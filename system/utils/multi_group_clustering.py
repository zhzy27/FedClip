"""Training-update-only balanced kernel spherical clustering, fixed after R1.

Similarity is the mean of K within-column delta cosines, never full-weight cosine.
Hungarian assignment repeats each center capacity times. Local NumPy Generator
and sorted row/slot order make all ties deterministic without global RNG calls.
"""
import hashlib
import json
import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from common.experiment import jsonable
from common.models import checked_parameters


def canonical_groups(groups):
    return sorted((sorted(map(int, group)) for group in groups), key=lambda g: (-len(g), tuple(g)))


def digest_json(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def grouping_context(args, data, initial_hash):
    files = [{key: row[key] for key in ('client_id', 'split', 'sha256', 'decoded_x_sha256', 'decoded_y_sha256', 'count')}
             for row in data['files']]
    return dict(num_clients=args.num_clients, K=args.num_server_models, model_family=args.model_family,
                capacities=args.multi_capacities, initial_parameter_sha256=initial_hash,
                data_fingerprint=digest_json(dict(config_sha256=data['config_sha256'], files=files)),
                training=dict(seed=args.seed, lr=args.local_learning_rate, batch=args.batch_size,
                              local_epochs=args.local_epochs, grad_clip=args.local_grad_clip))


@torch.no_grad()
def update_similarity(servers, clients, chunk_size=65536):
    n, k = len(clients), len(servers)
    similarity = np.zeros((n, n), dtype=np.float64)
    zero_by_model, column_cosines = [], []
    update_hash = hashlib.sha256()
    for column, server in enumerate(servers):
        base = dict(server.named_parameters())
        posts = [checked_parameters(c.model_set[column], server)[0] for c in clients]
        gram = torch.zeros((n, n), dtype=torch.float64, device=next(server.parameters()).device)
        # Subtract in original parameter dtype, then reduce in double precision.
        # Bound temporary memory to N x chunk_size, not N flattened models.
        for name, parameter in base.items():
            old = parameter.reshape(-1)
            for start in range(0, old.numel(), chunk_size):
                block = torch.stack([(post[name].reshape(-1)[start:start+chunk_size] - old[start:start+chunk_size])
                                     for post in posts])
                if not torch.isfinite(block).all():
                    raise ValueError('Nonfinite first-round update')
                update_hash.update(block.cpu().contiguous().numpy().tobytes())
                values = block.double()
                gram.add_(values @ values.T)
        gram = gram.cpu().numpy()
        norms = np.sqrt(np.maximum(0., np.diag(gram)))
        denom = np.outer(norms, norms)
        cosines = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
        cosines = np.clip(cosines, -1., 1.)
        zero_by_model.append(np.flatnonzero(norms == 0).tolist())
        column_cosines.append(cosines.tolist())
        similarity += cosines / k
    return similarity, dict(zero_update_clients_by_model=zero_by_model,
        zero_vector_cosine=0., per_model_cosine_matrices=column_cosines,
        first_upload_delta_sha256=update_hash.hexdigest(), chunk_size=chunk_size)


def balanced_cluster(similarity, sizes, *, seed=0, restarts=8, iterations=50):
    s = np.asarray(similarity, dtype=np.float64)
    n, k = sum(sizes), len(sizes)
    if s.shape != (n, n) or not np.isfinite(s).all() or not np.allclose(s, s.T, atol=1e-12):
        raise ValueError('Require finite symmetric N x N update cosine similarity')
    if any(size < 1 for size in sizes) or restarts < 1 or iterations < 1:
        raise ValueError('Require positive capacities/restarts/iterations')
    rng = np.random.default_rng(seed)
    def scores_for(groups):
        result = np.zeros((n, k), dtype=np.float64)
        for column, ids in enumerate(groups):
            norm = math.sqrt(max(0., float(s[np.ix_(ids, ids)].mean())))
            if norm > 0:
                result[:, column] = s[:, ids].mean(axis=1) / norm
        return result
    best, candidates = None, []
    for restart in range(restarts):
        centers = [int(rng.integers(n))]
        for _ in range(1, k):
            distances = np.maximum(0., 1. - s[:, centers].max(axis=1))
            distances[centers] = 0.
            remaining = [i for i in range(n) if i not in centers]
            centers.append(int(rng.choice(n, p=distances/distances.sum())) if distances.sum() else remaining[0])
        scores = s[:, centers]
        previous, tie_count, converged = None, 0, False
        for iteration in range(iterations):
            slots = np.repeat(np.arange(k), sizes)
            rows, assigned = linear_sum_assignment(-scores[:, slots])
            labels = np.empty(n, dtype=int)
            labels[rows] = slots[assigned]
            groups = [np.flatnonzero(labels == c).tolist() for c in range(k)]
            tie_count += sum(len(np.unique(row)) < k for row in scores)
            if previous == groups:
                converged = True
                break
            previous, scores = groups, scores_for(groups)
        canonical = canonical_groups(groups)
        objective = float(sum(scores_for(groups)[cid, column] for column, ids in enumerate(groups) for cid in ids))
        candidates.append(dict(restart=restart, objective=objective, iterations=iteration+1, tied_score_rows=tie_count,
                               converged=converged))
        key = tuple(tuple(g) for g in canonical)
        if best is None or objective > best[0] + 1e-12 or (abs(objective-best[0]) <= 1e-12 and key < best[1]):
            best = objective, key, canonical
    off = s[~np.eye(n, dtype=bool)]
    return best[2], dict(algorithm='balanced_kernel_spherical_hungarian_v1', seed=seed, restarts=restarts,
        max_iterations=iterations, objective=best[0], candidates=candidates,
        capacities=sorted(sizes, reverse=True), numbering='descending size then lexicographic sorted members',
        tie_rule='sorted client/center slots, deterministic scipy assignment; equal restart objective uses lexicographic groups',
        zero_centroid_rule='all center similarities=0', degenerate_similarity=bool(not len(off) or np.ptp(off) < 1e-12))


def validate_group_file(payload, context):
    if payload.get('schema_version') != 1 or payload.get('context_sha256') != digest_json(context) or payload.get('context') != context:
        raise ValueError('Grouping file does not match data/initialization/N/K/capacities/training protocol')
    groups = payload.get('groups')
    if not isinstance(groups, list) or any(not isinstance(g, list) or not g or any(type(i) is not int for i in g) for g in groups):
        raise ValueError('Invalid grouping members')
    if groups != canonical_groups(groups) or sorted(i for g in groups for i in g) != list(range(context['num_clients'])):
        raise ValueError('Grouping must canonically partition all client IDs exactly once')
    if [len(g) for g in groups] != context['capacities'] or len(groups) != context['K']:
        raise ValueError('Grouping sizes differ')
    expected = {str(cid): k for k, group in enumerate(groups) for cid in group}
    if payload.get('client_to_model') != expected:
        raise ValueError('Grouping mapping differs from member lists')
    if payload.get('group_fingerprint') != digest_json(dict(context=context, groups=groups)):
        raise ValueError('Grouping fingerprint mismatch')
    return groups
