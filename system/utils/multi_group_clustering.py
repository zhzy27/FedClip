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
    context = dict(num_clients=args.num_clients, K=args.num_server_models, model_family=args.model_family,
                capacities=args.multi_capacities, initial_parameter_sha256=initial_hash,
                data_fingerprint=digest_json(dict(config_sha256=data['config_sha256'], files=files)),
                training=dict(seed=args.seed, lr=args.local_learning_rate, batch=args.batch_size,
                              local_epochs=args.local_epochs, grad_clip=args.local_grad_clip))
    from utils.multi_group_protocol import compatibility_default, grouping_protocol
    if not compatibility_default(args):
        context['grouping_protocol'] = grouping_protocol(args)
    from utils.multi_group_svd import enabled, capacity_protocol, initialization_protocol
    if enabled(args):
        context['svd_capacity_protocol'] = capacity_protocol(args)
        # Missing initialization metadata in existing artifacts means legacy
        # dense initialization. Keep that context byte-for-byte compatible.
        # A recovered initialization always needs this additional protocol key,
        # even at rank=1 where the initial values can equal the dense path.
        if initialization_protocol(args)['mode'] != 'dense':
            context['svd_initialization'] = initialization_protocol(args)
    return context


@torch.no_grad()
def update_similarity(servers, clients, chunk_size=65536, *, basis='single_delta'):
    if basis not in ('single_delta', 'full_weights'):
        raise ValueError('Similarity input must be single_delta or full_weights')
    n, k = len(clients), len(servers)
    similarity = np.zeros((n, n), dtype=np.float64)
    zero_by_model, column_cosines = [], []
    update_hash = hashlib.sha256()
    for column, server in enumerate(servers):
        base = dict(server.named_parameters())
        heterogeneous = any(hasattr(c, 'full_upload_parameters') for c in clients)
        if heterogeneous:
            if not all(hasattr(c, 'full_upload_parameters') for c in clients):
                raise ValueError('Mixed full and SVD client upload protocols')
            posts = [c.full_upload_parameters(column) for c in clients]
            for post in posts:
                if post.keys() != base.keys() or any(post[n].shape != base[n].shape for n in base):
                    raise ValueError('Recovered grouping parameters differ from full mother')
        else:
            posts = [checked_parameters(c.model_set[column], server)[0] for c in clients]
        gram = torch.zeros((n, n), dtype=torch.float64, device=next(server.parameters()).device)
        # Subtract in original parameter dtype, then reduce in double precision.
        # Bound temporary memory to N x chunk_size, not N flattened models.
        for name, parameter in base.items():
            old = parameter.reshape(-1)
            for start in range(0, old.numel(), chunk_size):
                pieces = [post[name].reshape(-1)[start:start+chunk_size] for post in posts]
                if heterogeneous:
                    pieces = [p.to(parameter) for p in pieces]
                block = torch.stack([p if basis == 'full_weights' else p-old[start:start+chunk_size] for p in pieces])
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
    diagnostics = dict(zero_update_clients_by_model=zero_by_model,
        zero_vector_cosine=0., per_model_cosine_matrices=column_cosines,
        first_upload_delta_sha256=update_hash.hexdigest(), chunk_size=chunk_size)
    if basis == 'full_weights':
        diagnostics['zero_full_weight_clients_by_model'] = diagnostics.pop('zero_update_clients_by_model')
        diagnostics['upload_full_weights_sha256'] = diagnostics.pop('first_upload_delta_sha256')
        diagnostics['basis'] = basis
    return similarity, diagnostics


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
    if not isinstance(payload, dict):
        raise ValueError('Grouping file must contain a JSON object')
    if payload.get('schema_version') not in (1, 2) or payload.get('context_sha256') != digest_json(context) or payload.get('context') != context:
        raise ValueError('Grouping file does not match data/initialization/N/K/capacities/training protocol')
    from types import SimpleNamespace
    from utils.multi_group_protocol import grouping_protocol
    expected_protocol = context.get('grouping_protocol', grouping_protocol(SimpleNamespace()))
    if payload.get('grouping_protocol', expected_protocol) != expected_protocol:
        raise ValueError('Grouping basis/warmup/history protocol mismatch')
    if 'grouping_protocol' in context:
        protocol = context['grouping_protocol']
        evidence = payload.get('evidence', {})
        if not isinstance(evidence, dict):
            raise ValueError('Grouping file needs structured provenance evidence')
        if (payload.get('grouping_protocol') != protocol or evidence.get('protocol') != protocol or
                payload.get('completed_train_count') != protocol['formation_completed_train_count'] or
                [r.get('loop_round') for r in evidence.get('rounds', [])] != protocol['evidence_loop_rounds']):
            raise ValueError('Grouping basis/warmup/history window/formation/evidence rounds mismatch')
        unsigned = {k: v for k, v in evidence.items() if k != 'evidence_sha256'}
        if evidence.get('evidence_sha256') != digest_json(unsigned):
            raise ValueError('Grouping evidence fingerprint mismatch')
        for row in evidence['rounds']:
            if row.get('matrix_sha256') != digest_json(row.get('matrix')) or row.get('completed_train_count') != row['loop_round']+1:
                raise ValueError('Grouping history matrix/round fingerprint mismatch')
            columns = np.asarray(row.get('diagnostics',{}).get('per_model_cosine_matrices',[]),dtype=np.float64)
            n = context['num_clients']
            if columns.shape != (context['K'],n,n) or not np.isfinite(columns).all():
                raise ValueError('Grouping evidence model-column count/shape differs from N/K')
            if not np.allclose(np.mean(columns,axis=0),np.asarray(row['matrix']),atol=1e-14,rtol=0):
                raise ValueError('Grouping round matrix is not the mean of its K column cosines')
        matrices = np.asarray([r['matrix'] for r in evidence['rounds']], dtype=np.float64)
        n = context['num_clients']
        if matrices.shape != (len(protocol['evidence_loop_rounds']), n, n) or not np.isfinite(matrices).all():
            raise ValueError('Invalid grouping history similarity matrices')
        final = matrices[0] if len(matrices)==1 else np.mean(matrices, axis=0)
        if (payload.get('similarity_matrix') != final.tolist() or
                evidence.get('final_similarity_sha256') != digest_json(final)):
            raise ValueError('Grouping final similarity does not match per-round evidence average')
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
