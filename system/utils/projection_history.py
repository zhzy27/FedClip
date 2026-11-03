"""Opt-in finite-history Project: current hard halfspace, historical soft hinge.

No client training, scoring, or legacy projection arithmetic is changed here.
The active set is solved in the span of two unit directions, without flattening
parameters or constructing a parameter-space matrix.
"""
import json
import math
import os
import time

import torch

from utils.projection_variants import aggregate_projection_weighting
from utils.target_projection import EPS, _cosine, _delta, _dot
from utils.target_head import preserved_head_rng

MODE = 'projection_history_softmax'
REFERENCE_NORM_MIN = math.sqrt(EPS)


def add_options(parser):
    parser.add_argument('--projection_history_window', type=int, default=10)
    parser.add_argument('--projection_history_decay', type=float, default=.8)
    parser.add_argument('--projection_history_strength', type=float, default=1.)
    parser.add_argument('--projection_history_diagnostics', type=int, choices=[0, 1], default=1)


def enabled(args):
    return getattr(args, 'target_proj_mode', '') == MODE


def validate(args):
    window = getattr(args, 'projection_history_window', 10)
    decay = getattr(args, 'projection_history_decay', .8)
    strength = getattr(args, 'projection_history_strength', 1.)
    if type(window) is not int or window < 1:
        raise ValueError('projection_history_window must be a positive integer')
    if not math.isfinite(decay) or not 0 <= decay <= 1:
        raise ValueError('projection_history_decay must be finite in [0,1] (0: previous round only)')
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('projection_history_strength must be finite and nonnegative')
    if getattr(args, 'projection_history_diagnostics', 1) not in (0, 1):
        raise ValueError('projection_history_diagnostics must be 0/1')
    if not enabled(args):
        # Reject nondefault controls in modes that cannot use them.
        if (window, decay, strength) != (10, .8, 1.) or getattr(args, 'projection_history_diagnostics', 1) != 1:
            raise ValueError('History controls require projection_history_softmax')
        return
    family = getattr(args, 'model_family', getattr(args, 'model', ''))
    if (getattr(args, 'algorithm', '') != 'FedTargetProj' or family != 'Decom_CNN-5-512'
            or getattr(args, 'target_head_mode', 'shared') != 'shared'
            or getattr(args, 'target_local_schedule', 'joint') != 'joint'):
        raise ValueError('History Project supports only single-target FedTargetProj Decom_CNN-5-512, shared head, joint')
    forbidden = ('target_prox_mu', 'target_head_l2_lambda', 'target_z2_lambda', 'cnn512_control',
                 'meta_c0_split', 'meta_collect_snapshots', 'meta_weight_file', 'head_diag_collect',
                 'resume', 'models_folder_name', 'use_clip_alignment', 'use_asymmetric_lr',
                 'multi_svd', 'multi_proto_lambda', 'multi_avg_proto_ablation', 'learning_rate_decay')
    if (any(getattr(args, key, False) for key in forbidden)
            or getattr(args, 'target_c0_replay_mode', 'none') != 'none'
            or getattr(args, 'multi_proto_mode', 'none') != 'none'
            or getattr(args, 'multi_kd_mode', 'none') != 'none'):
        raise ValueError('History Project forbids Prox/head L2/Z2/replay/meta/prototypes/feedback/KD/control/resume combinations')
    if (getattr(args, 'join_ratio', 1.) != 1. or getattr(args, 'random_join_ratio', False)
            or any(getattr(args, key, 0) for key in ('client_drop_rate', 'train_slow_rate',
                'send_slow_rate', 'few_shot', 'auto_break', 'time_select'))):
        raise ValueError('History Project requires complete ordinary epochs/data and full participation')


def tag(args):
    return (f'{MODE}_L{getattr(args,"projection_history_window",10)}_'
            f'decay{getattr(args,"projection_history_decay",.8):g}_'
            f'strength{getattr(args,"projection_history_strength",1.):g}').replace('.', 'p')


def ensure_names(args):
    if not enabled(args):
        return
    value = tag(args)
    if value not in getattr(args, 'exp_name', ''):
        args.exp_name = getattr(args, 'exp_name', '') + '_' + value
    for key in ('save_folder_name', 'final_model_root', 'h5_result_root', 'training_log_root',
                'run_log_dir', 'training_log_dir'):
        path = getattr(args, key, '')
        if path and value not in path:
            setattr(args, key, os.path.join(path, value))


def schema(parameters):
    return [(name, list(p.shape), str(p.dtype)) for name, p in parameters.items()]


class History:
    """Transaction-style cache: mean is read-only; commit happens after success."""
    def __init__(self, window, decay):
        if type(window) is not int or window < 1 or not math.isfinite(decay) or not 0 <= decay <= 1:
            raise ValueError('History requires a positive window and finite decay in [0,1]')
        self.window, self.decay = window, decay
        self.entries = []  # ascending loop_round, independent CPU tensors
        self.parameter_schema = None

    @torch.no_grad()
    def mean(self, current_round, reference):
        if not self.entries:
            return None, []
        if self.parameter_schema != schema(reference):
            raise ValueError('History parameter schema changed')
        rounds = [r for r, _ in self.entries]
        if rounds[-1] != current_round - 1 or any(r >= current_round for r in rounds):
            raise ValueError('History must contain only completed, consecutive past rounds')
        recent = list(reversed(self.entries))
        weights = [self.decay ** j for j in range(len(recent))]
        denominator = math.fsum(weights)
        result = {n: torch.zeros_like(p) for n, p in reference.items()}
        for (_, values), weight in zip(recent, weights):
            for name in result:
                result[name].add_(values[name].to(result[name]), alpha=weight / denominator)
        return result, [r for r, _ in recent]

    @torch.no_grad()
    def commit(self, loop_round, delta):
        if self.entries and loop_round != self.entries[-1][0] + 1:
            raise ValueError('Cannot advance/repeat the history out of order')
        current_schema = schema(delta)
        if self.parameter_schema is not None and current_schema != self.parameter_schema:
            raise ValueError('Cannot commit a different parameter schema')
        if not delta or any(not torch.isfinite(p).all() for p in delta.values()):
            raise ValueError('Cannot commit empty/nonfinite history')
        copy = {n: p.detach().to('cpu', copy=True) for n, p in delta.items()}
        self.parameter_schema = current_schema
        self.entries.append((loop_round, copy))
        self.entries = self.entries[-self.window:]

    @property
    def bytes(self):
        return sum(p.numel() * p.element_size() for _, values in self.entries for p in values.values())

    def state(self):
        return dict(format='target_global_delta_history_v1', window=self.window, decay=self.decay,
                    parameter_schema=self.parameter_schema, entries=self.entries,
                    source_loop_rounds=[r for r, _ in self.entries],
                    source_completed_rounds=[r+1 for r, _ in self.entries], resume_supported=False)


def solve_coefficients(a, b, cosine, strength):
    """Return A,B for v=d+A*u+B*q, plus active-set diagnostics.

    a=u.d, b=q.d, c=u.q. Unconstrained soft prox first determines whether
    the hard constraint is active. On its boundary the penalized direction is
    q-c*u. The only denominator is 1+lambda*(1-c*c), which is >=1.
    """
    if not all(math.isfinite(x) for x in (a, b, cosine, strength)) or strength < 0:
        raise ValueError('Nonfinite/negative history solver input')
    if abs(cosine) > 1 + 1e-10:
        raise ValueError('Reference cosine outside [-1,1]')
    c = max(-1., min(1., cosine))
    k = strength / (1 + strength)
    B = -k * min(b, 0.)
    hard = a + B * c < 0
    if not hard:
        A = 0.
    else:
        tangent_dot = b - c * a
        # Equivalent to lambda/(1+lambda*delta), without product overflow.
        factor = 1 / (1 / strength + max(0., 1 - c*c)) if strength else 0.
        B = -factor * min(tangent_dot, 0.)
        A = -a - B * c
    if not math.isfinite(A) or not math.isfinite(B):
        raise ValueError('Nonfinite history projection coefficients')
    after_current = a + A + B*c
    after_history = b + A*c + B
    return A, B, dict(current_constraint_active=int(hard),
        history_penalty_active=int(after_history < 0 and strength > 0),
        solver_current_unit_dot=after_current, solver_history_unit_dot=after_history)


@torch.no_grad()
def aggregate(global_params, target_post, uploads_factory, target_id, history, loop_round,
              strength=1., self_weight=.125, softmax_tau=.2, diagnostics=True):
    """Old kernel supplies exactly the old scores/weights and RNG evolution.

    Only an active historical solve rereads uploads, inside RNG protection.
    The cache is never changed here. Caller commits pending_target after a
    successful server update. strength=0 never constructs or reads a cache.
    """
    start = time.perf_counter()
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('History strength must be finite and nonnegative')
    old, metrics, rows, _, matrices = aggregate_projection_weighting(
        global_params, target_post, uploads_factory, target_id, 'projection_softmax', self_weight, softmax_tau)
    legacy_finished = time.perf_counter()
    solve_seconds = 0.
    sources, h, pending = [], None, None
    if strength > 0:
        if history is None:
            raise ValueError('Positive history strength requires a cache')
        pending = _delta(target_post, global_params)
        h, sources = history.mean(loop_round, pending)
    rn = metrics['target_delta_norm']
    hn = math.sqrt(_dot(h, h)) if h is not None else 0.
    reason = ('strength_zero' if strength == 0 else 'no_past_rounds' if h is None
              else 'current_reference_near_zero' if rn <= REFERENCE_NORM_MIN
              else 'history_reference_near_zero' if hn <= REFERENCE_NORM_MIN else '')
    reference_cosine = _cosine(_dot(pending, h), rn*rn, hn*hn) if h is not None else None
    tangent = None
    tangent_sq = 0.
    exactly_aligned = False
    if not reason and abs(reference_cosine) > 1-1e-8:
        # Avoid cancellation in 1-c*c and b-c*a near parallel/antiparallel
        # directions. Keep the small orthogonal direction explicitly in double.
        exactly_aligned = all(torch.equal(h[n].double()/hn, pending[n].double()/rn) for n in h)
        exactly_opposite = all(torch.equal(h[n].double()/hn, -pending[n].double()/rn) for n in h)
        if exactly_aligned:
            reference_cosine = 1.
        elif exactly_opposite:
            reference_cosine = -1.
        tangent = {n: h[n].double()/hn-reference_cosine*pending[n].double()/rn for n in h}
        tangent_sq = _dot(tangent, tangent)
    # Exact aligned directions make the historical penalty redundant. Retain
    # the legacy EPS convention rather than adding a different hard correction.
    if not reason and exactly_aligned:
        reason = 'aligned_references_history_redundant'
    if not reason:
        r = pending
        result_delta = {n: torch.zeros_like(p) for n, p in global_params.items()}
        sum_r = sum_h = 0.
        tangent_correction = {n: torch.zeros_like(p) for n, p in global_params.items()} if tangent is not None else None
        by_id = {row['client_id']: row for row in rows}
        seen = set()
        with preserved_head_rng():
            for cid, sample_weight, post in uploads_factory():
                if cid in seen or cid not in by_id or sample_weight != by_id[cid]['sample_weight']:
                    raise ValueError('History upload IDs/weights changed between reads')
                seen.add(cid)
                row = by_id[cid]
                d = r if cid == target_id else _delta(post, global_params)
                raw_sq = _dot(d, d)
                rd, hd = _dot(d, r), _dot(d, h)
                if cid == target_id:
                    A = B = 0.
                    flags = dict(current_constraint_active=0, history_penalty_active=0)
                else:
                    solve_started = time.perf_counter()
                    A, B, flags = solve_coefficients(rd/rn, hd/hn, reference_cosine, strength)
                    solve_seconds += time.perf_counter()-solve_started
                cr, ch = A/rn, B/hn
                stable_boundary = tangent is not None and flags['current_constraint_active']
                if stable_boundary:
                    tangent_dot = _dot(d, tangent)
                    B = -min(tangent_dot, 0.)/(1/strength+tangent_sq)
                    v = {n: (p.double()-(rd/rn)*r[n].double()/rn+B*tangent[n]).to(p) for n, p in d.items()}
                    # Keep the boundary correction in its orthogonal basis;
                    # very large cancelling u/q coefficients are never applied.
                    cr, ch = -rd/(rn*rn), 0.
                    for n in tangent_correction:
                        tangent_correction[n].add_((B*tangent[n]).to(d[n])*row['aggregation_weight'])
                    flags['history_penalty_active'] = int(tangent_dot < 0)
                else:
                    v = {n: p + cr*r[n] + ch*h[n] for n, p in d.items()}
                after_r, after_h, v_sq = _dot(v, r), _dot(v, h), _dot(v, v)
                if stable_boundary:
                    flags.update(solver_current_unit_dot=after_r/rn, solver_history_unit_dot=after_h/hn)
                # Unit-dot residual relative to helper magnitude; explicit dtype
                # tolerance accommodates model-dtype reconstruction roundoff.
                tolerance = 64*max(torch.finfo(p.dtype).eps for p in d.values())*max(math.sqrt(raw_sq), 1e-12)
                residual = max(0., -after_r/rn) if cid != target_id else 0.
                if residual > tolerance or any(not torch.isfinite(p).all() for p in v.values()):
                    raise ValueError(f'History current constraint/nonfinite solve failed: client={cid}, residual={residual}, tolerance={tolerance}')
                weight = row['aggregation_weight']
                for n in result_delta:
                    result_delta[n].add_(d[n] * weight)
                sum_r += weight * cr
                sum_h += weight * ch
                if diagnostics:
                    legacy_c = row['projection_coefficient'] if cid != target_id else 0.
                    raw_correction = {n: v[n]-d[n] for n in d}
                    old_correction = {n: v[n]-(d[n]-legacy_c*r[n]) for n in d}
                    row.update(flags, current_dot_before=rd, current_dot_after=after_r,
                        history_dot_before=hd, history_dot_after=after_h,
                        current_cosine_after=_cosine(after_r, v_sq, rn*rn),
                        history_cosine_before=_cosine(hd, raw_sq, hn*hn),
                        history_cosine_after=_cosine(after_h, v_sq, hn*hn),
                        correction_from_raw_norm=math.sqrt(_dot(raw_correction, raw_correction)),
                        correction_from_legacy_norm=math.sqrt(_dot(old_correction, old_correction)),
                        current_constraint_residual=residual, current_constraint_tolerance=tolerance,
                        current_correction_coefficient=cr, history_correction_coefficient=ch,
                        orthogonal_history_coefficient=B if stable_boundary else 0.,
                        finite_check_passed=1)
                row['projected_update_norm'] = math.sqrt(v_sq)
            if seen != by_id.keys():
                raise ValueError('Missing history projection upload')
        for n in result_delta:
            result_delta[n].add_(r[n], alpha=sum_r)
            result_delta[n].add_(h[n], alpha=sum_h)
            if tangent_correction is not None:
                result_delta[n].add_(tangent_correction[n])
        old = {n: p.detach()+result_delta[n] for n, p in global_params.items()}
        metrics.update(proj_update_norm=math.sqrt(_dot(result_delta, result_delta)),
                       proj_target_cos=_cosine(_dot(result_delta, r), _dot(result_delta, result_delta), rn*rn))
    for row in rows:
        # Keep legacy conflict/removal fields explicitly identified as legacy,
        # since a dual-reference correction is not a single projected component.
        row['legacy_projection_coefficient'] = row.pop('projection_coefficient')
        row['legacy_removed_component_norm'] = row.pop('removed_component_norm')
        defaults = dict(current_constraint_active=None, history_penalty_active=None,
            solver_current_unit_dot=None, solver_history_unit_dot=None,
            current_dot_before=row['dot_before_projection'], current_dot_after=None,
            history_dot_before=None, history_dot_after=None, current_cosine_after=None,
            history_cosine_before=None, history_cosine_after=None, correction_from_raw_norm=None,
            correction_from_legacy_norm=0., current_constraint_residual=None, current_constraint_tolerance=None,
            current_correction_coefficient=None, history_correction_coefficient=0., finite_check_passed=1)
        defaults['orthogonal_history_coefficient'] = 0.
        for key, value in defaults.items():
            row.setdefault(key, value)
        if reason:
            # Legacy EPS arithmetic, no extra recovery/forward or history work.
            dot_after = row['dot_before_projection'] - row['legacy_projection_coefficient'] * rn*rn
            row.update(current_dot_after=dot_after, current_constraint_active=row['conflict'],
                history_penalty_active=0 if h is None else None,
                current_cosine_after=_cosine(dot_after, row['projected_update_norm']**2, rn*rn),
                correction_from_raw_norm=row['legacy_removed_component_norm'],
                current_constraint_residual=max(0., -dot_after/rn) if rn else 0.,
                current_correction_coefficient=-row['legacy_projection_coefficient'])
        row.update(history_applied=int(not reason), fallback_reason=reason)
    # Stable CSV columns/order in no-history, active, zero, and degenerate rounds.
    rows = [dict(sorted(row.items())) for row in rows]
    metrics.update(history_strength=strength, current_reference_norm=rn, history_reference_norm=hn,
        history_length=len(sources), history_source_loop_rounds=json.dumps(sources),
        history_source_completed_rounds=json.dumps([r+1 for r in sources]),
        current_history_cosine=reference_cosine, history_applied=int(not reason), history_fallback_reason=reason,
        reference_norm_min=REFERENCE_NORM_MIN, history_solve_seconds=solve_seconds,
        legacy_kernel_seconds=legacy_finished-start,
        history_extra_compute_seconds=time.perf_counter()-legacy_finished,
        history_cache_bytes_before=history.bytes if history is not None else 0,
        history_diagnostics_enabled=int(diagnostics), finite_check_passed=1)
    matrices.update(weighting_mode=MODE, projection_scope='full_recovered_global_delta', clients=rows,
        legacy_removed_update_ratio_scope='old current-only projection; not dual-reference removal',
        current_constraint='u dot v >= 0; model-dtype tolerance = 64*eps*raw_update_norm',
        history_penalty='strength/2 * min(0,q dot v)^2', history_source_loop_rounds=sources)
    return old, metrics, rows, [], matrices, pending


def initialize(server, model):
    if not enabled(server.args):
        return
    args = server.args
    strength = getattr(args, 'projection_history_strength', 1.)
    server.projection_history_cache = (History(getattr(args, 'projection_history_window', 10),
        getattr(args, 'projection_history_decay', .8)) if strength else None)
    server.projection_history_records = []
    from utils.head_snapshot import tensor_fingerprint
    from utils.cnn512_control import effective_grad_clip
    server.projection_history_policy = dict(mode=MODE, target_client_id=server.target_client_id,
        history_window=getattr(args, 'projection_history_window', 10),
        history_decay=getattr(args, 'projection_history_decay', .8), history_strength=strength,
        diagnostics=bool(getattr(args, 'projection_history_diagnostics', 1)),
        requested_self_weight=getattr(args, 'projection_self_weight', None),
        actual_self_weight=.125 if getattr(args, 'projection_self_weight', None) is None else args.projection_self_weight,
        softmax_tau=getattr(args, 'projection_softmax_tau', .2),
        score_scope='original helper/current-target global-delta cosine; unchanged stable softmax',
        history_scope='finite window of previous recovered-full-W target global deltas; excludes current round',
        delta_scope='post-local recovered named_parameters minus pre-aggregation server; includes shared fc3; no buffers',
        initialization='unchanged single-target Model_Distribe initialization and random protocol',
        initial_server_sha256=tensor_fingerprint(model.state_dict()), parameter_schema=schema(dict(model.named_parameters())),
        seed=server.seed, model_family=args.model_family, client_model_formulas=list(args.models),
        local_epochs=server.local_epochs, batch_size=server.batch_size, lr=server.learning_rate,
        regularization_enabled=args.is_regular, regularization_lambda=args.regular_lamda,
        requested_local_grad_clip=getattr(args, 'local_grad_clip', None),
        local_grad_clip=effective_grad_clip(args), grad_clip_enabled=effective_grad_clip(args) is not None,
        planned_train_count=server.global_rounds+1,
        models_per_client_per_round=1, extra_train_epochs=0, extra_communication_bytes=0,
        primary_metric='target_post_local_acc', accuracy_unit='fraction', resume_supported=False,
        buffers='retain original server buffers; ordinary client/download/evaluation conventions unchanged',
        reference_norm_min=REFERENCE_NORM_MIN,
        legacy_diagnostics='conflict/removed_update_ratio describe old current-only Project; dual corrections have separate fields')
    # File-only audit does not construct loaders, iterate data, or consume RNG.
    from pathlib import Path
    from common.experiment import sha256
    from utils.data_utils import get_dataset_sub_dir
    directory = (Path('../dataset')/args.dataset/get_dataset_sub_dir(args)).resolve()
    explicit = bool(getattr(args, 'data_root', None))
    if explicit:
        from common.client_data import data_directory, client_file
        directory = data_directory(args)
    files = []
    for client in server.clients:
        for train in (True, False):
            path = (client_file(args, client.id, train) if explicit else
                    directory/('train' if train else 'test')/f'{client.id}.npz')
            files.append(dict(client_id=client.id, split='train' if train else 'test',
                path=str(path.resolve()), sha256=sha256(path) if path.is_file() else None,
                samples=client.train_samples if train else client.test_samples))
    server.projection_history_policy.update(data_directory=str(directory), data_files=files,
        data_fingerprint_scope='selected shard files, without extra decoding/loader traversal',
        dataset=args.dataset, partition=getattr(args, 'partition', None),
        class_per_client=getattr(args, 'class_per_client', None))


def commit_success(server, pending, metrics):
    """Called only after aggregation/save/target evaluation have succeeded."""
    cache = server.projection_history_cache
    start = time.perf_counter()
    if cache is not None:
        previous_entries, previous_schema = cache.entries[:], cache.parameter_schema
        path = os.path.join(server.save_folder_name, 'projection_history_cache.pt')
        temporary = path + '.tmp'
        try:
            cache.commit(server.cur_ground, pending)
            # Separate from prediction checkpoints, atomically replace latest cache.
            torch.save(cache.state(), temporary)
            os.replace(temporary, path)
        except BaseException:
            cache.entries, cache.parameter_schema = previous_entries, previous_schema
            try:
                os.unlink(temporary)
            except OSError:
                pass  # Preserve the original failure; previous atomic cache stays valid.
            raise
    metrics.update(history_cache_bytes_after=cache.bytes if cache is not None else 0,
        history_cache_maintenance_seconds=time.perf_counter()-start,
        history_window=getattr(server.args, 'projection_history_window', 10),
        history_decay=getattr(server.args, 'projection_history_decay', .8))


def record(server, row, clients, matrices):
    server.projection_history_records.append(dict(**row, clients=clients, protocol=matrices))
    server._append_csv('projection_history_metrics.csv', [row])
    server._append_csv('projection_history_clients.csv', [dict(round=row['round'], loop_round=row['loop_round'], **c) for c in clients])
    from common.experiment import write_json
    write_json(os.path.join(server.report_dir(), 'projection_history.json'),
        dict(policy=server.projection_history_policy, summary=server._local_accuracy_summary(), history=server.projection_history_records))


def save_h5(server, parent):
    if not enabled(server.args):
        return
    import h5py
    group = parent.create_group('projection_history')
    group.attrs['policy_json'] = json.dumps(server.projection_history_policy)
    group.attrs['summary_json'] = json.dumps(server._local_accuracy_summary())
    group.create_dataset('rounds_json', data=[json.dumps(r, allow_nan=False) for r in server.projection_history_records], dtype=h5py.string_dtype('utf-8'))
    root = parent.file
    if 'rs_test_acc' in root:
        root.create_dataset('legacy_server_diagnostic_test_acc', data=root['rs_test_acc'][:])
        del root['rs_test_acc']
    root.create_dataset('rs_test_acc', data=[r['target_post_local_acc'] for r in server.target_proj_history])
    root.attrs['rs_test_acc_scope'] = 'ordinary_target_post_local_before_aggregation'


def save_manifest(server):
    if not enabled(server.args):
        return
    import shutil
    from pathlib import Path
    path = Path(server.final_model_dir())/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['projection_history'] = dict(policy=server.projection_history_policy,
        primary_summary=server._local_accuracy_summary(),
        cache='projection_history_cache.pt' if server.projection_history_cache is not None else None,
        cache_required_for_prediction=False, resume_supported=False,
        ordinary_post_local_model=f'Client_{server.target_client_id}_model.pt',
        aggregate_server_model='Server_model.pt', metrics_dir=str(Path(server.report_dir()).resolve()))
    source = Path(server.save_folder_name)/'projection_history_cache.pt'
    if source.is_file() and not (path.parent/source.name).exists():
        shutil.copy2(source, path.parent/source.name)
    manifest['legacy_server_diagnostic_best_accuracy'] = manifest.get('best_accuracy')
    manifest['best_accuracy'] = server._local_accuracy_summary()['best_target_local_acc']
    manifest['primary_accuracy_metric'] = 'target_post_local_acc'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
