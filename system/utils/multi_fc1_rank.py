"""K1 fc1 local rank scheduling and the legacy single-layer shared download SVD."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

from utils.multi_group_svd import factorized_layers


MODES = ('full', 'random_weighted', 'random_equal', 'staged_weighted',
         'staged_weighted_frozen', 'staged_equal', 'staged_equal_frozen')
REORDER = 'coverage_reorder'


def add_options(parser):
    parser.add_argument('--multi_fc1_local_mode', choices=MODES, default='full')


def uses_shared_download(args):
    return factorized_layers(args) == ('fc1',)


def uses_local_schedule(args):
    layers = factorized_layers(args)
    # Keep the per-rank logs of the original single-layer full training case.
    return 'fc1' in layers and (layers == ('fc1',) or getattr(args, 'multi_fc1_local_mode', 'full') != 'full')


def validate(args):
    layers = factorized_layers(args)
    mode = getattr(args, 'multi_fc1_local_mode', 'full')
    spectral = getattr(args, 'multi_svd_spectral_mode', 'none')
    if mode not in MODES:
        raise ValueError('Invalid fc1 local schedule')
    if mode != 'full' and ('fc1' not in layers or not getattr(args, 'multi_svd', False)):
        raise ValueError(f'{mode} requires low-rank fc1 via --multi_svd and --multi_svd_factorized_layers including fc1')
    if spectral == REORDER and not uses_shared_download(args):
        raise ValueError('coverage_reorder requires --multi_svd_factorized_layers fc1')
    if not uses_local_schedule(args):
        return
    aggregation = getattr(args, 'multi_svd_aggregation', 'full_avg')
    if (args.algorithm != 'FedMultiGroupProj' or not getattr(args, 'multi_avg_proto_ablation', False)
            or not args.multi_svd or args.num_server_models != 1 or args.join_ratio != 1.
            or args.multi_group_mode != 'group_avg_only' or args.target_head_mode != 'shared'
            or args.target_local_schedule != 'joint' or getattr(args, 'multi_no_pre_eval', False)
            or getattr(args, 'multi_last_epoch_protocol', False) or getattr(args, 'multi_head_mode', 'shared') != 'shared'
            or getattr(args, 'multi_proto_mode', 'none') != 'none' or args.multi_proto_lambda != 0
            or getattr(args, 'multi_fc1_tail_scale_strength', 0) != 0 or getattr(args, 'multi_svd_five_diagnostics', 0)
            or getattr(args, 'multi_kd_mode', 'none') != 'none' or getattr(args, 'target_c0_replay_mode', 'none') != 'none'
            or getattr(args, 'target_curriculum_mode', 'none') != 'none' or getattr(args, 'target_z2_lambda', 0)
            or getattr(args, 'target_prox_mu', 0) or getattr(args, 'target_head_l2_lambda', 0)):
        raise ValueError('fc1 experiment requires K1 full-participation SVD Avg, shared/joint, pre evaluation, no auxiliary modules or three-layer diagnostics')
    if any(rate >= 1 for rate in args.multi_svd_effective_ratios):
        raise ValueError('fc1 experiment requires a valid local schedule and low-rank clients')
    if uses_shared_download(args) and (spectral not in ('none', REORDER) or (spectral == REORDER and aggregation != 'full_avg')):
        raise ValueError('Single-layer shared download supports plain SVD or coverage_reorder with full_avg')


def protocol(args):
    return dict(factorized_layers=list(factorized_layers(args)), scheduled_layer='fc1',
                local_mode=getattr(args, 'multi_fc1_local_mode', 'full'),
                aggregation=getattr(args, 'multi_svd_aggregation', 'full_avg'),
                spectral_mode=getattr(args, 'multi_svd_spectral_mode', 'none'),
                allocation='integer largest remainder; ties use ascending rank',
                random_full_share=.60, random_stream='private seed/client/zero-based round',
                optimizer=dict(name='SGD', momentum=0, weight_decay=0),
                segment_update_steps='optimizer steps with that factor interval active and unfrozen',
                ce_averaging='sample visits within each active rank or stage',
                evaluation_and_upload='full assigned rank')


def assigned_ranks(args, layer):
    from flcore.trainmodel.models import factorized_linear_rank
    rates = args.multi_svd_effective_ratios
    return [factorized_linear_rank(layer.in_features, layer.out_features, rates[level])
            for level in args.multi_svd_effective_assignment]


def coverage_reorder(u, singular, vh, ranks, samples):
    """Amplify by original index, then apply one stable permutation to all three outputs."""
    import torch
    if (len(ranks) != len(samples) or not ranks or any(not 0 < r <= len(singular) for r in ranks)
            or any(n <= 0 for n in samples) or not torch.isfinite(singular).all()
            or (singular < 0).any() or (singular[1:] > singular[:-1]).any()):
        raise ValueError('Invalid original spectrum, actual ranks or train counts')
    maximum = max(ranks)
    mass = [sum(n for rank, n in zip(ranks, samples) if rank > j) for j in range(maximum)]
    coverage = singular.new_tensor(mass) / sum(samples)
    amplified = singular.clone()
    amplified[:maximum] = singular[:maximum] / coverage
    if not torch.isfinite(amplified).all():
        raise FloatingPointError('Nonfinite amplified fc1 spectrum')
    order = torch.argsort(amplified, descending=True, stable=True)
    return u[:, order], amplified[order], vh[order, :], amplified, coverage, order


def _boundary_ratios(spectrum, tiers):
    return [dict(boundary_rank=r, sigma_before=spectrum[r-1], sigma_after=spectrum[r],
                 next_over_boundary=spectrum[r]/spectrum[r-1] if spectrum[r-1] > 0 else None)
            for r in tiers if r < len(spectrum)]


class Download:
    """Run one fc1 SVD on the complete mother and reuse its triplet for this round."""

    def __init__(self, server):
        self.loop = server.loop_round
        self.mode = getattr(server.args, 'multi_svd_spectral_mode', 'none')
        self.layer = server.global_model_set[0].fc1
        self.ranks = assigned_ranks(server.args, self.layer)
        self.samples = [c.train_samples for c in server.clients]
        self.server_hash = server.initial_hash if self.loop == 0 else None
        self.triplet = None
        self.record = None

    def svd(self, matrix):
        import torch
        if self.triplet is not None:
            return self.triplet
        if not torch.equal(matrix, self.layer.weight):
            raise ValueError('Shared fc1 SVD must use the current server weight')
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        self.triplet = (u, singular, vh)
        applied = self.loop > 0 and self.mode == REORDER
        if applied:
            u, sorted_values, vh, amplified, coverage, order = coverage_reorder(u, singular, vh, self.ranks, self.samples)
            self.triplet = (u, sorted_values, vh)
            q = coverage.tolist()
            gains = (1 / coverage).tolist() + [1.] * (len(singular) - len(q))
            ordering = [i+1 for i in order.tolist()]
        else:
            amplified = sorted_values = singular
            gains = [1.] * len(singular)
            q = None
            ordering = list(range(1, len(singular)+1))
        tiers = sorted(set(self.ranks))
        raw, sorted_list = singular.tolist(), sorted_values.tolist()
        moved = sum(j > tiers[0] for j in ordering[:tiers[0]])
        self.record = dict(loop_round=self.loop, completed_train_count=self.loop+1, layer='fc1',
            initial_mother_sha256=self.server_hash, mode=self.mode, correction_applied=applied,
            client_ranks=self.ranks, original_train_samples=self.samples, rank_boundaries=tiers, original_spectrum=raw,
            original_position_coverage=q, original_position_applied_gain=gains, amplified_before_sort=amplified.tolist(),
            sorted_spectrum=sorted_list, sorted_to_original_rank_one_based=ordering,
            directions_entering_smallest_prefix=moved, directions_leaving_smallest_prefix=moved,
            original_boundary_ratios=_boundary_ratios(raw, tiers[:-1]),
            download_boundary_ratios=_boundary_ratios(sorted_list, tiers[:-1]), shared_download_svd_calls=1)
        return self.triplet


@contextmanager
def use_prefix(layer, active, frozen):
    if active == layer.rank and frozen == 0:
        yield
        return
    layer._fc1_training_prefix = (active, frozen)
    try:
        yield
    finally:
        del layer._fc1_training_prefix


def batch_plan(mode, ranks, steps, seed):
    widths = [right-left for left, right in zip([0]+ranks[:-1], ranks)]
    if mode == 'full' or len(ranks) == 1:
        weights = [0] * (len(ranks)-1) + [1]
    elif mode == 'random_weighted':
        weights = [2*w for w in widths[:-1]] + [3*sum(widths[:-1])]
    elif 'weighted' in mode:
        weights = widths
    else:
        weights = [1] * len(ranks)
    counts, remainder = map(list, zip(*(divmod(steps*w, sum(weights)) for w in weights)))
    for index in sorted(range(len(ranks)), key=lambda i: (-remainder[i], i))[:steps-sum(counts)]:
        counts[index] += 1
    plan = [(r, 0) for r, n in zip(ranks, counts) for _ in range(n)]
    if mode.startswith('random_'):
        random.Random(seed).shuffle(plan)
    elif mode.endswith('_frozen'):
        previous_trained = 0
        plan = []
        for rank, count in zip(ranks, counts):
            if count:
                plan.extend([(rank, previous_trained)] * count)
                previous_trained = rank
    return plan, counts


class Schedule:
    def __init__(self, client, layer, batches, optimizer):
        args = client.args
        self.mode = getattr(args, 'multi_fc1_local_mode', 'full')
        self.layer = layer
        assignments = assigned_ranks(args, layer)
        if assignments[client.id] != layer.rank:
            raise ValueError('fc1 local rank differs from the assigned rank')
        self.ranks = sorted({r for r in assignments if r <= layer.rank})
        for group in optimizer.param_groups:
            if group['momentum'] != 0 or group['weight_decay'] != 0:
                raise ValueError('Rank schedules require SGD with zero momentum and weight decay')
        seed_text = f'fc1-rank-schedule:{args.seed}:{client.id}:{client.download_round}'
        self.seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'big')
        self.plan, self.counts = batch_plan(self.mode, self.ranks, client.local_epochs*batches, self.seed)
        self.rows = {r: dict(active_rank=r, batches=0, sample_visits=0, ce_sum=0.) for r in self.ranks}
        self.segments = [dict(start_exclusive=lo, end_inclusive=hi, update_steps=0)
                         for lo, hi in zip([0]+self.ranks[:-1], self.ranks)]

    def batch(self, step):
        return use_prefix(self.layer, *self.plan[step])

    def observe(self, step, samples, ce):
        active, frozen = self.plan[step]
        row = self.rows[active]
        row['batches'] += 1
        row['sample_visits'] += samples
        row['ce_sum'] += ce*samples
        for segment in self.segments:
            if frozen <= segment['start_exclusive'] and segment['end_inclusive'] <= active:
                segment['update_steps'] += 1

    def fields(self, steps):
        if steps != len(self.plan) or [r['batches'] for r in self.rows.values()] != self.counts:
            raise RuntimeError('Actual SGD batches do not match the fc1 schedule')
        stages = []
        if self.mode.startswith('staged'):
            done = 0
            for rank, count in zip(self.ranks, self.counts):
                stages.append(dict(active_rank=rank, start_step=done+1 if count else None,
                                   end_step=done+count if count else None,
                                   frozen_prefix=self.plan[done][1] if count else None))
                done += count
        by_rank = []
        for row in self.rows.values():
            ce_sum = row['ce_sum']
            by_rank.append(dict(active_rank=row['active_rank'], batches=row['batches'], sample_visits=row['sample_visits'],
                                train_mean_ce=ce_sum/row['sample_visits'] if row['sample_visits'] else None))
        return dict(fc1_local_mode=self.mode, fc1_rank_by_batch=by_rank, fc1_update_segments=self.segments,
                    fc1_stages=stages, fc1_rank_schedule_seed=self.seed if self.mode.startswith('random_') else None,
                    fc1_rank_schedule_sha256=hashlib.sha256(json.dumps(self.plan).encode()).hexdigest())


def finish_round(server, download):
    import torch
    from utils.multi_rank_partition import protected_rng
    if download.record is None:
        raise RuntimeError('Missing shared download spectrum')
    record = download.record
    if getattr(server.args, 'multi_svd_aggregation', 'full_avg') == 'rank_partition':
        mother = server.global_model_set[0]
        with torch.no_grad(), protected_rng([mother]):
            values = torch.linalg.svdvals(mother.fc1.weight.detach().to(device='cpu', dtype=torch.float64)).tolist()
        record['post_aggregation_spectrum'] = values
    server.fc1_download_records.append(record)
    with (Path(server.report_dir())/'fc1_download_spectrum.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False)+'\n')
