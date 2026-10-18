"""Opt-in homogeneous comparison protocol; legacy defaults are untouched."""
import math
import random

import numpy as np
import torch


def add_control_arguments(parser):
    parser.add_argument('--cnn512_control', action='store_true', help='Explicit shared CNN-512 comparison protocol')
    parser.add_argument('--local_grad_clip', type=float, default=None,
                        help='Gradient norm cap; 0 disables; omitted comparison/TargetProj=10, original FedFewS=off')
    parser.add_argument('--group_softmax_tau', type=float, default=.2, help='Group outside-helper cosine softmax divisor (not other algorithms temperature)')
    parser.add_argument('--projection_softmax_tau', type=float, default=.2, help='Single-target helper cosine softmax divisor')
    parser.add_argument('--control_test_batch_size', type=int, default=16)
    parser.add_argument('--control_eval_scope', choices=['c0', 'all', 'group'], default='c0',
                        help='Post-local primary evaluation population; default preserves the C0 control')
    parser.add_argument('--control_preview', action='store_true', help='Audit existing comparison shards and print effective settings; no model or training')


def primary_scope(args):
    return getattr(args, 'control_eval_scope', 'c0')


def population_control(args):
    return getattr(args, 'cnn512_control', False) and primary_scope(args) != 'c0'


def full_target(args):
    return args.algorithm == 'FedTargetProj' and args.model_family == 'CNN-512'


def evaluation_due(args, loop):
    """Same zero-based schedule as FedFewS; the last training round is observed."""
    return loop % args.eval_gap == 0 or loop == args.global_rounds


def effective_grad_clip(args):
    value = getattr(args, 'local_grad_clip', None)
    if value is None:
        return 10. if getattr(args, 'cnn512_control', False) or args.algorithm == 'FedTargetProj' else None
    return value if value > 0 else None  # Explicit zero disables clipping.


def validate_control(args):
    if args.algorithm == 'FedMultiGroupProj':
        from utils.multi_group_options import validate_multi_group
        validate_multi_group(args)
        return
    if primary_scope(args) not in ('c0', 'all', 'group'):
        raise ValueError('control_eval_scope must be c0/all/group')
    clip = getattr(args, 'local_grad_clip', None)
    if clip is not None and (not math.isfinite(clip) or clip < 0):
        raise ValueError('local_grad_clip must be finite and nonnegative; 0 disables it')
    if clip is not None and args.algorithm not in ('FedTargetProj', 'FedFewS'):
        raise ValueError('local_grad_clip is supported only by FedTargetProj/FedFewS')
    for key in ('group_softmax_tau', 'projection_softmax_tau'):
        value = getattr(args, key, .2)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        supported = ('group_projection_softmax', 'group_softmax_only') if key == 'group_softmax_tau' else ('projection_softmax', 'softmax_only')
        if value != .2 and (args.algorithm != 'FedTargetProj' or args.target_proj_mode not in supported):
            raise ValueError(f'Nondefault {key} requires the corresponding cosine softmax mode; it is not a generic temperature')
    if full_target(args):
        if args.dataset != 'Cifar100' or args.num_classes != 100:
            raise ValueError('Full CNN-512 currently supports only RGB CIFAR-100/100 classes')
        if args.target_proj_mode not in ('avg', 'projection_softmax', 'group_projection_softmax'):
            raise ValueError('Full CNN-512 supports avg/projection_softmax/group_projection_softmax')
        if (args.is_regular or args.target_prox_mu or args.target_head_mode != 'shared'
                or args.target_local_schedule != 'joint' or args.meta_c0_split
                or args.meta_collect_snapshots or args.head_diag_collect):
            raise ValueError('Full CNN-512 requires shared head, joint CE, no F/Prox/meta/head diagnostics')
    if not getattr(args, 'cnn512_control', False):
        if getattr(args, 'control_preview', False):
            raise ValueError('control_preview requires cnn512_control')
        if primary_scope(args) != 'c0':
            raise ValueError('control_eval_scope all/group requires cnn512_control')
        return
    if args.algorithm not in ('FedTargetProj', 'FedFewS') or args.model_family != 'CNN-512' or args.dataset != 'Cifar100' or args.num_classes != 100:
        raise ValueError('Comparison currently supports only full CNN-512 on CIFAR-100/100 classes')
    if (args.join_ratio != 1 or args.random_join_ratio or args.client_drop_rate or args.train_slow_rate
            or args.send_slow_rate or args.auto_break or args.learning_rate_decay or args.resume
            or args.few_shot or args.is_regular or args.target_prox_mu or args.meta_c0_split
            or args.meta_collect_snapshots or args.target_head_mode != 'shared'
            or args.target_local_schedule != 'joint' or args.head_diag_collect or args.time_select
            or getattr(args, 'models_folder_name', '')):
        raise ValueError('Comparison requires full participation, fresh complete local training, shared joint CE; partial/drop/slow/resume/decay/early-stop/F/Prox/meta are not implemented')
    for key in ('num_clients', 'batch_size', 'local_epochs', 'eval_gap', 'control_test_batch_size', 'num_server_models'):
        value = getattr(args, key, 3 if key == 'num_server_models' else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if getattr(args, 'global_rounds', 1) < 0:
        raise ValueError('global_rounds must be nonnegative (actual training count is gr+1)')
    if not 0 <= args.target_client_id < args.num_clients:
        raise ValueError('target_client_id must be in [0,num_clients)')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0,2**32) for NumPy/Torch controlled streams')
    for key in ('local_learning_rate', 'smooth_mu'):
        value = getattr(args, key, .01 if key == 'smooth_mu' else None)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    alpha = getattr(args, 'data_expected_alpha', None)
    if alpha is not None and (not math.isfinite(alpha) or alpha <= 0):
        raise ValueError('data-expected-alpha must be finite and positive')
    if not args.data_root or not args.data_subdir:
        raise ValueError('Comparison requires explicit data-root and exact existing data-subdir; no partition discovery')
    partition = getattr(args, 'partition', None) or getattr(args, 'data_expected_partition', None) or 'pat'
    if partition not in ('pat', 'dir'):
        raise ValueError('Comparison currently supports existing pat/dir image shards only')
    expected = getattr(args, 'data_expected_partition', None)
    if expected is not None and expected != partition:
        raise ValueError('data-expected-partition conflicts with -pt')
    if partition == 'pat' and not 1 <= getattr(args, 'class_per_client', 20) <= args.num_classes:
        raise ValueError('pat requires 1 <= class_per_client <= num_classes')
    scope = primary_scope(args)
    group_mode = args.algorithm == 'FedTargetProj' and args.target_proj_mode == 'group_projection_softmax'
    if scope == 'group' or group_mode:
        from utils.target_group_options import validate_group_options
        if not group_mode or scope != 'group' or args.projection_self_weight is not None:
            raise ValueError('Group scope requires group_projection_softmax and no single-target self weight')
        validate_group_options(args.target_group_ids, args.target_group_weight, args.target_proj_mode, args.num_clients)
    if args.algorithm == 'FedFewS' and (getattr(args, 'target_group_ids', None) is not None or getattr(args, 'target_group_weight', None) is not None):
        raise ValueError('FedFewS does not use target group aggregation options')
    if args.algorithm == 'FedTargetProj':
        from utils.projection_self_weight import validate_projection_self_weight
        validate_projection_self_weight(args.projection_self_weight, args.target_proj_mode)
        if args.target_proj_mode == 'projection_softmax' and args.num_clients == 1 and args.projection_self_weight is not None and args.projection_self_weight < 1:
            raise ValueError('Positive helper mass requires at least one helper')
    # Snapshot requested settings ONCE, before deriving comparison defaults.
    if not hasattr(args, 'control_requested_config'):
        keys = ('data_root','dataset','data_subdir','data_formats','partition','class_per_client',
                'data_expected_partition','data_expected_alpha','num_clients','num_classes','seed','target_client_id',
                'local_learning_rate','batch_size','local_epochs','global_rounds','eval_gap','local_grad_clip',
                'control_test_batch_size','num_server_models','smooth_mu','projection_self_weight',
                'target_group_ids','target_group_weight','projection_softmax_tau','group_softmax_tau',
                'fedfews_rng','fedfews_control_seed','fedfews_test_batch_size')
        args.control_requested_config = {key: getattr(args, key, None) for key in keys}
    if clip is None:
        args.local_grad_clip = 10.  # Prior comparison default, including FedFewS.
    args.target_prox_experiment = False
    args.data_formats = getattr(args, 'data_formats', None) or 'npz_only'
    if args.data_formats not in ('npz_only','pkl_first','pkl_only'):
        raise ValueError('Unknown explicit shard format policy')
    args.data_expected_partition = partition
    args.test_batch_size = args.control_test_batch_size
    if args.algorithm == 'FedFewS':
        if args.fedfews_test_batch_size not in (None, args.control_test_batch_size):
            raise ValueError('Conflicting FedFewS/control test batch')
        args.fedfews_rng, args.fedfews_control_seed = 'controlled', args.seed
        args.fedfews_test_batch_size = args.control_test_batch_size
    args.control_effective_config = {key: getattr(args, key, None) for key in args.control_requested_config}
    args.control_effective_config['gradient_norm_cap'] = effective_grad_clip(args)
    args.control_effective_config['seed_protocol'] = 'seed controls public initialization and all training streams; client internal seed0 retained, then streams reset'


def seed_control(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def resolve_data_settings(args, audit):
    """Record data-dependent mass defaults without changing the kernel arguments."""
    effective = args.control_effective_config
    effective['actual_data_alpha'] = audit['actual_alpha']
    effective['actual_data_partition'] = audit['config']['partition']
    effective['models_trained_per_client'] = args.num_server_models if args.algorithm == 'FedFewS' else 1
    samples = {r['client_id']: r['count'] for r in audit['files'] if r['split'] == 'train'}
    total = sum(samples.values())
    if args.algorithm == 'FedTargetProj' and args.target_proj_mode == 'group_projection_softmax':
        from utils.target_group_options import normalize_group_ids
        ids = normalize_group_ids(args.target_group_ids, args.num_clients)
        effective['target_group_ids'] = ids
        weights = {cid: n/total for cid, n in samples.items()}
        effective['target_group_weight'] = (math.fsum(weights[cid] for cid in ids)/math.fsum(weights.values())
                                            if args.target_group_weight is None else args.target_group_weight)
        effective['group_mass_rule'] = 'train_sample_share' if args.target_group_weight is None else 'explicit'
    if args.algorithm == 'FedTargetProj' and args.target_proj_mode == 'projection_softmax':
        effective['projection_self_weight'] = (samples[args.target_client_id]/total
            if args.projection_self_weight is None else args.projection_self_weight)
    return effective


def preview_control(args):
    from common.client_data import audit_image_clients
    from common.experiment import environment
    data = audit_image_clients(args)
    return dict(training_started=False, requested=args.control_requested_config, effective=resolve_data_settings(args, data),
        data=data, environment=environment(), model='common.models.CNN_512',
        model_kwargs=dict(in_channels=3,n_kernels=16,out_dim=args.num_classes,input_size=32),
        optimizer=dict(name='SGD',lr=args.local_learning_rate,momentum=0,weight_decay=0),
        gradient_norm_cap=effective_grad_clip(args), train_batch_size=args.batch_size, test_batch_size=args.test_batch_size,
        data_epochs_per_client=(args.num_server_models if args.algorithm=='FedFewS' else 1)*args.local_epochs,
        planned_train_count=args.global_rounds+1, actual_train_count=0,
        evaluation_loop_rounds=[r for r in range(args.global_rounds+1) if evaluation_due(args,r)])
