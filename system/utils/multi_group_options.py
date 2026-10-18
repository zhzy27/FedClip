"""Independent fixed-K, fixed-membership multi-group protocol."""
import math


def add_multi_group_arguments(parser):
    parser.add_argument('--multi_warmup_rounds', type=int, default=0, help='All-client Avg rounds INCLUDED in gr+1; no extra training')
    parser.add_argument('--multi_group_basis', choices=['single_delta', 'full_weights', 'delta_history'], default='single_delta')
    parser.add_argument('--multi_history_window', type=int, default=None, help='delta_history only: positive tail window <= warmup; omitted=all warmup rounds')
    parser.add_argument('--multi_group_publish', default='', help='Atomically publish grouping at formation; refuses to overwrite existing files')
    parser.add_argument('--multi_group_wait_seconds', type=float, default=0., help='Explicit bounded wait for multi_group_file at formation; zero=missing file error')
    parser.add_argument('--multi_group_poll_seconds', type=float, default=1., help='Wait polling interval; positive, no training RNG used')
    parser.add_argument('--multi_group_mode', choices=['group_projection_softmax', 'group_avg_only'], default='group_projection_softmax')
    parser.add_argument('--multi_group_sizes', default=None, help='Positive comma-separated capacities, sum=N; canonical order is descending')
    parser.add_argument('--multi_group_weight', type=float, default=.5, help='Total within-group mass (outside mass=1-beta)')
    parser.add_argument('--multi_group_file', default='', help='Reuse a compatible fixed_groups.json; no reclustering')
    parser.add_argument('--multi_cluster_seed', type=int, default=0, help='Independent clustering RNG, never advances training RNG')
    parser.add_argument('--multi_cluster_restarts', type=int, default=8)
    parser.add_argument('--multi_cluster_iterations', type=int, default=50)
    parser.add_argument('--multi_no_pre_eval', action='store_true', help='Disable observation before training; does not change training')
    parser.add_argument('--multi_preview', action='store_true', help='Audit data and show protocol without constructing models or training')


def group_sizes(value, n, k):
    if not 1 <= k <= n:
        raise ValueError('Require 1 <= num_server_models <= num_clients: every group is nonempty')
    if value is None:
        return [n // k + (i < n % k) for i in range(k)]
    try:
        values = [int(v.strip()) for v in value.split(',')] if isinstance(value, str) else list(value)
    except (ValueError, TypeError) as error:
        raise ValueError('multi_group_sizes must be K positive integer capacities') from error
    if len(values) != k or any(type(v) is not int or v < 1 for v in values) or sum(values) != n:
        raise ValueError('multi_group_sizes must contain K positive integers summing to N')
    return sorted(values, reverse=True)


def validate_multi_group(args):
    if args.algorithm != 'FedMultiGroupProj':
        for key, default in dict(multi_warmup_rounds=0, multi_group_basis='single_delta', multi_history_window=None,
            multi_group_publish='', multi_group_wait_seconds=0., multi_group_poll_seconds=1.).items():
            if getattr(args,key,default) != default:
                raise ValueError(f'{key} requires algorithm FedMultiGroupProj; refusing to ignore it')
        return
    # Attribute defaults also preserve existing programmatic callers.
    for key, default in dict(multi_warmup_rounds=0, multi_group_basis='single_delta', multi_history_window=None,
        multi_group_publish='', multi_group_wait_seconds=0., multi_group_poll_seconds=1.).items():
        if not hasattr(args, key):
            setattr(args, key, default)
    if type(args.multi_warmup_rounds) is not int or args.multi_warmup_rounds < 0:
        raise ValueError('multi_warmup_rounds must be a nonnegative integer')
    if args.multi_group_basis not in ('single_delta', 'full_weights', 'delta_history'):
        raise ValueError('Unsupported multi_group_basis')
    if args.multi_group_basis == 'delta_history':
        if args.multi_warmup_rounds < 1:
            raise ValueError('delta_history requires positive warmup rounds')
        window = args.multi_warmup_rounds if args.multi_history_window is None else args.multi_history_window
        if type(window) is not int or not 1 <= window <= args.multi_warmup_rounds:
            raise ValueError('delta_history requires 1 <= history_window <= warmup_rounds')
    elif args.multi_history_window is not None:
        raise ValueError('multi_history_window applies only to delta_history; refusing to ignore it')
    if args.global_rounds < args.multi_warmup_rounds:
        raise ValueError('gr+1 must include all warmup rounds AND at least one grouped round')
    if not math.isfinite(args.multi_group_wait_seconds) or args.multi_group_wait_seconds < 0:
        raise ValueError('multi_group_wait_seconds must be finite and nonnegative')
    if not math.isfinite(args.multi_group_poll_seconds) or args.multi_group_poll_seconds <= 0:
        raise ValueError('multi_group_poll_seconds must be finite and positive')
    if args.multi_group_wait_seconds and not args.multi_group_file:
        raise ValueError('Waiting requires multi_group_file')
    if not args.multi_group_wait_seconds and args.multi_group_poll_seconds != 1.:
        raise ValueError('Nondefault polling interval requires waiting to be enabled')
    if args.multi_group_publish and args.multi_group_file:
        raise ValueError('Publish or reuse a group file, not both in one run')
    if args.model_family != 'CNN-512' or args.dataset != 'Cifar100' or args.num_classes != 100:
        raise ValueError('FedMultiGroupProj currently supports complete CNN-512, RGB32 CIFAR-100/100 classes')
    for key in ('num_clients', 'num_server_models', 'batch_size', 'local_epochs', 'eval_gap',
                'control_test_batch_size', 'multi_cluster_restarts', 'multi_cluster_iterations'):
        value = getattr(args, key)
        if type(value) is not int or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if args.global_rounds < 0 or not 0 <= args.seed < 2**32 or not 0 <= args.multi_cluster_seed < 2**32:
        raise ValueError('Require gr>=0 and seed/cluster seed in [0,2**32)')
    for key in ('local_learning_rate', 'group_softmax_tau'):
        value = getattr(args, key)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(args.multi_group_weight) or not 0 <= args.multi_group_weight <= 1:
        raise ValueError('multi_group_weight must be finite in [0,1]')
    if args.local_grad_clip is not None and (not math.isfinite(args.local_grad_clip) or args.local_grad_clip < 0):
        raise ValueError('local_grad_clip must be finite and nonnegative; zero disables')
    forbidden = ['random_join_ratio', 'client_drop_rate', 'train_slow_rate', 'send_slow_rate',
                 'auto_break', 'learning_rate_decay', 'time_select', 'few_shot', 'resume', 'models_folder_name',
                 'is_regular', 'target_prox_mu', 'meta_c0_split', 'meta_collect_snapshots', 'meta_weight_file',
                 'meta_snapshot_dir', 'head_diag_collect', 'head_diag_dir']
    if args.join_ratio != 1 or any(getattr(args, key, False) for key in forbidden):
        raise ValueError('Multi-group requires fresh full participation and CE-only training; drop/slow/resume/decay/F/Prox/meta/diagnostics unsupported')
    if args.target_head_mode != 'shared' or args.target_local_schedule != 'joint':
        raise ValueError('Multi-group requires shared classifier and joint training')
    if args.control_eval_scope == 'group':
        raise ValueError('Multi-group primary scope is all clients by fixed membership, not a single group')
    if any(getattr(args, key, None) is not None for key in ('target_group_ids', 'target_group_weight', 'projection_self_weight', 'feedback_anchor_weight')):
        raise ValueError('Use multi_group_sizes/file/weight; single-target/group/feedback parameters do not apply')
    if args.projection_softmax_tau != .2 or args.fedfews_preview:
        raise ValueError('Use group_softmax_tau / multi_preview; no single-target or Few selection protocol')
    if not 0 <= args.prev < args.times:
        raise ValueError('Require 0 <= prev < times')
    if not args.data_root or not args.data_subdir or args.partition not in ('pat', 'dir'):
        raise ValueError('Specify exact existing data-root/data-subdir and -pt pat/dir')
    if args.partition == 'pat' and not 1 <= args.class_per_client <= args.num_classes:
        raise ValueError('pat class_per_client must be in [1,num_classes]')
    if args.data_expected_partition is not None and args.data_expected_partition != args.partition:
        raise ValueError('data-expected-partition conflicts with -pt')
    if args.data_expected_alpha is not None and (not math.isfinite(args.data_expected_alpha) or args.data_expected_alpha <= 0):
        raise ValueError('data-expected-alpha must be finite and positive')
    args.multi_capacities = group_sizes(args.multi_group_sizes, args.num_clients, args.num_server_models)
    if args.num_server_models == 1 and args.multi_group_mode == 'group_projection_softmax' and args.multi_group_weight < 1:
        raise ValueError('K=1 contains all clients: positive outside mass has no outside members; use beta=1')
    if not hasattr(args, 'multi_requested_config'):
        args.multi_requested_config = {key: getattr(args, key, None) for key in
            ('num_clients', 'num_server_models', 'seed', 'multi_group_sizes', 'multi_group_weight', 'group_softmax_tau',
             'local_grad_clip', 'batch_size', 'local_epochs', 'local_learning_rate', 'global_rounds', 'eval_gap',
             'control_test_batch_size', 'data_formats', 'data_expected_partition', 'multi_group_mode', 'multi_group_file',
             'multi_cluster_seed', 'multi_cluster_restarts', 'multi_cluster_iterations', 'multi_no_pre_eval', 'control_eval_scope',
             'multi_warmup_rounds', 'multi_group_basis', 'multi_history_window', 'multi_group_publish',
             'multi_group_wait_seconds', 'multi_group_poll_seconds')}
    # Opt into the existing strict image audit and seed-forked public model factory.
    args.cnn512_control = True
    args.control_eval_scope = 'all'
    args.local_grad_clip = 10. if args.local_grad_clip is None else args.local_grad_clip
    args.test_batch_size = args.control_test_batch_size
    args.data_formats = args.data_formats or 'npz_only'
    args.data_expected_partition = args.partition
    args.control_effective_config = {}  # Filled from audited metadata, never folder names.


def multi_preview(args):
    from common.client_data import audit_image_clients
    from common.experiment import environment
    from utils.multi_group_protocol import grouping_protocol
    return dict(training_started=False, algorithm=args.algorithm, requested=args.multi_requested_config,
                grouping_protocol=grouping_protocol(args), planned_warmup_rounds=args.multi_warmup_rounds,
                planned_grouped_rounds=args.global_rounds+1-args.multi_warmup_rounds,
                effective=vars(args), data=audit_image_clients(args), environment=environment(),
                data_epochs_per_client=args.num_server_models*args.local_epochs,
                planned_train_count=args.global_rounds+1, actual_train_count=0,
                evaluation_loop_rounds=[r for r in range(args.global_rounds+1) if r % args.eval_gap == 0 or r == args.global_rounds])
