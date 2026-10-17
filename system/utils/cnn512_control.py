"""Opt-in homogeneous comparison protocol; legacy defaults are untouched."""
import math
import random

import numpy as np
import torch


def add_control_arguments(parser):
    parser.add_argument('--cnn512_control', action='store_true', help='Explicit shared CNN-512 comparison protocol')
    parser.add_argument('--local_grad_clip', type=float, default=None,
                        help='Optional positive gradient norm cap; omitted: TargetProj=10, FedFewS=off')
    parser.add_argument('--control_test_batch_size', type=int, default=16)
    parser.add_argument('--control_eval_scope', choices=['c0', 'all', 'group'], default='c0',
                        help='Post-local primary evaluation population; default preserves the C0 control')


def primary_scope(args):
    return getattr(args, 'control_eval_scope', 'c0')


def population_control(args):
    return getattr(args, 'cnn512_control', False) and primary_scope(args) != 'c0'


def full_target(args):
    return args.algorithm == 'FedTargetProj' and args.model_family == 'CNN-512'


def validate_control(args):
    if primary_scope(args) not in ('c0', 'all', 'group'):
        raise ValueError('control_eval_scope must be c0/all/group')
    clip = getattr(args, 'local_grad_clip', None)
    if clip is not None and (not math.isfinite(clip) or clip <= 0):
        raise ValueError('local_grad_clip must be finite and positive')
    if clip is not None and args.algorithm not in ('FedTargetProj', 'FedFewS'):
        raise ValueError('local_grad_clip is supported only by FedTargetProj/FedFewS')
    if full_target(args):
        if args.dataset != 'Cifar100' or args.num_classes != 100:
            raise ValueError('Full CNN-512 TargetProj currently supports RGB CIFAR-100/100 classes')
        if args.target_proj_mode not in ('avg', 'projection_softmax', 'group_projection_softmax'):
            raise ValueError('Full CNN-512 TargetProj supports avg/projection_softmax/group_projection_softmax')
        if (args.is_regular or args.target_prox_mu or args.target_head_mode != 'shared'
                or args.target_local_schedule != 'joint' or args.meta_c0_split
                or args.meta_collect_snapshots or args.head_diag_collect):
            raise ValueError('Full CNN-512 requires shared head, joint CE, no F/Prox/meta/head diagnostics')
    if not getattr(args, 'cnn512_control', False):
        if primary_scope(args) != 'c0':
            raise ValueError('control_eval_scope all/group requires cnn512_control')
        return
    if (args.algorithm not in ('FedTargetProj', 'FedFewS') or args.model_family != 'CNN-512'
            or args.dataset != 'Cifar100' or args.num_classes != 100 or args.num_clients != 20
            or args.target_client_id != 0 or args.seed != 0):
        raise ValueError('cnn512_control requires FedTargetProj/FedFewS, CNN-512, Cifar100, 20 clients/100 classes/C0/seed0')
    if (args.join_ratio != 1 or args.random_join_ratio or args.client_drop_rate or args.train_slow_rate
            or args.send_slow_rate or args.auto_break or args.learning_rate_decay or args.resume
            or args.few_shot or args.is_regular or args.target_prox_mu or args.meta_c0_split
            or args.meta_collect_snapshots or args.target_head_mode != 'shared'
            or args.target_local_schedule != 'joint' or args.head_diag_collect or args.time_select):
        raise ValueError('cnn512_control requires full participation, full data, joint/shared CE and no extra protocols')
    if (args.local_learning_rate != .005 or args.batch_size != 16 or args.local_epochs != 5
            or args.eval_gap != 1 or args.control_test_batch_size < 1):
        raise ValueError('cnn512_control requires lr=.005, batch16, local5, eval_gap1 and positive test batch')
    if clip not in (None, 10.):
        raise ValueError('cnn512_control fixes local_grad_clip=10')
    if not args.data_root or args.data_subdir != 'pat_20' or args.data_formats not in (None, 'npz_only'):
        raise ValueError('cnn512_control requires explicit data-root, data-subdir pat_20 and npz_only')
    if args.algorithm == 'FedTargetProj' and args.target_proj_mode == 'projection_softmax' and args.projection_self_weight != .125:
        raise ValueError('CNN-512 control ProjectionSoftmax requires projection_self_weight=.125')
    if args.algorithm == 'FedFewS' and (args.num_server_models != 3 or args.smooth_mu != .01):
        raise ValueError('CNN-512 control FedFewS requires K=3/smooth_mu=.01')
    scope = primary_scope(args)
    if scope == 'all' and not (args.algorithm == 'FedFewS' or args.target_proj_mode == 'avg'):
        raise ValueError('The all-client control supports only Avg/FedFewS')
    group_mode = args.algorithm == 'FedTargetProj' and args.target_proj_mode == 'group_projection_softmax'
    if scope == 'group' or group_mode:
        from utils.target_group_options import normalize_group_ids, DEFAULT_GROUP_IDS
        if (not group_mode or scope != 'group' or args.target_group_weight != .50
                or normalize_group_ids(args.target_group_ids, args.num_clients) != list(DEFAULT_GROUP_IDS)
                or args.projection_self_weight is not None):
            raise ValueError('Group control requires group_projection_softmax, scope=group, fixed seven IDs and beta=.50, no self weight')
    args.local_grad_clip = 10.
    args.target_prox_experiment = False
    args.data_formats = 'npz_only'
    args.data_expected_partition = 'pat'
    args.test_batch_size = args.control_test_batch_size
    if args.algorithm == 'FedFewS':
        if args.fedfews_test_batch_size not in (None, args.control_test_batch_size):
            raise ValueError('Conflicting FedFewS/control test batch')
        args.fedfews_rng, args.fedfews_control_seed = 'controlled', args.seed
        args.fedfews_test_batch_size = args.control_test_batch_size


def seed_control(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
