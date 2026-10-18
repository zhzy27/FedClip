"""Native standard FedFewS configuration; no dependency on compatibility code."""
import argparse
import math
import random

import numpy as np
import torch

from common.client_data import data_directory


def add_fedfews_arguments(parser):
    parser.add_argument('-nsm', '--num_server_models', type=int, default=3)
    parser.add_argument('-smu', '--smooth_mu', type=float, default=.01)
    parser.add_argument('--data-root', dest='data_root', default=None, help='Explicit existing image dataset root; no partition discovery')
    parser.add_argument('--data-subdir', dest='data_subdir', default=None, help='Optional exact suffix; do not also include it in -data')
    parser.add_argument('--data-formats', dest='data_formats', choices=['pkl_first', 'npz_only', 'pkl_only'], default=None)
    parser.add_argument('--data-expected-alpha', type=float, default=None)
    parser.add_argument('--data-expected-partition', choices=['dir', 'pat'], default=None)
    parser.add_argument('--fedfews_rng', choices=['original', 'controlled'], default='original')
    parser.add_argument('--fedfews_control_seed', type=int, default=0)
    parser.add_argument('--fedfews_torch_threads', type=int, default=None)
    parser.add_argument('--fedfews_test_batch_size', type=int, default=None)
    parser.add_argument('--fedfews_no_cache', action='store_true')
    parser.add_argument('--fedfews_preview', action='store_true', help='Audit existing data and print effective protocol; do not construct/train a model')
    parser.add_argument('--final_model_root', default=argparse.SUPPRESS, help='Structured final-model export root')


def validate_fedfews(args):
    if args.algorithm != 'FedFewS':
        if getattr(args, 'model_family', '') == 'FedAvgCNN-plain':
            raise ValueError('FedAvgCNN-plain entry is currently supported only by FedFewS')
        return
    if args.model_family not in ('FedAvgCNN-plain', 'CNN-512') or args.num_classes != 100 or 'cifar100' not in args.dataset.lower():
        raise ValueError('FedFewS requires -m FedAvgCNN-plain or CNN-512, CIFAR-100, 100 classes')
    if args.num_server_models < 1 or args.num_clients < 1 or args.global_rounds < (0 if getattr(args, 'cnn512_control', False) else 1):
        raise ValueError('FedFewS requires positive K/clients and >=2 inclusive training rounds')
    if min(args.batch_size, args.local_epochs, args.eval_gap) < 1:
        raise ValueError('Batch size, local epochs and evaluation gap must be positive')
    if not math.isfinite(args.smooth_mu) or args.smooth_mu <= 0 or not math.isfinite(args.local_learning_rate) or args.local_learning_rate <= 0:
        raise ValueError('FedFewS smooth_mu and learning rate must be finite and positive')
    if args.join_ratio != 1 or any(getattr(args, k, False) for k in
        ['random_join_ratio', 'client_drop_rate', 'train_slow_rate', 'send_slow_rate', 'auto_break',
         'learning_rate_decay', 'time_select', 'few_shot', 'resume', 'models_folder_name',
         'target_prox_mu', 'meta_c0_split', 'meta_collect_snapshots', 'meta_weight_file', 'meta_snapshot_dir',
         'head_diag_collect', 'head_diag_dir']):
        raise ValueError('FedFewS standard requires full participation, no drop/slow/early-stop/decay/resume/few-shot/Prox/meta/head diagnostics')
    if args.target_head_mode != 'shared' or args.target_local_schedule != 'joint' or args.is_regular:
        raise ValueError('FedFewS uses only full-model CE: no private head, alternating training or F regularizer')
    for name in ['fedfews_torch_threads', 'fedfews_test_batch_size']:
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f'{name} must be positive')
    if args.data_expected_alpha is not None and (not math.isfinite(args.data_expected_alpha) or args.data_expected_alpha <= 0):
        raise ValueError('Expected data alpha must be finite and positive')
    if not 0 <= args.prev < args.times:
        raise ValueError('Require 0 <= prev < times')
    data_directory(args)  # Validate the explicit path policy before training.
    # Source test-batch exact-name rule, applied to the full dataset string.
    full_dataset = args.dataset + ('/' + args.data_subdir if args.data_subdir else '')
    args.test_batch_size = args.fedfews_test_batch_size or (1024 if full_dataset == 'Cifar100' else 256)
    args.use_cache = not args.fedfews_no_cache
    args.use_rep_mode = False
    if not hasattr(args, 'final_model_root'):
        args.final_model_root = './final_models'


def prepare_fedfews_random(args):
    # Same order as the accepted compatibility entry; never invoke main.set_seed
    # (which changes backend flags). Reset ONCE per process, not per repetition.
    if args.fedfews_rng == 'controlled':
        random.seed(args.fedfews_control_seed)
        np.random.seed(args.fedfews_control_seed)
        torch.manual_seed(args.fedfews_control_seed)
    torch.manual_seed(0)
    if args.fedfews_torch_threads is not None:
        torch.set_num_threads(args.fedfews_torch_threads)
