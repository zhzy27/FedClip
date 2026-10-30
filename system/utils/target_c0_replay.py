"""Opt-in paired target TRAIN batches; no change to aggregation sample counts."""
import hashlib
import json
import math
import os
from pathlib import Path


def add_options(parser):
    parser.add_argument('--target_c0_replay_mode', choices=['none', 'batch_pair'], default='none')
    parser.add_argument('--target_c0_replay_weight', type=float, default=.5)
    parser.add_argument('--target_c0_replay_diagnostics', type=int, choices=[0, 1], default=1)


def enabled(args):
    return (getattr(args, 'target_c0_replay_mode', 'none') == 'batch_pair'
            and getattr(args, 'target_c0_replay_weight', .5) > 0)


def effective_weight(args, client_id):
    return args.target_c0_replay_weight if enabled(args) and client_id != args.target_client_id else 0.


def validate(args):
    weight = getattr(args, 'target_c0_replay_weight', .5)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('target_c0_replay_weight must be finite and in [0,1]')
    if getattr(args, 'target_c0_replay_mode', 'none') not in ('none', 'batch_pair'):
        raise ValueError('target_c0_replay_mode must be none or batch_pair')
    if getattr(args, 'target_c0_replay_diagnostics', 1) not in (0, 1):
        raise ValueError('target_c0_replay_diagnostics must be 0/1')
    if not enabled(args):
        return  # In particular, batch_pair + zero performs NO extra read/loader/forward.
    if (getattr(args, 'algorithm', '') != 'FedTargetProj'
            or getattr(args, 'target_proj_mode', '') != 'projection_softmax'
            or getattr(args, 'model_family', '') != 'Decom_CNN-5-512'
            or getattr(args, 'target_head_mode', 'shared') != 'shared'
            or getattr(args, 'target_local_schedule', 'joint') != 'joint'):
        raise ValueError('Target replay supports only FedTargetProj projection_softmax, Decom_CNN-5-512, shared head, joint')
    if (getattr(args, 'target_prox_mu', 0.) or getattr(args, 'target_head_l2_lambda', 0.)
            or getattr(args, 'target_z2_lambda', 0.) or getattr(args, 'cnn512_control', False)
            or getattr(args, 'meta_c0_split', '') or getattr(args, 'meta_collect_snapshots', False)
            or getattr(args, 'meta_weight_file', '') or getattr(args, 'head_diag_collect', False)
            or getattr(args, 'resume', False) or getattr(args, 'models_folder_name', '')
            or getattr(args, 'learning_rate_decay', False)):
        raise ValueError('Target replay forbids Prox/head L2/Z2/control/meta/head_diag/resume/pretrained/decay combinations')
    if (getattr(args, 'join_ratio', 1.) != 1. or getattr(args, 'random_join_ratio', False)
            or getattr(args, 'client_drop_rate', 0.) or getattr(args, 'train_slow_rate', 0.)
            or getattr(args, 'send_slow_rate', 0.) or getattr(args, 'few_shot', 0)
            or getattr(args, 'auto_break', False) or getattr(args, 'time_select', False)):
        raise ValueError('Target replay requires ordinary full participation, full data/epochs and no slow/drop/early-stop')
    n, target = getattr(args, 'num_clients', 20), getattr(args, 'target_client_id', 0)
    if type(n) is not int or n < 2 or type(target) is not int or not 0 <= target < n:
        raise ValueError('Target replay requires at least two clients and a valid target_client_id')


def tag(args):
    return f'projection_softmax_replay_t{getattr(args,"target_client_id",0)}_pair{args.target_c0_replay_weight:g}'.replace('.', 'p')


def ensure_names(args):
    if not enabled(args):
        return
    value = tag(args)
    if value not in getattr(args, 'exp_name', ''):
        args.exp_name = getattr(args, 'exp_name', '') + '_' + value
    for key, default in dict(save_folder_name='temp', final_model_root='final_models', h5_result_root='h5_results',
                             training_log_root='training_logs', run_log_dir='', training_log_dir='').items():
        path = getattr(args, key, default)
        if path and value not in path:
            setattr(args, key, os.path.join(path, value))


class TargetBatchStream:
    """One helper/round-local sampler; its iterator AND shuffle use a private RNG.

    The source reader is exactly the ordinary client's reader, with is_train=True.
    No source iterator is shared between helpers; wraps continue its private RNG.
    """
    def __init__(self, client, loop_round):
        import torch
        from common.data_loading import make_data_loader
        from flcore.clients import clientbase
        from utils.target_head import preserved_head_rng
        source = client.args.target_client_id
        seed_key = f'target-replay-v1:{getattr(client.args,"seed",0)}:{loop_round}:{client.id}:{source}'
        self.seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], 'little') % (2**63)
        generator = torch.Generator().manual_seed(self.seed)
        with preserved_head_rng():
            data = clientbase.read_client_data(client.dataset, source, is_train=True,
                                               few_shot=client.few_shot, args=client.args)
            if len(data) == 0:
                raise ValueError('Target replay requires nonempty target TRAIN data')
            self.loader = make_data_loader(data, client.batch_size, shuffle=True, drop_last=False,
                                          num_workers=0, generator=generator)
        self.iterator = None
        self.wraps = 0

    def next(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.wraps += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)


def initialize(server, model):
    if not enabled(server.args):
        return
    from utils.head_snapshot import tensor_fingerprint
    from utils.cnn512_control import effective_grad_clip
    from utils.target_head import preserved_head_rng
    from flcore.clients.clientbase import load_item
    from common.experiment import sha256
    from utils.data_utils import get_dataset_sub_dir
    args = server.args
    explicit = getattr(args, 'data_root', None)
    if explicit:
        from common.client_data import data_directory, client_file, image_policy
        directory = data_directory(args); formats, _ = image_policy(args)
    else:
        directory = (Path('../dataset') / args.dataset / get_dataset_sub_dir(args)).resolve(); formats = ('.npz',)
    files, capacities = [], []
    with preserved_head_rng():
        for c in server.clients:
            local = load_item(c.role, 'model', c.save_folder_name)
            capacities.append(dict(client_id=c.id, rank_rate=local.ratio_LR,
                trainable_parameters=sum(p.numel() for p in local.parameters() if p.requires_grad),
                train_samples=c.train_samples, test_samples=c.test_samples))
            del local
            for train in (True, False):
                path = client_file(args, c.id, train) if explicit else directory / ('train' if train else 'test') / f'{c.id}.npz'
                files.append(dict(client_id=c.id, split='train' if train else 'test', path=str(path.resolve()),
                                  sha256=sha256(path) if path.is_file() else None,
                                  samples=c.train_samples if train else c.test_samples))
    server.target_c0_replay_clients = []
    server.target_c0_replay_budget = dict(local_train_calls=0, own_batches=0, c0_batches=0,
        own_sample_visits=0, c0_sample_visits=0, optimizer_steps=0, train_seconds=0.)
    server.target_c0_replay_policy = dict(replay_mode=args.target_c0_replay_mode,
        replay_source_client_id=server.target_client_id, configured_weight=args.target_c0_replay_weight,
        diagnostics=bool(getattr(args, 'target_c0_replay_diagnostics', 1)),
        applies_to_client_ids=[c.id for c in server.clients if c.id != server.target_client_id],
        target_effective_weight=0., seed=server.seed,
        formula='(1-alpha)*mean_CE_own + alpha*mean_CE_target_train + original_weighted_F (once)',
        CE_mean_unit='each source weighted by its own actual sample visits',
        total_and_F_mean_unit='arithmetic mean over actual SGD objectives/steps, not endpoint evaluation loss',
        sampler='private Torch Generator: SHA256(target-replay-v1:seed:loop_round:helper_id:target_id); new per helper/round; wraps continue generator',
        training_budget='own loader alone drives epochs and optimizer steps; extra target visits and two forwards per helper step; target unchanged',
        aggregation_samples='original own shard train_samples only; replay never increases aggregation mass',
        evaluation_data='original own train/test definitions, never replay mixture',
        optimizer=dict(name='SGD', lr=server.learning_rate, momentum=0., weight_decay=0., grad_norm_cap=effective_grad_clip(args)),
        batch_size=server.batch_size, local_epochs=server.local_epochs, planned_train_count=server.global_rounds+1,
        regularization_enabled=args.is_regular, regularization_lambda=args.regular_lamda,
        projection_self_weight=getattr(args, 'projection_self_weight', None),
        projection_softmax_tau=getattr(args, 'projection_softmax_tau', .2),
        initialization='unchanged legacy Model_Distribe, seed and SVD download/recovery',
        initial_server_state_sha256=tensor_fingerprint(model.state_dict()),
        actual_client_capacities=capacities, data_directory=str(directory), format_policy=list(formats), data_files=files,
        data_fingerprint_scope='actual selected file SHA256; no extra dataset traversal',
        data_config_sha256=sha256(directory/'config.json') if (directory/'config.json').is_file() else None,
        primary_metric='target_post_local_acc', accuracy_unit='fraction',
        aggregate_metric='target_client_test_acc (post-aggregation download, no extra training)')
    print('[TargetReplayPolicy] ' + json.dumps(server.target_c0_replay_policy, sort_keys=True))


def record_client(server, client):
    if not enabled(server.args):
        return
    row = dict(client.last_c0_replay_train_stats, mode=server.target_proj_mode, seed=server.seed)
    if row['loop_round'] != server.cur_ground or row['client_id'] != client.id:
        raise RuntimeError('Stale target replay training record')
    server.target_c0_replay_budget['local_train_calls'] += 1
    for key in ('own_batches', 'c0_batches', 'own_sample_visits', 'c0_sample_visits', 'optimizer_steps', 'train_seconds'):
        server.target_c0_replay_budget[key] += row[key]
    if not getattr(server.args, 'target_c0_replay_diagnostics', 1):
        return
    server.target_c0_replay_clients.append(row)
    server._append_csv('target_c0_replay_clients.csv', [row])
    with (Path(server.report_dir())/'target_c0_replay.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')


def primary_rows(server):
    return [dict(loop_round=r['loop_round'], completed_train_count=r['round'], client_id=server.target_client_id,
                 target_post_local_acc=r['target_post_local_acc'], target_aggregated_download_acc=r['target_client_test_acc'])
            for r in server.target_proj_history]


def save_json(server):
    if not enabled(server.args):
        return
    import csv
    from common.experiment import write_json
    root = Path(server.report_dir()); summary = server._local_accuracy_summary(); rows = primary_rows(server)
    write_json(root/'target_c0_replay_clients.json', dict(policy=server.target_c0_replay_policy,
        actual_budget=server.target_c0_replay_budget, history=server.target_c0_replay_clients))
    write_json(root/'target_c0_replay_primary.json', dict(history=rows, summary=summary, accuracy_unit='fraction'))
    write_json(root/'target_c0_replay_summary.json', dict(primary_metric='target_post_local_acc', accuracy_unit='fraction', **summary))
    if rows:
        with (root/'target_c0_replay_primary.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_h5(server, parent):
    if not enabled(server.args):
        return
    import h5py
    group = parent.create_group('target_c0_replay')
    group.attrs['policy_json'] = json.dumps(server.target_c0_replay_policy)
    group.attrs['actual_budget_json'] = json.dumps(server.target_c0_replay_budget)
    group.attrs['summary_json'] = json.dumps(server._local_accuracy_summary())
    for name, rows in [('clients', server.target_c0_replay_clients), ('primary', primary_rows(server))]:
        group.create_dataset(name+'_json', data=[json.dumps(r, allow_nan=False) for r in rows], dtype=h5py.string_dtype('utf-8'))
    root = parent.file
    if 'rs_test_acc' in root:
        root.create_dataset('legacy_server_diagnostic_test_acc', data=root['rs_test_acc'][:]); del root['rs_test_acc']
    root.create_dataset('rs_test_acc', data=[r['target_post_local_acc'] for r in server.target_proj_history])
    root.attrs['rs_test_acc_scope'] = 'ordinary_target_post_local_before_aggregation'
    root.attrs['legacy_server_diagnostic_test_acc_scope'] = 'original all-client evaluate() at pre-download timing'
    root.attrs['rs_train_loss_scope'] = 'original evaluation CE plus F on own data; not paired training loss'


def save_manifest(server):
    if not enabled(server.args):
        return
    path = Path(server.final_model_dir())/'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['target_c0_replay'] = dict(policy=server.target_c0_replay_policy, primary_summary=server._local_accuracy_summary(),
        actual_budget=server.target_c0_replay_budget,
        actual_train_count=len(server.target_proj_history), ordinary_target_model=f'Client_{server.target_client_id}_model.pt',
        metrics_dir=str(Path(server.report_dir()).resolve()))
    manifest['legacy_server_diagnostic_best_accuracy'] = manifest.get('best_accuracy')
    manifest['best_accuracy'] = server._local_accuracy_summary()['best_target_local_acc']
    manifest['primary_accuracy_metric'] = 'target_post_local_acc'
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
