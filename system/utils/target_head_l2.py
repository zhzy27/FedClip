"""Explicit local-fc3 weight L2 and isolated observations for the Avg ablation.

The ordinary CE/F objective and zero-coefficient training path remain separate.
Imports used by the command launcher do not load Torch.
"""

import copy
import json
import math
import os
from pathlib import Path


def add_head_l2_options(parser):
    parser.add_argument('--target_head_l2_lambda', type=float, default=0.,
                        help='Explicit lambda/2 * sum(fc3.weight**2); local-head Avg joint only.')
    parser.add_argument('--target_head_l2_diagnostics', type=int, choices=(0, 1), default=1,
                        help='Read-only observations/exports for nonzero head L2; 0 disables observation.')


def validate_head_l2(args):
    coefficient = float(getattr(args, 'target_head_l2_lambda', 0.))
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('target_head_l2_lambda must be finite and nonnegative.')
    if coefficient == 0:
        return
    if (getattr(args, 'algorithm', '') != 'FedTargetProj'
            or getattr(args, 'target_proj_mode', '') != 'avg'
            or getattr(args, 'target_head_mode', 'shared') != 'local'
            or getattr(args, 'target_local_schedule', 'joint') != 'joint'
            or getattr(args, 'model_family', '') != 'Decom_CNN-5-512'
            or getattr(args, 'target_prox_mu', 0.) != 0):
        raise ValueError('Nonzero target_head_l2_lambda supports only FedTargetProj avg + local head + '
                         'joint + Decom_CNN-5-512 with target_prox_mu=0.')
    if getattr(args, 'cnn512_control', False) or getattr(args, 'meta_c0_split', '') or getattr(args, 'meta_collect_snapshots', False):
        raise ValueError('Head L2 does not support control/meta protocols.')


def head_l2_tag(coefficient):
    return 'head_l2_' + repr(float(coefficient)).replace('.', 'p').replace('-', 'm').replace('+', '')


def ensure_head_l2_names(args):
    if getattr(args, 'target_head_l2_lambda', 0.) <= 0:
        return
    tag = head_l2_tag(args.target_head_l2_lambda)
    if tag not in getattr(args, 'exp_name', ''):
        args.exp_name = getattr(args, 'exp_name', '') + '_' + tag
    defaults = dict(save_folder_name='temp', final_model_root='final_models', h5_result_root='h5_results',
                    training_log_root='training_logs')
    for key in (*defaults, 'run_log_dir'):
        value = getattr(args, key, defaults.get(key, ''))
        if value and tag not in value:
            setattr(args, key, os.path.join(value, tag))


def l2_weight(model):
    from utils.target_head import classifier_names
    names = classifier_names(model)  # Validate aliases by identity; never iterate head + fc3 twice.
    return dict(model.named_parameters())[next(n for n in names if n == 'fc3.weight')]


def head_l2_penalty(weight, coefficient):
    return (coefficient / 2.) * weight.square().sum()


def head_weight_norm(model):
    return float(l2_weight(model).detach().double().norm())


def diagnostics_enabled(args):
    return getattr(args, 'target_head_l2_lambda', 0.) > 0 and bool(getattr(args, 'target_head_l2_diagnostics', 1))


def policy(args):
    return dict(target_head_l2_lambda=args.target_head_l2_lambda,
        target_head_l2_diagnostics=bool(getattr(args, 'target_head_l2_diagnostics', 1)),
        parameter_names=['fc3.weight'], excluded_classifier_parameters=['fc3.bias'],
        formula='lambda_h/2 * sum(fc3.weight**2)', applies_to_client_ids=list(range(args.num_clients)),
        target_local_schedule='joint', optimizer_weight_decay=0.,
        loss_means='training process means weighted by actual sample visits; weighted terms already include lambda',
        endpoint_loss='CE only; same saved ordinary post-local C0 model before aggregation',
        accuracy_unit='fraction', round='completed_round=loop_round+1')


def initialize(server):
    if getattr(server.args, 'target_head_l2_lambda', 0.) > 0:
        server.head_l2_policy = policy(server.args)
        server.head_l2_clients, server.head_l2_target = [], []
        print('[HeadL2] ' + json.dumps(server.head_l2_policy, sort_keys=True))


def record_client(server, client):
    if not diagnostics_enabled(server.args):
        return
    row = dict(client.last_head_l2_train_stats, mode=server.target_proj_mode, seed=server.seed)
    if row['loop_round'] != server.cur_ground or row['client_id'] != client.id:
        raise RuntimeError('Stale or mismatched head L2 training statistics.')
    server.head_l2_clients.append(row)
    server._append_csv('target_head_l2_clients.csv', [row])


def observe_target(server):
    if not diagnostics_enabled(server.args):
        return
    import torch
    from utils.target_head import preserved_head_rng
    client = server.clients[server.target_client_id]
    # The loaded checkpoint and the inference copy are both inside RNG protection.
    # Copying also protects buffers/mixed modes if a model has inference side effects.
    with preserved_head_rng(), torch.no_grad():
        candidate = copy.deepcopy(client._load_model())
        candidate.eval()
        row = dict(round=server.cur_ground + 1, loop_round=server.cur_ground, client_id=client.id,
            mode=server.target_proj_mode, seed=server.seed, target_head_l2_lambda=server.args.target_head_l2_lambda,
            head_weight_norm=head_weight_norm(candidate))
        for name, loader in (('train', client.load_train_data()), ('test', client.load_test_data())):
            loss_sum, correct, samples = 0., 0, 0
            for x, y in loader:
                x, y = client._move_batch(x, y)
                output = candidate(x)
                loss_sum += float(client.loss(output, y)) * len(y)
                correct += int((output.argmax(1) == y).sum())
                samples += len(y)
            if samples <= 0:
                raise ValueError('Head L2 endpoint observation requires nonempty train/test data.')
            row.update({f'endpoint_{name}_ce': loss_sum / samples, f'endpoint_{name}_correct': correct,
                        f'endpoint_{name}_samples': samples, f'endpoint_{name}_accuracy': correct / samples})
    if row['endpoint_test_accuracy'] != server._post_local_acc:
        raise RuntimeError('Head L2 endpoint must match ordinary post-local primary accuracy.')
    server.head_l2_target.append(row)
    server._append_csv('target_head_l2_target.csv', [row])
    print('[HeadL2TargetEndpoint] ' + json.dumps(row, sort_keys=True))


def save_json(server):
    if not diagnostics_enabled(server.args):
        return
    for suffix, rows in (('clients', server.head_l2_clients), ('target', server.head_l2_target)):
        document = dict(policy=server.head_l2_policy, history=rows)
        if suffix == 'target':
            document['primary_summary'] = server._local_accuracy_summary()
        (Path(server.report_dir()) / f'target_head_l2_{suffix}.json').write_text(
            json.dumps(document, indent=2, allow_nan=False), encoding='utf-8')


def save_h5(server, parent):
    group = parent.create_group('target_head_l2')
    group.attrs['policy_json'] = json.dumps(server.head_l2_policy)
    for suffix, rows in (('clients', server.head_l2_clients), ('target', server.head_l2_target)):
        child = group.create_group(suffix)
        if not rows:
            continue
        for key, value in rows[0].items():
            if isinstance(value, (bool, float, int)):
                child.create_dataset(key, data=[r[key] for r in rows])
            else:
                child.attrs[key] = value
