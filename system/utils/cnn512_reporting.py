"""Read-only comparison artifacts; no new loader traversal or random draws."""
import csv
import hashlib
from pathlib import Path
import sys

import h5py
import torch

from common.client_data import audit_image_clients
from common.experiment import tensor_digest, write_json, environment, code_version
from common.metrics import summarize_evaluations, sample_weighted_accuracy
from utils.cnn512_control import primary_scope, population_control, effective_grad_clip, resolve_data_settings
from utils.training_logs import mirror_result


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor_digest(tensor).encode())
    return h.hexdigest()


def initialize_control(server, model, clients):
    if not getattr(server.args, 'cnn512_control', False):
        return
    folder = Path(server.report_dir())
    audit = audit_image_clients(server.args)
    resolve_data_settings(server.args, audit)
    write_json(folder / 'data_manifest.json', audit)
    digest = model_digest(model)
    client_hashes = [model_digest(m) for m in clients]
    server_hashes = [model_digest(m) for m in getattr(server, 'global_model_set', [model])]
    if any(h != digest for h in client_hashes):
        raise AssertionError('Common initial parameters differ')
    if any(h != digest for h in server_hashes):
        raise AssertionError('Common initial K-server parameters differ')
    server.control_points, server.control_training = [], []
    server._control_training_rounds = set()
    server.primary_points, server.control_client_metrics = [], []
    server.control_manifest = dict(protocol='cnn512_control', args=vars(server.args),
        command=[sys.executable] + sys.argv, environment=environment(),
        code=code_version(Path(__file__).resolve().parents[2]), model='common.models.CNN_512',
        model_kwargs=dict(in_channels=3, n_kernels=16, out_dim=server.args.num_classes, input_size=32),
        initial_parameter_sha256=digest, initial_client_model_hashes=client_hashes,
        initial_server_model_hashes=server_hashes,
        parameter_shapes={n: list(p.shape) for n,p in model.named_parameters()},
        data_manifest='data_manifest.json', objective='CE only', gradient_norm_cap=effective_grad_clip(server.args),
        requested_config=server.args.control_requested_config, effective_config=server.args.control_effective_config,
        optimizer=dict(name='SGD', lr=server.args.local_learning_rate, momentum=0, weight_decay=0),
        loaders=dict(train=dict(batch_size=server.args.batch_size, shuffle=True, drop_last=False, num_workers=0),
                     test=dict(batch_size=server.args.test_batch_size, shuffle=False, drop_last=False, num_workers=0)),
        buffers='Not aggregated; local buffers retained on download (CNN-512 has no buffers)',
        primary_metric=f'{primary_scope(server.args)} ordinary post-local; FedFewS selected by training endpoint CE',
        primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
        primary_weighting='test_sample_count', primary_accuracy_unit='percent',
        data_epochs_per_client=(server.K if server.args.algorithm == 'FedFewS' else 1)*server.args.local_epochs,
        planned_train_count=server.global_rounds+1, actual_train_count=0, evaluation_count=0,
        paths=dict(checkpoints=server.save_folder_name, logs=server.report_dir(), final_models=server.final_model_dir()))
    if population_control(server.args):
        server.control_manifest.update(protocol='cnn512_population_v1',
            population_comparison=f'All {server.num_clients} clients vs declared group {getattr(server.args, "target_group_ids", None)} evaluate different populations; group is a stage proxy, not the completed method')
    print('[CNN512Config] requested=', server.args.control_requested_config)
    print('[CNN512Config] effective=', server.args.control_effective_config)
    write_json(folder / 'comparison_manifest.json', server.control_manifest)
    torch.save(dict(model='common.models.CNN_512', kwargs=server.control_manifest['model_kwargs'],
                    state_dict=model.state_dict()), Path(server.save_folder_name) / 'initial_model.pt')
    print(f'[CNN512Control] initial_parameter_sha256={digest}; shared head; full parameters; no SVD')
    if population_control(server.args):
        print('[CNN512Population] scope='+primary_scope(server.args)+'; '+server.control_manifest['population_comparison'])


def scope_ids(args):
    scope = primary_scope(args)
    return (list(range(args.num_clients)) if scope == 'all' else list(args.target_group_ids)
            if scope == 'group' else [getattr(args, 'target_client_id', 0)])


def primary_point(args, loop, counts):
    """Pure reduction of THIS round's counts; never average per-client bests."""
    ids = scope_ids(args)
    actual_ids = [row['client_id'] for row in counts]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(ids):
        raise ValueError('Primary evaluation counts must cover the declared population exactly once')
    correct = sum(row['correct'] for row in counts)
    samples = sum(row['test_samples'] for row in counts)
    accuracy = sample_weighted_accuracy([r['correct'] for r in counts], [r['test_samples'] for r in counts],
                                        client_ids=actual_ids)
    return dict(loop_round=loop, completed_train_count=loop+1, round=loop+1,
        evaluation_object=primary_scope(args)+'_ordinary_post_local', eval_scope=primary_scope(args),
        member_ids=','.join(map(str,ids)), weighting='test_sample_count', accuracy_unit='percent',
        correct=correct, test_samples=samples, test_acc_percent=100.*accuracy)


def target_object(args):
    cid = getattr(args, 'target_client_id', 0)
    return 'C0_ordinary_post_local' if cid == 0 else f'Client_{cid}_ordinary_post_local'


def record_control_training(server, loop):
    if not getattr(server.args, 'cnn512_control', False) or loop in server._control_training_rounds:
        return
    for client in server.clients:
        stats = client.last_normal_train_stats
        models = server.K if server.args.algorithm == 'FedFewS' else 1
        server.control_training.append(dict(loop_round=loop, completed_train_count=loop+1,
            client_id=client.id, model_count=models, **stats,
            optimizer_steps=stats['train_batches'], data_epochs=stats['local_epochs']*models))
    server._control_training_rounds.add(loop)


def record_control(server, loop, correct=None, samples=None, client_counts=None):
    if not getattr(server.args, 'cnn512_control', False):
        return
    target = getattr(server.args, 'target_client_id', 0)
    counts = client_counts if client_counts is not None else [dict(client_id=target, correct=correct, test_samples=samples)]
    point = primary_point(server.args, loop, counts)
    server.primary_points.append(point)
    auxiliary = next((r for r in counts if r['client_id'] == target), None)
    if auxiliary is not None:
        correct, samples = auxiliary['correct'], auxiliary['test_samples']
        row = dict(loop_round=loop, completed_train_count=loop+1, round=loop+1,
                   evaluation_object=target_object(server.args), correct=correct, samples=samples,
                   test_acc_percent=100.*correct/samples)
        server.control_points.append(row)
        stem = 'c0' if target == 0 else f'target_{target}'
        write_json(Path(server.report_dir()) / (stem+'_post_local.json'), server.control_points)
    for item in counts:
        server.control_client_metrics.append(dict(loop_round=loop, completed_train_count=loop+1,
            **item, post_local_acc_percent=100.*item['correct']/item['test_samples'], accuracy_unit='percent'))
    record_control_training(server, loop)
    aux_text = 'not observed (outside primary population; no extra test)' if auxiliary is None else f'{100.*correct/samples:.6f}%'
    print(f'[CNN512Control] completed={loop+1}; scope={point["eval_scope"]}; members={point["member_ids"]}; '
          f'test-sample-weighted post-local={point["test_acc_percent"]:.6f}%; Client {target} auxiliary={aux_text}')


def save_control(server):
    if not getattr(server.args, 'cnn512_control', False):
        return
    folder = Path(server.report_dir())
    target = getattr(server.args, 'target_client_id', 0)
    stem = 'c0' if target == 0 else f'target_{target}'
    summary = (summarize_evaluations(server.control_points, value_key='test_acc_percent', unit='percent',
                                   evaluation_object=target_object(server.args)) if server.control_points else None)
    auxiliary_status = dict(target_client_id=target, observed=bool(server.control_points),
        reason='from existing primary counts' if summary is not None else 'target outside primary population; no extra evaluation')
    write_json(folder/'target_auxiliary.json', auxiliary_status)
    if summary is not None:
        write_json(folder/(stem+'_summary.json'), summary)
    primary = summarize_evaluations(server.primary_points, value_key='test_acc_percent', unit='percent',
                                    evaluation_object=primary_scope(server.args)+'_ordinary_post_local')
    primary.update(eval_scope=primary_scope(server.args), member_ids=scope_ids(server.args),
        weighting='test_sample_count', accuracy_unit='percent',
        evaluation_loop_rounds=[r['loop_round'] for r in server.primary_points],
        completed_train_counts=[r['completed_train_count'] for r in server.primary_points],
        last10_loop_rounds=[r['loop_round'] for r in server.primary_points[-10:]],
        last20_loop_rounds=[r['loop_round'] for r in server.primary_points[-20:]])
    completed = len(server._control_training_rounds)
    server.control_manifest.update(actual_train_count=completed, evaluation_count=len(server.primary_points),
        evaluation_loop_rounds=primary['evaluation_loop_rounds'], target_auxiliary=auxiliary_status,
        train_sample_visits=sum(r['processed_train_samples'] for r in server.control_training),
        optimizer_steps=sum(r['optimizer_steps'] for r in server.control_training))
    write_json(folder/'comparison_manifest.json', server.control_manifest)
    write_json(folder/'primary_summary.json', primary)
    write_json(folder/'primary_metrics.json', server.primary_points)
    write_json(folder/'client_metrics.json', server.control_client_metrics)
    write_json(folder/'local_training.json', server.control_training)
    outputs = [('local_training',server.control_training), ('primary_metrics',server.primary_points),
               ('client_metrics',server.control_client_metrics)]
    if summary is not None: outputs.append((stem+'_post_local',server.control_points))
    for name, rows in outputs:
        with (folder/f'{name}.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    path = server.args.save_file_paths[-1]
    clip = effective_grad_clip(server.args)
    with h5py.File(path,'a') as result:
        group = result.create_group('cnn512_control')
        active = server.primary_points if population_control(server.args) else server.control_points
        active_summary = primary if population_control(server.args) else summary
        group.attrs.update(model=server.control_manifest['model'], initial_parameter_sha256=server.control_manifest['initial_parameter_sha256'],
                           gradient_norm_cap=clip if clip is not None else 0., gradient_clipping_enabled=clip is not None,
                           test_shuffle=False, actual_train_count=completed, evaluation_count=len(server.primary_points), **active_summary)
        for name in active[0]:
            if not isinstance(active[0][name], str):
                group.create_dataset(name, data=[r[name] for r in active])
        population = result.create_group('primary_metrics')
        population.attrs.update(**primary)
        for name in server.primary_points[0]:
            if not isinstance(server.primary_points[0][name], str):
                population.create_dataset(name, data=[r[name] for r in server.primary_points])
        population.create_dataset('client_ids', data=scope_ids(server.args))
        by_key = {(r['loop_round'],r['client_id']):r for r in server.control_client_metrics}
        for name in ('correct','test_samples','post_local_acc_percent'):
            population.create_dataset('client_'+name, data=[[by_key[(r['loop_round'],cid)][name]
                for cid in scope_ids(server.args)] for r in server.primary_points])
        import json
        result.attrs.update(primary_accuracy_metric='primary_metrics/test_acc_percent',
            primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
            primary_weighting='test_sample_count', primary_accuracy_unit='percent',
            requested_config_json=json.dumps(server.args.control_requested_config),
            effective_config_json=json.dumps(server.args.control_effective_config))
        train = group.create_group('training')
        for name in server.control_training[0]:
            train.create_dataset(name, data=[r[name] for r in server.control_training])
    mirror_result(path, folder)
    final_document = dict(**server.control_manifest, target_summary=summary, primary_summary=primary,
        reload='common.models.create_full_model(args); load_state_dict; FedFewS_final.pt contains actual K servers/client models')
    if target == 0 and summary is not None: final_document['c0_summary'] = summary
    write_json(Path(server.final_model_dir())/'comparison_manifest.json', final_document)
    manifest_path = Path(server.final_model_dir())/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(primary_accuracy_metric='primary_metrics/test_acc_percent',
        primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
        primary_weighting='test_sample_count', primary_accuracy_unit='percent', primary_summary=primary,
        requested_config=server.args.control_requested_config, effective_config=server.args.control_effective_config,
        actual_train_count=completed, evaluation_count=len(server.primary_points), target_auxiliary=auxiliary_status)
    write_json(manifest_path, manifest)
    print('[CNN512Control] PRIMARY summary (percent; tail EVALUATION points):', primary)
