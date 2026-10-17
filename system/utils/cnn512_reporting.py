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
from utils.cnn512_control import primary_scope, population_control
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
    write_json(folder / 'data_manifest.json', audit)
    digest = model_digest(model)
    client_hashes = [model_digest(m) for m in clients]
    server_hashes = [model_digest(m) for m in getattr(server, 'global_model_set', [model])]
    if any(h != digest for h in client_hashes):
        raise AssertionError('Common initial parameters differ')
    if any(h != digest for h in server_hashes):
        raise AssertionError('Common initial K-server parameters differ')
    server.control_points, server.control_training = [], []
    server.primary_points, server.control_client_metrics = [], []
    server.control_manifest = dict(protocol='cnn512_control', args=vars(server.args),
        command=[sys.executable] + sys.argv, environment=environment(),
        code=code_version(Path(__file__).resolve().parents[2]), model='common.models.CNN_512',
        model_kwargs=dict(in_channels=3, n_kernels=16, out_dim=100, input_size=32),
        initial_parameter_sha256=digest, initial_client_model_hashes=client_hashes,
        initial_server_model_hashes=server_hashes,
        parameter_shapes={n: list(p.shape) for n,p in model.named_parameters()},
        data_manifest='data_manifest.json', objective='CE only', gradient_norm_cap=10,
        optimizer=dict(name='SGD', lr=.005, momentum=0, weight_decay=0),
        loaders=dict(train=dict(batch_size=16, shuffle=True, drop_last=False, num_workers=0),
                     test=dict(batch_size=server.args.test_batch_size, shuffle=False, drop_last=False, num_workers=0)),
        buffers='Not aggregated; local buffers retained on download (CNN-512 has no buffers)',
        primary_metric=f'{primary_scope(server.args)} ordinary post-local; FedFewS selected by training endpoint CE',
        primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
        primary_weighting='test_sample_count', primary_accuracy_unit='percent',
        data_epochs_per_client=(3 if server.args.algorithm == 'FedFewS' else 1)*5,
        actual_train_count=server.global_rounds+1,
        paths=dict(checkpoints=server.save_folder_name, logs=server.report_dir(), final_models=server.final_model_dir()))
    if population_control(server.args):
        server.control_manifest.update(protocol='cnn512_population_v1',
            population_comparison='All20 Avg/FedFewS and fixed-group7 Projection evaluate different populations; group7 is a stage proxy for the incomplete full method')
    write_json(folder / 'comparison_manifest.json', server.control_manifest)
    torch.save(dict(model='common.models.CNN_512', kwargs=server.control_manifest['model_kwargs'],
                    state_dict=model.state_dict()), Path(server.save_folder_name) / 'initial_model.pt')
    print(f'[CNN512Control] initial_parameter_sha256={digest}; shared head; full parameters; no SVD')
    if population_control(server.args):
        print('[CNN512Population] scope='+primary_scope(server.args)+'; '+server.control_manifest['population_comparison'])


def scope_ids(args):
    scope = primary_scope(args)
    return (list(range(args.num_clients)) if scope == 'all' else list(args.target_group_ids)
            if scope == 'group' else [0])


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


def record_control(server, loop, correct, samples, client_counts=None):
    if not getattr(server.args, 'cnn512_control', False):
        return
    if samples <= 0:
        raise ValueError('C0 control evaluation requires positive sample count')
    row = dict(loop_round=loop, completed_train_count=loop+1, round=loop+1,
               evaluation_object='C0_ordinary_post_local', correct=correct, samples=samples,
               test_acc_percent=100.*correct/samples)
    server.control_points.append(row)
    counts = client_counts if client_counts is not None else [dict(client_id=0, correct=correct, test_samples=samples)]
    point = primary_point(server.args, loop, counts)
    server.primary_points.append(point)
    for item in counts:
        server.control_client_metrics.append(dict(loop_round=loop, completed_train_count=loop+1,
            **item, post_local_acc_percent=100.*item['correct']/item['test_samples'], accuracy_unit='percent'))
    for client in server.clients:
        stats = client.last_normal_train_stats
        training = dict(loop_round=loop, completed_train_count=loop+1,
            client_id=client.id, model_count=server.K if server.args.algorithm=='FedFewS' else 1, **stats)
        if population_control(server.args):
            training.update(optimizer_steps=stats['train_batches'],
                            data_epochs=stats['local_epochs']*training['model_count'])
        server.control_training.append(training)
    write_json(Path(server.report_dir()) / 'c0_post_local.json', server.control_points)
    print(f'[CNN512Control] completed={loop+1}; scope={point["eval_scope"]}; members={point["member_ids"]}; '
          f'test-sample-weighted post-local={point["test_acc_percent"]:.6f}%; C0 auxiliary={row["test_acc_percent"]:.6f}%')


def save_control(server):
    if not getattr(server.args, 'cnn512_control', False):
        return
    folder = Path(server.report_dir())
    summary = summarize_evaluations(server.control_points, value_key='test_acc_percent', unit='percent',
                                    evaluation_object='C0_ordinary_post_local')
    primary = summarize_evaluations(server.primary_points, value_key='test_acc_percent', unit='percent',
                                    evaluation_object=primary_scope(server.args)+'_ordinary_post_local')
    primary.update(eval_scope=primary_scope(server.args), member_ids=scope_ids(server.args),
                   weighting='test_sample_count', accuracy_unit='percent')
    write_json(folder/'c0_summary.json', summary)
    write_json(folder/'primary_summary.json', primary)
    write_json(folder/'primary_metrics.json', server.primary_points)
    write_json(folder/'client_metrics.json', server.control_client_metrics)
    write_json(folder/'local_training.json', server.control_training)
    for name, rows in [('c0_post_local',server.control_points), ('local_training',server.control_training),
                       ('primary_metrics',server.primary_points), ('client_metrics',server.control_client_metrics)]:
        with (folder/f'{name}.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    path = server.args.save_file_paths[-1]
    with h5py.File(path,'a') as result:
        group = result.create_group('cnn512_control')
        active = server.primary_points if population_control(server.args) else server.control_points
        active_summary = primary if population_control(server.args) else summary
        group.attrs.update(model='common.models.CNN_512', initial_parameter_sha256=server.control_manifest['initial_parameter_sha256'],
                           gradient_norm_cap=10., test_shuffle=False, **active_summary)
        for name in active[0]:
            if isinstance(active[0][name], str):
                continue
            if name != 'evaluation_object':
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
        result.attrs.update(primary_accuracy_metric='primary_metrics/test_acc_percent',
            primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
            primary_weighting='test_sample_count', primary_accuracy_unit='percent')
        train = group.create_group('training')
        for name in server.control_training[0]:
            train.create_dataset(name, data=[r[name] for r in server.control_training])
    mirror_result(path, folder)
    write_json(Path(server.final_model_dir())/'comparison_manifest.json', dict(**server.control_manifest, c0_summary=summary,
        primary_summary=primary,
        reload='common.models.create_full_model(args); load_state_dict; FedFewS_final.pt contains K actual servers/client models'))
    # Canonical primary metadata also goes into the ordinary final manifest.
    import json
    manifest_path = Path(server.final_model_dir())/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(primary_accuracy_metric='primary_metrics/test_acc_percent',
        primary_eval_scope=primary_scope(server.args), primary_member_ids=scope_ids(server.args),
        primary_weighting='test_sample_count', primary_accuracy_unit='percent', primary_summary=primary)
    write_json(manifest_path, manifest)
    print('[CNN512Control] PRIMARY summary (percent; tail evaluation points):', primary)
