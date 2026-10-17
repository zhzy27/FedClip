"""Read-only comparison artifacts; no new loader traversal or random draws."""
import csv
import hashlib
from pathlib import Path
import sys

import h5py
import torch

from common.client_data import audit_image_clients
from common.experiment import tensor_digest, write_json, environment, code_version
from common.metrics import summarize_evaluations
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
        primary_metric='C0 ordinary post-local; FedFewS selected by training endpoint CE',
        data_epochs_per_client=(3 if server.args.algorithm == 'FedFewS' else 1)*5,
        actual_train_count=server.global_rounds+1,
        paths=dict(checkpoints=server.save_folder_name, logs=server.report_dir(), final_models=server.final_model_dir()))
    write_json(folder / 'comparison_manifest.json', server.control_manifest)
    torch.save(dict(model='common.models.CNN_512', kwargs=server.control_manifest['model_kwargs'],
                    state_dict=model.state_dict()), Path(server.save_folder_name) / 'initial_model.pt')
    print(f'[CNN512Control] initial_parameter_sha256={digest}; shared head; full parameters; no SVD')


def record_control(server, loop, correct, samples):
    if not getattr(server.args, 'cnn512_control', False):
        return
    if samples <= 0:
        raise ValueError('C0 control evaluation requires positive sample count')
    row = dict(loop_round=loop, completed_train_count=loop+1, round=loop+1,
               evaluation_object='C0_ordinary_post_local', correct=correct, samples=samples,
               test_acc_percent=100.*correct/samples)
    server.control_points.append(row)
    for client in server.clients:
        stats = client.last_normal_train_stats
        server.control_training.append(dict(loop_round=loop, completed_train_count=loop+1,
            client_id=client.id, model_count=server.K if server.args.algorithm=='FedFewS' else 1, **stats))
    write_json(Path(server.report_dir()) / 'c0_post_local.json', server.control_points)
    print(f'[CNN512Control] completed={loop+1}; C0 ordinary post-local={row["test_acc_percent"]:.6f}%')


def save_control(server):
    if not getattr(server.args, 'cnn512_control', False):
        return
    folder = Path(server.report_dir())
    summary = summarize_evaluations(server.control_points, value_key='test_acc_percent', unit='percent',
                                    evaluation_object='C0_ordinary_post_local')
    write_json(folder/'c0_summary.json', summary)
    write_json(folder/'local_training.json', server.control_training)
    for name, rows in [('c0_post_local',server.control_points), ('local_training',server.control_training)]:
        with (folder/f'{name}.csv').open('w',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    path = server.args.save_file_paths[-1]
    with h5py.File(path,'a') as result:
        group = result.create_group('cnn512_control')
        group.attrs.update(model='common.models.CNN_512', initial_parameter_sha256=server.control_manifest['initial_parameter_sha256'],
                           gradient_norm_cap=10., test_shuffle=False, **summary)
        for name in server.control_points[0]:
            if name != 'evaluation_object':
                group.create_dataset(name, data=[r[name] for r in server.control_points])
        train = group.create_group('training')
        for name in server.control_training[0]:
            train.create_dataset(name, data=[r[name] for r in server.control_training])
    mirror_result(path, folder)
    write_json(Path(server.final_model_dir())/'comparison_manifest.json', dict(**server.control_manifest, c0_summary=summary,
        reload='common.models.create_full_model(args); load_state_dict; FedFewS_final.pt contains K actual servers/client models'))
    print('[CNN512Control] PRIMARY C0 summary (percent; tail evaluation points):', summary)
