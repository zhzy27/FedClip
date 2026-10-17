"""Export collected FedFewS counts; never evaluate a model or iterate a loader."""
import csv
import json
import math
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from common.client_data import data_directory, image_policy
from common.data_loading import loader_description
from common.experiment import code_version, environment, jsonable, write_json
from common.metrics import sample_weighted_accuracy, summarize_evaluations
from utils.training_logs import mirror_result


def write_manifest(server):
    args = server.args
    formats, mode = image_policy(args)
    write_json(Path(server.report_dir()) / 'manifest.json', dict(
        args=vars(args), command=[sys.executable] + sys.argv, code=code_version(Path(__file__).resolve().parents[2]),
        environment=environment(), model=f'{type(server.global_model).__module__}.{type(server.global_model).__name__}', K=server.K,
        data_directory=str(data_directory(args)), format_policy=formats, image_list_mode=mode,
        rng_protocol=args.fedfews_rng, control_seed=args.fedfews_control_seed,
        python_numpy_seed=args.fedfews_control_seed if args.fedfews_rng == 'controlled' else None,
        initialization='One CNN; K deep copies; client torch.manual_seed(0) retained',
        optimizer=dict(name='SGD', lr=args.local_learning_rate, momentum=0, weight_decay=0),
        objective='CE only; no F/Prox/Projection/private head/low-rank',
        gradient_norm_cap=getattr(args, 'local_grad_clip', None),
        budget=dict(epochs_per_model=args.local_epochs, data_epochs_per_client=server.K * args.local_epochs,
                    actual_train_count=args.global_rounds + 1),
        loaders={stage: loader_description(batch, shuffle=stage != 'test' or not getattr(args, 'cnn512_control', False)) for stage, batch in
                 [('train', args.batch_size), ('deployment_selection', args.batch_size),
                  ('test', args.test_batch_size), ('train_metrics', args.test_batch_size)]},
        paths=dict(checkpoints=server.save_folder_name, logs=server.report_dir(), final_models=server.final_model_dir()),
        source=dict(repository='https://github.com/pgg3/FedFew', commit='1b29dec70b66d89b956d90e0da024897f698897a',
                    license='common/LICENSE.fedfew', accepted_reference='FedClip c73ec5b compatibility entry')))


def round_record(server, evaluated):
    s = server
    ids = [c.id for c in s.clients]
    evaluation = None
    if evaluated:
        correct = [c.last_test[0] for c in s.clients]
        samples = [c.last_test[1] for c in s.clients]
        accuracy = sample_weighted_accuracy(correct, samples, client_ids=ids)
        if accuracy != s.rs_test_acc[-1]:
            raise AssertionError('Exported weighted count accuracy differs from algorithm evaluation')
        evaluation = dict(loop_round=s.loop_round, completed_train_count=s.loop_round + 1,
                          evaluation_object='selected_post_local_clients', test_acc_percent=100 * accuracy,
                          train_loss=s.rs_train_loss[-1], total_test_correct=sum(correct), total_test_samples=sum(samples),
                          clients=[dict(client_id=c.id, selected_model=c.best_model_idx, deployment_ce=c.deployment_ce,
                                        correct=c.last_test[0], test_samples=c.last_test[1],
                                        test_acc_percent=100 * c.last_test[0] / c.last_test[1],
                                        endpoint_train_loss_sum=c.last_train_metrics[0], endpoint_train_samples=c.last_train_metrics[1])
                                   for c in s.clients],
                          selected_model_counts=np.bincount([c.best_model_idx for c in s.clients], minlength=s.K))
        s.evaluations.append(jsonable(evaluation))
    distances = {}
    with torch.no_grad():
        for k in range(s.K):
            for j in range(k):
                squared = sum((p.double() - q.double()).square().sum().item()
                              for p, q in zip(s.global_model_set[k].parameters(), s.global_model_set[j].parameters()))
                distances[f'{j}-{k}'] = squared ** .5
    clients = [dict(client_id=cid, train_samples=s.clients[cid].train_samples,
                    sample_weight=s.uploaded_weights[row], train_model_losses=s.clients[cid].train_model_losses,
                    epochs_per_model=s.local_epochs, model_count=s.K, data_epochs=s.K*s.local_epochs,
                    train_sample_visits=s.clients[cid].train_samples*s.K*s.local_epochs,
                    train_batch_count=math.ceil(s.clients[cid].train_samples/s.batch_size)*s.K*s.local_epochs,
                    train_seconds=getattr(s.clients[cid], 'last_normal_train_stats', {}).get('train_seconds'),
                    alpha=float(s.last_alpha[row, 0]), w=s.last_w[row], actual_weights=s.last_alpha[row, 0] * s.last_w[row])
               for row, cid in enumerate(s.uploaded_ids)]
    row = jsonable(dict(loop_round=s.loop_round, completed_train_count=s.loop_round+1,
                       train_order=[c.id for c in s.selected_clients], uploaded_ids=s.uploaded_ids, clients=clients,
                       server_received_mass=(s.last_alpha * s.last_w).sum(axis=0), server_pairwise_l2=distances, evaluation=evaluation))
    s.rows.append(row)
    with (Path(s.report_dir()) / 'rounds.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def collected_accuracy(evaluation, client_ids):
    """C0/group extraction from existing counts, with no new test pass. FRACTION."""
    clients = evaluation['clients']
    return sample_weighted_accuracy([c['correct'] for c in clients], [c['test_samples'] for c in clients],
                                    client_ids=[c['client_id'] for c in clients], subset_ids=client_ids)


def save_fedfews_results(server):
    s = server
    summary = summarize_evaluations(s.evaluations, value_key='test_acc_percent', unit='percent',
                                    evaluation_object='selected_post_local_clients')
    folder = Path(s.report_dir())
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'rounds.json', s.rows)  # Included by the common log-only ZIP facility.
    write_json(folder / 'evaluations.json', s.evaluations)
    with (folder / 'evaluations.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['loop_round', 'completed_train_count', 'test_acc_percent',
                              'total_test_correct', 'total_test_samples', 'train_loss', 'evaluation_object'], extrasaction='ignore')
        writer.writeheader()
        writer.writerows(s.evaluations)
    with (folder / 'clients.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['loop_round', 'completed_train_count', 'client_id', 'train_samples',
                  'sample_weight', 'epochs_per_model', 'model_count', 'data_epochs', 'train_sample_visits', 'train_batch_count', 'train_seconds',
                  'train_model_losses', 'alpha', 'w', 'actual_weights', 'deployment_ce', 'selected_model',
                  'correct', 'test_samples', 'test_acc_percent'], extrasaction='ignore')
        writer.writeheader()
        for row in s.rows:
            ev = {c['client_id']: c for c in row['evaluation']['clients']} if row['evaluation'] else {}
            for c in row['clients']:
                writer.writerow(dict(loop_round=row['loop_round'], completed_train_count=row['completed_train_count'], **c,
                                     **{k:v for k,v in ev.get(c['client_id'], {}).items() if k != 'client_id'}))
    path = s.args.save_file_paths[-1]
    with h5py.File(path, 'a') as hf:
        fields = dict(rs_model_losses=s.rs_model_losses, rs_alpha_weights=s.rs_alpha_weights, rs_soft_weights=s.rs_soft_weights,
                      rs_client_test_acc=s.rs_client_test_acc, rs_client_train_loss=s.rs_client_train_loss,
                      rs_client_test_auc=s.rs_client_test_auc, rs_client_ids=s.rs_client_ids,
                      evaluation_loop_round=[e['loop_round'] for e in s.evaluations],
                      completed_train_count=[e['completed_train_count'] for e in s.evaluations],
                      server_received_mass=[r['server_received_mass'] for r in s.rows], uploaded_ids=[r['uploaded_ids'] for r in s.rows],
                      actual_update_weights=[[[c['alpha'] * wk for wk in c['w']] for c in r['clients']] for r in s.rows],
                      deployment_ce=[[c['deployment_ce'] for c in e['clients']] for e in s.evaluations],
                      deployed_model_idx=[[c['selected_model'] for c in e['clients']] for e in s.evaluations],
                      test_correct=[[c['correct'] for c in e['clients']] for e in s.evaluations],
                      test_samples=[[c['test_samples'] for c in e['clients']] for e in s.evaluations],
                      selected_model_counts=[e['selected_model_counts'] for e in s.evaluations])
        for key, value in fields.items():
            hf.create_dataset(key, data=value)
        hf.attrs.update(K=s.K, mu=s.mu, smooth_mu=s.mu, accuracy_unit='fraction; CSV/JSON percent',
                        tail_average_unit='evaluation points', evaluation_object='selected_post_local_clients',
                        data_directory=str(data_directory(s.args)), actual_alpha=json.dumps(s.data_manifest['actual_alpha']),
                        model_class=f'{type(s.global_model).__module__}.{type(s.global_model).__name__}', rng_protocol=s.args.fedfews_rng,
                        data_config_sha256=s.data_manifest['config_sha256'],
                        shard_format_policy=json.dumps(s.data_manifest['format_policy']))
    mirror_result(path, s.report_dir())
    # Base export already copied the actual K-model bundle. Add explicit semantics.
    write_json(Path(s.final_model_dir()) / 'fedfews_manifest.json', dict(
        model_bundle='FedFewS_final.pt', K=s.K, clients=s.num_clients, summary=summary,
        server_models='K aggregated servers', client_models='K ordinary post-local models/client with optimizers and selected index',
        unused_template_exported=False, args=vars(s.args)))
    print('FedFewS PRIMARY summary (percent; tail windows count EVALUATIONS):', summary)
