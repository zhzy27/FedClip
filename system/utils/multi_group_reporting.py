"""Collected-count metrics and consistent CSV/JSON/H5 for fixed multi-groups."""
import csv
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

from common.experiment import code_version, environment, jsonable, write_json
from common.metrics import sample_weighted_accuracy, summarize_evaluations
from utils.group_projection import accuracy_statistics
from utils.target_projection import EPS, _delta, _dot, _cosine
from utils.training_logs import mirror_result
from utils import multi_group_svd as svd


def _weighted(rows):
    return sample_weighted_accuracy([r['correct'] for r in rows], [r['test_samples'] for r in rows],
                                    client_ids=[r['client_id'] for r in rows])


def record_evaluation(server, before, after):
    loop, k, n = server.loop_round, server.K, server.num_clients
    def index(records):
        lookup = {(r['client_id'], r['model_id']): r for r in records}
        if len(records) != n*k or set(lookup) != {(i, j) for i in range(n) for j in range(k)}:
            raise ValueError('Require exactly N x K endpoint counts')
        for row in records:
            _weighted([row])  # Pure count validation, no model/data access.
        return lookup
    post, pre = index(after), index(before) if before is not None else None
    selected, selected_pre = [], []
    for cid in range(n):
        column = int(server.group_payload['client_to_model'][str(cid)])
        selected.append(post[cid, column])
        if pre is not None:
            selected_pre.append(pre[cid, column])
    point = dict(loop_round=loop, completed_train_count=loop+1,
        phase='grouped',
        evaluation_object='all_clients_fixed_membership_ordinary_post_local', unit='fraction',
        aggregation_mode=server.args.multi_group_mode, post_local_accuracy=_weighted(selected),
        pre_local_accuracy=_weighted(selected_pre) if pre else None,
        pre_observed=pre is not None, test_samples=sum(r['test_samples'] for r in selected),
        correct=sum(r['correct'] for r in selected), member_ids=list(range(n)))
    point['same_round_finetune_gain'] = point['post_local_accuracy']-point['pre_local_accuracy'] if pre else None
    server.primary_records.append(point)
    server.rs_test_acc.append(point['post_local_accuracy'])
    for stage, records in [('pre', before), ('post', after)]:
        if records is not None:
            server.endpoint_records.extend(dict(loop_round=loop, completed_train_count=loop+1, stage=stage,
                phase='grouped', membership_status='fixed',
                assigned_model_id=int(server.group_payload['client_to_model'][str(row['client_id'])]),
                selected_by_membership=int(int(server.group_payload['client_to_model'][str(row['client_id'])]) == row['model_id']),
                **row) for row in records)
    for column, ids in enumerate(server.groups):
        members = [dict(post[cid, column], post_local_acc=post[cid, column]['accuracy'],
                        post_local_correct=post[cid, column]['correct']) for cid in ids]
        stats = accuracy_statistics(members, 'post_local_acc')
        record = dict(loop_round=loop, completed_train_count=loop+1, model_id=column, members=ids,
            phase='grouped',
            post_local_accuracy=_weighted([post[cid, column] for cid in ids]), worst=stats['worst'],
            std=stats['std'], macro_accuracy=stats['macro'], test_samples=sum(post[cid, column]['test_samples'] for cid in ids),
            correct=sum(post[cid, column]['correct'] for cid in ids),
            pre_local_accuracy=_weighted([pre[cid, column] for cid in ids]) if pre else None)
        record['same_round_finetune_gain'] = record['post_local_accuracy']-record['pre_local_accuracy'] if pre else None
        server.group_records.append(record)
    if svd.enabled(server.args):
        for level, ratio in enumerate(server.args.multi_svd_effective_ratios):
            ids = [cid for cid, assigned in enumerate(server.args.multi_svd_effective_assignment) if assigned == level]
            if not ids:
                continue
            rows = [selected[cid] for cid in ids]
            pre_rows = [selected_pre[cid] for cid in ids] if pre else None
            server.capacity_records.append(dict(loop_round=loop, completed_train_count=loop+1,
                capacity_level=level, rank_rate=ratio, member_ids=ids, unit='fraction', weighted_by='test_samples',
                correct=sum(r['correct'] for r in rows), test_samples=sum(r['test_samples'] for r in rows),
                post_local_accuracy=_weighted(rows), pre_local_accuracy=_weighted(pre_rows) if pre else None,
                same_round_finetune_gain=_weighted(rows)-_weighted(pre_rows) if pre else None))
    print(f'Primary all {n} clients, fixed membership, test-sample weighted POST={100*point["post_local_accuracy"]:.6f}% '
          f'PRE={None if pre is None else 100*point["pre_local_accuracy"]}; gain={point["same_round_finetune_gain"]} (fraction)')


def record_warmup_evaluation(server, before, after):
    """No deployment rule exists yet: keep each column, never make a primary."""
    loop, k, n = server.loop_round, server.K, server.num_clients
    def index(records):
        lookup = {(r['client_id'], r['model_id']): r for r in records}
        if len(records) != n*k or set(lookup) != {(i,j) for i in range(n) for j in range(k)}:
            raise ValueError('Require N x K warmup endpoint counts')
        for row in records:
            _weighted([row])
        return lookup
    post, pre = index(after), index(before) if before is not None else None
    for stage, records in [('pre', before), ('post', after)]:
        if records is not None:
            server.endpoint_records.extend(dict(loop_round=loop, completed_train_count=loop+1, stage=stage,
                phase='warmup', assigned_model_id=None, selected_by_membership=None,
                membership_status='undetermined', **row) for row in records)
    for column in range(k):
        members = [post[cid,column] for cid in range(n)]
        record = dict(loop_round=loop, completed_train_count=loop+1, phase='warmup', model_id=column,
            evaluation_object=f'warmup_all_clients_model_{column}', unit='fraction', weighted_by='test_samples',
            post_local_accuracy=_weighted(members), correct=sum(r['correct'] for r in members),
            test_samples=sum(r['test_samples'] for r in members),
            pre_local_accuracy=_weighted([pre[cid,column] for cid in range(n)]) if pre else None)
        record['same_round_finetune_gain'] = record['post_local_accuracy']-record['pre_local_accuracy'] if pre else None
        server.warmup_records.append(record)
        print(f'Warmup model {column}, all {n} clients, sample-weighted post={100*record["post_local_accuracy"]:.6f}%; no assigned deployment/primary')


@torch.no_grad()
def model_geometry(server):
    for left in range(server.K):
        a = dict(server.global_model_set[left].named_parameters())
        for right in range(left+1, server.K):
            b = dict(server.global_model_set[right].named_parameters())
            difference = _delta(b, a)
            server.geometry_records.append(dict(loop_round=server.loop_round, completed_train_count=server.loop_round+1,
                left_model=left, right_model=right, parameter_distance=_dot(difference, difference)**.5,
                parameter_cosine=_cosine(_dot(a, b), _dot(a, a), _dot(b, b))))


def _csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(jsonable(v), ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v
                         for key, v in row.items()} for row in rows)


def primary_summary(server):
    result = summarize_evaluations(server.primary_records, value_key='post_local_accuracy', unit='fraction',
        evaluation_object='all_clients_fixed_membership_ordinary_post_local')
    result.update(member_ids=list(range(server.num_clients)), weighted_by='test_samples',
        evaluation_loop_rounds=[p['loop_round'] for p in server.primary_records],
        evaluation_completed_train_counts=[p['completed_train_count'] for p in server.primary_records],
        actual_train_count=server.loop_round+1, evaluation_count=len(server.primary_records),
        phase='grouped', warmup_train_count=min(server.loop_round+1, getattr(server.args,'multi_warmup_rounds',0)),
        grouped_train_count=max(0,server.loop_round+1-getattr(server.args,'multi_warmup_rounds',0)),
        last10_loop_rounds=[p['loop_round'] for p in server.primary_records[-10:]],
        last20_loop_rounds=[p['loop_round'] for p in server.primary_records[-20:]])
    return result


def write_multi_manifest(server, path=None):
    summary = primary_summary(server) if server.primary_records else None
    groups = server.group_payload
    from utils.multi_group_protocol import grouping_protocol
    protocol = grouping_protocol(server.args)
    values = dict(algorithm=server.algorithm, aggregation_mode=server.args.multi_group_mode,
        server_model_count=server.K, client_count=server.num_clients,
        aggregation=dict(kernel='utils.group_projection.aggregate_group_updates',
            effective_group_mass=server.args.multi_group_weight if server.args.multi_group_mode=='group_projection_softmax' else 1.,
            effective_outside_mass=1-server.args.multi_group_weight if server.args.multi_group_mode=='group_projection_softmax' else 0.,
            group_softmax_tau=server.args.group_softmax_tau, epsilon=EPS),
        **svd.model_spec(server.args, model=server.global_model_set[0]), model_family=server.args.model_family,
        parameter_schema=server.initial_schema, initial_parameter_sha256=server.initial_hash,
        requested=server.args.multi_requested_config, effective=vars(server.args),
        grouping=groups, group_source_sha256=getattr(server, 'group_source_sha256', None),
        grouping_protocol=protocol,
        clustering_execution='source_file_reused' if server.args.multi_group_file else 'computed_once_at_formation',
        effective_clustering=groups.get('algorithm') if groups else None,
        group_reuse_verification=getattr(server,'group_reuse_receipt',None),
        data_manifest=server.data_manifest, environment=environment(), code=code_version(Path(__file__).resolve().parents[2]),
        command=sys.argv, primary_summary=summary, checkpoint_dir=server.save_folder_name, metrics_dir=server.report_dir(),
        seed_protocol='public seed-forked initialization; Python/NumPy/Torch streams reset after client construction; local independent clustering Generator',
        final_model_dir=server.final_model_dir(), planned_train_count=server.global_rounds+1,
        actual_train_count=server.loop_round+1, evaluation_count=len(server.primary_records),
        actual_warmup_train_count=min(server.loop_round+1, protocol['warmup_rounds']),
        actual_grouped_train_count=max(0,server.loop_round+1-protocol['warmup_rounds']),
        warmup_evaluation_point_count=len(getattr(server, 'warmup_records', []))//server.K,
        grouped_evaluation_point_count=len(server.primary_records),
        budget=dict(models_trained_per_client=server.K, epochs_per_model=server.local_epochs,
            data_epochs_per_client_per_round=server.K*server.local_epochs,
            actual_sample_visits=sum(r['processed_train_samples'] for r in server.training_records),
            actual_optimizer_steps=sum(r['train_batches'] for r in server.training_records),
            training_seconds=sum(r['train_seconds'] for r in server.training_records)),
        loader=dict(train_batch=server.batch_size, test_batch=server.args.test_batch_size,
                    train_shuffle=True, test_shuffle=False, drop_last=False, workers=0),
        optimizer=dict(name='SGD', lr=server.learning_rate, momentum=0., weight_decay=0.,
                       gradient_norm_cap=server.args.local_grad_clip if server.args.local_grad_clip else None),
        buffers='Not aggregated or downloaded after initialization; local buffers preserved. Current CNN-512 has no buffers.',
        deployment='Client_i_model_g(i).pt is the final ordinary post-local primary model; Server_model_k.pt is AFTER aggregation.',
        loading="m=common.models.create_full_model(args); m.load_state_dict(torch.load(file,map_location='cpu')['state_dict']); m.eval()",
        resume_supported=False, pre_observation_enabled=not server.args.multi_no_pre_eval)
    if svd.enabled(server.args):
        values.update(svd_capacity_protocol=svd.capacity_protocol(server.args),
            svd_initialization=svd.initialization_protocol(server.args),
            objective='CE_plus_original_Frobenius' if server.args.is_regular else 'CE_only',
            is_regular=server.args.is_regular, regular_lamda=server.args.regular_lamda,
            client_capacity_metadata={str(c.id): c.capacity_metadata for c in server.clients},
            upload_storage='recovered detached CPU named_parameter tensors; not prediction checkpoints',
            prediction_storage='Client_i_model_k.pt: ordinary post-local low-rank state_dict + actual rank metadata',
            loading="m,payload=utils.multi_group_svd.load_checkpoint(path,device); m.eval()",
            buffers='client-local, retained across SVD downloads; no buffers aggregated',
            svd_diagnostics_enabled=bool(server.args.multi_svd_diagnostics))
        write_json(Path(server.report_dir())/'capacity_manifest.json', dict(protocol=svd.capacity_protocol(server.args),
            initialization=svd.initialization_protocol(server.args),
            clients=values['client_capacity_metadata']))
    from utils.multi_group_kd_options import requested,protocol as kd_protocol
    if requested(server.args):
        rows=server.kd_training_records
        values['online_kd']=dict(protocol=kd_protocol(server.args),
            durable_diagnostics=str(Path(server.report_dir())/'online_kd.jsonl'),
            effective_kd_training_tasks=sum(r['kd_enabled'] for r in rows),
            cache_total_seconds=sum(r['kd_cache_seconds'] for r in rows),
            teacher_forward_seconds=sum(r['teacher_forward_seconds'] for r in rows),
            teacher_sample_forwards=sum(r['teacher_sample_forwards'] for r in rows),
            probe_total_seconds=sum(r['observation_seconds'] for r in server.kd_probe_records),
            probe_sample_forwards=sum(r['probe_sample_forwards'] for r in server.kd_probe_records),
            probe_records=len(server.kd_probe_records),cache_persistence='one_client_one_round_memory_only_no_history')
    from utils.multi_group_proto_options import requested as proto_requested
    if proto_requested(server.args):
        from utils.multi_group_proto import manifest_value
        values['class_prototypes']=manifest_value(server)
    write_json(path or Path(server.report_dir())/'manifest.json', values)
    return values


def save_multi_reports(server):
    root = Path(server.report_dir())
    summary = primary_summary(server)
    tables = dict(primary_metrics=server.primary_records, client_metrics=server.endpoint_records,
        group_metrics=server.group_records, local_training=server.training_records, round_budget=server.round_records,
        aggregation_metrics=server.aggregation_records, aggregation_weights=server.weight_records,
        server_geometry=server.geometry_records, warmup_metrics=getattr(server,'warmup_records',[]))
    if svd.enabled(server.args):
        tables['capacity_metrics'] = server.capacity_records
        tables['svd_diagnostics'] = [{k: v for k, v in row.items() if k in ('loop_round', 'completed_train_count',
            'client_id', 'model_id', 'capacity_level', 'rank_rate') or k.startswith(('svd_truncation_',
            'local_training_recovered_', 'global_delta_'))} for row in server.training_records]
    from utils.multi_group_kd_options import requested,protocol as kd_protocol
    if requested(server.args):
        tables['online_kd_training']=server.kd_training_records
        tables['online_kd_probes']=server.kd_probe_records
    from utils.multi_group_proto_options import requested as proto_requested,protocol as proto_protocol
    if proto_requested(server.args):
        tables.update(prototype_training=server.proto_training_records,prototype_sources=server.proto_source_records,
            prototype_client_directions=server.proto_direction_records,prototype_capacity_directions=server.proto_capacity_records,
            prototype_collection=server.proto_collection_records,prototype_uploads=server.proto_upload_records,
            prototype_rounds=server.proto_round_records)
    for name, rows in tables.items():
        _csv(root/f'{name}.csv', rows)
        write_json(root/f'{name}.json', rows)
    write_json(root/'primary_summary.json', summary)
    warmup_summary = dict(phase='warmup', primary_metric=False, deployment='undetermined', models={})
    for column in range(server.K):
        rows = [p for p in getattr(server,'warmup_records',[]) if p['model_id']==column]
        warmup_summary['models'][str(column)] = (summarize_evaluations(rows, value_key='post_local_accuracy', unit='fraction',
            evaluation_object=f'warmup_all_clients_model_{column}') if rows else None)
    write_json(root/'warmup_summary.json', warmup_summary)
    # Same summary logic for each fixed group and each client's assigned endpoint.
    extra = dict(groups={}, clients={})
    for column in range(server.K):
        rows = [p for p in server.group_records if p['model_id'] == column]
        extra['groups'][str(column)] = summarize_evaluations(rows, value_key='post_local_accuracy', unit='fraction', evaluation_object=f'group_{column}')
    for cid in range(server.num_clients):
        rows = [p for p in server.endpoint_records if p['client_id']==cid and p['stage']=='post' and p['selected_by_membership']]
        extra['clients'][str(cid)] = summarize_evaluations(rows, value_key='accuracy', unit='fraction', evaluation_object=f'client_{cid}_assigned_post_local')
    if svd.enabled(server.args):
        extra['capacities'] = {}
        for level in range(len(server.args.multi_svd_effective_ratios)):
            rows = [p for p in server.capacity_records if p['capacity_level'] == level]
            if rows:
                extra['capacities'][str(level)] = summarize_evaluations(rows, value_key='post_local_accuracy',
                    unit='fraction', evaluation_object=f'capacity_level_{level}_fixed_membership_post_local')
    write_json(root/'member_summaries.json', extra)
    directory = Path(server.h5_result_dir())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'FedMultiGroupProj_{server.args.multi_group_mode}_{server.run_id}.h5'
    from utils.multi_group_protocol import grouping_protocol
    with h5py.File(path, 'w') as hf:
        hf.attrs.update(algorithm=server.algorithm, mode=server.args.multi_group_mode,
            model_class=svd.model_spec(server.args)['model_class'], primary_metric='all_clients_fixed_membership_ordinary_post_local',
            accuracy_unit='fraction', tail_unit='evaluation points', summary_json=json.dumps(summary),
            groups_json=json.dumps(server.groups), group_fingerprint=server.group_payload['group_fingerprint'],
            initial_parameter_sha256=server.initial_hash, data_fingerprint=server.context['data_fingerprint'],
            actual_train_count=server.loop_round+1, evaluation_count=len(server.primary_records),
            actual_warmup_train_count=summary['warmup_train_count'], actual_grouped_train_count=summary['grouped_train_count'],
            grouping_protocol_json=json.dumps(grouping_protocol(server.args)),
            source_evidence_sha256=server.group_payload.get('evidence',{}).get('evidence_sha256',''),
            group_reuse_verification_json=json.dumps(getattr(server,'group_reuse_receipt',None)),
            warmup_summary_json=json.dumps(warmup_summary),
            effective_json=json.dumps(jsonable(vars(server.args)), allow_nan=False))
        if svd.enabled(server.args):
            hf.attrs['svd_capacity_protocol_json'] = json.dumps(svd.capacity_protocol(server.args))
            hf.attrs['svd_initialization_json'] = json.dumps(svd.initialization_protocol(server.args))
        if requested(server.args):hf.attrs['online_kd_protocol_json']=json.dumps(kd_protocol(server.args))
        if proto_requested(server.args):hf.attrs['class_prototype_protocol_json']=json.dumps(proto_protocol(server.args))
        hf.create_dataset('rs_test_acc', data=[r['post_local_accuracy'] for r in server.primary_records])
        for name in ('loop_round', 'completed_train_count', 'correct', 'test_samples', 'post_local_accuracy'):
            hf.create_dataset('primary/'+name, data=[p[name] for p in server.primary_records])
        hf.create_dataset('client_to_model', data=[int(server.group_payload['client_to_model'][str(cid)]) for cid in range(server.num_clients)])
        for name, rows in tables.items():
            hf.create_dataset(name+'_json', data=[json.dumps(jsonable(r), allow_nan=False) for r in rows], dtype=h5py.string_dtype('utf-8'))
    server.args.save_file_paths.append(str(path))
    mirror_result(str(path), server.report_dir())
    write_multi_manifest(server)
    print(f'Primary final={100*summary["final"]:.6f}%, best={100*summary["best"]:.6f}% '
          f'at completed={summary["best_completed_train_count"]}; last10={100*summary["last10"]:.6f}%, '
          f'last20={100*summary["last20"]:.6f}% over evaluation points; metrics={root}')
    print(f'Actual completed training: total={summary["actual_train_count"]}, warmup={summary["warmup_train_count"]}, '
          f'grouped={summary["grouped_train_count"]}; grouped evaluations={summary["evaluation_count"]}')
