"""Opt-in K-column last-epoch protocol, sharing K1 math and actual SGD code."""
from pathlib import Path
import time
import math

from utils import single_avg_proto as shared


def enabled(args):return bool(getattr(args,'multi_last_epoch_protocol',False))


def uses_bank(args):return enabled(args) and args.multi_proto_mode=='last_epoch' and args.multi_proto_lambda>0


def bank_mode(args):return getattr(args,'multi_proto_bank_mode','per_model')


def shared_bank(args):return bank_mode(args)=='shared'


def validate(args):
    for key,default in dict(multi_proto_scope='group_priority',multi_group_reuse_policy='strict',multi_proto_bank_mode='per_model').items():
        if not hasattr(args,key):setattr(args,key,default)
    if args.algorithm!='FedMultiGroupProj' or not getattr(args,'multi_svd',False) or getattr(args,'multi_avg_proto_ablation',False):
        raise ValueError('multi_last_epoch_protocol requires FedMultiGroupProj --multi_svd, separate from K1 ablation')
    if args.multi_proto_mode not in ('none','last_epoch') or args.multi_proto_loss!='mse' or args.multi_proto_feature!='base_output':
        raise ValueError('New last-epoch protocol supports none/last_epoch, raw base_output MSE only')
    if getattr(args,'multi_proto_scope','group_priority') not in ('group_priority','global'):
        raise ValueError('Invalid last-epoch contributor scope')
    if bank_mode(args) not in ('per_model','shared'):raise ValueError('Invalid prototype bank mode')
    if shared_bank(args) and uses_bank(args) and args.multi_proto_scope!='global':
        raise ValueError('Enabled shared prototype bank requires multi_proto_scope=global')
    if getattr(args,'multi_group_reuse_policy','strict') not in ('strict','members_only'):raise ValueError('Invalid group reuse policy')
    if not args.multi_group_file or getattr(args,'multi_group_publish','') or getattr(args,'multi_warmup_rounds',0) or getattr(args,'multi_group_wait_seconds',0.):
        raise ValueError('Last-epoch protocol requires an existing fixed group file, no publication/warmup/wait/reclustering')
    if (getattr(args,'multi_kd_mode','none')!='none' or args.target_head_mode!='shared' or args.target_local_schedule!='joint'
            or getattr(args,'target_prox_mu',0.) or getattr(args,'target_head_l2_lambda',0.) or getattr(args,'target_z2_lambda',0.)
            or getattr(args,'target_c0_replay_mode','none')!='none'):
        raise ValueError('Last-epoch protocol requires shared joint training without KD/Prox/head L2/Z2/replay')
    if not math.isfinite(args.multi_proto_lambda) or args.multi_proto_lambda<0:raise ValueError('Prototype coefficient must be finite and nonnegative')
    if args.multi_proto_mode=='none' and args.multi_proto_lambda:raise ValueError('Nonzero coefficient requires last_epoch mode')
    if args.multi_proto_mode=='last_epoch' and args.multi_proto_target_type!='raw_mean':raise ValueError('Last-epoch targets require raw_mean')
    if args.multi_proto_target_norm!=1. or args.multi_proto_batch_size!=16 or args.multi_proto_eps!=1e-12:
        raise ValueError('Last-epoch raw means do not use target norm, scan batch size or normalization epsilon; leave defaults')
    if type(args.multi_proto_start_round) is not int or args.multi_proto_start_round<2:raise ValueError('First round has no prototype MSE')
    if uses_bank(args) and args.multi_proto_start_round>args.global_rounds+1:raise ValueError('No eligible prototype round in training budget')
    if args.multi_proto_diagnostics not in (0,1):raise ValueError('Invalid diagnostics flag')


def protocol(args):
    values=shared.protocol(args)
    values.update(name='K_column_last_epoch_class_means_v1',prototype_enabled=uses_bank(args),
        effective_coefficient=args.multi_proto_lambda if uses_bank(args) else 0.,contributor_scope=args.multi_proto_scope,
        collection='each column last local epoch existing TRAIN forward; detach float64 sum/int64 count',
        aggregation='equal client class means, selected by contributor_scope independently per column',
        normalization='none; no fixed norm or EMA',group_reuse_policy=args.multi_group_reuse_policy,
        sample_order='column 0 exactly K1 loader_seed; other columns SHA256(seed,loop,client,column), independent stable generators',
        loss_mean_unit='training objectives sample-visit weighted; batch MSE reduction is entire batch x feature dimension with missing-class zero error',
        interpretation='fixed K/groups with original parameter kernels plus optional next-round class-mean MSE')
    values.update(bank_mode=bank_mode(args),K=args.num_server_models,
        actual_prototype_bank_count=(1 if shared_bank(args) else args.num_server_models) if uses_bank(args) else 0)
    if shared_bank(args):
        values.update(aggregation='equal mean over model-column class means WITHIN each client, THEN equal mean over owning clients',
            prototype_representation='client_then_column_equal_raw_class_mean_shared_across_all_models',
            feature_coordinate_assumption='direct arithmetic cross-model coordinate sharing is the experimental variable',
            communication='N x K local sum/count packets uploaded; one encoded shared bank per client downloaded, not K times')
    return values


def initialize(server):
    from utils.multi_group_reuse import load_members
    load_members(server)
    server.post_proto_packet=None;server.post_proto_classes=[];server.post_proto_rounds=[]


def prepare_round(server):
    if shared_bank(server.args):
        from utils.multi_shared_proto import prepare_round as prepare_shared
        return prepare_shared(server)
    from utils.multi_group_proto import decode_packet,packet_sha
    tick=time.perf_counter();packet=server.post_proto_packet;server.previous_post_proto_packet=packet
    if packet is not None and not uses_bank(server.args):raise RuntimeError('Disabled last-epoch protocol has a library')
    values=meta=None
    if packet is not None:
        values,meta=decode_packet(packet,'last_epoch_columns_library')
        model=server.global_model_set[0]
        if (meta['source_completed_train_count']!=server.loop_round or meta['K']!=server.K or
                meta['feature_position']!=server.args.multi_proto_feature or meta['contributor_scope']!=server.args.multi_proto_scope or
                meta['group_fingerprint']!=server.group_payload['group_fingerprint'] or
                values['prototypes'].shape!=(server.K,server.num_classes,model.fc3.in_features) or
                values['valid'].shape!=(server.K,server.num_classes)):
            raise ValueError('Last-epoch library round/column/shape/feature/scope/group mismatch')
    for c in server.clients:
        c.post_proto_banks=None;c.post_proto_receipt=None
        c.post_proto_assignment=int(server.group_payload['client_to_model'][str(c.id)])
        c.post_proto_group_source_sha256=server.group_source_sha256
        c.post_proto_group_fingerprint=server.group_payload['group_fingerprint']
        if packet is not None:
            c.post_proto_banks=[dict(prototypes=values['prototypes'][k].detach().clone(),valid=values['valid'][k].detach().clone(),model_id=k) for k in range(server.K)]
            c.post_proto_receipt=dict(meta,sha256=packet_sha(packet))
    server.post_proto_distribution_seconds=time.perf_counter()-tick


def contributors(packets,group,scope,classes):
    if scope not in ('group_priority','global'):raise ValueError('Invalid contributor scope')
    selected=[];sources=[];members=set(group)
    for label in range(classes):
        owners=sorted(p['client_id'] for p in packets if int(p['counts'][label])>0)
        inside=[i for i in owners if i in members]
        chosen=owners if scope=='global' else inside or [i for i in owners if i not in members]
        selected.append(chosen)
        sources.append('invalid' if not chosen else 'global' if scope=='global' else 'group' if inside else 'outside')
    return selected,sources


def finish_round(server):
    if shared_bank(server.args):
        from utils.multi_shared_proto import finish_round as finish_shared
        return finish_shared(server)
    if not uses_bank(server.args):return
    import torch
    from utils.multi_group_proto import decode_packet,encode_packet,packet_sha
    tick=time.perf_counter();source=server.loop_round+1;uploads=0;columns=[[] for _ in range(server.K)]
    for c in server.clients:
        if c.upload_round!=server.loop_round or len(c.post_proto_local_packets)!=server.K:raise RuntimeError('Incomplete N x K prototype uploads')
        for column,packet in enumerate(c.post_proto_local_packets):
            tensors,meta=decode_packet(packet,'last_epoch_column_local')
            if (meta.get('model_id')!=column or meta['client_id']!=c.id or meta['source_completed_train_count']!=source
                    or meta['feature_position']!=server.args.multi_proto_feature):raise ValueError('Prototype upload wrong column/client/round/feature')
            if int(tensors['counts'].sum())!=c.train_samples:raise ValueError('Collect only one last epoch per column')
            columns[column].append(dict(tensors,**meta));uploads+=len(packet)
    decode_seconds=time.perf_counter()-tick;merge_started=time.perf_counter()
    banks=[];details=[];d=server.global_model_set[0].fc3.in_features
    for k,packets in enumerate(columns):
        selected,sources=contributors(packets,server.groups[k],server.args.multi_proto_scope,server.num_classes)
        values,ids=shared.equal_client_means(packets,server.num_classes,d,source,allowed_by_class=selected)
        banks.append(values)
        for label in range(server.num_classes):
            valid=bool(values['valid'][label]);norm=float(values['prototypes'][label].norm()) if valid else None
            details.append(dict(loop_round=server.loop_round,completed_train_count=source,model_id=k,class_id=label,
                contributor_scope=server.args.multi_proto_scope,source=sources[label],contributor_ids=ids[label],
                contributor_count=int(values['contributors'][label]),class_samples=int(values['samples'][label]),
                prototype_l2_norm=norm,valid=valid,valid_zero_mean=valid and norm==0.,invalid_reason=None if valid else 'no_contributors',
                source_completed_train_count=source,first_usable_completed_train_count=source+1,
                group_fingerprint=server.group_payload['group_fingerprint']))
    tensors={key:torch.stack([b[key] for b in banks]) for key in banks[0]}
    meta=dict(source_completed_train_count=source,K=server.K,feature_position=server.args.multi_proto_feature,
        contributor_scope=server.args.multi_proto_scope,aggregation='equal_client_class_means',
        statistics_representation='raw_sum_count_last_training_epoch_per_column',prototype_representation='equal_client_raw_class_means_per_column',
        group_fingerprint=server.group_payload['group_fingerprint'],initial_sha256=server.initial_hash,
        data_fingerprint=server.context['data_fingerprint'])
    merge_seconds=time.perf_counter()-merge_started;encode_started=time.perf_counter()
    server.post_proto_packet=encode_packet('last_epoch_columns_library',tensors,meta);sha=packet_sha(server.post_proto_packet)
    encode_seconds=time.perf_counter()-encode_started
    server.post_proto_classes.extend(dict(row,library_sha256=sha) for row in details)
    server.post_proto_rounds.append(dict(loop_round=server.loop_round,completed_train_count=source,
        source_completed_train_count=source,first_usable_completed_train_count=source+1,library_sha256=sha,
        collected_training_samples=sum(int(p['counts'].sum()) for packets in columns for p in packets),extra_collection_forwards=0,
        statistics_upload_bytes=uploads,library_packet_bytes=len(server.post_proto_packet),
        actual_download_bytes=len(server.previous_post_proto_packet)*len(server.clients) if server.previous_post_proto_packet else 0,
        decode_seconds=decode_seconds,merge_seconds=merge_seconds,library_encode_seconds=encode_seconds,
        server_prototype_total_seconds=time.perf_counter()-tick,distribution_seconds=server.post_proto_distribution_seconds))
    for c in server.clients:c.post_proto_local_packets=[]
    print(f'[LastEpochColumns] R{source}->R{source+1}, K={server.K}, scope={server.args.multi_proto_scope}, valid={int(tensors["valid"].sum())}, upload={uploads}B')


def train(client):
    from utils.cnn512_reporting import model_digest
    if client.download_round is None or client.upload_round==client.download_round or len(client.model_set)!=client.K:
        raise RuntimeError('Require fresh complete K-model download')
    if client.download_hashes!=[model_digest(m) for m in client.download_servers]:raise RuntimeError('Downloaded server changed before training')
    started=time.perf_counter();records=[];packets=[]
    for column in range(client.K):
        row,packet=shared.train(client,column,multicolumn=True)
        row.update(assigned_model_id=client.post_proto_assignment,contributor_scope=client.args.multi_proto_scope,
            group_fingerprint=client.post_proto_group_fingerprint,group_source_sha256=client.post_proto_group_source_sha256,
            prototype_library_sha256=client.post_proto_receipt['sha256'] if client.post_proto_receipt else None)
        if shared_bank(client.args):
            row.update(bank_mode='shared',actual_prototype_bank_count=1 if uses_bank(client.args) and client.post_proto_receipt is not None else 0,K=client.K)
        records.append(row);packets.append(packet)
        print(f'[LastEpochLocal] R{client.download_round+1} client={client.id} column={column} CE={row["ce_loss"]:.8g} F={row["regularization_loss"]:.8g} MSE={row["raw_mse"]:.8g} source={row["used_proto_source_completed_train_count"]} steps={row["train_batches"]}')
    client.train_stats=records;client.post_proto_local_packets=packets if uses_bank(client.args) else []
    client.upload_round=client.download_round
    client.train_time_cost['num_rounds']+=1;client.train_time_cost['total_cost']+=time.perf_counter()-started


def save_latest(server):
    if shared_bank(server.args):
        from utils.multi_shared_proto import save_latest as save_shared
        return save_shared(server)
    if server.post_proto_packet:
        for root in (server.save_folder_name,server.final_model_dir()):
            (Path(root)/'Last_epoch_columns_prototypes.packet').write_bytes(server.post_proto_packet)


def manifest_value(server):
    used=[r['used_proto_source_completed_train_count'] for r in server.training_records if r.get('used_proto_source_completed_train_count') is not None]
    result=dict(protocol=protocol(server.args),group_source=server.group_reuse_receipt,
        class_records=len(server.post_proto_classes),round_records=len(server.post_proto_rounds),extra_collection_forward_count=0,
        latest_generated_source_completed_train_count=server.post_proto_rounds[-1]['source_completed_train_count'] if server.post_proto_rounds else None,
        latest_training_source_completed_train_count=max(used) if used else None,
        actual_prototype_upload_bytes=sum(r['statistics_upload_bytes'] for r in server.post_proto_rounds),
        actual_prototype_download_bytes=sum(r['actual_download_bytes'] for r in server.post_proto_rounds),
        inference='ordinary post-local low-rank model assigned by fixed membership; no prototype inference/model selection')
    if shared_bank(server.args):
        from utils.multi_shared_proto import manifest_fields
        result.update(manifest_fields(server))
    return result
