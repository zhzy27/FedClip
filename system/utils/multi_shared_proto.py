"""One cross-model raw global prototype bank for the opt-in last-epoch protocol.

No local SGD, features, MSE, sampling, model aggregation or grouping is changed.
Only this branch changes the contributor reduction and bank distribution shape.
"""
from pathlib import Path
import time

import torch

from utils import single_avg_proto as shared
from utils.multi_group_proto import decode_packet,encode_packet,packet_sha

KIND='last_epoch_shared_library'
FORMULA='mean_owning_clients(mean_model_columns(raw_class_sum/class_count))'


def uses_bank(args):
    from utils.multi_last_epoch_proto import uses_bank as active
    return active(args)


def validate_library(values,meta,*,K,classes,dimension,source,feature,group,initial,data):
    if (meta.get('bank_mode')!='shared' or meta.get('actual_prototype_bank_count')!=1
            or meta.get('K')!=K or meta.get('source_completed_train_count')!=source
            or meta.get('feature_position')!=feature or meta.get('contributor_scope')!='global'
            or meta.get('aggregation')!=FORMULA or meta.get('group_fingerprint')!=group
            or meta.get('initial_sha256')!=initial or meta.get('data_fingerprint')!=data):
        raise ValueError('Shared prototype library round/schema/feature/scope/group/initialization/data mismatch')
    expected=dict(prototypes=((classes,dimension),torch.float64),valid=((classes,),torch.bool),
        contributors=((classes,),torch.int64),samples=((classes,),torch.int64),
        model_contributors=((classes,),torch.int64),statistics_samples=((classes,),torch.int64))
    if set(values)!=set(expected):raise ValueError('Shared prototype tensor set mismatch')
    for name,(shape,dtype) in expected.items():
        value=values[name]
        if value.shape!=shape or value.dtype!=dtype or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError('Invalid shared prototype shape/dtype/gradient/finite state')
    if (not torch.equal(values['valid'],values['contributors']>0) or (values['samples']<0).any()
            or (values['contributors']<0).any()
            or not torch.equal(values['model_contributors'],values['contributors']*K)
            or not torch.equal(values['statistics_samples'],values['samples']*K)
            or ((values['samples']>0)!=values['valid']).any()
            or (values['prototypes'][~values['valid']]!=0).any()):
        raise ValueError('Shared prototype validity/contributor/sample accounting mismatch')


def prepare_round(server):
    tick=time.perf_counter();packet=server.post_proto_packet;server.previous_post_proto_packet=packet
    if packet is not None and not uses_bank(server.args):raise RuntimeError('Disabled shared protocol has a library')
    values=meta=None
    if packet is not None:
        values,meta=decode_packet(packet,KIND)
        validate_library(values,meta,K=server.K,classes=server.num_classes,
            dimension=server.global_model_set[0].fc3.in_features,source=server.loop_round,
            feature=server.args.multi_proto_feature,group=server.group_payload['group_fingerprint'],
            initial=server.initial_hash,data=server.context['data_fingerprint'])
        if (values['contributors']>server.num_clients).any():raise ValueError('Shared library contributor count exceeds client population')
        if server.loop_round+1>=server.args.multi_proto_start_round:
            server.last_used_shared_proto_packet=packet
            server.last_used_shared_proto_completed_train_count=server.loop_round+1
    # Each client receives ONE P+mask pair and reuses that object for all K
    # columns. No K clones and no per-column logical download are created.
    for c in server.clients:
        c.post_proto_banks=None;c.post_proto_shared_bank=None;c.post_proto_receipt=None
        c.post_proto_assignment=int(server.group_payload['client_to_model'][str(c.id)])
        c.post_proto_group_source_sha256=server.group_source_sha256
        c.post_proto_group_fingerprint=server.group_payload['group_fingerprint']
        if packet is not None:
            c.post_proto_shared_bank=dict(prototypes=values['prototypes'].detach().clone(),
                valid=values['valid'].detach().clone(),bank_mode='shared')
            c.post_proto_receipt=dict(meta,sha256=packet_sha(packet))
    server.post_proto_distribution_seconds=time.perf_counter()-tick


def training_bank(client,column):
    if not 0<=column<client.K:raise RuntimeError('Invalid training model column')
    bank=getattr(client,'post_proto_shared_bank',None)
    if bank is not None:
        receipt=client.post_proto_receipt
        model=client.model_set[column]
        if (not uses_bank(client.args) or receipt is None or receipt.get('bank_mode')!='shared'
                or receipt.get('K')!=client.K or receipt.get('contributor_scope')!='global'
                or bank.get('bank_mode')!='shared' or bank['prototypes'].shape!=(model.fc3.out_features,model.fc3.in_features)
                or bank['valid'].shape!=(model.fc3.out_features,)
                or bank['prototypes'].requires_grad or bank['valid'].requires_grad):
            raise RuntimeError('Wrong shared library bound to training session')
    return bank


def finish_round(server):
    if not uses_bank(server.args):return
    tick=time.perf_counter();source=server.loop_round+1;uploads=0;columns=[[] for _ in range(server.K)]
    if len(server.clients)!=server.num_clients or {c.id for c in server.clients}!=set(range(server.num_clients)):
        raise RuntimeError('Incomplete shared-prototype client population')
    for c in server.clients:
        if c.upload_round!=server.loop_round or len(c.post_proto_local_packets)!=server.K:
            raise RuntimeError('Incomplete N x K shared prototype uploads')
        for k,packet in enumerate(c.post_proto_local_packets):
            if packet is None:raise RuntimeError('Missing shared-prototype model-column statistics')
            values,meta=decode_packet(packet,'last_epoch_column_local')
            if (meta.get('model_id')!=k or meta.get('client_id')!=c.id or meta.get('source_completed_train_count')!=source
                    or meta.get('feature_position')!=server.args.multi_proto_feature):
                raise ValueError('Shared prototype upload wrong column/client/round/feature')
            if int(values['counts'].sum())!=c.train_samples:raise ValueError('Shared statistics must cover exactly one last epoch')
            columns[k].append(dict(values,**meta));uploads+=len(packet)
    decoded=time.perf_counter();dimension=server.global_model_set[0].fc3.in_features
    values,ids=shared.equal_client_column_means(columns,server.num_classes,dimension,source)
    meta=dict(bank_mode='shared',actual_prototype_bank_count=1,K=server.K,
        source_completed_train_count=source,first_usable_completed_train_count=source+1,
        feature_position=server.args.multi_proto_feature,contributor_scope='global',aggregation=FORMULA,
        statistics_representation='raw_sum_count_last_training_epoch_per_column',
        prototype_representation='client_then_column_equal_raw_class_means_shared',
        group_fingerprint=server.group_payload['group_fingerprint'],initial_sha256=server.initial_hash,
        data_fingerprint=server.context['data_fingerprint'])
    merged=time.perf_counter();packet=encode_packet(KIND,values,meta);sha=packet_sha(packet)
    encoded=time.perf_counter()
    previous=server.previous_post_proto_packet
    used_meta=decode_packet(previous,KIND)[1] if previous is not None else None
    for label in range(server.num_classes):
        valid=bool(values['valid'][label]);norm=float(values['prototypes'][label].norm()) if valid else None
        server.post_proto_classes.append(dict(loop_round=server.loop_round,completed_train_count=source,
            model_id=None,bank_mode='shared',actual_prototype_bank_count=1,K=server.K,class_id=label,
            contributor_scope='global',aggregation=FORMULA,source='global' if valid else 'invalid',
            contributor_ids=ids[label],contributor_count=int(values['contributors'][label]),
            contributor_client_count=int(values['contributors'][label]),
            client_model_contributor_count=int(values['model_contributors'][label]),
            class_samples=int(values['samples'][label]),deduplicated_class_samples=int(values['samples'][label]),
            cross_column_class_sample_visits=int(values['statistics_samples'][label]),prototype_l2_norm=norm,
            valid=valid,valid_zero_mean=valid and norm==0.,invalid_reason=None if valid else 'no_contributors',
            source_completed_train_count=source,first_usable_completed_train_count=source+1,
            group_fingerprint=server.group_payload['group_fingerprint'],library_sha256=sha))
    server.post_proto_packet=packet
    server.post_proto_rounds.append(dict(loop_round=server.loop_round,completed_train_count=source,
        bank_mode='shared',actual_prototype_bank_count=1,K=server.K,prototype_shape=list(values['prototypes'].shape),
        aggregation=FORMULA,source_completed_train_count=source,first_usable_completed_train_count=source+1,
        training_used_source_completed_train_count=used_meta['source_completed_train_count'] if used_meta is not None and source>=server.args.multi_proto_start_round else None,
        training_used_library_sha256=packet_sha(previous) if used_meta is not None and source>=server.args.multi_proto_start_round else None,
        library_sha256=sha,collected_training_samples=int(values['statistics_samples'].sum()),
        deduplicated_training_samples=int(values['samples'].sum()),extra_collection_forwards=0,
        statistics_upload_bytes=uploads,library_packet_bytes=len(packet),
        actual_download_bytes=len(previous)*len(server.clients) if previous else 0,
        actual_library_delivery_count=len(server.clients) if previous else 0,
        decode_seconds=decoded-tick,merge_seconds=merged-decoded,library_encode_seconds=encoded-merged,
        client_statistics_encode_seconds=sum(row['prototype_statistics_encode_seconds'] for c in server.clients for row in c.train_stats),
        server_prototype_total_seconds=time.perf_counter()-tick,distribution_seconds=server.post_proto_distribution_seconds))
    for c in server.clients:c.post_proto_local_packets=[]
    print(f'[SharedLastEpoch] R{source}->R{source+1}, ONE [{server.num_classes},{dimension}] bank, K={server.K}, upload={uploads}B')


def save_latest(server):
    for root in (Path(server.save_folder_name),Path(server.final_model_dir())):
        if server.post_proto_packet:(root/'Last_epoch_shared_prototypes.packet').write_bytes(server.post_proto_packet)
        used=getattr(server,'last_used_shared_proto_packet',None)
        if used:(root/'Last_epoch_shared_training_prototypes.packet').write_bytes(used)


def manifest_fields(server):
    packet=server.post_proto_packet;used=getattr(server,'last_used_shared_proto_packet',None)
    return dict(bank_mode='shared',actual_prototype_bank_count=1 if uses_bank(server.args) else 0,K=server.K,
        aggregation=FORMULA,prototype_shape=[server.num_classes,server.global_model_set[0].fc3.in_features] if packet else None,
        latest_generated_file='Last_epoch_shared_prototypes.packet' if packet else None,
        latest_generated_sha256=packet_sha(packet) if packet else None,
        latest_training_used_file='Last_epoch_shared_training_prototypes.packet' if used else None,
        latest_training_used_sha256=packet_sha(used) if used else None,
        latest_training_completed_train_count=getattr(server,'last_used_shared_proto_completed_train_count',None),
        library_loading="values,meta=utils.multi_group_proto.decode_packet(Path(file).read_bytes(),'last_epoch_shared_library'); prototypes.shape=[C,D]",
        communication_scope='actual encoded in-process packets; upload N*K local statistics; download ONE shared packet per client per round; not physical network measurements')
