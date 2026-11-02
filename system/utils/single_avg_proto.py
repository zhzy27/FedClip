"""Explicit K1 Avg mechanism ablation; legacy group-priority code is untouched.

Only this protocol has private per-client/round loader generators. Prototypes
come from last-epoch TRAIN forwards, are merged at round end, and used NEXT round.
"""
import hashlib
import math
from pathlib import Path
import time


def enabled(args):return bool(getattr(args,'multi_avg_proto_ablation',False))


def uses_bank(args):
    return enabled(args) and args.multi_proto_mode=='global_last_epoch' and args.multi_proto_lambda>0


def validate(args):
    if args.algorithm!='FedMultiGroupProj' or args.num_server_models!=1 or args.multi_group_mode!='group_avg_only':
        raise ValueError('multi_avg_proto_ablation requires FedMultiGroupProj K1 group_avg_only (all clients Avg)')
    if args.model_family!='Decom_CNN-5-512' or args.target_head_mode!='shared' or args.target_local_schedule!='joint':
        raise ValueError('Avg prototype ablation requires matching Hyper_CNN_512, shared head and joint')
    if (getattr(args,'multi_kd_mode','none')!='none' or getattr(args,'target_prox_mu',0.)
            or getattr(args,'target_head_l2_lambda',0.) or getattr(args,'target_z2_lambda',0.)
            or getattr(args,'target_c0_replay_mode','none')!='none'):
        raise ValueError('Avg prototype ablation does not mix KD/Prox/head L2/Z2/replay')
    if (getattr(args,'multi_warmup_rounds',0) or getattr(args,'multi_group_file','') or
            getattr(args,'multi_group_publish','') or getattr(args,'multi_group_wait_seconds',0.)):
        raise ValueError('K1 all-client Avg has no warmup or grouping files/publication/waiting')
    if args.multi_proto_mode not in ('none','global_last_epoch') or args.multi_proto_loss!='mse':
        raise ValueError('This ablation supports only none/global_last_epoch with MSE; no Z2 or group-priority protocol')
    if not math.isfinite(args.multi_proto_lambda) or args.multi_proto_lambda<0:
        raise ValueError('multi_proto_lambda must be finite and nonnegative')
    if args.multi_proto_mode=='none' and args.multi_proto_lambda:
        raise ValueError('Nonzero prototype coefficient requires global_last_epoch')
    if args.multi_proto_target_norm!=1.:
        raise ValueError('Global raw prototypes do not use a fixed target norm')
    if args.multi_proto_mode=='global_last_epoch' and args.multi_proto_target_type!='raw_mean':
        raise ValueError('global_last_epoch requires explicit multi_proto_target_type=raw_mean')
    if args.multi_proto_feature not in ('base_output','classifier_input') or args.multi_proto_diagnostics not in (0,1):
        raise ValueError('Invalid ablation feature position or diagnostics flag')
    if type(args.multi_proto_start_round) is not int or args.multi_proto_start_round<2:
        raise ValueError('Global prototype MSE starts no earlier than completed round 2')
    if uses_bank(args) and args.multi_proto_start_round>args.global_rounds+1:
        raise ValueError('Training budget has no round eligible for previous-round prototype MSE')
    if not math.isfinite(args.multi_proto_eps) or args.multi_proto_eps<=0:
        raise ValueError('Prototype epsilon must be finite positive')
    if args.multi_proto_eps!=1e-12:
        raise ValueError('Raw class means/MSE do not use normalization epsilon; leave it at the default')
    if args.multi_proto_batch_size!=16:
        raise ValueError('No extra collection loader in global_last_epoch; leave unused collection batch option at default')
    if args.is_regular not in (0,1) or not math.isfinite(args.regular_lamda) or args.regular_lamda<0:
        raise ValueError('Invalid Frobenius switch/coefficient')
    if not getattr(args,'multi_svd',False) and args.is_regular:
        raise ValueError('Full-model ablation must disable F; no dense F implementation is added')


def protocol(args):
    return dict(name='single_model_Avg_global_last_epoch_v1',
        model_training='svd_heterogeneous' if getattr(args,'multi_svd',False) else 'full_Hyper_CNN_512',
        prototype_enabled=uses_bank(args),requested_mode=args.multi_proto_mode,
        coefficient=args.multi_proto_lambda,effective_coefficient=args.multi_proto_lambda if uses_bank(args) else 0.,
        feature_position=args.multi_proto_feature,loss='mean_over_batch_and_features((raw_feature-detached_previous_global_class_mean)^2); missing-class positions have zero loss/gradient',
        collection='only last local epoch existing TRAIN forwards; immediate detach; float64 sum/int64 count; no extra forward',
        aggregation='equal weight among contributing CLIENT CLASS MEANS, not class-sample weighted',
        timing='end round t -> frozen targets throughout round t+1; first round CE plus configured F only',
        missing='no contributors invalid; zero or near-zero class mean remains valid',
        normalization='none; no fixed norm; no EMA; no group priority',
        is_regular=args.is_regular,regular_lamda=args.regular_lamda,local_grad_clip=args.local_grad_clip,
        diagnostics=bool(args.multi_proto_diagnostics),
        loss_mean_unit='actual SGD sample-visit weighted training objectives; MSE batch reduction uses valid samples and feature dimensions',
        gradient_mean_unit='one pre-clip total L2 norm / clipping coefficient per SGD step; trigger fraction per step',
        sample_order='private SHA256(single-avg-proto-v1:seed:loop:client_id) Torch Generator; identical across all eight configurations',
        initialization='independent mother option multi_svd_init, shared by full/SVD; only initial recovery for full training',
        classification='original fc3; prototypes never used at inference',
        interpretation='parameter aggregation PLUS class prototypes; mechanism ablation, not full FedProto; dense and low-rank capacities differ')


def loader_seed(args,cid,loop):
    key=f'single-avg-proto-v1:{args.seed}:{loop}:{cid}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63)


class LastEpochStats:
    def __init__(self,classes,dimension):
        import torch
        self.sums=torch.zeros(classes,dimension,dtype=torch.float64)
        self.counts=torch.zeros(classes,dtype=torch.int64)

    def add(self,features,labels):
        import torch
        z=features.detach().to(device='cpu',dtype=torch.float64)
        y=labels.detach().to(device='cpu',dtype=torch.int64)
        if z.ndim!=2 or z.shape[1]!=self.sums.shape[1] or len(z)!=len(y) or not torch.isfinite(z).all():
            raise ValueError('Invalid last-epoch raw features')
        if len(y)==0 or y.min()<0 or y.max()>=len(self.counts):raise ValueError('Invalid last-epoch labels')
        self.sums.index_add_(0,y,z);self.counts.add_(torch.bincount(y,minlength=len(self.counts)))
        if not torch.isfinite(self.sums).all():raise FloatingPointError('Nonfinite local class sum')


def equal_client_means(packets,classes,dimension,source):
    """Inputs are sum/count statistics; first divide locally, THEN average clients."""
    import torch
    if not packets or len({p['client_id'] for p in packets})!=len(packets):raise ValueError('Missing/duplicate prototype contributors')
    means=torch.zeros(classes,dimension,dtype=torch.float64)
    contributors=torch.zeros(classes,dtype=torch.int64);samples=torch.zeros(classes,dtype=torch.int64)
    ids=[[] for _ in range(classes)]
    for p in sorted(packets,key=lambda p:p['client_id']):
        if p['source_completed_train_count']!=source:raise ValueError('Stale local prototype statistics')
        s,n=p['sums'],p['counts']
        if (s.shape!=means.shape or n.shape!=samples.shape or s.dtype!=torch.float64 or n.dtype!=torch.int64
                or s.requires_grad or n.requires_grad or (n<0).any() or not torch.isfinite(s).all()):
            raise ValueError('Invalid last-epoch sum/count packet')
        mask=n>0
        means[mask]+=s[mask]/n[mask,None]
        contributors+=mask.long();samples+=n
        for label in mask.nonzero().flatten().tolist():ids[label].append(p['client_id'])
    valid=contributors>0
    means[valid]/=contributors[valid,None]
    if not torch.isfinite(means).all():raise FloatingPointError('Nonfinite global class prototype')
    return dict(prototypes=means,valid=valid,contributors=contributors,samples=samples),ids


def global_mse(features,labels,prototypes,valid):
    """Official missing-label convention: no error there, denominator is B*d."""
    import torch
    import torch.nn.functional as F
    if features.ndim!=2 or prototypes.shape[1]!=features.shape[1]:raise ValueError('Prototype feature shape mismatch')
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():raise FloatingPointError('Nonfinite raw MSE input')
    mask=valid.detach().to(labels.device)[labels];nvalid=int(mask.sum())
    if not nvalid:return features.reshape(-1)[:1].sum()*0.,0
    target=features.detach().clone()  # Missing classes get zero ERROR, never a fabricated zero prototype.
    target[mask]=prototypes.detach().to(features)[labels[mask]]
    value=F.mse_loss(features,target,reduction='mean')
    if not torch.isfinite(value):raise FloatingPointError('Nonfinite global prototype MSE')
    return value,nvalid


def initialize(server):
    from utils.multi_group_clustering import digest_json
    ids=list(range(server.num_clients));server.groups=[ids]
    server.group_payload=dict(groups=[ids],client_to_model={str(i):0 for i in ids},
        group_fingerprint=digest_json(dict(protocol='fixed_all_client_Avg',clients=ids)),
        algorithm=dict(name='fixed_all_client_Avg_no_clustering'),context=server.context)
    server.avg_proto_packet=None;server.avg_proto_classes=[];server.avg_proto_rounds=[]


def prepare_round(server):
    from utils.multi_group_proto import decode_packet,packet_sha
    packet=server.avg_proto_packet
    server.previous_avg_proto_packet=packet
    if not uses_bank(server.args) and packet is not None:raise RuntimeError('Disabled prototype protocol has a library')
    for c in server.clients:
        c.avg_proto_bank=None;c.avg_proto_receipt=None
        if packet is not None:
            values,meta=decode_packet(packet,'global_last_epoch_library')
            if (meta['source_completed_train_count']!=server.loop_round or
                    meta['feature_position']!=server.args.multi_proto_feature or
                    meta['aggregation']!='equal_client_class_means'):
                raise ValueError('Wrong round/feature/representation for previous global prototypes')
            c.avg_proto_bank={k:v.detach().clone() for k,v in values.items()}
            c.avg_proto_receipt=dict(meta,sha256=packet_sha(packet))


def finish_round(server):
    if not uses_bank(server.args):return
    import torch
    from utils.multi_group_proto import encode_packet,decode_packet,packet_sha
    tick=time.perf_counter();packets=[];byte_count=0;source=server.loop_round+1
    for c in server.clients:
        if c.upload_round!=server.loop_round or c.avg_proto_local is None:raise RuntimeError('Incomplete current local prototypes')
        packet=c.avg_proto_local;values,meta=decode_packet(packet,'global_last_epoch_local')
        if (meta['client_id']!=c.id or meta['source_completed_train_count']!=source
                or meta['feature_position']!=server.args.multi_proto_feature):raise ValueError('Local prototype packet mismatch')
        packets.append(dict(values,**meta));byte_count+=len(packet)
        if int(values['counts'].sum())!=c.train_samples:raise ValueError('Prototype count must cover only ONE complete final local epoch')
    d=server.global_model_set[0].fc3.in_features
    values,ids=equal_client_means(packets,server.num_classes,d,source)
    meta=dict(source_completed_train_count=source,feature_position=server.args.multi_proto_feature,
        aggregation='equal_client_class_means',statistics_representation='raw_sum_count_last_training_epoch',
        prototype_representation='equal_client_raw_class_means',initial_sha256=server.initial_hash,
        data_fingerprint=server.context['data_fingerprint'])
    server.avg_proto_packet=encode_packet('global_last_epoch_library',values,meta)
    sha=packet_sha(server.avg_proto_packet)
    for label in range(server.num_classes):
        valid=bool(values['valid'][label]);norm=float(values['prototypes'][label].norm()) if valid else None
        server.avg_proto_classes.append(dict(loop_round=server.loop_round,completed_train_count=source,class_id=label,
            contributor_ids=ids[label],contributor_count=int(values['contributors'][label]),
            class_samples=int(values['samples'][label]),prototype_l2_norm=norm,valid=valid,
            invalid_reason=None if valid else 'no_samples',valid_zero_mean=valid and norm==0.,
            source_completed_train_count=source,first_usable_completed_train_count=source+1,library_sha256=sha))
    server.avg_proto_rounds.append(dict(loop_round=server.loop_round,completed_train_count=source,
        source_completed_train_count=source,first_usable_completed_train_count=source+1,
        collected_training_samples=sum(int(p['counts'].sum()) for p in packets),extra_collection_forwards=0,
        statistics_upload_bytes=byte_count,library_packet_bytes=len(server.avg_proto_packet),
        actual_download_bytes=(len(server.clients)*len(server.previous_avg_proto_packet) if server.previous_avg_proto_packet else 0),
        valid_classes=int(values['valid'].sum()),merge_seconds=time.perf_counter()-tick,library_sha256=sha))
    print(f'[GlobalLastEpochProto] source=R{source}, usable=R{source+1}, equal-client means, valid={int(values["valid"].sum())}')
    for c in server.clients:c.avg_proto_local=None  # No local statistics history or per-sample features.


def train(client):
    """Shared full/SVD SGD path, entered ONLY by the explicit new protocol."""
    import torch
    from flcore.clients.clientTargetProj import clientTargetProj
    from utils.multi_group_proto import Features,encode_packet
    from utils import multi_group_svd as svd
    if client.K!=1 or client.download_round is None or client.upload_round==client.download_round:
        raise RuntimeError('Require one fresh K1 download before training')
    model=client.model_set[0];optimizer=client.optimizer_set[0];args=client.args
    params=[p for p in model.parameters() if p.requires_grad]
    if [id(p) for p in params]!=[id(p) for g in optimizer.param_groups for p in g['params']]:
        raise RuntimeError('Optimizer bound to stale local parameters')
    proto=uses_bank(args);bank=getattr(client,'avg_proto_bank',None)
    receipt=getattr(client,'avg_proto_receipt',None)
    if bank is not None and (not proto or receipt['source_completed_train_count']!=client.download_round):
        raise RuntimeError('Only previous-round global prototypes may train this round')
    if proto and client.download_round>0 and bank is None:raise RuntimeError('Missing previous-round global prototypes')
    can_use=proto and bank is not None and client.download_round+1>=args.multi_proto_start_round
    collector=LastEpochStats(model.fc3.out_features,model.fc3.in_features) if proto else None
    loader=client.load_train_data();started=time.perf_counter();model.train()
    ce_sum=f_sum=mse_sum=weighted_sum=total_sum=feature_norm=feature_square=norm_sum=coef_sum=0.
    visits=steps=valid_visits=triggered=observed=0
    for epoch in range(client.local_epochs):
        for x,y in loader:
            x,y=x.to(client.device),y.to(client.device);optimizer.zero_grad()
            if proto or args.multi_proto_diagnostics:
                with Features(model,args.multi_proto_feature) as capture:
                    ce,f=clientTargetProj._objective(client,model,x,y);features=capture.get()
                    # Detach NOW: training feature, before backward/step; never retain epoch graphs.
                    if collector is not None and epoch==client.local_epochs-1:collector.add(features,y)
                    raw,nvalid=(global_mse(features,y,bank['prototypes'],bank['valid']) if can_use else (None,0))
                    if args.multi_proto_diagnostics:
                        z=features.detach();feature_norm+=float(z.norm(dim=1).sum())
                        feature_square+=float(z.square().mean(dim=1).sum());observed+=len(y)
            else:ce,f=clientTargetProj._objective(client,model,x,y);raw,nvalid=None,0
            # Off/zero/first round: preserve the exact CE+F backward graph.
            weighted=args.multi_proto_lambda*raw if raw is not None else None
            loss=ce+f if weighted is None or nvalid==0 else ce+f+weighted
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Avg prototype objective')
            loss.backward();cap=args.local_grad_clip
            if cap is not None and cap>0:
                norm=torch.nn.utils.clip_grad_norm_(params,cap,error_if_nonfinite=True)
            elif args.multi_proto_diagnostics:
                norms=[p.grad.detach().norm(2).to(params[0].device) for p in params if p.grad is not None]
                norm=torch.stack(norms).norm(2)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite unclipped gradient norm')
            if args.multi_proto_diagnostics:
                value=float(norm);coef=float((cap/(norm+1e-6)).clamp(max=1.)) if cap is not None and cap>0 else 1.
                norm_sum+=value;coef_sum+=coef;triggered+=int(coef<1.)
            optimizer.step()
            ce_sum+=float(ce.detach())*len(y);f_sum+=float(f.detach())*len(y)
            mse_sum+=(float(raw.detach()) if raw is not None else 0.)*len(y)
            weighted_sum+=(float(weighted.detach()) if weighted is not None else 0.)*len(y)
            total_sum+=float(loss.detach())*len(y);visits+=len(y);steps+=1;valid_visits+=nvalid
    if not visits:raise ValueError('Nonempty local data required')
    if str(client.device).startswith('cuda'):torch.cuda.synchronize(client.device)
    train_seconds=time.perf_counter()-started
    row=dict(client_id=client.id,model_id=0,local_epochs=client.local_epochs,train_batches=steps,
        processed_train_samples=visits,ce_loss=ce_sum/visits,regularization_loss=f_sum/visits,proximal_loss=0.,
        train_seconds=train_seconds,raw_mse=mse_sum/visits,weighted_mse=weighted_sum/visits,total_loss=total_sum/visits,
        feature_position=args.multi_proto_feature,train_feature_l2_mean=feature_norm/observed if observed else None,
        train_feature_square_mean=feature_square/observed if observed else None,
        head_weight_l2_norm=float(model.fc3.weight.detach().norm()),
        pre_clip_grad_norm_mean_per_step=norm_sum/steps if args.multi_proto_diagnostics else None,
        clip_coefficient_mean_per_step=coef_sum/steps if args.multi_proto_diagnostics else None,
        clip_trigger_fraction=triggered/steps if args.multi_proto_diagnostics else None,
        effective_grad_clip=cap if cap is not None and cap>0 else None,
        effective_F_lambda=args.regular_lamda if args.is_regular else 0.,
        effective_proto_lambda=args.multi_proto_lambda if can_use else 0.,proto_valid_sample_visits=valid_visits,
        used_proto_source_completed_train_count=receipt['source_completed_train_count'] if can_use else None,
        trainable_parameter_count=sum(p.numel() for p in params),
        train_loader_seed=loader_seed(args,client.id,client.download_round),
        prototype_collected_samples=int(collector.counts.sum()) if collector is not None else 0)
    client.avg_proto_local=None
    if collector is not None:
        client.avg_proto_local=encode_packet('global_last_epoch_local',dict(sums=collector.sums,counts=collector.counts),
            dict(client_id=client.id,source_completed_train_count=client.download_round+1,feature_position=args.multi_proto_feature))
    if svd.enabled(args):
        tick=time.perf_counter();full=svd.recover_parameters(model);client.full_uploads[0]=full
        base=dict(client.download_servers[0].named_parameters())
        if full.keys()!=base.keys() or any(full[n].shape!=base[n].shape for n in base):
            raise ValueError('Recovered K1 upload parameter names/shapes differ from the full mother')
        if args.multi_svd_diagnostics:
            for key,left,right in [('svd_truncation',client.download_starts[0],base),
                ('local_training_recovered',full,client.download_starts[0]),('global_delta',full,base)]:
                squared=svd.distance_squared(left,right);row[key+'_squared_norm']=squared;row[key+'_norm']=squared**.5
        client.download_starts[0]=None
        row.update(capacity_level=client.capacity_level,rank_rate=client.rank_rate,recovered_parameter_count=sum(p.numel() for p in full.values()))
        torch.save(svd.client_payload(client,0),Path(client.save_folder_name)/f'Client_{client.id}_model_0.pt')
        row['upload_recovery_checkpoint_seconds']=time.perf_counter()-tick
    else:row.update(rank_rate=1.,recovered_parameter_count=sum(p.numel() for p in model.parameters()))
    client.train_stats=[row];client.upload_round=client.download_round
    client.train_time_cost['num_rounds']+=1;client.train_time_cost['total_cost']+=time.perf_counter()-started
    print(f'[AvgProtoLocal] round={client.download_round+1} client={client.id} CE={row["ce_loss"]:.8g} F={row["regularization_loss"]:.8g} '
        f'MSE={row["raw_mse"]:.8g} source={row["used_proto_source_completed_train_count"]} collected={row["prototype_collected_samples"]} steps={steps}')


def save_latest(server):
    if server.avg_proto_packet:
        for directory in (server.save_folder_name,server.final_model_dir()):
            (Path(directory)/'Global_last_epoch_prototypes.packet').write_bytes(server.avg_proto_packet)


def manifest_value(server):
    return dict(protocol=protocol(server.args),class_records=len(server.avg_proto_classes),
        round_records=len(server.avg_proto_rounds),extra_collection_forward_count=0,
        training_initialization_independent=True,grouping='none: all clients assigned model 0',
        actual_prototype_upload_bytes=sum(r['statistics_upload_bytes'] for r in server.avg_proto_rounds),
        actual_prototype_download_bytes=sum(r['actual_download_bytes'] for r in server.avg_proto_rounds))
