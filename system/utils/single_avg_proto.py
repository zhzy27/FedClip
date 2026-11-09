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
    return enabled(args) and args.multi_proto_mode in ('global_last_epoch','global_pre_train') and args.multi_proto_lambda>0


def extended(args):
    return enabled(args) and (getattr(args,'multi_proto_aggregation','client_equal')!='client_equal' or args.multi_proto_mode=='global_pre_train')


def pre_train(args):return args.multi_proto_mode=='global_pre_train'


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
    if args.multi_proto_mode not in ('none','global_last_epoch','global_pre_train') or args.multi_proto_loss!='mse':
        raise ValueError('This ablation supports only none/global_last_epoch with MSE; no Z2 or group-priority protocol')
    if not math.isfinite(args.multi_proto_lambda) or args.multi_proto_lambda<0:
        raise ValueError('multi_proto_lambda must be finite and nonnegative')
    if args.multi_proto_mode=='none' and args.multi_proto_lambda:
        raise ValueError('Nonzero prototype coefficient requires global_last_epoch')
    if args.multi_proto_target_norm!=1.:
        raise ValueError('Global raw prototypes do not use a fixed target norm')
    if args.multi_proto_mode in ('global_last_epoch','global_pre_train') and args.multi_proto_target_type!='raw_mean':
        raise ValueError('global_last_epoch requires explicit multi_proto_target_type=raw_mean')
    if args.multi_proto_feature not in ('base_output','classifier_input') or args.multi_proto_diagnostics not in (0,1):
        raise ValueError('Invalid ablation feature position or diagnostics flag')
    if type(args.multi_proto_start_round) is not int or args.multi_proto_start_round<2:
        raise ValueError('Global prototype MSE starts no earlier than completed round 2')
    if uses_bank(args) and args.multi_proto_start_round>args.global_rounds+1:
        raise ValueError('Training budget has no round eligible for previous-round prototype MSE')
    if extended(args) and uses_bank(args) and not getattr(args,'multi_svd',False):
        raise ValueError('K1 class_sample v2 is validated only for the actual heterogeneous SVD training path')
    if not math.isfinite(args.multi_proto_eps) or args.multi_proto_eps<=0:
        raise ValueError('Prototype epsilon must be finite positive')
    if args.multi_proto_eps!=1e-12:
        raise ValueError('Raw class means/MSE do not use normalization epsilon; leave it at the default')
    if pre_train(args):
        if getattr(args,'multi_proto_aggregation','client_equal')!='class_sample':
            raise ValueError('global_pre_train v1 requires explicit multi_proto_aggregation=class_sample')
        if type(args.multi_proto_batch_size) is not int or args.multi_proto_batch_size<1:
            raise ValueError('Pre-training collection batch size must be a positive integer')
    elif args.multi_proto_batch_size!=16:
        raise ValueError('No extra collection loader in global_last_epoch; leave unused collection batch option at default')
    if args.is_regular not in (0,1) or not math.isfinite(args.regular_lamda) or args.regular_lamda<0:
        raise ValueError('Invalid Frobenius switch/coefficient')
    if not getattr(args,'multi_svd',False) and args.is_regular:
        raise ValueError('Full-model ablation must disable F; no dense F implementation is added')


def protocol(args):
    if extended(args):
        return extended_protocol(args)
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


def loader_seed(args,cid,loop,column=0):
    key=(f'single-avg-proto-v1:{args.seed}:{loop}:{cid}'+(f':column:{column}' if column else '')).encode()
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


def equal_client_means(packets,classes,dimension,source,allowed_by_class=None):
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
        if allowed_by_class is not None:
            if len(allowed_by_class)!=classes:raise ValueError('Contributor-class count mismatch')
            mask=mask & torch.tensor([p['client_id'] in selected for selected in allowed_by_class],dtype=torch.bool)
        means[mask]+=s[mask]/n[mask,None]
        contributors+=mask.long();samples+=torch.where(mask,n,0) if allowed_by_class is not None else n
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


def equal_client_column_means(columns,classes,dimension,source):
    """One raw shared bank: equal columns WITHIN client, then equal clients.

    Reuse the existing single-client validation and equal-client reducer. The
    second reducer sees unit counts for owning clients (means are already
    computed); real class sample counts are retained separately and never used
    as aggregation weights. Both merge orders are deterministic.
    """
    import torch
    if not columns or any(not packets for packets in columns):raise ValueError('Missing prototype model column')
    lookup=[]
    for k,packets in enumerate(columns):
        ids=[p['client_id'] for p in packets]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate client in prototype column')
        if any(p.get('model_id')!=k for p in packets):raise ValueError('Wrong prototype model column')
        lookup.append({p['client_id']:p for p in packets})
    if any(set(c)!=set(lookup[0]) for c in lookup):raise ValueError('Incomplete client x model statistics')
    reduced=[];samples=torch.zeros(classes,dtype=torch.int64);K=len(columns)
    for cid in sorted(lookup[0]):
        total=torch.zeros(classes,dimension,dtype=torch.float64);counts=None
        for k in range(K):
            p=lookup[k][cid]
            single,_=equal_client_means([p],classes,dimension,source)
            if p['sums'].device.type!='cpu' or p['counts'].device.type!='cpu':raise ValueError('Decoded statistics must be CPU tensors')
            if counts is None:counts=p['counts']
            elif not torch.equal(counts,p['counts']):raise ValueError('Same client class counts differ across model columns')
            if (p['sums'][p['counts']==0]!=0).any():raise ValueError('Nonzero class sum without samples')
            total.add_(single['prototypes'])
        total.div_(K)
        reduced.append(dict(client_id=cid,source_completed_train_count=source,sums=total,counts=(counts>0).long()))
        samples.add_(counts)
    values,ids=equal_client_means(reduced,classes,dimension,source)
    values.update(samples=samples,model_contributors=values['contributors']*K,statistics_samples=samples*K)
    return values,ids


def initialize(server):
    from utils.multi_group_clustering import digest_json
    ids=list(range(server.num_clients));server.groups=[ids]
    server.group_payload=dict(groups=[ids],client_to_model={str(i):0 for i in ids},
        group_fingerprint=digest_json(dict(protocol='fixed_all_client_Avg',clients=ids)),
        algorithm=dict(name='fixed_all_client_Avg_no_clustering'),context=server.context)
    server.avg_proto_packet=None;server.avg_proto_classes=[];server.avg_proto_rounds=[]


def prepare_round(server):
    if extended(server.args) and uses_bank(server.args):return prepare_extended_round(server)
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
    if extended(server.args):return finish_extended_round(server)
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


def train(client,column=0,*,multicolumn=False):
    """Shared full/SVD SGD path, entered ONLY by the explicit new protocol."""
    import torch
    from flcore.clients.clientTargetProj import clientTargetProj
    from utils.multi_group_proto import Features,encode_packet
    from utils import multi_group_svd as svd
    if (not multicolumn and client.K!=1) or client.download_round is None or client.upload_round==client.download_round:
        raise RuntimeError('Require one fresh K1 download before training')
    model=client.model_set[column];optimizer=client.optimizer_set[column];args=client.args
    params=[p for p in model.parameters() if p.requires_grad]
    if [id(p) for p in params]!=[id(p) for g in optimizer.param_groups for p in g['params']]:
        raise RuntimeError('Optimizer bound to stale local parameters')
    from utils.multi_helper_head import monitor as helper_monitor
    helper=helper_monitor(client,column)
    if multicolumn:
        from utils.multi_last_epoch_proto import uses_bank as post_uses_bank,shared_bank
        proto=post_uses_bank(args)
        if shared_bank(args):
            from utils.multi_shared_proto import training_bank
            bank=training_bank(client,column)
        else:
            bank=client.post_proto_banks[column] if client.post_proto_banks is not None else None
        receipt=client.post_proto_receipt
        if bank is not None and not shared_bank(args) and bank.get('model_id')!=column:raise RuntimeError('Prototype library bound to wrong model column')
    else:
        proto=uses_bank(args);bank=getattr(client,'avg_proto_bank',None)
        receipt=getattr(client,'avg_proto_receipt',None)
    new_protocol=not multicolumn and extended(args)
    if new_protocol:
        validate_training_receipt(client,bank,receipt)
    else:
        if bank is not None and (not proto or receipt['source_completed_train_count']!=client.download_round):
            raise RuntimeError('Only previous-round global prototypes may train this round')
        if proto and client.download_round>0 and bank is None:raise RuntimeError('Missing previous-round global prototypes')
    can_use=proto and bank is not None and client.download_round+1>=args.multi_proto_start_round
    collector=LastEpochStats(model.fc3.out_features,model.fc3.in_features) if proto and (multicolumn or not pre_train(args)) else None
    loader=client.load_train_data(model_column=column) if multicolumn else client.load_train_data()
    started=time.perf_counter();model.train()
    ce_sum=f_sum=mse_sum=weighted_sum=total_sum=feature_norm=feature_square=norm_sum=coef_sum=0.
    visits=steps=valid_visits=triggered=observed=0
    collection_seconds=0.;collection_batches=0
    for epoch in range(client.local_epochs):
        for x,y in loader:
            x,y=x.to(client.device),y.to(client.device);optimizer.zero_grad()
            if proto or args.multi_proto_diagnostics:
                with Features(model,args.multi_proto_feature) as capture:
                    ce,f=clientTargetProj._objective(client,model,x,y);features=capture.get()
                    # Detach NOW: training feature, before backward/step; never retain epoch graphs.
                    if collector is not None and epoch==client.local_epochs-1:
                        if new_protocol:
                            collection_tick=time.perf_counter();collector.add(features,y)
                            collection_seconds+=time.perf_counter()-collection_tick;collection_batches+=1
                        else:collector.add(features,y)
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
            if helper is not None:helper.before_clip()
            if cap is not None and cap>0:
                norm=torch.nn.utils.clip_grad_norm_(params,cap,error_if_nonfinite=True)
            elif args.multi_proto_diagnostics:
                norms=[p.grad.detach().norm(2).to(params[0].device) for p in params if p.grad is not None]
                norm=torch.stack(norms).norm(2)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite unclipped gradient norm')
            if args.multi_proto_diagnostics:
                value=float(norm);coef=float((cap/(norm+1e-6)).clamp(max=1.)) if cap is not None and cap>0 else 1.
                norm_sum+=value;coef_sum+=coef;triggered+=int(coef<1.)
            if helper is not None:helper.after_clip(norm if cap is not None and cap>0 or args.multi_proto_diagnostics else None,cap)
            optimizer.step()
            ce_sum+=float(ce.detach())*len(y);f_sum+=float(f.detach())*len(y)
            mse_sum+=(float(raw.detach()) if raw is not None else 0.)*len(y)
            weighted_sum+=(float(weighted.detach()) if weighted is not None else 0.)*len(y)
            total_sum+=float(loss.detach())*len(y);visits+=len(y);steps+=1;valid_visits+=nvalid
    if not visits:raise ValueError('Nonempty local data required')
    if str(client.device).startswith('cuda'):torch.cuda.synchronize(client.device)
    train_seconds=time.perf_counter()-started
    row=dict(client_id=client.id,model_id=column,local_epochs=client.local_epochs,train_batches=steps,
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
        train_loader_seed=loader_seed(args,client.id,client.download_round,column),
        prototype_collected_samples=int(collector.counts.sum()) if collector is not None else 0)
    if helper is not None:helper.finish(row)
    if new_protocol:
        if collector is not None:
            client.avg_proto_collection_cost=dict(statistics_seconds=collection_seconds,statistics_batches=collection_batches)
        row.update(prototype_collection_phase='pre_local_download' if pre_train(args) else 'last_local_epoch',
            prototype_aggregation=getattr(args,'multi_proto_aggregation','client_equal'),
            used_proto_library_sha256=receipt['sha256'] if can_use else None,
            used_proto_collection_loop_round=receipt['collection_loop_round'] if can_use else None,
            used_proto_collection_phase=receipt['collection_phase'] if can_use else None,
            used_proto_intended_use_loop_round=receipt['intended_use_loop_round'] if can_use else None,
            extra_collection_samples=getattr(client,'avg_proto_collection_cost',{}).get('samples',0),
            extra_collection_batches=getattr(client,'avg_proto_collection_cost',{}).get('batches',0),
            extra_collection_seconds=getattr(client,'avg_proto_collection_cost',{}).get('seconds',0.))
        row.update(last_epoch_statistics_seconds=collection_seconds,last_epoch_statistics_batches=collection_batches)
        row['prototype_download_bytes']=receipt['packet_bytes'] if receipt is not None else 0
    packet=None
    packet_started=time.perf_counter() if multicolumn or new_protocol else None
    if collector is not None:
        if new_protocol:
            packet=encode_packet('single_avg_local_v2',dict(sums=collector.sums,counts=collector.counts),
                local_metadata(client,'last_local_epoch'))
        else:
            packet=encode_packet('last_epoch_column_local' if multicolumn else 'global_last_epoch_local',dict(sums=collector.sums,counts=collector.counts),
                dict(client_id=client.id,source_completed_train_count=client.download_round+1,feature_position=args.multi_proto_feature,
                     **(dict(model_id=column) if multicolumn else {})))
    if multicolumn:
        row.update(prototype_statistics_bytes=len(packet) if packet else 0,
            prototype_statistics_encode_seconds=time.perf_counter()-packet_started if packet else 0.)
    elif new_protocol:
        row['prototype_statistics_upload_bytes']=len(packet) if packet else getattr(client,'avg_proto_collection_cost',{}).get('packet_bytes',0)
        row['prototype_statistics_encode_seconds']=time.perf_counter()-packet_started if packet else 0.
    if svd.enabled(args):
        tick=time.perf_counter();full=svd.recover_parameters(model);client.full_uploads[column]=full
        base=dict(client.download_servers[column].named_parameters())
        if full.keys()!=base.keys() or any(full[n].shape!=base[n].shape for n in base):
            raise ValueError('Recovered K1 upload parameter names/shapes differ from the full mother')
        from utils import multi_group_head as multi_heads
        if multi_heads.enabled(args):
            client.full_uploads[column]=multi_heads.finalize_upload(client,column,full,row)
        elif args.multi_svd_diagnostics:
            for key,left,right in [('svd_truncation',client.download_starts[column],base),
                ('local_training_recovered',full,client.download_starts[column]),('global_delta',full,base)]:
                squared=svd.distance_squared(left,right);row[key+'_squared_norm']=squared;row[key+'_norm']=squared**.5
        client.download_starts[column]=None
        row.update(capacity_level=client.capacity_level,rank_rate=client.rank_rate,recovered_parameter_count=sum(p.numel() for p in full.values()))
        torch.save(svd.client_payload(client,column),Path(client.save_folder_name)/f'Client_{client.id}_model_{column}.pt')
        row['upload_recovery_checkpoint_seconds']=time.perf_counter()-tick
    else:row.update(rank_rate=1.,recovered_parameter_count=sum(p.numel() for p in model.parameters()))
    if multicolumn:return row,packet
    client.avg_proto_local=packet
    client.train_stats=[row];client.upload_round=client.download_round
    client.train_time_cost['num_rounds']+=1;client.train_time_cost['total_cost']+=time.perf_counter()-started
    print(f'[AvgProtoLocal] round={client.download_round+1} client={client.id} CE={row["ce_loss"]:.8g} F={row["regularization_loss"]:.8g} '
        f'MSE={row["raw_mse"]:.8g} source={row["used_proto_source_completed_train_count"]} collected={row["prototype_collected_samples"]} steps={steps}')


def save_latest(server):
    if extended(server.args):
        for directory in (server.save_folder_name,server.final_model_dir()):
            for name,packet in [('latest_generated',server.avg_proto_packet),('last_used',getattr(server,'previous_avg_proto_packet',None))]:
                if packet:(Path(directory)/f'Global_class_sample_{name}.packet').write_bytes(packet)
        return
    if server.avg_proto_packet:
        for directory in (server.save_folder_name,server.final_model_dir()):
            (Path(directory)/'Global_last_epoch_prototypes.packet').write_bytes(server.avg_proto_packet)


def manifest_value(server):
    if extended(server.args):return extended_manifest(server)
    return dict(protocol=protocol(server.args),class_records=len(server.avg_proto_classes),
        round_records=len(server.avg_proto_rounds),extra_collection_forward_count=0,
        training_initialization_independent=True,grouping='none: all clients assigned model 0',
        actual_prototype_upload_bytes=sum(r['statistics_upload_bytes'] for r in server.avg_proto_rounds),
        actual_prototype_download_bytes=sum(r['actual_download_bytes'] for r in server.avg_proto_rounds))


def extended_protocol(args):
    before=pre_train(args)
    return dict(name='single_model_Avg_class_sample_v2',requested_mode=args.multi_proto_mode,
        prototype_enabled=uses_bank(args),aggregation=getattr(args,'multi_proto_aggregation','client_equal'),
        aggregation_formula='sum_client_class_sums / sum_actual_class_counts; model Avg weights unchanged',
        collection_phase='pre_local_download' if before else 'last_local_epoch',
        timing='this download -> collect ALL clients -> bind -> this SGD' if before else 'last epoch of round t -> frozen targets in t+1',
        collection='one complete own TRAIN eval/no_grad pass, after all downloads and before ANY SGD' if before else
            'last local epoch existing TRAIN forwards only; detach immediately; NO extra pass',
        model_training='svd_heterogeneous' if getattr(args,'multi_svd',False) else 'full_Hyper_CNN_512',
        coefficient=args.multi_proto_lambda,effective_coefficient=args.multi_proto_lambda if uses_bank(args) else 0.,
        start_completed_train_count=args.multi_proto_start_round,feature_position=args.multi_proto_feature,
        statistics_dtype='float64 class sum / int64 actual class count',normalization='none; raw signed features and raw class means',
        missing='no class samples invalid; an observed zero mean is VALID',
        loss='CE + configured original F + lambda*MSE over ENTIRE batch x feature dimensions; missing targets have zero error/gradient',
        is_regular=args.is_regular,regular_lamda=args.regular_lamda,local_grad_clip=args.local_grad_clip,
        collection_batch_size=args.multi_proto_batch_size if before else None,diagnostics=bool(args.multi_proto_diagnostics),
        sample_order='UNCHANGED single-avg-proto-v1 private seed/client/round training loader; separate collection generator',
        classification='original fc3; prototypes are training-only',
        feature_alignment='actual client SVD encoders; heterogeneous coordinates only approximately aligned, not a single shared encoder')


def class_sample_means(packets,classes,dimension,source):
    """Strict sum/count merge. A client's TOTAL data size is never a weight."""
    import torch
    if not packets or any(not isinstance(p,dict) or not {'client_id','source_completed_train_count','sums','counts'}<=p.keys() for p in packets):
        raise ValueError('Incomplete prototype statistic packets')
    ids=[p['client_id'] for p in packets]
    if any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):raise ValueError('Invalid/duplicate prototype contributor')
    total=torch.zeros(classes,dimension,dtype=torch.float64);counts=torch.zeros(classes,dtype=torch.int64)
    owners=[[] for _ in range(classes)];by_class=[[] for _ in range(classes)]
    for p in sorted(packets,key=lambda p:p['client_id']):
        s,n=p['sums'],p['counts']
        if p['source_completed_train_count']!=source:raise ValueError('Wrong statistic source round')
        if (not isinstance(s,torch.Tensor) or not isinstance(n,torch.Tensor) or s.shape!=total.shape or n.shape!=counts.shape
            or s.dtype!=torch.float64 or n.dtype!=torch.int64 or s.device.type!='cpu' or n.device.type!='cpu'
            or s.requires_grad or n.requires_grad or (n<0).any() or not torch.isfinite(s).all()
            or (s[n==0]!=0).any()):raise ValueError('Invalid raw class sum/count packet')
        if (n>torch.iinfo(torch.int64).max-counts).any():raise ValueError('Class count overflow')
        total.add_(s);counts.add_(n)
        if not torch.isfinite(total).all():raise FloatingPointError('Nonfinite merged raw class sum')
        for c in (n>0).nonzero().flatten().tolist():
            owners[c].append(p['client_id']);by_class[c].append(int(n[c]))
    valid=counts>0;total[valid]/=counts[valid,None]
    contributors=torch.tensor([len(x) for x in owners],dtype=torch.int64)
    return dict(prototypes=total,valid=valid,contributors=contributors,samples=counts),owners,by_class


def _phase(args):return 'pre_local_download' if pre_train(args) else 'last_local_epoch'


def local_metadata(client,phase):
    from utils.head_snapshot import tensor_fingerprint
    loop=client.download_round;context=client.avg_proto_download_context
    return dict(protocol_version=2,client_id=client.id,model_id=0,collection_loop_round=loop,collection_phase=phase,
        source_completed_train_count=loop if phase=='pre_local_download' else loop+1,
        intended_use_loop_round=loop if phase=='pre_local_download' else loop+1,
        intended_use_completed_train_count=loop+1 if phase=='pre_local_download' else loop+2,
        source_server_completed_aggregations=loop,source_server_sha256=context['server_sha256'],
        source_download_model_sha256=context['download_sha256'],data_fingerprint=context['data_fingerprint'],
        source_post_local_model_sha256=tensor_fingerprint(client.model_set[0].state_dict()) if phase=='last_local_epoch' else None,
        encoder_evolves_within_collection=phase=='last_local_epoch',
        initial_sha256=context['initial_sha256'],feature_position=client.args.multi_proto_feature,
        aggregation='class_sample',statistics_representation='raw_feature_class_sum',prototype_representation='raw_class_mean')


def _validate_library(values,meta,args,loop):
    import torch
    phase=_phase(args);collection=loop if pre_train(args) else loop-1
    expected=dict(protocol_version=2,model_id=0,aggregation='class_sample',collection_phase=phase,collection_loop_round=collection,
        intended_use_loop_round=loop,intended_use_completed_train_count=loop+1,
        source_completed_train_count=loop,source_server_completed_aggregations=collection,
        feature_position=args.multi_proto_feature,statistics_representation='raw_feature_class_sum',prototype_representation='raw_class_mean')
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Prototype phase/source/use round or aggregation mismatch')
    downloads=meta.get('source_download_model_sha256_by_client',{})
    if (not isinstance(downloads,dict) or set(downloads)!={str(i) for i in range(args.num_clients)}
        or any(not isinstance(v,str) or not v for v in downloads.values())
        or any(not isinstance(meta.get(k),str) or not meta[k] for k in ('source_server_sha256','data_fingerprint','initial_sha256'))):
        raise ValueError('Missing prototype source model/data fingerprints or client statistics')
    if not {'prototypes','valid','contributors','samples'}<=values.keys():raise ValueError('Incomplete prototype library')
    p,v,c,n=(values[k] for k in ('prototypes','valid','contributors','samples'))
    if (p.ndim!=2 or p.shape[0]!=args.num_classes or p.dtype!=torch.float64 or not torch.isfinite(p).all()
        or v.dtype!=torch.bool or v.shape!=(len(p),) or c.dtype!=torch.int64 or n.dtype!=torch.int64
        or c.shape!=v.shape or n.shape!=v.shape or (c<0).any() or (n<0).any() or (c>n).any()
        or not torch.equal(v,n>0) or not torch.equal(v,c>0) or (p[~v]!=0).any()):raise ValueError('Invalid prototype library tensors')


def validate_training_receipt(client,bank,receipt):
    if not uses_bank(client.args):
        if bank is not None or receipt is not None:raise ValueError('Disabled prototype protocol received a bank')
        return
    loop=client.download_round;expected=loop>0 and (not pre_train(client.args) or loop+1>=client.args.multi_proto_start_round)
    if not expected:
        if bank is not None or receipt is not None:raise ValueError('Prototype bank supplied before its collection/use stage')
        return
    if bank is None or receipt is None:raise RuntimeError('Missing required prototype collection barrier/bank')
    _validate_library(bank,receipt,client.args,loop)
    if bank['prototypes'].shape[1]!=client.model_set[0].fc3.in_features:raise ValueError('Wrong prototype feature dimension')
    context=client.avg_proto_download_context
    if receipt.get('data_fingerprint')!=context['data_fingerprint'] or receipt.get('initial_sha256')!=context['initial_sha256']:
        raise ValueError('Wrong prototype data/initialization fingerprint')
    if pre_train(client.args):
        if (receipt.get('source_server_sha256')!=context['server_sha256'] or
            receipt.get('source_download_model_sha256_by_client',{}).get(str(client.id))!=context['download_sha256']):
            raise ValueError('Pre-training prototypes belong to another actual download')


def collect_pre_train(client):
    """Read the actual downloaded prediction model, with a separate loader/RNG."""
    from contextlib import contextmanager
    import random
    import numpy as np
    import torch
    from common.data_loading import make_data_loader
    from utils.multi_group_proto import Features,encode_packet
    model=client.model_set[0]
    @contextmanager
    def preserve():
        cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state()
        devices={p.device.index for p in list(model.parameters())+list(model.buffers()) if p.device.type=='cuda'}
        cuda={d:torch.cuda.get_rng_state(d) for d in sorted(devices)}
        state={n:p.detach().clone() for n,p in model.state_dict().items()};modes=[(m,m.training) for m in model.modules()]
        try:yield
        finally:
            model.load_state_dict(state)
            for m,mode in modes:m.training=mode
            random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
            for device,value in cuda.items():torch.cuda.set_rng_state(value,device)
    tick=time.perf_counter();batches=0
    with preserve(),torch.no_grad():
        original=client.load_train_data()  # Dataset only: NEVER iterate the training sampler.
        generator=torch.Generator().manual_seed(loader_seed(client.args,client.id,client.download_round)^0x505245)
        loader=make_data_loader(original.dataset,client.args.multi_proto_batch_size,shuffle=False,drop_last=False,num_workers=0,
            loader_cls=type(original),generator=generator,collate_fn=original.collate_fn)
        stats=LastEpochStats(model.fc3.out_features,model.fc3.in_features);model.eval()
        for x,y in loader:
            x,y=x.to(client.device),y.to(client.device)
            with Features(model,client.args.multi_proto_feature) as features:model(x);stats.add(features.get(),y)
            batches+=1
        if str(client.device).startswith('cuda'):torch.cuda.synchronize(client.device)
        if len(loader.dataset)!=client.train_samples or int(stats.counts.sum())!=client.train_samples:
            raise ValueError('Collection must cover the complete own TRAIN dataset exactly once')
    packet=encode_packet('single_avg_local_v2',dict(sums=stats.sums,counts=stats.counts),local_metadata(client,'pre_local_download'))
    return packet,dict(samples=int(stats.counts.sum()),batches=batches,seconds=time.perf_counter()-tick,packet_bytes=len(packet))


def prepare_extended_round(server):
    from utils.head_snapshot import tensor_fingerprint
    from utils.cnn512_reporting import model_digest
    from utils.multi_group_proto import decode_packet,packet_sha
    loop=server.loop_round
    if {c.id for c in server.clients}!=set(range(server.num_clients)) or len(server.clients)!=server.num_clients:
        raise ValueError('Incomplete/duplicate all-client collection population')
    mother=model_digest(server.global_model_set[0])
    for client in server.clients:
        if client.download_round!=loop or client.upload_round is not None or client.download_hashes[0]!=mother:
            raise RuntimeError('ALL current K1 downloads must complete before prototype collection')
        client.avg_proto_bank=None;client.avg_proto_receipt=None;client.avg_proto_local=None;client.avg_proto_collection_cost={}
        client.avg_proto_download_context=dict(server_sha256=mother,download_sha256=tensor_fingerprint(client.model_set[0].state_dict()),
            initial_sha256=server.initial_hash,data_fingerprint=server.context['data_fingerprint'])
    if pre_train(server.args):
        server.avg_proto_packet=None  # NEVER reuse the preceding round's pre-training library.
        if loop>0 and loop+1>=server.args.multi_proto_start_round:
            # Hard barrier: no client.train call exists inside collection/merge/bind.
            for client in server.clients:
                client.avg_proto_local,client.avg_proto_collection_cost=collect_pre_train(client)
            merge_extended_round(server)
    packet=server.avg_proto_packet;server.previous_avg_proto_packet=packet
    if packet is not None:
        values,meta=decode_packet(packet,'single_avg_library_v2');_validate_library(values,meta,server.args,loop)
        for client in server.clients:
            client.avg_proto_bank={k:v.detach().clone() for k,v in values.items()}
            client.avg_proto_receipt=dict(meta,sha256=packet_sha(packet),packet_bytes=len(packet))
            validate_training_receipt(client,client.avg_proto_bank,client.avg_proto_receipt)


def merge_extended_round(server):
    import torch
    from utils.multi_group_proto import decode_packet,encode_packet,packet_sha
    loop=server.loop_round;before=pre_train(server.args);tick=time.perf_counter();packets=[];byte_count=0
    for client in server.clients:
        if (before and client.upload_round is not None) or (not before and client.upload_round!=loop):raise RuntimeError('Wrong collection/SGD phase at merge')
        if client.avg_proto_local is None:raise ValueError('Missing full-client prototype statistics')
        values,meta=decode_packet(client.avg_proto_local,'single_avg_local_v2');expected=local_metadata(client,_phase(server.args))
        if meta!=expected or set(values)!=set(('sums','counts')):raise ValueError('Wrong/incomplete prototype phase, download, data or metadata')
        if values['counts'].dtype!=torch.int64 or int(values['counts'].sum())!=client.train_samples:
            raise ValueError('Prototype counts must cover exactly ONE full TRAIN traversal')
        packets.append(dict(values,**meta));byte_count+=len(client.avg_proto_local)
    source=loop if before else loop+1;d=server.global_model_set[0].fc3.in_features
    values,owners,counts=class_sample_means(packets,server.num_classes,d,source)
    meta={k:v for k,v in packets[0].items() if k not in ('sums','counts','client_id','source_download_model_sha256','source_post_local_model_sha256')}
    meta['source_download_model_sha256_by_client']={str(p['client_id']):p['source_download_model_sha256'] for p in packets}
    if not before:meta['source_post_local_model_sha256_by_client']={str(p['client_id']):p['source_post_local_model_sha256'] for p in packets}
    server.avg_proto_packet=encode_packet('single_avg_library_v2',values,meta);sha=packet_sha(server.avg_proto_packet)
    for c in range(server.num_classes):
        n=int(values['samples'][c]);valid=bool(values['valid'][c]);norm=float(values['prototypes'][c].norm()) if valid else None
        server.avg_proto_classes.append(dict(loop_round=loop,completed_train_count=loop if before else loop+1,
            collection_loop_round=loop,collection_phase=_phase(server.args),aggregation='class_sample',class_id=c,
            contributor_ids=owners[c],contributor_count=len(owners[c]),contributor_class_samples=counts[c],
            contributor_weights=[x/n for x in counts[c]] if n else [],class_samples=n,valid=valid,prototype_l2_norm=norm,
            invalid_reason=None if valid else 'no_samples',valid_zero_mean=valid and norm==0.,
            source_completed_train_count=source,source_server_completed_aggregations=loop,
            intended_use_loop_round=meta['intended_use_loop_round'],intended_use_completed_train_count=meta['intended_use_completed_train_count'],
            library_sha256=sha,source_server_sha256=meta['source_server_sha256'],data_fingerprint=meta['data_fingerprint'],feature_position=meta['feature_position']))
    cost=[getattr(c,'avg_proto_collection_cost',{}) for c in server.clients]
    server.avg_proto_rounds.append(dict(loop_round=loop,collection_phase=_phase(server.args),collection_loop_round=loop,
        completed_train_count=loop if before else loop+1,source_completed_train_count=source,source_server_completed_aggregations=loop,
        intended_use_loop_round=meta['intended_use_loop_round'],intended_use_completed_train_count=meta['intended_use_completed_train_count'],
        aggregation='class_sample',collected_training_samples=0 if before else sum(int(p['counts'].sum()) for p in packets),
        extra_collection_samples=sum(c.get('samples',0) for c in cost),extra_collection_forwards=sum(c.get('batches',0) for c in cost),
        last_epoch_statistics_batches=sum(c.get('statistics_batches',0) for c in cost),
        collection_seconds=sum(c.get('seconds',0.)+c.get('statistics_seconds',0.) for c in cost),statistics_upload_bytes=byte_count,
        actual_download_bytes=len(server.clients)*len(server.avg_proto_packet) if before else len(server.clients)*len(server.previous_avg_proto_packet or b''),
        library_packet_bytes=len(server.avg_proto_packet),merge_seconds=time.perf_counter()-tick,valid_classes=int(values['valid'].sum()),library_sha256=sha))
    for client in server.clients:client.avg_proto_local=None
    print(f'[SingleAvgPrototype] collection_loop={loop} phase={meta["collection_phase"]} '
          f'use_loop={meta["intended_use_loop_round"]} aggregation=class_sample valid={int(values["valid"].sum())} '
          f'extra_samples={server.avg_proto_rounds[-1]["extra_collection_samples"]} '
          f'upload_bytes={byte_count} library_sha256={sha}')


def finish_extended_round(server):
    import json
    if not pre_train(server.args):merge_extended_round(server)
    # Append after SGD/aggregation, but retain the TRUE pre/last collection stage.
    rows=[r for r in server.avg_proto_rounds if r['loop_round']==server.loop_round]
    classes=[r for r in server.avg_proto_classes if r['loop_round']==server.loop_round]
    with (Path(server.report_dir())/'avg_proto_protocol.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(loop_round=server.loop_round,collection=rows,classes=classes),allow_nan=False)+'\n')


def checkpoint_value(client):
    receipt=getattr(client,'avg_proto_receipt',None)
    return dict(protocol=protocol(client.args),training_loop_round=client.download_round,
        downloaded_model=getattr(client,'avg_proto_download_context',None),actual_used_library=receipt,
        mse_enabled=uses_bank(client.args) and receipt is not None and client.download_round+1>=client.args.multi_proto_start_round)


def extended_manifest(server):
    from utils.multi_group_proto import decode_packet,packet_sha
    def description(packet):
        if packet is None:return None
        _,meta=decode_packet(packet,'single_avg_library_v2');return dict(meta,sha256=packet_sha(packet))
    return dict(protocol=protocol(server.args),latest_generated_library=description(server.avg_proto_packet),
        last_actually_used_library=description(getattr(server,'previous_avg_proto_packet',None)),
        class_records=len(server.avg_proto_classes),round_records=len(server.avg_proto_rounds),
        actual_prototype_upload_bytes=sum(r['statistics_upload_bytes'] for r in server.avg_proto_rounds),
        actual_prototype_download_bytes=sum(r['actual_download_bytes'] for r in server.avg_proto_rounds),
        extra_collection_forward_count=sum(r['extra_collection_forwards'] for r in server.avg_proto_rounds),
        extra_collection_sample_visits=sum(r['extra_collection_samples'] for r in server.avg_proto_rounds),
        collection_seconds=sum(r['collection_seconds'] for r in server.avg_proto_rounds),
        grouping='none: all clients assigned model 0',training_initialization_independent=True)
