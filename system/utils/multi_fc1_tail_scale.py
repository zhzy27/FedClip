"""Opt-in K1 full-Avg download gauge; no SGD, server SVD or aggregation change."""
import json
import math
from pathlib import Path
import time


def add_options(parser):
    parser.add_argument('--multi_fc1_tail_scale_strength',type=float,default=0.,
        help='0: exact old download; positive c: floor fc1 tail U scales at c*sqrt(sigma_min_rank)')


def enabled(args):return getattr(args,'multi_fc1_tail_scale_strength',0.)>0


def validate(args):
    value=getattr(args,'multi_fc1_tail_scale_strength',0.)
    if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('multi_fc1_tail_scale_strength must be finite and nonnegative')
    if not enabled(args):return
    if (getattr(args,'algorithm','')!='FedMultiGroupProj' or not getattr(args,'multi_avg_proto_ablation',False)
        or getattr(args,'model_family','')!='Decom_CNN-5-512'
        or not getattr(args,'multi_svd',False) or args.num_server_models!=1 or args.multi_group_mode!='group_avg_only'
        or getattr(args,'multi_svd_aggregation','full_avg')!='full_avg' or args.join_ratio!=1.
        or args.target_head_mode!='shared' or args.target_local_schedule!='joint'
        or args.multi_proto_mode!='none' or args.multi_proto_lambda!=0
        or getattr(args,'multi_last_epoch_protocol',False) or getattr(args,'multi_head_mode','shared')!='shared'
        or getattr(args,'multi_projection_scope','full')!='full' or getattr(args,'multi_helper_head_update','train')!='train'):
        raise ValueError('fc1 tail scaling v1 requires K1 all-client multi_avg_proto_ablation + multi_svd + full_avg/group_avg_only, no prototypes, shared/joint')
    if (getattr(args,'target_prox_mu',0) or getattr(args,'target_head_l2_lambda',0) or getattr(args,'target_z2_lambda',0)
        or getattr(args,'target_c0_replay_mode','none')!='none' or getattr(args,'target_curriculum_mode','none')!='none'
        or getattr(args,'multi_kd_mode','none')!='none'):
        raise ValueError('fc1 tail scaling does not mix Prox/head L2/Z2/replay/curriculum/KD')
    if any(rate>=1 for rate in args.multi_svd_effective_ratios):raise ValueError('fc1 tail scaling needs actual low-rank factors; dense clients are unsupported')


def policy(args):
    return dict(configured_strength=getattr(args,'multi_fc1_tail_scale_strength',0.),enabled=enabled(args),
        stage='each actual fc1 download SVD, before SGD optimizer construction',
        minimum_rank='original FactorizedLinear rank rule, minimum ACTUALLY assigned capacity, per fc1 shape',
        scope='fc1 slots [minimum_rank,client_rank) only; prefix, bias and other layers unchanged',
        rule='a=max(sqrt(sigma_j),c*sqrt(sigma_min)); U=a*P, V=(sigma_j/a)*Qh, SAME download SVD',
        zero_rule='same-SVD unit vectors; sigma=0 and a=0 -> both scales zero',
        server='unchanged complete-parameter sample-weighted full_avg; no gauge or SVD on server aggregation',
        off_path='strength 0 calls original decomposition directly; no callbacks, collectors or matrix checks',
        inference='ordinary post-local model; no truncation at inference; no added epochs or SGD steps')


def decompose(model,rate,args,client_id,column,loop):
    from flcore.trainmodel.models import factorized_linear_rank
    from utils.fc1_tail_factors import apply_from_svd
    tick=time.perf_counter();rates=args.multi_svd_effective_ratios
    minimum_rate=min(rates[level] for level in args.multi_svd_effective_assignment)
    minimum=factorized_linear_rank(model.fc1.in_features,model.fc1.out_features,minimum_rate)
    rank=factorized_linear_rank(model.fc1.in_features,model.fc1.out_features,rate)
    strength=args.multi_fc1_tail_scale_strength
    record=dict(loop_round=loop,completed_train_count=loop+1,client_id=client_id,model_id=column,
        capacity_level=args.multi_svd_effective_assignment[client_id],rank_rate=rate,
        configured_strength=strength,effective_strength=strength if rank>minimum else 0.,
        actual_fc1_rank=rank,minimum_actual_fc1_rank=minimum,
        tail_start_inclusive=minimum,tail_end_exclusive=rank,tail_slot_count=max(0,rank-minimum),
        model_download_decomposition_count=1,fc1_actual_svd_count=1,extra_svd_count=0,
        fc1_svd_callback_count=0,fc1_trainable_parameter_count=None,
        skipped_reason='minimum_actual_capacity_no_tail' if rank<=minimum else None,
        floor_singular_value=None,floor_factor_scale=None,zero_tail_singular_values=None,zero_floor=None)
    def receive(layer,left,singular,right_h):
        if layer.rank!=rank:raise RuntimeError('Actual fc1 rank disagrees with original constructor rule')
        record['fc1_svd_callback_count']+=1
        started=time.perf_counter()
        record.update(apply_from_svd(layer,left,singular,right_h,minimum,strength,verify_product=bool(args.multi_svd_diagnostics)))
        record['gauge_seconds']=time.perf_counter()-started
    if rank>minimum:model.decom_larger_model(rate,fc1_svd_observer=receive)
    else:model.decom_larger_model(rate)  # Minimum capacity: exactly the unmodified balanced download.
    if (model.fc1.rank!=rank or record['fc1_svd_callback_count']!=int(rank>minimum)):
        raise RuntimeError('Need exactly one same-SVD callback for each adjusted fc1')
    record.update(fc1_trainable_parameter_count=sum(p.numel() for p in model.fc1.parameters()),
        download_decomposition_and_gauge_seconds=time.perf_counter()-tick)
    return record


def bind_receipts(client,records):
    from utils.cnn512_reporting import model_digest
    for row in records:
        column=row['model_id'];row['source_server_sha256']=client.download_hashes[column]
        row['download_model_sha256']=model_digest(client.model_set[column])
        row['local_trainable_parameter_count']=sum(p.numel() for p in client.model_set[column].parameters() if p.requires_grad)
    client.fc1_tail_scale_latest=records
    if not hasattr(client,'fc1_tail_scale_records'):client.fc1_tail_scale_records=[]
    client.fc1_tail_scale_records.extend(records)


def rows(server):return [r for c in server.clients for r in getattr(c,'fc1_tail_scale_records',[])]


def manifest(server):
    records=rows(server)
    return dict(policy=policy(server.args),download_records=len(records),
        adjusted_download_count=sum(r['effective_strength']>0 for r in records),
        skipped_minimum_capacity_count=sum(r['effective_strength']==0 for r in records),
        gauge_observation_seconds=sum(r.get('gauge_seconds',0.) for r in records),
        actual_minimum_ranks=sorted({r['minimum_actual_fc1_rank'] for r in records}))


def persist(server):
    if not enabled(server.args):return
    records=[r for c in server.clients for r in getattr(c,'fc1_tail_scale_latest',[]) if r['loop_round']==server.loop_round]
    if len(records)!=server.num_clients:raise RuntimeError('Incomplete all-client tail-scale download receipts')
    with (Path(server.report_dir())/'fc1_tail_scale.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(loop_round=server.loop_round,downloads=records),allow_nan=False)+'\n')
    ranges=sorted({(r['rank_rate'],r['tail_start_inclusive'],r['tail_end_exclusive']) for r in records})
    print(f'[FC1TailScale] completed={server.loop_round+1} c={server.args.multi_fc1_tail_scale_strength} '
          f'adjusted={sum(r["effective_strength"]>0 for r in records)}/{len(records)} '
          f'(rate,start,end)={ranges}; extra_SVD=0; before_SGD')
