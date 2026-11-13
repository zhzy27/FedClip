"""Opt-in K1 CNN rank-position aggregation; no LoRA or server truncation.

Only post-local ORIGINAL factors are read. Diagnostic singular values are
computed on private float64 CPU matrices and never feed aggregation/downloads.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time

LAYERS=('conv2','fc1','fc2')


def add_options(parser):
    parser.add_argument('--multi_svd_aggregation',choices=['full_avg','rank_partition'],default='full_avg')
    parser.add_argument('--multi_rank_spectrum',type=int,choices=[0,1],default=None,
        help='Read-only sampled spectrum: default ON for rank_partition, OFF for old full_avg')
    parser.add_argument('--multi_rank_spectrum_rounds',default=None,
        help='Completed-training numbers, comma-separated; final round always included; default 1,10,50,final (within budget)')
    parser.add_argument('--multi_rank_spectrum_client_id',type=int,default=0)


def enabled(args):return getattr(args,'multi_svd_aggregation','full_avg')=='rank_partition'


def spectrum_enabled(args):
    value=getattr(args,'multi_rank_spectrum',None)
    return enabled(args) if value is None else bool(value)


def reporting(args):return enabled(args) or spectrum_enabled(args)


def validate(args):
    for key,value in dict(multi_svd_aggregation='full_avg',multi_rank_spectrum=None,multi_rank_spectrum_rounds=None,
                          multi_rank_spectrum_client_id=0).items():
        if not hasattr(args,key):setattr(args,key,value)
    if getattr(args,'multi_svd_aggregation','full_avg') not in ('full_avg','rank_partition'):raise ValueError('Invalid multi_svd_aggregation')
    if getattr(args,'multi_rank_spectrum',None) not in (None,0,1):raise ValueError('multi_rank_spectrum must be 0/1')
    requested=getattr(args,'multi_rank_spectrum_rounds',None);cid=getattr(args,'multi_rank_spectrum_client_id',0)
    if not spectrum_enabled(args) and (requested is not None or cid!=0):raise ValueError('Spectrum rounds/client need enabled spectrum diagnostics')
    if not reporting(args):return
    if (getattr(args,'algorithm','')!='FedMultiGroupProj' or not getattr(args,'multi_avg_proto_ablation',False)
        or not getattr(args,'multi_svd',False) or args.num_server_models!=1 or args.multi_group_mode!='group_avg_only'
        or args.target_head_mode!='shared' or args.target_local_schedule!='joint' or args.join_ratio!=1.
        or getattr(args,'multi_last_epoch_protocol',False) or getattr(args,'multi_head_mode','shared')!='shared'):
        raise ValueError('Rank partition/spectrum v1 requires full-participation K1 multi_avg_proto_ablation + multi_svd + group_avg_only, shared/joint')
    if args.multi_proto_mode not in ('none','global_last_epoch'):
        raise ValueError('Rank partition/spectrum v1 supports none or last-epoch prototypes; no global_pre_train')
    if args.multi_proto_lambda>0 and (getattr(args,'multi_proto_aggregation','client_equal')!='class_sample'
        or args.multi_proto_target_type!='raw_mean' or args.multi_proto_loss!='mse' or args.multi_proto_feature!='base_output'):
        raise ValueError('Rank experiment prototypes require explicit last_epoch/class_sample/raw_mean/base_output MSE')
    if any(r>=1 for r in args.multi_svd_effective_ratios):
        raise ValueError('Rank partition/spectrum v1 requires real factors in conv2/fc1/fc2; dense rank_rate=1 clients unsupported')
    if type(cid) is not int or not 0<=cid<args.num_clients:raise ValueError('Spectrum client ID out of range')
    total=args.global_rounds+1
    if requested is None:rounds=[r for r in (1,10,50) if r<=total]
    else:
        try:rounds=[int(x.strip()) for x in requested.split(',')]
        except (ValueError,AttributeError):raise ValueError('Spectrum rounds must be comma-separated completed-training integers')
        if not rounds or len(set(rounds))!=len(rounds) or any(not 1<=r<=total for r in rounds):raise ValueError('Invalid/duplicate/out-of-budget spectrum rounds')
    if spectrum_enabled(args):args.multi_rank_effective_spectrum_rounds=sorted(set(rounds+[total]))


def protocol(args):
    return dict(method=getattr(args,'multi_svd_aggregation','full_avg'),
        factorized_layers=list(LAYERS),partition_boundaries='sorted distinct ACTUAL integer layer.rank values, independently per layer',
        factor_identity='original download-position U columns / V rows AFTER ordinary local training; never re-SVD or reorder',
        factor_weights='n_i / sum(n_j for rank_j >= partition_end); original TRAIN shard sample counts',
        other_parameters='original all-client sample-weighted full-matrix Avg; includes conv1, fc3 and every bias ONCE',
        convolution_layout='A[out*k,in*k] -> reshape(out,k,in,k).permute(0,2,1,3), original Recover_COV',
        server='complete current CNN; NO extra SVD or truncation of aggregation result; next download uses original SVD',
        equal_rank_fallback='reuse original Avg arithmetic/result for each uniform-rank layer (including one-client case)',
        spectrum_enabled=spectrum_enabled(args),spectrum_completed_rounds=getattr(args,'multi_rank_effective_spectrum_rounds',[]),
        spectrum_client_id=getattr(args,'multi_rank_spectrum_client_id',0),
        spectrum='actual matrix singular-value-squared energy split at minimum ACTUAL client layer rank; float64 private CPU copy',
        block_norm='factor-POSITION partition Frobenius norm, NOT spectral energy; partitions may be nonorthogonal',
        reference='raFLoRA, arXiv:2602.13486v1, rank-partitioned contributor normalization only; CNN adaptation, not full LoRA reproduction')


def factors(layer):
    import torch
    from flcore.trainmodel.models import FactorizedConv,FactorizedLinear
    if isinstance(layer,FactorizedConv):u,v=layer.conv_u,layer.conv_v
    elif isinstance(layer,FactorizedLinear):u,v=layer.weight_u,layer.weight_v
    else:raise ValueError('Need actual trained FactorizedConv/FactorizedLinear; never reconstruct factors from full uploads')
    rank=layer.rank
    if (type(rank) is not int or rank<=0 or u.ndim!=2 or v.ndim!=2 or u.shape[1]!=rank or v.shape[0]!=rank
        or rank>min(u.shape[0],v.shape[1]) or not torch.isfinite(u).all() or not torch.isfinite(v).all()):
        raise ValueError('Invalid actual layer rank/factor shapes or nonfinite factors')
    return dict(U=u.detach(),V=v.detach(),rank=rank)


def partition_product(entries,samples):
    """Pure ordered sum of normalized rank blocks. Never writes factor inputs."""
    import torch
    ids=[item['client_id'] for item in entries]
    if not entries or len(set(ids))!=len(ids) or set(ids)!=set(samples):raise ValueError('Incomplete/duplicate factor population')
    if any(type(n) is not int or n<=0 for n in samples.values()):raise ValueError('Require positive original TRAIN sample counts')
    first=entries[0];shape=(first['U'].shape[0],first['V'].shape[1]);dtype=first['U'].dtype;device=first['U'].device
    for item in entries:
        u,v,r=item['U'],item['V'],item['rank']
        if (type(r) is not int or r<=0 or u.ndim!=2 or v.ndim!=2 or u.shape!=(shape[0],r) or v.shape!=(r,shape[1])
            or r>min(shape) or u.dtype!=dtype or v.dtype!=dtype or u.device!=device or v.device!=device
            or not torch.isfinite(u).all() or not torch.isfinite(v).all()):raise ValueError('Invalid rank-position matrix factors')
    ranks=sorted({item['rank'] for item in entries});total=torch.zeros(shape,dtype=dtype,device=device);rows=[];left=0
    with torch.no_grad():
        for right in ranks:
            tick=time.perf_counter();owners=[item for item in entries if item['rank']>=right]
            mass=sum(samples[item['client_id']] for item in owners);weights=[samples[item['client_id']]/mass for item in owners]
            block=torch.zeros_like(total)
            for item,w in zip(owners,weights):block.add_((item['U'][:,left:right]@item['V'][left:right,:])*w)
            if not torch.isfinite(block).all():raise FloatingPointError('Nonfinite rank partition product')
            total.add_(block)
            rows.append(dict(partition_start=left,partition_end=right,contributor_ids=[p['client_id'] for p in owners],
                contributor_count=len(owners),contributor_train_samples=[samples[p['client_id']] for p in owners],
                contributor_train_samples_total=mass,normalized_weights=weights,weight_sum=math.fsum(weights),
                factor_position_block_frobenius_norm=float(block.double().norm()),partition_seconds=time.perf_counter()-tick))
            left=right
    if not torch.isfinite(total).all():raise FloatingPointError('Nonfinite partition aggregate')
    return total,rows


def matrix_to_weight(matrix,layer):
    from flcore.trainmodel.models import FactorizedConv
    if isinstance(layer,FactorizedConv):
        k=layer.kernel_size
        return matrix.reshape(layer.out_channels,k,layer.in_channels,k).permute(0,2,1,3)
    return matrix


def actual_matrix(layer):
    import torch
    from flcore.trainmodel.models import FactorizedConv,FactorizedLinear
    if isinstance(layer,(FactorizedConv,FactorizedLinear)):
        values=factors(layer);return values['U']@values['V']
    if isinstance(layer,torch.nn.Conv2d):
        if layer.kernel_size[0]!=layer.kernel_size[1]:raise ValueError('Need original square convolution layout')
        k=layer.kernel_size[0];return layer.weight.detach().permute(0,2,1,3).reshape(layer.out_channels*k,layer.in_channels*k)
    if isinstance(layer,torch.nn.Linear):return layer.weight.detach()
    raise ValueError('Unsupported spectrum layer')


@contextmanager
def protected_rng(models):
    import random
    import numpy as np
    import torch
    cpu=torch.get_rng_state();python=random.getstate();numpy=np.random.get_state()
    devices={p.device.index for m in models for p in list(m.parameters())+list(m.buffers()) if p.device.type=='cuda'}
    cuda={d:torch.cuda.get_rng_state(d) for d in sorted(devices)}
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        for d,value in cuda.items():torch.cuda.set_rng_state(value,d)


def initialize(server):
    if reporting(server.args):server.rank_partition_records=[];server.rank_spectrum_records=[];server.rank_aggregation_records=[]


def aggregate(server):
    import torch
    from utils.group_projection import aggregate_group_updates
    if server.K!=1 or server.groups!=[list(range(server.num_clients))]:raise RuntimeError('Require one ALL-client group')
    order=[c.id for c in server.selected_clients]
    if sorted(order)!=list(range(server.num_clients)) or any(c.upload_round!=server.loop_round for c in server.clients):
        raise RuntimeError('Require complete fresh post-local factors/uploads')
    start=time.perf_counter();mother=server.global_model_set[0];params=dict(mother.named_parameters())
    weights={c.id:c.train_samples for c in server.clients};mass=sum(weights.values());fractions={cid:n/mass for cid,n in weights.items()}
    def uploads():
        for cid in order:yield cid,fractions[cid],server.clients[cid].full_upload_parameters(0,server.device)
    with torch.no_grad(),protected_rng([mother]+[c.model_set[0] for c in server.clients]):
        # Keep the legacy baseline arithmetic for biases/other weights and exact uniform-rank fallback.
        result,metrics,rows,matrices=aggregate_group_updates(params,uploads,fractions,list(range(server.num_clients)),'group_avg_only')
        legacy_weights={row['client_id']:row['aggregation_weight'] for row in rows}
        names={id(p):n for n,p in mother.named_parameters()};replaced=[]
        for name in LAYERS:
            target=getattr(mother,name);key=names.get(id(target.weight))
            if key is None or key in replaced:raise ValueError('Missing/aliased decomposed server weight')
            entries=[dict(client_id=cid,**factors(getattr(server.clients[cid].model_set[0],name))) for cid in order]
            uniform=len({p['rank'] for p in entries})==1
            if uniform:
                # Avoid even a different split/matmul/add ordering on the exact legacy path.
                n=sum(weights.values());block_rows=[dict(partition_start=0,partition_end=entries[0]['rank'],contributor_ids=order,
                    contributor_count=len(order),contributor_train_samples=[weights[i] for i in order],contributor_train_samples_total=n,
                    normalized_weights=[legacy_weights[i] for i in order],weight_sum=math.fsum(legacy_weights[i] for i in order),
                    factor_position_block_frobenius_norm=float(result[key].double().norm()),partition_seconds=0.)]
            else:
                product,block_rows=partition_product(entries,weights)
                weight=matrix_to_weight(product,getattr(server.clients[order[0]].model_set[0],name)).to(result[key])
                if weight.shape!=result[key].shape:raise ValueError('Original convolution recovery layout/schema mismatch')
                result[key]=weight
            replaced.append(key)
            server.rank_partition_records.extend(dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,
                layer=name,canonical_weight_name=key,actual_client_ranks={str(p['client_id']):p['rank'] for p in entries},
                uniform_rank_legacy_fallback=uniform,**row) for row in block_rows)
        for name,p in mother.named_parameters():p.copy_(result[name])
    if str(server.device).startswith('cuda'):torch.cuda.synchronize(server.device)
    elapsed=time.perf_counter()-start;timing=dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,model_id=0,
        phase='grouped',aggregation_mode='rank_partition',aggregation_seconds=elapsed)
    server.uploaded_ids=order;server.rank_aggregation_records.append(dict(**timing,partitioned_weights=replaced))
    matrices.update(factor_weighting='rank-dependent contributor-normalized; not the scalar weights for nonfactorized parameters',
        reference_diagnostic_scope='legacy full-upload sample-average delta only; NOT the new partition aggregate',
        scalar_weight_scope='conv1/fc3/bias and all other nonfactorized trainable parameters only')
    server.aggregation_records.append(dict(**timing,**metrics,matrices=matrices))
    server.weight_records.extend(dict(**timing,weight_scope=matrices['scalar_weight_scope'],**row) for row in rows)
    server.last_aggregation=[dict(metrics=metrics,rows=rows,matrices=matrices)]
    print(f'[RankPartition] completed={server.loop_round+1} layers={replaced} time={elapsed:.4f}s; no server truncation')
    return server.last_aggregation


def observe(server,stage):
    import torch
    if not spectrum_enabled(server.args) or server.loop_round+1 not in server.args.multi_rank_effective_spectrum_rounds:return
    cid=server.args.multi_rank_spectrum_client_id;client=server.clients[cid]
    model=server.global_model_set[0] if stage=='server_post_aggregation' else client.model_set[0]
    with torch.no_grad(),protected_rng([model]):
        for name in LAYERS:
            tick=time.perf_counter();matrix=actual_matrix(getattr(model,name)).detach().to(device='cpu',dtype=torch.float64).clone()
            if not torch.isfinite(matrix).all():raise FloatingPointError('Nonfinite actual matrix for spectral observation')
            minimum=min(getattr(c.model_set[0],name).rank for c in server.clients)
            singular=torch.linalg.svdvals(matrix);energy=singular.square()
            if not torch.isfinite(energy).all():raise FloatingPointError('Nonfinite sampled spectrum')
            inside=float(energy[:minimum].sum());outside=float(energy[minimum:].sum())
            from utils.multi_svd_spectral import spectrum_boundary_fields,WINDOWS
            tiers=sorted({getattr(c.model_set[0],name).rank for c in server.clients})
            boundary=spectrum_boundary_fields(singular,tiers,WINDOWS[name])
            server.rank_spectrum_records.append(dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,
                stage=stage,client_id=None if stage=='server_post_aggregation' else cid,layer=name,
                matrix_shape=list(matrix.shape),minimum_actual_client_rank=minimum,
                spectral_energy_within_min_rank=inside,spectral_energy_outside_min_rank=outside,
                spectral_total_energy=inside+outside,actual_matrix_frobenius_norm=float(matrix.norm()),
                spectral_outside_fraction=outside/(inside+outside) if inside+outside else 0.,
                completed_aggregations_at_observation=server.loop_round+int(stage=='server_post_aggregation'),
                completed_local_trainings_at_observation=server.loop_round+int(stage!='client_download_svd'),
                spectrum_seconds=time.perf_counter()-tick,diagnostic_dtype='float64 CPU copy',
                energy_scope='actual singular values, NOT original factor-position blocks',**boundary))


def persist(server):
    if not reporting(server.args):return
    row=dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,method=protocol(server.args)['method'],
        partitions=[r for r in server.rank_partition_records if r['loop_round']==server.loop_round],
        spectra=[r for r in server.rank_spectrum_records if r['loop_round']==server.loop_round])
    with (Path(server.report_dir())/'rank_aggregation.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,allow_nan=False)+'\n')


def manifest(server):
    return dict(protocol=protocol(server.args),partition_records=len(getattr(server,'rank_partition_records',[])),
        spectrum_records=len(getattr(server,'rank_spectrum_records',[])),
        spectrum_total_seconds=sum(r['spectrum_seconds'] for r in getattr(server,'rank_spectrum_records',[])),
        partition_aggregation_seconds=sum(r['aggregation_seconds'] for r in getattr(server,'rank_aggregation_records',[])))
