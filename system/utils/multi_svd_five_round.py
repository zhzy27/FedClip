"""Explicit K1 full-Avg paired capacity experiment observations; never train here.

SVD observers see the SAME raw U/S/V as the real download. Counterfactual raw
factors are constructed directly, not re-SVD'd. All observations restore only
the devices actually used, plus CPU/Python/NumPy RNG, parameters and modes.
"""
import copy
from pathlib import Path
import time

LAYERS=('conv2','fc1','fc2')
DEFAULT_TIERS='.90,.37,.35,.25,.15'
_DOWNLOAD_SPECTRA={}


def add_options(p):
    p.add_argument('--multi_svd_five_diagnostics',type=int,choices=[0,1],default=0)
    p.add_argument('--multi_svd_fixed_cohort_ratios',default=DEFAULT_TIERS,
        help='Diagnostic fixed membership: client_id modulo this list; independent of actual S/L capacities')


def enabled(a):return bool(getattr(a,'multi_svd_five_diagnostics',0))


def validate(a):
    if not hasattr(a,'multi_svd_fixed_cohort_ratios'):a.multi_svd_fixed_cohort_ratios=DEFAULT_TIERS
    if getattr(a,'multi_svd_five_diagnostics',0) not in (0,1):raise ValueError('five diagnostics must be 0/1')
    if not enabled(a):
        value=getattr(a,'multi_svd_fixed_cohort_ratios',DEFAULT_TIERS)
        if value!=DEFAULT_TIERS:
            try:same=[float(v) for v in value.split(',')]==[float(v) for v in DEFAULT_TIERS.split(',')]
            except (ValueError,AttributeError):same=False
            if not same:raise ValueError('Fixed cohorts need explicitly enabled five diagnostics')
        return
    from utils.multi_fc1_tail_scale import enabled as tail
    if (a.algorithm!='FedMultiGroupProj' or not getattr(a,'multi_avg_proto_ablation',False)
        or not getattr(a,'multi_svd',False) or a.num_server_models!=1 or a.join_ratio!=1.
        or a.multi_group_mode!='group_avg_only' or getattr(a,'multi_svd_aggregation','full_avg')!='full_avg'
        or a.multi_proto_mode!='none' or a.multi_proto_lambda!=0 or tail(a)
        or a.target_head_mode!='shared' or a.target_local_schedule!='joint'
        or getattr(a,'multi_last_epoch_protocol',False) or getattr(a,'multi_no_pre_eval',False)):
        raise ValueError('Five-run diagnostics need K1 full-participation low-rank full_avg, shared/joint, no auxiliary module, with pre evaluation')
    if a.eval_gap!=1:raise ValueError('Five-run diagnostics collect every training round: require eval_gap=1')
    if getattr(a,'multi_svd_spectral_mode','none') not in ('none','all_tail'):
        raise ValueError('Five-run comparison supports original or all_tail downloads')
    try:rates=[float(v) for v in a.multi_svd_fixed_cohort_ratios.split(',')]
    except (AttributeError,ValueError):raise ValueError('Invalid fixed cohort rank rates')
    import math
    if not rates or len(set(rates))!=len(rates) or any(not math.isfinite(v) or not 0<v<1 for v in rates):
        raise ValueError('Fixed cohort ratios must be distinct finite low-rank rates')
    a.multi_svd_fixed_cohort_effective_ratios=rates


def configure(server):
    # This is the SAME unmodified sample count used by the real Avg kernel.
    if getattr(server.args,'multi_svd_spectral_mode','none')=='all_tail':
        server.args.multi_svd_avg_train_samples=[c.train_samples for c in server.clients]
    if enabled(server.args):
        _DOWNLOAD_SPECTRA.clear()
        server.five_spectra=[];server.five_endpoints=[];server.five_metrics=[];server.five_slots=[]
        server.five_costs=[]


def capture_spectrum(record,name,raw,used,rank):
    if record is None:raise RuntimeError('Missing same-SVD observation receiver')
    record[name]=dict(raw=raw.detach().cpu().clone(),used=used.detach().cpu().clone(),rank=rank)


def svd_observers(record):
    def observe(name,layer,u,s,v):
        if name not in record:raise RuntimeError('SVD spectrum was not captured before factors')
        r=layer.rank;root=s[:r].sqrt()
        record[name]['raw_factors']=(u[:,:r]@root.diag(),root.diag()@v[:r,:])
    return {name:(lambda layer,u,s,v,name=name:observe(name,layer,u,s,v)) for name in LAYERS}


def tier_ranks(model,args,name):
    from flcore.trainmodel.models import factorized_linear_rank
    from utils.multi_rank_partition import actual_matrix
    matrix=actual_matrix(getattr(model,name))
    rates=args.multi_svd_fixed_cohort_effective_ratios
    return sorted({factorized_linear_rank(matrix.shape[1],matrix.shape[0],v) for v in rates})


def spectrum_row(s,tiers,**fields):
    import torch
    s=s.detach().cpu().double()
    if not torch.isfinite(s).all() or (s<0).any():raise FloatingPointError('Invalid diagnostic spectrum')
    ratios=[];blocks=[];left=0
    for right in tiers:
        if right>len(s):raise ValueError('Diagnostic tier exceeds actual matrix')
        ratios.append(dict(boundary_rank=right,sigma_at_boundary=float(s[right-1]),
            sigma_next=float(s[right]) if right<len(s) else None,
            next_over_boundary=float(s[right]/s[right-1]) if right<len(s) and s[right-1]>0 else None))
        blocks.append(dict(slot_start=left+1,slot_end=right,singular_value_squared_energy=float(s[left:right].square().sum())))
        left=right
    blocks.append(dict(slot_start=left+1,slot_end=len(s),singular_value_squared_energy=float(s[left:].square().sum())))
    return dict(**fields,one_based_rank_start=1,singular_values=s.tolist(),boundary_ratios=ratios,
        spectral_segments=blocks,total_spectral_energy=float(s.square().sum()),
        spectrum_unit='actual singular value; energy=sum(sigma^2); NOT fixed-factor-slot energy')


def evaluate(client,model,stage):
    import torch
    from common.data_loading import make_data_loader
    from utils.data_utils import read_client_data
    from utils.multi_rank_partition import protected_rng
    start=time.perf_counter();rows=[]
    with protected_rng([model]),torch.no_grad():
        state={n:p.detach().clone() for n,p in model.state_dict().items()};modes=[(m,m.training) for m in model.modules()]
        try:
            model.eval()
            for split in ('train','test'):
                data=read_client_data(client.dataset,client.id,client.args,is_train=split=='train',few_shot=0)
                loader=make_data_loader(data,client.args.test_batch_size,shuffle=False,drop_last=False,num_workers=0)
                loss=0.;correct=samples=0
                for x,y in loader:
                    y=y.to(client.device);logits=model(x.to(client.device))
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite five-run endpoint')
                    loss+=float(torch.nn.functional.cross_entropy(logits,y,reduction='sum'))
                    correct+=int((logits.argmax(1)==y).sum());samples+=len(y)
                if not samples or samples!=(client.train_samples if split=='train' else client.test_samples):
                    raise ValueError('Diagnostic dataset differs from audited client samples')
                rows.append(dict(loop_round=client.download_round,completed_train_count=client.download_round+1,
                    client_id=client.id,model_id=0,stage=stage,split=split,CE=loss/samples,
                    correct=correct,samples=samples,accuracy=correct/samples,accuracy_unit='fraction',
                    rank_rate=client.rank_rate,fixed_cohort=client.id%len(client.args.multi_svd_fixed_cohort_effective_ratios)))
        finally:
            model.load_state_dict(state,strict=True)
            for m,training in modes:m.training=training
    client.five_endpoint_rows.extend(rows)
    client.five_cost_rows.append(dict(loop_round=client.download_round,client_id=client.id,stage=stage,
        kind='extra_readonly_endpoint',forward_samples=sum(r['samples'] for r in rows),seconds=time.perf_counter()-start))


def after_download(client,record,loop):
    import torch
    from utils.multi_rank_partition import factors,actual_matrix,protected_rng
    model=client.model_set[0];client.five_endpoint_rows=[];client.five_spectrum_rows=[];client.five_cost_rows=[]
    with protected_rng([model]),torch.no_grad():
        for name in LAYERS:
            tick=time.perf_counter();entry=record[name];tiers=tier_ranks(model,client.args,name)
            base=dict(loop_round=loop,completed_train_count=loop+1,layer=name,model_id=0,
                completed_aggregations_before_download=loop,diagnostic_seconds=0.)
            if client.id==0:
                client.five_spectrum_rows.append(spectrum_row(entry['raw'],tiers,**base,client_id=None,
                    stage='server_original_download',spectral_basis='full original same-download SVD'))
            matrix=actual_matrix(getattr(model,name)).detach().cpu().double()
            from utils.head_snapshot import tensor_fingerprint
            cache_key=(loop,name,tensor_fingerprint({'actual_matrix':matrix}))
            cached=cache_key in _DOWNLOAD_SPECTRA
            if not cached:_DOWNLOAD_SPECTRA[cache_key]=torch.linalg.svdvals(matrix)
            actual=_DOWNLOAD_SPECTRA[cache_key]
            client.five_spectrum_rows.append(spectrum_row(actual,tiers,**base,client_id=client.id,
                stage='client_actual_download',diagnostic_cache_hit=cached,
                actual_matrix_sha256=cache_key[-1],spectral_basis='actual restored factor product; float64 observation SVD; reuse ONLY exact matrix hash'))
            client.five_spectrum_rows[-1]['diagnostic_seconds']=time.perf_counter()-tick
        if client.id==0:
            client.five_slot_start={name:{k:v.detach().cpu().clone() for k,v in factors(getattr(model,name)).items() if k in ('U','V')} for name in LAYERS}
        # Raw twin is made from the SAME SVD vectors before local training.
        if loop>0 and client.args.multi_svd_spectral_mode=='all_tail':
            tick=time.perf_counter();raw=copy.deepcopy(model)
            for name in LAYERS:
                f=factors(getattr(raw,name));u,v=record[name]['raw_factors'];f['U'].copy_(u);f['V'].copy_(v)
            construction_seconds=time.perf_counter()-tick
            evaluate(client,raw,'same_mother_raw_counterfactual')
            del raw
            client.five_cost_rows.append(dict(loop_round=loop,client_id=client.id,stage='same_mother_raw_counterfactual',
                kind='raw_twin_construction',seconds=construction_seconds))
        evaluate(client,model,'download')


def fixed_slot_rows(before,after,tiers,**fields):
    import torch
    rows=[];left=0;r=before['U'].shape[1]
    for bound in sorted(set(tiers+[r])):
        right=min(bound,r)
        if right<=left:continue
        a=before['U'][:,left:right].double()@before['V'][left:right].double()
        b=after['U'][:,left:right].double()@after['V'][left:right].double()
        norm=float(a.norm());change=float((b-a).norm())
        if not torch.isfinite(b).all():raise FloatingPointError('Nonfinite slot product')
        rows.append(dict(**fields,slot_start=left+1,slot_end=right,initial_block_norm=norm,
            trained_block_norm=float(b.norm()),absolute_product_change=change,relative_product_change=change/norm if norm else None,
            scope='download original factor slots; NOT re-SVD components; blocks need not be orthogonal'))
        left=right
    return rows


def after_local(server):
    import torch
    from utils.multi_rank_partition import actual_matrix,factors,protected_rng
    if not enabled(server.args):return
    for c in server.clients:
        evaluate(c,c.model_set[0],'post_local')
        server.five_endpoints.extend(c.five_endpoint_rows);server.five_spectra.extend(c.five_spectrum_rows);server.five_costs.extend(c.five_cost_rows)
        c.five_endpoint_rows=[];c.five_spectrum_rows=[];c.five_cost_rows=[]
    c=server.clients[0];model=c.model_set[0]
    with protected_rng([model]),torch.no_grad():
        for name in LAYERS:
            tick=time.perf_counter();tiers=tier_ranks(model,server.args,name);layer=getattr(model,name)
            server.five_spectra.append(spectrum_row(torch.linalg.svdvals(actual_matrix(layer).detach().cpu().double()),tiers,
                loop_round=server.loop_round,completed_train_count=server.loop_round+1,stage='C0_post_local',client_id=0,model_id=0,layer=name))
            server.five_spectra[-1]['diagnostic_seconds']=time.perf_counter()-tick
            current={k:v.detach().cpu() for k,v in factors(layer).items() if k in ('U','V')}
            server.five_slots.extend(fixed_slot_rows(c.five_slot_start[name],current,tiers,loop_round=server.loop_round,
                completed_train_count=server.loop_round+1,client_id=0,layer=name))
    del c.five_slot_start


def weighted_points(rows,cohorts):
    points=[]
    for stage in sorted({r['stage'] for r in rows}):
        for split in ('train','test'):
            subset=[r for r in rows if r['stage']==stage and r['split']==split]
            if len({r['client_id'] for r in subset})!=len(subset):raise ValueError('Duplicate endpoint')
            for name,ids in [('all',sorted(cohorts)),('C0',[0])]+[(f'fixed_tier_{k}',[i for i,g in cohorts.items() if g==k]) for k in sorted(set(cohorts.values()))]:
                chosen=[r for r in subset if r['client_id'] in ids]
                if len(chosen)!=len(ids):raise ValueError('Incomplete fixed-cohort endpoint')
                samples=sum(r['samples'] for r in chosen);correct=sum(r['correct'] for r in chosen)
                points.append(dict(loop_round=chosen[0]['loop_round'],completed_train_count=chosen[0]['completed_train_count'],
                    stage=stage,split=split,evaluation_scope=name,member_ids=ids,correct=correct,samples=samples,
                    accuracy=correct/samples,CE=sum(r['CE']*r['samples'] for r in chosen)/samples,weighted_by='samples',accuracy_unit='fraction'))
    return points


def after_aggregation(server):
    import json
    import torch
    from utils.multi_rank_partition import actual_matrix,protected_rng
    if not enabled(server.args):return
    model=server.global_model_set[0]
    with protected_rng([model]),torch.no_grad():
        for name in LAYERS:
            tick=time.perf_counter();tiers=tier_ranks(model,server.args,name)
            server.five_spectra.append(spectrum_row(torch.linalg.svdvals(actual_matrix(getattr(model,name)).detach().cpu().double()),tiers,
                loop_round=server.loop_round,completed_train_count=server.loop_round+1,stage='server_post_aggregation',client_id=None,model_id=0,layer=name))
            server.five_spectra[-1]['diagnostic_seconds']=time.perf_counter()-tick
    rows=[r for r in server.five_endpoints if r['loop_round']==server.loop_round]
    cohorts={c.id:c.id%len(server.args.multi_svd_fixed_cohort_effective_ratios) for c in server.clients}
    metrics=weighted_points(rows,cohorts);server.five_metrics.extend(metrics)
    with (Path(server.report_dir())/'five_round_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(loop_round=server.loop_round,metrics=metrics,
            slots=[r for r in server.five_slots if r['loop_round']==server.loop_round],
            spectra=[r for r in server.five_spectra if r['loop_round']==server.loop_round]),allow_nan=False)+'\n')
    _DOWNLOAD_SPECTRA.clear()  # At most current-round vectors, never model/feature history.


def tables(server):
    indexed={(r['completed_train_count'],r['evaluation_scope'],r['split'],r['stage']):r for r in server.five_metrics}
    paired=[]
    for (round_id,scope,split,stage),post in indexed.items():
        if stage!='post_local':continue
        pre=indexed[round_id,scope,split,'download'];raw=indexed.get((round_id,scope,split,'same_mother_raw_counterfactual'))
        paired.append(dict(completed_train_count=round_id,loop_round=round_id-1,evaluation_scope=scope,split=split,
            accuracy_unit='fraction',pre_accuracy=pre['accuracy'],post_accuracy=post['accuracy'],
            same_round_local_accuracy_gain=post['accuracy']-pre['accuracy'],post_minus_pre_CE=post['CE']-pre['CE'],
            same_mother_corrected_minus_raw_accuracy=pre['accuracy']-raw['accuracy'] if raw else None,
            same_mother_corrected_minus_raw_CE=pre['CE']-raw['CE'] if raw else None))
    return dict(five_client_endpoints=server.five_endpoints,five_cohort_metrics=server.five_metrics,
        five_layer_spectra=server.five_spectra,five_C0_fixed_slots=server.five_slots,five_diagnostic_costs=server.five_costs,
        five_paired_gains=paired)


def manifest(server):
    return dict(enabled=True,fixed_cohort_ratios=server.args.multi_svd_fixed_cohort_effective_ratios,
        fixed_cohort_rule='client_id modulo reference tier count, independent of actual S/L/H capacity',
        shared_initialization_sha256=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],
        instant_counterfactual='same download SVD U/V, raw vs corrected; raw eval ONLY, corrected SGD ONLY',
        spectrum_format='one row per stage/client/layer; singular_values[j-1] is one-based rank j; boundary ratios/segment energies separate',
        slot_scope='C0 download factor slots; never confuse with retrained SVD components',
        extra_SGD_steps=0,extra_training_epochs=0,
        extra_observation_SVD_count=sum(r['stage']!='server_original_download' and not r.get('diagnostic_cache_hit',False) for r in server.five_spectra),
        extra_training_download_SVD_count=0,
        endpoint_cost_seconds=sum(r['seconds'] for r in server.five_costs),
        spectrum_seconds=sum(r.get('diagnostic_seconds',0.) for r in server.five_spectra))
