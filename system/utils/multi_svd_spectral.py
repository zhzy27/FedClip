"""D01 full-Avg client spectra: same SVD, after first aggregation only.

All fitting/curve construction uses the FULL spectrum and shared actual rank
tiers, never a client's truncated prefix. No train/eval forward lives here.
"""
import hashlib
import json
import math
from pathlib import Path
import time

from utils.multi_group_svd import DEFAULT_FACTORIZED_LAYERS as LAYERS, factorized_layers
WINDOWS=dict(conv2=8,fc1=20,fc2=20)


def add_options(parser):
    parser.add_argument('--multi_svd_spectral_mode',choices=['none','single_point','next_tier','all_tail','first_band','coverage_reorder'],default='none')
    parser.add_argument('--multi_svd_spectral_layers',default='conv2,fc1,fc2')
    parser.add_argument('--multi_svd_spectral_max_gain',type=float,default=1.5)


def enabled(args):return getattr(args,'multi_svd_spectral_mode','none')!='none'


def layer_names(args):
    value=getattr(args,'multi_svd_spectral_layers','conv2,fc1,fc2')
    try:names=[v.strip() for v in value.split(',')] if isinstance(value,str) else list(value)
    except TypeError:raise ValueError('Spectral layers must be a nonempty subset of conv2,fc1,fc2')
    if not names or len(set(names))!=len(names) or any(v not in LAYERS for v in names):
        raise ValueError('Spectral layers must be nonempty, known and without duplicates')
    return [v for v in LAYERS if v in names]


def validate(args):
    mode=getattr(args,'multi_svd_spectral_mode','none');gain=getattr(args,'multi_svd_spectral_max_gain',1.5)
    if mode not in ('none','single_point','next_tier','all_tail','first_band','coverage_reorder'):raise ValueError('Invalid multi_svd_spectral_mode')
    if not isinstance(gain,(int,float)) or not math.isfinite(gain) or gain<1:
        raise ValueError('multi_svd_spectral_max_gain must be finite and >=1')
    layer_names(args)
    if not enabled(args):return
    if not set(layer_names(args)) <= set(factorized_layers(args)):
        raise ValueError('Spectral corrections must select only factorized layers')
    if mode == 'coverage_reorder' and layer_names(args) != ['fc1']:
        raise ValueError('coverage_reorder requires --multi_svd_spectral_layers fc1')
    if (getattr(args,'algorithm','')!='FedMultiGroupProj' or not getattr(args,'multi_avg_proto_ablation',False)
        or not getattr(args,'multi_svd',False) or getattr(args,'model_family','')!='Decom_CNN-5-512'
        or args.num_server_models!=1 or args.multi_group_mode!='group_avg_only' or args.join_ratio!=1.
        or getattr(args,'multi_svd_aggregation','full_avg')!='full_avg' or getattr(args,'multi_fc1_tail_scale_strength',0.)!=0
        or args.target_head_mode!='shared' or args.target_local_schedule!='joint'
        or getattr(args,'multi_last_epoch_protocol',False) or getattr(args,'multi_head_mode','shared')!='shared'
        or getattr(args,'multi_projection_scope','full')!='full' or getattr(args,'multi_helper_head_update','train')!='train'
        or args.multi_proto_mode!='none' or args.multi_proto_lambda!=0):
        raise ValueError('Spectral correction v1 needs K1 full-participation SVD full_avg/group_avg_only, shared/joint, no prototypes, old fc1 tail scale=0')
    if (getattr(args,'target_prox_mu',0) or getattr(args,'target_z2_lambda',0) or getattr(args,'target_head_l2_lambda',0)
        or getattr(args,'target_curriculum_mode','none')!='none' or getattr(args,'target_c0_replay_mode','none')!='none'
        or getattr(args,'multi_kd_mode','none')!='none'):
        raise ValueError('Spectral correction does not mix other local training modules')
    if any(r>=1 for r in args.multi_svd_effective_ratios):raise ValueError('Spectral correction requires actual low-rank client factors')


def sampled(args,loop):
    upcoming=loop+1
    return upcoming in set(range(1,11))|{20,50,args.global_rounds+1}


def policy(args):
    value=dict(mode=getattr(args,'multi_svd_spectral_mode','none'),layers=layer_names(args),
        max_gain=getattr(args,'multi_svd_spectral_max_gain',1.5),fit_windows=WINDOWS,
        timing='loop=0 / upcoming training R1 unchanged; first correction after aggregation #1, before R2 SGD',
        ranks='two smallest distinct ACTUALLY assigned integer ranks, independent per layer; FULL shared mother spectrum',
        fit='float64 CPU OLS log(positive sigma) vs one-based rank in lowest-tier final window; slope clamped <=0; measured sigma_k anchor',
        zero_rule='never lift zero; single_point sigma_(k+1)=0 skips; next_tier sigma_b=0 leaves b and later zero tail unchanged',
        dtype_tolerance='gains checked within 8*machine-epsilon after original dtype rounding; no sorting/renormalization repair',
        log_upcoming_train_counts=sorted(r for r in set(range(1,11))|{20,50,args.global_rounds+1} if r<=args.global_rounds+1),
        server='NO correction/SVD/truncation on server; ordinary sample-weighted full matrix Avg',
        local_objective='unchanged CE plus configured original F, original SGD/clip/epochs',
        global_delta='unchanged restored post-local minus full server; download-distance includes spectral correction as well as truncation')
    if value['mode']=='all_tail':
        value.update(max_gain=None,fit=None,fit_windows=None,
            ranks='all distinct actually assigned integer layer ranks; per-client original train-sample weights',
            correction='q_j=sum(n_i for rank_i>=j)/sum(n_i); candidate=sigma_j/q_j; forward nonincreasing clamp at preceding corrected value; ONLY min_rank<j<=max_rank',
            max_gain_parameter_effective=False,zero_rule='original zero stays zero; beyond maximum client rank unchanged',
            actual_train_samples=getattr(args,'multi_svd_avg_train_samples',None))
    elif value['mode']=='first_band':
        value.update(max_gain=None,fit=None,fit_windows=None,max_gain_parameter_effective=False,
            correction='all_tail sample-mass/forward clamp formula ONLY for min_rank<j<=second_distinct_rank; original prefix and all j>second_distinct_rank EXACTLY preserved',
            ranks='two smallest distinct actually assigned integer layer ranks; all clients share the same full first-band curve',
            zero_rule='original zero stays zero; no change outside (k,b]',
            actual_train_samples=getattr(args,'multi_svd_avg_train_samples',None))
    elif value['mode']=='coverage_reorder':
        value.update(max_gain=None,fit=None,fit_windows=None,max_gain_parameter_effective=False,
            ranks='actual assigned fc1 ranks; per-client original train-sample weights',
            dtype_tolerance=None,
            correction='full sigma/q through largest assigned rank, stable descending U/S/Vh permutation before truncation',
            log_upcoming_train_counts=list(range(1,args.global_rounds+2)),
            local_objective='CE plus active-prefix Frobenius with the configured fc1 local schedule',
            zero_rule='original zero stays zero; values beyond maximum assigned rank unamplified')
    return value


def boundary_ranks(model,args):
    from flcore.trainmodel.models import factorized_linear_rank
    rates=args.multi_svd_effective_ratios;active=[rates[i] for i in args.multi_svd_effective_assignment]
    result={}
    layers = factorized_layers(args)
    for name in layers:
        layer=getattr(model,name)
        if name=='conv2':
            kernel=layer.kernel_size[0] if isinstance(layer.kernel_size,tuple) else layer.kernel_size
            rows,cols=layer.out_channels*kernel,layer.in_channels*kernel
        else:rows,cols=layer.out_features,layer.in_features
        ranks=sorted({factorized_linear_rank(cols,rows,r) for r in active})
        result[name]=dict(k=ranks[0],b=ranks[1] if len(ranks)>1 else None,actual_ranks=ranks,
            client_ranks=[factorized_linear_rank(cols,rows,r) for r in active],
            matrix_rows=rows,matrix_columns=cols,fit_window=min(ranks[0],WINDOWS[name]))
    # Guard current requested configuration, never use this table to derive ranks.
    if (layers==LAYERS and set(active)=={.90,.37,.35,.25,.15} and [(result[n]['matrix_rows'],result[n]['matrix_columns']) for n in layers]
        ==[(160,80),(2000,800),(512,2000)]):
        expected=dict(conv2=(12,20),fc1=(120,200),fc2=(77,128))
        if any((result[n]['k'],result[n]['b'])!=expected[n] for n in LAYERS):raise RuntimeError('D01 actual integer rank boundary mismatch')
    for name in layer_names(args):
        if enabled(args) and result[name]['b'] is None:raise ValueError(f'{name} needs at least two distinct actually assigned ranks')
    return result


def all_tail_spectrum(singular,client_ranks,train_samples,*,details=True):
    """Sample-mass correction on the ORIGINAL full SVD spectrum, not factors."""
    import torch
    if (singular.ndim!=1 or not singular.is_floating_point() or not len(singular)
        or not torch.isfinite(singular).all() or (singular<0).any() or (singular[:-1]<singular[1:]).any()
        or not client_ranks or len(client_ranks)!=len(train_samples)
        or any(type(r) is not int or not 1<=r<=len(singular) for r in client_ranks)
        or any(type(n) is not int or n<=0 for n in train_samples)):
        raise ValueError('all_tail needs ordered finite spectrum, actual layer ranks and positive original train counts')
    with torch.no_grad():
        raw=singular.detach().cpu().double();out=raw.clone();k=min(client_ranks);top=max(client_ranks)
        total=sum(train_samples);records=[];limited=0
        for j in range(k+1,top+1):
            mass=sum(n for r,n in zip(client_ranks,train_samples) if r>=j);q=mass/total
            candidate=raw[j-1]/q
            value=min(float(candidate),float(out[j-2]))
            out[j-1]=value;constrained=bool(candidate>out[j-2]);limited+=int(constrained and raw[j-1]>0)
            if details:records.append(dict(rank=j,contributor_mass=q,contributor_train_samples=mass,
                theoretical_gain=1/q,original=float(raw[j-1]),corrected=value,
                actual_gain=value/float(raw[j-1]) if raw[j-1]>0 else None,nonincreasing_limited=constrained))
        output=out.to(singular)
        if (not torch.isfinite(output).all() or (output[:-1]<output[1:]).any() or (output<singular).any()
            or not torch.equal(output[:k],singular[:k]) or not torch.equal(output[top:],singular[top:])
            or (output[singular==0]!=0).any()):raise FloatingPointError('Invalid all_tail spectrum')
        # Record gains of the actual dtype-rounded values used by sqrt().
        for row in records:
            row['corrected']=float(output[row['rank']-1]);row['actual_gain']=row['corrected']/row['original'] if row['original'] else None
        return (singular if torch.equal(output,singular) else output),dict(k=k,maximum_client_rank=top,
            actual_rank_tiers=sorted(set(client_ranks)),changed=not torch.equal(output,singular),
            nonincreasing_limited_ranks=limited,rank_corrections=records,max_gain=None,skipped_reason=None,
            correction_formula='sigma_i / sum_original_sample_weights(rank>=i), then forward nonincreasing clamp')


def first_band_spectrum(singular,client_ranks,train_samples,*,details=True):
    """Reuse the exact all_tail recurrence, retain ONLY its first tier interval.

    Later all_tail computations never influence the first band. Copying that
    prefix back to the ORIGINAL spectrum preserves every value outside (k,b].
    No SVD, model construction, randomness, sorting or gain cap here.
    """
    import torch
    tiers=sorted(set(client_ranks))
    if len(tiers)<2:raise ValueError('first_band needs two distinct actually assigned integer ranks')
    full,receipt=all_tail_spectrum(singular,client_ranks,train_samples,details=True)
    k,b=tiers[:2];output=singular.clone();output[k:b]=full[k:b]
    rows=[]
    for row in receipt['rank_corrections']:
        if row['rank']>b:break
        owners=[i for i,r in enumerate(client_ranks) if r>=row['rank']]
        rows.append(dict(row,contributor_ids=owners,contributor_count=len(owners),all_train_samples=sum(train_samples)))
    if (not torch.equal(output[:k],singular[:k]) or not torch.equal(output[b:],singular[b:])
        or not torch.isfinite(output).all() or (output[:-1]<output[1:]).any() or (output[singular==0]!=0).any()):
        raise FloatingPointError('Invalid first_band prefix/tail/monotonicity/zero invariant')
    changed=not torch.equal(output,singular)
    return (output if changed else singular),dict(k=k,b=b,correction_end_rank=b,maximum_client_rank=max(client_ranks),
        actual_rank_tiers=tiers,changed=changed,nonincreasing_limited_ranks=sum(r['nonincreasing_limited'] and r['original']>0 for r in rows),
        rank_corrections=rows if details else [],max_gain=None,skipped_reason=None,
        correction_formula='sigma_j/q_j then clamp at previous corrected value ONLY k<j<=b; outside original exactly')


def boundary_indices(k,b,length,window=20):
    end=b if b is not None else k
    return sorted(set(range(max(1,k-window+1),min(length,end+4)+1)))


def spectrum_boundary_fields(singular,ranks,window=20):
    """Pure detail over an ALREADY computed actual matrix spectrum; no SVD here."""
    import torch
    k=ranks[0];b=ranks[1] if len(ranks)>1 else None
    positions=boundary_indices(k,b,len(singular),window)
    return dict(boundary_k=k,boundary_b=b,boundary_actual_rank_tiers=list(ranks),
        boundary_singular_values=[dict(rank=j,value=float(singular[j-1])) for j in positions],
        next_tier_spectral_energy=float(singular[k:b].square().sum()) if b is not None else None,
        boundary_spectrum_scope='actual restored parameter matrix at named observation stage; not source download S or factor slots')


def correct_spectrum(singular,k,b,mode,max_gain,window,*,retain_rank=None,details=True):
    """Return spectrum in SAME dtype/device. Input is never mutated or re-SVD'd."""
    import torch
    if mode not in ('single_point','next_tier') or not math.isfinite(max_gain) or max_gain<1:
        raise ValueError('Invalid correction mode/gain')
    if (singular.ndim!=1 or not singular.is_floating_point() or not 1<=k<b<=len(singular)
        or type(window) is not int or window<1 or not torch.isfinite(singular).all() or (singular<0).any()
        or (singular[:-1]<singular[1:]).any()):raise ValueError('Require finite nonnegative nonincreasing full SVD spectrum and legal k/b/window')
    r=len(singular) if retain_rank is None else retain_rank
    if not k<=r<=len(singular):raise ValueError('Invalid retained rank for diagnostics')
    with torch.no_grad():
        raw=singular.detach().cpu().double();start=max(0,k-window)
        index=torch.arange(start+1,k+1,dtype=torch.float64);prefix=raw[start:k];positive=prefix>0
        meta=dict(k=k,b=b,fit_window_requested=window,fit_window_actual=k-start,
            fit_rank_start=start+1,fit_rank_end=k,fit_positive_count=int(positive.sum()),raw_slope=None,slope=None,
            anchor_sigma_k=float(raw[k-1]),zero_sigma_k_plus_1=bool(raw[k]==0),skipped_reason=None)
        reason=None;curves=None
        if int(positive.sum())<2:reason='insufficient_positive_fit_values'
        elif raw[k-1]==0:reason='zero_anchor_sigma_k'
        else:
            x=index[positive];y=prefix[positive].log();center=x-x.mean()
            slope_raw=float((center*(y-y.mean())).sum()/center.square().sum());slope=min(slope_raw,0.)
            if not math.isfinite(slope):raise FloatingPointError('Nonfinite fitted log-slope')
            meta.update(raw_slope=slope_raw,slope=slope)
            target=raw[k-1]*torch.exp(slope*(torch.arange(k+1,len(raw)+1,dtype=torch.float64)-k))
            A=raw.clone();B=raw.clone()
            def uniform(tail,gain,ceiling):
                result=tail*gain;roundoff=0
                # The analytic multiplier already respects this boundary.
                # Round a uniform multiplier DOWN (never reorder S) if a
                # float64 divide/multiply pair overshoots by an ulp.
                while len(result) and result[0]>ceiling and roundoff<4:
                    gain=max(1.,math.nextafter(gain,0.));result=tail*gain;roundoff+=1
                return result,gain,roundoff
            single_gain=None;later_gain=None;single_roundoff=later_roundoff=0
            if raw[k]>0:
                gain=max(1.,min(max_gain,float(target[0]/raw[k])))
                A[k:],single_gain,single_roundoff=uniform(raw[k:],gain,raw[k-1])
            B[k:b]=torch.maximum(raw[k:b],torch.minimum(target[:b-k],max_gain*raw[k:b]))
            if raw[b-1]>0:
                gain=max(1.,min(max_gain,float(B[b-1]/raw[b-1])))
                B[b:],later_gain,later_roundoff=uniform(raw[b:],gain,B[b-1])
            meta.update(single_point_uniform_gain=single_gain,next_tier_after_b_uniform_gain=later_gain,
                uniform_multiplier_roundoff_steps=single_roundoff if mode=='single_point' else later_roundoff)
            curves=dict(single_point=A,next_tier=B)
            if mode=='single_point' and raw[k]==0:reason='zero_sigma_k_plus_1'
        proposed=raw if curves is None or reason else curves[mode]
        output=proposed.to(singular)
        if (not torch.isfinite(output).all() or (output[:-1]<output[1:]).any() or not torch.equal(output[:k],singular[:k])
            or (output[singular==0]!=0).any()):raise FloatingPointError('Corrected spectrum violated prefix/monotonicity/zero/finite invariant')
        effective=output.detach().cpu().double();mask=raw>0;gains=torch.ones_like(raw);gains[mask]=effective[mask]/raw[mask]
        tolerance=8*torch.finfo(singular.dtype).eps
        if ((gains[mask]<1-tolerance).any() or (gains[mask]>max_gain+tolerance*max_gain).any()):
            raise FloatingPointError('Actual correction gain outside [1,max_gain] after dtype rounding')
        mid=(k+1+b)//2;cap=(gains>=max_gain-tolerance*max_gain)&mask
        meta.update(skipped_reason=reason,changed=not torch.equal(output,singular),
            gain_k_plus_1=float(gains[k]) if raw[k]>0 else None,
            midpoint_rank=mid,gain_midpoint=float(gains[mid-1]) if raw[mid-1]>0 else None,
            gain_b=float(gains[b-1]) if raw[b-1]>0 else None,
            zero_sigma_b=bool(raw[b-1]==0),next_tier_cap_hits=int(cap[k:b].sum()),
            next_tier_positive_ranks=int(mask[k:b].sum()),retained_tail_cap_hits=int(cap[k:r].sum()),
            full_tail_cap_hits=int(cap[k:].sum()),
            next_tier_energy_before=float(raw[k:b].square().sum()),next_tier_energy_after=float(effective[k:b].square().sum()),
            full_tail_energy_before=float(raw[k:].square().sum()),full_tail_energy_after=float(effective[k:].square().sum()))
        if details:
            positions=boundary_indices(k,b,len(raw),window)
            meta['boundary_values']=[dict(rank=j,original=float(raw[j-1]),corrected=float(effective[j-1]),
                actual_gain=float(gains[j-1]) if raw[j-1]>0 else None,retained=j<=r) for j in positions]
            if curves is not None:
                a=curves['single_point'].to(singular);c=curves['next_tier'].to(singular)
                ac=a.cpu().double();bc=c.cpu().double();eligible=raw[k:b]>0
                a_cap=eligible & ((ac[k:b]/raw[k:b].clamp_min(torch.finfo(torch.float64).tiny)-max_gain).abs()<=tolerance*max_gain)
                b_cap=eligible & ((bc[k:b]/raw[k:b].clamp_min(torch.finfo(torch.float64).tiny)-max_gain).abs()<=tolerance*max_gain)
                both=bool(eligible.all() and a_cap.sum()==eligible.sum() and b_cap.sum()==eligible.sum())
                same=torch.equal(a,c)
                meta.update(counterfactual_same_full_corrected_spectrum=same,counterfactual_both_entire_interval_at_cap=both,
                    counterfactual_A_B_max_absolute_difference=float((ac-bc).abs().max()),
                    equivalence_note='same-mother A/B both fully saturated; actual corrections identical' if both and same else
                        'same-mother actual float spectra identical' if same else 'same-mother correction rules differ')
        if not meta['changed']:output=singular  # Preserve exact old construction for a no-op.
        return output,meta


def decompose(model,rate,args,client_id,column,loop,*,observation=None):
    import torch
    from flcore.trainmodel.models import factorized_linear_rank
    bounds=boundary_ranks(model,args);selected=set(layer_names(args));records=[];transforms={}
    from utils import multi_svd_five_round as five
    observing=five.enabled(args)
    detail=sampled(args,loop)
    for name in bounds:
        bound=bounds[name];k,b=bound['k'],bound['b']
        rank=factorized_linear_rank(bound['matrix_columns'],bound['matrix_rows'],rate)
        row=dict(loop_round=loop,completed_aggregations_before_download=loop,upcoming_completed_train_count=loop+1,
            client_id=client_id,model_id=column,layer=name,capacity_level=args.multi_svd_effective_assignment[client_id],
            rank_rate=rate,actual_rank=rank,k=k,b=b,selected=name in selected,requested_mode=args.multi_svd_spectral_mode,
            effective_mode='none',max_gain=None if args.multi_svd_spectral_mode in ('all_tail','first_band') else args.multi_svd_spectral_max_gain,changed=False,
            skipped_reason='initial_download_original_path' if loop==0 else 'minimum_actual_rank_original_path' if rank==k else 'layer_not_selected',
            extra_download_svd_count=0,full_spectrum_before_client_truncation=True)
        records.append(row)
        common_tail=args.multi_svd_spectral_mode in ('all_tail','first_band')
        if (loop==0 or rank==k and not common_tail) and not observing:continue
        if name not in selected and not detail and not observing:continue
        def transform(singular,actual_rank,name=name,row=row,bound=bound,rank=rank):
            if actual_rank!=rank:raise RuntimeError(f'{name} actual module rank disagrees with capacity rule')
            tick=time.perf_counter()
            if row['selected'] and loop>0 and (rank>bound['k'] or args.multi_svd_spectral_mode in ('all_tail','first_band')) and enabled(args):
                if args.multi_svd_spectral_mode in ('all_tail','first_band'):
                    if not hasattr(args,'multi_svd_avg_train_samples'):
                        raise ValueError('Sample-mass spectral modes require audited original per-client train counts bound by the server')
                    kernel=first_band_spectrum if args.multi_svd_spectral_mode=='first_band' else all_tail_spectrum
                    output,metrics=kernel(singular,bound['client_ranks'],args.multi_svd_avg_train_samples,details=detail)
                else:
                    output,metrics=correct_spectrum(singular,bound['k'],bound['b'],args.multi_svd_spectral_mode,
                        args.multi_svd_spectral_max_gain,WINDOWS[name],retain_rank=rank,details=detail)
                row.update(metrics,effective_mode=args.multi_svd_spectral_mode)
                if args.multi_svd_spectral_mode in ('all_tail','first_band'):
                    row['retained_spectrum_changed']=not torch.equal(output[:rank],singular[:rank])
            else:
                output=singular
                # Unselected layers are observed only at requested sample points,
                # and their original S object goes straight back to sqrt/SVD factors.
                if bound['b'] is not None and not observing:
                    _,metrics=correct_spectrum(singular,bound['k'],bound['b'],'single_point',1.,WINDOWS[name],retain_rank=rank,details=True)
                    for key in list(metrics):
                        if key.startswith('counterfactual_') or key=='equivalence_note':metrics.pop(key)
                    metrics.update(next_tier_cap_hits=0,retained_tail_cap_hits=0,full_tail_cap_hits=0)
                    row.update(metrics,skipped_reason='layer_not_selected',effective_mode='none')
            if observing:five.capture_spectrum(observation,name,singular,output,actual_rank)
            raw=singular.detach().cpu().contiguous()
            row.update(source_full_spectrum_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                corrected_full_spectrum_sha256=hashlib.sha256(output.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                correction_seconds=time.perf_counter()-tick)
            return output
        transforms[name]=transform
    if observing:
        model.decom_larger_model(rate,singular_value_transforms=transforms,svd_observers=five.svd_observers(observation))
    elif transforms:model.decom_larger_model(rate,singular_value_transforms=transforms)
    else:model.decom_larger_model(rate)
    return records


def bind(client,records,loop):
    for row in records:row['source_server_sha256']=client.download_hashes[row['model_id']]
    client.svd_spectral_latest=records
    if sampled(client.args,loop):
        if not hasattr(client,'svd_spectral_records'):client.svd_spectral_records=[]
        client.svd_spectral_records.extend(records)


def rows(server):return [r for client in server.clients for r in getattr(client,'svd_spectral_records',[])]


def manifest(server):
    if getattr(server.args, 'multi_svd_spectral_mode', 'none') == 'coverage_reorder':
        return dict(policy=policy(server.args), shared_download_table='fc1_download_spectrum',
                    shared_download_records=len(server.fc1_download_records))
    records=rows(server)
    return dict(policy=policy(server.args),sampled_download_layer_records=len(records),
        correction_seconds=sum(r.get('correction_seconds',0.) for r in records),
        same_mother_saturated_identical_records=sum(bool(r.get('counterfactual_both_entire_interval_at_cap')) and bool(r.get('counterfactual_same_full_corrected_spectrum')) for r in records))


def persist(server):
    if getattr(server.args,'multi_svd_spectral_mode','none') == 'coverage_reorder':return
    if not enabled(server.args) or not sampled(server.args,server.loop_round):return
    records=[r for c in server.clients for r in getattr(c,'svd_spectral_latest',[]) if r['loop_round']==server.loop_round]
    if len(records)!=server.num_clients*len(factorized_layers(server.args)):
        raise RuntimeError('Incomplete all-client/layer spectral download receipts')
    with (Path(server.report_dir())/'svd_spectral_downloads.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(loop_round=server.loop_round,completed_aggregations=server.loop_round,
            upcoming_completed_train_count=server.loop_round+1,downloads=records),allow_nan=False)+'\n')
    identical=sum(bool(r.get('counterfactual_both_entire_interval_at_cap')) and bool(r.get('counterfactual_same_full_corrected_spectrum')) for r in records if r['selected'])
    if server.args.multi_svd_spectral_mode in ('all_tail','first_band'):
        print(f'[SVDSpectral] after_aggregation={server.loop_round} upcoming_R={server.loop_round+1} mode={server.args.multi_svd_spectral_mode} '
              f'layers={layer_names(server.args)} max_gain=not_applied; theoretical gain=1/actual_Avg_contributor_mass; '
              'actual gains include forward nonincreasing clamp; minimum retained prefix unchanged')
        return
    gain=server.args.multi_svd_spectral_max_gain
    print(f'[SVDSpectral] after_aggregation={server.loop_round} upcoming_R={server.loop_round+1} mode={server.args.multi_svd_spectral_mode} '
          f'layers={layer_names(server.args)} max_gain={gain}; same-mother A/B fully capped & identical={identical} layer-downloads')
