"""Read-only prediction-capacity sweep; never imported by a federated trainer."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch

from common.client_data import audit_image_clients,read_image_client
from common.experiment import write_json,sha256,environment,code_version
from utils import multi_group_svd as svd,fc1_tail_diagnostic as old
from utils.cnn512_reporting import model_digest
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_rank_partition import protected_rng
from utils.whole_network_spectrum_diagnostic import truncated_model

LAYERS=('conv2','fc1','fc2')
DATA_FIELDS=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')


def integer_ids(text):
    try:ids=[int(v.strip()) for v in text.split(',')]
    except (ValueError,AttributeError):raise ValueError('client-ids must be comma-separated integers')
    if not ids or len(set(ids))!=len(ids) or any(i<0 for i in ids):raise ValueError('client-ids must be unique nonnegative integers')
    return sorted(ids)


def rates(text):
    try:values=[float(v.strip()) for v in text.split(',')]
    except (ValueError,AttributeError):raise ValueError('rank-rates must be comma-separated numbers')
    if not values or len(set(values))!=len(values) or any(not math.isfinite(v) or not 0<v<=.9 for v in values):
        raise ValueError('Sweep v1 requires unique finite 0<rank_rate<=native .90')
    return [.9]+[v for v in values if v!=.9]  # mandatory numerical round-trip control, never native reference


def resolve_inputs(cli,ids):
    files={};missing=[];diagnostics={}
    if cli.checkpoint:
        for item in cli.checkpoint:
            if '=' in item:
                cid,path=item.split('=',1);cid=int(cid)
            elif len(ids)==1:cid,path=ids[0],item
            else:raise ValueError('Use --checkpoint CLIENT_ID=PATH for multiple clients')
            if cid not in ids or cid in files:raise ValueError('Explicit checkpoint IDs must match selected client-ids exactly')
            files[cid]=Path(path).expanduser().resolve()
        for cid in ids:
            if cid not in files or not files[cid].is_file():missing.append(f'C{cid}: '+str(files.get(cid,'missing --checkpoint assignment')))
    else:
        root=Path(cli.source_run).expanduser().resolve()
        for cid in ids:
            name=f'Client_{cid}_model_{cli.model_id}.pt'
            direct=[root/name] if (root/name).is_file() else []
            final=sorted((root/'final_models').rglob(name)) if (root/'final_models').is_dir() else []
            others=sorted(root.rglob(name)) if root.is_dir() else []
            candidates=list(dict.fromkeys(p.resolve() for p in direct+final+others));valid=[];seen=[]
            for path in candidates:
                payload=torch.load(path,map_location='cpu',weights_only=False)
                status={k:payload.get(k) for k in ('client_id','model_id','completed_train_count','role')};seen.append(dict(path=str(path),**status))
                if status==dict(client_id=cid,model_id=cli.model_id,completed_train_count=cli.completed_round,role='ordinary_post_local_low_rank'):
                    priority=0 if path in direct else 1 if path in final else 2;valid.append((priority,path))
                del payload
            if not valid:missing.append(f'C{cid}: expected R{cli.completed_round} {root}/**/{name}');diagnostics[str(cid)]=seen
            else:
                priority=min(v[0] for v in valid);best=[v[1] for v in valid if v[0]==priority]
                if len(best)>1 and len({sha256(p) for p in best})>1:raise ValueError(f'C{cid}: ambiguous same-round files; use --checkpoint: {best}')
                files[cid]=best[0]
    if missing:raise FileNotFoundError('Missing requested post-local checkpoints (no other-round fallback):\n'+'\n'.join(missing)+'\nObserved candidates: '+json.dumps(diagnostics))
    manifests={cid:Path(cli.manifest).expanduser().resolve() if cli.manifest else file.parent/'manifest.json' for cid,file in files.items()}
    absent=[str(p) for p in manifests.values() if not p.is_file()]
    if absent:raise FileNotFoundError('Missing same-run manifest; use --manifest explicitly: '+str(sorted(set(absent))))
    if len({sha256(p) for p in manifests.values()})!=1:raise ValueError('Selected checkpoints have different run manifests')
    return files,next(iter(manifests.values()))


def audit_source(cli,manifest,ids):
    m=manifest;a=SimpleNamespace(**copy.deepcopy(m['effective']))
    if (cli.dataset!='Cifar100' or a.num_classes!=100 or a.model_family!='Decom_CNN-5-512'
        or any(i>=a.num_clients for i in ids) or m['actual_train_count']<cli.completed_round):
        raise ValueError('Unsupported source model/data/client range or manifest predates requested round')
    a.data_root=cli.data_root;a.dataset=cli.dataset;a.data_subdir=cli.data_subdir;a.data_formats='npz_only'
    a.data_expected_partition=m['data_manifest']['config']['partition'];a.cnn512_control=True
    audit=audit_image_clients(a)
    actual=[{k:r[k] for k in DATA_FIELDS} for r in audit['files']]
    expected=[{k:r[k] for k in DATA_FIELDS} for r in m['data_manifest']['files']]
    if audit['config_sha256']!=m['data_manifest']['config_sha256'] or actual!=expected:
        raise ValueError('Actual config/files/decoded tensor fingerprints/counts differ from source run manifest')
    return a,audit,digest_json(dict(config_sha256=audit['config_sha256'],files=actual))


def validate_checkpoint(model,p,cid,cli,manifest,data_fp):
    if tuple(getattr(model,'factorized_layers',LAYERS)) != LAYERS:
        raise ValueError('Whole-network capacity sweep requires conv2,fc1,fc2 factorization; fc1-only runs provide fc1_download_spectrum instead')
    if (p.get('role')!='ordinary_post_local_low_rank' or not p.get('prediction_checkpoint') or p.get('client_id')!=cid
        or p.get('model_id')!=cli.model_id or p.get('completed_train_count')!=cli.completed_round
        or p.get('loop_round',cli.completed_round-1)!=cli.completed_round-1 or model.ratio_LR!=.9):
        raise ValueError(f'C{cid}: require selected-round .90 ordinary post-local prediction checkpoint, not a server/upload/other capacity')
    if p.get('model_family')!=manifest['effective']['model_family']:raise ValueError(f'C{cid}: checkpoint model family differs from manifest')
    if (model.conv1.in_channels!=3 or model.conv1.out_channels!=16 or model.fc1.in_features!=800
        or model.fc2.in_features!=model.fc1.out_features or model.fc2.out_features!=512 or model.fc3.out_features!=100):
        raise ValueError('Sweep v1 supports RGB32 Hyper_CNN_512 with original convolution channels and 512 features')
    if model.fc1.out_features!=manifest.get('model_kwargs',{}).get('hidden_width',manifest['effective'].get('cnn_hidden_width',2000)):
        raise ValueError(f'C{cid}: checkpoint width differs from run configuration')
    for name,t in model.state_dict().items():
        if t.dtype!=p['state_dict'][name].dtype or not torch.isfinite(t).all() or not torch.equal(t,p['state_dict'][name]):raise ValueError('Checkpoint dtype/nonfinite/alias mismatch: '+name)
    for key,mkey in [('initial_parameter_sha256','initial_parameter_sha256'),('group_fingerprint',None),('capacity_protocol','svd_capacity_protocol')]:
        expected=manifest.get('grouping',{}).get('group_fingerprint') if mkey is None else manifest.get(mkey)
        if key in p and p[key]!=expected:raise ValueError(f'C{cid}: checkpoint/manifest {key} mismatch')
    usage=p.get('single_avg_prototype_usage') or {};context=usage.get('downloaded_model') or {};bindings=[]
    for label,fields in [('checkpoint',p),('download context',context)]:
        if fields.get('data_fingerprint') is not None:
            if fields['data_fingerprint']!=data_fp:raise ValueError(f'C{cid}: {label} data fingerprint mismatch')
            bindings.append(label)
    return dict(data_binding='direct' if bindings else 'independent_same_run_manifest_audit',direct_binding_locations=bindings,
        checkpoint_has_top_level_data_fingerprint='data_fingerprint' in p,download_context_is_null=usage.get('downloaded_model') is None,
        missing_data_binding_is_not_fabricated=True)


class ObservedDataset:
    def __init__(self,data):self.data=data;self.indices=[]
    def __len__(self):return len(self.data)
    def __getitem__(self,index):self.indices.append(index);return self.data[index]


def evaluate(model,data,batch,device):
    observed=ObservedDataset(data);row,logits=old.evaluate(model,observed,batch,device)
    if observed.indices!=list(range(len(data))):raise RuntimeError('Evaluation did not visit each test sample exactly once in original order')
    if not torch.isfinite(logits).all() or not math.isfinite(row['ce']):raise FloatingPointError('Nonfinite sweep logits/CE')
    return dict(row,sample_order_sha256=digest_json(observed.indices)),logits


@torch.no_grad()
def shared_download(native,max_rate):
    receipts=[]
    with protected_rng([native]):
        parameter=next(native.parameters());device=parameter.device;prediction_dtype=parameter.dtype
        # A trained model can amplify float32 recovery/SVD rounding even with
        # gesvd. Promote BEFORE recovery; its original layer constructors must
        # also allocate float64, otherwise copy_ silently rounds back to float32.
        compute_dtype=torch.float64 if device.type=='cuda' else prediction_dtype
        svd_driver='gesvd' if device.type=='cuda' else None
        previous_dtype=torch.get_default_dtype()
        try:
            if device.type=='cuda':torch.set_default_dtype(compute_dtype)
            mother=copy.deepcopy(native).to(dtype=compute_dtype)
            mother.recover_larger_model();mother.ratio_LR=1.;mother.to(device=device,dtype=compute_dtype)
            balanced=copy.deepcopy(mother)
            def observe(name,layer,u,s,v):
                receipts.append(dict(layer=name,retained_rank=layer.rank,svd_driver=svd_driver,svd_device=str(s.device),
                    svd_compute_dtype=str(s.dtype),prediction_dtype=str(prediction_dtype),
                    SVD_sha256=tensor_fingerprint(dict(U=u,S=s,Vh=v)),singular_values_sha256=tensor_fingerprint(dict(S=s))))
            balanced.decom_larger_model(max_rate,svd_driver=svd_driver,
                svd_observers={name:(lambda layer,u,s,v,name=name:observe(name,layer,u,s,v)) for name in LAYERS})
        finally:
            torch.set_default_dtype(previous_dtype)
        balanced.ratio_LR=max_rate;balanced.to(device=device,dtype=prediction_dtype)
    if len(receipts)!=3 or {r['layer'] for r in receipts}!=set(LAYERS):raise RuntimeError('Need exactly one original download SVD per layer')
    return balanced,receipts


def threshold_report(rows,limit):
    ordered=sorted(rows,key=lambda r:r['parameter_count']);good=[r for r in ordered if r['drop_pp']<=limit]
    chosen=good[0] if good else None
    neighbors=[]
    if chosen:
        i=ordered.index(chosen);neighbors=ordered[max(0,i-1):min(len(ordered),i+2)]
    return dict(reference_drop_pp=limit,minimum_tested_feasible=chosen,adjacent_tested_points=neighbors,
        adjacent_crossing_intervals=[dict(lower_rate=a['rank_rate'],upper_rate=b['rank_rate'],parameter_bounds=[a['parameter_count'],b['parameter_count']],
            observed_drop_pp_endpoints=[a['drop_pp'],b['drop_pp']]) for a,b in zip(ordered,ordered[1:]) if (a['drop_pp']<=limit)!=(b['drop_pp']<=limit)],
        interpretation='Minimum among MEASURED ranks only. No monotonicity or unmeasured accuracy assumption; negative drop means improvement.')


def summarize(rows,ids,limit):
    reference={r['client_id']:r for r in rows if r['variant']=='native'};output=[]
    if set(reference)!=set(ids):raise ValueError('Missing native references')
    for rate in sorted({r['rank_rate'] for r in rows if r['variant']=='resvd'},reverse=True):
        chosen=[r for r in rows if r['variant']=='resvd' and r['rank_rate']==rate]
        if {r['client_id'] for r in chosen}!=set(ids) or len(chosen)!=len(ids):raise ValueError('Incomplete rank cohort')
        if len({r['parameter_count'] for r in chosen})!=1:raise ValueError('Mixed model structures in source cohort')
        n=sum(r['samples'] for r in chosen);native_correct=sum(reference[i]['correct'] for i in ids)
        if any(reference[r['client_id']]['samples']!=r['samples'] or reference[r['client_id']]['sample_order_sha256']!=r['sample_order_sha256'] for r in chosen):raise ValueError('Native/swept evaluation sample mismatch')
        correct=sum(r['correct'] for r in chosen)
        output.append(dict(source=chosen[0]['source'],completed_round=chosen[0]['completed_round'],rank_rate=rate,
            client_ids=ids,parameter_count=chosen[0]['parameter_count'],hidden_width=chosen[0]['hidden_width'],
            correct=correct,samples=n,accuracy=correct/n,native_accuracy=native_correct/n,drop_pp=100*(native_correct-correct)/n,
            accuracy_unit='fraction',drop_unit='percentage points',weighted_by='test samples'))
    return dict(cohort=output,cohort_threshold=threshold_report(output,limit),
        client_thresholds={str(i):threshold_report([r for r in rows if r['client_id']==i and r['variant']=='resvd'],limit) for i in ids})


def render(root):
    with (root/'render.log').open('w',encoding='utf-8') as stream:
        r=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'plot_svd_capacity_sweep.py'),'--output-dir',str(root)],stdout=stream,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError('Plot failed; tables retained, see '+str(root/'render.log'))


def run(cli):
    ids=integer_ids(cli.client_ids);effective_rates=rates(cli.rank_rates)
    if cli.completed_round<1 or cli.model_id<0 or cli.batch_size<1 or cli.threads<1:raise ValueError('Round/batch/threads must be positive; model-id nonnegative')
    if any(not math.isfinite(v) or v<0 for v in (cli.max_drop_pp,cli.roundtrip_atol,cli.roundtrip_rtol)):raise ValueError('Threshold/tolerances must be finite nonnegative')
    torch.set_num_threads(cli.threads);files,manifest_file=resolve_inputs(cli,ids)
    source_manifest=json.loads(manifest_file.read_text(encoding='utf-8'));args,audit,data_fp=audit_source(cli,source_manifest,ids)
    device='cpu' if cli.device=='cpu' else 'cuda:0'
    if cli.device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise ValueError('Requested GPU unavailable; use --device cpu explicitly')
    backend=source_manifest.get('environment',{})
    for key,owner,attr in [('cudnn_deterministic',torch.backends.cudnn,'deterministic'),('cudnn_benchmark',torch.backends.cudnn,'benchmark'),
        ('matmul_tf32',torch.backends.cuda.matmul,'allow_tf32'),('cudnn_tf32',torch.backends.cudnn,'allow_tf32')]:
        if key in backend:setattr(owner,attr,backend[key])
    if 'deterministic' in backend:torch.use_deterministic_algorithms(backend['deterministic'])
    for cid,path in files.items():
        model,payload=svd.load_checkpoint(path,'cpu');validate_checkpoint(model,payload,cid,cli,source_manifest,data_fp)
        del model,payload
    root=Path(cli.output_dir).expanduser().resolve()
    protected=[manifest_file.parent,Path(audit['directory'])]
    if any(root==p or p in root.parents or root in p.parents for p in protected):raise ValueError('Output must be independent of input checkpoints/data')
    root.mkdir(parents=True,exist_ok=False)
    hashes={p:sha256(p) for p in list(files.values())+[manifest_file,Path(audit['config_path'])]}
    hashes.update({Path(r['path']):r['sha256'] for r in audit['files']})
    manifest=dict(status='running',role='offline_post_local_whole_network_capacity_sweep',requested=vars(cli),
        effective_rates=effective_rates,mandatory_roundtrip_rate=.9,client_ids=ids,data_fingerprint=data_fp,source_manifest=str(manifest_file),
        source_manifest_sha256=hashes[manifest_file],source_initial_sha256=source_manifest.get('initial_parameter_sha256'),
        checkpoint_files={str(i):str(p) for i,p in files.items()},environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),
        source_training=dict(algorithm=source_manifest.get('algorithm'),aggregation=source_manifest.get('aggregation_mode'),
            spectral_mode=source_manifest['effective'].get('multi_svd_spectral_mode','none'),manifest_completed_round=source_manifest['actual_train_count']),
        svd_driver='gesvd' if cli.device=='cuda' else None,svd_driver_policy='offline sweep only: CUDA gesvd, CPU default; federated training defaults unchanged',
        reconstruction_precision_policy='CUDA recovery and SVD in float64, factors cast back to checkpoint dtype before all inference; CPU unchanged',
        trained=False,prediction='original classifier; native .90 reference distinct from re-SVD .90',parameter_count='all unique model parameters including factors/bias/classifier, not a communication or peak-memory measure')
    write_json(root/'manifest.json',manifest);write_json(root/'data_manifest.json',audit)
    rows=[];checks=[];svds=[];source_width=None
    try:
        for cid,path in files.items():
            print(f'Sweep {cli.source_name} R{cli.completed_round} C{cid}: native .90 then {effective_rates}',flush=True)
            native,p=svd.load_checkpoint(path,device);provenance=validate_checkpoint(native,p,cid,cli,source_manifest,data_fp)
            width=native.fc1.out_features
            if source_width is not None and source_width!=width:raise ValueError('Mixed checkpoint widths in one source run')
            source_width=width;native_hash=model_digest(native);before_rng=old.rng_fingerprint(old.rng_snapshot(native))
            data=read_image_client(args,cid,False);start=time.perf_counter();reference,reference_logits=evaluate(native,data,cli.batch_size,device)
            identity=dict(source=cli.source_name,completed_round=cli.completed_round,client_id=cid,model_id=cli.model_id,hidden_width=width,
                checkpoint=str(path),checkpoint_sha256=hashes[path],data_fingerprint=data_fp)
            def result(model,rate,variant,point):
                return dict(identity,variant=variant,rank_rate=rate,actual_ranks={n:getattr(model,n).rank for n in LAYERS},
                    parameter_count=sum(p.numel() for p in model.parameters()),**point,drop_pp=100*(reference['accuracy']-point['accuracy']))
            rows.append(result(native,.9,'native',reference));balanced,receipts=shared_download(native,max(effective_rates))
            shared_hash=digest_json(receipts);svds.append(dict(client_id=cid,svd_driver=receipts[0]['svd_driver'],
                svd_compute_dtype=receipts[0]['svd_compute_dtype'],prediction_dtype=receipts[0]['prediction_dtype'],
                shared_SVD_sha256=shared_hash,layers=receipts,native_download_SVD_calls=3))
            for rate in effective_rates:
                model=truncated_model(balanced,rate);point,logits=evaluate(model,data,cli.batch_size,device)
                row=result(model,rate,'resvd',point);row['shared_SVD_sha256']=shared_hash;rows.append(row)
                if rate==.9:
                    delta=(logits-reference_logits).double()
                    check=dict(identity,**provenance,svd_driver=receipts[0]['svd_driver'],
                        svd_compute_dtype=receipts[0]['svd_compute_dtype'],prediction_dtype=receipts[0]['prediction_dtype'],
                        roundtrip_logit_max_absolute_error=float(delta.abs().max()),
                        roundtrip_logit_relative_Frobenius_error=float(delta.norm()/reference_logits.double().norm().clamp_min(1e-30)),
                        prediction_change_count=int((logits.argmax(1)!=reference_logits.argmax(1)).sum()),
                        tolerance_atol=cli.roundtrip_atol,tolerance_rtol=cli.roundtrip_rtol,
                        passed=bool(torch.allclose(logits,reference_logits,atol=cli.roundtrip_atol,rtol=cli.roundtrip_rtol)))
                    checks.append(check);write_json(root/'roundtrip_checks.json',checks)
                    if not check['passed']:
                        raise ValueError(f"Unexpected .90 roundtrip logits error: max_abs={check['roundtrip_logit_max_absolute_error']:.8g}, "
                            f"relative_Frobenius={check['roundtrip_logit_relative_Frobenius_error']:.8g}, "
                            f"prediction_changes={check['prediction_change_count']}, atol={cli.roundtrip_atol}, rtol={cli.roundtrip_rtol}; "
                            'see roundtrip_checks.json (tolerances never auto-relaxed)')
                del model,logits
            if model_digest(native)!=native_hash or old.rng_fingerprint(old.rng_snapshot(native))!=before_rng:raise RuntimeError('Sweep changed source model or RNG')
            checks[-1].update(source_model_and_RNG_unchanged=True,client_sweep_seconds=time.perf_counter()-start)
            del native,p,balanced,data,reference_logits
        summary=summarize(rows,ids,cli.max_drop_pp)
        for name,values in [('client_results',rows),('cohort_results',summary['cohort'])]:write_json(root/(name+'.json'),values);old.write_csv(root/(name+'.csv'),values)
        write_json(root/'summary.json',summary);write_json(root/'roundtrip_checks.json',checks);write_json(root/'shared_svd.json',svds)
        render(root)
        if any(sha256(p)!=value for p,value in hashes.items()):raise RuntimeError('Source artifact/data changed during sweep')
        manifest.update(status='complete',hidden_width=source_width,native_download_SVD_calls=3*len(ids),input_files_unchanged=True,
            svd_compute_dtypes=sorted({r['svd_compute_dtype'] for r in svds}),prediction_dtypes=sorted({r['prediction_dtype'] for r in svds}),
            all_variants_same_samples=True,model_exports=0,plot='parameters_vs_drop.png')
        write_json(root/'manifest.json',manifest)
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));write_json(root/'manifest.json',manifest);raise


def compare_results(cli):
    root=Path(cli.output_dir).expanduser().resolve();manifests=[];rows=[];summaries={}
    for folder in cli.compare_results:
        folder=Path(folder).expanduser().resolve();m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if m['status']!='complete' or m['role']!='offline_post_local_whole_network_capacity_sweep':raise ValueError('Require completed sweep results')
        if manifests and any(m[k]!=manifests[0][k] for k in ('client_ids','data_fingerprint')):raise ValueError('Paired sources must use identical clients and audited data')
        if manifests and m['requested']['completed_round']!=manifests[0]['requested']['completed_round']:raise ValueError('Paired source rounds differ')
        name=m['requested']['source_name']
        if name in summaries:raise ValueError('Use distinct source-name values')
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));summaries[name]=summary;manifests.append(m)
        rows.extend(summary['cohort'])
    paired=[];baseline=manifests[0]['requested']['source_name']
    first={r['rank_rate']:r for r in summaries[baseline]['cohort']}
    for name,summary in summaries.items():
        if name==baseline:continue
        for row in summary['cohort']:
            if row['rank_rate'] not in first:continue
            ref=first[row['rank_rate']]
            if row['samples']!=ref['samples']:raise ValueError('Paired test sample totals differ')
            paired.append(dict(contrast=name+'_minus_'+baseline,completed_round=row['completed_round'],client_ids=row['client_ids'],rank_rate=row['rank_rate'],
                first_parameters=ref['parameter_count'],second_parameters=row['parameter_count'],
                accuracy_difference_pp=100*(row['accuracy']-ref['accuracy']),native_accuracy_difference_pp=100*(row['native_accuracy']-ref['native_accuracy']),
                drop_difference_pp=row['drop_pp']-ref['drop_pp']))
    root.mkdir(parents=True,exist_ok=False);old.write_csv(root/'cohort_results.csv',rows);write_json(root/'cohort_results.json',rows)
    old.write_csv(root/'paired_comparisons.csv',paired);write_json(root/'paired_comparisons.json',paired)
    write_json(root/'summary.json',summaries);write_json(root/'manifest.json',dict(role='paired_offline_sweep_comparison',sources=manifests,trained=False))
    render(root)
