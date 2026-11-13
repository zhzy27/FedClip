"""Real D01 main, original CNN width, five capacities, N20/K1/2 rounds.

Synthetic fixture ONLY. Independent scalar OLS/oracle, frozen 444ad38 reference,
and actual batch/RNG traces. Does not start or alter a real-data experiment.
"""
import argparse
from contextlib import ExitStack
import copy
import csv
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def verify_csv_exports(root):
    """Check saved CSV against saved JSON, including nested per-rank details."""
    assert (Path(root)/'primary_metrics.json').is_file(),'Not a completed acceptance run'
    checked=[]
    for name in ('primary_metrics','capacity_metrics','svd_spectral_downloads','rank_spectrum'):
        file=Path(root)/(name+'.json')
        if not file.is_file():continue
        expected=json.loads(file.read_text(encoding='utf-8'))
        with file.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as stream:
            saved=list(csv.DictReader(stream))
        assert len(saved)==len(expected),name
        for actual,row in zip(saved,expected):
            for key,cell in actual.items():
                value=row.get(key)
                if isinstance(value,(list,dict)):assert json.loads(cell)==value,(name,key)
                else:assert cell==('' if value is None else str(value)),(name,key)
        checked.append(name)
    return checked


def independent_curve(s,k,b,mode,gain,window):
    # No production helper and no NumPy LAPACK dependency. This host's NumPy
    # lstsq has a DLL failure; retain an independent closed-form OLS oracle.
    raw=s.detach().cpu().double().tolist();out=list(raw);start=max(0,k-window)
    points=[(float(i+1),math.log(raw[i])) for i in range(start,k) if raw[i]>0]
    if len(points)<2 or raw[k-1]==0:return s.clone()
    mx=math.fsum(x for x,y in points)/len(points);my=math.fsum(y for x,y in points)/len(points)
    slope=min(math.fsum((x-mx)*(y-my) for x,y in points)/math.fsum((x-mx)**2 for x,y in points),0.)
    target=[raw[k-1]*math.exp(slope*(j-k)) for j in range(k+1,len(raw)+1)]
    if mode=='single_point':
        if raw[k]>0:
            g=min(gain,max(1.,target[0]/raw[k]));out[k:]=[v*g for v in raw[k:]]
    else:
        for i in range(k,b):out[i]=max(raw[i],min(target[i-k],gain*raw[i]))
        if raw[b-1]>0:
            g=min(gain,max(1.,out[b-1]/raw[b-1]));out[b:]=[v*g for v in raw[b:]]
    import torch
    return torch.tensor(out,dtype=s.dtype,device=s.device)


def worker(out,label):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from common.experiment import rng_state,set_rng
    from fedfews_compat.observe import rng_digest
    from utils.cnn512_reporting import model_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils import multi_svd_spectral as sp,multi_group_svd as svd
    from utils.group_projection import aggregate_group_updates
    import flcore.clients.clientMultiGroupSVD as cs
    import flcore.clients.clientMultiGroupProj as cm
    import flcore.servers.serverMultiGroupProj as sm
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    sc,cc=sm.FedMultiGroupProj,cs.clientMultiGroupSVD
    if job.get('frozen'):
        for path,name in [('system/flcore/clients/clientMultiGroupSVD.py','clientMultiGroupSVD'),('system/flcore/servers/serverMultiGroupProj.py','FedMultiGroupProj')]:
            source=subprocess.check_output(['git','show',f'444ad38:{path}'],cwd=ROOT,text=True,encoding='utf-8')
            ns={'__name__':'frozen_444ad38_'+name};exec(compile(source,'444ad38/'+path,'exec'),ns)
            if name=='clientMultiGroupSVD':cc=ns[name]
            else:sc=ns[name]
    ctx=dict(training=False,download=False);events=[];batches=[];created=[];steps=[0];calls=[0];evidence={};errors=[]
    native_svd=torch.linalg.svd;native_step=torch.optim.SGD.step
    def SVD(matrix,*a,**kw):
        values=native_svd(matrix,*a,**kw)
        if ctx['download']:
            names={(160,80):'conv2',(2000,800):'fc1',(512,2000):'fc2'}
            name=names[tuple(matrix.shape)];evidence[name]=tuple(v.detach().cpu().clone() for v in values);calls[0]+=1
        return values
    def step(opt,*a,**kw):
        if ctx['training']:steps[0]+=1
        return native_step(opt,*a,**kw)
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if ctx['training']:batches.append([ctx['loop'],ctx['cid'],x[:,0,0,0].tolist()])
                yield x,y
    native_init=sc.__init__
    def initialize(server,*a,**kw):
        native_init(server,*a,**kw);created.append(server)
        assert server.K==1 and server.groups==[list(range(20))]
        events.append(dict(kind='initial',weights=server.initial_hash,RNG=rng_digest()))
        for c in server.clients:
            native_download=c.set_parameters;native_train=c.train
            def download(servers,loop,c=c,native_download=native_download):
                ctx.update(download=True,cid=c.id,loop=loop);evidence.clear()
                try:native_download(servers,loop)
                finally:ctx['download']=False
                bounds=sp.boundary_ranks(servers[0],c.args);local=c.model_set[0]
                for name in sp.LAYERS:
                    layer=getattr(local,name);r=layer.rank;k,b=bounds[name]['k'],bounds[name]['b'];p,s,q=evidence[name]
                    active=job.get('mode','none')!='none' and loop>=1 and r>k and name in job.get('layers','conv2,fc1,fc2').split(',')
                    target=independent_curve(s,k,b,job['mode'],1.5,sp.WINDOWS[name]) if active else s
                    root=target[:r].sqrt();expected_u=p[:,:r]@torch.diag(root);expected_v=torch.diag(root)@q[:r]
                    u=layer.conv_u if name=='conv2' else layer.weight_u;v=layer.conv_v if name=='conv2' else layer.weight_v
                    if not active:
                        assert torch.equal(u,expected_u) and torch.equal(v,expected_v),(name,loop,c.id)
                    else:
                        assert torch.allclose(u,expected_u,atol=1e-6,rtol=1e-5) and torch.allclose(v,expected_v,atol=1e-6,rtol=1e-5)
                        errors.append(max(float((u-expected_u).abs().max()),float((v-expected_v).abs().max())))
                        original_root=s[:r].sqrt()
                        assert torch.equal(u[:,:k],(p[:,:r]@torch.diag(original_root))[:,:k])
                        assert torch.equal(v[:k],(torch.diag(original_root)@q[:r])[:k])
                    assert torch.equal(layer.bias,getattr(servers[0],name).bias)
                for name in ('conv1','fc3'):
                    for p,q in zip(getattr(local,name).parameters(),getattr(servers[0],name).parameters()):assert torch.equal(p,q)
                assert [id(p) for p in local.parameters()]==[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']]
                events.append(dict(kind='download',loop=loop,cid=c.id,weights=model_digest(local),RNG=rng_digest()))
            c.set_parameters=download
            def train(c=c,native_train=native_train):
                ctx.update(training=True,cid=c.id,loop=server.loop_round)
                try:result=native_train()
                finally:ctx['training']=False
                row=c.train_stats[0];assert row['train_batches']==5 and row['processed_train_samples']==5*c.train_samples
                assert row['effective_proto_lambda']==0 and c.avg_proto_local is None
                events.append(dict(kind='local',loop=server.loop_round,cid=c.id,weights=model_digest(c.model_set[0]),
                    upload=tensor_fingerprint(c.full_uploads[0]),RNG=rng_digest()))
                return result
            c.train=train
        native_aggregate=server.aggregate_parameters
        def aggregate():
            base=dict(server.global_model_set[0].named_parameters());total=sum(c.train_samples for c in server.clients)
            weights={c.id:c.train_samples/total for c in server.clients}
            factory=lambda:iter((c.id,weights[c.id],c.full_upload_parameters(0,'cpu')) for c in server.selected_clients)
            expected=aggregate_group_updates(base,factory,weights,list(range(20)),'group_avg_only')[0]
            result=native_aggregate()
            for name,p in server.global_model_set[0].named_parameters():assert torch.equal(p,expected[name]),name
            events.append(dict(kind='aggregate',loop=server.loop_round,weights=model_digest(server.global_model_set[0]),RNG=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    sys.argv=['main.py','-did','-1','-dev','cpu','-algo','FedMultiGroupProj','-data','Cifar100','-m','Decom_CNN-5-512',
        '-t','1','-pv','0','-nc','20','-ncl','100','-nsm','1','-jr','1','--seed','0','-lr','.005','-lbs','16','-ls','5','-gr','1','-eg','1',
        '-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),'--data-subdir','fixture','--data-formats','npz_only',
        '--multi_avg_proto_ablation','--multi_svd','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.90',
        '--multi_group_basis','full_weights','--multi_group_mode','group_avg_only','--multi_group_sizes','20','--multi_group_weight','1',
        '--multi_proto_mode','none','--multi_proto_lambda','0','--multi_proto_aggregation','class_sample','--multi_proto_loss','mse',
        '--multi_proto_feature','base_output','--multi_proto_target_type','raw_mean','--multi_svd_aggregation','full_avg',
        '--multi_fc1_tail_scale_strength','0','--multi_rank_spectrum',str(job.get('spectrum',1)),
        '-is_regular','1','-regular_lamda','.001','--local_grad_clip','10','--control_test_batch_size','16','-clip_cpu_threads','2',
        '-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),'--h5_result_root',str(run/'h'),'--final-model-root',str(run/'f')]
    if 'mode' in job:sys.argv+=['--multi_svd_spectral_mode',job['mode'],'--multi_svd_spectral_layers',job.get('layers','conv2,fc1,fc2')]
    with ExitStack() as stack:
        for obj,name,value in [(sm,'FedMultiGroupProj',sc),(cs,'clientMultiGroupSVD',cc),(sc,'__init__',initialize),
            (cm,'DataLoader',TraceLoader),(torch.linalg,'svd',SVD),(torch.optim.SGD,'step',step)]:stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];root=Path(server.report_dir());summary=json.loads((root/'primary_summary.json').read_text())
    verify_csv_exports(root)
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text());values=[r['post_local_accuracy'] for r in server.primary_records]
    assert summary==manifest['primary_summary'] and summary['best']==max(values) and summary['best_loop_round']==values.index(max(values))
    assert summary['final']==values[-1] and summary['last10']==summary['last20']==sum(values)/len(values)
    for row in server.primary_records:
        members=[r for r in server.endpoint_records if r['loop_round']==row['loop_round'] and r['stage']=='post']
        assert row['post_local_accuracy']==sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
        assert row['pre_observed'] and row['member_ids']==list(range(20))
    for row in server.capacity_records:
        members=[r for r in server.endpoint_records if r['loop_round']==row['loop_round'] and r['stage']=='post' and r['client_id'] in row['member_ids']]
        assert row['post_local_accuracy']==sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
    assert steps[0]==len(batches)==200 and calls[0]==120 and len(server.capacity_records)==10
    receipts=sp.rows(server)
    if job.get('mode','none')!='none':
        assert len(receipts)==120
        assert all(not r['changed'] and r['skipped_reason']=='initial_download_original_path' for r in receipts if r['loop_round']==0)
        assert all(not r['changed'] for r in receipts if r['actual_rank']==r['k'])
        assert json.loads((root/'svd_spectral_downloads.json').read_text())==receipts
        assert len((root/'svd_spectral_downloads.jsonl').read_text().splitlines())==2
        for name in sp.LAYERS:
            active=[r for r in receipts if r['layer']==name and r['loop_round']==1 and r['actual_rank']>r['k']]
            assert len({r['corrected_full_spectrum_sha256'] for r in active})==1
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values and json.loads(hf.attrs['svd_spectral_json'])==manifest['svd_spectral']
        if receipts:assert [json.loads(r) for r in hf['svd_spectral_downloads_json'][:]]==receipts
    if job.get('spectrum',1):
        spectra=json.loads((root/'rank_spectrum.json').read_text());assert len(spectra)==18
        assert {(r['layer'],r['boundary_k'],r['boundary_b']) for r in spectra}=={('conv2',12,20),('fc1',120,200),('fc2',77,128)}
        assert all(r['boundary_singular_values'] and r['next_tier_spectral_energy']>=0 for r in spectra)
    state=rng_state();final_rng=rng_digest()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(server.final_model_dir())/f'{role}_model_0.pt')
        old=server.global_model_set[0] if role=='Server' else server.clients[int(role.split('_')[1])].model_set[0]
        assert model_digest(model)==model_digest(old)
        with torch.no_grad():assert torch.equal(model(torch.zeros(1,3,32,32)),copy.deepcopy(old).eval()(torch.zeros(1,3,32,32)))
        if job.get('mode','none')!='none':
            assert payload['svd_spectral_policy']['mode']==job['mode'] if role=='Server' else payload['svd_spectral']['policy']['mode']==job['mode']
    set_rng(state)
    report=dict(status='PASS',job=job,events=events,batches=batches,final_rng=final_rng,primary=server.primary_records,summary=summary,
        initial_hash=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],SGD_steps=steps[0],SVD_calls=calls[0],models_reloaded=21,
        independent_balanced_factor_max_error=max(errors,default=0.))
    (run/'verification.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--resume-owned-output',action='store_true');args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    jobs=dict(single=dict(mode='single_point'),next=dict(mode='next_tier'),none=dict(mode='none'),
        frozen=dict(frozen=True),quiet=dict(mode='single_point',spectrum=0))
    if args.resume_owned_output:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Not this verifier\'s synthetic workspace')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        generator=np.random.default_rng(94);data=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(generator.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        if args.resume_owned_output and (out/label/'verification.json').is_file():continue
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Acceptance exit={result.returncode}: {out}/{label}.log')
        print('PASS',label,flush=True)
    reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    for key in ('events','batches','final_rng','primary','summary'):assert reports['none'][key]==reports['frozen'][key],key
    for key in ('events','batches','final_rng','primary','summary'):assert reports['single'][key]==reports['quiet'][key],key
    initial_round=lambda r:[e for e in r['events'] if e.get('loop',-1)==0]
    for report in reports.values():
        assert report['initial_hash']==reports['none']['initial_hash'] and report['data_fingerprint']==reports['none']['data_fingerprint']
        assert report['batches']==reports['none']['batches'] and report['final_rng']==reports['none']['final_rng']
        assert initial_round(report)==initial_round(reports['none'])
    result=dict(status='PASS_CPU',synthetic_only=True,N=20,K=1,completed_rounds=2,local_epochs=5,initialization_rank=.90,
        SVD_calls_per_case=120,SGD_steps_per_case=200,models_reloaded_per_case=21,
        none_vs_frozen_444ad38_tensors_uploads_aggregates_batches_RNG_exact=True,
        first_round_all_modes_vs_old_exact=True,minimum_and_unselected_layer_parameters_exact=True,
        independent_Python_OLS_balanced_factor_max_error=max(r['independent_balanced_factor_max_error'] for r in reports.values()),
        full_shared_curves_across_capacities_verified=True,diagnostics_toggle_trajectory_and_RNG_exact=True,
        ordinary_full_Avg_kernel_error=0.,weighted_primary_and_capacity_curves_exports_reload_verified=True,
        cases=list(jobs),real_accuracy_improvement_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
