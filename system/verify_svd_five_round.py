"""CPU synthetic actual-main acceptance. Never reads/generates real pat20 data."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import torch
    import h5py
    from torch.utils.data import DataLoader
    import flcore.clients.clientMultiGroupSVD as cs
    import flcore.clients.clientMultiGroupProj as cm
    import flcore.servers.serverMultiGroupProj as sm
    from fedfews_compat.observe import rng_digest
    from utils.cnn512_reporting import model_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils import multi_group_svd as svd,multi_svd_five_round as five
    from utils.group_projection import aggregate_group_updates
    from verify_multi_svd_spectral import verify_csv_exports
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label;events=[];batches=[];created=[]
    cls=sm.FedMultiGroupProj;clientcls=cs.clientMultiGroupSVD
    if job.get('frozen'):
        for file,name in [('system/flcore/clients/clientMultiGroupSVD.py','clientMultiGroupSVD'),('system/flcore/servers/serverMultiGroupProj.py','FedMultiGroupProj')]:
            source=subprocess.check_output(['git','show',f'b150128:{file}'],cwd=ROOT,text=True,encoding='utf-8')
            ns={'__name__':'frozen_b150128_'+name};exec(compile(source,file,'exec'),ns)
            if name=='FedMultiGroupProj':cls=ns[name]
            else:clientcls=ns[name]
    context=dict(training=False,downloading=False);svds=[0];steps=[0];native_svd=torch.linalg.svd;native_step=torch.optim.SGD.step
    def SVD(*p,**kw):
        if context['downloading']:svds[0]+=1
        return native_svd(*p,**kw)
    def step(opt,*p,**kw):
        if context['training']:steps[0]+=1
        return native_step(opt,*p,**kw)
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append([context['loop'],context['cid'],x[:,0,0,0].tolist()])
                yield x,y
    native_init=cls.__init__
    def initialize(server,*p,**kw):
        native_init(server,*p,**kw);created.append(server)
        events.append(dict(kind='initial',sha=server.initial_hash,rng=rng_digest()))
        for c in server.clients:
            download=c.set_parameters;train=c.train
            def down(models,loop,c=c,download=download):
                context.update(downloading=True,cid=c.id,loop=loop)
                try:download(models,loop)
                finally:context['downloading']=False
                local=c.model_set[0]
                assert [id(p) for p in local.parameters() if p.requires_grad]==[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']]
                events.append(dict(kind='download',loop=loop,cid=c.id,sha=model_digest(local),rng=rng_digest()))
            c.set_parameters=down
            def local(c=c,train=train):
                context.update(training=True,cid=c.id,loop=server.loop_round)
                try:value=train()
                finally:context['training']=False
                r=c.train_stats[0];assert r['train_batches']==5 and r['processed_train_samples']==5*c.train_samples
                assert r['effective_proto_lambda']==0 and c.avg_proto_local is None
                events.append(dict(kind='local',loop=server.loop_round,cid=c.id,sha=model_digest(c.model_set[0]),
                    upload=tensor_fingerprint(c.full_uploads[0]),CE=r['ce_loss'],F=r['regularization_loss'],rng=rng_digest()))
                return value
            c.train=local
        aggregate=server.aggregate_parameters
        def avg():
            weights={c.id:c.train_samples/sum(c.train_samples for c in server.clients) for c in server.clients}
            factory=lambda:iter((c.id,weights[c.id],c.full_upload_parameters(0,'cpu')) for c in server.selected_clients)
            expected=aggregate_group_updates(dict(server.global_model_set[0].named_parameters()),factory,weights,list(range(20)),'group_avg_only')[0]
            result=aggregate()
            assert all(torch.equal(p,expected[n]) for n,p in server.global_model_set[0].named_parameters())
            events.append(dict(kind='aggregate',loop=server.loop_round,sha=model_digest(server.global_model_set[0]),rng=rng_digest()))
            return result
        server.aggregate_parameters=avg
    words=(ROOT/'system/svd_five_round_commands.sh').read_text().splitlines()
    import shlex
    line=next(l.strip()[1:-1] for l in words if l.strip().startswith('"') and f'/{job["case"]}/checkpoints' in l)
    argv=shlex.split(line)[3:]
    def change(flag,value):argv[argv.index(flag)+1]=str(value)
    for flag,value in [('-did','-1'),('-dev','cpu'),('-gr',job['rounds']-1),('--multi_svd_five_diagnostics',job['diagnostics']),
        ('--data-root',out/'data'),('--data-subdir','fixture'),('-clip_cpu_threads',2),('-exp_name',label),
        ('-sfn',run/'c'),('--run_log_dir',run/'l'),('--h5_result_root',run/'h'),('--final-model-root',run/'f')]:change(flag,value)
    sys.argv=['main.py']+argv
    with ExitStack() as stack:
        for obj,name,val in [(sm,'FedMultiGroupProj',cls),(cs,'clientMultiGroupSVD',clientcls),(cls,'__init__',initialize),
            (cm,'DataLoader',TraceLoader),(torch.linalg,'svd',SVD),(torch.optim.SGD,'step',step)]:stack.enter_context(patch.object(obj,name,val))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    s=created[0];root=Path(s.report_dir());n=job['rounds'];assert svds[0]==n*20*3 and steps[0]==n*20*5
    verify_csv_exports(root)
    primary=json.loads((root/'primary_metrics.json').read_text());summary=json.loads((root/'primary_summary.json').read_text())
    values=[p['post_local_accuracy'] for p in primary]
    assert summary['best']==max(values) and summary['best_completed_train_count']==values.index(max(values))+1
    if job['diagnostics']:
        assert len(s.five_endpoints)==20*2*2*n+(20*2*(n-1) if job['case']=='Hplus' else 0)
        for row in s.five_metrics:
            members=[r for r in s.five_endpoints if r['loop_round']==row['loop_round'] and r['stage']==row['stage'] and r['split']==row['split'] and r['client_id'] in row['member_ids']]
            assert row['accuracy']==sum(r['correct'] for r in members)/sum(r['samples'] for r in members)
        for p in primary:
            diag=next(r for r in s.five_metrics if r['loop_round']==p['loop_round'] and r['stage']=='post_local' and r['split']=='test' and r['evaluation_scope']=='all')
            assert p['post_local_accuracy']==diag['accuracy']
        assert len(s.five_spectra)==n*(3+20*3+3+3) and len(s.five_slots)>0
        for name,rows in five.tables(s).items():
            assert json.loads((root/(name+'.json')).read_text())==rows
            with (root/(name+'.csv')).open(encoding='utf-8') as f:assert sum(1 for _ in f)==len(rows)+1
    with h5py.File(s.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values
        if job['diagnostics']:
            for name,rows in five.tables(s).items():assert [json.loads(r) for r in hf[name+'_json'][:]]==rows
    if job['case']=='Hplus':
        records=[r for c in s.clients for r in c.svd_spectral_records]
        assert not any(r['changed'] for r in records if r['loop_round']==0)
        for name,top in [('conv2',72),('fc1',720),('fc2',461)]:
            for loop in range(1,n):
                row=next(r for r in records if r['loop_round']==loop and r['client_id']==0 and r['layer']==name)
                assert row['rank_corrections'][-1]['rank']==top
                assert row['rank_corrections'][-1]['actual_gain']>1
                assert any(r['actual_gain'] is not None and r['actual_gain']>1 for r in row['rank_corrections'] if r['rank']>row['k'])
                peers=[r for r in records if r['loop_round']==loop and r['layer']==name]
                assert len({r['corrected_full_spectrum_sha256'] for r in peers})==1
        flattened=json.loads((root/'svd_all_tail_ranks.json').read_text())
        assert len(flattened)==sum(len(r.get('rank_corrections',[])) for r in records)
        import csv
        with (root/'svd_all_tail_ranks.csv').open(encoding='utf-8',newline='') as f:
            saved=list(csv.DictReader(f))
        assert len(saved)==len(flattened)
        for row,actual in zip(flattened,saved):
            assert all(actual[k]==('' if v is None else str(v)) for k,v in row.items())
        with h5py.File(s.args.save_file_paths[-1]) as hf:
            assert [json.loads(r) for r in hf['svd_all_tail_ranks_json'][:]]==flattened
    final_rng=rng_digest()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(s.final_model_dir())/f'{role}_model_0.pt')
        original=s.global_model_set[0] if role=='Server' else s.clients[int(role.split('_')[1])].model_set[0]
        assert model_digest(model)==model_digest(original)
        if role!='Server':
            from utils.multi_rank_partition import protected_rng
            cid=int(role.split('_')[1]);correct=samples=0
            with protected_rng([model]),torch.no_grad():
                for x,y in s.clients[cid].load_test_data():
                    correct+=int((model(x).argmax(1)==y).sum());samples+=len(y)
            expected=next(r for r in s.endpoint_records if r['loop_round']==n-1 and r['client_id']==cid and r['stage']=='post')
            assert correct==expected['correct'] and samples==expected['test_samples']
    assert final_rng==rng_digest()
    manifest=json.loads((root/'manifest.json').read_text());assert manifest['initial_parameter_sha256']==s.initial_hash
    report=dict(status='PASS',events=events,batches=batches,primary=primary,summary=summary,final_rng=final_rng,
        initialization_sha256=s.initial_hash,data_fingerprint=s.context['data_fingerprint'],report_dir=str(root),SVD_calls=svds[0],SGD_steps=steps[0])
    (run/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Owned verifier model copies only; keep diagnostics, hashes and raw logs.
    for folder in [run/'c',run/'f']:
        if folder.resolve().is_relative_to(out.resolve()):
            for file in folder.rglob('*.pt'):file.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker');p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve()
    if a.worker:return worker(out,a.worker)
    jobs={k:dict(case=k,rounds=5,diagnostics=1) for k in ('S','L','H','Hplus')}
    jobs.update(Hoff=dict(case='H',rounds=2,diagnostics=0),Hplusoff=dict(case='Hplus',rounds=2,diagnostics=0),
        frozen=dict(case='H',rounds=2,diagnostics=0,frozen=True))
    if a.resume:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Wrong owned workspace')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        gen=np.random.default_rng(416);data=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(gen.normal(size=(n,3,32,32))*.1).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)+cid)%6))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        if a.resume and (out/label/'verification.json').exists():continue
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{label} failed: {out}/{label}.log')
        print('PASS',label,flush=True)
    reports={name:json.loads((out/name/'verification.json').read_text()) for name in jobs}
    for report in reports.values():
        values=[r['post_local_accuracy'] for r in report['primary']];summary=report['summary']
        assert summary['final']==values[-1] and summary['last10']==sum(values[-10:])/len(values[-10:])
        assert summary['last20']==sum(values[-20:])/len(values[-20:])
    prefix=lambda rows,rounds:[r for r in rows if (r.get('loop',-1) if isinstance(r,dict) else r[0])<rounds]
    for a,b in [('H','Hoff'),('Hplus','Hplusoff'),('Hoff','frozen')]:
        for key in ('events','batches'):assert prefix(reports[a][key],2)==prefix(reports[b][key],2),(a,b,key)
        assert reports[a]['primary'][:2]==reports[b]['primary'][:2]
    assert prefix(reports['H']['events'],1)==prefix(reports['Hplus']['events'],1)
    assert reports['H']['batches']==reports['Hplus']['batches']
    for case in ('S','L'):assert reports[case]['batches']==reports['H']['batches'],case
    assert len({r['initialization_sha256'] for r in reports.values()})==1 and len({r['data_fingerprint'] for r in reports.values()})==1
    from compare_svd_five_round import compare
    pairs=compare({k:reports[k]['report_dir'] for k in ('S','L','H','Hplus')})
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',N=20,K=1,main_cases_rounds=5,local_epochs=5,
        original_SVD_calls_per_five_round_case=300,SGD_steps_per_five_round_case=500,
        frozen_reference='b150128 client/server old default H path',default_vs_frozen_exact=True,
        diagnostics_toggle_upload_aggregate_batches_RNG_exact=True,H_Hplus_R1_exact=True,
        shared_init_and_data=True,all_tail_reaches_top_tier_R2=True,unequal_sample_weighted_metrics_verified=True,
        exports_and_21_models_per_case_reload_verified=True,roundwise_comparison_rows=len(pairs),real_accuracy_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
