"""Short actual-main acceptance for grouping basis/warmup and frozen 2fc0e09.

Only isolated synthetic fixtures. Most cases suppress raw model exports in the
acceptance worker to conserve disk; B_projection exports/reloads all actual K
server/client models. Production save code is never disabled by a CLI flag.
"""
import argparse
from contextlib import ExitStack
import io
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
# Independent float64 per-tensor reductions versus chunked GEMM over ~2.7M
# coordinates have different rounding orders (~2e-14 observed for full weights).
# This bound is far below float32 precision; all aggregation/frozen-path checks
# below still require torch.equal / identical SHA and RNG, with ZERO tolerance.
SIMILARITY_ATOL=1e-12


def worker(output,label,reference=False):
    job=json.loads((output/'jobs.json').read_text())[label]
    runtime=output/'pinned' if reference else ROOT
    sys.path.insert(0,str(runtime/'system'))
    import numpy as np
    import torch
    from unittest.mock import patch
    from torch.utils.data import DataLoader
    from common.experiment import write_json
    from common.models import create_full_model
    from fedfews_compat.observe import rng_digest
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    import flcore.clients.clientMultiGroupProj as module
    from utils.cnn512_reporting import model_digest
    from utils.group_projection import aggregate_group_updates
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()
    events=[];batches=[];context={};created=[];oracles=[]
    class TraceLoader(DataLoader):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            original=self.batch_sampler;where=dict(context);iteration=[0]
            class Sampler:
                def __len__(self):return len(original)
                def __iter__(self):
                    indices=[]
                    for batch in original:
                        indices.append(list(map(int,batch)));yield batch
                    if where:
                        batches.append(dict(context=where,iteration=iteration[0],indices=indices,rng=rng_digest()))
                    iteration[0]+=1
            object.__setattr__(self,'batch_sampler',Sampler())
    def oracle(servers,clients,full):
        result=np.zeros((job['N'],job['N']),dtype=np.float64)
        for k,s in enumerate(servers):
            base=dict(s.named_parameters());vectors=[]
            for c in clients:
                post=dict(c.model_set[k].named_parameters())
                vectors.append({n:(p.detach() if full else p.detach()-base[n].detach()) for n,p in post.items()})
            def dot(a,b):return sum(float((a[n].double()*b[n].double()).sum()) for n in a)
            norms=[dot(v,v)**.5 for v in vectors]
            for i,x in enumerate(vectors):
                for j,y in enumerate(vectors):
                    result[i,j]+=max(-1.,min(1.,dot(x,y)/(norms[i]*norms[j]))) / job['K'] if norms[i]*norms[j] else 0.
        return result
    init=FedMultiGroupProj.__init__
    def observed(s,*a,**kw):
        init(s,*a,**kw);created.append(s)
        events.append(dict(kind='initial',parameters=[model_digest(m) for m in s.global_model_set],rng=rng_digest()))
        for c in s.clients:
            train=c.train
            def trained(train=train,c=c):
                context.update(client_id=c.id,loop_round=s.loop_round)
                bases=[model_digest(m) for m in s.global_model_set]
                assert [model_digest(m) for m in c.model_set]==bases
                try:answer=train()
                finally:context.clear()
                assert len(c.train_stats)==job['K']
                events.append(dict(kind='local',loop=s.loop_round,client_id=c.id,bases=bases,
                    parameters=[model_digest(m) for m in c.model_set],rng=rng_digest()))
                return answer
            c.train=trained
        if not reference and label!='compat':
            collect=s.group_evidence.collect
            def collect_checked(loop,servers,clients):
                if loop in s.group_protocol['evidence_loop_rounds']:
                    expected=oracle(servers,clients,job['basis']=='full_weights')
                    answer=collect(loop,servers,clients)
                    actual=np.asarray(s.group_evidence.records[-1]['matrix'])
                    error=float(np.max(np.abs(actual-expected)))
                    assert error<SIMILARITY_ATOL,error
                    oracles.append(dict(loop_round=loop,matrix=expected.tolist(),max_error=error))
                    return answer
                return collect(loop,servers,clients)
            s.group_evidence.collect=collect_checked
        aggregate=s.aggregate_parameters
        def aggregated():
            warm=s.loop_round<job['warmup']
            total=sum(c.train_samples for c in s.clients)
            samples={c.id:c.train_samples/total for c in s.clients}
            expected=[]
            for k,model in enumerate(s.global_model_set):
                if warm:
                    values={n:torch.zeros_like(p) for n,p in model.named_parameters()}
                    # Independent exact legacy Avg parameter accumulation order.
                    with torch.no_grad():
                        for c in s.selected_clients:
                            for n,p in c.model_set[k].named_parameters():values[n].add_(p.detach()*samples[c.id])
                    expected.append(values)
                else:
                    def uploads():
                        for c in s.selected_clients:yield c.id,samples[c.id],dict(c.model_set[k].named_parameters())
                    expected.append(aggregate_group_updates(dict(model.named_parameters()),uploads,samples,s.groups[k],
                        s.args.multi_group_mode,s.args.multi_group_weight if s.args.multi_group_mode=='group_projection_softmax' else None,
                        softmax_tau=s.args.group_softmax_tau)[0])
            before=rng_digest();answer=aggregate();assert rng_digest()==before
            for model,w in zip(s.global_model_set,expected):
                for n,p in model.named_parameters():assert torch.equal(p,w[n]),(s.loop_round,n)
            events.append(dict(kind='aggregate',loop=s.loop_round,parameters=[model_digest(m) for m in s.global_model_set],rng=rng_digest()))
            return answer
        s.aggregate_parameters=aggregated
    out=output/('reference_compat' if reference else label)
    argv=['main.py','-algo','FedMultiGroupProj','-data','Cifar100','-m','CNN-512','-ncl','100','-nc',str(job['N']),
        '-nsm',str(job['K']),'--seed',str(job['seed']),'-gr',str(job['rounds']-1),'-eg',str(job['gap']),
        '-ls',str(job['epochs']),'-lbs','16','-lr','.005','-jr','1','-t','1','-pv','0','-niid','1',
        '-pt','pat','-cpc','20','-is_regular','0','--target_prox_mu','0','--local_grad_clip','10',
        '--control_test_batch_size',str(job.get('test_batch',16 if label=='compat' else 3)),
        '-clip_cpu_threads','2','-dev','cpu','--multi_cluster_seed',str(job.get('cluster_seed',0 if label=='compat' else 9)),
        '--multi_cluster_restarts',str(8 if label=='compat' else 2),
        '--multi_cluster_iterations',str(50 if label=='compat' else 10),'--multi_group_mode',job['mode'],
        '--multi_group_weight','.5','--group_softmax_tau','.2','--data-root',str(output/'synthetic'),
        '--data-subdir',job['subdir'],'--data-formats','npz_only','--control_eval_scope','all',
        '-sfn',str(out/'checkpoints'),'--run_log_dir',str(out/'logs'),'--h5_result_root',str(out/'h5'),
        '--final_model_root',str(out/'final_models'),'-exp_name','warmup_'+label]
    if not reference:
        argv+=['--multi_warmup_rounds',str(job['warmup']),'--multi_group_basis',job['basis']]
        if job.get('window') is not None:argv+=['--multi_history_window',str(job['window'])]
        if job.get('publish'):argv+=['--multi_group_publish',str(output/job['publish'])]
        if job.get('read'):argv+=['--multi_group_file',str(output/job['read']),'--multi_group_wait_seconds','120','--multi_group_poll_seconds','.05']
    if job.get('no_pre'):argv+=['--multi_no_pre_eval']
    import shutil
    original_copy=shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and not Path(dst).exists():os.link(src,dst);return dst
        return original_copy(src,dst,*a,**kw)
    sys.argv=argv
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedMultiGroupProj,'__init__',observed))
        stack.enter_context(patch.object(module,'DataLoader',TraceLoader))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        if not job.get('save_models'):
            stack.enter_context(patch.object(FedMultiGroupProj,'save_global_model',lambda s:None))
        runpy.run_path(str(runtime/'system/main.py'),run_name='__main__')
    s=created[0];final_rng=rng_digest();folder=Path(s.report_dir())
    summary=json.loads((folder/'primary_summary.json').read_text())
    due=[r for r in range(job['rounds']) if r%job['gap']==0 or r==job['rounds']-1]
    grouped=[r for r in due if r>=job['warmup']]
    assert summary['evaluation_loop_rounds']==grouped
    assert len(s.training_records)==job['N']*job['K']*job['rounds']
    for r in s.training_records:
        c=s.clients[r['client_id']]
        assert r['processed_train_samples']==c.train_samples*job['epochs']
        assert r['train_batches']==math.ceil(c.train_samples/16)*job['epochs']
    values=[]
    for r in s.primary_records:
        members=[m for m in s.endpoint_records if m['loop_round']==r['loop_round'] and m['stage']=='post' and m['selected_by_membership']]
        assert len(members)==job['N']
        value=sum(m['correct'] for m in members)/sum(m['test_samples'] for m in members)
        assert value==r['post_local_accuracy'];values.append(value)
    assert summary['best']==max(values) and summary['best_loop_round']==grouped[values.index(max(values))]
    if not reference:
        assert len(s.warmup_records)==len([r for r in due if r<job['warmup']])*job['K']
        for r in s.endpoint_records:
            if r['loop_round']<job['warmup']:assert r['assigned_model_id'] is None and r['selected_by_membership'] is None
        for r in s.aggregation_records:
            assert r['aggregation_mode']==('group_avg_all' if r['loop_round']<job['warmup'] else job['mode'])
        assert s.group_payload['completed_train_count']==max(1,job['warmup'])
        assert len(s.group_evidence.records)==(job['window'] or job['warmup'] if job['basis']=='delta_history' else 1)
        actual=np.asarray(s.group_payload['similarity_matrix'])
        if oracles:
            assert np.max(np.abs(actual-np.mean([r['matrix'] for r in oracles],axis=0)))<SIMILARITY_ATOL
    import csv,h5py
    with (folder/'primary_metrics.csv').open() as stream:
        assert [float(r['post_local_accuracy']) for r in csv.DictReader(stream)]==values
    with h5py.File(s.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values and json.loads(hf.attrs['summary_json'])==summary
        if not reference:
            assert hf.attrs['actual_warmup_train_count']==job['warmup']
            assert hf.attrs['actual_grouped_train_count']==job['rounds']-job['warmup']
            assert len(hf['warmup_metrics_json'])==len(s.warmup_records)
    manifest=json.loads((Path(s.final_model_dir())/'manifest.json').read_text())
    assert manifest['primary_summary']==summary
    assert manifest['budget']['actual_sample_visits']==sum(r['processed_train_samples'] for r in s.training_records)
    assert manifest['budget']['actual_optimizer_steps']==sum(r['train_batches'] for r in s.training_records)
    if not reference:
        assert manifest['actual_warmup_train_count']==job['warmup']
        assert manifest['actual_grouped_train_count']==job['rounds']-job['warmup']
    reloads=0
    if job.get('save_models'):
        probe=torch.ones(1,3,32,32)/100
        restored=create_full_model(s.args)
        for name,model in ([(f'Server_model_{k}.pt',m) for k,m in enumerate(s.global_model_set)]+
            [(f'Client_{c.id}_model_{k}.pt',m) for c in s.clients for k,m in enumerate(c.model_set)]):
            payload=torch.load(Path(s.final_model_dir())/name,map_location='cpu')
            restored.load_state_dict(payload['state_dict'],strict=True);restored.eval();model.eval()
            assert model_digest(restored)==model_digest(model)
            with torch.no_grad():assert torch.equal(restored(probe),model(probe))
            if 'optimizer' in payload:
                opt=torch.optim.SGD(restored.parameters(),lr=.005);opt.load_state_dict(payload['optimizer'])
                assert opt.state_dict()==payload['optimizer']
            reloads+=1
    write_json(out/'verification.json',dict(status='PASS',events=events,batches=batches,final_rng=final_rng,
        groups=s.group_payload,primary_values=values,primary_loops=grouped,logs=str(folder),oracles=oracles,
        training_tasks=len(s.training_records),model_files_reloaded=reloads,
        raw_model_exports_suppressed=not job.get('save_models'),max_aggregation_parameter_error=0.))


def verify_rendezvous(out):
    """Start the follower first; require a real post-warmup wait before producer."""
    jobs=json.loads((out/'jobs.json').read_text())
    base=dict(jobs['C_projection'],warmup=2,window=2,rounds=3,save_models=False)
    jobs['ipc_producer']=dict(base,publish='shared/ipc.json')
    jobs['ipc_follower']=dict(base,mode='group_avg_only',read='shared/ipc.json')
    jobs['ipc_follower'].pop('publish',None)
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    path=out/'ipc_follower.log'
    with path.open('w',encoding='utf-8') as follower_log:
        follower=subprocess.Popen([sys.executable,__file__,'--output',str(out),'--worker','ipc_follower'],
            cwd=ROOT/'system',env=env,stdout=follower_log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+90
            while '[GroupWait] still waiting' not in path.read_text(encoding='utf-8',errors='replace'):
                if follower.poll() is not None:raise RuntimeError(f'Follower exited before rendezvous: {path}')
                if time.monotonic()>deadline:raise TimeoutError('Acceptance follower did not reach grouping barrier')
                time.sleep(.1)
            with (out/'ipc_producer.log').open('w',encoding='utf-8') as producer_log:
                producer=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker','ipc_producer'],
                    cwd=ROOT/'system',env=env,stdout=producer_log,stderr=subprocess.STDOUT)
            if producer.returncode or follower.wait(timeout=150):raise RuntimeError('Actual-main rendezvous failed; inspect ipc logs')
        finally:
            if follower.poll() is None:follower.terminate();follower.wait(timeout=15)
    left=json.loads((out/'ipc_producer/verification.json').read_text())
    right=json.loads((out/'ipc_follower/verification.json').read_text())
    assert left['groups']==right['groups']
    assert left['training_tasks']==right['training_tasks']==base['N']*base['K']*3
    at_barrier=lambda events:[e for e in events if e.get('loop',-1)<2 or (e['kind']=='local' and e['loop']==2)]
    assert at_barrier(left['events'])==at_barrier(right['events'])
    assert left['batches']==right['batches'] and left['final_rng']==right['final_rng']
    for report in [left,right]:
        assert len([e for e in report['events'] if e['kind']=='aggregate' and e['loop']<2])==2
    receipt=json.loads((Path(right['logs'])/'group_reuse_verification.json').read_text())
    assert receipt['waited_seconds']>0 and receipt['evidence_bitwise_match']
    (out/'ipc_acceptance.json').write_text(json.dumps(dict(status='PASS',follower_started_first=True,
        completed_warmup_before_wait=True,waited_seconds=receipt['waited_seconds'],no_retraining=True),indent=2))
    print('PASS actual-main publication/wait rendezvous',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--worker');parser.add_argument('--reference',action='store_true')
    parser.add_argument('--compat-only',action='store_true',help='Only strict frozen default regression; no warmup experiments')
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker,args.reference)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    for n in [20,4,5]:
        directory=out/'synthetic'/'Cifar100'/f'clients{n}';rng=np.random.default_rng(123)
        for split in ['train','test']:
            (directory/split).mkdir(parents=True)
            for cid in range(n):
                count=2+cid%3 if split=='train' else 1+cid%4
                np.savez(directory/split/f'{cid}.npz',data=dict(x=rng.standard_normal((count,3,32,32)).astype('float32'),y=(np.arange(count)+cid*4)%100))
        (directory/'config.json').write_text(json.dumps(dict(num_clients=n,num_classes=100,partition='pat',class_per_client=20,alpha=.1)))
    blob=subprocess.check_output(['git','archive','--format=zip','2fc0e09','system'],cwd=ROOT)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:archive.extractall(out/'pinned')
    common=dict(N=4,K=2,seed=7,epochs=1,gap=2,subdir='clients4',mode='group_projection_softmax')
    jobs={'compat':dict(common,N=20,K=3,seed=0,epochs=5,rounds=2,gap=1,subdir='clients20',warmup=0,basis='single_delta')}
    for case,warmup,basis in [('A',0,'full_weights'),('B',10,'full_weights'),('C',10,'delta_history')]:
        job=dict(common,warmup=warmup,basis=basis,rounds=3 if warmup==0 else 12)
        if basis=='delta_history':job['window']=10
        jobs[case+'_projection']=dict(job,publish=f'shared/{case}.json',save_models=case=='B')
        jobs[case+'_avg_only']=dict(job,mode='group_avg_only',read=f'shared/{case}.json')
    jobs['tail']=dict(common,N=5,K=3,subdir='clients5',seed=11,warmup=4,basis='delta_history',window=2,rounds=6,gap=3,publish='shared/tail.json')
    jobs['tail_no_pre']=dict(jobs['tail'],no_pre=True,gap=1,read='shared/tail.json');jobs['tail_no_pre'].pop('publish')
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    reports={}
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def execute(label,reference=False):
        name='reference_compat' if reference else label
        with (out/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label]+(['--reference'] if reference else []),cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{name} failed; see {out}/{name}.log')
        report=json.loads((out/name/'verification.json').read_text());reports[name]=report;print('PASS',name,flush=True)
    execute('compat',True)
    for label in (['compat'] if args.compat_only else jobs):execute(label)
    old,new=reports['reference_compat'],reports['compat']
    for key in ['events','batches','final_rng','primary_values','primary_loops']:assert old[key]==new[key],key
    assert old['groups']['group_fingerprint']==new['groups']['group_fingerprint']
    if args.compat_only:
        (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',reference_commit='2fc0e09',
            clients=20,K=3,seed=0,cluster_seed=0,restarts=8,max_iterations=50,test_batch=16,
            rounds=2,default_single_delta_parameter_batch_rng_parity=True),indent=2))
        print('PASS frozen default-only acceptance',flush=True)
        return
    for case in ['A','B','C']:
        left,right=reports[case+'_projection'],reports[case+'_avg_only']
        assert left['groups']==right['groups']
        receipt=json.loads((Path(right['logs'])/'group_reuse_verification.json').read_text())
        assert receipt['evidence_bitwise_match'] and receipt['similarity_bitwise_match']
        # Same uploads/RNG all the way through first grouped local training.
        bound=max(0,jobs[case+'_projection']['warmup'])
        assert [e for e in left['events'] if e.get('loop',-1)<bound or (e['kind']=='local' and e['loop']==bound)]==[
            e for e in right['events'] if e.get('loop',-1)<bound or (e['kind']=='local' and e['loop']==bound)]
    for key in ['events','batches','final_rng']:
        assert reports['tail'][key]==reports['tail_no_pre'][key],key
    assert reports['tail']['groups']==reports['tail_no_pre']['groups']
    on=dict(zip(reports['tail']['primary_loops'],reports['tail']['primary_values']))
    off=dict(zip(reports['tail_no_pre']['primary_loops'],reports['tail_no_pre']['primary_values']))
    assert all(on[loop]==off[loop] for loop in on.keys() & off.keys())
    # Basis collection / eval schedule cannot change warmup training trajectories.
    assert [e for e in reports['B_projection']['events'] if e.get('loop',-1)<10]==[
        e for e in reports['C_projection']['events'] if e.get('loop',-1)<10]
    verify_rendezvous(out)
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',cases=list(reports),
        frozen_default_parameter_batch_rng_parity=True,exact_avg_and_group_kernels=True,
        full_weights_and_history_independent_formula=True,observation_eval_gap_parameter_rng_parity=True),indent=2))
    print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
