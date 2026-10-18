"""Short actual-main acceptance. Optional frozen 7e236dd parameter/batch/RNG parity.

Only isolated synthetic fixtures are created; never touches the real dataset.
Use --compact for same-volume hardlinked duplicate model exports.
"""
import argparse
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def worker(job, kind, reference=False):
    runtime=Path(job['reference_root']) if reference else ROOT
    sys.path.insert(0,str(runtime/'system'))
    import torch
    from unittest.mock import patch
    from common.experiment import write_json
    from common.models import create_full_model
    from utils.cnn512_reporting import model_digest
    from flcore.clients.clientbase import load_item
    from flcore.servers.serverTargetProj import FedTargetProj
    from flcore.servers.serverfedfews import FedFewS
    from fedfews_compat.observe import rng_digest
    from torch.utils.data import DataLoader
    import flcore.clients.clientbase as base_module
    import flcore.clients.clientfedfews as few_module
    if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
    events=[];batches=[];context={};created=[];test_calls={}
    class TraceLoader(DataLoader):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            original=self.batch_sampler;stage=dict(context)
            class Sampler:
                def __len__(self): return len(original)
                def __iter__(self):
                    indices=[]
                    for batch in original:
                        indices.append(list(map(int,batch)));yield batch
                    batches.append(dict(context=stage,indices=indices,rng=rng_digest()))
            object.__setattr__(self,'batch_sampler',Sampler())
    cls=FedFewS if kind=='few' else FedTargetProj
    initialize=cls.__init__
    def observed(s,*a,**kw):
        initialize(s,*a,**kw);created.append(s)
        initial=s.global_model if kind=='few' else load_item('Server','model',s.save_folder_name)
        events.append(dict(name='initialization',hash=model_digest(initial),rng=rng_digest()))
        for c in s.clients:
            for name in ['load_train_data','load_test_data']:
                native=getattr(c,name)
                def loader(*a,native=native,c=c,name=name,**kw):
                    previous=dict(context);context.update(client_id=c.id,loader=name)
                    try:return native(*a,**kw)
                    finally:context.clear();context.update(previous)
                setattr(c,name,loader)
            name='test_metrics' if kind=='few' else 'test_post_local'
            native=getattr(c,name)
            def test(*a,native=native,c=c,**kw):
                test_calls[c.id]=test_calls.get(c.id,0)+1
                return native(*a,**kw)
            setattr(c,name,test)
            native=c.train
            def train(*a,native=native,c=c,**kw):
                result=native(*a,**kw)
                models=c.model_set if kind=='few' else [load_item(c.role,'model',c.save_folder_name)]
                events.append(dict(name='local',client_id=c.id,hashes=[model_digest(m) for m in models],rng=rng_digest(),
                    losses=c.train_model_losses.tolist() if kind=='few' else {k:v for k,v in c.last_normal_train_stats.items() if k!='train_seconds'}))
                return result
            c.train=train
        name='aggregate_parameters' if kind=='few' else 'aggregate_parameters_avg'
        native=getattr(s,name)
        def aggregate(*a,**kw):
            result=native(*a,**kw)
            models=s.global_model_set if kind=='few' else [load_item('Server','model',s.save_folder_name)]
            payload = dict(alpha=s.last_alpha.tolist(),w=s.last_w.tolist(),
                           choices=[c.best_model_idx for c in s.clients],
                           deployment_ce=[c.deployment_ce.tolist() if c.deployment_ce is not None else None for c in s.clients]) if kind=='few' else {}
            events.append(dict(name='aggregate',hashes=[model_digest(m) for m in models],
                upload_ids=s.uploaded_ids,weights=s.uploaded_weights,payload=payload,rng=rng_digest()))
            return result
        setattr(s,name,aggregate)
        if kind!='few':
            s._recover_if_needed=lambda *a: (_ for _ in ()).throw(AssertionError('No recovery in full CNN path'))
    out=Path(job['output'])/('reference' if reference else 'current')/kind
    scope='group' if kind=='group' else 'c0' if kind=='projection' else 'all'
    argv=['main.py','-algo','FedFewS' if kind=='few' else 'FedTargetProj','-m','CNN-512','-data','Cifar100',
        '--cnn512_control','--control_eval_scope',scope,'--seed',str(job['seed']),'--target_client_id',str(job['target']),
        '--data-root',job['data_root'],'--data-subdir',job['subdir'],'--data-formats','npz_only',
        '-pt',job['partition'],'-cpc',str(job['cpc']),'-nc',str(job['clients']),'-ncl','100','-jr','1',
        '-lr',str(job['lr']),'-lbs',str(job['batch']),'-ls',str(job['epochs']),'-gr',str(job['rounds']-1),'-eg',str(job['gap']),
        '-nsm',str(job['K']),'-smu',str(job['mu']),'--local_grad_clip',str(job['clip']),
        '--control_test_batch_size',str(job['test_batch']),'-is_regular','0','--target_prox_mu','0',
        '-dev','cpu','-did','0','-t','1','-pv','0','-niid','1','-clip_cpu_threads','2','--fedfews_torch_threads','2',
        '-sfn',str(out/'checkpoints'),'--run_log_dir',str(out/'logs'),'--h5_result_root',str(out/'h5'),
        '--final_model_root',str(out/'final_models'),'-exp_name','configurable_'+kind]
    if kind!='few':argv+=['--target_proj_mode',{'avg':'avg','group':'group_projection_softmax','projection':'projection_softmax'}[kind]]
    if kind=='group':
        argv+=['--target_group_ids',','.join(map(str,job['group']))]
        if job['beta'] is not None:argv+=['--target_group_weight',str(job['beta'])]
        if job['group_tau']!=.2:argv+=['--group_softmax_tau',str(job['group_tau'])]
    if kind=='projection':
        if job['self'] is not None:argv+=['--projection_self_weight',str(job['self'])]
        if job['projection_tau']!=.2:argv+=['--projection_softmax_tau',str(job['projection_tau'])]
    if job['partition']=='dir':argv+=['--data-expected-alpha',str(job['alpha'])]
    sys.argv=argv
    with ExitStack() as stack:
        stack.enter_context(patch.object(cls,'__init__',observed))
        stack.enter_context(patch.object(base_module,'DataLoader',TraceLoader))
        stack.enter_context(patch.object(few_module,'DataLoader',TraceLoader))
        stack.enter_context(patch('flcore.clients.clientbase.Model_Distribe',side_effect=AssertionError('No SVD distribution')))
        stack.enter_context(patch('flcore.servers.serverTargetProj.Model_Distribe',side_effect=AssertionError('No SVD distribution')))
        if job['compact']:
            import shutil
            copy=shutil.copy2
            def link(src,dst,**kw):
                target=Path(dst)/Path(src).name if Path(dst).is_dir() else Path(dst)
                if Path(src).suffix=='.pt':os.link(src,target);return str(target)
                return copy(src,dst,**kw)
            stack.enter_context(patch('shutil.copy2',link))
        runpy.run_path(str(runtime/'system/main.py'),run_name='__main__')
    s=created[0];final_rng=rng_digest();folder=Path(s.report_dir())
    points=json.loads((folder/'primary_metrics.json').read_text())
    summary=json.loads((folder/'primary_summary.json').read_text())
    due=[r for r in range(job['rounds']) if r%job['gap']==0 or r==job['rounds']-1]
    assert [p['loop_round'] for p in points]==due
    assert len(s.control_training)==job['rounds']*job['clients']
    ids=job['group'] if kind=='group' else [job['target']] if kind=='projection' else list(range(job['clients']))
    rows=json.loads((folder/'client_metrics.json').read_text()) if (folder/'client_metrics.json').exists() else []
    if not rows:  # Frozen implementation has this file too; explicit failure if absent.
        raise AssertionError('Missing client metrics')
    values=[]
    for point in points:
        selected=[r for r in rows if r['loop_round']==point['loop_round']]
        assert sorted(r['client_id'] for r in selected)==sorted(ids)
        values.append(100.*(sum(r['correct'] for r in selected)/sum(r['test_samples'] for r in selected)))
        assert point['test_acc_percent']==values[-1]
    assert summary['best']==max(values) and summary['best_loop_round']==due[values.index(max(values))]
    assert summary['point_count']==len(due)
    if not reference:
        manifest=json.loads((folder/'comparison_manifest.json').read_text())
        assert manifest['actual_train_count']==job['rounds'] and manifest['evaluation_count']==len(due)
        assert manifest['optimizer']['lr']==job['lr'] and manifest['loaders']['train']['batch_size']==job['batch']
        assert manifest['gradient_norm_cap']==(None if job['clip']==0 else job['clip'])
        assert manifest['data_epochs_per_client']==job['epochs']*(job['K'] if kind=='few' else 1)
        auxiliary=json.loads((folder/'target_auxiliary.json').read_text())
        assert auxiliary['target_client_id']==job['target']
        assert auxiliary['observed']==(job['target'] in ids)
        if kind=='few':
            native_manifest=json.loads((folder/'manifest.json').read_text())
            assert native_manifest['budget']['actual_train_count']==job['rounds']
            assert native_manifest['budget']['evaluation_count']==len(due)
        if kind=='group':
            assert manifest['effective_config']['target_group_weight'] is not None
        for c in s.clients:
            assert c.train_time_cost['num_rounds']==job['rounds']
            stats=[r for r in s.control_training if r['client_id']==c.id]
            k=job['K'] if kind=='few' else 1
            assert all(r['processed_train_samples']==c.train_samples*job['epochs']*k for r in stats)
            import math
            assert all(r['optimizer_steps']==math.ceil(c.train_samples/job['batch'])*job['epochs']*k for r in stats)
        import h5py
        with h5py.File(s.args.save_file_paths[-1]) as hf:
            assert hf['primary_metrics/test_acc_percent'][:].tolist()==values
            assert len(hf['cnn512_control/training/train_batches'])==job['rounds']*job['clients']
            assert hf['primary_metrics'].attrs['best']==summary['best']
        for name in ['manifest.json','comparison_manifest.json']:
            assert json.loads((Path(s.final_model_dir())/name).read_text())['primary_summary']==summary
    if kind=='few':
        bundle=load_item('FedFewS','final',s.save_folder_name)
        assert len(bundle['server_models'])==job['K'] and len(bundle['clients'])==job['clients']
        for weights in bundle['server_models']:create_full_model(s.args).load_state_dict(weights,strict=True)
    else:create_full_model(s.args).load_state_dict(load_item('Server','model',s.save_folder_name).state_dict(),strict=True)
    write_json(out/'verification.json',dict(status='PASS',events=events,batches=batches,final_rng=final_rng,
        primary_values=values,primary_loop_rounds=due,test_calls=test_calls,logs=str(folder),
        initial_sha256=events[0]['hash'],data=json.loads((folder/'data_manifest.json').read_text())))


def case(name):
    job=dict(clients=20,cpc=20,seed=0,target=0,lr=.005,batch=16,epochs=5,rounds=2,gap=1,
        K=3,mu=.01,clip=10,test_batch=16,subdir='pat_20',partition='pat',alpha=.1,
        group=[0,1,2,3,4,6,7],beta=.5,self=.125,group_tau=.2,projection_tau=.2)
    if name=='pat10':job.update(cpc=10,subdir='pat_10')
    if name=='custom':job.update(clients=4,cpc=3,seed=7,target=2,lr=.012,batch=4,epochs=2,rounds=4,gap=2,
        K=2,mu=.025,clip=0,test_batch=3,subdir='custom_pat',group=[0,1,3],beta=None,self=.3,group_tau=.65,projection_tau=.45)
    if name=='dir':job.update(clients=4,cpc=99,seed=11,target=3,lr=.007,batch=3,epochs=1,rounds=2,gap=3,
        K=2,mu=.04,clip=2.5,test_batch=2,subdir='existing_dir',partition='dir',alpha=.5)
    return job


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--case',choices=['default','pat10','custom','dir'],default='default')
    parser.add_argument('--reference-commit',help='Frozen default parity, normally 7e236dd')
    parser.add_argument('--compact',action='store_true');parser.add_argument('--worker',choices=['avg','few','group','projection'])
    parser.add_argument('--reference',action='store_true')
    args=parser.parse_args()
    out=Path(args.output).resolve()
    if args.worker:return worker(json.loads((out/'job.json').read_text()),args.worker,args.reference)
    out.mkdir(parents=True,exist_ok=False)
    job=case(args.case);job.update(output=str(out),compact=args.compact,data_root=str(out/'synthetic_data'))
    if args.reference_commit:
        if args.case!='default':raise ValueError('Frozen fixed protocol can only accept the default case')
        pin=out/'pinned';pin.mkdir()
        blob=subprocess.check_output(['git','archive','--format=zip',args.reference_commit,'system'],cwd=ROOT)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:archive.extractall(pin)
        job['reference_root']=str(pin)
    import numpy as np
    directory=Path(job['data_root'])/'Cifar100'/job['subdir'];rng=np.random.RandomState(321)
    for split in ['train','test']:
        (directory/split).mkdir(parents=True)
        for cid in range(job['clients']):
            count=2+cid%3 if args.case=='default' else (job['cpc'] if job['partition']=='pat' else 5)+cid%3
            labels=(np.arange(count)+cid)%100 if args.case=='default' else ((np.arange(count)%job['cpc'])+cid*5)%100 if job['partition']=='pat' else (np.arange(count)+cid)%100
            np.savez(directory/split/f'{cid}.npz',data=dict(x=rng.randn(count,3,32,32).astype('float32'),y=labels))
    config=dict(num_clients=job['clients'],num_classes=100,partition=job['partition'],alpha=job['alpha'])
    if args.case!='default' and job['partition']=='pat':config['class_per_client']=job['cpc']
    (directory/'config.json').write_text(json.dumps(config))
    (out/'job.json').write_text(json.dumps(job,indent=2))
    kinds=['avg','few'] if args.case in ['pat10','dir'] else ['avg','group','few'] if args.case=='default' else ['avg','group','projection','few']
    reports=[]
    for kind in kinds:
        for ref in ([True,False] if args.reference_commit else [False]):
            label='reference' if ref else 'current'
            env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            command=[sys.executable,__file__,'--output',str(out),'--worker',kind]+(['--reference'] if ref else [])
            with (out/f'{label}_{kind}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'{label}/{kind} failed: see {out}/{label}_{kind}.log')
        current=json.loads((out/'current'/kind/'verification.json').read_text())
        if args.reference_commit:
            reference=json.loads((out/'reference'/kind/'verification.json').read_text())
            for key in ['events','batches','final_rng','primary_values','primary_loop_rounds']:
                assert reference[key]==current[key],f'{kind} frozen parity failed: {key}'
            assert reference['data']['files']==current['data']['files']
        reports.append(current)
    assert len({r['initial_sha256'] for r in reports})==1
    assert all(r['events'][0]['rng']==reports[0]['events'][0]['rng'] for r in reports)
    assert all(r['data']==reports[0]['data'] for r in reports)
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',case=args.case,
        reference_commit=args.reference_commit,parameter_batch_rng_parity=bool(args.reference_commit),
        initial_sha256=reports[0]['initial_sha256'],kinds=kinds,rounds=job['rounds'],clients=job['clients']),indent=2))
    print('PASS:',out/'acceptance_report.json')


if __name__=='__main__':main()
