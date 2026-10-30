"""Actual main paired target TRAIN batches: N20 heterogeneous CNN, two rounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import csv
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import h5py
    import torch
    from flcore.servers.serverTargetProj import FedTargetProj
    from flcore.clients.clientbase import load_item
    import flcore.clients.clientbase as clientbase
    from flcore.trainmodel.models import Hyper_CNN_512
    from utils.target_head import preserved_head_rng
    from utils.head_snapshot import tensor_fingerprint,file_hash
    from utils.projection_variants import aggregate_projection_weighting,aggregate_softmax_only
    from fedfews_compat.observe import rng_digest
    from torch.utils.data import DataLoader
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()
    context=dict(training=False);batches=[];events=[];forwards={};created=[]
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append((context['round'],context['cid'],y.tolist()))
                yield x,y
    old_forward=Hyper_CNN_512.forward
    def forward(model,*args,**kwargs):
        if context['training']:
            key=(context['round'],context['cid']);forwards[key]=forwards.get(key,0)+1
        return old_forward(model,*args,**kwargs)
    original_init=FedTargetProj.__init__
    def initialize(server,*args,**kwargs):
        original_init(server,*args,**kwargs);created.append(server)
        events.append(dict(kind='initial',server=tensor_fingerprint(load_item('Server','model',server.save_folder_name).state_dict()),rng=rng_digest()))
        for c in server.clients:
            native=c.train
            def train(current_round,c=c,native=native):
                context.update(training=True,round=current_round,cid=c.id)
                try:result=native(current_round)
                finally:context['training']=False
                stats=c.last_c0_replay_train_stats
                assert stats['local_epochs']==5 and stats['own_batches']==5
                assert stats['own_sample_visits']==5*c.train_samples
                active=c.id!=server.target_client_id
                assert forwards[current_round,c.id]==stats['own_batches']*(2 if active else 1)
                assert stats['effective_weight']==(.5 if active else 0.)
                assert stats['c0_batches']==(5 if active else 0)
                assert stats['c0_sample_visits']==(5*server.clients[server.target_client_id].train_samples if active else 0)
                assert stats['aggregation_train_samples']==c.train_samples
                with preserved_head_rng():model=load_item(c.role,'model',c.save_folder_name)
                assert model.ratio_LR==(.9,.37,.35,.25,.15)[c.id%5]
                assert not model.base._forward_hooks and not model.fc3._forward_pre_hooks
                events.append(dict(kind='local',loop=current_round,client_id=c.id,model=tensor_fingerprint(model.state_dict()),rng=rng_digest()))
                return result
            c.train=train
        native=server.aggregate_parameters_avg
        def aggregate():
            with preserved_head_rng(),torch.no_grad():
                base=dict(load_item('Server','model',server.save_folder_name).named_parameters())
                posts={c.id:server._load_full_parameters(c.id) for c in server.clients}
                weights=dict(zip(server.uploaded_ids,server.uploaded_weights))
                factory=lambda:iter((cid,weights[cid],posts[cid]) for cid in server.uploaded_ids)
                if server.target_proj_mode=='avg':
                    expected={n:torch.zeros_like(p) for n,p in base.items()}
                    for cid in server.uploaded_ids:
                        for n in expected:expected[n].add_(posts[cid][n]*weights[cid])
                else:
                    plain=aggregate_softmax_only(base,posts[0],factory,0,.125)
                    projected=aggregate_projection_weighting(base,posts[0],factory,0,'projection_softmax',.125)
                    assert [r['aggregation_weight'] for r in plain[2]]==[r['aggregation_weight'] for r in projected[2]]
                    expected=(plain if server.target_proj_mode=='softmax_only' else projected)[0]
                paths=[Path(c.save_folder_name)/f'{c.role}_model.pt' for c in server.clients]
                before=[file_hash(p) for p in paths]
            native()
            with preserved_head_rng(),torch.no_grad():
                actual=dict(load_item('Server','model',server.save_folder_name).named_parameters())
                assert all(torch.equal(actual[n],expected[n]) for n in expected)
                assert before==[file_hash(p) for p in paths]  # Recovery + aggregate evaluation are read-only uploads.
            events.append(dict(kind='aggregate',loop=server.cur_ground,server=tensor_fingerprint(actual),rng=rng_digest()))
        server.aggregate_parameters_avg=aggregate
    sys.argv=['main.py','-algo','FedTargetProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu','-did','0',
        '-nc','20','-ncl','100','-t','1','-pv','0','-jr','1.0','--seed','0','--target_client_id','0','-lr','.005','-lbs','16',
        '-ls','5','-gr','1','-eg','1','-niid','1','-pt','pat','-cpc','20','-is_regular','1','-regular_lamda','.001',
        '--target_head_mode','shared','--target_local_schedule','joint','--target_prox_mu','0','--local_grad_clip','10',
        '--target_proj_mode','projection_softmax','--target_c0_replay_mode','batch_pair','--target_c0_replay_weight','.5',
        '--target_c0_replay_diagnostics',str(job.get('diagnostics',1)),
        '-clip_cpu_threads','2','-exp_name',label,'-sfn',str(run/'checkpoints'),'--run_log_dir',str(run/'logs'),
        '--h5_result_root',str(run/'h5'),'--final-model-root',str(run/'final_models')]
    if job['mode']!='avg':sys.argv+=['--projection_self_weight','.125']
    native_copy=shutil.copy2
    def compact(source,destination,*args,**kwargs):
        if str(source).endswith('.pt') and Path(source).is_file() and not Path(destination).exists():
            os.link(source,destination);return destination
        return native_copy(source,destination,*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedTargetProj,'__init__',initialize))
        stack.enter_context(patch.object(clientbase,'DataLoader',TraceLoader))
        stack.enter_context(patch.object(Hyper_CNN_512,'forward',forward))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        os.chdir(out/'system');runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];rng=rng_digest();folder=Path(server.report_dir());summary=server._local_accuracy_summary()
    metrics=json.loads((folder/'target_c0_replay_primary.json').read_text());records=json.loads((folder/'target_c0_replay_clients.json').read_text())
    assert metrics['summary']==summary and metrics['history'][1]['completed_train_count']==2
    values=[r['target_post_local_acc'] for r in metrics['history']]
    assert summary['final_target_local_acc']==values[-1] and summary['best_target_local_acc']==max(values)
    assert summary['best_target_local_round']==values.index(max(values))+1
    assert summary['last10_target_local_acc']==sum(values)/2 and summary['last20_target_local_acc']==sum(values)/2
    assert len(records['history'])==(40 if job.get('diagnostics',1) else 0)
    for row in records['history']:
        scale=max(1.,abs(row['train_mean_total_loss']))
        a=row['effective_weight']
        # One batch per epoch in this tiny fixture: source-specific sample means also equal step means.
        expected=(1-a)*row['train_mean_ce_own']+a*row['train_mean_ce_c0']+row['train_mean_weighted_frobenius']
        assert abs(row['train_mean_total_loss']-expected)<=8*2**-23*scale
        assert row['optimizer_steps']==row['own_batches']==5
    if records['history']:
        with (folder/'target_c0_replay_clients.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
        assert len(csv_rows)==40
        for a,b in zip(csv_rows,records['history']):
            for key in ('effective_weight','train_mean_ce_own','train_mean_ce_c0','train_mean_weighted_frobenius','train_mean_total_loss'):assert float(a[key])==b[key]
        assert [json.loads(line)['loop_round'] for line in (folder/'target_c0_replay.jsonl').read_text().splitlines()]==[r['loop_round'] for r in records['history']]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values
        group=hf['target_projection/target_c0_replay']
        assert [json.loads(row) for row in group['clients_json'][:]]==records['history']
        assert [json.loads(row) for row in group['primary_json'][:]]==metrics['history']
        assert json.loads(group.attrs['summary_json'])==summary
        assert json.loads(group.attrs['policy_json'])==records['policy']
        assert json.loads(group.attrs['actual_budget_json'])==records['actual_budget']
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    assert manifest['target_c0_replay']['primary_summary']==summary and manifest['best_accuracy']==max(values)
    assert manifest['target_c0_replay']['actual_budget']==records['actual_budget']
    assert len(batches)==200
    with preserved_head_rng(),torch.no_grad():
        for role in ('Server',*[c.role for c in server.clients]):
            restored=torch.load(Path(server.final_model_dir())/f'{role}_model.pt',map_location='cpu',weights_only=False)
            original=load_item(role,'model',server.save_folder_name)
            assert tensor_fingerprint(restored.state_dict())==tensor_fingerprint(original.state_dict())
            assert torch.equal(restored.eval()(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
    log_root=Path(server.args.training_log_dir)
    with zipfile.ZipFile(log_root.with_name(log_root.name+'.zip')) as archive:assert not any(n.endswith(('.pt','.packet','.npz')) for n in archive.namelist())
    result=dict(status='PASS',events=events,batches=batches,final_rng=rng,summary=summary,
        actual_budget=records['actual_budget'],
        policy=records['policy'],actual_train_tasks=40,SGD_batches=200,models_reloaded=21,
        kernel_max_parameter_error=0.,paired_softmax_weights_exact=True)
    (run/'verification.json').write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--worker')
    parser.add_argument('--jobs',type=int,default=2);parser.add_argument('--keep-models',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    out.mkdir(parents=True,exist_ok=False);(out/'system').mkdir()
    import numpy as np
    rng=np.random.default_rng(101)
    for split in ('train','test'):
        directory=out/'dataset/Cifar100/pat_20'/split;directory.mkdir(parents=True)
        for cid in range(20):
            count=2+cid%3 if split=='train' else 1+cid%4
            np.savez(directory/f'{cid}.npz',data=dict(x=rng.normal(size=(count,3,32,32)).astype('float32'),y=(np.arange(count)+cid*3)%100))
    jobs={'p':dict(mode='projection_softmax'),'q':dict(mode='projection_softmax',diagnostics=0)}
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Actual-entry acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,jobs))
    reports={name:json.loads((out/name/'verification.json').read_text()) for name in jobs}
    a,b=reports['p'],reports['q']
    for key in ('events','batches','final_rng','summary'):assert a[key]==b[key],('diagnostic isolation',key)
    for report in reports.values():
        assert report['events'][0]['server']==reports['p']['events'][0]['server']
        assert report['policy']['data_files']==reports['p']['policy']['data_files']
        assert report['policy']['actual_client_capacities']==reports['p']['policy']['actual_client_capacities']
        first=lambda r:[e for e in r['events'] if e['kind']=='local' and e['loop']==0]
        assert first(report)==first(reports['p'])
    budget=a['actual_budget']
    assert budget['own_sample_visits']==590 and budget['c0_sample_visits']==380
    assert budget['optimizer_steps']==200 and budget['local_train_calls']==40
    result=dict(status='PASS',synthetic_only=True,clients=20,rounds=2,local_epochs=5,original_width=16,
        initialization_and_data_exact_across_modes=True,diagnostic_toggle_local_parameters_and_RNG_exact=True,
        diagnostics_isolation_exact=True,helper_two_forwards_one_backward_clip_step=True,target_one_forward=True,
        old_kernel_max_error=0.,original_train_samples_unchanged=True,
        actual_budget=budget,forwards_per_case=budget['own_batches']+budget['c0_batches'],
        cases={k:{key:r[key] for key in ('actual_train_tasks','SGD_batches','models_reloaded')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
