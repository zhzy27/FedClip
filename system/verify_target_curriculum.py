"""Actual-main N20 heterogeneous CNN curriculum acceptance; synthetic data only.

Three ordinary rounds, five local epochs, warmup=1 to exercise the boundary.
Formal commands independently use gr100/warmup10. No real data are accessed.
"""
import argparse
from contextlib import ExitStack
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
    from flcore.trainmodel.models import Hyper_CNN_512
    from utils.target_head import preserved_head_rng
    from utils.head_snapshot import tensor_fingerprint,file_hash
    from utils.projection_variants import aggregate_projection_weighting,aggregate_softmax_only
    from fedfews_compat.observe import rng_digest
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    context=dict(training=False);batches=[];events=[];forwards={};created=[]
    old_forward=Hyper_CNN_512.forward
    def forward(model,x,*args,**kwargs):
        if context['training'] and model.training and torch.is_grad_enabled():
            key=(context['round'],context['cid']);forwards[key]=forwards.get(key,0)+1
            batches.append((context['round'],context['cid'],x[:,0,0,0].detach().cpu().tolist()))
        return old_forward(model,x,*args,**kwargs)
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
                stats=c.last_normal_train_stats
                assert stats['local_epochs']==5 and stats['train_batches']==5
                assert stats['processed_train_samples']==5*c.train_samples
                assert forwards[current_round,c.id]==5
                active=job['mode']!='none' and current_round>=1 and c.id!=server.target_client_id
                if active:
                    rows=c.last_curriculum_rows
                    assert len(rows)==5 and all(r['active'] for r in rows)
                    assert sum(r['score_samples'] for r in rows)==c.train_samples
                    assert sum(r['score_batches'] for r in rows)==1
                    assert [r['beta'] for r in rows]==[.5,.375,.25,.125,0.]
                    assert rows[-1]['weight_min']==rows[-1]['weight_max']==1.
                with preserved_head_rng():model=load_item(c.role,'model',c.save_folder_name)
                assert model.ratio_LR==(.9,.37,.35,.25,.15)[c.id%5]
                assert not model.base._forward_hooks and not model.fc3._forward_pre_hooks
                assert all(torch.isfinite(p).all() for p in model.parameters())
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
                plain=aggregate_softmax_only(base,posts[server.target_client_id],factory,server.target_client_id,.125)
                projected=aggregate_projection_weighting(base,posts[server.target_client_id],factory,server.target_client_id,'projection_softmax',.125)
                assert [r['aggregation_weight'] for r in plain[2]]==[r['aggregation_weight'] for r in projected[2]]
                expected=projected[0]
                paths=[Path(c.save_folder_name)/f'{c.role}_model.pt' for c in server.clients]
                before=[file_hash(p) for p in paths]
            native()
            with preserved_head_rng(),torch.no_grad():
                actual=dict(load_item('Server','model',server.save_folder_name).named_parameters())
                assert all(torch.equal(actual[n],expected[n]) for n in expected)
                assert before==[file_hash(p) for p in paths]
            events.append(dict(kind='aggregate',loop=server.cur_ground,server=tensor_fingerprint(actual),rng=rng_digest()))
        server.aggregate_parameters_avg=aggregate
    sys.argv=['main.py','-algo','FedTargetProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu','-did','0',
        '-nc','20','-ncl','100','-t','1','-pv','0','-jr','1.0','--seed','0','--target_client_id','0','-lr','.005','-lbs','16',
        '-ls','5','-gr','2','-eg','1','-niid','1','-pt','pat','-cpc','20','-is_regular','1','-regular_lamda','.001',
        '--target_head_mode','shared','--target_local_schedule','joint','--target_prox_mu','0','--local_grad_clip','10',
        '--target_proj_mode','projection_softmax','--projection_self_weight','.125','--projection_softmax_tau','.2',
        '--target_curriculum_mode',job['mode'],'--target_curriculum_strength','.5','--target_curriculum_warmup_rounds','1',
        '--target_curriculum_diagnostics',str(job.get('diagnostics',1)),
        '-clip_cpu_threads','2','-exp_name',label,'-sfn',str(run/'checkpoints'),'--run_log_dir',str(run/'logs'),
        '--h5_result_root',str(run/'h5'),'--final-model-root',str(run/'final_models')]
    native_copy=shutil.copy2
    def compact(source,destination,*args,**kwargs):
        if str(source).endswith('.pt') and Path(source).is_file() and not Path(destination).exists():
            try:os.link(source,destination);return destination
            except OSError:pass  # External SSD may not support hardlinks.
        return native_copy(source,destination,*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedTargetProj,'__init__',initialize))
        stack.enter_context(patch.object(Hyper_CNN_512,'forward',forward))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        os.chdir(out/'system');runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];rng=rng_digest();folder=Path(server.report_dir());summary=server._local_accuracy_summary()
    metrics=json.loads((folder/'target_curriculum_primary.json').read_text());records=json.loads((folder/'target_curriculum_clients.json').read_text())
    assert metrics['summary']==summary and metrics['history'][-1]['completed_train_count']==3
    values=[r['target_post_local_acc'] for r in metrics['history']]
    assert summary['final_target_local_acc']==values[-1] and summary['best_target_local_acc']==max(values)
    assert summary['best_target_local_round']==values.index(max(values))+1
    assert summary['last10_target_local_acc']==sum(values)/3 and summary['last20_target_local_acc']==sum(values)/3
    rows=records['history'];expected_rows=60 if job['mode']=='none' else 212
    assert len(rows)==expected_rows,(len(rows),expected_rows)
    assert sum(r['score_samples'] for r in rows)==(0 if job['mode']=='none' else sum(c.train_samples for c in server.clients if c.id!=0)*2)
    assert sum(r['optimizer_steps'] for r in rows)==300
    for row in rows:
        scale=max(1.,abs(row['total_loss_sample_mean']))
        assert abs(row['total_loss_sample_mean']-row['weighted_ce_sample_mean']-row['weighted_F_sample_mean'])<=8*2**-23*scale
        if row['client_id']==0 or row['loop_round']==0:assert row['score_samples']==0 and not row['active']
    with (folder/'target_curriculum_clients.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==expected_rows
    for a,b in zip(csv_rows,rows):
        for key in ('weighted_ce_sample_mean','unweighted_ce_sample_mean','weighted_F_sample_mean','total_loss_sample_mean'):assert float(a[key])==b[key]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values
        group=hf['target_projection/target_curriculum']
        assert [json.loads(row) for row in group['clients_json'][:]]==rows
        assert json.loads(group.attrs['summary_json'])==summary
        assert json.loads(group.attrs['policy_json'])==records['policy']
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    assert manifest['target_curriculum']['summary']==summary and manifest['best_accuracy']==max(values)
    assert manifest['target_curriculum']['actual_optimizer_steps']==300
    assert len(batches)==300
    with preserved_head_rng(),torch.no_grad():
        for role in ('Server',*[c.role for c in server.clients]):
            restored=torch.load(Path(server.final_model_dir())/f'{role}_model.pt',map_location='cpu',weights_only=False)
            original=load_item(role,'model',server.save_folder_name)
            assert tensor_fingerprint(restored.state_dict())==tensor_fingerprint(original.state_dict())
            assert torch.equal(restored.eval()(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
    log_root=Path(server.args.training_log_dir)
    with zipfile.ZipFile(log_root.with_name(log_root.name+'.zip')) as archive:assert not any(n.endswith(('.pt','.packet','.npz')) for n in archive.namelist())
    result=dict(status='PASS',events=events,batches=batches,final_rng=rng,summary=summary,
        policy=records['policy'],actual_train_tasks=60,SGD_batches=300,models_reloaded=21,
        kernel_max_parameter_error=0.,paired_softmax_weights_exact=True,rows=expected_rows)
    (run/'verification.json').write_text(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--worker')
    parser.add_argument('--keep-models',action='store_true');args=parser.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    out.mkdir(parents=True,exist_ok=False);(out/'system').mkdir()
    import numpy as np
    rng=np.random.default_rng(101)
    for split in ('train','test'):
        directory=out/'dataset/Cifar100/pat_20'/split;directory.mkdir(parents=True)
        for cid in range(20):
            count=6+cid%3 if split=='train' else 1+cid%4
            x=rng.normal(size=(count,3,32,32)).astype('float32');x[:,0,0,0]=np.arange(count)/10
            y=(np.arange(count)//2+cid*3)%100
            np.savez(directory/f'{cid}.npz',data=dict(x=x,y=y))
    jobs={'none':dict(mode='none'),'loss':dict(mode='loss_rank'),'random':dict(mode='random_rank'),'quiet':dict(mode='loss_rank',diagnostics=0)}
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Actual-entry acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                if owned.exists():shutil.rmtree(owned)
        print('PASS',label,flush=True)
    reports={name:json.loads((out/name/'verification.json').read_text()) for name in jobs}
    for key in ('events','batches','final_rng','summary'):assert reports['loss'][key]==reports['quiet'][key],('diagnostic isolation',key)
    for report in reports.values():
        assert report['events'][0]==reports['none']['events'][0]
        assert report['batches']==reports['none']['batches'] and report['final_rng']==reports['none']['final_rng']
        assert report['policy']['data_files']==reports['none']['policy']['data_files']
        first=lambda r:[e for e in r['events'] if e.get('loop')==0]
        assert first(report)==first(reports['none'])
        c0_boundary=lambda r:[e for e in r['events'] if e.get('loop')==1 and e.get('client_id')==0]
        assert c0_boundary(report)==c0_boundary(reports['none'])
    result=dict(status='PASS',synthetic_only=True,clients=20,rounds=3,local_epochs=5,warmup=1,original_width=16,
        initial_weights_data_batches_RNG_exact=True,warmup_all_uploads_aggregates_exact=True,
        diagnostics_toggle_parameters_and_RNG_exact=True,one_SGD_forward_per_step=True,
        old_projection_kernel_max_parameter_error=0.,paired_softmax_weights_exact=True,
        cases={k:{key:r[key] for key in ('actual_train_tasks','SGD_batches','models_reloaded','rows')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
