"""Real main N20/K1, original-width CNN, 2 rounds, E5; synthetic shards only."""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from common.experiment import rng_state,set_rng
    from fedfews_compat.observe import rng_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils.cnn512_reporting import model_digest
    from utils import multi_fc1_tail_scale as tail,multi_group_svd as svd
    from utils.group_projection import aggregate_group_updates
    from utils.fc1_tail_diagnostic import read_only,finite
    import flcore.clients.clientMultiGroupSVD as cs
    import flcore.clients.clientMultiGroupProj as cm
    import flcore.servers.serverMultiGroupProj as sm
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    server_class=sm.FedMultiGroupProj;client_class=cs.clientMultiGroupSVD
    if job.get('frozen'):
        ns={'__name__':'frozen_17e5ebb_client'}
        exec(compile(subprocess.check_output(['git','show','17e5ebb:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,text=True,encoding='utf-8'),
            '17e5ebb/clientMultiGroupSVD.py','exec'),ns);client_class=ns['clientMultiGroupSVD']
        ns={'__name__':'frozen_17e5ebb_server'}
        exec(compile(subprocess.check_output(['git','show','17e5ebb:system/flcore/servers/serverMultiGroupProj.py'],cwd=ROOT,text=True,encoding='utf-8'),
            '17e5ebb/serverMultiGroupProj.py','exec'),ns);server_class=ns['FedMultiGroupProj']
    context=dict(training=False,downloading=False);events=[];batches=[];created=[];steps=[0];decompositions=[0];pending={};parity=[]
    native_apply=__import__('utils.fc1_tail_factors',fromlist=['apply_from_svd']).apply_from_svd
    # Independent FROZEN offline B formula, not the shared live helper.
    import ast
    code=subprocess.check_output(['git','show','17e5ebb:system/utils/fc1_tail_diagnostic.py'],cwd=ROOT,text=True,encoding='utf-8')
    tree=ast.parse(code);tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rescale_tail']
    ns=dict(torch=torch,copy=copy,finite=finite);exec(compile(ast.fix_missing_locations(tree),'17e5ebb/offline_B.py','exec'),ns);offline_B=ns['rescale_tail']
    def apply(layer,p,s,q,minimum,strength=1.,verify_product=False):
        from types import SimpleNamespace
        balanced=copy.deepcopy(layer)
        expected,_=offline_B(SimpleNamespace(fc1=balanced),dict(P=p.cpu(),S=s.cpu(),Qh=q.cpu()),minimum)
        result=native_apply(layer,p,s,q,minimum,strength,verify_product)
        for a,b in zip(layer.parameters(),expected.fc1.parameters()):assert torch.equal(a,b),'Online/frozen offline B mismatch'
        pending[context['cid']]=balanced
        return result
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append([context['loop'],context['cid'],x[:,0,0,0].tolist()])
                yield x,y
    native_svd=torch.linalg.svd
    def SVD(*a,**kw):
        if context['downloading']:decompositions[0]+=1
        return native_svd(*a,**kw)
    native_step=torch.optim.SGD.step
    def SGD(opt,*a,**kw):
        if context['training']:steps[0]+=1
        return native_step(opt,*a,**kw)
    native_init=server_class.__init__
    def initialize(server,*a,**kw):
        native_init(server,*a,**kw);created.append(server)
        assert server.K==1 and server.groups==[list(range(20))]
        events.append(dict(kind='initial',weights=server.initial_hash,RNG=rng_digest()))
        for c in server.clients:
            native_download=c.set_parameters;native_train=c.train
            def download(servers,loop,c=c,native_download=native_download):
                context.update(downloading=True,cid=c.id,loop=loop)
                try:native_download(servers,loop)
                finally:context['downloading']=False
                actual=c.model_set[0]
                assert [id(p) for p in actual.parameters()]==[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']]
                if c.id in pending:
                    balanced=copy.deepcopy(actual);balanced.fc1.load_state_dict(pending.pop(c.id).state_dict())
                    with read_only(actual),torch.no_grad():
                        actual.eval();balanced.eval();x=torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32)
                        logits,expected=actual(x),balanced(x)
                        assert torch.allclose(logits,expected,atol=1e-6,rtol=1e-5)
                        y=torch.tensor([0,1]);ce=torch.nn.functional.cross_entropy(logits,y);ce0=torch.nn.functional.cross_entropy(expected,y)
                        f=actual.frobenius_decay()*.001;f0=balanced.frobenius_decay()*.001
                        assert torch.allclose(torch.stack([ce,f]),torch.stack([ce0,f0]),atol=1e-6,rtol=1e-5)
                        assert sum(p.numel() for p in actual.parameters())==sum(p.numel() for p in balanced.parameters())
                        parity.append(dict(loop_round=loop,client_id=c.id,logits_max_error=float((logits-expected).abs().max()),
                            ordinary_CE_F_max_error=float((torch.stack([ce,f])-torch.stack([ce0,f0])).abs().max())))
                events.append(dict(kind='download',loop=loop,cid=c.id,weights=model_digest(actual),RNG=rng_digest()))
            c.set_parameters=download
            def train(c=c,native_train=native_train):
                context.update(training=True,cid=c.id,loop=server.loop_round)
                try:result=native_train()
                finally:context['training']=False
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
            before=[model_digest(c.model_set[0]) for c in server.clients];result=native_aggregate()
            assert before==[model_digest(c.model_set[0]) for c in server.clients]
            for name,p in server.global_model_set[0].named_parameters():assert torch.equal(p,expected[name]),name
            events.append(dict(kind='aggregate',loop=server.loop_round,weights=model_digest(server.global_model_set[0]),RNG=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    sys.argv=['main.py','-did','-1','-dev','cpu','-algo','FedMultiGroupProj','-data','Cifar100','-m','Decom_CNN-5-512',
        '-t','1','-pv','0','-nc','20','-ncl','100','-nsm','1','-jr','1','--seed','0','-lr','.005','-lbs','16','-ls','5','-gr','1','-eg','1',
        '-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),'--data-subdir','fixture','--data-formats','npz_only',
        '--multi_avg_proto_ablation','--multi_svd','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.15',
        '--multi_group_basis','full_weights','--multi_group_mode','group_avg_only','--multi_group_sizes','20','--multi_group_weight','1',
        '--multi_proto_mode','none','--multi_proto_lambda','0','--multi_proto_aggregation','class_sample','--multi_proto_loss','mse',
        '--multi_proto_feature','base_output','--multi_proto_target_type','raw_mean','--multi_svd_aggregation','full_avg',
        '--multi_rank_spectrum',str(job.get('spectrum',1)),'--multi_svd_diagnostics',str(job.get('diagnostics',1)),
        '-is_regular','1','-regular_lamda','.001','--local_grad_clip','10','--control_test_batch_size','16','-clip_cpu_threads','2',
        '-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),'--h5_result_root',str(run/'h'),'--final-model-root',str(run/'f')]
    if 'strength' in job:sys.argv.extend(['--multi_fc1_tail_scale_strength',str(job['strength'])])
    with ExitStack() as stack:
        import utils.fc1_tail_factors as factors
        for obj,name,value in [(sm,'FedMultiGroupProj',server_class),(cs,'clientMultiGroupSVD',client_class),(server_class,'__init__',initialize),
            (cm,'DataLoader',TraceLoader),(torch.linalg,'svd',SVD),(torch.optim.SGD,'step',SGD),(factors,'apply_from_svd',apply)]:
            stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];root=Path(server.report_dir());manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    summary=json.loads((root/'primary_summary.json').read_text());values=[p['post_local_accuracy'] for p in server.primary_records]
    assert manifest['primary_summary']==summary and summary['best']==max(values) and summary['best_loop_round']==values.index(max(values))
    assert summary['final']==values[-1] and summary['last10']==summary['last20']==sum(values)/len(values)
    for row in server.primary_records:
        members=[r for r in server.endpoint_records if r['loop_round']==row['loop_round'] and r['stage']=='post']
        assert row['post_local_accuracy']==sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
        assert row['pre_observed'] and row['member_ids']==list(range(20))
    assert steps[0]==len(batches)==200 and decompositions[0]==120
    receipts=tail.rows(server)
    if job.get('strength',0)>0:
        assert len(receipts)==40 and len(parity)==32
        assert sum(r['effective_strength']>0 for r in receipts)==32
        assert all(r['minimum_actual_fc1_rank']==120 and r['tail_slot_count']==r['actual_fc1_rank']-120 for r in receipts)
        for cid in range(20):
            records=[r for r in receipts if r['client_id']==cid]
            assert [r['loop_round'] for r in records]==[0,1]
            assert records[0]['source_server_sha256']!=records[1]['source_server_sha256']
        assert json.loads((root/'fc1_tail_scale_downloads.json').read_text())==receipts
        assert len((root/'fc1_tail_scale.jsonl').read_text().splitlines())==2
    else:assert not receipts
    assert len(server.capacity_records)==10
    for record in server.capacity_records:
        members=[r for r in server.endpoint_records if r['loop_round']==record['loop_round'] and r['stage']=='post' and r['client_id'] in record['member_ids']]
        assert record['post_local_accuracy']==sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
    if job.get('spectrum',1):
        spectra=json.loads((root/'rank_spectrum.json').read_text())
        fc1=[r for r in spectra if r['layer']=='fc1']
        assert len(fc1)==6 and {r['stage'] for r in fc1}=={'client_download_svd','client_post_local','server_post_aggregation'}
        assert all(r['minimum_actual_client_rank']==120 for r in fc1)
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values and json.loads(hf.attrs['fc1_tail_scale_json'])==manifest['fc1_tail_scale']
        if receipts:assert [json.loads(row) for row in hf['fc1_tail_scale_downloads_json'][:]]==receipts
    final_rng=rng_digest();state=rng_state()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(server.final_model_dir())/f'{role}_model_0.pt')
        original=server.global_model_set[0] if role=='Server' else server.clients[int(role.split('_')[1])].model_set[0]
        assert model_digest(model)==model_digest(original)
        if job.get('strength',0)>0:
            if role=='Server':assert payload['fc1_tail_scale_policy']['configured_strength']==1
            else:assert payload['fc1_tail_scale']['download']['completed_train_count']==2
        with torch.no_grad():assert torch.equal(model(torch.zeros(1,3,32,32)),copy.deepcopy(original).eval()(torch.zeros(1,3,32,32)))
    set_rng(state)
    report=dict(status='PASS',job=job,events=events,batches=batches,final_rng=final_rng,primary=server.primary_records,
        summary=summary,initial_hash=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],
        offline_parity=parity,SGD_steps=steps[0],SVD_download_calls=decompositions[0],download_records=len(receipts),models_reloaded=21)
    (run/'verification.json').write_text(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--resume-owned-output',action='store_true');args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    jobs=dict(scale1=dict(strength=1.),zero=dict(strength=0.),unset={},frozen=dict(frozen=True),quiet=dict(strength=1.,spectrum=0,diagnostics=0))
    if args.resume_owned_output:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Not this verifier\'s synthetic workspace')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        rng=np.random.default_rng(94);data=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        if args.resume_owned_output and (out/label/'verification.json').is_file():continue
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Acceptance failed: {out}/{label}.log')
        print('PASS',label,flush=True)
    reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    for label in ('zero','unset','frozen'):
        for key in ('events','batches','final_rng','primary','summary','initial_hash','data_fingerprint'):
            assert reports[label][key]==reports['zero'][key],(label,key)
    for key in ('events','batches','final_rng','primary','summary','initial_hash','data_fingerprint'):
        assert reports['scale1'][key]==reports['quiet'][key],('diagnostics',key)
    for report in reports.values():
        assert report['initial_hash']==reports['zero']['initial_hash'] and report['data_fingerprint']==reports['zero']['data_fingerprint']
        assert report['batches']==reports['zero']['batches'] and report['final_rng']==reports['zero']['final_rng']
    receipt=dict(status='PASS_CPU',synthetic_only=True,N=20,K=1,completed_rounds=2,epochs_per_round=5,
        actual_CNN_width=16,SGD_steps_per_case=200,download_SVD_calls_per_case=120,models_reloaded_per_case=21,
        default_and_explicit_zero_vs_frozen_17e5ebb_parameters_uploads_batches_RNG_exact=True,
        scale1_same_SVD_vs_frozen_offline_B_parameter_max_error=0.,
        diagnostics_toggle_parameters_uploads_batches_RNG_exact=True,
        same_input_initial_logits_CE_F_and_restored_matrices_close=True,
        adjusted_downloads=32,minimum_capacity_downloads_skipped=8,
        ordinary_full_Avg_kernel_parameter_max_error=0.,
        per_round_reapplication_and_fresh_optimizer_verified=True,
        primary_sample_weighted_curves_and_earliest_best_exports_verified=True,
        cases=list(jobs),real_accuracy_improvement_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(receipt,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
