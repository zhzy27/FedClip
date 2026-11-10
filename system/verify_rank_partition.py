"""Real main, original-width N20/K1 SVD smoke; synthetic fixtures ONLY."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    # CPU acceptance must not allocate unrelated CUDA contexts for legacy RNG helpers.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from common.experiment import rng_state,set_rng
    from fedfews_compat.observe import rng_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils import multi_rank_partition as rp,multi_group_svd as svd
    from utils.multi_group_proto import packet_sha
    from utils.group_projection import aggregate_group_updates
    from flcore.trainmodel.models import Hyper_CNN_512
    import flcore.servers.serverMultiGroupProj as sm
    import flcore.clients.clientMultiGroupProj as cm
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    cls=sm.FedMultiGroupProj
    if job.get('frozen'):
        source=subprocess.check_output(['git','show','7fd0685:system/flcore/servers/serverMultiGroupProj.py'],cwd=ROOT,text=True)
        ns={'__name__':'frozen_7fd0685_server'};exec(compile(source,'7fd0685/serverMultiGroupProj.py','exec'),ns);cls=ns['FedMultiGroupProj']
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()  # Existing legacy diagnostic machinery, outside training.
    context=dict(training=False,aggregating=False);events=[];batches=[];created=[];steps=[0]
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append([context['loop'],context['cid'],x[:,0,0,0].tolist()])
                yield x,y
    native_svd=torch.linalg.svd
    def svd_call(*a,**kw):
        if context['aggregating']:raise AssertionError('Aggregation may NOT re-SVD factors/uploads/server')
        return native_svd(*a,**kw)
    native_step=torch.optim.SGD.step
    def step(opt,*a,**kw):
        if context['training']:steps[0]+=1
        return native_step(opt,*a,**kw)
    native_init=cls.__init__
    def initialize(server,*a,**kw):
        native_init(server,*a,**kw);created.append(server)
        assert server.K==1 and server.groups==[list(range(20))]
        events.append(dict(kind='initial',weights=server.initial_hash,rng=rng_digest()))
        for c in server.clients:
            native_train=c.train
            def train(c=c,native_train=native_train):
                context.update(training=True,loop=server.loop_round,cid=c.id)
                try:result=native_train()
                finally:context['training']=False
                row=c.train_stats[0];assert row['train_batches']==5 and row['processed_train_samples']==5*c.train_samples
                assert row['effective_proto_lambda']==(1. if job['proto'] and server.loop_round>0 else 0.)
                events.append(dict(kind='local',loop=server.loop_round,cid=c.id,weights=tensor_fingerprint(c.model_set[0].state_dict()),
                    upload=tensor_fingerprint(c.full_uploads[0]),packet=packet_sha(c.avg_proto_local) if c.avg_proto_local else None,rng=rng_digest()))
                return result
            c.train=train
        native_aggregate=server.aggregate_parameters
        def aggregate():
            random=rng_state();mother=server.global_model_set[0];base=dict(mother.named_parameters())
            mass=sum(c.train_samples for c in server.clients);samples={c.id:c.train_samples/mass for c in server.clients}
            posts={c.id:c.full_upload_parameters(0,'cpu') for c in server.clients}
            factory=lambda:iter((c.id,samples[c.id],posts[c.id]) for c in server.selected_clients)
            expected=aggregate_group_updates(base,factory,samples,list(range(20)),'group_avg_only')[0]
            if job['method']=='rank_partition':
                # Independent direct formula on ORIGINAL trained factor positions.
                for name in ('conv2','fc1','fc2'):
                    layers={c.id:getattr(c.model_set[0],name) for c in server.clients};ranks=sorted({l.rank for l in layers.values()})
                    first=layers[0];conv=name=='conv2'
                    left,right_names=('conv_u','conv_v') if conv else ('weight_u','weight_v')
                    u0=getattr(first,left);v0=getattr(first,right_names);matrix=torch.zeros(u0.shape[0],v0.shape[1]);start=0
                    for end in ranks:
                        owners=[c for c in server.selected_clients if layers[c.id].rank>=end];total=sum(c.train_samples for c in owners)
                        block=torch.zeros_like(matrix)
                        for c in owners:
                            u,v=getattr(layers[c.id],left).detach(),getattr(layers[c.id],right_names).detach()
                            block.add_((u[:,start:end]@v[start:end,:])*(c.train_samples/total))
                        matrix.add_(block);start=end
                    if conv:matrix=matrix.reshape(first.out_channels,first.kernel_size,first.in_channels,first.kernel_size).permute(0,2,1,3)
                    expected[name+'.weight']=matrix
            before=[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
            set_rng(random);context['aggregating']=True
            try:result=native_aggregate()
            finally:context['aggregating']=False
            assert before==[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
            for name,p in mother.named_parameters():assert torch.equal(p,expected[name]),name
            assert mother.ratio_LR==1. and all(isinstance(getattr(mother,n),torch.nn.Conv2d if n=='conv2' else torch.nn.Linear) for n in rp.LAYERS)
            events.append(dict(kind='aggregate',loop=server.loop_round,weights=tensor_fingerprint(mother.state_dict()),rng=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    sys.argv=['main.py','-did','-1','-algo','FedMultiGroupProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu',
        '-nc','20','-ncl','100','-nsm','1','-jr','1','-t','1','-pv','0','--seed','0','-lr','.005','-lbs','16','-ls','5',
        '-gr','1','-eg','1','-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),'--data-subdir','fixture',
        '--data-formats','npz_only','-is_regular','1','-regular_lamda','.001','--local_grad_clip','10',
        '--multi_avg_proto_ablation','--multi_svd','--multi_group_mode','group_avg_only','--multi_group_weight','1','--multi_group_sizes','20',
        '--multi_group_basis','full_weights','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.15',
        '--multi_svd_aggregation',job['method'],'--multi_rank_spectrum',str(job.get('spectrum',0)),
        '--multi_proto_mode','global_last_epoch' if job['proto'] else 'none','--multi_proto_aggregation','class_sample',
        '--multi_proto_loss','mse','--multi_proto_target_type','raw_mean','--multi_proto_lambda',str(int(job['proto'])),
        '--multi_proto_feature','base_output','--control_test_batch_size','16','-clip_cpu_threads','2',
        '-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),'--h5_result_root',str(run/'h'),'--final-model-root',str(run/'f')]
    native_copy=shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():
            try:os.link(src,dst);return dst
            except OSError:pass
        return native_copy(src,dst,*a,**kw)
    with ExitStack() as stack:
        for obj,name,value in [(sm,'FedMultiGroupProj',cls),(cls,'__init__',initialize),(cm,'DataLoader',TraceLoader),
            (torch.linalg,'svd',svd_call),(torch.optim.SGD,'step',step),(shutil,'copy2',compact)]:stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];folder=Path(server.report_dir());summary=json.loads((folder/'primary_summary.json').read_text())
    assert steps[0]==len(batches)==200 and len(server.training_records)==40
    for point in server.primary_records:
        rows=[r for r in server.endpoint_records if r['loop_round']==point['loop_round'] and r['stage']=='post']
        assert point['post_local_accuracy']==sum(r['correct'] for r in rows)/sum(r['test_samples'] for r in rows)
    values=[r['post_local_accuracy'] for r in server.primary_records]
    assert summary['final']==values[-1] and summary['best']==max(values) and summary['best_loop_round']==values.index(max(values))
    assert summary['last10']==summary['last20']==sum(values)/len(values)
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text());rank_info=manifest['svd_aggregation']
    assert rank_info['protocol']['method']==job['method'] and manifest['primary_summary']==summary
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values and json.loads(hf.attrs['svd_aggregation_json'])==rank_info
        if rp.reporting(server.args):
            for name,rows in [('rank_partitions',server.rank_partition_records),('rank_spectrum',server.rank_spectrum_records)]:
                assert json.loads((folder/(name+'.json')).read_text())==rows
                assert [json.loads(r) for r in hf[name+'_json'][:]]==rows
    if job['method']=='rank_partition':
        assert len(server.rank_partition_records)==30
        for row in server.rank_partition_records:
            assert abs(row['weight_sum']-1)<1e-15
            assert row['contributor_train_samples_total']==sum(server.clients[i].train_samples for i in row['contributor_ids'])
    if job.get('spectrum'):
        assert len(server.rank_spectrum_records)==18
        assert {r['stage'] for r in server.rank_spectrum_records}=={'client_download_svd','client_post_local','server_post_aggregation'}
        for row in server.rank_spectrum_records:
            assert math_isclose(row['spectral_total_energy'],row['actual_matrix_frobenius_norm']**2)
    random=rng_state()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(server.final_model_dir())/f'{role}_model_0.pt')
        original=server.global_model_set[0] if role=='Server' else server.clients[int(role.split('_')[1])].model_set[0]
        assert tensor_fingerprint(model.state_dict())==tensor_fingerprint(original.state_dict())
        assert payload['svd_aggregation']['method']==job['method']
        with torch.no_grad():assert torch.equal(model(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
    set_rng(random)
    report=dict(status='PASS',job=job,events=events,batches=batches,final_rng=rng_digest(),summary=summary,
        initial_hash=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],SGD_steps=steps[0],models_reloaded=21,
        independent_aggregation_max_error=0.,spectrum_records=len(getattr(server,'rank_spectrum_records',[])))
    (run/'verification.json').write_text(json.dumps(report,indent=2))


def math_isclose(a,b):
    import math
    return math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text());reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    for a,b in [('rp1','quiet'),('full0','frozen0'),('full1','frozen1')]:
        for key in ('events','batches','final_rng','summary'):assert reports[a][key]==reports[b][key],(a,b,key)
    for report in reports.values():
        for key in ('initial_hash','data_fingerprint','batches','final_rng'):assert report[key]==reports['rp0'][key],key
    first_local=lambda r:[e for e in r['events'] if e.get('loop')==0 and e['kind']=='local']
    for a,b in [('rp0','full0'),('rp1','full1')]:assert first_local(reports[a])==first_local(reports[b]),'First-round local factors/uploads/prototypes must match'
    first_aggregate=lambda r:next(e for e in r['events'] if e['kind']=='aggregate' and e['loop']==0)['weights']
    assert first_aggregate(reports['rp0'])==first_aggregate(reports['rp1'])
    assert first_aggregate(reports['full0'])==first_aggregate(reports['full1'])
    assert first_aggregate(reports['rp0'])!=first_aggregate(reports['full0']),'Heterogeneous rules should differ'
    result=dict(status='PASS',synthetic_only=True,N=20,K=1,rounds=2,local_epochs=5,actual_width=16,
        frozen_7fd0685_full_avg_parameters_uploads_packets_batches_RNG_exact=True,
        first_local_factors_and_prototypes_exact_before_different_aggregation=True,
        prototype_switch_first_aggregation_unchanged_within_same_model_aggregation=True,
        diagnostic_toggle_parameters_and_RNG_exact=True,independent_aggregation_max_parameter_error=0.,
        actual_training_steps_per_case=200,models_reloaded_per_case=21,cases=list(jobs))
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--jobs',type=int,default=1);p.add_argument('--keep-models',action='store_true');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--resume-owned-output',action='store_true',help='Reuse only this verifier\'s identical synthetic jobs and successful receipts')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.audit_only:return audit(out)
    if args.resume_owned_output:
        if not (out/'jobs.json').is_file() or not (out/'data/Cifar100/fixture/config.json').is_file():raise ValueError('Not an owned synthetic verification workspace')
    else:out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(94);data=out/'data/Cifar100/fixture'
    if not args.resume_owned_output:
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    jobs=dict(rp0=dict(method='rank_partition',proto=False,spectrum=1),rp1=dict(method='rank_partition',proto=True,spectrum=1),
        quiet=dict(method='rank_partition',proto=True,spectrum=0),full0=dict(method='full_avg',proto=False),
        full1=dict(method='full_avg',proto=True),frozen0=dict(method='full_avg',proto=False,frozen=True),
        frozen1=dict(method='full_avg',proto=True,frozen=True))
    if args.resume_owned_output:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Verification case definitions changed; use a fresh directory')
    else:(out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        existing=out/label/'verification.json'
        if args.resume_owned_output and existing.is_file():
            receipt=json.loads(existing.read_text())
            if receipt['status']!='PASS' or receipt['job']!=jobs[label]:raise ValueError('Incompatible existing verification receipt')
            print('REUSE PASS',label,flush=True);return
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            r=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'Synthetic acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('c','f'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup path')
                if owned.exists():shutil.rmtree(owned)
        print('PASS',label,flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,jobs))
    audit(out)


if __name__=='__main__':main()
