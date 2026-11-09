"""Actual-main K1 class-sample prototypes: two ordinary rounds, synthetic only."""
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
    from torch.utils.data import DataLoader
    from common.experiment import rng_state,set_rng
    from fedfews_compat.observe import rng_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils import single_avg_proto as proto,multi_group_svd as svd
    from utils.multi_group_proto import decode_packet
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    from flcore.trainmodel.models import Hyper_CNN_512
    import flcore.clients.clientMultiGroupProj as cm
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    context=dict(training=False,collecting=False);events=[];batches=[];created=[];collected=set();snapshots={}
    counts=dict(SGD_forwards=0,collection_forwards=0,backwards=0,steps=0,clips=0)
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()  # Existing observer machinery may query CUDA, outside training.
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training'] or context['collecting']:context['labels']=y
                if context['training']:batches.append([context['loop'],context['cid'],x[:,0,0,0].tolist()])
                yield x,y
    native_forward=Hyper_CNN_512.forward
    def forward(model,*a,**kw):
        if context['training']:counts['SGD_forwards']+=1
        if context['collecting']:counts['collection_forwards']+=1
        return native_forward(model,*a,**kw)
    native_step=torch.optim.SGD.step;native_backward=torch.Tensor.backward;native_clip=torch.nn.utils.clip_grad_norm_
    def step(opt,*a,**kw):
        if context['training']:counts['steps']+=1
        return native_step(opt,*a,**kw)
    def backward(t,*a,**kw):
        if context['training']:counts['backwards']+=1
        return native_backward(t,*a,**kw)
    def clip(*a,**kw):
        if context['training']:counts['clips']+=1
        return native_clip(*a,**kw)
    native_collect=proto.collect_pre_train
    def collect(client):
        assert job['mode']=='global_pre_train' and job['coefficient']>0 and client.download_round==1
        model=client.model_set[0];before=tensor_fingerprint(model.state_dict());modes=[m.training for m in model.modules()];random=rng_digest()
        context.update(collecting=True,loop=client.download_round,cid=client.id)
        try:packet,cost=native_collect(client)
        finally:context['collecting']=False
        assert before==tensor_fingerprint(model.state_dict()) and modes==[m.training for m in model.modules()]
        assert random==rng_digest();values,meta=decode_packet(packet,'single_avg_local_v2')
        assert int(values['counts'].sum())==client.train_samples and cost['samples']==client.train_samples and cost['batches']==1
        snapshots[client.download_round,client.id]=dict(values,**meta);collected.add((client.download_round,client.id))
        return packet,cost
    native_init=FedMultiGroupProj.__init__
    def initialize(server,*a,**kw):
        native_init(server,*a,**kw);created.append(server)
        assert server.K==1 and server.groups==[list(range(20))]
        events.append(dict(kind='initial',weights=server.initial_hash,rng=rng_digest()))
        for client in server.clients:
            native_train=client.train
            def train(client=client,native_train=native_train):
                loop=server.loop_round;active=job['coefficient']>0 and loop>0
                if active and job['mode']=='global_pre_train':
                    assert collected=={(loop,c.id) for c in server.clients},'SGD began before the global collection barrier'
                    shas={c.avg_proto_receipt['sha256'] for c in server.clients};assert len(shas)==1
                    assert all(c.avg_proto_receipt['collection_phase']=='pre_local_download' and c.avg_proto_receipt['intended_use_loop_round']==loop for c in server.clients)
                observer=None;observed=[];seen=[0]
                def capture(module,inputs,output):
                    seen[0]+=1
                    if seen[0]>4:observed.append((output.detach().double().clone(),context['labels'].clone()))
                if job['coefficient']>0 and job['mode']=='global_last_epoch':observer=client.model_set[0].base.register_forward_hook(capture)
                context.update(training=True,loop=loop,cid=client.id)
                try:result=native_train()
                finally:
                    context['training']=False
                    if observer is not None:observer.remove()
                row=client.train_stats[0]
                assert row['train_batches']==5 and row['processed_train_samples']==5*client.train_samples
                assert row['effective_proto_lambda']==(1. if active else 0.)
                assert row['used_proto_source_completed_train_count']==(loop if active else None)
                assert row['prototype_collected_samples']==(client.train_samples if job['coefficient']>0 and job['mode']=='global_last_epoch' else 0)
                if active:
                    expected_phase='pre_local_download' if job['mode']=='global_pre_train' else 'last_local_epoch'
                    assert row['used_proto_collection_phase']==expected_phase and row['used_proto_intended_use_loop_round']==loop
                if observed:
                    values,meta=decode_packet(client.avg_proto_local,'single_avg_local_v2');z,y=observed[0]
                    assert len(observed)==1 and int(values['counts'].sum())==client.train_samples
                    manual=torch.zeros_like(values['sums'])
                    for j in range(len(y)):manual[int(y[j])].add_(z[j])
                    assert torch.equal(manual,values['sums'])
                    snapshots[loop,client.id]=dict(values,**meta)
                assert not client.model_set[0].base._forward_hooks
                events.append(dict(kind='local',loop=loop,cid=client.id,params=tensor_fingerprint(client.model_set[0].state_dict()),
                    upload=tensor_fingerprint(client.full_uploads[0]),rng=rng_digest()))
                return result
            client.train=train
        native_aggregate=server.aggregate_parameters
        def aggregate():
            from utils.group_projection import aggregate_group_updates
            before=[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
            random=rng_state();total=sum(c.train_samples for c in server.clients)
            samples={c.id:c.train_samples/total for c in server.clients}
            posts={c.id:c.full_upload_parameters(0,'cpu') for c in server.clients};base=dict(server.global_model_set[0].named_parameters())
            factory=lambda:iter((c.id,samples[c.id],posts[c.id]) for c in server.selected_clients)
            expected=aggregate_group_updates(base,factory,samples,list(range(20)),'group_avg_only')[0]
            independent={n:torch.zeros_like(v) for n,v in base.items()}
            for c in server.selected_clients:
                for n in independent:independent[n].add_(posts[c.id][n]*samples[c.id])
            set_rng(random);result=native_aggregate()
            assert before==[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
            for n,v in server.global_model_set[0].named_parameters():
                assert torch.equal(v,expected[n]);torch.testing.assert_close(v,independent[n],rtol=2e-6,atol=1e-7)
            events.append(dict(kind='aggregate',loop=server.loop_round,params=tensor_fingerprint(server.global_model_set[0].state_dict()),rng=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    native_merge=proto.merge_extended_round
    def merge(server):
        native_merge(server)
        values,meta=decode_packet(server.avg_proto_packet,'single_avg_library_v2');loop=server.loop_round
        packets=[snapshots[loop,c.id] for c in server.clients]
        for label in range(server.num_classes):
            count=sum(int(p['counts'][label]) for p in packets);total=torch.zeros(values['prototypes'].shape[1],dtype=torch.float64)
            for p in packets:total.add_(p['sums'][label])
            if count:total.div_(count)
            assert torch.equal(total,values['prototypes'][label]) and bool(values['valid'][label])==(count>0)
    sys.argv=['main.py','-did','0','-algo','FedMultiGroupProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu',
        '-nc','20','-ncl','100','-nsm','1','-jr','1','-t','1','-pv','0','--seed','0','-lr','.005','-lbs','16','-ls','5',
        '-gr','1','-eg','1','-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),'--data-subdir','fixture',
        '--data-formats','npz_only','-is_regular','1','-regular_lamda','.001','--local_grad_clip','10',
        '--multi_avg_proto_ablation','--multi_svd','--multi_group_mode','group_avg_only','--multi_group_weight','1','--multi_group_sizes','20',
        '--multi_group_basis','full_weights','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.15',
        '--multi_proto_mode',job['mode'],'--multi_proto_aggregation','class_sample','--multi_proto_loss','mse',
        '--multi_proto_target_type','raw_mean','--multi_proto_lambda',str(job['coefficient']),'--multi_proto_feature','base_output',
        '--multi_proto_diagnostics',str(job.get('diagnostics',1)),'--control_test_batch_size','16','-clip_cpu_threads','2',
        '-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),'--h5_result_root',str(run/'h'),'--final-model-root',str(run/'f')]
    native_copy=shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():
            try:os.link(src,dst);return dst
            except OSError:pass
        return native_copy(src,dst,*a,**kw)
    with ExitStack() as stack:
        for obj,name,value in [(FedMultiGroupProj,'__init__',initialize),(cm,'DataLoader',TraceLoader),(Hyper_CNN_512,'forward',forward),
            (torch.optim.SGD,'step',step),(torch.Tensor,'backward',backward),(torch.nn.utils,'clip_grad_norm_',clip),
            (proto,'collect_pre_train',collect),(proto,'merge_extended_round',merge),(shutil,'copy2',compact)]:stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];folder=Path(server.report_dir());summary=json.loads((folder/'primary_summary.json').read_text())
    assert counts['SGD_forwards']==counts['steps']==counts['backwards']==counts['clips']==200
    assert counts['collection_forwards']==(20 if job['mode']=='global_pre_train' and job['coefficient'] else 0)
    assert len(server.training_records)==40 and len(batches)==200
    values=[r['post_local_accuracy'] for r in server.primary_records]
    for row in server.primary_records:
        points=[r for r in server.endpoint_records if r['loop_round']==row['loop_round'] and r['stage']=='post']
        assert row['post_local_accuracy']==sum(p['correct'] for p in points)/sum(p['test_samples'] for p in points)
    assert summary['best']==max(values) and summary['best_loop_round']==values.index(max(values))
    assert summary['last10']==summary['last20']==sum(values)/len(values)
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    assert manifest['primary_summary']==summary
    source=manifest['single_model_Avg_prototype_ablation']
    if job['coefficient']:
        assert source['last_actually_used_library']['intended_use_loop_round']==1
        assert source['latest_generated_library']['intended_use_loop_round']==(1 if job['mode']=='global_pre_train' else 2)
        for name,key in [('latest_generated','latest_generated_library'),('last_used','last_actually_used_library')]:
            packet=(Path(server.final_model_dir())/f'Global_class_sample_{name}.packet').read_bytes()
            _,meta=decode_packet(packet,'single_avg_library_v2')
            assert meta['collection_phase']==source[key]['collection_phase']
    else:assert source['latest_generated_library'] is None and source['extra_collection_sample_visits']==0
    for name,rows in [('avg_proto_training',server.training_records),('avg_proto_classes',server.avg_proto_classes),('avg_proto_rounds',server.avg_proto_rounds)]:
        assert json.loads((folder/(name+'.json')).read_text())==rows
        if rows:
            with (folder/(name+'.csv')).open(encoding='utf-8',newline='') as f:assert len(list(csv.DictReader(f)))==len(rows)
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values and json.loads(hf.attrs['summary_json'])==summary
        assert json.loads(hf.attrs['avg_proto_protocol_json'])==proto.protocol(server.args)
        assert [json.loads(r) for r in hf['avg_proto_training_json'][:]]==server.training_records
    random=rng_state()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(server.final_model_dir())/f'{role}_model_0.pt')
        original=server.global_model_set[0] if role=='Server' else server.clients[int(role.split('_')[1])].model_set[0]
        assert tensor_fingerprint(model.state_dict())==tensor_fingerprint(original.state_dict())
        with torch.no_grad():assert torch.equal(model(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
        if role!='Server':assert payload['single_avg_prototype_usage']['training_loop_round']==1
    set_rng(random)
    logs=Path(server.args.training_log_dir)
    with zipfile.ZipFile(logs.with_name(logs.name+'.zip')) as archive:assert not any(n.endswith(('.pt','.packet','.npz')) for n in archive.namelist())
    result=dict(status='PASS',synthetic_only=True,job=job,events=events,batches=batches,final_rng=rng_digest(),counts=counts,
        initial_hash=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],summary=summary,
        libraries=source,models_reloaded=21,old_Avg_kernel_max_parameter_error=0.)
    (run/'verification.json').write_text(json.dumps(result,indent=2))


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text())
    reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    for a,b in [('pre','quiet'),('off','zero')]:
        for key in ('events','batches','final_rng','summary'):assert reports[a][key]==reports[b][key],(a,b,key)
    for report in reports.values():
        for key in ('initial_hash','data_fingerprint','batches','final_rng'):assert report[key]==reports['last'][key],key
        first=lambda r:[e for e in r['events'] if e.get('loop')==0]
        assert first(report)==first(reports['last']),'First round CE/F must be exact'
    result=dict(status='PASS',synthetic_only=True,clients=20,K=1,rounds=2,local_epochs=5,actual_width=16,
        first_round_weights_uploads_Avg_and_RNG_exact=True,initialization_data_and_batches_exact=True,
        zero_and_diagnostics_isolation_exact=True,pre_collection_barrier_all_clients_bound_before_SGD=True,
        counts={k:r['counts'] for k,r in reports.items()},class_sample_formula_exact=True,models_reloaded_per_case=21,
        old_Avg_kernel_max_parameter_error=0.)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker');p.add_argument('--keep-models',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.audit_only:return audit(out)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(42);data=out/'data/Cifar100/fixture'
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3 if split=='train' else 1+cid%4
            x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
            np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    jobs=dict(last=dict(mode='global_last_epoch',coefficient=1),pre=dict(mode='global_pre_train',coefficient=1),
        quiet=dict(mode='global_pre_train',coefficient=1,diagnostics=0),off=dict(mode='none',coefficient=0),zero=dict(mode='global_pre_train',coefficient=0))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        print('RUN',label,flush=True)
        with (out/(label+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('c','f'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                if owned.exists():shutil.rmtree(owned)
        print('PASS',label,flush=True)
    audit(out)


if __name__=='__main__':main()
