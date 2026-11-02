"""Eight real-main synthetic K1 mechanisms, no real data or formal experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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
    context=dict(training=False);events=[];batches=[];created=[];counts=dict(forwards=0,backwards=0,steps=0,clips=0,svd=0,recover=0)
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append([context['loop'],context['cid'],y.tolist()])
                yield x,y
    native_forward=Hyper_CNN_512.forward;native_svd=Hyper_CNN_512.decom_larger_model;native_rec=Hyper_CNN_512.recover_larger_model
    def forward(m,*a,**kw):
        if context['training']:counts['forwards']+=1
        return native_forward(m,*a,**kw)
    def decompose(m,*a,**kw):
        counts['svd']+=1
        if not job['svd']:raise AssertionError('Full training must not decompose')
        return native_svd(m,*a,**kw)
    def recover(m,*a,**kw):
        counts['recover']+=1
        if not job['svd'] and created:raise AssertionError('Full training must never recover after initialization')
        return native_rec(m,*a,**kw)
    clip_original=torch.nn.utils.clip_grad_norm_
    def clip(*a,**kw):
        if context['training']:
            counts['clips']+=1
            if job['clip']==0:raise AssertionError('Disabled clipping must not call modifying clip function')
        return clip_original(*a,**kw)
    step_original=torch.optim.SGD.step;backward_original=torch.Tensor.backward
    def step(o,*a,**kw):
        if context['training']:counts['steps']+=1
        return step_original(o,*a,**kw)
    def backward(t,*a,**kw):
        if context['training']:counts['backwards']+=1
        return backward_original(t,*a,**kw)
    init_original=FedMultiGroupProj.__init__
    def initialize(server,*a,**kw):
        init_original(server,*a,**kw);created.append(server)
        events.append(dict(kind='initial',model=server.initial_hash,rng=rng_digest()))
        assert server.K==1 and server.groups==[list(range(20))]
        assert server.global_model_set[0].ratio_LR==1.
        for c in server.clients:
            train_original=c.train
            def train(c=c,train_original=train_original):
                context.update(training=True,loop=server.loop_round,cid=c.id)
                try:result=train_original()
                finally:context['training']=False
                s=c.train_stats[0]
                assert s['train_batches']==5 and s['processed_train_samples']==5*c.train_samples
                assert s['effective_F_lambda']==(.001 if job['F'] else 0)
                assert s['regularization_loss']>0 if job['F'] else s['regularization_loss']==0
                assert s['effective_proto_lambda']==(1. if job['proto'] and server.loop_round>0 else 0.)
                assert s['used_proto_source_completed_train_count']==(server.loop_round if job['proto'] and server.loop_round>0 else None)
                assert s['prototype_collected_samples']==(c.train_samples if job['proto'] else 0)
                if job['proto']:
                    values,meta=decode_packet(c.avg_proto_local,'global_last_epoch_local')
                    assert int(values['counts'].sum())==c.train_samples
                    assert meta['source_completed_train_count']==server.loop_round+1
                assert not c.model_set[0].base._forward_hooks
                events.append(dict(kind='local',loop=server.loop_round,cid=c.id,
                    params=tensor_fingerprint(c.model_set[0].state_dict()),rng=rng_digest()))
                return result
            c.train=train
        aggregate_original=server.aggregate_parameters
        def aggregate():
            random=rng_state()
            with torch.no_grad():
                base=dict(server.global_model_set[0].named_parameters());expected={n:torch.zeros_like(v) for n,v in base.items()}
                total=sum(c.train_samples for c in server.clients)
                before=[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
                for c in server.selected_clients:
                    post=c.full_upload_parameters(0,'cpu') if job['svd'] else dict(c.model_set[0].named_parameters())
                    for n in expected:expected[n].add_(post[n]* (c.train_samples/total))
            set_rng(random);result=aggregate_original()
            assert before==[tensor_fingerprint(c.model_set[0].state_dict()) for c in server.clients]
            # group_avg_only retains the pre-existing parameter-add arithmetic.
            for n,v in server.global_model_set[0].named_parameters():
                torch.testing.assert_close(v,expected[n],rtol=2e-6,atol=1e-7)
            assert rng_digest()==events[-1]['rng']  # No random data/diagnostic sampler in aggregation.
            events.append(dict(kind='aggregate',loop=server.loop_round,params=tensor_fingerprint(server.global_model_set[0].state_dict()),rng=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    finish_original=proto.finish_round
    def finish(server):
        packets=[]
        if job['proto']:
            for c in server.clients:
                values,meta=decode_packet(c.avg_proto_local,'global_last_epoch_local');packets.append(dict(values,**meta))
        old_rng=rng_digest();finish_original(server);assert old_rng==rng_digest()
        if job['proto']:
            values,meta=decode_packet(server.avg_proto_packet,'global_last_epoch_library')
            assert meta['source_completed_train_count']==server.loop_round+1
            for label in range(server.num_classes):
                owners=[p for p in packets if int(p['counts'][label])>0]
                if owners:
                    mean=torch.zeros(512,dtype=torch.float64)
                    for packet in sorted(owners,key=lambda p:p['client_id']):mean+=packet['sums'][label]/packet['counts'][label]
                    mean/=len(owners)
                    assert torch.equal(mean,values['prototypes'][label])
                assert bool(values['valid'][label])==bool(owners)
        else:assert server.avg_proto_packet is None
    sys.argv=['main.py','-algo','FedMultiGroupProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu',
        '-nc','20','-ncl','100','-nsm','1','-jr','1','-t','1','-pv','0','--seed','0','-lr','.005','-lbs','16','-ls','5',
        '-gr','1','-eg','1','-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),'--data-subdir','pat_fixture',
        '--data-formats','npz_only','-is_regular',str(job['F']),'-regular_lamda','.001','--local_grad_clip',str(job['clip']),
        '--multi_avg_proto_ablation','--multi_group_mode','group_avg_only','--multi_group_weight','1','--multi_group_sizes','20',
        '--multi_group_basis','full_weights','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.15',
        '--multi_proto_mode','global_last_epoch' if job['proto'] or job.get('zero') else 'none',
        '--multi_proto_loss','mse','--multi_proto_lambda',str(1. if job['proto'] else 0.),'--multi_proto_feature','base_output',
        '--multi_proto_diagnostics',str(job.get('diagnostics',1)),'--control_test_batch_size','16','-clip_cpu_threads','2',
        '-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),'--h5_result_root',str(run/'h'),
        '--final-model-root',str(run/'f')]
    if job['svd']:sys.argv+=['--multi_svd','--multi_svd_diagnostics','1']
    if job['proto'] or job.get('zero'):sys.argv+=['--multi_proto_target_type','raw_mean']
    original_copy=shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():os.link(src,dst);return dst
        return original_copy(src,dst,*a,**kw)
    with ExitStack() as stack:
        for obj,name,value in [(FedMultiGroupProj,'__init__',initialize),(cm,'DataLoader',TraceLoader),
            (Hyper_CNN_512,'forward',forward),(Hyper_CNN_512,'decom_larger_model',decompose),
            (Hyper_CNN_512,'recover_larger_model',recover),(torch.nn.utils,'clip_grad_norm_',clip),
            (torch.optim.SGD,'step',step),(torch.Tensor,'backward',backward),(proto,'finish_round',finish),
            (shutil,'copy2',compact)]:stack.enter_context(patch.object(obj,name,value))
        if not job['F']:stack.enter_context(patch.object(Hyper_CNN_512,'frobenius_decay',side_effect=AssertionError('F disabled must not execute')))
        if not job['proto']:
            stack.enter_context(patch.object(proto,'LastEpochStats',side_effect=AssertionError('Disabled prototype work')))
            stack.enter_context(patch('utils.multi_group_proto.encode_packet',side_effect=AssertionError('Disabled prototype communication')))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];folder=Path(server.report_dir());summary=json.loads((folder/'primary_summary.json').read_text())
    assert counts['forwards']==counts['backwards']==counts['steps']==200
    assert counts['clips']==(200 if job['clip'] else 0)
    if not job['svd']:assert counts['svd']==0 and counts['recover']==1
    assert len(server.training_records)==40 and len(batches)==200
    for point in server.primary_records:
        rows=[r for r in server.endpoint_records if r['loop_round']==point['loop_round'] and r['stage']=='post']
        assert point['post_local_accuracy']==sum(r['correct'] for r in rows)/sum(r['test_samples'] for r in rows)
    values=[r['post_local_accuracy'] for r in server.primary_records]
    assert summary['best']==max(values) and summary['best_loop_round']==values.index(max(values))
    assert summary['last10']==summary['last20']==sum(values)/len(values)
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values
        assert json.loads(hf.attrs['summary_json'])==summary
        assert [json.loads(r) for r in hf['avg_proto_training_json'][:]]==server.training_records
    training=json.loads((folder/'avg_proto_training.json').read_text());assert training==server.training_records
    with (folder/'avg_proto_training.csv').open(encoding='utf-8',newline='') as stream:
        for a,b in zip(csv.DictReader(stream),training):assert float(a['total_loss'])==b['total_loss']
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    assert manifest['primary_summary']==summary and manifest['initial_parameter_sha256']==server.initial_hash
    assert manifest['model_class']=='flcore.trainmodel.models.Hyper_CNN_512'
    random=rng_state()
    for role in ['Server']+[f'Client_{i}' for i in range(20)]:
        model,payload=svd.load_checkpoint(Path(server.final_model_dir())/f'{role}_model_0.pt')
        original=server.global_model_set[0] if role=='Server' else server.clients[int(role.split('_')[1])].model_set[0]
        assert tensor_fingerprint(model.state_dict())==tensor_fingerprint(original.state_dict())
        with torch.no_grad():assert torch.equal(model(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
    set_rng(random)
    result=dict(status='PASS',synthetic_only=True,job=job,initial_hash=server.initial_hash,data_fingerprint=server.context['data_fingerprint'],
        initial_schema=server.initial_schema,events=events,batches=batches,final_rng=rng_digest(),counts=counts,summary=summary,
        primary=server.primary_records,all_prediction_checkpoints_reloaded=True,train_sample_visits=sum(r['processed_train_samples'] for r in training))
    (run/'verification.json').write_text(json.dumps(result,indent=2))


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text())
    reports={key:json.loads((out/key/'verification.json').read_text()) for key in jobs}
    for a,b in [('01','zero'),('02','diag0')]:
        for key in ('events','batches','final_rng','summary'):assert reports[a][key]==reports[b][key],(a,b,key)
    base=reports['01']
    for key,r in reports.items():
        for field in ('initial_hash','initial_schema','data_fingerprint','batches'):assert r[field]==base[field],(key,field)
        # Dense skips real SVD constructors. Preserve that genuine RNG difference;
        # private data generators still give every paired client identical batches.
        representative=reports['01' if r['job']['svd'] else '05']
        assert r['final_rng']==representative['final_rng'],(key,'within-training-path RNG')
    for a,b in [('01','02'),('03','04'),('05','06'),('07','08')]:
        local=lambda r:[x for x in r['events'] if x['kind']=='local' and x['loop']==0]
        assert local(reports[a])==local(reports[b]),(a,b,'first round exact')
    receipt=dict(status='PASS',synthetic_only=True,clients=20,K=1,rounds=2,local_epochs=5,actual_width=16,
        eight_initial_weights_and_data_exact=True,all_batches_exact=True,within_training_path_RNG_exact=True,
        first_round_proto_on_off_exact=True,zero_lambda_exact=True,diagnostics_isolation_exact=True,
        full_training_no_SVD_or_recovery_after_initialization=True,
        cases={k:{field:r[field] for field in ('counts','summary','train_sample_visits','all_prediction_checkpoints_reloaded')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(receipt,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--keep-models',action='store_true')
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--resume-owned-output',action='store_true',help='Verification only: reuse this tool\'s synthetic data and completed case receipts')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.audit_only:return audit(out)
    if args.resume_owned_output:
        if not (out/'jobs.json').is_file() or not (out/'data/Cifar100/pat_fixture/config.json').is_file():
            raise ValueError('Resume only a complete synthetic workspace made by this tool')
    else:out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(91);data=out/'data/Cifar100/pat_fixture'
    if not args.resume_owned_output:
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%2 if split=='train' else 1+cid%4
                np.savez(data/split/f'{cid}.npz',data=dict(x=(rng.normal(size=(n,3,32,32))*.2).astype('float32'),y=(np.arange(n)+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    settings=[(1,1,10,0),(1,1,10,1),(1,0,10,0),(1,0,10,1),(0,0,10,0),(0,0,10,1),(1,1,0,0),(1,1,0,1)]
    jobs={f'{i+1:02}':dict(svd=bool(svd),F=f,clip=clip,proto=bool(proto)) for i,(svd,f,clip,proto) in enumerate(settings)}
    jobs['zero']=dict(jobs['01'],zero=True);jobs['diag0']=dict(jobs['02'],diagnostics=0)
    if args.resume_owned_output:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Synthetic case definitions changed; use a fresh output directory')
    else:(out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        if args.resume_owned_output and (out/label/'verification.json').is_file():
            receipt=json.loads((out/label/'verification.json').read_text())
            if receipt['status']!='PASS' or receipt['job']!=jobs[label]:raise ValueError('Incompatible verification receipt')
            print('REUSE PASS',label,flush=True);return
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Actual main smoke failed: {out}/{label}.log')
        if not args.keep_models:
            for directory in ('c','f'):
                owned=(out/label/directory).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,jobs))
    audit(out)


if __name__=='__main__':main()
