"""Serial N20/K3 actual-main shared-bank acceptance; synthetic data only."""
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

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import h5py
    import torch
    from utils import single_avg_proto as common,multi_last_epoch_proto as post,multi_shared_proto as shared
    from utils.multi_group_proto import decode_packet,packet_sha
    from utils.head_snapshot import tensor_fingerprint
    from common.experiment import sha256
    from fedfews_compat.observe import rng_digest
    from flcore.trainmodel.models import Hyper_CNN_512
    from verify_multi_group_svd import worker as native_worker
    job=json.loads((out/'jobs.json').read_text())[label]
    context=dict(training=False);counts=dict(forwards=0,backwards=0,steps=0,clips=0)
    created=[];libraries=[];received=[]
    native_entry=runpy.run_path
    def entry(path,*args,**kwargs):
        sys.argv+=['--multi_last_epoch_protocol','--multi_group_reuse_policy','members_only',
            '--multi_proto_mode','last_epoch','--multi_proto_loss','mse','--multi_proto_target_type','raw_mean',
            '--multi_proto_lambda','1.0','--multi_proto_scope','global','--multi_proto_bank_mode','shared',
            '--multi_proto_feature','base_output','--multi_proto_start_round','2',
            '--multi_proto_diagnostics',str(job.get('diagnostics',1))]
        return native_entry(path,*args,**kwargs)
    native_prepare=post.prepare_round
    def prepare(server):
        if not created:created.append(server)
        before=rng_digest();native_prepare(server);assert rng_digest()==before
        if server.loop_round:
            values,meta=decode_packet(server.post_proto_packet,shared.KIND)
            fingerprint=packet_sha(server.post_proto_packet)
            assert values['prototypes'].shape==(100,512) and meta['source_completed_train_count']==server.loop_round
            for c in server.clients:
                assert c.post_proto_banks is None
                assert c.post_proto_receipt['sha256']==fingerprint
                assert torch.equal(c.post_proto_shared_bank['prototypes'],values['prototypes'])
                assert torch.equal(c.post_proto_shared_bank['valid'],values['valid'])
    native_train=common.train
    def train(client,column=0,**kwargs):
        old_bank=getattr(client,'post_proto_shared_bank',None)
        before=tensor_fingerprint({k:v for k,v in old_bank.items() if torch.is_tensor(v)}) if old_bank else None
        receipt=client.post_proto_receipt
        start=counts['forwards'];context['training']=True
        try:row,packet=native_train(client,column,**kwargs)
        finally:context['training']=False
        assert counts['forwards']-start==row['train_batches']==5
        assert row['effective_grad_clip']==10
        assert row['prototype_collected_samples']==client.train_samples
        assert row['used_proto_source_completed_train_count']==(1 if client.download_round else None)
        if old_bank:
            assert old_bank is shared.training_bank(client,column)
            assert before==tensor_fingerprint({k:v for k,v in old_bank.items() if torch.is_tensor(v)})
            received.append((client.download_round,client.id,column,receipt['sha256'],before))
            assert not old_bank['prototypes'].requires_grad and old_bank['prototypes'].grad is None
        else:assert row['raw_mse']==0.
        return row,packet
    fwd=Hyper_CNN_512.forward;bwd=torch.Tensor.backward;step=torch.optim.SGD.step;clip=torch.nn.utils.clip_grad_norm_
    def forward(model,*args,**kwargs):
        if context['training']:counts['forwards']+=1
        return fwd(model,*args,**kwargs)
    def backward(t,*args,**kwargs):
        if context['training']:counts['backwards']+=1
        return bwd(t,*args,**kwargs)
    def update(optimizer,*args,**kwargs):
        if context['training']:counts['steps']+=1
        return step(optimizer,*args,**kwargs)
    def clipping(*args,**kwargs):
        if context['training']:counts['clips']+=1
        return clip(*args,**kwargs)
    source_sha=sha256(out/'pair.json');native_finish=post.finish_round
    def finish(server):
        packets={}
        for c in server.clients:
            packets[c.id]=[decode_packet(p,'last_epoch_column_local')[0] for p in c.post_proto_local_packets]
            assert len(packets[c.id])==3
            assert all(torch.equal(p['counts'],packets[c.id][0]['counts']) for p in packets[c.id])
        expected=torch.zeros(100,512,dtype=torch.float64);contributors=torch.zeros(100,dtype=torch.int64)
        samples=torch.zeros(100,dtype=torch.int64)
        # Independent explicit nested reduction, client first then columns;
        # no call to the production equal-means routines.
        for label_id in range(100):
            for cid in sorted(packets):
                n=int(packets[cid][0]['counts'][label_id])
                if not n:continue
                mu=torch.zeros(512,dtype=torch.float64)
                for k in range(3):mu+=packets[cid][k]['sums'][label_id]/n
                expected[label_id]+=mu/3;contributors[label_id]+=1;samples[label_id]+=n
            if contributors[label_id]:expected[label_id]/=int(contributors[label_id])
        previous=server.previous_post_proto_packet
        before=rng_digest();native_finish(server);assert rng_digest()==before
        assert sha256(out/'pair.json')==source_sha
        values,meta=decode_packet(server.post_proto_packet,shared.KIND)
        assert torch.equal(values['prototypes'],expected)
        assert torch.equal(values['valid'],contributors>0)
        assert torch.equal(values['contributors'],contributors)
        assert torch.equal(values['samples'],samples)
        assert torch.equal(values['model_contributors'],3*contributors)
        assert torch.equal(values['statistics_samples'],3*samples)
        row=server.post_proto_rounds[-1]
        assert row['actual_download_bytes']==(len(previous)*20 if previous else 0)
        assert row['deduplicated_training_samples']==sum(c.train_samples for c in server.clients)
        assert row['collected_training_samples']==3*row['deduplicated_training_samples']
        assert row['extra_collection_forwards']==0 and row['actual_prototype_bank_count']==1
        libraries.append(dict(source=server.loop_round+1,tensor_sha256=tensor_fingerprint(values),packet_sha256=packet_sha(server.post_proto_packet)))
    with ExitStack() as stack:
        for obj,key,value in [(runpy,'run_path',entry),(post,'prepare_round',prepare),(post,'finish_round',finish),
            (common,'train',train),(Hyper_CNN_512,'forward',forward),(torch.Tensor,'backward',backward),
            (torch.optim.SGD,'step',update),(torch.nn.utils,'clip_grad_norm_',clipping)]:stack.enter_context(patch.object(obj,key,value))
        native_worker(out,label)
    server=created[0];root=Path(server.report_dir());final=Path(server.final_model_dir())
    assert counts==dict(forwards=600,backwards=600,steps=600,clips=600)
    assert len(received)==60 and len({r[3] for r in received})==1 and len({r[4] for r in received})==1
    rows=json.loads((root/'last_epoch_training.json').read_text())
    assert len(rows)==120 and all(r['bank_mode']=='shared' and r['actual_prototype_bank_count']==int(r['loop_round']>0) for r in rows)
    with (root/'last_epoch_training.csv').open(newline='',encoding='utf-8') as stream:
        assert [float(r['total_loss']) for r in csv.DictReader(stream)]==[r['total_loss'] for r in rows]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert [json.loads(r) for r in hf['last_epoch_training_json'][:]]==rows
        assert [json.loads(r) for r in hf['last_epoch_rounds_json'][:]]==server.post_proto_rounds
        assert json.loads(hf.attrs['last_epoch_prototype_protocol_json'])['bank_mode']=='shared'
        assert len(hf['last_epoch_classes_json'])==200
    manifest=json.loads((final/'manifest.json').read_text())['last_epoch_prototypes']
    assert manifest['bank_mode']=='shared' and manifest['actual_prototype_bank_count']==1
    assert manifest['latest_generated_source_completed_train_count']==2
    assert manifest['latest_training_source_completed_train_count']==1 and manifest['latest_training_completed_train_count']==2
    for name,source in [('Last_epoch_shared_prototypes.packet',2),('Last_epoch_shared_training_prototypes.packet',1)]:
        raw=(final/name).read_bytes();v,m=decode_packet(raw,shared.KIND)
        assert m['source_completed_train_count']==source and v['prototypes'].shape==(100,512)
        assert packet_sha(raw)==libraries[source-1]['packet_sha256']
    assert not (final/'Last_epoch_columns_prototypes.packet').exists()
    for c in server.clients:
        for k in range(3):
            payload=torch.load(final/f'Client_{c.id}_model_{k}.pt',map_location='cpu',weights_only=False)
            assert payload['model_id']==k and payload['last_epoch_prototypes']['bank_mode']=='shared'
            assert payload['prototype_training_source_completed_train_count']==1
            assert payload['prototype_training_library_sha256']==received[0][3]
    receipt_path=out/label/'verification.json';receipt=json.loads(receipt_path.read_text())
    receipt.update(counts=counts,prototype_libraries=libraries,received=received,all_63_models_reloaded=True,
        independent_two_level_parameter_max_error=0.,prototype_rounds=server.post_proto_rounds)
    receipt_path.write_text(json.dumps(receipt,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--make-source',action='store_true',help=argparse.SUPPRESS);p.add_argument('--keep-models',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.make_source:
        from verify_multi_last_epoch_proto import make_source
        return make_source(out)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(83);data=out/'data/Cifar100/pat_fixture'
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%2 if split=='train' else 1+cid%4
            np.savez(data/split/f'{cid}.npz',data=dict(x=(rng.normal(size=(n,3,32,32))*.1).astype('float32'),y=(np.arange(n)+cid)%8))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    subprocess.run([sys.executable,__file__,'--output',str(out),'--make-source'],check=True,env=env)
    common=dict(K=3,rounds=2,gap=1,mode='group_projection_softmax',read=True,initialization='low_rank_recovered',init_rank_rate=.15)
    jobs=dict(on=dict(common,diagnostics=1),off=dict(common,diagnostics=0));(out/'jobs.json').write_text(json.dumps(jobs))
    for label in jobs:
        print('RUN SERIAL',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Shared-bank actual-entry smoke failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if owned==out or not owned.is_relative_to(out):raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    reports={label:json.loads((out/label/'verification.json').read_text()) for label in jobs}
    for key in ('events','batches','final_rng','primary_summary','prototype_libraries','received'):
        assert reports['on'][key]==reports['off'][key],('diagnostic isolation',key)
    receipt=dict(status='PASS',synthetic_only=True,clients=20,K=3,rounds=2,local_epochs=5,original_width=16,
        grad_clip=10,diagnostics_exact=True,independent_two_level_max_error=0.,one_bank_per_client=True,
        bank_shape=[100,512],shared_next_round_values_and_fingerprints_exact=True,
        cases={k:{key:r[key] for key in ('counts','training_tasks','all_63_models_reloaded','prototype_rounds')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(receipt,indent=2));print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
