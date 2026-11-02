"""N20/K3 real-main five-case smoke, with zero/diagnostics and fixed source reuse."""
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
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import h5py
    import torch
    from utils import single_avg_proto as shared,multi_last_epoch_proto as post
    from utils.multi_group_proto import decode_packet
    from utils.head_snapshot import tensor_fingerprint
    from common.experiment import sha256
    from fedfews_compat.observe import rng_digest
    from flcore.trainmodel.models import Hyper_CNN_512
    from verify_multi_group_svd import worker as svd_worker
    job=json.loads((out/'jobs.json').read_text())[label]
    counts=dict(forwards=0,backwards=0,steps=0,clips=0);context=dict(training=False);created=[];libraries=[]
    original_run=runpy.run_path
    def main_entry(path,*a,**kw):
        argv=sys.argv;argv[argv.index('--local_grad_clip')+1]='0'
        argv+=['--multi_last_epoch_protocol','--multi_group_reuse_policy','members_only',
            '--multi_proto_mode','last_epoch' if job['proto'] or job.get('zero') else 'none',
            '--multi_proto_loss','mse','--multi_proto_target_type','raw_mean' if job['proto'] or job.get('zero') else 'fixed_norm',
            '--multi_proto_lambda',str(1. if job['proto'] else 0.),'--multi_proto_scope',job['scope'],
            '--multi_proto_feature','base_output','--multi_proto_diagnostics',str(job.get('diagnostics',1))]
        return original_run(path,*a,**kw)
    original_shared=shared.train
    def column(client,column=0,**kw):
        before=counts['forwards'];context.update(training=True,column=column)
        try:row,packet=original_shared(client,column,**kw)
        finally:context['training']=False
        assert counts['forwards']-before==row['train_batches']==5
        assert row['effective_proto_lambda']==(1. if job['proto'] and client.download_round>0 else 0.)
        assert row['used_proto_source_completed_train_count']==(client.download_round if job['proto'] and client.download_round>0 else None)
        assert row['prototype_collected_samples']==(client.train_samples if job['proto'] else 0)
        if not job['proto']:assert packet is None and row['prototype_statistics_bytes']==0
        return row,packet
    original_forward=Hyper_CNN_512.forward;original_backward=torch.Tensor.backward;original_step=torch.optim.SGD.step
    def forward(model,*a,**kw):
        if context['training']:counts['forwards']+=1
        return original_forward(model,*a,**kw)
    def backward(t,*a,**kw):
        if context['training']:counts['backwards']+=1
        return original_backward(t,*a,**kw)
    def step(o,*a,**kw):
        if context['training']:counts['steps']+=1
        return original_step(o,*a,**kw)
    def no_clip(*a,**kw):
        counts['clips']+=1;raise AssertionError('Unclipped protocol must never call clip_grad_norm_')
    source_hash=sha256(out/'pair.json');finish_original=post.finish_round
    def finish(server):
        if not created:created.append(server)
        raw=[]
        if job['proto']:
            for c in server.clients:
                entries=[]
                for k,packet in enumerate(c.post_proto_local_packets):
                    values,meta=decode_packet(packet,'last_epoch_column_local')
                    assert meta['model_id']==k and int(values['counts'].sum())==c.train_samples
                    entries.append(values)
                raw.append(entries)
        rng=rng_digest();finish_original(server);assert rng==rng_digest()
        assert source_hash==sha256(out/'pair.json')
        if job['proto']:
            values,meta=decode_packet(server.post_proto_packet,'last_epoch_columns_library')
            assert meta['source_completed_train_count']==server.loop_round+1 and meta['K']==3
            expected=torch.zeros_like(values['prototypes'])
            for k in range(3):
                for c in range(100):
                    owners=[i for i in range(20) if int(raw[i][k]['counts'][c])>0]
                    inside=[i for i in owners if i in server.groups[k]]
                    chosen=owners if job['scope']=='global' else inside or [i for i in owners if i not in server.groups[k]]
                    if chosen:
                        for i in chosen:expected[k,c]+=raw[i][k]['sums'][c]/raw[i][k]['counts'][c]
                        expected[k,c]/=len(chosen)
                    assert bool(values['valid'][k,c])==bool(chosen)
            assert torch.equal(expected,values['prototypes'])
            libraries.append(tensor_fingerprint(values))
        else:assert server.post_proto_packet is None
    with ExitStack() as stack:
        for obj,name,value in [(runpy,'run_path',main_entry),(shared,'train',column),(post,'finish_round',finish),
            (Hyper_CNN_512,'forward',forward),(torch.Tensor,'backward',backward),(torch.optim.SGD,'step',step),
            (torch.nn.utils,'clip_grad_norm_',no_clip)]:stack.enter_context(patch.object(obj,name,value))
        if not job['proto']:
            stack.enter_context(patch.object(shared,'LastEpochStats',side_effect=AssertionError('Disabled statistics')))
            stack.enter_context(patch('utils.multi_group_proto.encode_packet',side_effect=AssertionError('Disabled prototype communication')))
        svd_worker(out,label)
    server=created[0];folder=Path(server.report_dir());final=Path(server.final_model_dir())
    assert counts==dict(forwards=600,backwards=600,steps=600,clips=0)
    assert len(server.training_records)==120
    assert sha256(out/'pair.json')==source_hash
    rows=json.loads((folder/'last_epoch_training.json').read_text());assert rows==server.training_records
    with (folder/'last_epoch_training.csv').open(encoding='utf-8',newline='') as stream:
        assert [float(r['total_loss']) for r in csv.DictReader(stream)]==[r['total_loss'] for r in rows]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert [json.loads(r) for r in hf['last_epoch_training_json'][:]]==rows
        assert [json.loads(r) for r in hf['last_epoch_classes_json'][:]]==server.post_proto_classes
    manifest=json.loads((final/'manifest.json').read_text())
    assert manifest['last_epoch_prototypes']['group_source']['source_file_sha256']==source_hash
    for c in server.clients:
        for k in range(3):
            payload=torch.load(final/f'Client_{c.id}_model_{k}.pt',map_location='cpu',weights_only=False)
            assert payload['group_source_sha256']==source_hash
            assert payload['assigned_model_id']==int(server.group_payload['client_to_model'][str(c.id)])
            assert payload['last_epoch_prototypes']['contributor_scope']==job['scope']
            assert payload['prototype_training_source_completed_train_count']==(1 if job['proto'] else None)
            assert payload['prototype_statistics_source_completed_train_count']==(2 if job['proto'] else None)
    path=out/label/'verification.json';receipt=json.loads(path.read_text())
    receipt.update(counts=counts,prototype_library_tensor_hashes=libraries,group_source_sha256=source_hash,
        group_reuse_receipt=server.group_reuse_receipt,all_63_models_reloaded=True,
        proto_rounds=server.post_proto_rounds,prototype_class_count=len(server.post_proto_classes))
    path.write_text(json.dumps(receipt,indent=2))


def make_source(out):
    import numpy as np
    import torch
    from common.client_data import audit_image_clients
    from utils.multi_group_svd import initial_model,DEFAULT_RATIOS
    from utils.cnn512_reporting import model_digest
    from utils.multi_group_clustering import grouping_context,balanced_cluster,digest_json
    from utils.multi_group_protocol import grouping_protocol
    torch.set_num_threads(2)
    a=SimpleNamespace(algorithm='FedMultiGroupProj',multi_svd=True,model_family='Decom_CNN-5-512',num_classes=100,
        seed=0,num_clients=20,num_server_models=3,multi_capacities=[7,7,6],local_learning_rate=.005,batch_size=16,
        local_epochs=5,local_grad_clip=10,multi_group_basis='full_weights',multi_warmup_rounds=0,multi_history_window=None,
        multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=.15,multi_svd_effective_ratios=list(DEFAULT_RATIOS),
        multi_svd_effective_assignment=[i%5 for i in range(20)],is_regular=1,regular_lamda=.001,
        data_root=str(out/'data'),data_subdir='pat_fixture',dataset='Cifar100',cnn512_control=True,data_formats='npz_only',
        partition='pat',class_per_client=20,data_expected_partition='pat',data_expected_alpha=None)
    context=grouping_context(a,audit_image_clients(a),model_digest(initial_model(a)))
    matrix=np.eye(20);groups,algorithm=balanced_cluster(matrix,[7,7,6],seed=7,restarts=2,iterations=4)
    proto=grouping_protocol(a);row=dict(loop_round=0,completed_train_count=1,matrix=matrix.tolist(),
        matrix_sha256=digest_json(matrix),diagnostics=dict(per_model_cosine_matrices=[matrix.tolist()]*3))
    evidence=dict(protocol=proto,rounds=[row],final_similarity_sha256=digest_json(matrix));evidence['evidence_sha256']=digest_json(evidence)
    payload=dict(schema_version=2,context=context,context_sha256=digest_json(context),groups=groups,
        group_fingerprint=digest_json(dict(context=context,groups=groups)),client_to_model={str(i):k for k,g in enumerate(groups) for i in g},
        grouping_protocol=proto,evidence=evidence,similarity_matrix=matrix.tolist(),completed_train_count=1,algorithm=algorithm,diagnostics={})
    (out/'pair.json').write_text(json.dumps(payload,indent=2))


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text());reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    base=reports['01']
    for label,r in reports.items():
        for key in ('initial_sha256','data_fingerprint','groups','batches','final_rng','group_source_sha256'):
            assert r[key]==base[key],(label,key)
    for a,b in [('02','zero'),('04','diag0')]:
        for key in ('events','batches','primary_summary','final_rng'):assert reports[a][key]==reports[b][key],(a,b,key)
    for a,b in [('01','03'),('02','04'),('02','05')]:
        first=lambda r:[e for e in r['events'] if e['kind']=='initial' or e.get('loop')==0]
        assert first(reports[a])==first(reports[b]),(a,b,'same-kernel first round')
    assert reports['04']['prototype_library_tensor_hashes'][0]!=reports['05']['prototype_library_tensor_hashes'][0]
    receipt=dict(status='PASS',synthetic_only=True,clients=20,K=3,rounds=2,local_epochs=5,actual_width=16,
        first_round_same_aggregation_proto_toggle_exact=True,zero_coefficient_exact=True,diagnostics_exact=True,
        all_initialization_data_groups_column_batches_RNG_exact=True,independent_equal_mean_max_error=0.,old_column_kernel_max_error=0.,
        source_file_unchanged=True,group_and_global_contributors_differ=True,
        cases={k:{key:r[key] for key in ('counts','training_tasks','all_63_models_reloaded','proto_rounds')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(receipt,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--jobs',type=int,default=1);p.add_argument('--keep-models',action='store_true');p.add_argument('--audit-only',action='store_true')
    p.add_argument('--make-source',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=Path(a.output).resolve()
    if a.worker:return worker(out,a.worker)
    if a.make_source:return make_source(out)
    if a.audit_only:return audit(out)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(83);data=out/'data/Cifar100/pat_fixture'
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%2 if split=='train' else 1+cid%4
            np.savez(data/split/f'{cid}.npz',data=dict(x=(rng.normal(size=(n,3,32,32))*.1).astype('float32'),y=(np.arange(n)+cid)%8))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    # Keep Torch/model imports in a short-lived process; the scheduler itself
    # must not retain a large reference-model runtime during the smoke workers.
    subprocess.run([sys.executable,__file__,'--output',str(out),'--make-source'],check=True,
        env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    modes=['group_avg_only','group_projection_softmax','group_avg_only','group_projection_softmax','group_projection_softmax']
    jobs={f'{i+1:02}':dict(K=3,rounds=2,gap=1,mode=mode,read=True,initialization='low_rank_recovered',init_rank_rate=.15,
        proto=i>=2,scope='global' if i==4 else 'group_priority') for i,mode in enumerate(modes)}
    jobs['zero']=dict(jobs['02'],zero=True);jobs['diag0']=dict(jobs['04'],diagnostics=0)
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'K-column smoke failed: {out}/{label}.log')
        if not a.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:list(pool.map(run,jobs))
    audit(out)


if __name__=='__main__':main()
