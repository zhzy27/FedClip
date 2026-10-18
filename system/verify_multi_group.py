"""Actual-main synthetic acceptance. Isolated fixtures only; never real data.

Checks 20-client/K3 two-round paths, independent per-column group kernel,
strict grouping-file reuse, nondefault N/K/seed/capacities/sparse evaluation,
and observation on/off exact upload/server/RNG equality. Compact exports use
same-volume hardlinks only in this acceptance worker, not production training.
"""
import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker(output, label):
    sys.path.insert(0, str(ROOT/'system'))
    import torch
    from unittest.mock import patch
    from common.models import create_full_model
    from common.experiment import write_json
    from fedfews_compat.observe import rng_digest
    from utils.cnn512_reporting import model_digest
    from utils.group_projection import aggregate_group_updates
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    from flcore.clients.clientbase import Model_Distribe
    import shutil
    import h5py
    job = json.loads((output/'jobs.json').read_text())[label]
    if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
    events, created = [], []
    original_init = FedMultiGroupProj.__init__
    def observed(server, *a, **kw):
        original_init(server, *a, **kw)
        created.append(server)
        events.append(dict(kind='initial', parameters=[model_digest(m) for m in server.global_model_set], rng=rng_digest()))
        for client in server.clients:
            train = client.train
            def training(train=train, client=client):
                expected = [model_digest(m) for m in server.global_model_set]
                assert [model_digest(m) for m in client.model_set] == expected, 'Wrong column/base download'
                result = train()
                events.append(dict(kind='local', loop=server.loop_round, client=client.id,
                    uploads=[model_digest(m) for m in client.model_set], bases=expected, rng=rng_digest()))
                return result
            client.train = training
        aggregate = server.aggregate_parameters
        def checked_aggregation():
            total = sum(c.train_samples for c in server.clients)
            samples = {c.id:c.train_samples/total for c in server.clients}
            expected, expected_rows = [], []
            for k, model in enumerate(server.global_model_set):
                def uploads():
                    for c in server.selected_clients:yield c.id,samples[c.id],dict(c.model_set[k].named_parameters())
                result, metrics, rows, matrices = aggregate_group_updates(dict(model.named_parameters()), uploads,
                    samples, server.groups[k], server.args.multi_group_mode,
                    server.args.multi_group_weight if server.args.multi_group_mode=='group_projection_softmax' else None,
                    softmax_tau=server.args.group_softmax_tau)
                expected.append(result);expected_rows.append(rows)
            uploads_before = [[model_digest(m) for m in c.model_set] for c in server.clients]
            random_before = rng_digest()
            answer = aggregate()
            assert rng_digest() == random_before
            for k, model in enumerate(server.global_model_set):
                assert answer[k]['rows'] == expected_rows[k]
                for name, p in model.named_parameters():assert torch.equal(p,expected[k][name]), (k,name)
            assert uploads_before == [[model_digest(m) for m in c.model_set] for c in server.clients]
            events.append(dict(kind='aggregate', loop=server.loop_round,
                               parameters=[model_digest(m) for m in server.global_model_set], rng=rng_digest()))
            return answer
        server.aggregate_parameters = checked_aggregation
    out = output/label
    argv = ['main.py','-algo','FedMultiGroupProj','-m','CNN-512','-data','Cifar100','-ncl','100','-nc',str(job['N']),
        '-nsm',str(job['K']),'--seed',str(job['seed']),'-gr',str(job['rounds']-1),'-eg',str(job['gap']),
        '-ls',str(job['epochs']),'-lbs',str(job['batch']),'-lr',str(job['lr']),'-jr','1','-t','1','-pv','0',
        '-niid','1','-pt','pat','-cpc',str(job['cpc']),'-is_regular','0','--target_prox_mu','0',
        '--local_grad_clip',str(job['clip']),'--control_test_batch_size','3','-clip_cpu_threads','2','-dev','cpu',
        '--multi_group_mode',job['mode'],'--multi_group_weight',str(job['beta']),'--group_softmax_tau',str(job['tau']),
        '--multi_cluster_seed','19','--multi_cluster_restarts','3','--multi_cluster_iterations','15',
        '--data-root',str(output/job['fixture']),'--data-subdir','pat_fixture','--data-formats','npz_only',
        '-sfn',str(out/'checkpoints'),'--run_log_dir',str(out/'logs'),'--h5_result_root',str(out/'h5'),
        '--final_model_root',str(out/'final_models'),'-exp_name','multi_'+label]
    if job.get('sizes'):argv += ['--multi_group_sizes',job['sizes']]
    if job.get('group_file'):argv += ['--multi_group_file',str(output/job['group_file'])]
    if job.get('no_pre'):argv += ['--multi_no_pre_eval']
    original_copy = shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():
            os.link(src,dst);return dst
        return original_copy(src,dst,*a,**kw)
    sys.argv = argv
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedMultiGroupProj,'__init__',observed))
        stack.enter_context(patch.object(Model_Distribe,'__init__',side_effect=AssertionError('No low-rank constructor allowed')))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    final_rng = rng_digest()
    server = created[0];folder=Path(server.report_dir());summary=json.loads((folder/'primary_summary.json').read_text())
    due=[r for r in range(job['rounds']) if r%job['gap']==0 or r==job['rounds']-1]
    assert [p['loop_round'] for p in server.primary_records]==due
    assert len(server.training_records)==job['N']*job['K']*job['rounds']
    for p in server.training_records:
        client=server.clients[p['client_id']]
        assert p['processed_train_samples']==client.train_samples*job['epochs']
        assert p['train_batches']==math.ceil(client.train_samples/job['batch'])*job['epochs']
    for loop in due:
        selected=[r for r in server.endpoint_records if r['loop_round']==loop and r['stage']=='post' and r['selected_by_membership']]
        row=next(r for r in server.primary_records if r['loop_round']==loop)
        assert len(selected)==job['N'] and row['post_local_accuracy']==sum(r['correct'] for r in selected)/sum(r['test_samples'] for r in selected)
        for k in range(job['K']):
            members=[r for r in selected if r['model_id']==k]
            group=next(r for r in server.group_records if r['loop_round']==loop and r['model_id']==k)
            assert group['post_local_accuracy']==sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
    values=[r['post_local_accuracy'] for r in server.primary_records]
    assert summary['best']==max(values) and summary['best_loop_round']==due[values.index(max(values))]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist()==values
        assert json.loads(hf.attrs['summary_json'])==summary
        assert len(hf['local_training_json'])==len(server.training_records)
        assert json.loads(hf.attrs['groups_json'])==server.groups
    import csv
    with (folder/'primary_metrics.csv').open() as f:
        assert [float(r['post_local_accuracy']) for r in csv.DictReader(f)]==values
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    assert manifest['primary_summary']==summary and manifest['budget']['actual_optimizer_steps']==sum(r['train_batches'] for r in server.training_records)
    probe=torch.arange(3*32*32,dtype=torch.float32).reshape(1,3,32,32)/10000
    for k,model in enumerate(server.global_model_set):
        payload=torch.load(Path(server.final_model_dir())/f'Server_model_{k}.pt',map_location='cpu')
        restored=create_full_model(server.args);restored.load_state_dict(payload['state_dict'],strict=True)
        assert model_digest(restored)==model_digest(model)
        model.eval();restored.eval()
        assert torch.equal(restored(probe),model(probe))
    for c in server.clients:
        for k,model in enumerate(c.model_set):
            payload=torch.load(Path(server.final_model_dir())/f'Client_{c.id}_model_{k}.pt',map_location='cpu')
            restored.load_state_dict(payload['state_dict'],strict=True)
            assert model_digest(restored)==model_digest(model)
            model.eval()
            assert torch.equal(restored(probe),model(probe))
    write_json(out/'verification.json',dict(status='PASS',events=events,final_rng=final_rng,
        groups=server.group_payload,primary_summary=summary,logs=str(folder),
        training_tasks=len(server.training_records),max_column_parameter_error=0.,
        data_fingerprint=server.context['data_fingerprint'],initial_sha256=server.initial_hash))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    jobs={}
    for prefix,n,k,epochs,rounds,seed,gap,batch,lr,clip,beta,tau,cpc in [
        ('default',20,3,5,2,0,1,16,.005,10,.5,.2,20),
        ('custom',5,2,2,3,7,2,2,.012,0,.65,.4,3)]:
        directory=out/(prefix+'_data')/'Cifar100'/'pat_fixture';rng=np.random.default_rng(121)
        for split in ['train','test']:
            (directory/split).mkdir(parents=True)
            for cid in range(n):
                count=2+cid%3 if split=='train' else 1+cid%4
                np.savez(directory/split/f'{cid}.npz',data=dict(x=rng.standard_normal((count,3,32,32)).astype('float32'),
                    y=(np.arange(count)%min(3,cpc)+cid*3)%100))
        (directory/'config.json').write_text(json.dumps(dict(num_clients=n,num_classes=100,partition='pat',class_per_client=cpc,alpha=.1)))
        baseline=dict(N=n,K=k,epochs=epochs,rounds=rounds,seed=seed,gap=gap,batch=batch,lr=lr,clip=clip,beta=beta,tau=tau,
                      cpc=cpc,fixture=prefix+'_data',sizes='2,3' if n==5 else None,mode='group_projection_softmax')
        jobs[prefix+'_projection']=baseline
        jobs[prefix+'_avg_only']=dict(baseline,mode='group_avg_only',group_file=prefix+'_projection/fixed_groups.json')
        if prefix=='custom':jobs['custom_no_pre']=dict(baseline,no_pre=True,group_file='custom_projection/fixed_groups.json')
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    reports={}
    for label in jobs:
        env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{label} failed; see {out}/{label}.log')
        r=json.loads((out/label/'verification.json').read_text());reports[label]=r
        if label.endswith('_projection'):
            (out/label/'fixed_groups.json').write_text(json.dumps(r['groups'],indent=2))
        print('PASS',label,flush=True)
    for prefix in ['default','custom']:
        a,b=reports[prefix+'_projection'],reports[prefix+'_avg_only']
        assert a['groups']==b['groups']
        assert a['events'][0]==b['events'][0]
        assert [e for e in a['events'] if e['kind']=='local' and e['loop']==0]==[e for e in b['events'] if e['kind']=='local' and e['loop']==0]
    a,b=reports['custom_projection'],reports['custom_no_pre']
    for key in ['events','final_rng','groups','data_fingerprint','initial_sha256']:assert a[key]==b[key],key
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',cases=list(jobs),
        exact_column_kernel=True,observation_on_off_parameter_rng_parity=True,strict_group_reuse=True),indent=2))
    print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
