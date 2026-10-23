"""Actual-main synthetic SVD acceptance; never reads/regenerates real shards.

20 clients, original-width Hyper_CNN_512, K3, both kernels, two inclusive rounds
and five epochs/model. A paired file fixes membership. Optional extra cases
verify observation isolation and the K1 all-client baseline. Model files are
removed only after save/reload verification unless --keep-models is specified.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker(out, label):
    sys.path.insert(0, str(ROOT/'system'))
    import h5py
    import numpy as np
    import torch
    from unittest.mock import patch
    from torch.utils.data import DataLoader
    from common.experiment import write_json, sha256
    from fedfews_compat.observe import rng_digest
    from utils.cnn512_reporting import model_digest
    from utils.group_projection import aggregate_group_updates
    from utils.multi_group_svd import load_checkpoint
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    import flcore.clients.clientMultiGroupProj as client_module
    job = json.loads((out/'jobs.json').read_text())[label]
    if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
    events, batches, created, context = [], [], [], {}
    def upload_digest(post):
        h = hashlib.sha256()
        for n, p in post.items(): h.update(n.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x, y in super().__iter__():
                if context.get('training'):
                    batches.append([context['loop'], context['client'], y.tolist()])
                yield x, y
    native_init = FedMultiGroupProj.__init__
    def observe_init(server, *a, **kw):
        native_init(server, *a, **kw); created.append(server)
        assert len({model_digest(m) for m in server.global_model_set}) == 1
        events.append(dict(kind='initial', parameters=[model_digest(m) for m in server.global_model_set], rng=rng_digest()))
        for c in server.clients:
            download, train = c.set_parameters, c.train
            def checked_download(servers, loop, c=c, download=download):
                download(servers, loop)
                assert len(c.model_set) == job['K']
                for k, model in enumerate(c.model_set):
                    assert model.ratio_LR == [.9,.37,.35,.25,.15][c.id%5]
                    assert hasattr(model.fc1,'weight_u') and hasattr(model.conv2,'conv_u')
                    assert torch.equal(model.fc3.weight, servers[k].fc3.weight)
                    assert torch.equal(model.conv1.weight, servers[k].conv1.weight)
            c.set_parameters = checked_download
            def checked_train(c=c, train=train):
                context.update(training=True, loop=server.loop_round, client=c.id)
                before = [model_digest(m) for m in c.model_set]
                try: result = train()
                finally: context['training'] = False
                for k, model in enumerate(c.model_set):
                    assert model_digest(model) != before[k]
                    payload = torch.load(Path(c.save_folder_name)/f'Client_{c.id}_model_{k}.pt',map_location='cpu',weights_only=False)
                    assert payload['role'] == 'ordinary_post_local_low_rank'
                    assert payload['capacity']['rank_rate'] == model.ratio_LR
                    assert payload['state_dict'].keys() == model.state_dict().keys()
                events.append(dict(kind='local', loop=server.loop_round, client=c.id,
                    local=[model_digest(m) for m in c.model_set],
                    uploads=[upload_digest(p) for p in c.full_uploads], rng=rng_digest()))
                return result
            c.train = checked_train
        initialize = server._initialize_groups
        def grouping():
            assert server.loop_round == 0 and server.groups is None
            assert all(c.upload_round == 0 for c in server.clients)
            assert len(server.training_records) == server.num_clients*server.K
            # Independent, unchunked per-parameter float64 Gram reductions.
            matrices = []
            for k in range(server.K):
                gram = torch.zeros((server.num_clients,server.num_clients),dtype=torch.float64)
                for name in dict(server.global_model_set[k].named_parameters()):
                    values = torch.stack([c.full_upload_parameters(k)[name].reshape(-1).double() for c in server.clients])
                    gram.add_(values@values.T)
                norms = gram.diag().sqrt(); matrices.append((gram/(norms[:,None]*norms[None,:])).numpy())
            initialize()
            np.testing.assert_allclose(server.group_payload['similarity_matrix'],np.mean(matrices,axis=0),rtol=0,atol=1e-12)
            np.testing.assert_allclose(server.group_payload['diagnostics']['per_model_cosine_matrices'],matrices,rtol=0,atol=1e-12)
            events.append(dict(kind='group', loop=0, groups=server.groups, rng=rng_digest()))
        server._initialize_groups = grouping
        aggregate = server.aggregate_parameters
        def checked_aggregate():
            expected = []
            total = sum(c.train_samples for c in server.clients)
            samples = {c.id:c.train_samples/total for c in server.clients}
            hashes = [sha256(Path(c.save_folder_name)/f'Client_{c.id}_model_{k}.pt') for c in server.clients for k in range(server.K)]
            for k, model in enumerate(server.global_model_set):
                def uploads():
                    for c in server.selected_clients: yield c.id,samples[c.id],c.full_upload_parameters(k,'cpu')
                expected.append(aggregate_group_updates(dict(model.named_parameters()),uploads,samples,server.groups[k],
                    server.args.multi_group_mode,.5 if server.args.multi_group_mode=='group_projection_softmax' else None))
            rng = rng_digest(); result = aggregate(); assert rng_digest() == rng
            assert hashes == [sha256(Path(c.save_folder_name)/f'Client_{c.id}_model_{k}.pt') for c in server.clients for k in range(server.K)]
            for k, model in enumerate(server.global_model_set):
                for n, p in model.named_parameters(): assert torch.equal(p,expected[k][0][n]), (k,n)
                assert result[k]['rows'] == expected[k][2]
            events.append(dict(kind='aggregate', loop=server.loop_round,
                parameters=[model_digest(m) for m in server.global_model_set], rng=rng_digest()))
            return result
        server.aggregate_parameters = checked_aggregate
    run = out/label
    sys.argv = ['main.py','-algo','FedMultiGroupProj','-m','Decom_CNN-5-512','-data','Cifar100','-ncl','100','-nc','20',
        '-nsm',str(job['K']),'-dev','cpu','-t','1','-pv','0','-jr','1','--seed','0','-lr','.005','-lbs','16','-ls','5',
        '-gr',str(job['rounds']-1),'-eg',str(job['gap']),'-is_regular','1','-regular_lamda','.001',
        '--target_prox_mu','0','--target_head_l2_lambda','0','--local_grad_clip','10','--control_test_batch_size','16',
        '-clip_cpu_threads','2','-niid','1','-pt','pat','-cpc','20','--data-root',str(out/'data'),
        '--data-subdir','pat_fixture','--data-formats','npz_only','--multi_svd','--multi_group_basis','full_weights',
        '--multi_warmup_rounds','0','--multi_group_mode',job['mode'],'--multi_group_weight','.5','--group_softmax_tau','.2',
        '--multi_group_sizes','20' if job['K']==1 else '7,7,6','--multi_cluster_seed','0',
        '--multi_svd_diagnostics',str(0 if job.get('no_observation') else 1),
        '-sfn',str(run/'checkpoints'),'--run_log_dir',str(run/'logs'),'--h5_result_root',str(run/'h5'),
        '--final_model_root',str(run/'final_models'),'-exp_name','multi_svd_'+label]
    if job.get('no_observation'): sys.argv += ['--multi_no_pre_eval']
    if job.get('read'): sys.argv += ['--multi_group_file',str(out/'pair.json')]
    if job.get('publish'): sys.argv += ['--multi_group_publish',str(out/'pair.json')]
    # Avoid duplicate tensor storage only in this synthetic acceptance worker.
    native_copy = shutil.copy2
    def compact(src, dst, *a, **kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():
            os.link(src,dst); return dst
        return native_copy(src,dst,*a,**kw)
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedMultiGroupProj,'__init__',observe_init))
        stack.enter_context(patch.object(client_module,'DataLoader',TraceLoader))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server = created[0]; final_rng = rng_digest()
    assert len(server.training_records) == 20*job['K']*job['rounds']
    assert len(batches) == sum(r['train_batches'] for r in server.training_records)
    due = [r for r in range(job['rounds']) if r%job['gap']==0 or r==job['rounds']-1]
    assert [p['loop_round'] for p in server.primary_records] == due
    for row in server.training_records:
        c = server.clients[row['client_id']]
        assert row['processed_train_samples'] == 5*c.train_samples
        assert row['train_batches'] == 5*math.ceil(c.train_samples/16)
        assert row['regularization_loss'] > 0
    for loop in due:
        selected = [r for r in server.endpoint_records if r['loop_round']==loop and r['stage']=='post' and r['selected_by_membership']]
        primary = next(r for r in server.primary_records if r['loop_round']==loop)
        assert primary['post_local_accuracy'] == sum(r['correct'] for r in selected)/sum(r['test_samples'] for r in selected)
        for table, field, key in [(server.group_records,'model_id','model_id'),(server.capacity_records,'capacity_level','client_id')]:
            for row in [r for r in table if r['loop_round']==loop]:
                members = [r for r in selected if (r[key] if field=='model_id' else r[key]%5)==row[field]]
                assert row['post_local_accuracy'] == sum(r['correct'] for r in members)/sum(r['test_samples'] for r in members)
    folder = Path(server.report_dir()); summary = json.loads((folder/'primary_summary.json').read_text())
    values = [r['post_local_accuracy'] for r in server.primary_records]
    assert summary['best'] == max(values) and summary['best_loop_round'] == due[values.index(max(values))]
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['rs_test_acc'][:].tolist() == values
        assert json.loads(hf.attrs['summary_json']) == summary
        assert len(hf['capacity_metrics_json']) == 5*len(due)
        assert len(hf['local_training_json']) == len(server.training_records)
    final = Path(server.final_model_dir()); manifest = json.loads((final/'manifest.json').read_text())
    assert manifest['primary_summary'] == summary
    probe = torch.zeros(1,3,32,32)
    for k, model in enumerate(server.global_model_set):
        m, payload = load_checkpoint(final/f'Server_model_{k}.pt')
        assert model_digest(m) == model_digest(model) and payload['model_kwargs']['ratio_LR'] == 1.
        assert torch.equal(m(probe),copy_model_eval(model,probe))
    for c in server.clients:
        for k, model in enumerate(c.model_set):
            restored, payload = load_checkpoint(final/f'Client_{c.id}_model_{k}.pt')
            assert model_digest(restored) == model_digest(model)
            assert torch.equal(restored(probe),copy_model_eval(model,probe))
            assert payload['assigned_model_id'] == int(server.group_payload['client_to_model'][str(c.id)])
    write_json(run/'verification.json',dict(status='PASS',events=events,batches=batches,final_rng=final_rng,
        groups=server.group_payload,primary_summary=summary,training_tasks=len(server.training_records),
        evaluation_loop_rounds=due, eval_gap=job['gap'], last_loop_forced=bool((job['rounds']-1)%job['gap']),
        max_column_parameter_error=0.,data_fingerprint=server.context['data_fingerprint'],
        initial_sha256=server.initial_hash,all_prediction_checkpoints_reloaded=True))


def copy_model_eval(model, probe):
    import copy
    import torch
    with torch.no_grad(): return copy.deepcopy(model).eval()(probe)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True); p.add_argument('--worker')
    p.add_argument('--pair-only',action='store_true'); p.add_argument('--keep-models',action='store_true')
    a = p.parse_args(); out = Path(a.output).resolve()
    if a.worker: return worker(out,a.worker)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    data = out/'data/Cifar100/pat_fixture'; rng = np.random.default_rng(57)
    for split in ['train','test']:
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n = 2+cid%2 if split=='train' else 1+cid%4
            np.savez(data/split/f'{cid}.npz',data=dict(x=rng.standard_normal((n,3,32,32)).astype('float32'),
                y=(np.arange(n)+cid*3)%100))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    base = dict(K=3,rounds=2,gap=1,mode='group_projection_softmax')
    jobs = dict(projection=dict(base,publish=True),avg_only=dict(base,mode='group_avg_only',read=True))
    if not a.pair_only:
        jobs.update(no_observation=dict(base,no_observation=True,read=True),
            avg_all=dict(base,K=1,rounds=3,gap=3,mode='group_avg_only'))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    reports = {}
    for label in jobs:
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result = subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Acceptance failed: {out}/{label}.log')
        reports[label] = json.loads((out/label/'verification.json').read_text())
        print('PASS',label,flush=True)
        if not a.keep_models:
            for name in ('checkpoints','final_models'):
                target = (out/label/name).resolve()
                if not target.is_relative_to(out) or target == out: raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(target)
    left,right = reports['projection'],reports['avg_only']
    assert all(r['initial_sha256']==left['initial_sha256'] and r['data_fingerprint']==left['data_fingerprint']
               for r in reports.values())
    assert left['groups'] == right['groups']
    first = lambda r: [e for e in r['events'] if e['kind']=='initial' or (e['kind']=='local' and e['loop']==0)]
    assert first(left) == first(right)
    if 'no_observation' in reports:
        for key in ('events','batches','final_rng','groups'): assert left[key] == reports['no_observation'][key],key
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',cases=list(jobs),
        actual_width=16,clients=20,epochs_per_model=5,first_uploads_and_grouping_exact=True,
        column_kernel_max_parameter_error=0.,full_weight_cosine_atol=1e-12,
        observation_parity='no_observation' in reports,model_files_retained=a.keep_models),indent=2))
    print('PASS',out/'acceptance_report.json')


if __name__=='__main__': main()
