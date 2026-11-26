"""Exercise both aggregators and representative schedules on disposable N20 synthetic data."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from unittest.mock import patch


SYSTEM = Path(__file__).resolve().parent
CASES = (1, 6, 2, 10)


def worker(out, case):
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from flcore.clients import clientMultiGroupProj as client_module
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    from utils.cnn512_reporting import model_digest
    from utils.head_snapshot import tensor_fingerprint
    from utils.multi_group_svd import load_checkpoint
    from utils.group_projection import aggregate_group_updates
    from utils.multi_rank_partition import partition_product, factors

    run = out/f'{case:02}'
    commands = [shlex.split(line.strip()[1:-1]) for line in (SYSTEM/'fc1_rank_schedule_commands.sh').read_text().splitlines()
                if line.strip().startswith('"')]
    argv = commands[case-1][3:]
    for flag, value in {'-did':0, '-dev':'cpu', '-gr':1, '--data-root':out/'data', '--data-subdir':'fixture',
          '-clip_cpu_threads':2, '-sfn':run/'checkpoints', '--run_log_dir':run/'logs',
          '--h5_result_root':run/'h5', '--final-model-root':run/'models'}.items():
        argv[argv.index(flag)+1] = str(value)
    servers, batches, events = [], [], []
    active_client = None
    constructors = FedMultiGroupProj.__init__
    native_svd = torch.linalg.svd
    svd_calls = 0

    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if active_client is not None:
                    batches.append([*active_client,x[:,0,0,0].tolist()])
                yield x,y

    def counted_svd(*args, **kwargs):
        nonlocal svd_calls
        svd_calls += 1
        return native_svd(*args, **kwargs)

    def init(server, *args, **kwargs):
        constructors(server, *args, **kwargs)
        servers.append(server)
        for c in server.clients:
            def download(models, loop, *, fc1_svd, client=c, native=c.set_parameters):
                native(models, loop, fc1_svd=fc1_svd)
                events.append(['download',loop,client.id,model_digest(client.model_set[0])])
            def train(client=c, native=c.train):
                nonlocal active_client
                active_client = [client.download_round,client.id]
                try:
                    native()
                finally:
                    active_client = None
                row = client.train_stats[0]
                assert row['train_batches']==5 and row['processed_train_samples']==5*client.train_samples
                assert sum(r['batches'] for r in row['fc1_rank_by_batch'])==5
                assert sum(r['sample_visits'] for r in row['fc1_rank_by_batch'])==5*client.train_samples
                assert not hasattr(client.model_set[0].fc1,'_fc1_training_prefix') and client.avg_proto_local is None
                events.append(['upload',client.download_round,client.id,tensor_fingerprint(client.full_uploads[0])])
            c.set_parameters, c.train = download, train
        original_aggregate = server.aggregate_parameters
        def aggregate():
            sample_counts = {c.id:c.train_samples for c in server.clients}
            weights = {cid:n/sum(sample_counts.values()) for cid,n in sample_counts.items()}
            uploads = lambda: iter((c.id,weights[c.id],c.full_upload_parameters(0)) for c in server.selected_clients)
            mother = server.global_model_set[0]
            expected = aggregate_group_updates(dict(mother.named_parameters()),uploads,weights,list(sample_counts),'group_avg_only')[0]
            if server.args.multi_svd_aggregation == 'rank_partition':
                entries = [dict(client_id=c.id,**factors(c.model_set[0].fc1)) for c in server.selected_clients]
                expected['fc1.weight'] = partition_product(entries,sample_counts)[0]
            original_aggregate()
            for name, param in mother.named_parameters():
                assert torch.equal(expected[name],param), name
            events.append(['aggregate',server.loop_round,model_digest(mother)])
        server.aggregate_parameters = aggregate

    sys.argv = ['main.py', *argv]
    with patch.object(FedMultiGroupProj,'__init__',init), patch.object(client_module,'DataLoader',TraceLoader), patch.object(torch.linalg,'svd',counted_svd):
        runpy.run_path(str(SYSTEM/'main.py'),run_name='__main__')
    server = servers[0]
    assert svd_calls==2 and len(batches)==20*2*5
    assert [r['post_local_accuracy'] for r in server.primary_records]==[sum(r['correct'] for r in server.endpoint_records if r['loop_round']==t and r['stage']=='post')/sum(c.test_samples for c in server.clients) for t in range(2)]
    assert len(server.capacity_records)==10
    report = Path(server.report_dir())
    saved = json.loads((report/'fc1_download_spectrum.json').read_text())
    assert saved==server.fc1_download_records and len(saved)==2
    assert (report/'fc1_download_spectrum.csv').is_file()
    if server.args.multi_svd_aggregation=='rank_partition':
        assert len(server.rank_partition_records)==10 and all(r['layer']=='fc1' for r in server.rank_partition_records)
        assert all(len(row['post_aggregation_spectrum'])==800 for row in saved)
    else:
        assert [row['correction_applied'] for row in saved]==[False,True]
        row = saved[1]
        assert row['rank_boundaries']==[120,200,280,296,720]
        for j in range(800):
            mass=sum(n for r,n in zip(row['client_ranks'],row['original_train_samples']) if r>j)
            gain=sum(row['original_train_samples'])/mass if mass else 1.
            assert abs(row['original_position_applied_gain'][j]-gain)<2e-6
        assert row['sorted_spectrum']==[row['amplified_before_sort'][j-1] for j in row['sorted_to_original_rank_one_based']]
    with h5py.File(server.args.save_file_paths[-1]) as h5:
        assert [json.loads(r) for r in h5['fc1_download_spectrum_json'][:]]==saved
        assert len(h5['local_training_json'])==40
        assert list(h5['rs_test_acc'][:])==server.rs_test_acc
    final = Path(server.final_model_dir())
    for path in sorted(final.glob('*_model_0.pt')):
        loaded, payload = load_checkpoint(path)
        assert loaded.factorized_layers==('fc1',) and payload['completed_train_count']==2
        expected=server.global_model_set[0] if payload['role']=='post_aggregation_full_server' else server.clients[payload['client_id']].model_set[0]
        assert model_digest(expected)==model_digest(loaded)
    assert len(list(final.glob('*_model_0.pt')))==21
    result = dict(case=case, initial=server.initial_hash,data=server.context['data_fingerprint'],
                  assignment=server.args.multi_svd_effective_assignment, events=events, batches=batches,
                  download_svd_calls=svd_calls, checkpoint_reloads=21,metrics=str(report))
    (run/'verification.json').write_text(json.dumps(result,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--case',type=int,choices=CASES)
    args = parser.parse_args()
    out=Path(args.output).resolve()
    if args.case is not None:
        return worker(out,args.case)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    source=out/'data/Cifar100/fixture'
    random=np.random.default_rng(14)
    for split in ('train','test'):
        (source/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3 if split=='train' else 1+cid%4
            x=(random.normal(size=(n,3,32,32))*.1).astype('float32')
            x[:,0,0,0]=np.arange(n)/10
            np.savez(source/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)+cid)%6))
    (source/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    runs=[]
    for case in CASES:
        print('RUN',case,flush=True)
        with (out/f'{case:02}.log').open('x',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-u',__file__,'--output',str(out),'--case',str(case)],cwd=SYSTEM,check=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),
                stdout=stream,stderr=subprocess.STDOUT)
        runs.append(json.loads((out/f'{case:02}'/'verification.json').read_text()))
        print('PASS',case,flush=True)
    assert all(run[k]==runs[0][k] for run in runs for k in ('initial','data','assignment','batches'))
    initial_round = lambda run,kind:[r for r in run['events'] if r[0]==kind and r[1]==0]
    assert all(initial_round(run,'download')==initial_round(runs[0],'download') for run in runs)
    assert initial_round(runs[0],'upload')==initial_round(runs[1],'upload')
    assert initial_round(runs[0],'aggregate')!=initial_round(runs[1],'aggregate')
    assert [r for r in runs[0]['events'] if r[:2]==['download',1]]!=[r for r in runs[1]['events'] if r[:2]==['download',1]]
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',cases=CASES,clients=20,rounds=2,local_epochs=5,
        initial_mother_sha256=runs[0]['initial'],data_fingerprint=runs[0]['data'],assignment=runs[0]['assignment'],
        identical_batches_and_R1_downloads=True,baseline_R1_uploads_exact=True,distinct_aggregation_R1_and_download_R2=True,
        actual_entry_SVD_count_per_round=1,ckpt_reloads_per_case=21,steps_per_case=200)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2))
    print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':
    main()
