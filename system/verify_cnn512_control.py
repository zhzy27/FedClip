"""Two-round REAL main.py entry smoke on isolated synthetic pat_20 shards.

Never generates/replaces production data. Each algorithm runs in its own process.
The unchanged FedFewS mathematical regression is tested separately.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    import torch
    from common.experiment import write_json
    from common.models import create_full_model
    from utils.cnn512_reporting import model_digest
    from flcore.clients.clientbase import load_item
    from flcore.servers.serverfedfews import FedFewS
    from flcore.servers.serverTargetProj import FedTargetProj
    from unittest.mock import patch
    from contextlib import ExitStack
    population = getattr(args, 'population_protocol', False)
    created = []
    cls = FedFewS if args.kind == 'few' else FedTargetProj
    original = cls.__init__
    def observed(self, *a, **kw):
        original(self, *a, **kw)
        created.append(self)
        assert self.control_manifest['model'] == 'common.models.CNN_512'
        if args.kind != 'few':
            def forbidden(*a, **kw):
                raise AssertionError('Full path called recovery')
            self._recover_if_needed = forbidden
    cls.__init__ = observed
    run = Path(args.output)/args.kind
    argv = ['main.py','-algo','FedFewS' if args.kind=='few' else 'FedTargetProj', '-m','CNN-512',
        '-data','Cifar100','--data-root',str(Path(args.output)/'synthetic_data'),'--data-subdir','pat_20',
        '--data-formats','npz_only','--cnn512_control','--local_grad_clip','10','--seed','0','--target_prox_mu','0',
        '-nc','20','-ncl','100','-jr','1','-gr','1','-eg','1','-ls','5','-lbs','16','-lr','.005',
        '-is_regular','0','-dev',args.device,'-did',args.device_id,'-t','1','-pv','0',
        '-pt','pat','-cpc','20','-niid','1','-clip_cpu_threads','2','--fedfews_torch_threads','2',
        '-sfn',str(run/'checkpoints'),'--run_log_dir',str(run/'logs'),
        '--h5_result_root',str(run/'h5'),'--final_model_root',str(run/'final_models'),
        '-exp_name','cnn512_control_'+args.kind]
    if args.kind != 'few':
        argv += ['--target_proj_mode','avg' if args.kind=='avg' else
                 'group_projection_softmax' if args.kind=='group' else 'projection_softmax']
        if args.kind=='projection':
            argv += ['--projection_self_weight','.125']
    if population:
        argv += ['--control_eval_scope', 'group' if args.kind=='group' else 'all']
        if args.kind=='group':
            argv += ['--target_group_ids','0,1,2,3,4,6,7','--target_group_weight','.50']
    sys.argv = argv
    with ExitStack() as stack:
        stack.enter_context(patch('flcore.servers.serverTargetProj.Model_Distribe', side_effect=AssertionError('No model distribution')))
        stack.enter_context(patch('flcore.clients.clientbase.Model_Distribe', side_effect=AssertionError('No model distribution')))
        if getattr(args, 'compact', False):
            # Acceptance-only storage optimization: exported model files retain
            # the actual saved bytes, and are reloaded below. Training is over
            # before export. Production saving behavior is NOT changed.
            import shutil
            original_copy = shutil.copy2
            def copy_or_link(src, dst, **kw):
                destination = Path(dst)/Path(src).name if Path(dst).is_dir() else Path(dst)
                if Path(src).suffix == '.pt':
                    os.link(src,destination)
                    return str(destination)
                return original_copy(src,dst,**kw)
            stack.enter_context(patch('shutil.copy2',copy_or_link))
        runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    s = created[0]
    assert len(s.control_points)==2 and len(s.control_training)==40
    k = 3 if args.kind=='few' else 1
    for client in s.clients:
        assert client.train_time_cost['num_rounds']==2
        rows = [r for r in s.control_training if r['client_id']==client.id]
        assert all(r['processed_train_samples']==client.train_samples*5*k and r['train_batches']==5*k for r in rows)
    import h5py
    with h5py.File(s.args.save_file_paths[-1]) as hf:
        assert len(hf['cnn512_control/test_acc_percent'])==2
        assert len(hf['cnn512_control/training/train_batches'])==40
    if args.kind=='few':
        state=load_item('FedFewS','final',s.save_folder_name)
        assert len(state['server_models'])==3 and len(state['clients'])==20
        for weights in state['server_models']:
            create_full_model(s.args).load_state_dict(weights,strict=True)
        for i, point in enumerate(s.control_points):
            c0=s.evaluations[i]['clients'][0]
            assert point['correct']==c0['correct'] and point['samples']==c0['test_samples']
        assert state['model']=='common.models.CNN_512'
    else:
        model=load_item('Server','model',s.save_folder_name)
        assert type(model).__name__=='CNN_512' and not hasattr(model,'ratio_LR')
        create_full_model(s.args).load_state_dict(model.state_dict(),strict=True)
    for name in ['c0_summary.json','c0_post_local.csv','local_training.csv','data_manifest.json']:
        assert (Path(s.report_dir())/name).is_file()
    if population:
        verify_population_exports(s, args.kind)
    write_json(run/'smoke_report.json',dict(status='PASS', kind=args.kind, rounds=2, clients=20,
        gradient_cap=s.args.local_grad_clip, test_batch=s.args.test_batch_size,
        initial_sha256=s.control_manifest['initial_parameter_sha256'],
        data=json.loads((Path(s.report_dir())/'data_manifest.json').read_text()),
        model_parameters=sum(p.numel() for p in create_full_model(s.args).parameters()),
        data_epochs_per_client=k*5, logs=s.report_dir(), h5=s.args.save_file_paths, final_models=s.final_model_dir()))


def verify_population_exports(s, kind):
    import csv
    import h5py
    import numpy as np
    folder=Path(s.report_dir())
    points=json.loads((folder/'primary_metrics.json').read_text())
    clients=list(csv.DictReader((folder/'client_metrics.csv').open(encoding='utf-8')))
    summary=json.loads((folder/'primary_summary.json').read_text())
    ids=[0,1,2,3,4,6,7] if kind=='group' else list(range(20))
    expected=[]
    for point in points:
        records=[r for r in clients if int(r['loop_round'])==point['loop_round']]
        assert sorted(int(r['client_id']) for r in records)==ids
        assert len(set(float(r['test_samples']) for r in records))>1
        correct=sum(float(r['correct']) for r in records)
        count=sum(float(r['test_samples']) for r in records)
        # Reduce counts to a fraction first, then convert at the export boundary.
        # Multiplication-before-division differs by a float64 ULP for e.g. 9/19.
        expected.append(100.*(correct/count))
        assert point['test_acc_percent']==expected[-1]
        assert point['correct']==correct and point['test_samples']==count
    assert summary['best']==max(expected) and summary['best_loop_round']==expected.index(max(expected))
    assert summary['final']==expected[-1] and summary['last10']==float(np.mean(expected[-10:]))
    csv_points=list(csv.DictReader((folder/'primary_metrics.csv').open(encoding='utf-8')))
    assert [float(p['test_acc_percent']) for p in csv_points]==expected
    with h5py.File(s.args.save_file_paths[-1]) as hf:
        assert hf.attrs['primary_eval_scope']==('group' if kind=='group' else 'all')
        assert hf['primary_metrics'].attrs['best']==summary['best']
        assert hf['primary_metrics/test_acc_percent'][:].tolist()==expected
        assert hf['cnn512_control/test_acc_percent'][:].tolist()==expected
        assert hf['primary_metrics/client_ids'][:].tolist()==ids
        if kind=='group':
            assert hf['target_group'].attrs['primary_average']=='test_sample_weighted_average'
            assert hf['target_group'].attrs['primary_accuracy_metric']=='group_post_local_sample_weighted_acc'
            assert 'member_low_rank_ratio' not in hf['target_group']
            assert np.allclose(hf['target_group/group_post_local_sample_weighted_acc'][:]*100,expected,atol=1e-13,rtol=0)
            assert np.allclose(hf['target_group/actual_group_mass'][:],.5,atol=1e-15,rtol=0)
            assert np.allclose(hf['target_group/actual_outside_mass'][:],.5,atol=1e-15,rtol=0)
    for name in ['manifest.json','comparison_manifest.json']:
        manifest=json.loads((Path(s.final_model_dir())/name).read_text(encoding='utf-8'))
        assert manifest['primary_summary']==summary
    if kind=='few':
        assert [p['test_acc_percent'] for p in s.evaluations]==expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--device-id',default='0')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--kind',choices=['avg','projection','group','few'])
    parser.add_argument('--population-protocol',dest='population_protocol',action='store_true',
                        help='This batch: Avg/all, GroupProjection/group, FedFewS/all')
    parser.add_argument('--compact',action='store_true',help='Acceptance only: hardlink duplicate model exports on same volume')
    args=parser.parse_args()
    if args.worker:
        return worker(args)
    import numpy as np
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    args.output=str(out)
    directory=out/'synthetic_data/Cifar100/pat_20'
    rng=np.random.RandomState(321)
    for split in ['train','test']:
        (directory/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3  # Nonuniform counts; one minibatch, drop_last=False.
            np.savez(directory/split/f'{cid}.npz',data=dict(x=rng.randn(n,3,32,32).astype('float32'),y=(np.arange(n)+cid)%100))
    (directory/'config.json').write_text(json.dumps(dict(num_clients=20, num_classes=100, partition='pat',alpha=.1)))
    kinds=['avg','group','few'] if args.population_protocol else ['avg','projection','few']
    for kind in kinds:
        env=dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=args.device_id)
        with (out/f'{kind}.log').open('w',encoding='utf-8') as log:
            command=[sys.executable,__file__,'--worker','--kind',kind,'--output',str(out),
                '--device',args.device,'--device-id',args.device_id]
            if args.population_protocol: command+=['--population-protocol']
            if args.compact: command+=['--compact']
            result=subprocess.run(command,cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{kind} failed; inspect {out/kind}.log')
    reports=[json.loads((out/k/'smoke_report.json').read_text()) for k in kinds]
    assert len({r['initial_sha256'] for r in reports})==1
    assert reports[0]['data']==reports[1]['data']==reports[2]['data']
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS', real_data=False,
        initial_model_and_data_identical=True, reports=reports),indent=2),encoding='utf-8')
    print(f'PASS: three real entries, 20 clients, two rounds; {out}/acceptance_report.json')


if __name__=='__main__':
    main()
