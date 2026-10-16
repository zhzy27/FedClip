"""Fixed c73ec5b compatibility vs REAL main.py native entry; short runs only.

Reference code is exported from Git into the output, never edited. Worker
processes isolate imports/RNG. Observation hooks are acceptance-only and replay
no loaders. Production FedFewS imports none of this or the compatibility package.
"""
import argparse
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'c73ec5b'


def load_pinned(job, name):
    directory = Path(job['pinned']) / 'system/fedfews_compat'
    spec = importlib.util.spec_from_file_location(name, directory / '__init__.py', submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker(kind, job):
    os.environ['CUDA_VISIBLE_DEVICES'] = job['device_id']
    if kind == 'source_repeat':
        repeated = dict(job, output=str(Path(job['output'])/'source_repeat'))
        Path(repeated['output']).mkdir()
        worker('reference', repeated)
        from fedfews_compat.acceptance import Comparison, torch_load
        from fedfews_compat.metadata import write_json
        comparison = Comparison(Path(job['output'])/'repeat_scratch', job['atol'], job['rtol'])
        comparison.directory = Path(job['output'])/'reference_states'
        for path in sorted((Path(repeated['output'])/'reference_states').glob('event_*.pt')):
            actual = torch_load(path)
            comparison.migrated(actual['record'], actual['states'])
        report = comparison.report()
        report.update(reference_commit=job['reference_commit'], job=job,
                      comparison='Fixed c73ec5b vs ITSELF; repeatability diagnostic, NOT native acceptance')
        write_json(Path(job['output'])/'acceptance_report.json', report)
        print(json.dumps({k:report[k] for k in ['status','event_count','max_parameter_error','max_scalar_error']},indent=2))
        if report['status'] != 'PASS':
            raise SystemExit(1)
        return
    if kind == 'reference':
        sys.path.insert(0, str(Path(job['pinned']) / 'system'))
        from fedfews_compat.config import parse_args
        from fedfews_compat.data import audit_data
        from fedfews_compat.runtime import build_server, prepare_random
        from fedfews_compat.observe import rng_digest
        from fedfews_compat.metadata import write_json
        import torch
        args, _ = parse_args(['--data-root', job['data_root'], '--output-dir', job['output'] + '/reference',
             '-data', job['dataset'], '-nc', str(job['num_clients']), '-gr', str(job['rounds']-1),
             '-eg', str(job['eval_gap']), '-lbs', str(job['batch_size']), '-ls', str(job['local_epochs']),
             '-lr', str(job['lr']), '-nsm', str(job['K']), '-smu', str(job['smooth_mu']),
             '-dev', job['device'], '-did', job['device_id'], '--rng-mode', 'controlled',
             '--control-seed', str(job['seed']), '--torch-threads', str(job['threads']), '--trace-batches'])
        args.expected_alpha, args.expected_partition = job['expected_alpha'], job['expected_partition']
        write_json(Path(job['output'])/'reference_data_manifest.json', audit_data(args))
        prepare_random(args)
        if job['device'] == 'cuda':
            torch.empty(0, device='cuda')
        directory = Path(job['output']) / 'reference_states'
        directory.mkdir()
        def sink(record, states):
            torch.save(dict(record=record, states=states), directory / f"event_{record['event']:05d}.pt")
        server, recorder, cleanup = build_server(args, Path(job['output'])/'reference', event_sink=sink)
        try:
            server.train()
            write_json(Path(job['output'])/'reference_final_rng.json', rng_digest())
        finally:
            cleanup()
        return

    sys.path.insert(0, str(ROOT/'system'))
    # Only the verifier loads the frozen Recorder/Comparison. Native training is
    # still run by main.py and its NEW classes, not by the compatibility runner.
    load_pinned(job, 'fedfews_pinned')
    from fedfews_pinned.acceptance import Comparison, torch_load
    from fedfews_pinned.observe import Recorder, rng_digest
    from fedfews_pinned.metadata import write_json
    from flcore.clients import clientfedfews
    from flcore.servers.serverfedfews import FedFewS
    import torch
    if job['device'] == 'cuda':
        torch.empty(0, device='cuda')
    comparison = Comparison(Path(job['output'])/'comparison_scratch', job['atol'], job['rtol'])
    comparison.directory = Path(job['output'])/'reference_states'
    recorder = Recorder(Path(job['output'])/'native_observation', trace_batches=True, event_sink=comparison.migrated)
    clientfedfews.DataLoader = recorder.loader_class()
    init = FedFewS.__init__
    created = []
    def observed_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        recorder.attach(self)
        created.append(self)
    FedFewS.__init__ = observed_init
    native_root = Path(job['output'])/'native'
    argv = ['main.py', '-algo', 'FedFewS', '-m', 'FedAvgCNN-plain', '-data', job['dataset'],
            '--data-root', job['data_root'], '--data-formats', 'pkl_first', '-ncl', '100', '-nc', str(job['num_clients']),
            '-gr', str(job['rounds']-1), '-eg', str(job['eval_gap']), '-lbs', str(job['batch_size']),
            '-ls', str(job['local_epochs']), '-lr', str(job['lr']), '-nsm', str(job['K']), '-smu', str(job['smooth_mu']),
            '-dev', job['device'], '-did', job['device_id'], '--fedfews_rng', 'controlled',
            '--fedfews_control_seed', str(job['seed']), '--fedfews_torch_threads', str(job['threads']),
            '-sfn', str(native_root/'checkpoints'), '--run_log_dir', str(native_root/'logs'),
            '--h5_result_root', str(native_root/'h5'), '--final_model_root', str(native_root/'final_models')]
    if job['expected_alpha'] is not None:
        argv += ['--data-expected-alpha', str(job['expected_alpha'])]
    if job['expected_partition']:
        argv += ['--data-expected-partition', job['expected_partition'], '-pt', job['expected_partition']]
    sys.argv = argv
    runpy.run_path(str(ROOT/'system/main.py'), run_name='__main__')
    server = created[0]
    expected_data = json.loads((Path(job['output'])/'reference_data_manifest.json').read_text(encoding='utf-8'))
    actual_data = json.loads(json.dumps(server.data_manifest))
    comparison.compare_value(expected_data['actual_alpha'], actual_data['actual_alpha'], 'actual data alpha')
    comparison.compare_value(expected_data['config_sha256'], actual_data['config_sha256'], 'data config bytes')
    fields = ['client_id','split','path','format','sha256','count','labels','x_dtype','y_dtype',
              'decoded_x_sha256','decoded_y_sha256','npz_shadowed_by_pkl']
    comparison.compare_value([{k:r[k] for k in fields} for r in expected_data['files']],
                             [{k:r[k] for k in fields} for r in actual_data['files']], 'selected/decoded data')
    comparison.compare_value(json.loads((Path(job['output'])/'reference_final_rng.json').read_text()), rng_digest(), 'final RNG')
    reference = torch_load(Path(job['output'])/'reference/models/FedFewS_final.pt')
    actual = torch_load(Path(server.save_folder_name)/'FedFewS_final.pt')
    for ref, current in zip(reference['clients'], actual['clients']):
        for key in ['client_id', 'best_model_idx', 'deployed_model_is_post_local', 'optimizers', 'schedulers']:
            comparison.compare_value(ref[key], current[key], f"final client {ref['client_id']}.{key}")
    # Also inspect actual persisted tensors, not merely in-memory event snapshots.
    def compare_saved(left, right, where):
        if left.keys() != right.keys():
            comparison.failures.append(where + ' saved parameter keys differ')
            return
        for name in left:
            if not torch.allclose(left[name], right[name], atol=job['atol'], rtol=job['rtol']):
                comparison.failures.append(where + '.' + name + ' saved parameters differ')
    for k, (left, right) in enumerate(zip(reference['server_models'], actual['server_models'])):
        compare_saved(left, right, f'saved server {k}')
    for ref, current in zip(reference['clients'], actual['clients']):
        for k, (left, right) in enumerate(zip(ref['models'], current['models'])):
            compare_saved(left, right, f"saved client {ref['client_id']} model {k}")
    reference_events = [torch_load(p)['record'] for p in sorted(comparison.directory.glob('event_*.pt'))]
    evaluations = [e['loop_round'] for e in reference_events if e['name']=='evaluation']
    if len(evaluations) == job['rounds'] or evaluations[-1] != job['rounds']-1:
        comparison.failures.append('Must cover a skipped evaluation and forced final evaluation')
    report = comparison.report()
    report.update(reference_commit=job['reference_commit'], comparison='Fixed c73ec5b compatibility vs actual main.py native entry',
                  job=job, main_argv=argv, evaluation_rounds=evaluations, log_dir=server.report_dir(),
                  final_models=server.final_model_dir(), h5=server.args.save_file_paths,
                  reference_untouched=True, real_data=job['real_data'])
    write_json(Path(job['output'])/'acceptance_report.json', report)
    print(json.dumps({k:report[k] for k in ['status','event_count','max_parameter_error','max_scalar_error']}, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


def main():
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        return worker(sys.argv[2], json.loads(Path(sys.argv[3]).read_text(encoding='utf-8')))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root')
    parser.add_argument('--dataset', default='Cifar100/noniid_dir_20_a0p5')
    parser.add_argument('--num-clients', type=int, default=20)
    parser.add_argument('--expected-alpha', type=float)
    parser.add_argument('--expected-partition', choices=['dir','pat'])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--eval-gap', type=int, default=10)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--local-epochs', type=int, default=1)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--smooth-mu', type=float, default=.01)
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--device-id', default='0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--atol', type=float, default=0)
    parser.add_argument('--rtol', type=float, default=0)
    parser.add_argument('--reference-repeat', action='store_true', help='Diagnostic: fixed c73ec5b vs itself, NOT native acceptance')
    args = parser.parse_args()
    import math
    if not 2 <= args.rounds <= 5 or args.eval_gap < 2 or any(not math.isfinite(v) or v < 0 for v in [args.atol,args.rtol]):
        parser.error('Require 2..5 actual rounds, eval-gap>=2, finite nonnegative tolerances')
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(['git','-C',str(ROOT),'archive','--format=zip',REFERENCE,'system/fedfews_compat','system/common'])
    pinned = output/'pinned_c73ec5b'
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        z.extractall(pinned)
    (output/'reference_source_sha256.json').write_text(json.dumps(
        {str(p.relative_to(pinned)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(pinned.rglob('*')) if p.is_file()}, indent=2), encoding='utf-8')
    # Synthetic fixture creation uses the FROZEN helper and never existing data.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    job = vars(args).copy()
    job.update(output=str(output), pinned=str(pinned), real_data=bool(args.data_root),
               reference_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',REFERENCE],text=True).strip())
    if not args.data_root:
        if args.expected_alpha is not None or args.expected_partition:
            parser.error('Synthetic images are not real Dirichlet/pat shards; omit expected-alpha/partition')
        load_pinned(job, 'fedfews_fixture')
        from fedfews_fixture.acceptance import synthetic_data
        synthetic_data(output/'synthetic_data', args.num_clients, args.dataset)
        job['data_root'] = str(output/'synthetic_data')
    else:
        job['data_root'] = str(Path(args.data_root).resolve())
    job['batch_size'] = args.batch_size if args.batch_size is not None else (50 if args.data_root else 4)
    path = output/'job.json'
    path.write_text(json.dumps(job,indent=2), encoding='utf-8')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    for kind in ['reference', 'source_repeat' if args.reference_repeat else 'native']:
        with (output/f'{kind}_console.log').open('w',encoding='utf-8') as stream:
            result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',kind,str(path)],
                                    cwd=ROOT/'system', env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            report_path = output/'acceptance_report.json'
            if not report_path.exists():
                report_path.write_text(json.dumps(dict(status='ERROR', phase=kind, worker_exit_code=result.returncode,
                     log=str(output/f'{kind}_console.log'), reference_commit=job['reference_commit']),indent=2),encoding='utf-8')
            failure = json.loads(report_path.read_text(encoding='utf-8'))
            if failure['status'] == 'FAIL':
                print(json.dumps({k:failure[k] for k in ['status','event_count','max_parameter_error','max_parameter_location','max_scalar_error']},indent=2))
                print('Full FAIL report:', report_path)
            else:
                print((output/f'{kind}_console.log').read_text(encoding='utf-8')[-6000:])
            raise SystemExit(result.returncode)
    report = json.loads((output/'acceptance_report.json').read_text(encoding='utf-8'))
    print(json.dumps({k:report[k] for k in ['status','event_count','parameter_elements_compared','max_parameter_error','max_parameter_location','max_scalar_error']},indent=2))
    print('Full report:', output/'acceptance_report.json')


if __name__ == '__main__':
    main()
