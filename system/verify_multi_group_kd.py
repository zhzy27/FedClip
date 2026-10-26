"""N20/K3/two-round actual-main synthetic online KD acceptance, CPU only."""
import argparse
import csv
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def verify_exports(out,label):
    import h5py
    folder=next((out/label/'logs').rglob('online_kd_training.json')).parent
    training=json.loads((folder/'online_kd_training.json').read_text())
    probes=json.loads((folder/'online_kd_probes.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    with (folder/'online_kd_training.csv').open(newline='',encoding='utf-8') as stream:
        csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==len(training)
    for encoded,row in zip(csv_rows,training):
        for key in ('client_id','model_id','loop_round','completed_train_count','train_batches','processed_train_samples'):
            assert int(encoded[key])==row[key]
        for key in ('ce_loss','regularization_loss','raw_kl_loss','weighted_kd_loss','total_loss','kd_effective_lambda'):
            assert float(encoded[key])==row[key]
        assert (encoded['kd_enabled']=='True')==row['kd_enabled']
        assert json.loads(encoded['teacher_model_ids'])==row['teacher_model_ids']
        assert json.loads(encoded['teacher_state_sha256'])==row['teacher_state_sha256']
    with (folder/'online_kd_probes.csv').open(newline='',encoding='utf-8') as stream:
        probe_csv=list(csv.DictReader(stream))
    assert len(probe_csv)==len(probes)
    for encoded,row in zip(probe_csv,probes):
        assert json.loads(encoded['pairs'])==row['pairs']
        assert float(encoded['mean_js'])==row['mean_js']
        assert json.loads(encoded['probe_sample_ids'])==row['probe_sample_ids']
    durable=[json.loads(line) for line in (folder/'online_kd.jsonl').read_text().splitlines()]
    for event,expected in [('training',training),('probe',probes)]:
        assert [{k:v for k,v in r.items() if k!='event'} for r in durable if r['event']==event]==expected
    with h5py.File(next((out/label/'h5').rglob('*.h5'))) as hf:
        assert [json.loads(s) for s in hf['online_kd_training_json'][:]]==training
        assert [json.loads(s) for s in hf['online_kd_probes_json'][:]]==probes
        assert json.loads(hf.attrs['online_kd_protocol_json'])==manifest['online_kd']['protocol']
        effective=json.loads(hf.attrs['effective_json'])
        assert effective['multi_kd_mode']==manifest['online_kd']['protocol']['requested_mode']
    assert sum(r['kd_enabled'] for r in training)==manifest['online_kd']['effective_kd_training_tasks']
    assert sum(r['kd_cache_seconds'] for r in training)==manifest['online_kd']['cache_total_seconds']


def worker(out,label):
    import h5py
    import torch
    from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
    from verify_multi_group_svd import worker as svd_worker
    job=json.loads((out/'jobs.json').read_text())[label]
    context=nullcontext()
    if job.get('legacy_train'):
        source=subprocess.run(['git','show','e1e7f61:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,
                              capture_output=True,text=True,check=True).stdout
        namespace={'__name__':'fixed_e1e7f61_reference'}
        exec(compile(source,'e1e7f61/clientMultiGroupSVD.py','exec'),namespace)
        context=patch.object(clientMultiGroupSVD,'train',namespace['clientMultiGroupSVD'].train)
    with context:svd_worker(out,label)
    report=json.loads((out/label/'verification.json').read_text())
    logs=out/label/'logs';manifest_path=next(logs.rglob('manifest.json'));folder=manifest_path.parent
    manifest=json.loads(manifest_path.read_text())
    if job.get('online_kd'):
        rows=json.loads((folder/'online_kd_training.json').read_text())
        probes=json.loads((folder/'online_kd_probes.json').read_text())
        assert len(rows)==20*3*2
        expected=20 if job.get('kd_lambda',.1)>0 else 0
        assert sum(r['kd_enabled'] for r in rows)==expected
        for row in rows:
            assigned=int(report['groups']['client_to_model'][str(row['client_id'])])
            enabled=row['loop_round']>0 and row['model_id']==assigned and job.get('kd_lambda',.1)>0
            assert row['kd_enabled']==enabled
            assert row['train_batches']==5 and row['processed_train_samples'] in (10,15)
            assert abs(row['weighted_kd_loss']-job.get('kd_lambda',.1)*4*row['raw_kl_loss'])<=1e-8
            assert abs(row['total_loss']-row['ce_loss']-row['regularization_loss']-row['weighted_kd_loss'])<=5e-6
            if enabled:
                assert row['teacher_model_ids']==[k for k in range(3) if k!=assigned]
                assert row['teacher_download_loop_round']==row['loop_round']==1
                assert row['teacher_cache_bytes']>0 and len(row['teacher_state_sha256'])==2
                assert row['teacher_sample_forwards']==2*(row['processed_train_samples']//5)
                assert row['kd_cache_fingerprint'] and row['teacher_state_unchanged_during_cache']
            else:assert row['teacher_cache_bytes']==0 and row['raw_kl_loss']==row['weighted_kd_loss']==0
        durable=[json.loads(line) for line in (folder/'online_kd.jsonl').read_text().splitlines()]
        assert len([r for r in durable if r['event']=='training'])==len(rows)
        assert len([r for r in durable if r['event']=='probe'])==len(probes)
        assert len(probes)==(40 if job.get('probe_samples',0)>0 and expected else 0)
        for cid in range(20):
            records=[r for r in probes if r['client_id']==cid]
            if records:
                assert records[0]['probe_ids_sha256']==records[1]['probe_ids_sha256']
                assert all(r['loop_round']==1 and r['probe_count']==2 and len(r['pairs'])==3 for r in records)
        with next((out/label/'h5').rglob('*.h5')).open('rb') as stream,h5py.File(stream) as hf:
            assert len(hf['online_kd_training_json'])==len(rows)
            assert len(hf['online_kd_probes_json'])==len(probes)
            assert json.loads(hf.attrs['online_kd_protocol_json'])==manifest['online_kd']['protocol']
        assert manifest['online_kd']['effective_kd_training_tasks']==expected
        final_manifest=next((out/label/'final_models').rglob('manifest.json'))
        assert json.loads(final_manifest.read_text())['online_kd']==manifest['online_kd']
        for file in (out/label/'final_models').rglob('Client_*model_*.pt'):
            payload=torch.load(file,map_location='cpu',weights_only=False)
            assert payload['online_kd']==manifest['online_kd']['protocol']
        report['online_kd_validation']=dict(enabled_tasks=expected,records=len(rows),probes=len(probes),
            durable_jsonl_verified=True,H5_CSV_JSON_manifest_verified=True,
            teacher_forward_seconds=manifest['online_kd']['teacher_forward_seconds'],
            cache_seconds=manifest['online_kd']['cache_total_seconds'],
            probe_seconds=manifest['online_kd']['probe_total_seconds'])
    else:assert 'online_kd' not in manifest
    if job.get('online_kd'):verify_exports(out,label)
    (out/label/'verification.json').write_text(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--worker')
    parser.add_argument('--keep-models',action='store_true')
    parser.add_argument('--audit-only',action='store_true',help='Read/recheck completed synthetic reports; no training or cleanup')
    a=parser.parse_args();out=Path(a.output).resolve()
    if a.worker:return worker(out,a.worker)
    if a.audit_only:return audit(out)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    data=out/'data/Cifar100/pat_fixture';generator=np.random.default_rng(57)
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%2 if split=='train' else 1+cid%4
            np.savez(data/split/f'{cid}.npz',data=dict(x=generator.standard_normal((n,3,32,32)).astype('float32'),y=(np.arange(n)+cid*3)%100))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    base=dict(K=3,rounds=2,gap=1,mode='group_projection_softmax',initialization='low_rank_recovered',init_rank_rate=.15)
    jobs=dict(legacy=dict(base,legacy_train=True,publish=True),
        zero_full=dict(base,online_kd='full',kd_lambda=0,read=True),
        full=dict(base,online_kd='full',read=True,probe_samples=2),
        not_true=dict(base,online_kd='not_true',read=True,probe_samples=2),
        full_no_probe=dict(base,online_kd='full',read=True))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2));reports={}
    for label in jobs:
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Acceptance failed: {out}/{label}.log')
        reports[label]=json.loads((out/label/'verification.json').read_text())
        print('PASS',label,flush=True)
        if not a.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
    return audit(out,model_files_retained=a.keep_models)


def audit(out,model_files_retained=None):
    if model_files_retained is None:
        model_files_retained=any((out/label/'checkpoints').exists() for label in ('legacy','zero_full','full','not_true','full_no_probe'))
    reports={label:json.loads((out/label/'verification.json').read_text()) for label in
             ('legacy','zero_full','full','not_true','full_no_probe')}
    legacy=reports['legacy']
    for label in ('zero_full','full','not_true','full_no_probe'):verify_exports(out,label)
    for key in ('events','batches','final_rng','primary_summary'):
        assert legacy[key]==reports['zero_full'][key],('zero_KD_legacy_parity',key)
    first=lambda report:[e for e in report['events'] if e['kind']=='initial' or e.get('loop')==0]
    for label,report in reports.items():
        assert first(report)==first(legacy),('first_round_parity',label)
        for key in ('groups','data_fingerprint','initial_sha256'):assert report[key]==legacy[key],(label,key)
        for key in ('batches','final_rng'):assert report[key]==legacy[key],('same_training_random_stream',label,key)
    for key in ('events','batches','final_rng','primary_summary'):
        assert reports['full'][key]==reports['full_no_probe'][key],('probe_isolation',key)
    def cache_sources(label):
        path=next((out/label/'logs').rglob('online_kd_training.json'))
        rows=json.loads(path.read_text())
        keys=('client_id','model_id','teacher_model_ids','teacher_state_sha256','kd_cache_fingerprint')
        return sorted([{k:r[k] for k in keys} for r in rows if r['kd_enabled']],key=lambda r:r['client_id'])
    assert cache_sources('full')==cache_sources('not_true')==cache_sources('full_no_probe')
    result=dict(status='PASS',synthetic_only=True,clients=20,K=3,completed_rounds=2,epochs_per_model=5,actual_width=16,
        reference='e1e7f61 original SVD client.train + unchanged download/aggregation/evaluation kernels',
        zero_KD_old_parameters_batches_RNG_exact=True,first_round_all_modes_exact=True,shared_grouping_exact=True,
        same_all_modes_batch_order_and_RNG=True,second_round_teacher_cache_sources_exact=True,
        probe_isolation_exact=True,column_aggregation_parameter_error=0.,models_reloaded=True,
        cases={label:report.get('online_kd_validation') for label,report in reports.items()},model_files_retained=model_files_retained)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
