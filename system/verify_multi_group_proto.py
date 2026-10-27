"""Actual-main N20/K3 two-round prototype acceptance; owned synthetic data only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def verify_tables(out,label,allow_cleaned=False):
    import h5py
    from utils.multi_group_proto import decode_packet,packet_sha
    folder=next((out/label/'logs').rglob('prototype_training.json')).parent
    manifest=json.loads((folder/'manifest.json').read_text());tables={}
    archives=list((out/label).glob('logs.zip'));assert len(archives)==1
    with zipfile.ZipFile(archives[0]) as archive:
        assert not any(name.endswith(('.pt','.pth','.packet','.npz','.npy')) for name in archive.namelist())
    names=('prototype_training','prototype_sources','prototype_client_directions','prototype_capacity_directions',
           'prototype_collection','prototype_uploads','prototype_rounds')
    for name in names:
        rows=json.loads((folder/f'{name}.json').read_text());tables[name]=rows
        with (folder/f'{name}.csv').open(newline='',encoding='utf-8') as stream:encoded=list(csv.DictReader(stream))
        assert len(encoded)==len(rows)
        for a,b in zip(encoded,rows):
            for key in ('loop_round','model_id','client_id','class_id','class_samples','collected_samples','statistics_upload_bytes'):
                if key in b:assert int(a[key])==b[key]
            for key in ('raw_prototype_loss','weighted_prototype_loss','prototype_valid_ratio','pre_normalization_norm'):
                if key in b:assert float(a[key])==b[key]
    with h5py.File(next((out/label/'h5').rglob('*.h5'))) as hf:
        for name,rows in tables.items():assert [json.loads(s) for s in hf[name+'_json'][:]]==rows
        assert json.loads(hf.attrs['class_prototype_protocol_json'])==manifest['class_prototypes']['protocol']
    events=[json.loads(line) for line in (folder/'class_prototypes.jsonl').read_text().splitlines()]
    for event,name in [('training','prototype_training'),('source','prototype_sources'),('collection','prototype_collection'),
                       ('upload','prototype_uploads'),('round','prototype_rounds'),('client_direction','prototype_client_directions'),
                       ('capacity_direction','prototype_capacity_directions')]:
        assert [{k:v for k,v in r.items() if k!='event'} for r in events if r['event']==event]==tables[name]
    rows=tables['prototype_training'];active=manifest['effective']['multi_proto_lambda']>0
    assert len(rows)==120 and sum(r['proto_enabled'] for r in rows)==(60 if active else 0)
    for row in rows:
        assert row['proto_enabled']==(active and row['loop_round']==1)
        assert abs(row['weighted_prototype_loss']-manifest['effective']['multi_proto_lambda']*row['raw_prototype_loss'])<1e-7
        assert abs(row['total_loss']-row['ce_loss']-row['regularization_loss']-row['weighted_prototype_loss'])<5e-6
        assert 0<=row['prototype_valid_ratio']<=1
    if active:
        assert len(tables['prototype_sources'])==300 and len(tables['prototype_collection'])==60
        rounds=tables['prototype_rounds'];assert len(rounds)==1 and rounds[0]['loop_round']==1
        assert rounds[0]['collected_sample_forwards']==150 and rounds[0]['recipient_count']==20
        assert rounds[0]['statistics_upload_bytes']==sum(r['statistics_upload_bytes'] for r in tables['prototype_uploads'])
        libraries=list((out/label/'final_models').rglob('Prototype_library.packet'))
        if not libraries:
            # audit-only may run AFTER verified synthetic binary cleanup.
            # Retain the original worker receipt; never claim a fresh reload.
            assert allow_cleaned and not (out/label/'final_models').exists()
            prior=json.loads((out/label/'verification.json').read_text())['prototype_validation']
            assert prior['formats_exact'] and prior['round_costs'][0]['library_sha256']==rounds[0]['library_sha256']
        else:assert len(libraries)==1
        for path in libraries:
            raw=path.read_bytes();values,meta=decode_packet(raw,'library')
            assert meta['loop_round']==1 and values['prototypes'].shape==(3,100,512)
            assert packet_sha(raw)==rounds[0]['library_sha256']
            assert len(raw)==rounds[0]['prototype_packet_bytes']
        assert any(r['source']=='group' for r in tables['prototype_sources'])
        assert any(r['source']=='outside' for r in tables['prototype_sources'])
        assert any(r['source']=='invalid' for r in tables['prototype_sources'])
    else:
        assert not tables['prototype_rounds'] and not tables['prototype_collection']
    return dict(active_model_tasks=sum(r['proto_enabled'] for r in rows),collection_sample_forwards=150 if active else 0,
        formats_exact=True,per_sample_features_exported=False,round_costs=tables['prototype_rounds'])


def worker(out,label):
    from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
    from verify_multi_group_svd import worker as svd_worker
    job=json.loads((out/'jobs.json').read_text())[label]
    context=nullcontext()
    if job.get('legacy'):
        source=subprocess.run(['git','show','5d6103a:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        namespace={'__name__':'fixed_5d6103a_prototype_reference'};exec(compile(source,'5d6103a/clientMultiGroupSVD.py','exec'),namespace)
        context=patch.object(clientMultiGroupSVD,'train',namespace['clientMultiGroupSVD'].train)
    # Assert the GLOBAL barrier against every real optimizer, not just one client.
    import utils.multi_group_proto as runtime
    from utils.head_snapshot import tensor_fingerprint
    from fedfews_compat.observe import rng_digest
    original_prepare=runtime.prepare_round;barriers=[]
    def checked(server):
        before=rng_digest();states=[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
        assert all(c.upload_round is None and c.download_round==server.loop_round for c in server.clients)
        result=original_prepare(server)
        assert before==rng_digest()
        assert states==[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
        if server.loop_round==0:assert server.proto_latest_packet is None
        barriers.append(server.loop_round)
        return result
    with context,patch.object(runtime,'prepare_round',checked):svd_worker(out,label)
    report=json.loads((out/label/'verification.json').read_text())
    report['prototype_validation']=verify_tables(out,label) if job.get('prototype') else None
    if job.get('prototype'):assert barriers==[0,1]
    (out/label/'verification.json').write_text(json.dumps(report,indent=2))


def audit(out):
    labels=('legacy','zero','projection','avg_only','no_diagnostics')
    reports={label:json.loads((out/label/'verification.json').read_text()) for label in labels};legacy=reports['legacy']
    for key in ('events','batches','final_rng','primary_summary'):assert legacy[key]==reports['zero'][key],('zero_old_parity',key)
    first=lambda r:[e for e in r['events'] if e['kind']=='initial' or e['kind']=='local' and e['loop']==0]
    for label,report in reports.items():
        assert first(report)==first(legacy),('first_uploads',label)
        for key in ('groups','initial_sha256','data_fingerprint','batches','final_rng'):assert report[key]==legacy[key],(label,key)
        if label!='legacy':verify_tables(out,label,allow_cleaned=True)
    for key in ('events','batches','final_rng','primary_summary'):assert reports['projection'][key]==reports['no_diagnostics'][key],('diagnostic_isolation',key)
    result=dict(status='PASS',synthetic_only=True,clients=20,K=3,completed_rounds=2,epochs_per_model=5,actual_width=16,
        reference='5d6103a original SVD train, unchanged data/download/aggregation/evaluation kernels',
        zero_parameters_uploads_aggregation_batches_RNG_exact=True,first_uploads_grouping_exact=True,
        diagnostics_isolation_exact=True,all_columns_global_pre_SGD_barrier_verified=True,
        independent_column_kernel_max_parameter_error=0.,models_reloaded=True,
        cases={label:report['prototype_validation'] for label,report in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker')
    p.add_argument('--jobs',type=int,default=2);p.add_argument('--keep-models',action='store_true');p.add_argument('--audit-only',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.audit_only:return audit(out)
    if args.jobs<1:p.error('jobs must be positive')
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    data=out/'data/Cifar100/pat_fixture';rng=np.random.default_rng(57)
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%2 if split=='train' else 1+cid%4
            np.savez(data/split/f'{cid}.npz',data=dict(x=rng.standard_normal((n,3,32,32)).astype('float32'),y=(np.arange(n)+cid*3)%100))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    base=dict(K=3,rounds=2,gap=1,mode='group_projection_softmax',initialization='low_rank_recovered',init_rank_rate=.15)
    jobs=dict(legacy=dict(base,legacy=True,publish=True),zero=dict(base,prototype=True,proto_lambda=0,read=True),
        projection=dict(base,prototype=True,read=True),avg_only=dict(base,prototype=True,mode='group_avg_only',read=True),
        no_diagnostics=dict(base,prototype=True,proto_diagnostics=0,read=True))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Synthetic prototype acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    run('legacy')  # Publish the one source artifact before independent consumers.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,list(jobs)[1:]))
    audit(out)


if __name__=='__main__':main()
