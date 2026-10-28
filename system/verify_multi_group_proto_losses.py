"""N20/K3 actual-main, original-width SVD synthetic acceptance; no real data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
TABLES=('prototype_training','prototype_sources','prototype_client_directions','prototype_capacity_directions',
        'prototype_collection','prototype_uploads','prototype_rounds')


def tables(out,label):
    import h5py
    from utils.multi_group_proto import decode_packet,packet_sha
    job=json.loads((out/'jobs.json').read_text())[label]
    if not job.get('prototype'):return None
    with zipfile.ZipFile(out/label/'logs.zip') as archive:
        assert not any(name.endswith(('.pt','.pth','.packet','.npz','.npy')) for name in archive.namelist())
    folder=next((out/label/'logs').rglob('prototype_training.json')).parent
    manifest=json.loads((folder/'manifest.json').read_text());result={}
    for name in TABLES:
        rows=json.loads((folder/f'{name}.json').read_text());result[name]=rows
        with (folder/f'{name}.csv').open(encoding='utf-8',newline='') as stream:encoded=list(csv.DictReader(stream))
        assert len(rows)==len(encoded)
        for original,record in zip(rows,encoded):
            for key,value in original.items():
                if isinstance(value,float):assert float(record[key])==value,(name,key)
    with h5py.File(next((out/label/'h5').rglob('*.h5'))) as hf:
        for name,rows in result.items():assert [json.loads(s) for s in hf[name+'_json'][:]]==rows
        assert json.loads(hf.attrs['class_prototype_protocol_json'])==manifest['class_prototypes']['protocol']
    events=[json.loads(line) for line in (folder/'class_prototypes.jsonl').read_text().splitlines()]
    for event,name in zip(('training','source','client_direction','capacity_direction','collection','upload','round'),TABLES):
        assert [{k:v for k,v in r.items() if k!='event'} for r in events if r['event']==event]==result[name]
    active=job.get('proto_lambda',.1)>0;kind=job.get('proto_loss','cosine');feature=job.get('proto_feature','classifier_input')
    rows=result['prototype_training'];assert len(rows)==120
    for row in rows:
        enabled=active and row['loop_round']==1
        assert row['proto_enabled']==enabled
        if job.get('legacy'):continue  # Fixed d213477 fields retain their original meaning.
        assert row['aux_requested_loss']==kind and row['aux_feature_position']==feature
        assert row['aux_effective_loss']==(kind if enabled else 'none')
        assert abs(row['weighted_auxiliary_loss']-job.get('proto_lambda',.1)*row['raw_auxiliary_loss'])<1e-7
        assert abs(row['total_loss']-row['ce_loss']-row['regularization_loss']-row['weighted_auxiliary_loss'])<5e-6
        if enabled:
            assert row['auxiliary_valid_samples']==row['processed_train_samples']
            if job.get('proto_diagnostics',1):
                assert row['training_feature_observed_sample_visits']==row['processed_train_samples']
                assert row['preclip_gradient_observed_steps']==row['train_batches']
                assert 0<=row['gradient_clip_trigger_ratio']<=1
                if kind=='z2':
                    # Observation reduces in float64; the real SGD objective
                    # reduces in float32. Bound rounding relative to its scale,
                    # never relax parameter/upload/RNG equality (still exact).
                    error=abs(row['training_mean_feature_squared_per_dimension']-row['raw_auxiliary_loss'])
                    assert error<=8*(2**-23)*max(1.,abs(row['raw_auxiliary_loss']))
            else:assert row['training_mean_feature_l2'] is None and row['preclip_gradient_norm_mean'] is None
        if kind=='z2':assert row['prototype_library_sha256'] is None and row['raw_prototype_loss'] is None
    libraries=list((out/label/'final_models').rglob('Prototype_library.packet'))
    if not active or kind=='z2':
        assert not libraries
        assert all(not result[name] for name in TABLES[1:-1])
        assert manifest['class_prototypes']['statistics_upload_bytes']==0
        assert manifest['class_prototypes']['prototype_download_bytes']==0
        assert manifest['class_prototypes']['collection_sample_forwards']==0
        assert manifest['class_prototypes']['latest_library_sha256'] is None
        if active:
            assert len(result['prototype_rounds'])==1 and result['prototype_rounds'][0]['prototype_status']=='not_applicable'
        else:assert not result['prototype_rounds']
    else:
        assert len(libraries)==1 and len(result['prototype_collection'])==60
        raw=libraries[0].read_bytes();values,metadata=decode_packet(raw,'library')
        assert values['prototypes'].shape==(3,100,512) and metadata['loop_round']==1
        assert metadata.get('feature_position','classifier_input')==feature
        round_record=result['prototype_rounds'][0]
        assert packet_sha(raw)==round_record['library_sha256'] and len(raw)==round_record['prototype_packet_bytes']
        assert round_record['collected_sample_forwards']==150 and round_record['forward_batches']==90
    return dict(status='PASS',formats_exact=True,active_model_tasks=sum(r['proto_enabled'] for r in rows),
        rounds=result['prototype_rounds'],model_reload_receipt='verified by actual-main SVD worker before cleanup')


def worker(out,label):
    import utils.multi_group_proto as runtime
    from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
    from verify_multi_group_svd import worker as svd_worker
    from utils.head_snapshot import tensor_fingerprint
    from fedfews_compat.observe import rng_digest
    job=json.loads((out/'jobs.json').read_text())[label]
    with ExitStack() as stack:
        if job.get('legacy'):
            for filename in ('system/utils/multi_group_proto.py','system/flcore/clients/clientMultiGroupSVD.py'):
                source=subprocess.run(['git','show',f'd213477:{filename}'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
                namespace={'__name__':'fixed_d213477_reference'};exec(compile(source,'d213477/'+filename,'exec'),namespace)
                if 'utils/' in filename:
                    for name in ('prepare_round','Session','persist_training','manifest_value','save_latest'):stack.enter_context(patch.object(runtime,name,namespace[name]))
                else:stack.enter_context(patch.object(clientMultiGroupSVD,'train',namespace['clientMultiGroupSVD'].train))
        original=runtime.prepare_round
        def checked(server):
            before=rng_digest();states=[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
            assert all(c.upload_round is None for c in server.clients)
            original(server)
            assert before==rng_digest();assert states==[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
            if server.loop_round==0:assert server.proto_latest_packet is None
        stack.enter_context(patch.object(runtime,'prepare_round',checked))
        if job.get('proto_loss')=='z2' or job.get('proto_lambda')==0:
            for name in ('collect_client','merge_statistics','encode_packet','decode_packet'):
                stack.enter_context(patch.object(runtime,name,side_effect=AssertionError('Disabled/Z2 path attempted prototype work')))
        svd_worker(out,label)
    receipt=tables(out,label)
    path=out/label/'verification.json';report=json.loads(path.read_text());report['auxiliary_validation']=receipt
    path.write_text(json.dumps(report,indent=2))


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text());reports={label:json.loads((out/label/'verification.json').read_text()) for label in jobs}
    def same(a,b):
        for key in ('events','batches','final_rng','primary_summary'):assert reports[a][key]==reports[b][key],(a,b,key)
    same('old_control','zero_mse');same('old_cosine','default_cosine');same('z2','z2_no_diagnostics')
    old_round=reports['old_cosine']['auxiliary_validation']['rounds'][0]
    new_round=reports['default_cosine']['auxiliary_validation']['rounds'][0]
    for key in ('library_sha256','statistics_upload_bytes','prototype_packet_bytes','prototype_download_bytes'):
        assert old_round[key]==new_round[key],('default_packet_parity',key)
    assert (reports['cosine']['auxiliary_validation']['rounds'][0]['library_sha256']==
            reports['mse']['auxiliary_validation']['rounds'][0]['library_sha256']), 'same-position Cosine/MSE libraries differ'
    base=reports['old_control']
    for label,report in reports.items():
        for key in ('groups','initial_sha256','data_fingerprint','batches','final_rng'):assert report[key]==base[key],(label,key)
        first=lambda r:[e for e in r['events'] if e['kind']=='initial' or e['kind']=='local' and e['loop']==0]
        assert first(report)==first(base),('first_round',label)
    result=dict(status='PASS',synthetic_only=True,clients=20,K=3,completed_rounds=2,epochs_per_model=5,actual_width=16,
        reference='fixed d213477 train and prototype runtime, via git show; same actual-main data/download/aggregation path',
        old_default_cosine_exact=True,old_default_prototype_packet_exact=True,same_position_cosine_mse_prototypes_exact=True,
        zero_vs_old_control_exact=True,all_first_rounds_and_groups_exact=True,
        batches_and_final_RNG_exact=True,z2_no_prototype_operations=True,diagnostics_isolation_exact=True,
        independent_column_kernel_max_parameter_error=0.,all_prediction_models_reloaded=True,
        cases={label:report['auxiliary_validation'] for label,report in reports.items()})
    z2_rows=json.loads(next((out/'z2/logs').rglob('prototype_training.json')).read_text())
    errors=[(abs(r['training_mean_feature_squared_per_dimension']-r['raw_auxiliary_loss']),abs(r['raw_auxiliary_loss']))
            for r in z2_rows if r['proto_enabled']]
    result['diagnostic_precision']=dict(observation_dtype='float64',objective_dtype='float32',
        max_absolute_difference=max(e for e,scale in errors),max_scaled_difference=max(e/max(1.,scale) for e,scale in errors),
        rounding_bound='8 * float32 epsilon * max(1, abs(raw_loss)); applies ONLY to cross-dtype observations',
        parameter_upload_server_batch_RNG_tolerance='exact, unchanged')
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
    aux=dict(base,prototype=True,read=True)
    jobs=dict(old_control=dict(base,legacy=True,publish=True),zero_mse=dict(aux,proto_loss='mse',proto_feature='base_output',proto_lambda=0),
        old_cosine=dict(aux,legacy=True),default_cosine=dict(aux),cosine=dict(aux,proto_loss='cosine',proto_feature='base_output',proto_lambda=1),
        mse=dict(aux,proto_loss='mse',proto_feature='base_output',proto_lambda=1),z2=dict(aux,proto_loss='z2',proto_feature='base_output',proto_lambda=1),
        z2_no_diagnostics=dict(aux,proto_loss='z2',proto_feature='base_output',proto_lambda=1,proto_diagnostics=0))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Synthetic auxiliary acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup target')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    run('old_control')
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:list(pool.map(run,list(jobs)[1:]))
    audit(out)


if __name__=='__main__':main()
