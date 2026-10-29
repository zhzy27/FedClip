"""43eb806 parity plus actual-main N20/K3 natural-mean two-round acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import torch
    import utils.multi_group_proto as runtime
    from utils.multi_group_proto_options import representations
    from utils.head_snapshot import tensor_fingerprint
    from fedfews_compat.observe import rng_digest
    from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
    from verify_multi_group_svd import worker as svd_worker
    from verify_multi_group_proto_losses import tables
    job=json.loads((out/'jobs.json').read_text())[label];raw=job.get('proto_target_type')=='raw_mean'
    errors=[]
    with ExitStack() as stack:
        if job.get('legacy'):
            for filename in ('system/utils/multi_group_proto.py','system/flcore/clients/clientMultiGroupSVD.py'):
                source=subprocess.run(['git','show',f'43eb806:{filename}'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
                namespace={'__name__':'fixed_43eb806_reference'};exec(compile(source,'43eb806/'+filename,'exec'),namespace)
                if 'utils/' in filename:
                    for name in ('prepare_round','Session','persist_training','manifest_value','save_latest'):stack.enter_context(patch.object(runtime,name,namespace[name]))
                else:stack.enter_context(patch.object(clientMultiGroupSVD,'train',namespace['clientMultiGroupSVD'].train))
        original=runtime.prepare_round
        def checked(server):
            before=rng_digest();states=[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
            assert all(c.upload_round is None for c in server.clients)
            original(server)
            if raw and server.loop_round==1 and server.args.multi_proto_lambda>0:
                values,meta=runtime.decode_packet(server.proto_latest_packet,'library')
                runtime.validate_representation(meta,'raw_mean','independent check')
                sums={};counts={}
                with runtime.protected_rng(),torch.no_grad():
                    for client in server.clients:
                        data=client.load_train_data().dataset
                        counts[client.id]=torch.bincount(torch.stack([y for x,y in data]),minlength=server.num_classes)
                        sums[client.id]=[]
                        for model in client.model_set:
                            local=copy.deepcopy(model).eval();total=torch.zeros(server.num_classes,local.fc3.in_features,dtype=torch.float64)
                            for begin in range(0,len(data),server.args.multi_proto_batch_size):
                                batch=data[begin:begin+server.args.multi_proto_batch_size]
                                # Independent base_output path: no feature hook or merge kernel.
                                features=local.base(torch.stack([x for x,y in batch]).to(client.device)).double().cpu()
                                for feature,(_,y) in zip(features,batch):total[int(y)].add_(feature)
                            sums[client.id].append(total);del local
                    expected=torch.zeros_like(values['prototypes']);valid=torch.zeros_like(values['valid'])
                    for column,members in enumerate(server.groups):
                        for c in range(server.num_classes):
                            inside=[i for i in members if counts[i][c]>0]
                            selected=inside or [i for i in range(server.num_clients) if i not in members and counts[i][c]>0]
                            n=sum(int(counts[i][c]) for i in selected)
                            if n:
                                expected[column,c]=sum((sums[i][column][c] for i in selected),torch.zeros(expected.shape[-1],dtype=torch.float64))/n
                                valid[column,c]=True
                    error=float((expected-values['prototypes']).abs().max());errors.append(error)
                    assert error==0. and torch.equal(valid,values['valid'])
            assert before==rng_digest();assert states==[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
            if server.loop_round==0:assert server.proto_latest_packet is None
        stack.enter_context(patch.object(runtime,'prepare_round',checked))
        if job.get('proto_lambda')==0:
            for name in ('collect_client','merge_statistics','encode_packet'):stack.enter_context(patch.object(runtime,name,side_effect=AssertionError('Zero must skip prototypes')))
        svd_worker(out,label)
    receipt=tables(out,label)
    if raw and job.get('proto_lambda',1)>0:
        folder=next((out/label/'logs').rglob('prototype_training.json')).parent
        training=json.loads((folder/'prototype_training.json').read_text());sources=json.loads((folder/'prototype_sources.json').read_text())
        for row in training:
            assert row['prototype_target_type']=='raw_mean'
            assert row['aux_target_norm'] is None and row['aux_effective_target_norm'] is None and row['actual_fixed_target_norm'] is None
            assert row['statistics_representation']=='raw_feature_class_sum' and row['prototype_representation']=='raw_class_mean'
            if row['proto_enabled'] and job.get('proto_diagnostics',1):
                assert row['training_target_observed_valid_sample_visits']==row['processed_train_samples']
                assert row['training_actual_target_l2_mean'] is not None
        assert any(row['prototype_l2_norm']>1 for row in sources if row['valid'])
        assert all(row['pre_normalization_norm'] is None for row in sources)
        assert all(row['statistics_representation']=='raw_feature_class_sum' for row in sources)
        assert errors==[0.]
    path=out/label/'verification.json';report=json.loads(path.read_text());report.update(auxiliary_validation=receipt,independent_raw_mean_errors=errors)
    path.write_text(json.dumps(report,indent=2))


def audit(out):
    jobs=json.loads((out/'jobs.json').read_text());reports={label:json.loads((out/label/'verification.json').read_text()) for label in jobs}
    for a,b in [('old_control','zero_raw'),('old_fixed','fixed_default'),('raw_mean','raw_no_diagnostics')]:
        for key in ('events','batches','final_rng','primary_summary'):assert reports[a][key]==reports[b][key],(a,b,key)
    for key in ('library_sha256','statistics_upload_bytes','prototype_packet_bytes','prototype_download_bytes'):
        assert reports['old_fixed']['auxiliary_validation']['rounds'][0][key]==reports['fixed_default']['auxiliary_validation']['rounds'][0][key]
    assert reports['raw_mean']['auxiliary_validation']['rounds'][0]['library_sha256']==reports['raw_no_diagnostics']['auxiliary_validation']['rounds'][0]['library_sha256']
    base=reports['old_control']
    for label,report in reports.items():
        for key in ('groups','initial_sha256','data_fingerprint','batches','final_rng'):assert report[key]==base[key],(label,key)
        first=lambda r:[e for e in r['events'] if e['kind']=='initial' or e['kind']=='local' and e['loop']==0]
        assert first(report)==first(base),(label,'first round')
        assert report['all_prediction_checkpoints_reloaded'] and report['max_column_parameter_error']==0.
    result=dict(status='PASS',synthetic_only=True,reference='fixed git 43eb806 prototype runtime and client train',
        clients=20,K=3,completed_rounds=2,epochs_per_model=5,actual_width=16,
        fixed_norm_parameters_uploads_servers_batches_RNG_and_packet_exact=True,zero_vs_old_control_exact=True,
        first_round_uploads_and_grouping_exact=True,raw_diagnostics_isolation_exact=True,all_batches_and_final_RNG_exact=True,
        independent_raw_mean_max_error=0.,independent_column_kernel_max_error=0.,all_models_reloaded=True,
        cases={label:report['auxiliary_validation'] for label,report in reports.items()})
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
    aux=dict(base,prototype=True,read=True,proto_loss='mse',proto_feature='base_output',proto_lambda=1)
    jobs=dict(old_control=dict(base,legacy=True,publish=True),zero_raw=dict(aux,proto_target_type='raw_mean',proto_lambda=0),
        old_fixed=dict(aux,legacy=True),fixed_default=dict(aux),raw_mean=dict(aux,proto_target_type='raw_mean'),
        raw_no_diagnostics=dict(aux,proto_target_type='raw_mean',proto_diagnostics=0))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    def run(label):
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Synthetic raw-mean acceptance failed: {out}/{label}.log')
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
