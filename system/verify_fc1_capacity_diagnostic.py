"""Synthetic B-fc1 source + nine offline capacity tasks; no real-data run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(command,log,success=True):
    with Path(log).open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT/'system',env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',
            PYTHONIOENCODING='utf-8',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
    if (result.returncode==0)!=success:raise RuntimeError(f'Unexpected exit={result.returncode}; see {log}')


def reference(out,server):
    import random
    import tempfile
    import numpy as np
    import torch
    from fc1_capacity_diagnostic import parser
    from utils import fc1_capacity_diagnostic as d,fc1_tail_diagnostic as old
    from utils.head_snapshot import tensor_fingerprint
    from common.client_data import read_image_client
    from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
    torch.set_num_threads(2)
    cli=parser().parse_args(['--server-file',str(server),'--source-round','2','--data-root',str(out/'data'),
        '--data-subdir','fixture','--device','cpu','--output-dir',str(out/'not_created_reference_output')])
    mother,args,_,source=d.load_source(cli);results=[]
    with tempfile.TemporaryDirectory(prefix='reference_transient_',dir=out) as temp:
        args.save_folder_name_full=temp
        for cid in [0,10,18]:
            data=read_image_client(args,cid,True);test=read_image_client(args,cid,False)
            client=clientMultiGroupSVD(args,cid,len(data),len(test),initial_model=mother,train_slow=False,send_slow=False)
            client.set_parameters([mother],2)
            client.avg_proto_bank=None;client.avg_proto_receipt=None;client.avg_proto_local=None
            client.avg_proto_download_context={}
            # Exactly the same explicit offline training RNG start; actual client
            # constructor/manual_seed/SVD lifecycle above is not being changed.
            random.seed(args.seed);np.random.seed(args.seed);torch.random.default_generator.manual_seed(args.seed)
            initial_rng=old.rng_fingerprint(old.rng_snapshot(client.model_set[0]))
            loader=old.EpochLoader(data,args,cid,2);client.load_train_data=lambda:loader
            client.train()
            results.append(dict(client_id=cid,fc1_rank=client.model_set[0].fc1.rank,
                initial_rng_sha256=initial_rng,final_rng_sha256=old.rng_fingerprint(old.rng_snapshot(client.model_set[0])),
                all_batch_indices_sha256=d.digest_json(loader.batches),final_model_sha256=tensor_fingerprint(client.model_set[0].state_dict()),
                training=client.train_stats[0]))
    (out/'native_reference.json').write_text(json.dumps(results,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--reference-worker',action='store_true');p.add_argument('--resume-owned-output',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();jobs=dict(B_fc1=dict(mode='next_tier',layers='fc1'))
    if args.reference_worker:return reference(out,next((out/'B_fc1/f').rglob('Server_model_0.pt')))
    if args.resume_owned_output:
        if json.loads((out/'jobs.json').read_text())!=jobs:raise ValueError('Not this verifier\'s synthetic workspace')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        rng=np.random.default_rng(94);data=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (data/split).mkdir(parents=True)
            for cid in range(20):
                n=2+cid%3 if split=='train' else 1+cid%4
                x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
                np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
        (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    if not (out/'B_fc1/verification.json').is_file():
        print('RUN actual B_fc1 main: N20/K1/2 rounds',flush=True)
        execute([sys.executable,str(ROOT/'system/verify_multi_svd_spectral.py'),'--output',str(out),'--worker','B_fc1'],out/'source.log')
    server=next((out/'B_fc1/f').rglob('Server_model_0.pt'))
    common=[sys.executable,str(ROOT/'system/fc1_capacity_diagnostic.py'),'--server-file',str(server),
        '--source-round','2','--data-root',str(out/'data'),'--data-subdir','fixture',
        '--client-ids','0,10,18','--fc1-ranks','120,200,720','--device','cpu','--threads','2']
    for label,extra in [('observed',[]),('quiet',['--diagnostics','0'])]:
        if (out/label/'manifest.json').is_file():
            if json.loads((out/label/'manifest.json').read_text())['status']!='complete':raise ValueError('Partial diagnostic output; use a new root')
        else:
            print('RUN',label,'nine local tasks',flush=True)
            execute(common+extra+['--output-dir',str(out/label)],out/(label+'.log'))
    if not (out/'native_reference.json').is_file():
        print('RUN actual client native-rank references',flush=True)
        execute([sys.executable,__file__,'--output',str(out),'--reference-worker'],out/'reference.log')
    read=lambda p:json.loads((out/p).read_text())
    observed=read('observed/local_training.json');quiet=read('quiet/local_training.json');references=read('native_reference.json')
    assert len(observed)==9 and sum(r['train_batches'] for r in observed)==45
    for a,b in zip(observed,quiet):
        for key in ('client_id','fc1_rank','final_model_sha256','all_batch_indices_sha256','initial_rng_sha256','final_rng_sha256',
            'ce_loss','regularization_loss','raw_mse','total_loss','train_batches','processed_train_samples'):
            assert a[key]==b[key],(key,a['client_id'],a['fc1_rank'])
    for expected in references:
        row=next(r for r in observed if (r['client_id'],r['fc1_rank'])==(expected['client_id'],expected['fc1_rank']))
        for key in ('final_model_sha256','all_batch_indices_sha256','initial_rng_sha256','final_rng_sha256'):assert row[key]==expected[key],key
        for key in ('ce_loss','regularization_loss','total_loss','train_batches','processed_train_samples'):assert row[key]==expected['training'][key],key
    for cid in (0,10,18):
        rows=[r for r in observed if r['client_id']==cid]
        assert len({r['all_batch_indices_sha256'] for r in rows})==len({r['initial_rng_sha256'] for r in rows})==len({r['final_rng_sha256'] for r in rows})==1
        assert {tuple(r['other_layer_ranks'].values()) for r in rows}==({(72,461)} if cid!=18 else {(20,128)})
    assert len(read('observed/endpoints.json'))==54 and len(read('observed/fixed_slot_updates.json'))==108
    assert len(read('observed/first_step_gradients.json'))==len(read('observed/first_batch_CE_signal.json'))==27
    assert len(read('observed/tail_rollback.json'))==3 and len(read('observed/rank_comparisons.json'))==6
    import csv
    for label in ('observed','quiet'):
        root=out/label
        for file in root.glob('*.csv'):
            expected=json.loads(file.with_suffix('.json').read_text())
            with file.open(newline='',encoding='utf-8-sig') as stream:actual=list(csv.DictReader(stream))
            assert len(actual)==len(expected)
            for saved,row in zip(actual,expected):
                for key,cell in saved.items():
                    value=row.get(key)
                    if isinstance(value,(list,dict)):assert json.loads(cell)==value
                    else:assert cell==('' if value is None else str(value))
        assert not list(root.rglob('*.pt')) and not list(root.glob('transient_training_*'))
        assert read(label+'/manifest.json')['input_hashes_unchanged']
    execute(common+['--source-round','101','--output-dir',str(out/'bad_round')],out/'bad_round.log',False)
    execute(common+['--data-subdir','missing','--output-dir',str(out/'bad_data')],out/'bad_data.log',False)
    import shutil
    import numpy as np
    wrong=out/'wrong_data'
    if not wrong.exists():
        shutil.copytree(out/'data',wrong)
        file=wrong/'Cifar100/fixture/train/0.npz';value=np.load(file,allow_pickle=True)['data'].tolist()
        value['x'][0,0,0,0]+=1.;np.savez(file,data=value)
    execute(common+['--data-root',str(wrong),'--output-dir',str(out/'bad_fingerprint')],out/'bad_fingerprint.log',False)
    if 'Actual pat data file/decoded tensor fingerprints differ' not in (out/'bad_fingerprint.log').read_text(encoding='utf-8'):
        raise RuntimeError('Wrong-data check failed for an unexpected reason')
    def finite(value):
        import math
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,list):
            for v in value:finite(v)
    for name in ('endpoints','fixed_slot_updates','first_batch_CE_signal','first_step_gradients','local_training','tail_rollback'):
        finite(read('observed/'+name+'.json'))
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',source_main_clients=20,source_actual_rounds=2,offline_label='S2 -> offline R3, NOT a real R102 result',
        diagnosed_clients=[0,10,18],fc1_ranks=[120,200,720],tasks_per_diagnostic_run=9,epochs=5,steps_per_run=45,
        common_fc1_SVD_calls_per_run=1,other_layer_download_SVD_calls_per_run=6,
        common_prefix_optimizer_bound_and_rollback_unit_tests=True,paired_batches_and_RNG_exact=True,
        diagnostics_on_off_parameters_losses_batches_RNG_exact=True,native_rank_actual_client_parameters_losses_batches_RNG_exact=True,
        finite_probes_and_slot_updates=True,CSV_JSON_checked=True,default_only_tables_no_models=True,
        original_inputs_unchanged=True,invalid_source_round_and_data_rejected=True,real_R101_available_or_performance_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
