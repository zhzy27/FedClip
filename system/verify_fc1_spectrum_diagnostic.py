"""Actual CLI spectrum smoke + frozen 6697331 rank regression on synthetic data."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(command,path,success=True):
    with Path(path).open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT/'system',env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',
            PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
    if (result.returncode==0)!=success:raise RuntimeError(f'Unexpected exit={result.returncode}; {path}')


def prepare_source(root):
    root.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(94);data=root/'data/Cifar100/fixture'
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3 if split=='train' else 1+cid%4
            x=(rng.normal(size=(n,3,32,32))*.2).astype('float32');x[:,0,0,0]=np.arange(n)/10
            np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)//2+cid)%8))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    (root/'jobs.json').write_text(json.dumps(dict(B_fc1=dict(mode='next_tier',layers='fc1'))))
    execute([sys.executable,str(ROOT/'system/verify_multi_svd_spectral.py'),'--output',str(root),'--worker','B_fc1'],root/'source_main.log')


def worker(out,label):
    settings=json.loads((out/'settings.json').read_text());source=Path(settings['source_workspace'])
    server=next((source/'B_fc1/f').rglob('Server_model_0.pt'))
    args=['--server-file',str(server),'--source-round','2','--data-root',str(source/'data'),'--data-subdir','fixture',
        '--client-ids','0,10,18','--device','cpu','--threads','2','--output-dir',str(out/label)]
    if label in ('observed','quiet'):args+=['--comparison','spectrum','--fc1-ranks','200']
    else:args+=['--fc1-ranks','120,200,720']  # New rank case intentionally uses the DEFAULT comparison.
    if label=='quiet':args+=['--diagnostics','0']
    if label=='rank_frozen':
        from fc1_capacity_diagnostic import parser
        code=subprocess.check_output(['git','show','6697331:system/utils/fc1_capacity_diagnostic.py'],cwd=ROOT,text=True,encoding='utf-8')
        namespace=dict(__name__='frozen_6697331_capacity',__file__=str(ROOT/'system/utils/fc1_capacity_diagnostic.py'))
        exec(compile(code,'6697331/fc1_capacity_diagnostic.py','exec'),namespace)
        return namespace['run'](parser().parse_args(args))
    import runpy
    sys.argv=['fc1_capacity_diagnostic.py']+args
    runpy.run_path(str(ROOT/'system/fc1_capacity_diagnostic.py'),run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--source-workspace',help='Reuse a completed owned synthetic B_fc1 source from the prior verifier')
    parser.add_argument('--worker',choices=['observed','quiet','rank_new','rank_frozen'])
    parser.add_argument('--resume-owned-output',action='store_true');args=parser.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    if args.resume_owned_output:
        settings=json.loads((out/'settings.json').read_text())
        if settings.get('verifier')!='fc1-spectrum-v1':raise ValueError('Not this verifier workspace')
        source=Path(settings['source_workspace'])
    else:
        out.mkdir(parents=True,exist_ok=False)
        source=Path(args.source_workspace).resolve() if args.source_workspace else out/'source'
        if not args.source_workspace:prepare_source(source)
        if json.loads((source/'jobs.json').read_text())!=dict(B_fc1=dict(mode='next_tier',layers='fc1')):
            raise ValueError('Need an owned completed synthetic B_fc1 source')
        if json.loads((source/'B_fc1/verification.json').read_text())['status']!='PASS':raise ValueError('Source acceptance incomplete')
        settings=dict(verifier='fc1-spectrum-v1',source_workspace=str(source));(out/'settings.json').write_text(json.dumps(settings,indent=2))
    for label in ('observed','quiet','rank_new','rank_frozen'):
        manifest=out/label/'manifest.json'
        if manifest.is_file():
            if json.loads(manifest.read_text())['status']!='complete':raise ValueError('Partial output; use a new root')
        else:
            print('RUN',label,flush=True)
            execute([sys.executable,__file__,'--output',str(out),'--worker',label],out/(label+'.log'))
    read=lambda p:json.loads((out/p).read_text())
    observed=read('observed/local_training.json');quiet=read('quiet/local_training.json')
    assert len(observed)==len(quiet)==6 and sum(r['train_batches'] for r in observed)==30
    for cid in (0,10,18):
        rows=[r for r in observed if r['client_id']==cid]
        assert {r['spectrum_variant'] for r in rows}=={'raw','corrected'} and {r['fc1_rank'] for r in rows}=={200}
        for key in ('model_parameter_count','all_batch_indices_sha256','initial_rng_sha256','final_rng_sha256'):
            assert len({r[key] for r in rows})==1,(cid,key)
        assert {tuple(r['other_layer_ranks'].values()) for r in rows}==({(72,461)} if cid!=18 else {(20,128)})
    for a,b in zip(observed,quiet):
        for key in ('client_id','spectrum_variant','fc1_rank','final_model_sha256','all_batch_indices_sha256',
            'initial_rng_sha256','final_rng_sha256','ce_loss','regularization_loss','raw_mse','total_loss','train_batches','processed_train_samples'):
            assert a[key]==b[key],key
    assert read('observed/endpoints.json')==read('quiet/endpoints.json')
    assert len(read('observed/endpoints.json'))==36 and len(read('observed/fixed_slot_updates.json'))==48
    for table in ('endpoints','fixed_slot_updates','first_step_gradients','local_training','batch_order','first_batch_CE_signal'):
        assert {r['spectrum_variant'] for r in read('observed/'+table+'.json')}=={'raw','corrected'}
    points=read('observed/endpoints.json');pairs=read('observed/spectrum_comparisons.json');assert len(pairs)==3
    for row in pairs:
        cid=row['client_id']
        for epoch in (0,1,5):
            for split in ('train','test'):
                a=next(p for p in points if (p['client_id'],p['spectrum_variant'],p['epoch'],p['split'])==(cid,'raw',epoch,split))
                b=next(p for p in points if (p['client_id'],p['spectrum_variant'],p['epoch'],p['split'])==(cid,'corrected',epoch,split))
                assert row[f'epoch_{epoch}_corrected_minus_raw_{split}_accuracy']==b['accuracy']-a['accuracy']
                assert row[f'epoch_{epoch}_corrected_minus_raw_{split}_CE']==b['ce']-a['ce']
        assert row['epoch_5_minus_epoch_0_test_accuracy_gap_change']==row['epoch_5_corrected_minus_raw_test_accuracy']-row['epoch_0_corrected_minus_raw_test_accuracy']
        for variant in ('raw','corrected'):
            for split in ('train','test'):
                a=next(p for p in points if (p['client_id'],p['spectrum_variant'],p['epoch'],p['split'])==(cid,variant,1,split))
                b=next(p for p in points if (p['client_id'],p['spectrum_variant'],p['epoch'],p['split'])==(cid,variant,5,split))
                assert row[f'{variant}_{split}_CE_epoch5_minus_epoch1']==b['ce']-a['ce']
    for summary in read('observed/capacity_summary.json'):
        selected=[r for r in points if (r['rank_rate'],r['spectrum_variant'],r['epoch'],r['split'])==
            (summary['rank_rate'],summary['spectrum_variant'],summary['epoch'],summary['split'])]
        assert summary['accuracy']==sum(r['correct'] for r in selected)/sum(r['samples'] for r in selected)
    for name in ('endpoints','fixed_slot_updates','first_step_gradients','first_batch_CE_signal','batch_order','rank_comparisons','capacity_summary'):
        assert read('rank_new/'+name+'.json')==read('rank_frozen/'+name+'.json'),name
    for a,b in zip(read('rank_new/local_training.json'),read('rank_frozen/local_training.json')):
        for key in ('client_id','fc1_rank','final_model_sha256','all_batch_indices_sha256','initial_rng_sha256','final_rng_sha256',
            'ce_loss','regularization_loss','total_loss','train_batches','processed_train_samples','observed_clip_fraction'):
            assert a[key]==b[key],key
    for a,b in zip(read('rank_new/tail_rollback.json'),read('rank_frozen/tail_rollback.json')):
        assert {k:v for k,v in a.items() if k!='evaluation_seconds'}=={k:v for k,v in b.items() if k!='evaluation_seconds'}
    for label in ('observed','quiet'):
        root=out/label;manifest=read(label+'/manifest.json')
        assert manifest['common_fc1_svd_calls']==1 and manifest['other_layer_download_svd_calls']==6
        assert manifest['source']['source_config']['multi_svd_spectral_mode']=='next_tier'
        assert not manifest['spectrum_comparison']['tail_rollback_performed']
        assert not list(root.glob('tail_rollback.*')) and not list(root.rglob('*.pt')) and not list(root.glob('transient_training_*'))
        assert all(r['common_prefix_rank']==120 and r['same_parameter_count'] for r in manifest['common_prefix_checks'])
        assert manifest['input_hashes_unchanged']
    for label in ('observed','quiet','rank_new','rank_frozen'):
        for file in (out/label).glob('*.csv'):
            rows=json.loads(file.with_suffix('.json').read_text())
            with file.open(encoding='utf-8-sig',newline='') as stream:saved=list(csv.DictReader(stream))
            assert len(rows)==len(saved)
            for row,actual in zip(rows,saved):
                for key,cell in actual.items():
                    v=row.get(key)
                    if isinstance(v,(list,dict)):assert json.loads(cell)==v
                    else:assert cell==('' if v is None else str(v))
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',source='same accepted B_fc1 S2; offline R3, not real R102',
        clients=[0,10,18],rank=200,spectrum_variants=['raw','corrected'],training_tasks_per_spectrum_run=6,epochs=5,
        SGD_steps_per_spectrum_run=30,one_shared_fc1_SVD_and_equal_parameter_counts=True,
        prefix120_and_other_initial_state_exact=True,own_download_slot_changes=True,paired_batches_and_RNG_exact=True,
        diagnostics_toggle_parameters_losses_endpoints_batches_RNG_exact=True,
        default_rank_vs_frozen_6697331_parameters_losses_endpoints_slot_updates_batches_RNG_exact=True,
        summary_pairing_CE_changes_and_CSV_JSON_verified=True,tail_rollback_or_model_export=False,inputs_unchanged=True,
        real_accuracy_or_generalization_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
