"""Optional fifth local5 CLI smoke and frozen 222fe98 four-case regression."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(command,path):
    with Path(path).open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT/'system',env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',
            PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Exit {result.returncode}: {path}')


def worker(out,label):
    argv=['--device','cpu','--data-root',str(out/'data'),'--data-subdir','fixture','--threads','2','--output-dir',str(out/label)]
    if label=='five':argv+=['--include-e4-svd']
    if label=='frozen':
        from whole_network_spectrum_diagnostic import parser
        source=subprocess.check_output(['git','show','222fe98:system/utils/whole_network_spectrum_diagnostic.py'],cwd=ROOT,text=True,encoding='utf-8')
        namespace=dict(__name__='frozen_222fe98_whole',__file__=str(ROOT/'system/utils/whole_network_spectrum_diagnostic.py'))
        exec(compile(source,'222fe98/whole_network_spectrum_diagnostic.py','exec'),namespace)
        return namespace['run'](parser().parse_args(argv))
    import runpy
    sys.argv=['whole_network_spectrum_diagnostic.py']+argv
    runpy.run_path(str(ROOT/'system/whole_network_spectrum_diagnostic.py'),run_name='__main__')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--worker',choices=['default','five','frozen']);p.add_argument('--resume-owned-output',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();settings=dict(verifier='whole-e4-svd-v1',N=20,seed=0)
    if args.worker:return worker(out,args.worker)
    if args.resume_owned_output:
        if json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Not an owned verifier workspace')
    else:
        import numpy as np
        out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(144);directory=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (directory/split).mkdir(parents=True)
            for cid in range(20):
                count=3+cid%3 if split=='train' else 2+cid%2
                np.savez(directory/split/f'{cid}.npz',data=dict(x=(rng.normal(size=(count,3,32,32))*.2).astype('float32'),y=(np.arange(count)+cid)%8))
        (directory/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'settings.json').write_text(json.dumps(settings))
    for label in ('default','five','frozen'):
        file=out/label/'manifest.json'
        if file.is_file():
            if json.loads(file.read_text())['status']!='complete':raise ValueError('Partial output; choose a new workspace')
        else:
            print('RUN',label,flush=True);execute([sys.executable,__file__,'--output',str(out),'--worker',label],out/(label+'.log'))
    read=lambda name:json.loads((out/name).read_text())
    default=read('default/manifest.json');five=read('five/manifest.json');frozen=read('frozen/manifest.json')
    assert default['case_count']==frozen['case_count']==4 and five['case_count']==5
    assert five['actual_SGD_steps']==25 and five['actual_training_sample_visits']==75
    assert five['construction']['pre_aggregation_SVD_calls']==five['construction']['shared_E1_E2_download_SVD_calls']==3
    assert five['construction']['E4_SVD_source']['additional_original_SVD_calls']==0
    assert five['construction']['student_initial_state_sha256']['E4_SVD']==five['construction']['E4_SVD_source']['pre_aggregation_balanced_state_sha256']
    assert five['construction']['E4_SVD_source']['actual_layer_ranks']==dict(conv2=72,fc1=720,fc2=461)
    assert len(five['initial_function_checks'])==6 and len(five['construction']['construction_checks'])==15
    assert all(r['passed'] for r in five['initial_function_checks']+five['construction']['construction_checks'])
    assert len(read('five/endpoints.json'))==30 and len(read('five/comparisons.json'))==15
    contrasts={r['contrast'] for r in read('five/comparisons.json')}
    assert {'E2_minus_E4_SVD','E4_SVD_minus_E4','E2_minus_E1','E4_minus_E3','E2_minus_E4'}==contrasts
    for table in ('endpoints','comparisons','batch_order','initial_norms','spectra','repair_gains','E1_E2_spectral_gap'):
        assert read('default/'+table+'.json')==read('frozen/'+table+'.json'),table
    a,b,c=read('default/local_training.json'),read('five/local_training.json'),read('frozen/local_training.json')
    for expected,actual in [(a,b[:4]),(a,c)]:
        for left,right in zip(expected,actual):
            for key in ('case_id','actual_layer_ranks','model_parameter_count','initial_model_sha256','initial_rng_sha256','final_rng_sha256',
                'final_model_sha256','batch_order_sha256','ce_loss','regularization_loss','total_loss','train_batches','processed_train_samples','clip_trigger_fraction'):
                assert left[key]==right[key],key
    assert len({r['batch_order_sha256'] for r in b})==len({r['initial_rng_sha256'] for r in b})==len({r['final_rng_sha256'] for r in b})==1
    assert b[-1]['case_id']=='E4_SVD' and b[-1]['model_parameter_count']==b[0]['model_parameter_count']
    for row in read('five/comparisons.json'):
        high,low=row['contrast'].split('_minus_')
        for split in ('train','test'):
            points=read('five/endpoints.json')
            hi=next(r for r in points if (r['case_id'],r['epoch'],r['split'])==(high,row['epoch'],split))
            lo=next(r for r in points if (r['case_id'],r['epoch'],r['split'])==(low,row['epoch'],split))
            assert row[split+'_accuracy_difference']==hi['accuracy']-lo['accuracy']
            assert row[split+'_CE_difference']==hi['ce']-lo['ce']
    for label in ('default','five','frozen'):
        assert read(label+'/manifest.json')['data_unchanged']
        assert not list((out/label).rglob('*.pt')) and not list((out/label).glob('transient_training_*'))
        for file in (out/label).glob('*.csv'):
            rows=json.loads(file.with_suffix('.json').read_text())
            with file.open(encoding='utf-8-sig',newline='') as stream:actual=list(csv.DictReader(stream))
            assert len(rows)==len(actual)
            for row,saved in zip(rows,actual):
                for key,cell in saved.items():
                    v=row.get(key)
                    if isinstance(v,(list,dict)):assert json.loads(cell)==v
                    else:assert cell==('' if v is None else str(v))
    report=dict(status='PASS_CPU_SYNTHETIC_ONLY',case_count=5,epochs=5,SGD_steps=25,original_SVD_calls=6,extra_original_SVD_calls=0,
        E4_SVD_ranks=[72,720,461],all_five_batch_and_RNG_equal=True,
        original_four_cases_unchanged_when_fifth_enabled=True,four_default_vs_frozen_222fe98_parameters_losses_tables_batches_RNG_exact=True,
        new_initial_matrix_checks=[r for r in five['construction']['construction_checks'] if 'E4_SVD' in r['label']],
        new_initial_logit_checks=[r for r in five['initial_function_checks'] if 'E4_SVD' in r['label']],
        actual_training_contrasts=read('five/comparisons.json'),CSV_JSON_verified=True,source_data_unchanged=True,
        real_pat20_performance_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(report,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
