"""Four-way real CLI synthetic acceptance; never starts a real-data experiment."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(command,log,success=True):
    with Path(log).open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=ROOT/'system',env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',
            PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
    if (result.returncode==0)!=success:raise RuntimeError(f'Unexpected exit {result.returncode}: {log}')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--resume-owned-output',action='store_true')
    args=p.parse_args();out=Path(args.output).resolve();settings=dict(verifier='whole-network-spectrum-v1',clients=20,seed=0)
    if args.resume_owned_output:
        if json.loads((out/'settings.json').read_text())!=settings:raise ValueError('Not an owned verifier root')
    else:
        out.mkdir(parents=True,exist_ok=False)
        import numpy as np
        rng=np.random.default_rng(144);directory=out/'data/Cifar100/fixture'
        for split in ('train','test'):
            (directory/split).mkdir(parents=True)
            for cid in range(20):
                count=3+cid%3 if split=='train' else 2+cid%2
                x=(rng.normal(size=(count,3,32,32))*.2).astype('float32');y=(np.arange(count)+cid)%8
                np.savez(directory/split/f'{cid}.npz',data=dict(x=x,y=y))
        (directory/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
        (out/'settings.json').write_text(json.dumps(settings,indent=2))
    common=[sys.executable,str(ROOT/'system/whole_network_spectrum_diagnostic.py'),'-did','0','--device','cpu',
        '--data-root',str(out/'data'),'--data-subdir','fixture','--client-id','0','--seed','0','--capacity-ratios','.15,.25,.35,.9',
        '--epochs','5','--batch-size','16','--test-batch-size','16','--lr','.005','--f-lambda','.001','--grad-clip','10','--threads','2']
    for label,extra in [('observed',[]),('quiet',['--diagnostics','0'])]:
        file=out/label/'manifest.json'
        if file.is_file():
            if json.loads(file.read_text())['status']!='complete':raise ValueError('Partial diagnostic output; choose a new root')
        else:
            print('RUN',label,'four independent local5 cases',flush=True)
            execute(common+extra+['--output-dir',str(out/label)],out/(label+'.log'))
    read=lambda path:json.loads((out/path).read_text())
    manifest=read('observed/manifest.json');other=read('quiet/manifest.json')
    assert manifest['case_count']==4 and manifest['actual_SGD_steps']==20 and manifest['actual_training_sample_visits']==60
    assert manifest['construction']['actual_layer_ranks']==dict(conv2=[12,20,28,72],fc1=[120,200,280,720],fc2=[77,128,179,461])
    assert manifest['construction']['pre_aggregation_SVD_calls']==manifest['construction']['shared_E1_E2_download_SVD_calls']==3
    assert all(r['passed'] for r in manifest['construction']['construction_checks']+manifest['initial_function_checks'])
    for a,b in zip(read('observed/local_training.json'),read('quiet/local_training.json')):
        for key in ('case_id','actual_layer_ranks','model_parameter_count','initial_model_sha256','initial_rng_sha256','final_rng_sha256',
            'final_model_sha256','batch_order_sha256','ce_loss','regularization_loss','total_loss','raw_mse','train_batches','processed_train_samples','clip_trigger_fraction'):
            assert a[key]==b[key],(a['case_id'],key)
    training=read('observed/local_training.json');points=read('observed/endpoints.json')
    counts={r['case_id']:r['model_parameter_count'] for r in training}
    assert counts['E1']==counts['E2']==counts['E4'] and counts['E3']<counts['E4']
    assert next(r for r in training if r['case_id']=='E4')['initial_model_sha256']==manifest['construction']['native_large_state_sha256']
    assert len({r['batch_order_sha256'] for r in training})==len({r['initial_rng_sha256'] for r in training})==len({r['final_rng_sha256'] for r in training})==1
    assert len(points)==24 and points==read('quiet/endpoints.json')
    assert len(read('observed/comparisons.json'))==9 and read('observed/comparisons.json')==read('quiet/comparisons.json')
    for row in read('observed/comparisons.json'):
        high,low=row['contrast'].split('_minus_')
        for split in ('train','test'):
            a=next(r for r in points if (r['case_id'],r['epoch'],r['split'])==(high,row['epoch'],split))
            b=next(r for r in points if (r['case_id'],r['epoch'],r['split'])==(low,row['epoch'],split))
            assert row[split+'_accuracy_difference']==a['correct']/a['samples']-b['correct']/b['samples']
            assert row[split+'_CE_difference']==a['ce']-b['ce']
    gains=read('observed/repair_gains.json')
    assert max(r['actual_repair_gain'] or 0 for r in gains)>3.9  # No old 1.5 clamp.
    assert {r['theoretical_gain'] for r in gains}=={1.,4/3,2.,4.}
    assert not any('checkpoint' in k and k.endswith('_file') for k in manifest['requested'])
    for label in ('observed','quiet'):
        m=read(label+'/manifest.json');assert m['data_unchanged'] and not list((out/label).rglob('*.pt')) and not list((out/label).glob('transient_training_*'))
        for file in (out/label).glob('*.csv'):
            rows=json.loads(file.with_suffix('.json').read_text())
            with file.open(encoding='utf-8-sig',newline='') as stream:actual=list(csv.DictReader(stream))
            assert len(rows)==len(actual)
            for row,saved in zip(rows,actual):
                for key,cell in saved.items():
                    v=row.get(key)
                    if isinstance(v,(list,dict)):assert json.loads(cell)==v
                    else:assert cell==('' if v is None else str(v))
    def finite(value):
        if isinstance(value,float):assert math.isfinite(value)
        elif isinstance(value,dict):
            for x in value.values():finite(x)
        elif isinstance(value,list):
            for x in value:finite(x)
    for file in (out/'observed').glob('*.json'):finite(json.loads(file.read_text()))
    execute(common+['--data-subdir','missing','--output-dir',str(out/'bad_missing')],out/'bad_missing.log',False)
    execute(common+['--capacity-ratios','.15,1','--output-dir',str(out/'bad_rates')],out/'bad_rates.log',False)
    result=dict(status='PASS_CPU_SYNTHETIC_ONLY',client_id=0,case_count=4,epochs=5,SGD_steps_per_run=20,
        simulated_upload_count=4,pre_aggregation_SVD_calls=3,shared_E1_E2_download_SVD_calls=3,
        expected_attenuation=[1.,.75,.5,.25],ideal_gains=[1.,4/3,2.,4.],
        E2_E4_matrix_checks=[r for r in manifest['construction']['construction_checks'] if 'E2 repaired' in r['label']],
        E2_E4_initial_logit_checks=manifest['initial_function_checks'],
        native_factors_common_initial_parameters_and_optimizer_unit_checks=True,
        paired_batches_RNG_and_observation_toggle_parameters_losses_endpoints_exact=True,
        CSV_JSON_and_fixed_epoch_comparisons_verified=True,source_data_unchanged=True,default_tables_only=True,
        real_CIFAR100_accuracy_or_capacity_benefit_verified=False)
    (out/'acceptance_report.json').write_text(json.dumps(result,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


if __name__=='__main__':main()
