"""Full N20 R6/R7 CLI acceptance from actual formal H R1-R5 synthetic exports."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def prepare(out):
    import numpy as np
    import shlex
    data=out/'data/Cifar100/fixture';gen=np.random.default_rng(920)
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3 if split=='train' else 1+cid%4
            x=(gen.normal(size=(n,3,32,32))*.1).astype('float32')
            np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)+cid)%6))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    # Use the actual H command and main, including the unmodified formal
    # Client/Server lifecycle/export. Do NOT fill, strip or rewrite any payload.
    line=next(l.strip()[1:-1] for l in (ROOT/'system/svd_five_round_commands.sh').read_text().splitlines()
        if l.strip().startswith('"') and '/H/checkpoints' in l)
    argv=shlex.split(line)[3:]
    for flag,value in [('-did',0),('-dev','cpu'),('--data-root',out/'data'),('--data-subdir','fixture'),
        ('-clip_cpu_threads',2),('-exp_name','H_legacy_export_acceptance'),('-sfn',out/'source/checkpoints'),
        ('--run_log_dir',out/'source/logs'),('--h5_result_root',out/'source/h5'),('--final-model-root',out/'source/final_models')]:
        argv[argv.index(flag)+1]=str(value)
    sys.argv=['main.py']+argv
    runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')


def worker(out):
    import torch
    from utils import svd_r6_replay as d
    from flcore.trainmodel.models import Hyper_CNN_512
    from flcore.clients.clientTargetProj import clientTargetProj
    counts=dict(native_download_SVD=0,SGD=0,objective=0,F=0,training_tasks=0);context={}
    original_download=d.paired_download;original_train=d.train_branch;step=torch.optim.SGD.step
    objective=clientTargetProj._objective;f=Hyper_CNN_512.frobenius_decay
    def download(*a,**kw):
        native=torch.linalg.svd
        with patch.object(torch.linalg,'svd',wraps=native) as observed:
            result=original_download(*a,**kw)
        assert observed.call_count==3;counts['native_download_SVD']+=3;return result
    def train(model,*a,**kw):
        counts['training_tasks']+=1;context['model']=model
        try:return original_train(model,*a,**kw)
        finally:context.clear()
    def update(opt,*a,**kw):
        assert [id(p) for p in context['model'].parameters()]==[id(p) for g in opt.param_groups for p in g['params']]
        assert all(g['lr']==.005 and g['momentum']==g['weight_decay']==0 for g in opt.param_groups)
        counts['SGD']+=1;return step(opt,*a,**kw)
    def ce(*a,**kw):counts['objective']+=1;return objective(*a,**kw)
    def regular(*a,**kw):counts['F']+=1;return f(*a,**kw)
    sys.argv=['matrix_comp_replay.py','-did','0','--device','cpu','--source-run',str(out/'source'),
        '--data-root',str(out/'data'),'--data-subdir','fixture','--threads','2','--output-root',str(out/'results')]
    with ExitStack() as stack:
        for obj,name,value in [(d,'paired_download',download),(d,'train_branch',train),(torch.optim.SGD,'step',update),
            (clientTargetProj,'_objective',ce),(Hyper_CNN_512,'frobenius_decay',regular)]:stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/matrix_comp_replay.py'),run_name='__main__')
    (out/'counters.json').write_text(json.dumps(counts,indent=2))
    assert counts==dict(native_download_SVD=21,SGD=600,objective=600,F=600,training_tasks=120),counts


def verify(out):
    import csv
    from PIL import Image
    root=out/'results';read=lambda name:json.loads((root/(name+'.json')).read_text())
    m=read('manifest');assert m['status']=='complete' and m['source_Avg_validated'] and m['all_batches_and_RNG_paired']
    validation=read('source_validation')
    assert validation['legacy_compatible_client_ids']==list(range(20)) and validation['direct_data_bound_client_ids']==[]
    for row in validation['clients']:
        fields=row['checkpoint_fingerprint_fields']
        assert not fields['top_level_present'] and fields['downloaded_model_is_null']
        assert not fields['download_context_fingerprint_present']
    assert m['actual_training_tasks']==120 and m['actual_SGD_steps']==600 and m['actual_SGD_sample_visits']==1770
    endpoints=read('endpoints');assert len(endpoints)==1560 and len(read('epoch_training'))==600
    for row in read('cohort_endpoints'):
        selected=[r for r in endpoints if all(r[k]==row[k] for k in ('round','branch','phase','epoch','split')) and r['client_id'] in row['member_ids']]
        assert row['accuracy']==sum(r['correct'] for r in selected)/sum(r['samples'] for r in selected)
    batches=read('batch_fingerprints')
    for round_id in (6,7):
        for cid in range(20):
            rows=[r for r in batches if r['round']==round_id and r['client_id']==cid];assert len(rows)==3
            for key in ('batch_sha256','initial_rng_sha256','final_rng_sha256','epoch_batch_sha256'):
                assert all(r[key]==rows[0][key] for r in rows)
    assert len(read('compensation_segments'))==24 and len(read('capacity_boundaries'))==30
    for row in read('compensation_segments'):
        assert row['actual_taken_rank']<=min(row['planned_rank'],row['numerical_rank'])
        assert row['q']==row['contributor_train_samples_total']/row['all_train_samples']
    assert len(read('aggregations'))==4
    for branch in ('H','Hplus','matrix'):
        assert json.loads((root/branch/'manifest.json').read_text())['branch']==branch
        assert len(json.loads((root/branch/'local_training.json').read_text()))==40
    for path in root.rglob('*.csv'):
        expected=json.loads(path.with_suffix('.json').read_text())
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(expected)
        for row,value in zip(rows,expected):
            for key,v in value.items():assert (json.loads(row[key])==v if isinstance(v,(list,dict)) else row[key]==('' if v is None else str(v))),(path,key)
    figures=list(root.glob('*.png'));assert len(figures)==7
    for path in figures:
        with Image.open(path) as im:assert im.size[0]>=1000 and im.size[1]>=500
    assert not list(root.rglob('*.pt'))
    report=dict(status='PASS_CPU_SYNTHETIC_ONLY',N=20,branches=3,training_rounds=[6,7],epochs_per_task=5,
        synthetic_source='Actual main.py H command ran all R1-R5 on synthetic data; original formal final export untouched, downloaded_model=None and no top-level data fingerprint',
        legacy_compatible_clients=20,original_export_fields_untouched=True,
        counters=json.loads((out/'counters.json').read_text()),source_Avg_verified=True,
        actual_SGD_sample_visits=1770,paired_batches_and_RNG=True,weighted_metrics_and_CSV_JSON=True,
        compensation_segments=24,boundary_records=30,endpoint_rows=1560,curves=7,
        source_and_data_unchanged=m['source_artifacts_unchanged'],no_saved_models_or_transient_uploads=True,
        real_checkpoint_run=False,CUDA_tested=False,real_accuracy_conclusion=False)
    if (out/'negative.json').is_file():report['source_rejection_checks']=json.loads((out/'negative.json').read_text())
    (out/'acceptance_report.json').write_text(json.dumps(report,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


def negative(out):
    import copy
    import torch
    from matrix_comp_replay import parser
    from utils import matrix_comp_replay as d
    torch.set_num_threads(2)
    cli=parser().parse_args(['--source-run',str(out/'source'),'--data-root',str(out/'data'),'--data-subdir','fixture',
        '--device','cpu','--threads','2','--output-root',str(out/'not_created')])
    source,p,args,data,files,provenance=d.load_source(cli);checks=[]
    hashes={path:d.sha256(path) for path in Path(provenance['server_file']).parent.glob('*') if path.is_file()}
    model,cp=d.svd.load_checkpoint(files[0]);fp=args.data_fingerprint
    export=provenance['legacy_export_context']
    assert cp['single_avg_prototype_usage']['downloaded_model'] is None and 'data_fingerprint' not in cp
    result=d.check_client(model,cp,0,args,p,fp,legacy_export=export)
    assert result['data_fingerprint_binding']=='legacy_same_export';checks.append('formal_H_legacy_export_accepted_without_rewriting')
    for key,value in [('client_id',1),('completed_train_count',4),('role','post_aggregation_full_server')]:
        bad=dict(cp);bad[key]=value
        try:d.check_client(model,bad,0,args,p,fp,legacy_export=export)
        except ValueError:checks.append(key+'_rejected')
        else:raise AssertionError(key)
    for top,context in [(fp,'0'*64),('0'*64,fp),(fp,fp)]:
        sample=copy.deepcopy(cp);sample['data_fingerprint']=top
        sample['single_avg_prototype_usage']['downloaded_model']=dict(data_fingerprint=context,initial_sha256=args.initial_sha256)
        if top==context==fp:
            row=d.check_client(model,sample,0,args,p,fp,legacy_export=export)
            assert row['direct_data_binding_locations']==['checkpoint','download context'];checks.append('both_direct_bindings_checked')
        else:
            try:d.check_client(model,sample,0,args,p,fp,legacy_export=export)
            except ValueError:checks.append('top_mismatch_rejected' if top!=fp else 'context_mismatch_rejected')
            else:raise AssertionError('Contradictory direct fingerprint accepted')
    for key,value in [('group_fingerprint','other-run'),('initial_parameter_sha256','0'*64)]:
        sample=copy.deepcopy(cp);sample[key]=value
        try:d.check_client(model,sample,0,args,p,fp,legacy_export=export)
        except ValueError:checks.append(key+'_mismatch_rejected')
        else:raise AssertionError(key)
    try:d.check_client(model,cp,0,args,p,fp,legacy_export=dict(export,same_final_export_directory=False))
    except ValueError:checks.append('unbound_legacy_outside_final_export_rejected')
    else:raise AssertionError('Unbound external legacy file accepted')
    wrong=copy.deepcopy(source)
    with torch.no_grad():wrong.fc3.bias.add_(.01)
    try:d.check_source_Avg(wrong,source,1e-5,1e-5)
    except ValueError:checks.append('recomputed_Avg_mismatch_rejected')
    else:raise AssertionError('Wrong Avg')
    # A compatible-looking different-run upload must still fail the set proof.
    import tempfile
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='negative_upload_set_',dir=out) as folder:
        uploads={};samples={r['client_id']:r['count'] for r in data['files'] if r['split']=='train'}
        for cid,file in files.items():
            local,payload=d.svd.load_checkpoint(file)
            if cid==0:
                with torch.no_grad():local.fc3.bias.add_(.1)
                payload['state_dict']=local.state_dict()
                d.check_client(local,payload,cid,args,p,fp,legacy_export=export)
            upload=Path(folder)/f'{cid}.pt';torch.save(d.svd.recover_parameters(local),upload);uploads[cid]=upload
            del local,payload
        selector=np.random.RandomState(args.seed)
        for _ in range(5):order=selector.choice(np.arange(20),20,replace=False).tolist()
        mixed,_=d.replay.aggregate_once(source,uploads,samples,order)
        try:d.check_source_Avg(mixed,source,1e-5,1e-5)
        except ValueError:checks.append('different_run_weights_with_compatible_metadata_rejected_by_Avg_proof')
        else:raise AssertionError('Mixed upload set accepted')
    manifest=copy.deepcopy(data);manifest['files'][0]['decoded_x_sha256']='0'*64
    with patch.object(d,'audit_image_clients',return_value=manifest):
        try:d.load_source(cli)
        except ValueError:checks.append('actual_decoded_data_mismatch_rejected')
        else:raise AssertionError('Wrong decoded data')
    # Full entry must emit the mismatch report and stop BEFORE the first SGD.
    failure_cli=copy.copy(cli);failure_cli.output_root=str(out/'expected_rejected_Avg')
    original_avg=d.replay.aggregate_once
    def inconsistent_Avg(*a,**kw):
        combined,receipt=original_avg(*a,**kw)
        with torch.no_grad():combined.fc3.bias.add_(.1)
        return combined,receipt
    with patch.object(d.replay,'aggregate_once',side_effect=inconsistent_Avg),patch.object(d.replay,'train_branch') as training:
        try:d.run(failure_cli)
        except ValueError as error:assert 'upload-set Avg differs' in str(error)
        else:raise AssertionError('Entry accepted inconsistent Avg')
        training.assert_not_called()
    rejected=json.loads((out/'expected_rejected_Avg/source_validation.json').read_text())
    assert not rejected['all_passed'] and any(not r['passed'] for r in rejected['Avg_checks'])
    checks.append('full_entry_Avg_failure_reported_before_any_SGD')
    missing=copy.copy(cli);missing.source_run=str(out/'missing_source')
    try:d.resolve_source(missing)
    except FileNotFoundError:checks.append('missing_source_rejected')
    else:raise AssertionError('Missing source')
    assert all(d.sha256(path)==value for path,value in hashes.items())
    (out/'negative.json').write_text(json.dumps(dict(status='PASS',checks=checks,input_files_unchanged=True),indent=2))
    print('PASS source rejection checks',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['prepare','worker','negative','verify'])
    cli=p.parse_args();out=Path(cli.output).resolve()
    if cli.phase:return globals()[cli.phase](out)
    out.mkdir(parents=True,exist_ok=False)
    for phase in ('prepare','worker','negative','verify'):
        print('RUN',phase,flush=True)
        with (out/(phase+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-u',__file__,'--output',str(out),'--phase',phase],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{phase} exit={result.returncode}; see {out}/{phase}.log')
        print('PASS',phase,flush=True)


if __name__=='__main__':main()
