"""Isolated synthetic full-width N20 five-arm CLI acceptance; no real R5 input.

Fixtures deliberately use artificial R5-role metadata and controlled finite
weights, not a claimed real five-round history. They validate replay mechanics.
"""
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
    import copy
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'tests'))
    from test_single_avg_proto import options
    from common.experiment import environment,write_json
    from common.client_data import audit_image_clients
    from utils import multi_group_svd as svd,multi_svd_spectral as spectral,multi_rank_partition as rp
    from utils.cnn512_reporting import model_digest
    torch.set_num_threads(2);data=out/'data/Cifar100/fixture';gen=np.random.default_rng(616)
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        for cid in range(20):
            n=2+cid%3 if split=='train' else 1+cid%4
            x=(gen.normal(size=(n,3,32,32))*.1).astype('float32');x[:,0,0,0]=np.arange(n)/10
            np.savez(data/split/f'{cid}.npz',data=dict(x=x,y=(np.arange(n)+cid)%6))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='pat',class_per_client=20)))
    shared=dict(device='cpu',global_rounds=4,local_epochs=5,batch_size=16,local_learning_rate=.005,
        control_test_batch_size=16,multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_aggregation='class_sample',
        multi_svd_init_rank_rate=.9,multi_svd_spectral_mode='none',multi_svd_spectral_layers='conv2,fc1,fc2',
        multi_svd_spectral_max_gain=1.5,multi_svd_five_diagnostics=0,data_root=str(out/'data'),data_subdir='fixture',data_formats='npz_only')
    a=options(**shared);initial=svd.initial_model(a);initial_sha=model_digest(initial)
    for label,gain in [('h',.85),('hplus',.95),('l',1.)]:
        kw=dict(shared)
        if label=='l':kw['multi_svd_ratios']='.90'
        if label=='hplus':kw['multi_svd_spectral_mode']='all_tail'
        args=options(**kw);args.multi_svd_avg_train_samples=[2+i%3 for i in range(20)]
        model=copy.deepcopy(initial)
        with torch.no_grad():
            for name in ('conv2','fc1','fc2'):getattr(model,name).weight.mul_(gain)
            model.fc3.bias.add_(torch.linspace(-.001,.001,100))
        manifest=dict(algorithm='FedMultiGroupProj',actual_train_count=5,server_model_count=1,client_count=20,
            aggregation_mode='group_avg_only',effective=vars(args),svd_spectral=dict(policy=spectral.policy(args)),
            optimizer=dict(name='SGD',lr=.005,momentum=0.,weight_decay=0.,gradient_norm_cap=10.),environment=environment(),
            data_manifest=audit_image_clients(args),**svd.model_spec(args,model=model),model_family=args.model_family,
            svd_initialization=svd.initialization_protocol(args),svd_capacity_protocol=svd.capacity_protocol(args),
            initial_parameter_sha256=initial_sha,
            synthetic_fixture='Artificial R5-role fixture: common .90 initialization with known layer scalings; NOT a real five-round trajectory')
        folder=out/'sources'/label;folder.mkdir(parents=True)
        payload=dict(state_dict=model.state_dict(),model_id=0,role='post_aggregation_full_server',completed_train_count=5,
            svd_aggregation=rp.protocol(args),**svd.model_spec(args,model=model),model_family=args.model_family,
            svd_initialization=svd.initialization_protocol(args),capacity_protocol=svd.capacity_protocol(args),initial_parameter_sha256=initial_sha)
        if label=='hplus':payload['svd_spectral_policy']=spectral.policy(args)
        torch.save(payload,folder/'Server_model_0.pt');write_json(folder/'manifest.json',manifest)


def worker(out):
    import torch
    from utils import svd_r6_replay as d
    from flcore.trainmodel.models import Hyper_CNN_512
    counters=dict(SVD=0,SGD=0,objective=0,F=0,paired_downloads=0,ordinary_downloads=0);context=dict(training=False)
    native_svd=torch.linalg.svd;native_step=torch.optim.SGD.step;native_download=d.paired_download;native_train=d.train_branch
    from flcore.clients.clientTargetProj import clientTargetProj
    native_objective=clientTargetProj._objective;native_f=Hyper_CNN_512.frobenius_decay
    def SVD(*a,**kw):counters['SVD']+=1;return native_svd(*a,**kw)
    def download(mother,rate,args,cid,repair=True):
        before=counters['SVD'];result=native_download(mother,rate,args,cid,repair)
        assert counters['SVD']-before==3
        counters['paired_downloads' if repair else 'ordinary_downloads']+=1
        return result
    def train(model,*a,**kw):
        context.update(training=True,model=model)
        try:return native_train(model,*a,**kw)
        finally:context['training']=False
    def step(opt,*a,**kw):
        assert context['training'];counters['SGD']+=1
        assert [id(p) for p in context['model'].parameters()]==[id(p) for g in opt.param_groups for p in g['params']]
        assert all(g['lr']==.005 and g['momentum']==g['weight_decay']==0 for g in opt.param_groups)
        return native_step(opt,*a,**kw)
    def objective(*a,**kw):counters['objective']+=1;return native_objective(*a,**kw)
    def f(*a,**kw):counters['F']+=1;return native_f(*a,**kw)
    sys.argv=['svd_r6_replay.py','-did','0','--device','cpu','--h-run',str(out/'sources/h'),
        '--hplus-run',str(out/'sources/hplus'),'--l-run',str(out/'sources/l'),'--data-root',str(out/'data'),
        '--data-subdir','fixture','--threads','2','--output-dir',str(out/'results')]
    with ExitStack() as stack:
        for obj,name,value in [(torch.linalg,'svd',SVD),(torch.optim.SGD,'step',step),(d,'paired_download',download),
            (d,'train_branch',train),(clientTargetProj,'_objective',objective),(Hyper_CNN_512,'frobenius_decay',f)]:
            stack.enter_context(patch.object(obj,name,value))
        runpy.run_path(str(ROOT/'system/svd_r6_replay.py'),run_name='__main__')
    assert counters==dict(SVD=420,SGD=500,objective=500,F=500,paired_downloads=40,ordinary_downloads=100),counters
    (out/'counters.json').write_text(json.dumps(counters,indent=2))


def verify(out):
    import csv
    from PIL import Image
    root=out/'results';read=lambda key:json.loads((root/(key+'.json')).read_text())
    m=read('manifest');assert m['status']=='complete' and m['all_five_batches_and_RNG_paired'] and m['R7_SGD_steps']==0
    assert m['actual_training_tasks']==100 and m['actual_SGD_steps']==500 and m['actual_SGD_sample_visits']==1475
    endpoints=read('endpoints');assert len(endpoints)==1360 and len(read('epoch_training'))==500
    for row in read('cohort_endpoints'):
        members=[r for r in endpoints if r['branch']==row['branch'] and r['phase']==row['phase'] and r['epoch']==row['epoch'] and r['split']==row['split'] and r['client_id'] in row['member_ids']]
        assert row['accuracy']==sum(r['correct'] for r in members)/sum(r['samples'] for r in members)
    batches=read('batch_fingerprints')
    for cid in range(20):
        rows=[r for r in batches if r['client_id']==cid];assert len(rows)==5
        for field in ('batch_sha256','initial_rng_sha256','final_rng_sha256','epoch_batch_sha256'):
            assert all(r[field]==rows[0][field] for r in rows)
    for row in endpoints:
        if row['client_id'] in (0,5,10,15):
            assert row['rank_rate']==.9 and row['actual_layer_ranks']==dict(conv2=72,fc1=720,fc2=461)
        if row['phase']=='R7_pre_ordinary':assert row['branch']!='L_raw' and row['epoch']==0
    local_epochs=read('epoch_training')
    for row in read('cohort_epoch_training'):
        chosen=[r for r in local_epochs if r['branch']==row['branch'] and r['epoch']==row['epoch'] and r['client_id'] in row['member_ids']]
        assert row['train_mean_weighted_F']==sum(r['train_mean_weighted_F']*r['train_sample_visits'] for r in chosen)/sum(r['train_sample_visits'] for r in chosen)
        assert row['clip_trigger_fraction']==sum(r['clip_trigger_fraction']*r['optimizer_steps'] for r in chosen)/sum(r['optimizer_steps'] for r in chosen)
    assert len(read('aggregations'))==4 and len(read('comparisons'))==96
    for path in root.glob('*.csv'):
        expected=json.loads(path.with_suffix('.json').read_text())
        with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        assert len(rows)==len(expected),path
        for row,value in zip(rows,expected):
            for key,v in value.items():assert (json.loads(row[key])==v if isinstance(v,(list,dict)) else row[key]==('' if v is None else str(v))),(path,key)
    for path in root.glob('*.png'):
        with Image.open(path) as image:assert image.size[0]>=1000 and image.size[1]>=500
    assert len(list(root.glob('*.png')))==5 and not list(root.rglob('*.pt'))
    report=dict(status='PASS_CPU_SYNTHETIC_ONLY',N=20,branches=5,source_role_fixture_round=5,local_replay_round=6,R7_SGD=0,
        artificial_source_history=True,note='Finite synthetic R5-role fixtures test the complete replay; no real R5 checkpoint or real accuracy conclusion',
        original_SVD_calls=420,local_SGD_steps=500,actual_sample_visits=1475,epoch_endpoint_rows=1360,
        per_epoch_training_rows=500,comparison_rows=96,all_five_batches_and_RNG_paired=True,
        all_focus_clients_same_high_capacity=True,unequal_samples_weighted_metrics_and_CSV_JSON_verified=True,
        figures_verified=5,no_output_models_or_transient_uploads=True,source_and_data_hashes_unchanged=m['source_artifacts_unchanged'])
    if (out/'negative.json').is_file():report['negative_source_validation']=json.loads((out/'negative.json').read_text())
    (out/'acceptance_report.json').write_text(json.dumps(report,indent=2));print('PASS',out/'acceptance_report.json',flush=True)


def negative(out):
    import copy
    from svd_r6_replay import parser
    from utils import svd_r6_replay as d
    torch=d.torch;torch.set_num_threads(2)
    cli=parser().parse_args(['--h-run',str(out/'sources/h'),'--hplus-run',str(out/'sources/hplus'),
        '--l-run',str(out/'sources/l'),'--device','cpu','--data-root',str(out/'data'),'--data-subdir','fixture','--output-dir',str(out/'never_written')])
    files=[p for p in (out/'sources').rglob('*') if p.is_file()];hashes={p:d.sha256(p) for p in files}
    native=d.svd.load_checkpoint;checks=[]
    def wrong_role(*a,**kw):
        model,payload=native(*a,**kw);payload=dict(payload,role='ordinary_post_local_low_rank');return model,payload
    with patch.object(d.svd,'load_checkpoint',side_effect=wrong_role):
        try:d.load_sources(cli)
        except ValueError as e:assert 'SERVER' in str(e);checks.append('client_role_rejected')
        else:raise AssertionError('Wrong source role accepted')
    def wrong_dtype(*a,**kw):
        model,payload=native(*a,**kw);payload=copy.deepcopy(payload);payload['state_dict']['fc3.bias']=payload['state_dict']['fc3.bias'].double();return model,payload
    with patch.object(d.svd,'load_checkpoint',side_effect=wrong_dtype):
        try:d.load_sources(cli)
        except ValueError as e:assert 'dtype or alias' in str(e);checks.append('dtype_alias_mismatch_rejected')
        else:raise AssertionError('Wrong source dtype accepted')
    data=json.loads((out/'sources/h/manifest.json').read_text())['data_manifest'];data['files'][0]['sha256']='0'*64
    with patch.object(d,'audit_image_clients',return_value=data):
        try:d.load_sources(cli)
        except ValueError as e:assert 'fingerprints differ' in str(e);checks.append('file_decoded_fingerprint_mismatch_rejected')
        else:raise AssertionError('Wrong data accepted')
    missing=copy.copy(cli);missing.h_run=str(out/'missing_source')
    try:d.resolve_source(missing,'h')
    except FileNotFoundError:checks.append('missing_source_rejected_without_fallback')
    else:raise AssertionError('Missing source accepted')
    assert all(d.sha256(file)==value for file,value in hashes.items())
    (out/'negative.json').write_text(json.dumps(dict(status='PASS',checks=checks,input_files_unchanged=True),indent=2))
    print('PASS negative source checks',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--phase',choices=['prepare','worker','negative','verify'])
    cli=p.parse_args();out=Path(cli.output).resolve()
    if cli.phase:return globals()[cli.phase](out)
    out.mkdir(parents=True,exist_ok=False)
    for phase in ('prepare','worker','negative','verify'):
        print('RUN',phase,flush=True)
        with (out/(phase+'.log')).open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'-u',__file__,'--output',str(out),'--phase',phase],cwd=ROOT/'system',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONIOENCODING='utf-8',PYTHONFAULTHANDLER='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{phase} exit={r.returncode}: {out}/{phase}.log')
        print('PASS',phase,flush=True)


if __name__=='__main__':main()
