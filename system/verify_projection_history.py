"""Serial actual-main N20, original-width heterogeneous CNN, 2-round acceptance.

Synthetic shards are created only beneath --output. Trained binaries are checked
then deleted unless --keep-models is supplied. No real data or experiment runs.
"""
import argparse
from contextlib import ExitStack
import csv
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def worker(out,label):
    import h5py
    import torch
    from flcore.servers.serverTargetProj import FedTargetProj
    from flcore.clients.clientbase import load_item
    import flcore.clients.clientbase as cb
    from flcore.trainmodel.models import Hyper_CNN_512
    from utils import projection_history as ph
    from utils.target_head import preserved_head_rng
    from utils.head_snapshot import tensor_fingerprint,file_hash
    from utils.projection_variants import aggregate_projection_weighting
    from fedfews_compat.observe import rng_digest
    from torch.utils.data import DataLoader
    job=json.loads((out/'jobs.json').read_text())[label];run=out/label
    if torch.cuda.is_available():torch.cuda.get_rng_state_all()
    context=dict(training=False);batches=[];events=[];forwards={};created=[]
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append((context['round'],context['cid'],y.tolist()))
                yield x,y
    native_forward=Hyper_CNN_512.forward
    def forward(model,*args,**kwargs):
        if context['training']:
            key=(context['round'],context['cid']);forwards[key]=forwards.get(key,0)+1
        return native_forward(model,*args,**kwargs)
    native_init=FedTargetProj.__init__
    def initialize(server,*args,**kwargs):
        native_init(server,*args,**kwargs);created.append(server)
        events.append(dict(kind='initial',model=tensor_fingerprint(load_item('Server','model',server.save_folder_name).state_dict()),rng=rng_digest()))
        for c in server.clients:
            native=c.train
            def train(current_round,c=c,native=native):
                context.update(training=True,round=current_round,cid=c.id)
                try:value=native(current_round)
                finally:context['training']=False
                with preserved_head_rng():model=load_item(c.role,'model',c.save_folder_name)
                assert model.ratio_LR==(.9,.37,.35,.25,.15)[c.id%5]
                assert forwards[current_round,c.id]==5
                events.append(dict(kind='local',loop=current_round,client_id=c.id,model=tensor_fingerprint(model.state_dict()),rng=rng_digest()))
                return value
            c.train=train
        native_aggregate=server.aggregate_parameters_avg
        def aggregate():
            with preserved_head_rng(),torch.no_grad():
                base=dict(load_item('Server','model',server.save_folder_name).named_parameters())
                assert 'fc3.weight' in base and 'fc3.bias' in base
                assert not any('.U.' in n or '.V.' in n for n in base)
                target=server._load_full_parameters(server.target_client_id)
                weights=dict(zip(server.uploaded_ids,server.uploaded_weights))
                def factory():
                    for cid in server.uploaded_ids:
                        yield cid,weights[cid],target if cid==server.target_client_id else server._load_full_parameters(cid)
                old=aggregate_projection_weighting(base,target,factory,server.target_client_id,'projection_softmax',.125,.2)
                if ph.enabled(server.args):
                    result=ph.aggregate(base,target,factory,server.target_client_id,server.projection_history_cache,
                        server.cur_ground,job['strength'],.125,.2,bool(job.get('diagnostics',1)))
                    expected=result[0]
                    assert [r['aggregation_weight'] for r in result[2]]==[r['aggregation_weight'] for r in old[2]]
                    if server.cur_ground==0 or job['strength']==0:
                        assert all(torch.equal(expected[n],old[0][n]) for n in expected)
                    else:
                        assert result[1]['history_source_loop_rounds']=='[0]'
                else:expected=old[0]
                paths=[Path(c.save_folder_name)/f'{c.role}_model.pt' for c in server.clients]
                before=[file_hash(p) for p in paths]
            native_aggregate()
            with preserved_head_rng():actual=dict(load_item('Server','model',server.save_folder_name).named_parameters())
            assert all(torch.equal(actual[n],expected[n]) for n in expected)
            assert before==[file_hash(p) for p in paths]
            events.append(dict(kind='aggregate',loop=server.cur_ground,model=tensor_fingerprint(actual),rng=rng_digest()))
        server.aggregate_parameters_avg=aggregate
    sys.argv=['main.py','-did','0','-algo','FedTargetProj','-data','Cifar100','-m','Decom_CNN-5-512','-dev','cpu',
        '-nc','20','-ncl','100','-t','1','-pv','0','-jr','1.0','--seed','0','--target_client_id','0','-lr','.005',
        '-lbs','16','-ls','5','-gr','1','-eg','1','-niid','1','-pt','pat','-cpc','20','-is_regular','1','-regular_lamda','.001',
        '--target_head_mode','shared','--target_local_schedule','joint','--target_prox_mu','0','--local_grad_clip','0',
        '--target_proj_mode',job['mode'],'--projection_self_weight','.125','--projection_softmax_tau','.2',
        '-clip_cpu_threads','2','-exp_name',label,'-sfn',str(run/'c'),'--run_log_dir',str(run/'l'),
        '--h5_result_root',str(run/'h'),'--final-model-root',str(run/'f')]
    if job['mode']==ph.MODE:
        sys.argv+=['--projection_history_strength',str(job['strength']),
                   '--projection_history_diagnostics',str(job.get('diagnostics',1))]
    native_copy=shutil.copy2
    def compact(source,destination,*args,**kwargs):
        if str(source).endswith('.pt') and Path(source).is_file() and not Path(destination).exists():
            os.link(source,destination);return destination
        return native_copy(source,destination,*args,**kwargs)
    with ExitStack() as stack:
        stack.enter_context(patch.object(FedTargetProj,'__init__',initialize))
        stack.enter_context(patch.object(cb,'DataLoader',TraceLoader))
        stack.enter_context(patch.object(Hyper_CNN_512,'forward',forward))
        stack.enter_context(patch.object(shutil,'copy2',compact))
        os.chdir(out/'system');runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];rng=rng_digest();folder=Path(server.report_dir());summary=server._local_accuracy_summary()
    rows=json.loads((folder/'target_proj_metrics.json').read_text())['history']
    values=[r['target_post_local_acc'] for r in rows]
    assert summary['final_target_local_acc']==values[-1] and summary['best_target_local_acc']==max(values)
    assert summary['best_target_local_round']==values.index(max(values))+1
    assert summary['last10_target_local_acc']==sum(values)/2 and summary['last20_target_local_acc']==sum(values)/2
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert hf['target_projection/target_post_local_acc'][:].tolist()==values
        if ph.enabled(server.args):
            data=json.loads((folder/'projection_history.json').read_text())
            assert data['summary']==summary
            group=hf['target_projection/projection_history']
            assert [json.loads(r) for r in group['rounds_json'][:]]==data['history']
            assert json.loads(group.attrs['policy_json'])==data['policy']
            assert hf['rs_test_acc'][:].tolist()==values
            with (folder/'projection_history_clients.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
            assert len(csv_rows)==40
            for a,b in zip(csv_rows,[c for r in data['history'] for c in r['clients']]):
                assert float(a['aggregation_weight'])==b['aggregation_weight']
                assert a['fallback_reason']==b['fallback_reason']
    manifest=json.loads((Path(server.final_model_dir())/'manifest.json').read_text())
    if ph.enabled(server.args):
        assert manifest['projection_history']['primary_summary']==summary
        cache_path=Path(server.final_model_dir())/'projection_history_cache.pt'
        if job['strength']:
            cache=torch.load(cache_path,weights_only=False)
            assert cache['source_loop_rounds']==[0,1] and not cache['resume_supported']
            assert cache['parameter_schema']==ph.schema(dict(load_item('Server','model',server.save_folder_name).named_parameters()))
        else:
            assert server.projection_history_cache is None and not cache_path.exists()
    with preserved_head_rng(),torch.no_grad():
        for role in ('Server',*[c.role for c in server.clients]):
            restored=torch.load(Path(server.final_model_dir())/f'{role}_model.pt',map_location='cpu',weights_only=False)
            original=load_item(role,'model',server.save_folder_name)
            assert tensor_fingerprint(restored.state_dict())==tensor_fingerprint(original.state_dict())
            assert torch.equal(restored.eval()(torch.zeros(1,3,32,32)),original.eval()(torch.zeros(1,3,32,32)))
    log_root=Path(server.args.training_log_dir)
    with zipfile.ZipFile(log_root.with_name(log_root.name+'.zip')) as archive:
        assert not any(n.endswith(('.pt','.npz')) for n in archive.namelist())
    assert len(batches)==200
    (run/'verification.json').write_text(json.dumps(dict(status='PASS',events=events,batches=batches,final_rng=rng,
        summary=summary,train_tasks=40,SGD_steps=200,models_reloaded=21,kernel_max_error=0.),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--worker');parser.add_argument('--keep-models',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker)
    out.mkdir(parents=True,exist_ok=False);(out/'system').mkdir()
    import numpy as np
    generator=np.random.default_rng(101)
    for split in ('train','test'):
        directory=out/'dataset/Cifar100/pat_20'/split;directory.mkdir(parents=True)
        for cid in range(20):
            count=2+cid%2 if split=='train' else 1+cid%4
            np.savez(directory/f'{cid}.npz',data=dict(x=generator.normal(size=(count,3,32,32)).astype('float32'),y=(np.arange(count)+cid*3)%100))
    jobs=dict(old=dict(mode='projection_softmax'),zero=dict(mode='projection_history_softmax',strength=0.),
        s1=dict(mode='projection_history_softmax',strength=1.),s1off=dict(mode='projection_history_softmax',strength=1.,diagnostics=0),
        s10=dict(mode='projection_history_softmax',strength=10.))
    (out/'jobs.json').write_text(json.dumps(jobs,indent=2))
    for label in jobs:
        print('RUN',label,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--worker',label],cwd=ROOT/'system',
                env=dict(os.environ,PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Actual-entry acceptance failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('c','f'):
                owned=(out/label/name).resolve()
                if not owned.is_relative_to(out) or owned==out:raise RuntimeError('Unsafe synthetic cleanup path')
                shutil.rmtree(owned)
        print('PASS',label,flush=True)
    reports={name:json.loads((out/name/'verification.json').read_text()) for name in jobs}
    for key in ('events','batches','final_rng','summary'):
        assert reports['old'][key]==reports['zero'][key],('strength-zero',key)
        assert reports['s1'][key]==reports['s1off'][key],('diagnostics',key)
    for report in reports.values():
        assert report['events'][:22]==reports['old']['events'][:22]  # initialization, 20 uploads, first aggregation
    receipt=dict(status='PASS',synthetic_only=True,clients=20,rounds=2,original_width=16,local_epochs=5,
        strict_strength_zero=True,strict_first_round=True,strict_diagnostics_isolation=True,
        full_W_includes_head=True,kernel_max_parameter_error=0.,cases={k:{name:r[name] for name in
            ('train_tasks','SGD_steps','models_reloaded')} for k,r in reports.items()})
    (out/'acceptance_report.json').write_text(json.dumps(receipt,indent=2));print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
