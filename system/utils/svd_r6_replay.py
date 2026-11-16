"""Independent five-arm replay. No federated runtime imports this module.

Reuse original decomposers, CE/F trainer, private R6 epoch loader, evaluation,
and ordered full-matrix Avg kernel. Only offline copies are optimized.
"""
from contextlib import ExitStack
import copy
import math
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from common.client_data import audit_image_clients,read_image_client
from common.experiment import sha256,write_json,environment,code_version
from utils import fc1_tail_diagnostic as old,multi_group_svd as svd,multi_svd_spectral as spectral,single_avg_proto as proto
from utils.cnn512_reporting import model_digest
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_rank_partition import protected_rng,factors

LAYERS=('conv2','fc1','fc2')
TIERS=[.9,.37,.35,.25,.15]
BRANCHES=('H_raw','H_repair','Hplus_raw','Hplus_repair','L_raw')
BACKEND=('deterministic','cudnn_deterministic','cudnn_benchmark','matmul_tf32','cudnn_tf32')
FILE_FIELDS=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')


def resolve_source(cli,label):
    explicit=getattr(cli,label+'_server_file')
    if explicit:path=Path(explicit).expanduser().resolve()
    else:
        root=Path(getattr(cli,label+'_run')).expanduser().resolve()
        candidates=([root/'Server_model_0.pt'] if (root/'Server_model_0.pt').is_file() else
            sorted((root/'final_models').rglob('Server_model_0.pt')))
        if len(candidates)!=1:raise FileNotFoundError(f'{label}: expected exactly one final server, found {len(candidates)}; use --{label}-server-file')
        path=candidates[0]
    for file in (path,path.parent/'manifest.json'):
        if not file.is_file():raise FileNotFoundError(f'{label}: required same-run source missing: {file}')
    return path


def client_rates(a):
    rates=a.multi_svd_effective_ratios;assignment=a.multi_svd_effective_assignment
    if (not isinstance(rates,list) or not rates or len(set(rates))!=len(rates)
        or any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<1 for v in rates)
        or not isinstance(assignment,list) or len(assignment)!=a.num_clients
        or any(type(i) is not int or not 0<=i<len(rates) for i in assignment)):
        raise ValueError('Invalid actual source capacity ratios/assignment')
    return [rates[i] for i in assignment]


def validate_manifest(m,label,source):
    if type(source) is not int or source!=5:raise ValueError('This R6/R7 replay requires completed source round 5')
    a=SimpleNamespace(**copy.deepcopy(m['effective']))
    if any(type(getattr(a,k)) is not int or getattr(a,k)<1 for k in
        ('num_clients','num_server_models','num_classes','local_epochs','batch_size','test_batch_size')):
        raise ValueError(f'{label}: source counts/epochs/batch sizes must be positive integers')
    if type(a.global_rounds) is not int or a.global_rounds<0:raise ValueError('Invalid source round budget')
    if (m.get('algorithm')!='FedMultiGroupProj' or m.get('actual_train_count')!=source
        or m.get('server_model_count')!=1 or m.get('client_count')!=20 or m.get('aggregation_mode')!='group_avg_only'
        or a.global_rounds+1<source
        or a.num_clients!=20 or a.num_server_models!=1 or a.join_ratio!=1 or not a.multi_svd or not a.multi_avg_proto_ablation
        or a.model_family!='Decom_CNN-5-512' or a.dataset!='Cifar100' or a.num_classes!=100 or a.partition!='pat'
        or a.multi_proto_mode!='none' or a.multi_proto_lambda!=0 or a.multi_svd_aggregation!='full_avg'
        or a.multi_svd_init!='low_rank_recovered' or a.multi_svd_init_rank_rate!=.9
        or a.target_head_mode!='shared' or a.target_local_schedule!='joint'):
        raise ValueError(f'{label}: need matching K1/N20 full_avg SVD pat20 source, .90 recovered initialization, shared/joint, no prototypes')
    if (a.local_epochs!=5 or a.batch_size!=16 or a.local_learning_rate!=.005 or a.is_regular!=1
        or a.regular_lamda!=.001 or a.local_grad_clip!=10 or a.test_batch_size<1
        or any(getattr(a,k,False) for k in ('target_prox_mu','target_z2_lambda','target_head_l2_lambda','multi_warmup_rounds',
            'multi_fc1_tail_scale_strength','few_shot','resume','learning_rate_decay','auto_break','meta_c0_split','multi_last_epoch_protocol'))
        or any(getattr(a,k,'none')!='none' for k in ('multi_kd_mode','target_c0_replay_mode','target_curriculum_mode'))
        or getattr(a,'multi_head_mode','shared')!='shared' or getattr(a,'multi_projection_scope','full')!='full'
        or getattr(a,'multi_helper_head_update','train')!='train' or getattr(a,'random_join_ratio',False)):
        raise ValueError(f'{label}: need original local5/batch16/lr.005 CE+F.001/clip10 without other objectives')
    expected=[.9]*20 if label=='l' else [TIERS[i%len(TIERS)] for i in range(20)]
    if client_rates(a)!=expected:raise ValueError(f'{label}: unexpected actual per-client capacity assignment')
    mode='all_tail' if label=='hplus' else 'none'
    if a.multi_svd_spectral_mode!=mode or (mode=='all_tail' and spectral.layer_names(a)!=list(LAYERS)):
        raise ValueError(f'{label}: source must use {mode} over the expected layers')
    if m.get('svd_spectral',{}).get('policy')!=spectral.policy(a):raise ValueError(f'{label}: spectral policy/effective configuration mismatch')
    expected_optimizer=dict(name='SGD',lr=.005,momentum=0.,weight_decay=0.,gradient_norm_cap=10.)
    if any(m.get('optimizer',{}).get(k)!=v for k,v in expected_optimizer.items()):raise ValueError(f'{label}: incompatible optimizer')
    if type(a.seed) is not int or not 0<=a.seed<2**32:raise ValueError('Invalid source seed')
    if any(type(m.get('environment',{}).get(k)) is not bool for k in BACKEND):raise ValueError('Source backend configuration missing')
    return a


def common_protocol(a):
    return {k:getattr(a,k) for k in ('seed','num_clients','num_classes','num_server_models','model_family','local_epochs',
        'batch_size','test_batch_size','local_learning_rate','is_regular','regular_lamda','local_grad_clip',
        'multi_svd_init','multi_svd_init_rank_rate','class_per_client')}


def load_sources(cli):
    import json
    from flcore.trainmodel.models import Hyper_CNN_512
    sources={};identity=None
    for label in ('h','hplus','l'):
        path=resolve_source(cli,label);m=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
        a=validate_manifest(m,label,cli.source_round);mother,p=svd.load_checkpoint(path,'cpu')
        if (p.get('role')!='post_aggregation_full_server' or p.get('completed_train_count')!=cli.source_round
            or p.get('model_id')!=0 or mother.ratio_LR!=1. or p.get('svd_aggregation',{}).get('method')!='full_avg'
            or any(not isinstance(getattr(mother,n),torch.nn.Linear if n!='conv2' else torch.nn.Conv2d) for n in LAYERS)):
            raise ValueError(f'{label}: must load complete post-Avg R{cli.source_round} SERVER, not a client or upload')
        for key,mkey in [('model_kwargs','model_kwargs'),('model_family','model_family'),('svd_initialization','svd_initialization'),
            ('capacity_protocol','svd_capacity_protocol'),('initial_parameter_sha256','initial_parameter_sha256')]:
            if p.get(key)!=m.get(mkey):raise ValueError(f'{label}: checkpoint/manifest {key} mismatch')
        if p['capacity_protocol']!=svd.capacity_protocol(a):raise ValueError(f'{label}: effective capacity protocol mismatch')
        declared=m['svd_spectral']['policy']
        if p.get('svd_spectral_policy')!=(declared if label=='hplus' else None):raise ValueError(f'{label}: checkpoint spectral policy mismatch')
        if p['model_kwargs']!=dict(in_features=3,num_classes=100,n_kernels=16,ratio_LR=1.,input_size=32):
            raise ValueError('Replay v1 supports original RGB32/CIFAR100 Hyper_CNN_512 widths only')
        for name,value in mother.state_dict().items():
            if value.dtype!=p['state_dict'][name].dtype or not torch.isfinite(value).all() or not torch.equal(value,p['state_dict'][name]):
                raise ValueError('Nonfinite/inconsistent checkpoint dtype or alias '+name)
        current=(p['initial_parameter_sha256'],p['model_kwargs'],common_protocol(a),{k:m['environment'][k] for k in BACKEND})
        if identity is not None and current!=identity:raise ValueError('H/H+/L initial model, common training protocol or backends differ')
        identity=current
        sources[label]=dict(path=path,manifest=m,payload=p,args=a,mother=mother)
    a=copy.deepcopy(sources['h']['args'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(a.seed)
        initial=Hyper_CNN_512(**dict(sources['h']['payload']['model_kwargs'],ratio_LR=.9));initial.recover_larger_model();initial.ratio_LR=1.
        if model_digest(initial)!=identity[0]:raise ValueError('Common S0 hash cannot be reproduced from source initialization')
    if cli.dataset!='Cifar100':raise ValueError('Need original Cifar100 data')
    a.data_root=cli.data_root;a.dataset=cli.dataset;a.data_subdir=cli.data_subdir;a.data_formats='npz_only';a.data_expected_partition='pat'
    data=audit_image_clients(a);files=[{k:r[k] for k in FILE_FIELDS} for r in data['files']]
    fingerprint=digest_json(dict(config_sha256=data['config_sha256'],files=files));samples=[next(r['count'] for r in data['files'] if r['client_id']==cid and r['split']=='train') for cid in range(a.num_clients)]
    for label,s in sources.items():
        source_data=s['manifest']['data_manifest'];original_files=[{k:r[k] for k in FILE_FIELDS} for r in source_data['files']]
        if data['config_sha256']!=source_data['config_sha256'] or files!=original_files:raise ValueError(f'{label}: actual file/decoded data fingerprints differ')
        if label=='hplus' and s['args'].multi_svd_avg_train_samples!=samples:raise ValueError('H+ correction masses differ from audited original train samples')
        s['provenance']=dict(server_file=str(s['path']),server_file_sha256=sha256(s['path']),server_state_sha256=model_digest(s['mother']),
            manifest_file=str(s['path'].parent/'manifest.json'),manifest_file_sha256=sha256(s['path'].parent/'manifest.json'),
            source_completed_aggregations=cli.source_round,initial_parameter_sha256=identity[0],data_fingerprint=fingerprint,
            source_effective=vars(s['args']),source_backend=identity[3],source_environment=s['manifest']['environment'],
            source_fixture_note=s['manifest'].get('synthetic_fixture'))
    a.device='cuda:0' if cli.device=='cuda' else 'cpu'
    if cli.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Need exactly the requested visible GPU, or --device cpu')
        torch.cuda.set_device(0)
    a.offline_loop_round=cli.source_round;a.initial_sha256=identity[0];a.data_fingerprint=fingerprint
    # Download intervention is already explicit below. Disable ONLY transient
    # checkpoint decoration and old observations; never change CE/F/SGD.
    a.multi_svd_spectral_mode='none';a.multi_svd_five_diagnostics=0;a.multi_rank_spectrum=0;a.multi_svd_diagnostics=0
    a.multi_svd_avg_train_samples=samples
    backend=identity[3];torch.use_deterministic_algorithms(backend['deterministic'])
    torch.backends.cudnn.deterministic=backend['cudnn_deterministic'];torch.backends.cudnn.benchmark=backend['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=backend['matmul_tf32'];torch.backends.cudnn.allow_tf32=backend['cudnn_tf32']
    return sources,a,data


@torch.no_grad()
def paired_download(mother,rate,args,cid,repair=True):
    """ONE original decomposition per layer; both arms use its identical U/V."""
    model=copy.deepcopy(mother);bounds=spectral.boundary_ranks(mother,args);raw_factors={};receipts=[]
    def transform(name,s,rank):
        out,record=spectral.all_tail_spectrum(s,bounds[name]['client_ranks'],args.multi_svd_avg_train_samples)
        receipts.append(dict(client_id=cid,layer=name,actual_rank=rank,raw_spectrum_sha256=tensor_fingerprint({'S':s}),
            repaired_spectrum_sha256=tensor_fingerprint({'S':out}),**record))
        return out
    def observer(name,layer,u,s,v):
        root=s[:layer.rank].sqrt();raw_factors[name]=(u[:,:layer.rank]@root.diag(),root.diag()@v[:layer.rank,:])
    with protected_rng([mother]):
        if repair:
            transforms={n:(lambda s,r,n=n:transform(n,s,r)) for n in LAYERS}
            observers={n:(lambda layer,u,s,v,n=n:observer(n,layer,u,s,v)) for n in LAYERS}
            model.decom_larger_model(rate,singular_value_transforms=transforms,svd_observers=observers)
        else:model.decom_larger_model(rate)
        model.ratio_LR=rate;model.to(next(mother.parameters()).device)
        raw=copy.deepcopy(model) if repair else model
        for name in LAYERS if repair else ():
            f=factors(getattr(raw,name));u,v=raw_factors[name];f['U'].copy_(u);f['V'].copy_(v)
            k=bounds[name]['k'];changed=factors(getattr(model,name))
            if not torch.equal(f['U'][:,:k],changed['U'][:,:k]) or not torch.equal(f['V'][:k],changed['V'][:k]):
                raise RuntimeError('Same-mother download changed minimum prefix')
        if repair:
            # Compare by canonical names, excluding only factor tensors (bias stays shared).
            for n,p in model.named_parameters():
                if n.endswith(('weight_u','weight_v','conv_u','conv_v')):continue
                if not torch.equal(p,dict(raw.named_parameters())[n]):raise RuntimeError('Raw/repair nonfactor state differs '+n)
    return raw,(model if repair else None),receipts


class ReplayObserver:
    def __init__(self,client,data,test,collect=True):
        self.client=client;self.model=client.model_set[0];self.data=data;self.test=test;self.collect=collect
        self.endpoints=[];self.epochs=[];self.steps=0;self.epoch=1;self.acc=None;self.endpoint_seconds=0.
        self.stack=ExitStack();self.last_clip=None
    def endpoint(self,epoch):
        tick=time.perf_counter()
        for split,data in [('train',self.data),('test',self.test)]:
            row,_=old.evaluate(self.model,data,self.client.args.test_batch_size,self.client.device)
            self.endpoints.append(dict(epoch=epoch,split=split,**row))
        self.endpoint_seconds+=time.perf_counter()-tick
    def end_epoch(self,epoch):
        if self.collect:
            if str(self.client.device).startswith('cuda'):torch.cuda.synchronize(self.client.device)
            a=self.acc
            if a is None or not a['visits'] or not a['steps']:raise RuntimeError('Incomplete per-epoch SGD observations')
            self.epochs.append(dict(epoch=epoch,train_mean_CE=a['ce']/a['visits'],
                train_mean_weighted_F=a['f']/a['visits'],train_mean_raw_F=a['f']/a['visits']/self.client.args.regular_lamda,
                train_mean_total=a['total']/a['visits'],train_step_mean_weighted_F=a['f_steps']/a['steps'],
                train_step_mean_total=a['total_steps']/a['steps'],train_sample_visits=a['visits'],optimizer_steps=a['steps'],
                pre_clip_grad_norm_mean=a['norm']/a['steps'],clip_trigger_fraction=a['clips']/a['steps'],
                clip_coefficient_mean=a['coef']/a['steps'],SGD_epoch_seconds=time.perf_counter()-a['started'],
                loss_mean_unit='sample visits; additional step means explicitly named',gradient_mean_unit='SGD steps'))
            self.endpoint(epoch)
        self.epoch=epoch+1;self.acc=None
    def __enter__(self):
        from flcore.clients.clientTargetProj import clientTargetProj
        original_objective=clientTargetProj._objective;original_clip=torch.nn.utils.clip_grad_norm_
        opt=self.client.optimizer_set[0];original_step=opt.step
        def objective(client,model,x,y):
            tick=time.perf_counter()
            ce,f=original_objective(client,model,x,y)
            if client is self.client and self.collect:
                if self.acc is None:self.acc=dict(ce=0.,f=0.,total=0.,f_steps=0.,total_steps=0.,visits=0,steps=0,norm=0.,coef=0.,clips=0,started=tick)
                a=self.acc;fv=float(f.detach());cv=float(ce.detach());total=float(ce.detach()+f.detach())
                a['ce']+=cv*len(y);a['f']+=fv*len(y);a['total']+=total*len(y)
                a['f_steps']+=fv;a['total_steps']+=total;a['visits']+=len(y)
            return ce,f
        def clip(parameters,max_norm,*a,**kw):
            value=original_clip(parameters,max_norm,*a,**kw)
            if self.collect:
                self.last_clip=(float(value),float((max_norm/(value+1e-6)).clamp(max=1.)))
            return value
        def step(*a,**kw):
            result=original_step(*a,**kw);self.steps+=1
            if self.collect:
                if self.last_clip is None:raise RuntimeError('Original gradient clipping was not observed')
                n,c=self.last_clip;self.acc['steps']+=1;self.acc['norm']+=n;self.acc['coef']+=c;self.acc['clips']+=int(c<1)
                self.last_clip=None
            return result
        try:
            if self.collect:
                self.stack.enter_context(old.temporary_attribute(clientTargetProj,'_objective',objective))
                self.stack.enter_context(old.temporary_attribute(torch.nn.utils,'clip_grad_norm_',clip))
            # Restore method ownership exactly, including on exceptions.
            owned='step' in opt.__dict__;self.stack.callback(lambda:setattr(opt,'step',original_step) if owned else opt.__dict__.pop('step',None))
            opt.step=step
            return self
        except BaseException:self.stack.close();raise
    def __exit__(self,*exc):self.stack.close()


def train_branch(model,mother,args,cid,data,test,folder,*,collect=True,trainer=None):
    if model.ratio_LR!=client_rates(args)[cid]:raise ValueError('Offline optimizer/client capacity does not match actual downloaded model')
    client=old.local_client(model,mother,args,cid,None,None,data,folder)
    observer=ReplayObserver(client,data,test,collect);loader=old.EpochLoader(data,args,cid,args.offline_loop_round,observer)
    client.load_train_data=lambda:loader
    tick=time.perf_counter()
    try:
        if collect:observer.endpoint(0)
        initial_rng=old.rng_fingerprint(old.rng_snapshot(model))
        with observer:(trainer or proto.train)(client)
    finally:
        loader.observer=None  # Break client -> loader -> observer -> client; no GPU-state history.
    steps=args.local_epochs*math.ceil(len(data)/args.batch_size)
    if observer.steps!=steps or len(loader.batches)!=steps or client.train_stats[0]['processed_train_samples']!=len(data)*args.local_epochs:
        raise RuntimeError('R6 local budget mismatch')
    if client.avg_proto_local is not None or client.full_uploads[0] is None:raise RuntimeError('Need no prototypes and one real recovered upload')
    return dict(endpoints=observer.endpoints,epochs=observer.epochs,training=client.train_stats[0],
        batches=loader.batches,batch_sha256=digest_json(loader.batches),initial_rng_sha256=initial_rng,
        final_rng_sha256=old.rng_fingerprint(old.rng_snapshot(model)),model_sha256=tensor_fingerprint(model.state_dict()),
        wall_seconds=time.perf_counter()-tick,endpoint_seconds=observer.endpoint_seconds),client


def aggregate_once(mother,uploads,samples,order):
    from utils.group_projection import aggregate_group_updates
    if set(uploads)!=set(samples) or set(order)!=set(samples) or len(set(order))!=len(order):raise ValueError('Require complete unique original uploads')
    if any(type(n) is not int or n<=0 for n in samples.values()):raise ValueError('Need original train counts')
    weights={i:n/sum(samples.values()) for i,n in samples.items()};params=dict(mother.named_parameters())
    def factory():
        for cid in order:
            value=torch.load(uploads[cid],map_location='cpu',weights_only=False) if isinstance(uploads[cid],(str,Path)) else uploads[cid]
            yield cid,weights[cid],{n:p.to(next(mother.parameters()).device) for n,p in value.items()}
    with torch.no_grad(),protected_rng([mother]):
        result,metrics,rows,_=aggregate_group_updates(params,factory,weights,list(range(len(samples))),'group_avg_only')
        combined=copy.deepcopy(mother)
        for n,p in combined.named_parameters():p.copy_(result[n])
    return combined,dict(order=order,weights=rows,metrics=metrics,kernel='original aggregate_group_updates/group_avg_only; full matrix Avg',
        extra_local_SGD_steps=0)


def summarize_endpoints(rows,n,focus,target):
    groups={}
    for row in rows:groups.setdefault((row['branch'],row['phase'],row['epoch'],row['split']),[]).append(row)
    output=[]
    for (branch,phase,epoch,split),values in sorted(groups.items()):
        if len(values)!=n or {r['client_id'] for r in values}!=set(range(n)):raise ValueError('Incomplete/duplicate all-client endpoint')
        for scope,ids in [('all',list(range(n))),('target',[target]),('fixed_large',focus)]:
            chosen=[r for r in values if r['client_id'] in ids];samples=sum(r['samples'] for r in chosen);correct=sum(r['correct'] for r in chosen)
            if len(chosen)!=len(ids) or not samples:raise ValueError('Missing fixed members')
            output.append(dict(branch=branch,phase=phase,epoch=epoch,split=split,evaluation_scope=scope,member_ids=ids,
                correct=correct,samples=samples,accuracy=correct/samples,ce=sum(r['ce']*r['samples'] for r in chosen)/samples,
                weighted_by='split sample count',accuracy_unit='fraction',CE_unit='nats/sample'))
    return output


def summarize_training(rows,n,focus,target):
    groups={};output=[]
    for row in rows:groups.setdefault((row['branch'],row['epoch']),[]).append(row)
    for (branch,epoch),values in sorted(groups.items()):
        if {r['client_id'] for r in values}!=set(range(n)) or len(values)!=n:raise ValueError('Incomplete epoch budget')
        for scope,ids in [('all',list(range(n))),('target',[target]),('fixed_large',focus)]:
            chosen=[r for r in values if r['client_id'] in ids];visits=sum(r['train_sample_visits'] for r in chosen);steps=sum(r['optimizer_steps'] for r in chosen)
            output.append(dict(branch=branch,epoch=epoch,evaluation_scope=scope,member_ids=ids,train_sample_visits=visits,optimizer_steps=steps,
                **{key:sum(r[key]*r['train_sample_visits'] for r in chosen)/visits for key in ('train_mean_CE','train_mean_weighted_F','train_mean_total')},
                **{key:sum(r[key]*r['optimizer_steps'] for r in chosen)/steps for key in ('clip_trigger_fraction','clip_coefficient_mean','pre_clip_grad_norm_mean')},
                loss_weighting='train sample visits',gradient_weighting='SGD steps'))
    return output


def comparisons(metrics,epochs):
    lookup={(r['branch'],r['phase'],r['epoch'],r['evaluation_scope'],r['split']):r for r in metrics}
    if len(lookup)!=len(metrics):raise ValueError('Duplicate comparison endpoints')
    result=[]
    for high,low,phase,scopes,points in [
        ('H_repair','H_raw','R6_local',('all','target','fixed_large'),range(epochs+1)),
        ('Hplus_repair','Hplus_raw','R6_local',('all','target','fixed_large'),range(epochs+1)),
        ('Hplus_repair','L_raw','R6_local',('fixed_large',),range(epochs+1)),
        ('H_repair','H_raw','R7_pre_ordinary',('all','target','fixed_large'),[0]),
        ('Hplus_repair','Hplus_raw','R7_pre_ordinary',('all','target','fixed_large'),[0])]:
        for epoch in points:
            for scope in scopes:
                for split in ('train','test'):
                    a=lookup[high,phase,epoch,scope,split];b=lookup[low,phase,epoch,scope,split]
                    if a['member_ids']!=b['member_ids'] or a['samples']!=b['samples']:raise ValueError('Unpaired comparison members/data')
                    delta=a['accuracy']-b['accuracy'];ce=a['ce']-b['ce']
                    row=dict(contrast=high+'_minus_'+low,phase=phase,epoch=epoch,evaluation_scope=scope,member_ids=a['member_ids'],
                        split=split,accuracy_difference=delta,CE_difference=ce,accuracy_unit='fraction')
                    if phase=='R6_local':
                        p=lookup[high,phase,0,scope,split];q=lookup[low,phase,0,scope,split]
                        row.update(accuracy_gap_change_since_epoch0=delta-(p['accuracy']-q['accuracy']),CE_gap_change_since_epoch0=ce-(p['ce']-q['ce']))
                    result.append(row)
    return result


def fitting_trends(metrics,epochs):
    lookup={(r['branch'],r['phase'],r['epoch'],r['evaluation_scope'],r['split']):r for r in metrics};rows=[]
    for branch in BRANCHES:
        for scope in ('all','target','fixed_large'):
            for epoch in range(1,epochs+1):
                train=lookup[branch,'R6_local',epoch,scope,'train'];test=lookup[branch,'R6_local',epoch,scope,'test']
                p=lookup[branch,'R6_local',epoch-1,scope,'train'];q=lookup[branch,'R6_local',epoch-1,scope,'test']
                rows.append(dict(branch=branch,evaluation_scope=scope,epoch=epoch,previous_epoch=epoch-1,
                    train_CE_change=train['ce']-p['ce'],test_CE_change=test['ce']-q['ce'],
                    train_accuracy_change=train['accuracy']-p['accuracy'],test_accuracy_change=test['accuracy']-q['accuracy'],
                    train_CE_down_test_CE_up=train['ce']<p['ce'] and test['ce']>q['ce'],
                    use='post-hoc adjacent fixed-epoch observations only; never stopping or selecting a model'))
    return rows


def render_curves(root,epochs):
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'plot_svd_r6_replay.py'
    with (root/'render.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u',str(script),'--output-dir',str(root),'--epochs',str(epochs)],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Curve rendering exit={result.returncode}; tables retained, see {root}/render.log')
    import json
    return json.loads((root/'curves.json').read_text(encoding='utf-8'))


def preflight_curves():
    import subprocess
    import sys
    script=Path(__file__).resolve().parents[1]/'plot_svd_r6_replay.py'
    result=subprocess.run([sys.executable,'-u',str(script),'--check'],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(f'Renderer preflight exit={result.returncode}; no training started. Activate the existing Python environment (including its DLL paths), without changing packages. '+result.stderr[-2000:])



def summary_text(pairs,trends):
    lines=['# Offline R6 replay and R7 ordinary-download retention','Accuracy gaps in percentage points; CE gaps in nats/sample.',
        'Fixed largest clients have identical .90 capacity in all five branches. Test data are reporting only; every branch trains the fixed 5 epochs.','',
        '| contrast | phase | epoch | test accuracy gap (pp) | test CE gap |','| --- | --- | ---: | ---: | ---: |']
    for r in pairs:
        if r['evaluation_scope']=='fixed_large' and r['split']=='test':
            lines.append(f"| {r['contrast']} | {r['phase']} | {r['epoch']} | {100*r['accuracy_difference']:.6f} | {r['CE_difference']:.8g} |")
    lines+=['','First adjacent-epoch train-CE-down / test-CE-up observation (fixed clients; descriptive, not a selection rule):']
    for branch in BRANCHES:
        matching=[r['epoch'] for r in trends if r['branch']==branch and r['evaluation_scope']=='fixed_large' and r['train_CE_down_test_CE_up']]
        lines.append(f'- {branch}: '+(str(min(matching)) if matching else 'not observed'))
    lines+=['','R7 uses ordinary SVD for all four heterogeneous arms and no R7 SGD; it does not apply another repair.',
        'Repair-minus-raw starts can differ at epoch 0. Gap changes since epoch 0 are reported separately. No best-epoch or test-driven branch selection.']
    return '\n'.join(lines)+'\n'


def run(cli):
    if cli.threads<1:raise ValueError('threads must be positive')
    import importlib.util
    if importlib.util.find_spec('matplotlib') is None:raise RuntimeError('matplotlib is required for diagnostic curves; environment is left unchanged')
    output=Path(cli.output_dir).expanduser().resolve()
    if output.exists():raise FileExistsError(f'Output exists: {output}')
    data_root=(Path(cli.data_root)/cli.dataset/cli.data_subdir).expanduser().resolve()
    paths=[resolve_source(cli,label) for label in ('h','hplus','l')]
    for protected in [data_root]+[p.parent for p in paths]:
        if output==protected or protected in output.parents or output in protected.parents:raise ValueError('Output must be independent of source models and data')
    scratch=Path(cli.temp_root).expanduser().resolve() if cli.temp_root else output
    if cli.temp_root and any(scratch==p or p in scratch.parents or scratch in p.parents for p in [data_root]+[path.parent for path in paths]):
        raise ValueError('Scratch must not overlap source/data')
    preflight_curves()
    torch.set_num_threads(cli.threads)
    sources,args,data=load_sources(cli)
    try:focus=sorted(int(v.strip()) for v in cli.focus_client_ids.split(','))
    except ValueError:raise ValueError('focus-client-ids must be integers')
    if not focus or len(set(focus))!=len(focus) or any(not 0<=i<args.num_clients for i in focus):raise ValueError('Invalid fixed cohort')
    if not 0<=cli.target_client_id<args.num_clients:raise ValueError('Target client out of range')
    if any(client_rates(sources['h']['args'])[i]!=.9 for i in focus):raise ValueError('Fixed cohort must have the same .90 capacity in all branches')
    samples={r['client_id']:r['count'] for r in data['files'] if r['split']=='train'}
    # Original full-participation selector uses exactly this NumPy choice path.
    # A common isolated stream gives the R6 order; no source process RNG assumed.
    selection=np.random.RandomState(args.seed)
    for _ in range(cli.source_round+1):order=selection.choice(np.arange(args.num_clients),args.num_clients,replace=False).tolist()
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',role=f'offline_R{cli.source_round+1}_replay_then_R{cli.source_round+2}_ordinary_download',
        requested=vars(cli),sources={k:s['provenance'] for k,s in sources.items()},effective_common=common_protocol(args),
        environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),data_fingerprint=args.data_fingerprint,
        source_initial_parameter_sha256=args.initial_sha256,branches=list(BRANCHES),target_client_id=cli.target_client_id,focus_client_ids=focus,
        epoch_observations=list(range(args.local_epochs+1)),prototype_enabled=False,
        training='ORIGINAL single_avg_proto.train CE+F; NEW SGD per client/branch; no prototype/replay/KD/extra epochs',
        training_rng='independent stable per-client offline replay state, restored for all five arms; not source-process RNG continuation',
        batch_rng='original single_avg_proto.loader_seed(seed,client,source_round), i.e. R6 loop=5 by default',
        aggregation_order=order,aggregation_order_protocol='common isolated NumPy RandomState(seed); source_round+1 original all-client choice calls; not a recovered source RNG',
        artifacts='tables and fixed-epoch curves only; transient ordinary trainer checkpoints/uploads cleaned',
        R7='four heterogeneous arms each one original full-matrix Avg, then ordinary per-client SVD and inference only; L has no R7 aggregation',
        units='accuracy fraction in tables / percent on curves; CE nats per sample; loss-process and endpoint CE separate')
    write_json(output/'manifest.json',manifest);write_json(output/'data_manifest.json',data)
    input_hashes={p:sha256(p) for path in paths for p in (path,path.parent/'manifest.json')}
    input_hashes.update({Path(r['path']):r['sha256'] for r in data['files']})
    input_hashes[Path(data['config_path'])]=data['config_sha256']
    endpoints=[];epochs=[];training=[];batches=[];downloads=[];aggregations=[];costs=[]
    if cli.temp_root:
        scratch.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='offline_replay_',dir=scratch) as temp:
            temp=Path(temp);uploads={name:{} for name in BRANCHES if name!='L_raw'}
            for s in sources.values():s['mother']=s['mother'].to(args.device)
            configs={}
            for branch in BRANCHES:
                origin='hplus' if branch.startswith('Hplus') else 'h' if branch.startswith('H_') else 'l'
                a=copy.deepcopy(args);source_args=sources[origin]['args']
                a.multi_svd_effective_ratios=source_args.multi_svd_effective_ratios;a.multi_svd_effective_assignment=source_args.multi_svd_effective_assignment
                configs[branch]=a
            for cid in range(args.num_clients):
                train_data=read_image_client(args,cid,True);test_data=read_image_client(args,cid,False)
                models={}
                print(f'\n========== offline R{cli.source_round+1}, client={cid}, five paired tasks ========== ',flush=True)
                for origin in ('h','hplus','l'):
                    prefix={'h':'H','hplus':'Hplus','l':'L'}[origin];a=configs[prefix+'_raw']
                    rate=client_rates(a)[cid];tick=time.perf_counter()
                    raw,repair,rows=paired_download(sources[origin]['mother'],rate,a,cid,origin!='l')
                    name={'h':'H','hplus':'Hplus','l':'L'}[origin]
                    models[name+'_raw']=raw
                    if repair is not None:models[name+'_repair']=repair
                    for row in rows:
                        detail=row.pop('rank_corrections')
                        downloads.append(dict(source=origin,**row,max_theoretical_gain=max(r['theoretical_gain'] for r in detail),
                            max_actual_gain=max((r['actual_gain'] for r in detail if r['actual_gain'] is not None),default=1.)))
                    costs.append(dict(kind='paired_R6_download' if origin!='l' else 'ordinary_R6_download',source=origin,client_id=cid,seconds=time.perf_counter()-tick,original_SVD_calls=3))
                seed=int(digest_json(dict(purpose='offline_R6_paired_training',seed=args.seed,source_round=cli.source_round,client_id=cid))[:16],16)%(2**63)
                random.seed(seed);np.random.seed(seed%(2**32));torch.random.default_generator.manual_seed(seed)
                if args.device.startswith('cuda'):torch.cuda.default_generators[0].manual_seed(seed)
                base_rng=old.rng_snapshot(models['H_raw']);canonical=None;final_rng=None
                for branch in BRANCHES:
                    origin='hplus' if branch.startswith('Hplus') else 'h' if branch.startswith('H_') else 'l'
                    model=models.pop(branch);a=configs[branch];folder=temp/branch;folder.mkdir(exist_ok=True)
                    initial_model_sha256=tensor_fingerprint(model.state_dict())
                    old.restore_rng(base_rng);result,client=train_branch(model,sources[origin]['mother'],a,cid,train_data,test_data,folder)
                    if canonical is None:canonical=result['batches'];final_rng=result['final_rng_sha256']
                    if result['batches']!=canonical or result['initial_rng_sha256']!=old.rng_fingerprint(base_rng) or result['final_rng_sha256']!=final_rng:
                        raise RuntimeError('Five-arm actual batch order or training RNG differs')
                    identity=dict(branch=branch,client_id=cid,rank_rate=client.rank_rate,actual_layer_ranks={n:getattr(model,n).rank for n in LAYERS})
                    endpoints.extend(dict(**identity,phase='R6_local',replay_train_round=cli.source_round+1,**row) for row in result['endpoints'])
                    epochs.extend(dict(**identity,**row) for row in result['epochs'])
                    training.append(dict(result['training'],**identity,initial_model_sha256=initial_model_sha256,
                        final_model_sha256=result['model_sha256'],restored_upload_sha256=tensor_fingerprint(client.full_uploads[0])))
                    batches.append(dict(branch=branch,client_id=cid,loader_seed=proto.loader_seed(a,cid,cli.source_round),
                        paired_training_seed=seed,
                        actual_steps=len(result['batches']),batch_sha256=result['batch_sha256'],initial_rng_sha256=result['initial_rng_sha256'],
                        final_rng_sha256=result['final_rng_sha256'],all_five_paired=True,
                        epoch_batch_sha256={str(e):digest_json([r for r in result['batches'] if r['epoch']==e]) for e in range(1,a.local_epochs+1)}))
                    if branch in uploads:
                        file=folder/f'upload_{cid}.pt';torch.save(client.full_uploads[0],file);uploads[branch][cid]=file
                    transient=folder/f'Client_{cid}_model_0.pt'
                    transient.unlink();costs.append(dict(kind='R6_training_with_endpoints',branch=branch,client_id=cid,
                        seconds=result['wall_seconds'],endpoint_seconds=result['endpoint_seconds'],SGD_steps=result['training']['train_batches']))
                    del client,model,result
                del models
            for branch,files in uploads.items():
                origin='hplus' if branch.startswith('Hplus') else 'h';tick=time.perf_counter()
                combined,receipt=aggregate_once(sources[origin]['mother'],files,samples,order)
                aggregations.append(dict(branch=branch,server_state_sha256=model_digest(combined),seconds=time.perf_counter()-tick,**receipt))
                for cid in range(args.num_clients):
                    a=configs[branch];rate=client_rates(a)[cid];tick=time.perf_counter()
                    local,_,_=paired_download(combined,rate,a,cid,False)  # ALWAYS ordinary SVD; no R7 SGD.
                    for split,data_rows in [('train',read_image_client(args,cid,True)),('test',read_image_client(args,cid,False))]:
                        row,_=old.evaluate(local,data_rows,args.test_batch_size,args.device)
                        endpoints.append(dict(branch=branch,client_id=cid,rank_rate=rate,actual_layer_ranks={n:getattr(local,n).rank for n in LAYERS},
                            phase='R7_pre_ordinary',replay_train_round=cli.source_round+2,epoch=0,split=split,**row))
                    costs.append(dict(kind='ordinary_R7_download_and_inference',branch=branch,client_id=cid,seconds=time.perf_counter()-tick,
                        original_SVD_calls=3,SGD_steps=0));del local
                del combined
        metrics=summarize_endpoints(endpoints,args.num_clients,focus,cli.target_client_id)
        epoch_metrics=summarize_training(epochs,args.num_clients,focus,cli.target_client_id)
        pairs=comparisons(metrics,args.local_epochs);trends=fitting_trends(metrics,args.local_epochs)
        tables=dict(endpoints=endpoints,cohort_endpoints=metrics,epoch_training=epochs,cohort_epoch_training=epoch_metrics,
            local_training=training,batch_fingerprints=batches,paired_downloads=downloads,aggregations=aggregations,
            costs=costs,comparisons=pairs,fitting_trends=trends)
        for name,rows in tables.items():write_json(output/(name+'.json'),rows);old.write_csv(output/(name+'.csv'),rows)
        curves=render_curves(output,args.local_epochs)
        (output/'summary.md').write_text(summary_text(pairs,trends),encoding='utf-8')
        if any(sha256(file)!=value for file,value in input_hashes.items()):raise RuntimeError('Source artifact changed during replay')
        for source in sources.values():
            if model_digest(source['mother'])!=source['provenance']['server_state_sha256']:raise RuntimeError('Original mother modified by replay')
        manifest.update(status='complete',actual_training_tasks=len(training),actual_SGD_steps=sum(r['train_batches'] for r in training),
            actual_SGD_sample_visits=sum(r['processed_train_samples'] for r in training),all_five_batches_and_RNG_paired=True,
            original_R6_download_SVD_calls=3*3*args.num_clients,original_R7_download_SVD_calls=4*3*args.num_clients,
            R6_aggregations=4,R7_SGD_steps=0,source_artifacts_unchanged=True,transient_artifacts_cleaned=True,
            model_files_exported=0,curves=curves,comparison_rows=len(pairs))
        write_json(output/'manifest.json',manifest);print('Offline replay tables/curves:',output,flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(output/'manifest.json',manifest);raise
