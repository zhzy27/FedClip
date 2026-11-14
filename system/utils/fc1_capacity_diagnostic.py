"""Independent offline ranks. Federated code never imports this module.

Reuse the original trainer, indexed loader, read-only evaluation and RNG scopes.
One common fc1 SVD/corrected spectrum; private client conv2/fc2 capacities.
"""
import copy
import math
from pathlib import Path
import random
import sys
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
from utils.multi_rank_partition import protected_rng


def integers(text,name):
    try:values=[int(x.strip()) for x in text.split(',')]
    except (ValueError,AttributeError):raise ValueError(f'{name} must be comma-separated integers')
    if not values or len(set(values))!=len(values):raise ValueError(f'{name} must be nonempty and without duplicates')
    return sorted(values)


def comparison_mode(cli):
    mode=getattr(cli,'comparison','rank')
    if mode not in ('rank','spectrum'):raise ValueError('comparison must be rank or spectrum')
    if mode=='spectrum' and integers(cli.fc1_ranks,'fc1-ranks')!=[200]:
        raise ValueError('spectrum comparison requires --fc1-ranks 200; no silent rank override')
    return mode


def resolve_source(cli):
    if cli.server_file:path=Path(cli.server_file).expanduser().resolve()
    else:
        root=Path(cli.source_run).expanduser().resolve()
        candidates=([root/'Server_model_0.pt'] if (root/'Server_model_0.pt').is_file() else
            sorted((root/'final_models').rglob('Server_model_0.pt')))
        if len(candidates)!=1:raise FileNotFoundError(f'Need exactly one B_fc1 final server, found {len(candidates)} in {root}; use --server-file')
        path=candidates[0]
    for file in (path,path.parent/'manifest.json'):
        if not file.is_file():raise FileNotFoundError(f'Required same-run B_fc1 input missing: {file}')
    return path


def validate_manifest(manifest,source):
    if type(source) is not int or source<1:raise ValueError('source-round must be positive')
    args=SimpleNamespace(**copy.deepcopy(manifest['effective']))
    if (manifest.get('algorithm')!='FedMultiGroupProj' or manifest.get('server_model_count')!=1
        or manifest.get('client_count')!=20 or manifest.get('actual_train_count')!=source
        or manifest.get('aggregation_mode')!='group_avg_only'
        or not args.multi_avg_proto_ablation or not args.multi_svd or args.num_server_models!=1
        or args.join_ratio!=1 or args.num_clients!=20 or args.dataset!='Cifar100' or args.num_classes!=100
        or args.partition!='pat' or args.target_head_mode!='shared' or args.target_local_schedule!='joint'
        or args.multi_proto_mode!='none' or args.multi_proto_lambda!=0
        or getattr(args,'multi_svd_aggregation','full_avg')!='full_avg'
        or args.multi_svd_init!='low_rank_recovered' or args.multi_svd_init_rank_rate!=.9
        or args.multi_svd_spectral_mode!='next_tier' or spectral.layer_names(args)!=['fc1']
        or args.multi_svd_spectral_max_gain!=1.5 or getattr(args,'multi_fc1_tail_scale_strength',0)!=0):
        raise ValueError('Need B_fc1: K1/N20 full_avg, .90 recovered initialization, next_tier/fc1/gain1.5, no prototypes, shared/joint')
    if (args.multi_svd_effective_ratios!=[.9,.37,.35,.25,.15]
        or args.multi_svd_effective_assignment!=[i%5 for i in range(args.num_clients)]):
        raise ValueError('Need the source original five fixed capacity tiers and client_id % 5 assignment')
    if (args.is_regular!=1 or args.regular_lamda!=.001 or args.local_epochs!=5 or args.batch_size!=16
        or args.local_learning_rate!=.005 or args.local_grad_clip!=10
        or any(getattr(args,k,False) for k in ('target_prox_mu','target_z2_lambda','target_head_l2_lambda','multi_warmup_rounds'))
        or any(getattr(args,k,'none')!='none' for k in ('multi_kd_mode','target_c0_replay_mode','target_curriculum_mode'))
        or getattr(args,'multi_head_mode','shared')!='shared' or getattr(args,'multi_projection_scope','full')!='full'
        or getattr(args,'multi_helper_head_update','train')!='train' or getattr(args,'multi_last_epoch_protocol',False)
        or any(getattr(args,k,False) for k in ('few_shot','resume','learning_rate_decay','auto_break','meta_c0_split'))):
        raise ValueError('Need ordinary local5/batch16/lr.005 CE + F.001/clip10; no additional local objectives')
    optimizer=manifest.get('optimizer',{})
    if any(optimizer.get(k)!=v for k,v in dict(name='SGD',lr=.005,momentum=0.,weight_decay=0.,gradient_norm_cap=10.).items()):
        raise ValueError('Source optimizer must be the matching original SGD/clip protocol')
    declared=manifest.get('svd_spectral',{}).get('policy',{})
    if declared!=spectral.policy(args):raise ValueError('Source effective/declared spectral policy mismatch')
    return args


def load_source(cli):
    from flcore.trainmodel.models import Hyper_CNN_512
    path=resolve_source(cli);manifest_path=path.parent/'manifest.json'
    import json
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));args=validate_manifest(manifest,cli.source_round)
    backend=manifest.get('environment',{})
    keys=('deterministic','cudnn_deterministic','cudnn_benchmark','matmul_tf32','cudnn_tf32')
    if any(type(backend.get(k)) is not bool for k in keys):raise ValueError('Missing original Torch backend settings')
    torch.use_deterministic_algorithms(backend['deterministic'])
    torch.backends.cudnn.deterministic=backend['cudnn_deterministic'];torch.backends.cudnn.benchmark=backend['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=backend['matmul_tf32'];torch.backends.cudnn.allow_tf32=backend['cudnn_tf32']
    mother,payload=svd.load_checkpoint(path,'cpu')
    if (payload.get('role')!='post_aggregation_full_server' or payload.get('completed_train_count')!=cli.source_round
        or payload.get('model_id')!=0 or mother.ratio_LR!=1.
        or payload.get('svd_aggregation',{}).get('method')!='full_avg'
        or payload.get('svd_spectral_policy')!=manifest['svd_spectral']['policy']
        or not isinstance(mother.fc1,torch.nn.Linear) or not isinstance(mother.fc2,torch.nn.Linear)
        or not isinstance(mother.conv2,torch.nn.Conv2d)):
        raise ValueError('Need this run\'s complete post-Avg server, not a client/upload or other spectral policy')
    for key,manifest_key in [('model_kwargs','model_kwargs'),('model_family','model_family'),
        ('svd_initialization','svd_initialization'),('capacity_protocol','svd_capacity_protocol'),
        ('initial_parameter_sha256','initial_parameter_sha256')]:
        if payload.get(key)!=manifest.get(manifest_key):raise ValueError(f'Checkpoint/manifest {key} mismatch')
    if payload['model_kwargs']!=dict(in_features=3,num_classes=100,n_kernels=16,ratio_LR=1.,input_size=32):
        raise ValueError('Capacity diagnostic v1 requires the original RGB32/100-class Hyper_CNN_512 widths')
    for name,value in mother.state_dict().items():
        if not torch.equal(value,payload['state_dict'][name]) or not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite/inconsistent checkpoint alias: {name}')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        initial=Hyper_CNN_512(**dict(payload['model_kwargs'],ratio_LR=.9))
        initial.recover_larger_model();initial.ratio_LR=1.
        if model_digest(initial)!=payload['initial_parameter_sha256']:raise ValueError('Source initial parameter hash cannot be reproduced')
    original=copy.deepcopy(vars(args));args.data_root=cli.data_root;args.dataset=cli.dataset;args.data_subdir=cli.data_subdir
    if cli.dataset!='Cifar100':raise ValueError('Only the source CIFAR100 data can be used')
    args.data_formats='npz_only';args.data_expected_partition='pat'
    data=audit_image_clients(args);fields=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')
    actual=[{k:r[k] for k in fields} for r in data['files']]
    expected=[{k:r[k] for k in fields} for r in manifest['data_manifest']['files']]
    if data['config'].get('partition')!='pat' or data['config_sha256']!=manifest['data_manifest']['config_sha256'] or actual!=expected:
        raise ValueError('Actual pat data file/decoded tensor fingerprints differ from the source run')
    ids=integers(cli.client_ids,'client-ids');ranks=integers(cli.fc1_ranks,'fc1-ranks')
    if any(i<0 or i>=args.num_clients for i in ids):raise ValueError('client-ids out of source range')
    bounds=spectral.boundary_ranks(mother,args)['fc1'];k,b=bounds['k'],bounds['b'];maximum=max(bounds['actual_ranks'])
    if (k,b,maximum)!=(120,200,720):raise ValueError('Unexpected actual fc1 capacity boundaries')
    if any(r<k or r>maximum for r in ranks):raise ValueError(f'fc1-ranks must be distinct integers in [{k},{maximum}]')
    args.offline_loop_round=cli.source_round;args.data_fingerprint=digest_json(dict(config_sha256=data['config_sha256'],files=actual))
    args.initial_sha256=payload['initial_parameter_sha256'];args.device='cuda:0' if cli.device=='cuda' else 'cpu'
    args.multi_svd_diagnostics=0
    if cli.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Need exactly the requested visible GPU, or --device cpu')
        torch.cuda.set_device(0)
    record=dict(server_file=str(path),server_file_sha256=sha256(path),server_state_sha256=model_digest(mother),
        manifest_file=str(manifest_path),manifest_file_sha256=sha256(manifest_path),source_config=original,
        source_completed_aggregations=cli.source_round,offline_completed_train_count=cli.source_round+1,
        label=f'Offline R{cli.source_round+1} local training based on S{cli.source_round}',
        initial_parameter_sha256=args.initial_sha256,data_fingerprint=args.data_fingerprint,
        client_ids=ids,fc1_ranks=ranks,fc1_boundaries=bounds,
        original_backend_settings={key:backend[key] for key in keys},prototype_library_used=False,
        file_binding_note='Final source format declares metadata but no final server-file content hash; actual file/state hashes recorded here')
    return mother.to(args.device),args,data,record


@torch.no_grad()
def common_fc1(mother,args,ranks,*,include_raw=False):
    """Exactly ONE fc1 SVD, shared by all clients and all rank branches."""
    tick=time.perf_counter();bounds=spectral.boundary_ranks(mother,args)['fc1']
    p,s,qh=torch.linalg.svd(mother.fc1.weight.detach(),full_matrices=False)
    modified,receipt=spectral.correct_spectrum(s,bounds['k'],bounds['b'],'next_tier',args.multi_svd_spectral_max_gain,20)
    maximum=max(ranks);root=modified[:maximum].sqrt()
    u=(p[:,:maximum]@torch.diag(root)).cpu();v=(torch.diag(root)@qh[:maximum]).cpu()
    evidence=dict(P=p.detach().cpu(),S=s.detach().cpu(),Qh=qh.detach().cpu(),corrected_S=modified.detach().cpu(),
        weight_u=u,weight_v=v)
    if include_raw:
        raw_root=s[:maximum].sqrt()
        evidence.update(raw_weight_u=(p[:,:maximum]@torch.diag(raw_root)).cpu(),
            raw_weight_v=(torch.diag(raw_root)@qh[:maximum]).cpu())
    for value in evidence.values():old.finite(value)
    receipt.update(fc1_svd_calls=1,construction_rank=maximum,source_fc1_weight_sha256=tensor_fingerprint({'fc1':mother.fc1.weight}),
        common_corrected_spectrum_sha256=tensor_fingerprint({'S':modified}),seconds=time.perf_counter()-tick)
    if include_raw:receipt['common_raw_spectrum_sha256']=tensor_fingerprint({'S':s})
    return evidence,receipt


@torch.no_grad()
def client_template(mother,rate):
    from flcore.trainmodel.models import Decom_COV,Decom_LINEAR
    with protected_rng([mother]):
        model=copy.deepcopy(mother)
        model.conv2=Decom_COV(model.conv2,rate);model.fc2=Decom_LINEAR(model.fc2,rate)
        model.ratio_LR=rate;model._rebuild_base();model.to(next(mother.parameters()).device)
    return model


@torch.no_grad()
def rank_branch(template,evidence,rank):
    from flcore.trainmodel.models import FactorizedLinear
    with protected_rng([template]):
        model=copy.deepcopy(template);old_fc1=model.fc1
        layer=FactorizedLinear(old_fc1.in_features,old_fc1.out_features,
            rank/min(old_fc1.in_features,old_fc1.out_features),bias=old_fc1.bias is not None).to(old_fc1.weight)
        if layer.rank!=rank:raise ValueError('Actual FactorizedLinear rank differs from requested integer')
        layer.weight_u.copy_(evidence['weight_u'][:,:rank]);layer.weight_v.copy_(evidence['weight_v'][:rank,:])
        if layer.bias is not None:layer.bias.copy_(old_fc1.bias)
        model.fc1=layer;model._rebuild_base()
    return model


def spectrum_branch(template,evidence,rank,variant):
    if variant not in ('raw','corrected'):raise ValueError('Unknown spectrum variant')
    chosen=evidence if variant=='corrected' else dict(evidence,weight_u=evidence['raw_weight_u'],weight_v=evidence['raw_weight_v'])
    return rank_branch(template,chosen,rank)


def verify_prefix_and_other_state(left,right,*,prefix_rank=None):
    rank=min(left.fc1.rank,right.fc1.rank) if prefix_rank is None else prefix_rank
    if not 0<rank<=min(left.fc1.rank,right.fc1.rank):raise ValueError('Invalid common-prefix length')
    if (not torch.equal(left.fc1.weight_u[:,:rank],right.fc1.weight_u[:,:rank])
        or not torch.equal(left.fc1.weight_v[:rank,:],right.fc1.weight_v[:rank,:])):raise RuntimeError('Common factor prefix differs')
    a=dict(left.named_parameters());b=dict(right.named_parameters())
    if a.keys()!=b.keys():raise RuntimeError('Branch parameter names differ')
    for name,value in a.items():
        if name not in ('fc1.weight_u','fc1.weight_v') and not torch.equal(value,b[name]):raise RuntimeError(f'Non-fc1 state differs: {name}')
    ab=dict(left.named_buffers());bb=dict(right.named_buffers())
    if ab.keys()!=bb.keys() or any(not torch.equal(value,bb[name]) for name,value in ab.items()):raise RuntimeError('Branch buffers differ')
    if left.ratio_LR!=right.ratio_LR:raise RuntimeError('Original client capacity flag changed')


def factor_state(model):
    return dict(weight_u=model.fc1.weight_u.detach().cpu().clone(),weight_v=model.fc1.weight_v.detach().cpu().clone())


@torch.no_grad()
def rollback_tail(model,initial,start):
    r=model.fc1.rank
    if not 0<start<r or initial['weight_u'].shape!=model.fc1.weight_u.shape or initial['weight_v'].shape!=model.fc1.weight_v.shape:
        raise ValueError('Invalid rollback range or mismatched branch download snapshot')
    restored=copy.deepcopy(model)
    restored.fc1.weight_u[:,start:].copy_(initial['weight_u'][:,start:]);restored.fc1.weight_v[start:,:].copy_(initial['weight_v'][start:,:])
    target=dict(restored.named_parameters())
    for name,value in model.named_parameters():
        if name=='fc1.weight_u':
            if not torch.equal(target[name][:,:start],value[:,:start]) or not torch.equal(target[name][:,start:].cpu(),initial['weight_u'][:,start:]):raise RuntimeError('Wrong U rollback source')
        elif name=='fc1.weight_v':
            if not torch.equal(target[name][:start],value[:start]) or not torch.equal(target[name][start:].cpu(),initial['weight_v'][start:]):raise RuntimeError('Wrong V rollback source')
        elif not torch.equal(target[name],value):raise RuntimeError(f'Rollback changed other parameter {name}')
    if any(not torch.equal(v,dict(restored.named_buffers())[n]) for n,v in model.named_buffers()):raise RuntimeError('Rollback changed buffers')
    return restored


def slot_ranges(bounds):
    k,b=bounds['k'],bounds['b'];maximum=max(bounds['actual_ranks'])
    return [(0,k),(k,b),(b,maximum)]


class Observer:
    """Read-only probes around the UNMODIFIED CE+F trainer and optimizer."""
    def __init__(self,client,evidence,ranges,train_data,test_data,enabled=True):
        self.client=client;self.model=client.model_set[0];self.optimizer=client.optimizer_set[0]
        self.evidence=evidence;self.ranges=ranges;self.train_data=train_data;self.test_data=test_data;self.enabled=enabled
        self.steps=0;self.rows=[];self.gradients=[];self.signal=[];self.endpoints=[];self.handles=[];self.norms={};self.capture=None
        self.parameters=list(self.model.named_parameters());self.probing=False;self.seconds=0.;self.norm_sum=0.;self.norm_max=0.;self.clips=0;self.coef_sum=0.
        self.initial=factor_state(self.model);self.initial_products={}
        if enabled:
            for lo,hi in ranges:
                end=min(hi,self.model.fc1.rank)
                if end>lo:self.initial_products[lo]=self.initial['weight_u'][:,lo:end].double()@self.initial['weight_v'][lo:end].double()
    def endpoint(self,epoch):
        tick=time.perf_counter()
        for split,data in [('train',self.train_data),('test',self.test_data)]:
            values,_=old.evaluate(self.model,data,self.client.args.test_batch_size,self.client.device)
            self.endpoints.append(dict(epoch=epoch,split=split,**values))
        self.seconds+=time.perf_counter()-tick
    def snapshot(self,stage):
        if not self.enabled:return
        tick=time.perf_counter();u=self.model.fc1.weight_u.detach().cpu().double();v=self.model.fc1.weight_v.detach().cpu().double()
        for lo,hi in self.ranges:
            end=min(hi,self.model.fc1.rank);present=end>lo
            initial=self.initial_products.get(lo);start_norm=float(initial.norm()) if present else 0.
            change=float((u[:,lo:end]@v[lo:end]-initial).norm()) if present else 0.
            self.rows.append(dict(stage=stage,optimizer_step=self.steps,slot_start=lo+1,slot_end=hi,retained_slot_end=end if present else None,
                present=present,download_block_frobenius_norm=start_norm,absolute_product_change_frobenius=change,
                relative_product_change=change/start_norm if start_norm>0 else None,
                relative_change_reason=None if start_norm>0 else 'initial_zero_block' if present else 'slots_not_retained',
                scope='fixed download factor slots; float64 products of actual factor values, NOT re-SVD directions'))
        spectrum=torch.linalg.svdvals(u@v);energy=old.finite(spectrum.square());total=float(energy.sum())
        for row in self.rows[-len(self.ranges):]:
            lo=row['slot_start']-1;hi=row['slot_end'];block=float(energy[lo:hi].sum())
            row.update(resvd_energy_in_rank_interval=block,resvd_total_energy=total,resvd_energy_fraction=block/total if total else 0.,
                resvd_scope='actual current U@V re-SVD; singular values re-ordered independently of fixed factor slots')
        self.seconds+=time.perf_counter()-tick
    def end_epoch(self,epoch):
        if epoch in {1,self.client.local_epochs}:self.snapshot(f'epoch_{epoch}');self.endpoint(epoch)
    def __enter__(self):
        self.original_step=self.optimizer.step;self.step_was_instance_attribute='step' in self.optimizer.__dict__;self.original_loss=self.client.loss
        try:
            if self.enabled:
                def inputs(module,x):
                    if self.steps==0:self.capture=dict(inputs=x[0].detach().clone())
                def output(module,x,y):
                    if self.steps==0:self.capture['output']=y
                self.handles.extend([self.model.fc1.register_forward_pre_hook(inputs),self.model.fc1.register_forward_hook(output)])
                for name,p in self.parameters:
                    if not p.requires_grad:continue
                    def hook(g,name=name):
                        if self.probing:return
                        self.norms[name]=g.detach().norm(2)
                        if self.steps==0 and name in ('fc1.weight_u','fc1.weight_v'):
                            for lo,hi in self.ranges:
                                part=g[:,lo:hi] if name.endswith('_u') else g[lo:hi,:]
                                self.norms[(name,lo)]=float(part.detach().norm())
                    self.handles.append(p.register_hook(hook))
                def loss(logits,labels):
                    ce=self.original_loss(logits,labels)
                    if self.steps==0:
                        tick=time.perf_counter();before=[p.grad for _,p in self.parameters];self.probing=True
                        try:dy=torch.autograd.grad(ce,self.capture['output'],retain_graph=True)[0]
                        finally:self.probing=False
                        if any(p.grad is not g for (_,p),g in zip(self.parameters,before)):raise RuntimeError('CE probe modified .grad')
                        G=old.finite(dy.detach().cpu().double().T@self.capture['inputs'].cpu().double())
                        u=self.model.fc1.weight_u.detach().cpu().double();v=self.model.fc1.weight_v.detach().cpu().double()
                        for lo,hi in self.ranges:
                            end=min(hi,self.model.fc1.rank)
                            self.signal.append(dict(slot_start=lo+1,slot_end=hi,present=end>lo,
                                CE_restored_weight_gradient_frobenius=float(G.norm()),
                                CE_G_Q_download_interval_frobenius=float((G@self.evidence['Qh'][lo:end].double().T).norm()) if end>lo else 0.,
                                CE_G_transpose_P_download_interval_frobenius=float((G.T@self.evidence['P'][:,lo:end].double()).norm()) if end>lo else 0.,
                                CE_U_gradient_frobenius=float((G@v[lo:end].T).norm()) if end>lo else 0.,
                                CE_V_gradient_frobenius=float((u[:,lo:end].T@G).norm()) if end>lo else 0.,
                                loss_scope='CE only; probe reads existing training forward, no .grad writes',samples=len(labels)))
                        self.capture=None;self.seconds+=time.perf_counter()-tick
                    return ce
                self.client.loss=loss
            def step(*a,**kw):
                if self.enabled:
                    tick=time.perf_counter();norm=old.finite(torch.stack([self.norms[n] for n,p in self.parameters if p.grad is not None]).norm(2))
                    coefficient=float((self.client.args.local_grad_clip/(norm+1e-6)).clamp(max=1))
                    self.norm_sum+=float(norm);self.norm_max=max(self.norm_max,float(norm));self.clips+=int(coefficient<1);self.coef_sum+=coefficient
                    if self.steps==0:
                        u,v=self.model.fc1.weight_u,self.model.fc1.weight_v
                        post=old.finite(torch.stack([p.grad.detach().norm(2) for _,p in self.parameters if p.grad is not None]).norm(2))
                        for lo,hi in self.ranges:
                            self.gradients.append(dict(slot_start=lo+1,slot_end=hi,present=self.model.fc1.rank>lo,
                                total_pre_clip_grad_norm=float(norm),total_post_clip_grad_norm=float(post),clip_coefficient=coefficient,
                                U_grad_norm_pre_clip=self.norms[('fc1.weight_u',lo)],V_grad_norm_pre_clip=self.norms[('fc1.weight_v',lo)],
                                U_grad_norm_post_clip=float(u.grad[:,lo:hi].detach().norm()),V_grad_norm_post_clip=float(v.grad[lo:hi].detach().norm()),
                                loss_scope='actual CE + original weighted F total gradient; original clip and SGD unchanged'))
                    self.norms.clear();self.seconds+=time.perf_counter()-tick
                result=self.original_step(*a,**kw);self.steps+=1
                if self.steps==1:self.snapshot('step_1')
                return result
            self.optimizer.step=step
            return self
        except BaseException:
            self.__exit__(*sys.exc_info());raise
    def __exit__(self,*exc):
        if self.step_was_instance_attribute:self.optimizer.step=self.original_step
        elif 'step' in self.optimizer.__dict__:del self.optimizer.step
        self.client.loss=self.original_loss
        for handle in self.handles:handle.remove()
        self.capture=None;self.norms.clear()


def train_branch(model,mother,args,cid,data,test,folder,evidence,ranges,diagnostics=True,trainer=None):
    client=old.local_client(model,mother,args,cid,None,None,data,folder)
    # Only transient training checkpoints see this explicit offline receipt.
    client.svd_spectral_latest=[dict(layer='fc1',actual_rank=model.fc1.rank,offline_common_spectrum=True)]
    observer=Observer(client,evidence,ranges,data,test,diagnostics)
    loader=old.EpochLoader(data,args,cid,args.offline_loop_round,observer);client.load_train_data=lambda:loader
    tick=time.perf_counter();observer.snapshot('epoch_0');observer.endpoint(0)
    initial_rng_sha256=old.rng_fingerprint(old.rng_snapshot(model))
    with observer:(trainer or proto.train)(client)
    steps=args.local_epochs*math.ceil(len(data)/args.batch_size)
    if observer.steps!=steps or len(loader.batches)!=steps or client.train_stats[0]['processed_train_samples']!=len(data)*args.local_epochs:
        raise RuntimeError('Unexpected offline local training budget')
    if client.avg_proto_local is not None or client.full_uploads[0] is None:raise RuntimeError('Expected no prototypes and an ordinary recovered training result')
    return dict(training=client.train_stats[0],updates=observer.rows,endpoints=observer.endpoints,first_step=observer.gradients,
        CE_signal=observer.signal,batches=loader.batches,diagnostic_seconds=observer.seconds,
        wall_seconds_including_observations=time.perf_counter()-tick,
        initial_training_rng_sha256=initial_rng_sha256,
        final_rng_sha256=old.rng_fingerprint(old.rng_snapshot(model)),final_model_sha256=tensor_fingerprint(model.state_dict()),
        all_batch_indices_sha256=digest_json(loader.batches),initial_factors=observer.initial,
        observed_grad_norm_mean_per_step=observer.norm_sum/steps if diagnostics else None,
        observed_grad_norm_max=observer.norm_max if diagnostics else None,
        observed_clip_fraction=observer.clips/steps if diagnostics else None,
        observed_clip_coefficient_mean=observer.coef_sum/steps if diagnostics else None),client


def paired_comparisons(endpoints,clients,bounds):
    k,b=bounds['k'],bounds['b'];maximum=max(bounds['actual_ranks']);rows=[]
    for cid in clients:
        for low,high in [(k,b),(b,maximum)]:
            chosen=[r for r in endpoints if r['client_id']==cid and r['split']=='test' and r['fc1_rank'] in (low,high)]
            points={(r['fc1_rank'],r['epoch']):r for r in chosen}
            if not all((rank,epoch) in points for rank in (low,high) for epoch in (0,5)):continue
            row=dict(client_id=cid,lower_fc1_rank=low,higher_fc1_rank=high,accuracy_unit='fraction')
            for epoch in (0,5):
                a,c=points[low,epoch],points[high,epoch]
                if a['samples']!=c['samples']:raise ValueError('Unpaired endpoint sample counts')
                row[f'epoch_{epoch}_higher_minus_lower_test_accuracy']=c['accuracy']-a['accuracy']
                row[f'epoch_{epoch}_higher_minus_lower_test_CE']=c['ce']-a['ce']
            row['additional_rank_gap_change_after_local_training']=row['epoch_5_higher_minus_lower_test_accuracy']-row['epoch_0_higher_minus_lower_test_accuracy']
            rows.append(row)
    return rows


def capacity_summary(endpoints):
    groups={}
    for row in endpoints:
        key=(row['rank_rate'],row['fc1_rank'],row['epoch'],row['split']);groups.setdefault(key,[]).append(row)
    return [dict(rank_rate=rate,fc1_rank=rank,epoch=epoch,split=split,clients=[r['client_id'] for r in rows],
        samples=sum(r['samples'] for r in rows),correct=sum(r['correct'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/sum(r['samples'] for r in rows),
        ce=sum(r['ce']*r['samples'] for r in rows)/sum(r['samples'] for r in rows)) for (rate,rank,epoch,split),rows in sorted(groups.items())]


def spectrum_comparisons(endpoints,clients,rank=200):
    """Pair by client AND variant, never collapse the two identical rank IDs."""
    result=[]
    for cid in clients:
        rows=[r for r in endpoints if r['client_id']==cid]
        points={}
        for row in rows:
            key=(row['spectrum_variant'],row['epoch'],row['split'])
            if key in points or row['fc1_rank']!=rank:raise ValueError('Duplicate/incorrect-rank spectrum endpoint')
            points[key]=row
        expected={(v,e,s) for v in ('raw','corrected') for e in (0,1,5) for s in ('train','test')}
        if points.keys()!=expected:raise ValueError('Incomplete raw/corrected epoch0/1/5 train/test pair')
        item=dict(client_id=cid,fc1_rank=rank,variants=['raw','corrected'],accuracy_unit='fraction',CE_unit='nats/sample')
        for split in ('train','test'):
            for epoch in (0,1,5):
                a,b=points['raw',epoch,split],points['corrected',epoch,split]
                if a['samples']!=b['samples']:raise ValueError('Spectrum variants used different endpoint sample counts')
                item[f'epoch_{epoch}_{split}_samples']=a['samples']
                for variant,row in [('raw',a),('corrected',b)]:
                    for metric,key in [('accuracy','accuracy'),('CE','ce'),('correct','correct')]:
                        item[f'{variant}_epoch_{epoch}_{split}_{metric}']=row[key]
                for metric,key in [('accuracy','accuracy'),('CE','ce')]:
                    item[f'epoch_{epoch}_corrected_minus_raw_{split}_{metric}']=b[key]-a[key]
            for metric in ('accuracy','CE'):
                item[f'epoch_5_minus_epoch_0_{split}_{metric}_gap_change']=(
                    item[f'epoch_5_corrected_minus_raw_{split}_{metric}']-item[f'epoch_0_corrected_minus_raw_{split}_{metric}'])
            for variant in ('raw','corrected'):
                item[f'{variant}_{split}_CE_epoch5_minus_epoch1']=points[variant,5,split]['ce']-points[variant,1,split]['ce']
        result.append(item)
    return result


def spectrum_capacity_summary(endpoints):
    if any(r.get('spectrum_variant') not in ('raw','corrected') for r in endpoints):raise ValueError('Missing spectrum variant')
    return [dict(spectrum_variant=variant,**row) for variant in ('raw','corrected')
        for row in capacity_summary([r for r in endpoints if r['spectrum_variant']==variant])]


def spectrum_summary_text(pairs,label):
    lines=['# fc1 同秩原谱／修正谱离线对照','',label,'',
        '两分支均为200秩。差值为 corrected−raw，准确率差为百分点；固定epoch报告，测试集不用于选择。',
        '','| 客户端 | epoch0 测试差 | epoch1 测试差 | epoch5 测试差 | epoch5差−epoch0差 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in pairs:
        lines.append(f"| {r['client_id']} | {100*r['epoch_0_corrected_minus_raw_test_accuracy']:.4f} | {100*r['epoch_1_corrected_minus_raw_test_accuracy']:.4f} | {100*r['epoch_5_corrected_minus_raw_test_accuracy']:.4f} | {100*r['epoch_5_minus_epoch_0_test_accuracy_gap_change']:.4f} |")
    lines+=['','| 客户端 | 谱分支 | 训练CE：epoch5−epoch1 | 测试CE：epoch5−epoch1 |','| --- | --- | ---: | ---: |']
    for r in pairs:
        for variant in ('raw','corrected'):
            lines.append(f"| {r['client_id']} | {variant} | {r[f'{variant}_train_CE_epoch5_minus_epoch1']:.6g} | {r[f'{variant}_test_CE_epoch5_minus_epoch1']:.6g} |")
    lines+=['','来源检查点始终是B-fc1；raw仅表示本次下发不再校正S101的原始谱。没有验证集或尾部回退；谱干预可能无效，此时两分支允许相同。']
    return '\n'.join(lines)+'\n'


def summary_text(comparisons,rollback):
    lines=['# fc1 本地容量：离线配对结果','','准确率差为百分点；CE 为每样本交叉熵。测试只用于报告，不选模型。',
        '','| 客户端 | 增加秩 | epoch0 测试差 | epoch5 测试差 | 差距变化 |','| --- | --- | ---: | ---: | ---: |']
    for r in comparisons:
        lines.append(f"| {r['client_id']} | {r['higher_fc1_rank']}−{r['lower_fc1_rank']} | {100*r['epoch_0_higher_minus_lower_test_accuracy']:.4f} | {100*r['epoch_5_higher_minus_lower_test_accuracy']:.4f} | {100*r['additional_rank_gap_change_after_local_training']:.4f} |")
    lines+=['','| 客户端 | 回退 201–720 后准确率变化 | 回退后 CE 变化 |','| --- | ---: | ---: |']
    for r in rollback:lines.append(f"| {r['client_id']} | {100*r['rollback_minus_trained_test_accuracy']:.4f} | {r['rollback_minus_trained_test_CE']:.6g} |")
    lines+=['','固定槽位乘积变化与重新 SVD 谱能量分别记录。先看裁剪后的实际更新，再结合回退推理判断本地更新的分类价值；不能据此证明跨轮保留。']
    return '\n'.join(lines)+'\n'


def run(cli):
    comparison=comparison_mode(cli)
    if cli.threads<1:raise ValueError('threads must be positive')
    output=Path(cli.output_dir).expanduser().resolve();source_path=resolve_source(cli)
    if output.exists():raise FileExistsError(f'Output exists: {output}')
    roots=[Path(cli.data_root).expanduser().resolve()/cli.dataset/cli.data_subdir,
        Path(cli.source_run).expanduser().resolve() if cli.source_run else source_path.parent]
    if any(output==p or p in output.parents or output in p.parents for p in roots):raise ValueError('Output must be separate from source/data')
    torch.set_num_threads(cli.threads)
    mother,args,data_manifest,source=load_source(cli);output.mkdir(parents=True,exist_ok=False)
    input_hashes={source_path:sha256(source_path),source_path.parent/'manifest.json':sha256(source_path.parent/'manifest.json')}
    for row in data_manifest['files']:input_hashes[Path(row['path'])]=row['sha256']
    input_hashes[Path(data_manifest['config_path'])]=data_manifest['config_sha256']
    manifest=dict(status='running',source=source,requested=vars(cli),environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),
        role=source['label'],prediction='original classifier; no model selection, prototypes, aggregation or extra local epochs',
        units='accuracy fraction; CE nats/sample; changes Frobenius L2, relative changes dimensionless',
        trained_model_export=False,common_prefix_checks=[],initial_rng_equal_by_client=True)
    if comparison=='spectrum':
        manifest['spectrum_comparison']=dict(comparison='spectrum',checkpoint_origin='B_fc1 next_tier/fc1/max_gain1.5',
            offline_download_variants=dict(raw='unmodified singular values of THIS checkpoint',corrected='next_tier/max_gain1.5 applied once to THIS checkpoint spectrum'),
            fc1_rank=200,common_prefix_rank=source['fc1_boundaries']['k'],tail_rollback_performed=False,
            pairing='client_id x spectrum_variant; same rank is NOT a unique branch ID')
    write_json(output/'manifest.json',manifest);write_json(output/'data_manifest.json',data_manifest)
    endpoints=[];updates=[];signals=[];gradients=[];training=[];batches=[];rollbacks=[]
    random.seed(args.seed);np.random.seed(args.seed);torch.random.default_generator.manual_seed(args.seed)
    if args.device.startswith('cuda'):torch.cuda.default_generators[0].manual_seed(args.seed)
    try:
        evidence,receipt=(common_fc1(mother,args,source['fc1_ranks'],include_raw=True) if comparison=='spectrum' else
            common_fc1(mother,args,source['fc1_ranks']))
        manifest['common_fc1_correction']=receipt
        ranges=slot_ranges(source['fc1_boundaries'])
        if comparison=='spectrum':ranges=ranges[:2]
        for cid in source['client_ids']:
            rate=args.multi_svd_effective_ratios[args.multi_svd_effective_assignment[cid]]
            train=read_image_client(args,cid,True);test=read_image_client(args,cid,False)
            template=client_template(mother,rate);base_rng=old.rng_snapshot(template)
            branches=[(200,'raw'),(200,'corrected')] if comparison=='spectrum' else [(r,None) for r in source['fc1_ranks']]
            reference=(spectrum_branch(template,evidence,200,'raw') if comparison=='spectrum' else
                rank_branch(template,evidence,min(source['fc1_ranks'])))
            batch_hash=None;ending_rng=None
            for rank,variant in branches:
                print(f'\n========== {source["label"]}: client={cid}, other-layer rate={rate}, fc1 rank={rank}'+
                    (f', spectrum={variant}' if variant else '')+' ==========',flush=True)
                model=spectrum_branch(template,evidence,rank,variant) if variant else rank_branch(template,evidence,rank)
                if variant:
                    verify_prefix_and_other_state(reference,model,prefix_rank=source['fc1_boundaries']['k'])
                    if sum(p.numel() for p in reference.parameters())!=sum(p.numel() for p in model.parameters()):raise RuntimeError('Unequal parameter counts at fixed rank')
                else:verify_prefix_and_other_state(reference,model)
                check=dict(client_id=cid,fc1_rank=rank,reference_fc1_rank=reference.fc1.rank,
                    common_prefix_exact=True,other_layers_head_buffers_exact=True,model_ratio_LR=model.ratio_LR)
                if variant:check.update(spectrum_variant=variant,common_prefix_rank=source['fc1_boundaries']['k'],same_parameter_count=True)
                manifest['common_prefix_checks'].append(check)
                initial_rng_sha=old.rng_fingerprint(base_rng);old.restore_rng(base_rng)
                with tempfile.TemporaryDirectory(prefix='transient_training_',dir=output) as temp:
                    result,client=train_branch(model,mother,args,cid,train,test,temp,evidence,ranges,bool(cli.diagnostics))
                if result['initial_training_rng_sha256']!=initial_rng_sha:raise RuntimeError('Pre-training observations changed RNG')
                if batch_hash is None:batch_hash=result['all_batch_indices_sha256'];ending_rng=result['final_rng_sha256']
                if batch_hash!=result['all_batch_indices_sha256'] or ending_rng!=result['final_rng_sha256']:raise RuntimeError('Rank branches have different batch orders or training RNG')
                identity=dict(client_id=cid,rank_rate=rate,fc1_rank=rank,other_layer_ranks={n:getattr(model,n).rank for n in ('conv2','fc2')},
                    source_completed_aggregations=cli.source_round,offline_completed_train_count=cli.source_round+1)
                if variant:identity['spectrum_variant']=variant
                for key,table in [('endpoints',endpoints),('updates',updates),('CE_signal',signals),('first_step',gradients)]:table.extend(dict(identity,**row) for row in result[key])
                for row in result['batches']:
                    batch=dict(row);indices=batch.pop('indices');batches.append(dict(identity,**batch,samples=len(indices),sample_indices_sha256=digest_json(indices)))
                training.append(dict(identity,**result['training'],all_batch_indices_sha256=result['all_batch_indices_sha256'],
                    initial_rng_sha256=initial_rng_sha,final_rng_sha256=result['final_rng_sha256'],final_model_sha256=result['final_model_sha256'],
                    model_parameter_count=sum(p.numel() for p in model.parameters()),online_diagnostic_seconds=result['diagnostic_seconds'],
                    wall_seconds_including_observations=result['wall_seconds_including_observations'],
                    observed_grad_norm_mean_per_step=result['observed_grad_norm_mean_per_step'],observed_grad_norm_max=result['observed_grad_norm_max'],
                    observed_clip_fraction=result['observed_clip_fraction'],observed_clip_coefficient_mean=result['observed_clip_coefficient_mean']))
                if comparison=='rank' and rank==max(source['fc1_boundaries']['actual_ranks']):
                    tick=time.perf_counter();restored=rollback_tail(model,result['initial_factors'],source['fc1_boundaries']['b'])
                    values,_=old.evaluate(restored,test,args.test_batch_size,args.device)
                    final=next(r for r in result['endpoints'] if r['epoch']==args.local_epochs and r['split']=='test')
                    rollbacks.append(dict(identity,rollback_slot_start=source['fc1_boundaries']['b']+1,rollback_slot_end=rank,
                        trained_test_CE=final['ce'],trained_test_accuracy=final['accuracy'],trained_test_correct=final['correct'],
                        rollback_test_CE=values['ce'],rollback_test_accuracy=values['accuracy'],rollback_test_correct=values['correct'],test_samples=values['samples'],
                        rollback_minus_trained_test_accuracy=values['accuracy']-final['accuracy'],rollback_minus_trained_test_CE=values['ce']-final['ce'],
                        download_tail_sha256=tensor_fingerprint(result['initial_factors']),all_other_parameters_from_final_model_verified=True,
                        evaluation_seconds=time.perf_counter()-tick));del restored
                tables=[('endpoints',endpoints),('fixed_slot_updates',updates),('first_batch_CE_signal',signals),
                    ('first_step_gradients',gradients),('local_training',training),('batch_order',batches)]
                if comparison=='rank':tables.append(('tail_rollback',rollbacks))
                for name,table in tables:
                    write_json(output/(name+'.json'),table);old.write_csv(output/(name+'.csv'),table)
                write_json(output/'manifest.json',manifest)
                del model,client,result
            del template,reference
        pairs=(spectrum_comparisons(endpoints,source['client_ids']) if comparison=='spectrum' else
            paired_comparisons(endpoints,source['client_ids'],source['fc1_boundaries']))
        summary=spectrum_capacity_summary(endpoints) if comparison=='spectrum' else capacity_summary(endpoints)
        for name,table in [('spectrum_comparisons' if comparison=='spectrum' else 'rank_comparisons',pairs),('capacity_summary',summary)]:
            write_json(output/(name+'.json'),table);old.write_csv(output/(name+'.csv'),table)
        (output/'summary.md').write_text(spectrum_summary_text(pairs,source['label']) if comparison=='spectrum' else
            summary_text(pairs,rollbacks),encoding='utf-8')
        if any(sha256(file)!=value for file,value in input_hashes.items()):raise RuntimeError('Read-only input changed during diagnostics')
        manifest.update(status='complete',training_task_count=len(training),input_hashes_unchanged=True,
            actual_optimizer_steps=sum(r['train_batches'] for r in training),actual_training_sample_visits=sum(r['processed_train_samples'] for r in training),
            common_fc1_svd_calls=1,other_layer_download_svd_calls=len(source['client_ids'])*2,
            other_layer_capacity_note='ratio_LR keeps original client value; fc1 actual integer rank is overridden explicitly')
        write_json(output/'manifest.json',manifest);print('Offline diagnostics:',output,flush=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(output/'manifest.json',manifest);raise
