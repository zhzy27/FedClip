"""Read-only SVD evidence + paired gauges around the EXISTING K1 trainer.

No federated module imports this file. The only optimized states are offline
copies. SVD/SGD wrappers are process-local, scoped, and restored on exceptions.
"""
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.client_data import audit_image_clients,read_image_client
from common.data_loading import make_data_loader
from common.experiment import sha256,write_json,environment,code_version
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_group_proto import Features,decode_packet,packet_sha
from utils import multi_group_svd as svd,single_avg_proto as proto
from utils.multi_rank_partition import protected_rng


def rng_snapshot(model):
    devices={p.device.index for p in list(model.parameters())+list(model.buffers()) if p.device.type=='cuda'}
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state().clone(),
        cuda={d:torch.cuda.get_rng_state(d).clone() for d in sorted(devices)})


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    for device,value in state['cuda'].items():torch.cuda.set_rng_state(value,device)


def rng_fingerprint(state):
    # Pickled Tensor storage IDs are not stable: hash raw state bytes instead.
    h=hashlib.sha256(pickle.dumps((state['python'],state['numpy']),protocol=4))
    h.update(state['cpu'].numpy().tobytes())
    for device,value in state['cuda'].items():h.update(str(device).encode());h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def temporary_attribute(obj,name,value):
    original=getattr(obj,name)
    setattr(obj,name,value)
    try:yield original
    finally:setattr(obj,name,original)


def finite(value):
    if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite offline diagnostic tensor')
    return value


def factor_rank(model,rate):
    # Read the actual legacy constructor's integer-rank rule, no ratio*800 guess.
    from flcore.trainmodel.models import FactorizedLinear
    with protected_rng([model]):
        return FactorizedLinear(model.fc1.in_features,model.fc1.out_features,rate,bias=model.fc1.bias is not None).rank


@torch.no_grad()
def download_once(mother,rate):
    """Normal legacy SVD; capture the SAME fc1 decomposition, including zero directions."""
    local=copy.deepcopy(mother);target=local.fc1.weight
    address,shape=target.data_ptr(),target.shape
    original=torch.linalg.svd;evidence=[]
    def capture(matrix,*a,**kw):
        result=original(matrix,*a,**kw)
        if matrix.shape==shape and matrix.data_ptr()==address:
            evidence.append(tuple(t.detach().cpu().clone() for t in result))
        return result
    with temporary_attribute(torch.linalg,'svd',capture):local.decom_larger_model(rate)
    local.ratio_LR=rate;local.to(next(mother.parameters()).device)
    if len(evidence)!=1:raise RuntimeError('Must capture exactly the one actual fc1 download SVD')
    p,s,qh=evidence[0]
    for t in (p,s,qh):finite(t)
    if (s<0).any():raise ValueError('Negative singular values')
    return local,dict(P=p,S=s,Qh=qh,actual_rank=local.fc1.rank,svd_calls_for_fc1=1)


@torch.no_grad()
def rescale_tail(download,evidence,minimum):
    model=copy.deepcopy(download);layer=model.fc1;rank=layer.rank
    if not 0<minimum<rank<=len(evidence['S']):raise ValueError('Selected client needs fc1 rank strictly above minimum capacity')
    from utils.fc1_tail_factors import scales
    s=evidence['S'];a,b,floor=scales(s,minimum,rank,1.)
    # Zero sigma, positive floor => nonzero unit P column and zero V row.
    # If the floor itself is zero, sigma/a=0 is the finite zero convention.
    layer.weight_u[:,minimum:].copy_((evidence['P'][:,minimum:rank]*a).to(layer.weight_u))
    layer.weight_v[minimum:,:].copy_((b[:,None]*evidence['Qh'][minimum:rank,:]).to(layer.weight_v))
    for t in (layer.weight_u,layer.weight_v):finite(t)
    return model,dict(minimum_actual_fc1_rank=minimum,client_actual_fc1_rank=rank,
        floor_singular_value=float(s[minimum-1]),floor_factor_scale=float(floor),
        zero_tail_singular_values=int((s[minimum:rank]==0).sum()),
        zero_floor_convention='if floor and sigma are zero, both scales stay zero',
        rule='a_j=max(sqrt(sigma_j),sqrt(sigma_minimum)); U=a_j*P_j, V=(sigma_j/a_j)*Qh_j; prefix unchanged')


def full_matrix(layer):return finite(layer.weight_u.detach().cpu().double()@layer.weight_v.detach().cpu().double())


def matrix_summary(layer,minimum):
    matrix=full_matrix(layer);singular=torch.linalg.svdvals(matrix);energy=finite(singular.square())
    tail=float(energy[minimum:].sum());total=float(energy.sum())
    return dict(fc1_matrix_frobenius_norm=float(matrix.norm()),spectral_energy_within_min_rank=float(energy[:minimum].sum()),
        spectral_energy_outside_min_rank=tail,spectral_tail_energy_fraction=tail/total if total else 0.,
        spectrum_definition='re-SVD actual U@V in private float64 CPU copy; not fixed-slot partition norms')


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})


@contextmanager
def read_only(model):
    state={n:t.detach().clone() for n,t in model.state_dict().items()};modes=[(m,m.training) for m in model.modules()]
    flags=[(p,p.requires_grad) for p in model.parameters()]
    with protected_rng([model]):
        try:yield
        finally:
            model.load_state_dict(state,strict=True)
            for m,mode in modes:m.training=mode
            for p,flag in flags:p.requires_grad_(flag)


@torch.no_grad()
def evaluate(model,data,batch,device):
    with read_only(model):
        model.eval();outputs=[];labels=[];ce_sum=0.
        generator=torch.Generator().manual_seed(92841)
        loader=make_data_loader(data,batch,shuffle=False,drop_last=False,num_workers=0,generator=generator)
        for x,y in loader:
            out=finite(model(x.to(device)));ce_sum+=float(F.cross_entropy(out,y.to(device),reduction='sum'))
            outputs.append(out.cpu());labels.append(y.cpu())
        logits=torch.cat(outputs);y=torch.cat(labels);correct=int((logits.argmax(1)==y).sum())
        return dict(correct=correct,samples=len(y),accuracy=correct/len(y),ce=ce_sum/len(y)),logits


@torch.no_grad()
def truncated_fc1(model,minimum):
    """Best rank-minimum fc1 ONLY; direct dense inference, all other layers unchanged."""
    from torch import nn
    with protected_rng([model]):
        matrix=full_matrix(model.fc1);p,s,qh=torch.linalg.svd(matrix,full_matrices=False)
        weight=(p[:,:minimum]*s[:minimum])@qh[:minimum,:]
        result=copy.deepcopy(model)
        dense=nn.Linear(model.fc1.in_features,model.fc1.out_features,bias=model.fc1.bias is not None).to(model.fc1.weight_u)
        dense.weight.copy_(weight.to(dense.weight))
        if dense.bias is not None:dense.bias.copy_(model.fc1.bias)
        result.fc1=dense;result._rebuild_base()
        return result


class Indexed(Dataset):
    def __init__(self,data):self.data=data
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return *self.data[i],i


class EpochLoader:
    """Same private seed/shuffle as clientMultiGroupProj; IDs do not drive sampling."""
    def __init__(self,data,args,cid,loop,observer=None):
        self.epoch=0;self.batches=[];self.observer=observer
        generator=torch.Generator().manual_seed(proto.loader_seed(args,cid,loop))
        self.loader=make_data_loader(Indexed(data),args.batch_size,shuffle=True,drop_last=False,num_workers=0,generator=generator)
    def __iter__(self):
        self.epoch+=1
        for batch,(x,y,indices) in enumerate(self.loader,1):
            self.batches.append(dict(epoch=self.epoch,batch=batch,indices=indices.tolist()))
            yield x,y
        if self.observer is not None:self.observer.end_epoch(self.epoch)


class Observer:
    def __init__(self,client,evidence,minimum,enabled=True):
        self.client=client;self.model=client.model_set[0];self.optimizer=client.optimizer_set[0]
        self.minimum=minimum;self.evidence=evidence;self.enabled=enabled;self.steps=0
        self.rows=[];self.gradients=[];self.signal=[];self.norms={};self.seconds=0.;self.handles=[]
        self.parameters=list(self.model.named_parameters());self.capture=None;self.probing=False
        if enabled:
            u=self.model.fc1.weight_u.detach().cpu().double();v=self.model.fc1.weight_v.detach().cpu().double()
            self.initial_front=u[:,:minimum]@v[:minimum,:];self.initial_tail=u[:,minimum:]@v[minimum:,:]
    def snapshot(self,stage):
        if not self.enabled:return
        tick=time.perf_counter();u=self.model.fc1.weight_u.detach().cpu().double();v=self.model.fc1.weight_v.detach().cpu().double()
        self.rows.append(dict(stage=stage,optimizer_step=self.steps,
            fixed_slot_tail_product_change_frobenius=float((u[:,self.minimum:]@v[self.minimum:,:]-self.initial_tail).norm()),
            fixed_slot_front_product_change_frobenius=float((u[:,:self.minimum]@v[:self.minimum,:]-self.initial_front).norm()),
            fixed_slot_observation_dtype='float64 CPU products of ACTUAL current parameter values; fixed download slot IDs',
            **matrix_summary(self.model.fc1,self.minimum)))
        self.seconds+=time.perf_counter()-tick
    def end_epoch(self,epoch):
        if epoch in {1,self.client.local_epochs}:self.snapshot(f'epoch_{epoch}')
    def __enter__(self):
        self.original_step=self.optimizer.step;self.step_was_instance_attribute='step' in self.optimizer.__dict__
        self.original_loss=self.client.loss
        if not self.enabled:
            def count(*a,**kw):self.steps+=1;return self.original_step(*a,**kw)
            self.optimizer.step=count
            return self
        def input_hook(module,inputs):
            if self.steps==0:self.capture=dict(inputs=inputs[0].detach().clone())
        def output_hook(module,inputs,output):
            if self.steps==0:self.capture['output']=output
        self.handles.extend([self.model.fc1.register_forward_pre_hook(input_hook),self.model.fc1.register_forward_hook(output_hook)])
        for name,p in self.parameters:
            if not p.requires_grad:continue
            def hook(g,name=name):
                if self.probing:return
                self.norms[name]=g.detach().norm(2)
                if self.steps==0 and name in ('fc1.weight_u','fc1.weight_v'):
                    part=g[:,self.minimum:] if name.endswith('_u') else g[self.minimum:,:]
                    self.norms[name+'_tail_pre']=part.detach().norm(2)
            self.handles.append(p.register_hook(hook))
        def loss(logits,labels):
            ce=self.original_loss(logits,labels)
            if self.steps==0:
                tick=time.perf_counter();before=[p.grad for _,p in self.parameters]
                self.probing=True
                try:dy=torch.autograd.grad(ce,self.capture['output'],retain_graph=True)[0]
                finally:self.probing=False
                if any(p.grad is not grad for (_,p),grad in zip(self.parameters,before)):
                    raise RuntimeError('CE probe changed actual .grad')
                G=finite(dy.detach().cpu().double().T@self.capture['inputs'].cpu().double())
                m=self.minimum;r=self.model.fc1.rank;p=self.evidence['P'].double();qh=self.evidence['Qh'].double()
                self.signal.append(dict(stage='first_training_batch_CE_only',ce=float(ce.detach()),
                    CE_restored_weight_gradient_frobenius=float(G.norm()),
                    CE_G_Q_download_tail_frobenius=float((G@qh[m:r,:].T).norm()),
                    CE_G_transpose_P_download_tail_frobenius=float((G.T@p[:,m:r]).norm()),
                    CE_predicted_U_tail_gradient_frobenius=float((G@self.model.fc1.weight_v[m:,:].detach().cpu().double().T).norm()),
                    CE_predicted_V_tail_gradient_frobenius=float((self.model.fc1.weight_u[:,m:].detach().cpu().double().T@G).norm()),
                    samples=len(labels),probe='autograd.grad CE wrt existing fc1 output, retain_graph; G=dy.T@input; no .backward/.grad writes'))
                self.capture=None;self.seconds+=time.perf_counter()-tick
            return ce
        self.client.loss=loss
        def step(*a,**kw):
            tick=time.perf_counter()
            norms=[self.norms[n] for n,p in self.parameters if p.grad is not None]
            pre=torch.stack(norms).norm(2);finite(pre);cap=self.client.args.local_grad_clip
            coefficient=float((cap/(pre+1e-6)).clamp(max=1)) if cap and cap>0 else 1.
            if self.steps==0:
                u,v=self.model.fc1.weight_u,self.model.fc1.weight_v
                self.gradients.append(dict(stage='first_actual_SGD_step_total_CE_F_proto',
                    pre_clip_total_grad_norm=float(pre),clip_coefficient=coefficient,
                    post_clip_total_grad_norm=float(torch.stack([p.grad.detach().norm(2) for _,p in self.parameters if p.grad is not None]).norm(2)),
                    U_tail_grad_norm_pre_clip=float(self.norms['fc1.weight_u_tail_pre']),
                    V_tail_grad_norm_pre_clip=float(self.norms['fc1.weight_v_tail_pre']),
                    U_tail_grad_norm_post_clip=float(u.grad[:,self.minimum:].detach().norm()),
                    V_tail_grad_norm_post_clip=float(v.grad[self.minimum:,:].detach().norm()),
                    U_tail_step_norm=float((self.client.args.local_learning_rate*u.grad[:,self.minimum:]).detach().norm()),
                    V_tail_step_norm=float((self.client.args.local_learning_rate*v.grad[self.minimum:,:]).detach().norm())))
            self.norms.clear();self.seconds+=time.perf_counter()-tick
            result=self.original_step(*a,**kw);self.steps+=1
            if self.steps==1:self.snapshot('step_1')
            return result
        self.optimizer.step=step
        return self
    def __exit__(self,*exc):
        if self.step_was_instance_attribute:self.optimizer.step=self.original_step
        else:del self.optimizer.step
        self.client.loss=self.original_loss
        for h in self.handles:h.remove()
        self.capture=None;self.norms.clear()


def local_client(model,mother,args,cid,bank,receipt,data,folder):
    from utils.cnn512_reporting import model_digest
    level=args.multi_svd_effective_assignment[cid];rate=args.multi_svd_effective_ratios[level]
    c=SimpleNamespace(id=cid,K=1,args=args,device=args.device,model=model,model_set=[model],
        optimizer_set=[torch.optim.SGD(model.parameters(),lr=args.local_learning_rate,momentum=0.,weight_decay=0.)],
        loss=torch.nn.CrossEntropyLoss(),download_round=args.offline_loop_round,upload_round=None,
        local_epochs=args.local_epochs,batch_size=args.batch_size,dataset=args.dataset,train_samples=len(data),rank_rate=rate,capacity_level=level,
        avg_proto_bank=bank,avg_proto_receipt=receipt,download_servers=[mother],download_starts=[None],
        full_uploads=[None],capacity_metadata=[svd.model_capacity(model,dict(mother.named_parameters()))],
        save_folder_name=str(folder),train_time_cost=dict(num_rounds=0,total_cost=0.),
        avg_proto_download_context=dict(server_sha256=model_digest(mother),
            download_sha256=tensor_fingerprint(model.state_dict()),initial_sha256=args.initial_sha256,data_fingerprint=args.data_fingerprint))
    return c


def train_branch(model,mother,args,cid,bank,receipt,data,folder,evidence,minimum,diagnostics=True,trainer=None):
    c=local_client(model,mother,args,cid,bank,receipt,data,folder)
    observer=Observer(c,evidence,minimum,diagnostics);loader=EpochLoader(data,args,cid,args.offline_loop_round,observer)
    c.load_train_data=lambda:loader
    tick=time.perf_counter()
    with observer:(trainer or proto.train)(c)
    expected=args.local_epochs*math.ceil(len(data)/args.batch_size)
    if observer.steps!=expected or len(loader.batches)!=expected or c.train_stats[0]['processed_train_samples']!=len(data)*args.local_epochs:
        raise RuntimeError('Offline local epoch/step/sample budget mismatch')
    result=dict(training=c.train_stats[0],updates=observer.rows,first_step=observer.gradients,CE_signal=observer.signal,
        batches=loader.batches,diagnostic_seconds=observer.seconds,training_wall_including_online_diagnostics_seconds=time.perf_counter()-tick,
        final_rng_sha256=rng_fingerprint(rng_snapshot(model)),final_model_sha256=tensor_fingerprint(model.state_dict()),
        local_proto_packet_sha256=packet_sha(c.avg_proto_local))
    return result,c


def resolve_source(cli):
    if cli.server_file:path=Path(cli.server_file).expanduser().resolve()
    else:
        root=Path(cli.source_run).expanduser().resolve()
        candidates=([root/'Server_model_0.pt'] if (root/'Server_model_0.pt').is_file() else
            sorted((root/'final_models').rglob('Server_model_0.pt')))
        if len(candidates)!=1:
            raise FileNotFoundError(f'Expected exactly ONE final server in {root}/final_models, got {len(candidates)}; use --server-file explicitly')
        path=candidates[0]
    for file in (path,path.parent/'manifest.json',path.parent/'Global_class_sample_latest_generated.packet'):
        if not file.is_file():raise FileNotFoundError(f'Required same-run FA02 input missing: {file}')
    return path


def load_source(cli):
    from flcore.trainmodel.models import Hyper_CNN_512
    from utils.cnn512_reporting import model_digest
    path=resolve_source(cli);manifest_path=path.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    backend=manifest.get('environment',{})
    backend_keys=('deterministic','cudnn_deterministic','cudnn_benchmark','matmul_tf32','cudnn_tf32')
    if any(type(backend.get(k)) is not bool for k in backend_keys):raise ValueError('Missing/invalid original Torch backend configuration')
    torch.use_deterministic_algorithms(backend['deterministic'])
    torch.backends.cudnn.deterministic=backend['cudnn_deterministic'];torch.backends.cudnn.benchmark=backend['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=backend['matmul_tf32'];torch.backends.cudnn.allow_tf32=backend['cudnn_tf32']
    original=copy.deepcopy(manifest['effective']);args=SimpleNamespace(**copy.deepcopy(original))
    source=cli.source_round
    if type(source) is not int or source<1:raise ValueError('source-round must be positive')
    if (manifest.get('algorithm')!='FedMultiGroupProj' or manifest.get('server_model_count')!=1
        or manifest.get('actual_train_count')!=source or manifest.get('aggregation_mode')!='group_avg_only'
        or not args.multi_avg_proto_ablation or not args.multi_svd
        or args.multi_proto_mode!='global_last_epoch' or args.multi_proto_aggregation!='class_sample'
        or args.multi_proto_target_type!='raw_mean' or args.multi_proto_feature!='base_output' or args.multi_proto_loss!='mse'
        or args.multi_proto_lambda<=0 or args.target_head_mode!='shared' or args.target_local_schedule!='joint'
        or getattr(args,'multi_svd_aggregation','full_avg')!='full_avg'):
        raise ValueError('Need completed FA02 K1 full_avg + last_epoch/class_sample/raw_mean/base_output MSE run')
    if (args.is_regular!=1 or getattr(args,'multi_kd_mode','none')!='none' or getattr(args,'target_prox_mu',0)
        or getattr(args,'target_head_l2_lambda',0) or getattr(args,'target_z2_lambda',0)
        or getattr(args,'multi_head_mode','shared')!='shared' or getattr(args,'multi_projection_scope','full')!='full'
        or getattr(args,'target_c0_replay_mode','none')!='none' or getattr(args,'target_curriculum_mode','none')!='none'):
        raise ValueError('Source must be ordinary CE + original F + class_sample MSE, without other training modules')
    if manifest['optimizer'].get('momentum')!=0 or manifest['optimizer'].get('weight_decay')!=0:
        raise ValueError('Source optimizer must be original momentum=0, weight_decay=0 SGD')
    mother,payload=svd.load_checkpoint(path,'cpu')
    if (payload.get('role')!='post_aggregation_full_server' or payload.get('completed_train_count')!=source
        or payload.get('model_id')!=0 or mother.ratio_LR!=1.
        or payload.get('svd_aggregation',{}).get('method')!='full_avg'
        or not isinstance(mother.fc1,torch.nn.Linear) or not isinstance(mother.fc2,torch.nn.Linear)
        or not isinstance(mother.conv2,torch.nn.Conv2d)):
        raise ValueError('Input must be complete post-Avg SERVER, not an R101 client prediction/upload')
    for name,value in mother.state_dict().items():
        if not torch.equal(value,payload['state_dict'][name]) or not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite or inconsistent alias in source checkpoint: {name}')
    if payload['model_kwargs']!=manifest['model_kwargs'] or payload['model_family']!=manifest['model_family']:
        raise ValueError('Source checkpoint/manifest model mismatch')
    for key in ('svd_initialization','capacity_protocol'):
        manifest_key='svd_capacity_protocol' if key=='capacity_protocol' else key
        if payload.get(key)!=manifest.get(manifest_key):raise ValueError(f'Source {key} mismatch')
    if payload.get('initial_parameter_sha256')!=manifest.get('initial_parameter_sha256'):
        raise ValueError('Source checkpoint/manifest initialization fingerprint mismatch')
    packet_path=path.parent/'Global_class_sample_latest_generated.packet';packet=packet_path.read_bytes()
    bank,meta=decode_packet(packet,'single_avg_library_v2');proto._validate_library(bank,meta,args,source)
    expected_ids={str(i) for i in range(args.num_clients)}
    post_hashes=meta.get('source_post_local_model_sha256_by_client',{})
    if (set(post_hashes)!=expected_ids or any(not isinstance(v,str) or len(v)!=64 for v in post_hashes.values())
        or any(value.requires_grad for value in bank.values())):
        raise ValueError('Missing post-local source identities or non-frozen prototype library')
    declaration=manifest['single_model_Avg_prototype_ablation']['latest_generated_library']
    if declaration!=dict(meta,sha256=packet_sha(packet)):
        raise ValueError('Prototype is not this run\'s manifest-declared latest generated library')
    if meta.get('initial_sha256')!=payload.get('initial_parameter_sha256'):
        raise ValueError('Prototype/source initialization fingerprint mismatch')
    if bank['prototypes'].shape[1]!=mother.fc3.in_features:raise ValueError('Prototype feature dimension mismatch')
    # Verify the claimed S0 using the source model constructor/initialization.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(original['seed'])
        kwargs=copy.deepcopy(payload['model_kwargs']);mode=payload['svd_initialization']['mode']
        kwargs['ratio_LR']=payload['svd_initialization']['configured_rank_rate'] if mode=='low_rank_recovered' else 1.
        initial=Hyper_CNN_512(**kwargs)
        if mode=='low_rank_recovered':initial.recover_larger_model();initial.ratio_LR=1.
        if model_digest(initial)!=payload['initial_parameter_sha256']:raise ValueError('Initial S0 cannot be reproduced from source seed/model/init protocol')
    args.data_root=cli.data_root;args.dataset=cli.dataset;args.data_subdir=cli.data_subdir;args.data_formats='npz_only'
    if (cli.dataset!='Cifar100' or original['dataset']!='Cifar100' or args.num_classes!=100):
        raise ValueError('Offline v1 supports original CIFAR100/RGB32 Hyper_CNN only')
    args.data_expected_partition='pat'
    data=audit_image_clients(args)
    if data['config'].get('partition')!='pat':raise ValueError('Expected original pat data, not a directory-name guess')
    keys=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')
    fingerprint=digest_json(dict(config_sha256=data['config_sha256'],files=[{k:r[k] for k in keys} for r in data['files']]))
    if fingerprint!=meta['data_fingerprint']:raise ValueError('Actual whole dataset fingerprint differs from source prototype run')
    if (data['config_sha256']!=manifest['data_manifest']['config_sha256'] or
        [{k:r[k] for k in keys} for r in data['files']]!=[{k:r[k] for k in keys} for r in manifest['data_manifest']['files']]):
        raise ValueError('Original run data manifest does not match actual shards')
    args.offline_loop_round=source;args.data_fingerprint=fingerprint;args.initial_sha256=payload['initial_parameter_sha256']
    for requested,effective in [('epochs','local_epochs'),('batch_size','batch_size'),('test_batch_size','test_batch_size'),
        ('lr','local_learning_rate'),('f_lambda','regular_lamda'),('proto_lambda','multi_proto_lambda'),
        ('grad_clip','local_grad_clip'),('seed','seed')]:
        value=getattr(cli,requested)
        if value is not None:setattr(args,effective,value)
    for field in ('local_epochs','batch_size','test_batch_size'):
        if type(getattr(args,field)) is not int or getattr(args,field)<1:raise ValueError(f'{field} must be a positive integer')
    for field in ('local_learning_rate','regular_lamda','multi_proto_lambda','local_grad_clip'):
        value=getattr(args,field)
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or field=='local_learning_rate' and value==0:
            raise ValueError(f'Invalid {field}')
    if args.multi_proto_lambda==0:raise ValueError('This paired frozen-library diagnostic requires proto-lambda > 0; use the original no-prototype trainer for CE/F-only experiments')
    if type(args.seed) is not int or not 0<=args.seed<2**32:raise ValueError('Invalid seed')
    args.device='cuda:0' if cli.device=='cuda' else 'cpu';args.multi_svd_diagnostics=0
    args.multi_proto_diagnostics=1;args.multi_helper_head_update='train'
    if cli.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Select exactly the requested single GPU before Torch import, or use --device cpu')
        torch.cuda.set_device(0)
    rates=args.multi_svd_effective_ratios;assignment=args.multi_svd_effective_assignment
    if rates!=payload['capacity_protocol']['rank_rates'] or assignment!=payload['capacity_protocol']['client_level_ids']:
        raise ValueError('Effective capacity assignment differs from source checkpoint')
    if cli.client_ids:
        try:ids=[int(s) for s in cli.client_ids.split(',')]
        except ValueError:raise ValueError('client-ids must be comma-separated integers')
    else:
        try:requested=[float(s) for s in cli.client_rates.split(',')]
        except ValueError:raise ValueError('client-rates must be comma-separated finite rank rates')
        if not requested or any(not math.isfinite(r) or r<=0 or r>=1 for r in requested):raise ValueError('Invalid client-rates')
        if len(set(requested))!=len(requested) or any(not any(math.isclose(r,v,rel_tol=0,abs_tol=1e-12) for v in rates) for r in requested):
            raise ValueError('Requested client capacity absent/duplicate in source assignment')
        ids=[i for i,level in enumerate(assignment) if any(math.isclose(rates[level],r,rel_tol=0,abs_tol=1e-12) for r in requested)]
    if not ids or len(ids)!=len(set(ids)) or any(i<0 or i>=args.num_clients for i in ids):raise ValueError('Invalid/empty/duplicate client selection')
    minimum=factor_rank(mother,min(rates[level] for level in assignment))
    if any(factor_rank(mother,rates[assignment[i]])<=minimum for i in ids):raise ValueError('Selected client has no rank tail above minimum capacity')
    receipt=dict(meta,sha256=packet_sha(packet),packet_bytes=len(packet))
    source_record=dict(server_file=str(path),server_file_sha256=sha256(path),server_state_sha256=model_digest(mother),
        manifest_file=str(manifest_path),manifest_file_sha256=sha256(manifest_path),prototype_file=str(packet_path),
        prototype_packet_sha256=packet_sha(packet),prototype_metadata=meta,source_completed_train_count=source,
        offline_completed_train_count=source+1,offline_loop_round=source,initial_sha256=args.initial_sha256,
        data_fingerprint=fingerprint,source_config=original,effective_training=dict(epochs=args.local_epochs,batch_size=args.batch_size,
            lr=args.local_learning_rate,momentum=0.,weight_decay=0.,is_regular=args.is_regular,f_lambda=args.regular_lamda,
            grad_clip=args.local_grad_clip,proto_lambda=args.multi_proto_lambda,seed=args.seed),
        selected_client_ids=sorted(ids),minimum_actual_fc1_rank=minimum,
        original_backend_settings={k:backend[k] for k in backend_keys},
        packet_source_server_note='Packet source_server_sha256 names pre-R101 SGD server S100; it must NOT equal final S101. Latest-packet binding is checked against same-run manifest.')
    return mother.to(args.device),payload,args,bank,receipt,data,source_record


def compare_tensors(a,b,atol,rtol,label):
    if a.shape!=b.shape or not torch.allclose(a,b,atol=atol,rtol=rtol):raise ValueError(f'A/B initial equivalence failed: {label}')
    difference=(a.detach().cpu().double()-b.detach().cpu().double()).abs()
    return dict(max_absolute_error=float(difference.max()),rms_error=float(difference.square().mean().sqrt()),atol=atol,rtol=rtol)


def initial_loss(model,client,data,bank,args):
    with read_only(model):
        model.train();loader=EpochLoader(data,args,client.id,args.offline_loop_round)
        x,y=next(iter(loader));x,y=x.to(args.device),y.to(args.device)
        from flcore.clients.clientTargetProj import clientTargetProj
        with Features(model,'base_output') as f:
            ce,regular=clientTargetProj._objective(client,model,x,y)
            raw,valid=proto.global_mse(f.get(),y,bank['prototypes'],bank['valid'])
        return torch.stack([ce,regular,raw,ce+regular+args.multi_proto_lambda*raw]).detach().cpu(),valid


def capacity_summary(pairs):
    result=[]
    for rate in sorted({r['rank_rate'] for r in pairs}):
        rows=[r for r in pairs if r['rank_rate']==rate];total=sum(r['test_samples'] for r in rows)
        row=dict(rank_rate=rate,clients=[r['client_id'] for r in rows],test_samples=total)
        for branch in ('A','B'):
            row[f'{branch}_test_accuracy']=sum(r[f'{branch}_test_correct'] for r in rows)/total
            row[f'{branch}_test_CE']=sum(r[f'{branch}_test_CE']*r['test_samples'] for r in rows)/total
            row[f'{branch}_fc1_truncated_test_accuracy']=sum(r[f'{branch}_fc1_truncated_test_correct'] for r in rows)/total
            updates=[r.get(f'{branch}_epoch_final_tail_product_change') for r in rows]
            row[f'{branch}_mean_client_tail_product_change']=sum(updates)/len(updates) if all(x is not None for x in updates) else None
        row['B_minus_A_test_accuracy']=row['B_test_accuracy']-row['A_test_accuracy'];result.append(row)
    return result


def run(cli):
    if cli.threads<1:raise ValueError('threads must be positive')
    for name in ('init_atol','init_rtol'):
        if not math.isfinite(getattr(cli,name)) or getattr(cli,name)<0:raise ValueError('Invalid initial-equivalence tolerance')
    output=Path(cli.output_dir).expanduser().resolve()
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    source_path=resolve_source(cli);data_path=(Path(cli.data_root)/cli.dataset/cli.data_subdir).resolve()
    if output==data_path or data_path in output.parents or output in data_path.parents or output==source_path.parent or source_path.parent in output.parents:
        raise ValueError('Diagnostic output must be separate from source checkpoint and data directories')
    torch.set_num_threads(cli.threads)
    mother,payload,args,bank,receipt,data_manifest,source=load_source(cli)
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(role=f'offline_R{cli.source_round+1}_local_training_from_R{cli.source_round}_server_no_aggregation',
        requested=vars(cli),source=source,environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),
        interpretation='Actual clipped SGD product changes decide trainability; CE projection alone is not evidence of learning/classification value',
        collection='Original last-epoch collection remains inside reused trainer; generated offline packets discarded, never replace the frozen input library',
        default_artifacts='diagnostic tables only; no model or per-sample feature export',
        units='accuracy fraction; CE nats/sample; updates Frobenius L2; tail spectrum singular-value SQUARED energy',
        optimizer='NEW SGD per branch, lr from effective config, momentum=0, weight_decay=0; original CE+F+raw global MSE, clipping unchanged',
        initial_equivalence_tolerance=dict(atol=cli.init_atol,rtol=cli.init_rtol),initial_checks=[],status='running')
    write_json(output/'manifest.json',manifest);write_json(output/'data_manifest.json',data_manifest)
    endpoints=[];updates=[];signals=[];gradients=[];training=[];batches=[];pairs=[]
    input_hashes={source_path:sha256(source_path),source_path.parent/'Global_class_sample_latest_generated.packet':receipt['sha256'],
        source_path.parent/'manifest.json':sha256(source_path.parent/'manifest.json')}
    bank_hash=tensor_fingerprint(bank)
    random.seed(args.seed);np.random.seed(args.seed);torch.random.default_generator.manual_seed(args.seed)
    if args.device.startswith('cuda'):torch.cuda.default_generators[0].manual_seed(args.seed)
    try:
        for cid in source['selected_client_ids']:
            rate=args.multi_svd_effective_ratios[args.multi_svd_effective_assignment[cid]]
            train_data=read_image_client(args,cid,True);test_data=read_image_client(args,cid,False)
            print(f'\n========== offline R{cli.source_round+1} client={cid}, rate={rate} ========== ',flush=True)
            tick=time.perf_counter();A,evidence=download_once(mother,rate);B,gauge=rescale_tail(A,evidence,source['minimum_actual_fc1_rank'])
            minimum=source['minimum_actual_fc1_rank'];base_rng=rng_snapshot(A)
            check=dict(client_id=cid,rank_rate=rate,**gauge,download_seconds=time.perf_counter()-tick,
                restored_fc1=compare_tensors(full_matrix(A.fc1),full_matrix(B.fc1),cli.init_atol,cli.init_rtol,'fc1 restored product'))
            for name,value in A.state_dict().items():
                other=B.state_dict()[name]
                if 'fc1.weight_' not in name and not name.startswith('base.7.weight_') and not torch.equal(value,other):
                    raise RuntimeError(f'B changed a non-fc1-factor tensor: {name}')
            if not torch.equal(A.fc1.weight_u[:,:minimum],B.fc1.weight_u[:,:minimum]) or not torch.equal(A.fc1.weight_v[:minimum,:],B.fc1.weight_v[:minimum,:]):
                raise RuntimeError('B changed fc1 prefix')
            temporary_results={};initial={}
            for branch,model in [('A',A),('B',B)]:
                initial[branch]={}
                for split,dataset in [('train',train_data),('test',test_data)]:
                    metrics,logits=evaluate(model,dataset,args.test_batch_size,args.device);initial[branch][split]=(metrics,logits)
                    endpoints.append(dict(client_id=cid,rank_rate=rate,branch=branch,stage='download_before_training',split=split,**metrics))
            for split in ('train','test'):
                check[split+'_logits']=compare_tensors(initial['A'][split][1],initial['B'][split][1],cli.init_atol,cli.init_rtol,split+' logits')
                if not torch.equal(initial['A'][split][1].argmax(1),initial['B'][split][1].argmax(1)):
                    raise ValueError('A/B initial argmax differs despite small numerical errors; report/review tolerances before training')
            with tempfile.TemporaryDirectory(prefix='transient_training_',dir=output) as temporary:
                first_losses=[]
                for branch,model in [('A',A),('B',B)]:
                    probe_client=local_client(model,mother,args,cid,bank,receipt,train_data,temporary)
                    losses,valid=initial_loss(model,probe_client,train_data,bank,args);first_losses.append(losses)
                    check[branch+'_initial_loss_CE_F_rawMSE_total']=losses.tolist();check['initial_valid_proto_samples']=valid
                check['first_batch_objective']=compare_tensors(*first_losses,cli.init_atol,cli.init_rtol,'initial CE/F/MSE objective')
                manifest['initial_checks'].append(check)
                for branch,model in [('A',A),('B',B)]:
                    restore_rng(base_rng)
                    result,c=train_branch(model,mother,args,cid,bank,receipt,train_data,temporary,evidence,minimum,bool(cli.diagnostics))
                    if tensor_fingerprint(bank)!=bank_hash or any(value.grad is not None for value in bank.values()):
                        raise RuntimeError('Offline branch changed the frozen prototype bank')
                    temporary_results[branch]=result
                    identity=dict(client_id=cid,rank_rate=rate,actual_fc1_rank=model.fc1.rank,minimum_actual_fc1_rank=minimum,branch=branch,
                        source_completed_train_count=cli.source_round,offline_completed_train_count=cli.source_round+1,loop_round=args.offline_loop_round)
                    for key,table in [('updates',updates),('CE_signal',signals),('first_step',gradients)]:table.extend(dict(identity,**row) for row in result[key])
                    for row in result['batches']:
                        batch_copy=dict(row);indices=batch_copy.pop('indices')
                        batches.append(dict(identity,**batch_copy,samples=len(indices),sample_indices_sha256=digest_json(indices)))
                    training.append(dict(identity,**result['training'],
                        all_batch_indices_sha256=digest_json(result['batches']),final_rng_sha256=result['final_rng_sha256'],
                        final_model_sha256=result['final_model_sha256'],local_proto_packet_sha256=result['local_proto_packet_sha256'],
                        online_diagnostic_seconds=result['diagnostic_seconds'],
                        training_wall_including_online_diagnostics_seconds=result['training_wall_including_online_diagnostics_seconds']))
                    for split,dataset in [('train',train_data),('test',test_data)]:
                        metrics,_=evaluate(model,dataset,args.test_batch_size,args.device)
                        endpoints.append(dict(identity,stage=f'epoch_{args.local_epochs}',split=split,**metrics))
                    tick=time.perf_counter();truncated=truncated_fc1(model,minimum)
                    for split,dataset in [('train',train_data),('test',test_data)]:
                        metrics,_=evaluate(truncated,dataset,args.test_batch_size,args.device)
                        endpoints.append(dict(identity,stage=f'epoch_{args.local_epochs}_fc1_only_rank_{minimum}',split=split,**metrics))
                    training[-1]['truncated_fc1_evaluation_seconds']=time.perf_counter()-tick;del truncated
                    if cli.save_models:
                        folder=output/'models';folder.mkdir(exist_ok=True)
                        checkpoint=svd.client_payload(c,0,offline_diagnostic=dict(identity,prototype_packet_sha256=receipt['sha256'],
                            source_server_file_sha256=source['server_file_sha256'],gauge=gauge if branch=='B' else None))
                        torch.save(checkpoint,folder/f'Client_{cid}_{branch}_offline_R{cli.source_round+1}.pt')
                if temporary_results['A']['batches']!=temporary_results['B']['batches']:raise RuntimeError('A/B batch order differs')
                if temporary_results['A']['final_rng_sha256']!=temporary_results['B']['final_rng_sha256']:raise RuntimeError('A/B ending training RNG differs')
            row=dict(client_id=cid,rank_rate=rate,actual_fc1_rank=A.fc1.rank,minimum_actual_fc1_rank=minimum,
                test_samples=len(test_data),batches_per_branch=len(temporary_results['A']['batches']),
                SGD_steps_per_branch=args.local_epochs*math.ceil(len(train_data)/args.batch_size),
                training_sample_visits_per_branch=args.local_epochs*len(train_data),batch_order_equal=True,training_RNG_equal=True)
            for branch in ('A','B'):
                endpoint=next(r for r in endpoints if r['client_id']==cid and r['branch']==branch and r['stage']==f'epoch_{args.local_epochs}' and r['split']=='test')
                cut=next(r for r in endpoints if r['client_id']==cid and r['branch']==branch and r['stage']==f'epoch_{args.local_epochs}_fc1_only_rank_{minimum}' and r['split']=='test')
                change=next((r for r in updates if r['client_id']==cid and r['branch']==branch and r['stage']==f'epoch_{args.local_epochs}'),None)
                row.update({f'{branch}_test_correct':endpoint['correct'],f'{branch}_test_accuracy':endpoint['accuracy'],
                    f'{branch}_test_CE':endpoint['ce'],f'{branch}_fc1_truncated_test_correct':cut['correct'],
                    f'{branch}_fc1_truncation_accuracy_drop':endpoint['accuracy']-cut['accuracy'],
                    f'{branch}_epoch_final_tail_product_change':change['fixed_slot_tail_product_change_frobenius'] if change else None})
            row['B_minus_A_test_accuracy']=row['B_test_accuracy']-row['A_test_accuracy'];pairs.append(row)
            if cli.historical_client_dir:
                historical=Path(cli.historical_client_dir)/f'Client_{cid}_model_0.pt'
                history,hp=svd.load_checkpoint(historical,args.device)
                if (hp.get('completed_train_count')!=cli.source_round or hp.get('client_id')!=cid or hp.get('model_id')!=0
                    or hp.get('role')!='ordinary_post_local_low_rank' or history.ratio_LR!=rate
                    or hp.get('initial_parameter_sha256')!=args.initial_sha256):raise ValueError('Historical client identity/capacity/source mismatch')
                usage=hp.get('single_avg_prototype_usage',{}).get('downloaded_model',{})
                if usage.get('data_fingerprint')!=args.data_fingerprint:raise ValueError('Historical checkpoint data fingerprint mismatch')
                metrics,_=evaluate(history,test_data,args.test_batch_size,args.device)
                endpoints.append(dict(client_id=cid,rank_rate=rate,branch='historical_R'+str(cli.source_round),stage='original_post_local_not_download_anchor',split='test',**metrics));del history
            for name,table in [('client_pairs',pairs),('endpoints',endpoints),('fixed_slot_updates',updates),('first_batch_CE_signal',signals),
                ('first_step_gradients',gradients),('local_training',training),('batch_order',batches)]:
                write_json(output/(name+'.json'),table);write_csv(output/(name+'.csv'),table)
            write_json(output/'capacity_summary.json',capacity_summary(pairs));write_csv(output/'capacity_summary.csv',capacity_summary(pairs))
            write_json(output/'manifest.json',manifest)
            del A,B,evidence,initial,temporary_results,c
        if any(sha256(path)!=before for path,before in input_hashes.items()):raise RuntimeError('Input artifact changed')
        manifest.update(status='complete',prototype_frozen_sha256=receipt['sha256'],client_pair_count=len(pairs),
            prototype_frozen_tensor_sha256=bank_hash,server_aggregation_performed=False,diagnostics_only=not cli.save_models)
        write_json(output/'manifest.json',manifest)
        print(json.dumps(capacity_summary(pairs),indent=2));print('Offline diagnostics:',output)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(output/'manifest.json',manifest)
        raise
