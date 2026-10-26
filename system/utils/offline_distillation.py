"""Frozen-teacher, fixed-budget offline CE/F/KL training. No federation calls."""
from contextlib import contextmanager
import copy
import csv
import json
import math
import os
import hashlib
import pickle
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.data_loading import make_data_loader
from common.experiment import rng_state, set_rng, tensor_digest, write_json
from flcore.clients.clientTargetProj import clientTargetProj
from utils.head_diagnostic import evaluate
from utils.head_snapshot import tensor_fingerprint, backend_settings
from utils.offline_distillation_io import digest_json

MODES=('continue_ce','kd_mean','kl_dp')


@contextmanager
def preserved_rng():
    state=rng_state()
    try: yield
    finally: set_rng(state)


def training_rng_digest():
    state=rng_state()
    components=dict(python=hashlib.sha256(pickle.dumps(state['python'])).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(state['numpy'])).hexdigest(),torch=tensor_digest(state['torch']),
        cuda=[tensor_digest(s) for s in state['cuda']])
    return digest_json(components)


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


class IndexedData(Dataset):
    def __init__(self,data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,i): return *self.data[i],i


def select_teachers(logits,labels):
    """N x 3 x C original logits; first maximum gives A0, B0, C0 tie order."""
    with torch.no_grad():
        if logits.ndim!=3 or logits.shape[1]!=3 or len(labels)!=len(logits) or not torch.isfinite(logits).all():
            raise ValueError('Expected finite N x 3 x C teacher logits')
        probabilities=logits.detach().softmax(-1)
        true=probabilities.gather(2,labels[:,None,None].expand(-1,3,1)).squeeze(2)
        return true.argmax(1),true


def teacher_target(logits,mode,temperature,selected=None):
    with torch.no_grad():
        if mode=='kd_mean': return (logits.detach()[:,1:]/temperature).softmax(-1).mean(1)
        if mode!='kl_dp' or selected is None: raise ValueError('Expected fixed KL+DP selection')
        chosen=logits.detach()[torch.arange(len(logits),device=logits.device),selected.to(logits.device)]
        return (chosen/temperature).softmax(-1)


def kd_loss(student_logits,target,temperature):
    # Sum over C then mean over N. This function does NOT include T^2 or lambda.
    return F.kl_div(F.log_softmax(student_logits/temperature,dim=1),target.detach(),reduction='batchmean')


def freeze_teachers(models,device):
    for m in models:
        m.to(device).eval().requires_grad_(False)
        for p in m.parameters(): p.grad=None


def teacher_identity(models):
    return [tensor_fingerprint(m.state_dict()) for m in models]


def assert_frozen(models,expected):
    if teacher_identity(models)!=expected or any(any(mod.training for mod in m.modules()) or any(p.grad is not None or p.requires_grad for p in m.parameters()) for m in models):
        raise RuntimeError('Frozen teacher state/mode/gradient changed')


def forward_teachers(models,data,device,batch_size):
    """Canonical original sample order, no data transforms and no training RNG advance."""
    with preserved_rng(),torch.no_grad():
        start=time.perf_counter();sync(device)
        rows=[]; indices=[]
        loader=make_data_loader(IndexedData(data),batch_size,shuffle=False,drop_last=False,num_workers=0)
        for x,y,ids in loader:
            rows.append(torch.stack([m(x.to(device)).detach().cpu() for m in models],dim=1))
            indices.extend(ids.tolist())
        sync(device)
        if indices!=list(range(len(data))): raise RuntimeError('Teacher cache sample order changed')
        logits=torch.cat(rows)
        if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite teacher output')
        return logits,time.perf_counter()-start


def teacher_cache(models,data,source,data_record,args):
    labels=torch.stack([y for _,y in data])
    metadata=dict(schema=1,teacher_states=teacher_identity(models),
        checkpoint_sha256=[r['checkpoint_sha256'] for r in source['checkpoints']],
        class_order_sha256=source['class_order_sha256'],data=data_record,
        preprocessing='original float32/int64 no augmentation',teacher_eval=True,
        teacher_batch_size=args.teacher_batch_size,device=str(args.device),torch_version=torch.__version__,backend=backend_settings())
    # Paths are provenance, not cache identity; equal data at another path is still verified by its hashes.
    metadata['data']={k:v for k,v in data_record.items() if k!='path'}
    key=digest_json(metadata);root=Path(args.cache_dir);root.mkdir(parents=True,exist_ok=True)
    path=root/f'teacher_logits_{key}.pt';start=time.perf_counter();seconds=0.
    with preserved_rng():
        hit=path.is_file()
        if hit:
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved['metadata']!=metadata: raise ValueError('Teacher cache provenance mismatch')
            logits=saved['logits']
            if (logits.shape!=(len(data),3,len(source['class_order'])) or not torch.isfinite(logits).all()
                    or saved['logits_sha256']!=tensor_digest(logits)
                    or not torch.equal(saved['sample_ids'],torch.arange(len(data)))):
                raise ValueError('Teacher cache shape/content/index mismatch')
            selected,true=select_teachers(logits,labels)
            if not torch.equal(selected,saved['selected']): raise ValueError('Teacher cache selection mismatch')
        else:
            logits,seconds=forward_teachers(models,data,args.device,args.teacher_batch_size)
            selected,true=select_teachers(logits,labels)
            saved=dict(metadata=metadata,logits=logits,logits_sha256=tensor_digest(logits),selected=selected,
                sample_ids=torch.arange(len(data)))
            temp=root/f'.{path.name}.{uuid.uuid4().hex}.tmp'
            try:
                torch.save(saved,temp);os.replace(temp,path)  # Atomic complete publication; never read half a file.
            finally:
                if temp.exists(): temp.unlink()
        check=None
        if args.verify_cache:
            direct,extra=forward_teachers(models,data,args.device,args.teacher_batch_size);seconds+=extra
            torch.testing.assert_close(logits,direct,atol=1e-6,rtol=1e-5)
            if not torch.equal(selected,select_teachers(direct,labels)[0]):
                raise ValueError('Cache/direct teacher selection mismatch')
            check=dict(max_absolute_error=float((logits-direct).abs().max()),atol=1e-6,rtol=1e-5)
    return dict(logits=logits,selected=selected,true_probabilities=true,
        report=dict(key=key,path=str(path.resolve()),hit=hit,storage_bytes=path.stat().st_size,
            teacher_forward_seconds=seconds,cache_total_seconds=time.perf_counter()-start,
            teacher_sample_forwards_this_process=(0 if hit else 3*len(data))+(3*len(data) if args.verify_cache else 0),
            logits_sha256=saved['logits_sha256'],metadata=metadata,direct_verification=check))


def selection_report(cache,source):
    chosen=cache['selected'];p=cache['true_probabilities'];adv=p[:,1:]-p[:,0,None]
    best=adv.max(1).values;positive=best[best>0]
    return dict(sample_count=len(chosen),temperature_for_selection=1.,tie_priority=source['model_id_order'],
        selected_counts={role:int((chosen==i).sum()) for i,role in enumerate(('A0','B0','C0'))},
        selected_fractions={role:float((chosen==i).double().mean()) for i,role in enumerate(('A0','B0','C0'))},
        external_beats_A0_union=int((best>0).sum()),B0_beats_A0=int((adv[:,0]>0).sum()),C0_beats_A0=int((adv[:,1]>0).sum()),
        max_external_true_probability_advantage_mean=float(best.double().mean()),
        positive_advantage_mean=float(positive.double().mean()) if len(positive) else 0.,
        advantage_quantiles=dict(zip(('min','q25','median','q75','max'),torch.quantile(best.double(),torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist())))


def observe(model,data,args):
    """Reuse endpoint evaluator on an RNG-protected private model copy."""
    with preserved_rng(),torch.no_grad():
        return evaluate(copy.deepcopy(model),data,args.device,batch_size=args.test_batch_size,return_predictions=True)


def train_mode(mode,teachers,data,cache,args,initial_rng):
    if mode not in MODES: raise ValueError('Unknown offline mode')
    set_rng(initial_rng)
    initial_training_rng_sha256=training_rng_digest()
    student=copy.deepcopy(teachers[0]).requires_grad_(True)
    if any(id(p) in {id(q) for m in teachers for q in m.parameters()} for p in student.parameters()):
        raise RuntimeError('Student aliases frozen teacher')
    parameters=list(student.parameters())
    optimizer=torch.optim.SGD(parameters,lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    loader=make_data_loader(IndexedData(data['train']),args.batch_size,shuffle=True,drop_last=False,num_workers=0)
    criterion=torch.nn.CrossEntropyLoss();captured=[]
    def loss(logits,labels):
        captured[:] = [logits]
        return criterion(logits,labels)
    objective=SimpleNamespace(loss=loss,args=SimpleNamespace(is_regular=int(args.f_regular),regular_lamda=args.f_lambda))
    teacher_before=teacher_identity(teachers);rows=[];orders=[];evaluations=[]
    initial_sha=tensor_fingerprint(student.state_dict());base_test=observe(teachers[0],data['test'],args)
    def endpoint(epoch,means,seconds,visits,steps):
        sync(args.device);evaluation_start=time.perf_counter()
        train=observe(student,data['train'],args);test=observe(student,data['test'],args)
        sync(args.device);evaluation_seconds=time.perf_counter()-evaluation_start
        labels=torch.stack([y for _,y in data['test']]);original=base_test['predictions']==labels;now=test['predictions']==labels
        row=dict(mode=mode,epoch=epoch,train_ce=train['ce'],train_correct=train['correct'],train_samples=train['samples'],
            train_accuracy=train['accuracy'],test_ce=test['ce'],test_correct=test['correct'],test_samples=test['samples'],test_accuracy=test['accuracy'],
            test_accuracy_change_pp=100*(test['accuracy']-base_test['accuracy']),
            original_wrong_now_correct=int((~original & now).sum()),original_correct_now_wrong=int((original & ~now).sum()),
            train_seconds=seconds,evaluation_seconds=evaluation_seconds,sample_visits=visits,optimizer_steps=steps,parameter_count=sum(p.numel() for p in student.parameters()),
            student_state_sha256=tensor_fingerprint(student.state_dict()),**means)
        if any(not math.isfinite(v) for v in row.values() if isinstance(v,float)): raise FloatingPointError('Nonfinite offline metric')
        rows.append(row);evaluations.append(dict(epoch=epoch,predictions=test['predictions'].tolist()))
        print(f'[OfflineKD] mode={mode} epoch={epoch} train_CE={train["ce"]:.8f} test_CE={test["ce"]:.8f} test_acc={100*test["accuracy"]:.6f}%')
    names=('mean_ce','mean_weighted_frobenius','mean_raw_kl','mean_weighted_kd','mean_total_loss')
    endpoint(0,{n:None for n in names},0.,0,0)
    for epoch in range(1,args.epochs+1):
        student.train();sums=[0.]*5;visits=steps=0;trace=[];sync(args.device);start=time.perf_counter()
        for x,y,ids in loader:
            trace.append(ids.tolist());x,y=x.to(args.device),y.to(args.device);optimizer.zero_grad()
            ce,frob=clientTargetProj._objective(objective,student,x,y)
            raw=ce.new_zeros(());weighted=raw
            if mode!='continue_ce':
                logits=cache['logits'][ids].to(args.device)
                q=teacher_target(logits,mode,args.temperature,cache['selected'][ids])
                raw=kd_loss(captured[0],q,args.temperature)
                weighted=args.kd_lambda*(args.temperature**2)*raw
            # Zero lambda uses exactly the ordinary CE+F backward expression.
            total=ce+frob if mode=='continue_ce' or args.kd_lambda==0 else ce+frob+weighted
            if not torch.isfinite(total): raise FloatingPointError('Nonfinite offline training loss')
            total.backward()
            if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(parameters,args.grad_clip)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite offline student gradient')
            optimizer.step();captured.clear()
            for i,v in enumerate((ce,frob,raw,weighted,total)): sums[i]+=float(v.detach())*len(y)
            visits+=len(y);steps+=1
        sync(args.device);seconds=time.perf_counter()-start
        if visits!=len(data['train']): raise RuntimeError('Incomplete offline training epoch')
        orders.append(dict(epoch=epoch,batches=trace));endpoint(epoch,dict(zip(names,[x/visits for x in sums])),seconds,visits,steps)
        assert_frozen(teachers,teacher_before)
    return dict(student=student,optimizer=optimizer,rows=rows,batches=orders,test_predictions=evaluations,
        initial_state_sha256=initial_sha,initial_training_rng_sha256=initial_training_rng_sha256,
        final_rng_sha256=training_rng_digest(),teacher_states_unchanged=True,
        batch_order_sha256=digest_json(orders),fixed_final_epoch=args.epochs)


def write_csv(path,rows):
    if not rows: raise ValueError('Cannot export empty offline metric table')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
