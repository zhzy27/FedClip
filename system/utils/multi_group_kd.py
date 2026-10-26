"""Per-round downloaded low-rank teachers; no change to federation backend/RNG."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.data_loading import make_data_loader
from common.experiment import rng_state,set_rng,tensor_digest,jsonable
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_group_kd_options import requested,protocol


@contextmanager
def protected_rng():
    state=rng_state()
    try:yield
    finally:set_rng(state)


def synchronize(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def remove_true(logits,labels):
    """Delete, rather than zero/mask a probability after normalization."""
    if logits.shape[-1]<2:raise ValueError('not_true needs at least two classes')
    mask=torch.arange(logits.shape[-1],device=logits.device)[None,:]!=labels[:,None]
    if logits.ndim==2:return logits[mask].reshape(len(labels),-1)
    if logits.ndim==3:return logits[mask[:,None,:].expand_as(logits)].reshape(len(labels),logits.shape[1],-1)
    raise ValueError('Expected N x C or N x E x C logits')


def distillation_loss(student,teachers,labels,mode,temperature):
    if mode not in ('full','not_true') or teachers.ndim!=3 or teachers.shape[1]<1:
        raise ValueError('Expected a full/not_true mode with at least one external teacher')
    if teachers.shape[0]!=len(student) or teachers.shape[2]!=student.shape[1]:raise ValueError('KD shape mismatch')
    targets=teachers.detach()
    if mode=='not_true':student,targets=remove_true(student,labels),remove_true(targets,labels)
    with torch.no_grad():q=(targets/temperature).softmax(-1).mean(1)
    return F.kl_div(F.log_softmax(student/temperature,dim=-1),q,reduction='batchmean')


class IndexedData(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,index):return *self.base[index],index


def _forward_private(models,data,indices,device,batch):
    """Copy full low-rank state/modes; source models and flags never touched."""
    with protected_rng(),torch.no_grad():
        logits=[];hashes=[];seconds=0.
        source=[tensor_fingerprint(m.state_dict()) for m in models]
        for model in models:
            teacher=copy.deepcopy(model).eval().requires_grad_(False)
            if any(p is q for p in teacher.parameters() for q in model.parameters()):raise RuntimeError('Teacher parameter alias')
            before=tensor_fingerprint(teacher.state_dict());parts=[]
            synchronize(device);tick=time.perf_counter()
            for start in range(0,len(indices),batch):
                x=torch.stack([data[i][0] for i in indices[start:start+batch]]).to(device)
                parts.append(teacher(x).detach().cpu())
            synchronize(device);seconds+=time.perf_counter()-tick
            output=torch.cat(parts)
            if not torch.isfinite(output).all():raise FloatingPointError('Nonfinite downloaded teacher logits')
            if before!=tensor_fingerprint(teacher.state_dict()) or any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError('Frozen teacher state/gradient changed during cache')
            logits.append(output);hashes.append(before)
            del teacher
        if source!=[tensor_fingerprint(m.state_dict()) for m in models]:raise RuntimeError('Source teacher mutated by cache')
        return torch.stack(logits,dim=1),hashes,seconds


def bind_assignment(client,server):
    loop=server.loop_round
    if client.download_round!=loop or client.upload_round is not None:raise RuntimeError('KD binding requires fresh this-round download')
    assigned=None
    if loop>0:
        payload=server.group_payload
        if server.groups is None or payload is None:raise RuntimeError('Require validated fixed groups before KD')
        assigned=int(payload['client_to_model'][str(client.id)])
        if not 0<=assigned<client.K or client.id not in server.groups[assigned]:raise ValueError('Invalid KD membership')
    train=next(r for r in server.data_manifest['files'] if r['client_id']==client.id and r['split']=='train')
    client.kd_binding=dict(loop_round=loop,assigned_model_id=assigned,
        group_fingerprint=server.group_payload['group_fingerprint'] if server.group_payload else None,
        train_data_sha256=train['sha256'],decoded_x_sha256=train['decoded_x_sha256'],decoded_y_sha256=train['decoded_y_sha256'])


class Session:
    def __init__(self,client,loader):
        self.client=client;self.args=client.args;self.data=loader.dataset;self.loop=client.download_round
        binding=getattr(client,'kd_binding',None)
        if binding is None or binding['loop_round']!=self.loop:raise RuntimeError('Missing/stale server KD assignment')
        self.binding=copy.deepcopy(binding);self.assigned=binding['assigned_model_id'];self.cache=None;self.probes=[]
        self.teacher_ids=[];self.cost=0.;self.forward_seconds=0.;self.cache_fingerprint=None;self.source_hashes=[]
        self.reason=('first_round_no_KD' if self.loop==0 else 'zero_coefficient' if self.args.multi_kd_lambda==0 else
                     'K1_no_external_teachers' if client.K==1 else None)
        self.active=self.reason is None
        self.teacher_forwards=0
        if self.active:
            self._prepare_cache()
        self.probe_due=(self.active and self.args.multi_kd_probe_samples>0 and
            ((self.loop+1)%self.args.multi_kd_observe_gap==0 or self.loop==self.args.global_rounds))
        if self.probe_due:
            generator=np.random.default_rng(np.random.SeedSequence([self.args.seed,client.id,91273]))
            self.probe_ids=sorted(generator.choice(len(self.data),min(self.args.multi_kd_probe_samples,len(self.data)),replace=False).tolist())
            self.observe('pre')

    @protected_rng()
    def _prepare_cache(self):
        client=self.client
        if self.assigned is None or not 0<=self.assigned<client.K:raise RuntimeError('Invalid assigned KD student ID')
        if client.upload_round is not None:raise RuntimeError('Teacher caching must precede every local SGD task')
        tick=time.perf_counter()
        actual=[tensor_fingerprint(m.state_dict()) for m in client.model_set]
        if actual!=getattr(client,'kd_download_local_states',None):raise RuntimeError('Downloaded local state modified before teacher caching')
        self.labels=torch.stack([self.data[i][1] for i in range(len(self.data))])
        if (tensor_digest(torch.stack([self.data[i][0] for i in range(len(self.data))]))!=self.binding['decoded_x_sha256'] or
                tensor_digest(self.labels)!=self.binding['decoded_y_sha256']):raise ValueError('KD training data differs from audited download-round data')
        self.teacher_ids=[k for k in range(client.K) if k!=self.assigned]
        indices=list(range(len(self.data)))
        values,self.source_hashes,self.forward_seconds=_forward_private([client.model_set[k] for k in self.teacher_ids],
            self.data,indices,client.device,self.args.multi_kd_teacher_batch_size)
        if len(self.data)!=client.train_samples:raise ValueError('KD original training sample count mismatch')
        self.cache=values;self.logits_hash=tensor_digest(values);self.teacher_forwards=len(indices)*len(self.teacher_ids)
        self.cache_fingerprint=digest_json(dict(binding=self.binding,teacher_ids=self.teacher_ids,teacher_states=self.source_hashes,
            labels=tensor_digest(self.labels),logits=self.logits_hash,sample_ids=tensor_digest(torch.arange(len(indices)))))
        self.cost=time.perf_counter()-tick

    def indexed_loader(self,original):
        if not self.active:return original
        # Same sampler, generator defaults, drop_last, workers and original
        # client-module injectable loader class. No sampler is iterated twice.
        import importlib
        cls=importlib.import_module('flcore.clients.clientMultiGroupProj').DataLoader
        return make_data_loader(IndexedData(self.data),self.client.batch_size,shuffle=True,
                                drop_last=False,num_workers=0,loader_cls=cls)

    def objective(self,model,x,y,indices):
        from flcore.clients.clientTargetProj import clientTargetProj
        if self.client.download_round!=self.loop or self.client.kd_binding!=self.binding:raise RuntimeError('Stale KD source round/membership')
        if indices is None or indices.min()<0 or indices.max()>=len(self.data) or not torch.equal(self.labels[indices],y.detach().cpu()):
            raise ValueError('KD sample index/label mismatch')
        captured=[]
        def ce(logits,labels):captured.append(logits);return self.client.loss(logits,labels)
        adapter=SimpleNamespace(args=self.args,loss=ce)
        primary,regular=clientTargetProj._objective(adapter,model,x,y)
        raw=distillation_loss(captured[0],self.cache[indices].to(x.device),y,self.args.multi_kd_mode,self.args.multi_kd_temperature)
        weighted=self.args.multi_kd_lambda*self.args.multi_kd_temperature**2*raw
        if not torch.isfinite(raw) or not torch.isfinite(weighted):raise FloatingPointError('Nonfinite online KD KL/weighted loss')
        return primary,regular,raw,weighted

    def observe(self,stage):
        tick=time.perf_counter()
        logits,_,seconds=_forward_private(self.client.model_set,self.data,self.probe_ids,self.client.device,
                                         self.args.multi_kd_teacher_batch_size)
        p=logits.double().softmax(-1);pairs=[];tiny=torch.finfo(torch.float64).tiny
        for left in range(self.client.K):
            for right in range(left+1,self.client.K):
                a,b=p[:,left],p[:,right];mean=(a+b)/2
                js=.5*(a*(a.clamp_min(tiny).log()-mean.clamp_min(tiny).log())+
                       b*(b.clamp_min(tiny).log()-mean.clamp_min(tiny).log())).sum(-1).mean()
                pairs.append(dict(left_model_id=left,right_model_id=right,js=float(js),
                    prediction_agreement=float((a.argmax(-1)==b.argmax(-1)).double().mean())))
        self.probes.append(dict(client_id=self.client.id,loop_round=self.loop,completed_train_count=self.loop+1,
            assigned_model_id=self.assigned,stage=stage,probe_sample_ids=self.probe_ids,probe_count=len(self.probe_ids),
            probe_ids_sha256=digest_json(self.probe_ids),train_data_sha256=self.binding['train_data_sha256'],
            pairs=pairs,mean_js=sum(r['js'] for r in pairs)/len(pairs),
            probability_temperature=1.,js_unit='nats',probe_sample_forwards=len(self.probe_ids)*self.client.K,
            mean_pair_prediction_agreement=sum(r['prediction_agreement'] for r in pairs)/len(pairs),
            unanimous_prediction_fraction=float((logits.argmax(-1)==logits.argmax(-1)[:,:1]).all(1).double().mean()),
            observation_seconds=time.perf_counter()-tick,forward_seconds=seconds,training_use='none_diagnostic_only'))

    def fields(self,column,raw_sum,weighted_sum,loss_sum,visits):
        on=self.active and column==self.assigned;owner=on
        return dict(kd_requested_mode=self.args.multi_kd_mode,kd_effective_mode=self.args.multi_kd_mode if on else 'none',
            kd_requested_lambda=self.args.multi_kd_lambda,kd_effective_lambda=self.args.multi_kd_lambda if on else 0.,
            kd_temperature=self.args.multi_kd_temperature,kd_requested_temperature=self.args.multi_kd_temperature,
            kd_effective_temperature=self.args.multi_kd_temperature if on else None,kd_enabled=on,
            kd_disabled_reason=None if on else self.reason or 'not_assigned_model',assigned_model_id=self.assigned,
            teacher_model_ids=self.teacher_ids if on else [],teacher_download_loop_round=self.loop if on else None,
            teacher_state_sha256=self.source_hashes if on else [],kd_cache_fingerprint=self.cache_fingerprint if on else None,
            teacher_download_full_server_sha256=[self.client.download_hashes[k] for k in self.teacher_ids] if on else [],
            teacher_logits_sha256=self.logits_hash if on else None,
            teacher_cache_sample_count=len(self.data) if owner else 0,
            raw_kl_loss=raw_sum/visits,weighted_kd_loss=weighted_sum/visits,total_loss=loss_sum/visits,
            kd_cache_seconds=self.cost if owner else 0.,teacher_forward_seconds=self.forward_seconds if owner else 0.,
            teacher_sample_forwards=self.teacher_forwards if owner else 0,
            teacher_cache_bytes=self.cache.numel()*self.cache.element_size() if owner else 0,
            teacher_state_unchanged_during_cache=True if on else None,cache_indices_source_valid=True if on else None,
            teacher_freeze_scope='temporary_private_copies_only_original_models_train_normally',
            kd_binding_loop_round=self.binding['loop_round'],group_fingerprint=self.binding['group_fingerprint'],
            train_data_sha256=self.binding['train_data_sha256'],decoded_x_sha256=self.binding['decoded_x_sha256'],
            decoded_y_sha256=self.binding['decoded_y_sha256'])

    def finish(self):
        if self.cache is not None and tensor_digest(self.cache)!=self.logits_hash:raise RuntimeError('Teacher cache mutated during SGD')
        if self.probe_due:self.observe('post')
        self.client.kd_probe_records=self.probes
        self.cache=None  # No history/cross-round reuse or graph retained.


def persist_client(server,client):
    if not requested(server.args):return
    rows=[dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,**r) for r in client.train_stats]
    server.kd_training_records.extend(rows);server.kd_probe_records.extend(client.kd_probe_records)
    with (Path(server.report_dir())/'online_kd.jsonl').open('a',encoding='utf-8') as stream:
        for kind,records in [('training',rows),('probe',client.kd_probe_records)]:
            for row in records:stream.write(json.dumps(jsonable(dict(event=kind,**row)),allow_nan=False)+'\n')
