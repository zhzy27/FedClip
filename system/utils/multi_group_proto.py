"""Fresh round-local class prototypes; no per-sample features leave collection."""
import copy
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common.data_loading import make_data_loader
from common.experiment import rng_state,set_rng,tensor_digest,jsonable,write_json
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_group_proto_options import requested,active_round,protocol


@contextmanager
def protected_rng():
    before=rng_state()
    try:yield
    finally:set_rng(before)


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


class Features:
    """Capture the actual classifier input without replacing forward/modules."""
    def __init__(self,model):self.model=model;self.value=None;self.calls=0
    def __enter__(self):
        if not isinstance(getattr(self.model,'fc3',None),torch.nn.Linear):raise ValueError('Need the actual fc3 linear classifier')
        def capture(module,inputs):
            self.calls+=1;self.value=inputs[0].clone()  # Keep the student graph.
        self.hook=self.model.fc3.register_forward_pre_hook(capture)
        return self
    def get(self):
        if self.calls!=1 or self.value.ndim!=2 or self.value.shape[1]!=self.model.fc3.in_features:
            raise RuntimeError('Expected exactly one actual fc3 input per original forward')
        if not torch.isfinite(self.value).all():raise FloatingPointError('Nonfinite classifier features')
        return self.value
    def __exit__(self,*exc):self.hook.remove()


def prototype_loss(features,labels,prototypes,valid,eps):
    if features.ndim!=2 or prototypes.ndim!=2 or features.shape[1]!=prototypes.shape[1]:raise ValueError('Prototype feature dimension mismatch')
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():raise FloatingPointError('Nonfinite prototype loss input')
    selected=valid.detach().to(labels.device)[labels]
    count=int(selected.sum())
    if not count:return features.reshape(-1)[:1].sum()*0.,0  # Avoid overflow even for finite, very large features.
    target=prototypes.detach().to(features)[labels[selected]]
    return (1-F.cosine_similarity(features[selected],target,dim=1,eps=eps)).mean(),count


def encode_packet(kind,tensors,metadata):
    """Stable logical transport: u64 header size + UTF8 JSON + raw CPU tensors."""
    descriptions=[];parts=[]
    for name,tensor in tensors.items():
        array=tensor.detach().cpu().contiguous().numpy()
        array=np.asarray(array,dtype=array.dtype.newbyteorder('<'));raw=array.tobytes()
        descriptions.append(dict(name=name,dtype=array.dtype.str,shape=list(array.shape),nbytes=len(raw)));parts.append(raw)
    header=json.dumps(dict(schema=1,kind=kind,tensors=descriptions,metadata=metadata),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return struct.pack('<Q',len(header))+header+b''.join(parts)


def decode_packet(packet,kind):
    if len(packet)<8:raise ValueError('Truncated prototype packet')
    length=struct.unpack('<Q',packet[:8])[0]
    header=json.loads(packet[8:8+length]);offset=8+length;result={}
    if header['schema']!=1 or header['kind']!=kind:raise ValueError('Prototype packet schema/kind mismatch')
    for item in header['tensors']:
        dtype=np.dtype(item['dtype']);size=item['nbytes']
        if dtype.kind not in 'fbiu' or size!=int(np.prod(item['shape']))*dtype.itemsize or offset+size>len(packet):raise ValueError('Invalid prototype tensor payload')
        result[item['name']]=torch.from_numpy(np.frombuffer(packet,dtype=dtype,count=size//dtype.itemsize,offset=offset).copy().reshape(item['shape']))
        offset+=size
    if offset!=len(packet):raise ValueError('Unexpected prototype packet trailing bytes')
    return result,header['metadata']


def packet_sha(packet):return hashlib.sha256(packet).hexdigest()


class IndexedData(Dataset):
    def __init__(self,base):self.base=base
    def __len__(self):return len(self.base)
    def __getitem__(self,i):return *self.base[i],i


@protected_rng()
@torch.no_grad()
def collect_client(client,binding):
    tick=time.perf_counter();args=client.args
    if client.download_round!=binding['loop_round'] or client.upload_round is not None:raise RuntimeError('Collection requires fresh untrained downloads')
    states=[tensor_fingerprint(m.state_dict()) for m in client.model_set]
    if states!=client.proto_download_local_states:raise RuntimeError('Model changed before collective prototype collection')
    data=client.load_train_data().dataset
    x=torch.stack([a for a,b in data]);y=torch.stack([b for a,b in data])
    if len(data)!=client.train_samples or tensor_digest(x)!=binding['decoded_x_sha256'] or tensor_digest(y)!=binding['decoded_y_sha256']:
        raise ValueError('Prototype collection data differs from original audited training shard')
    del x,y
    loader=make_data_loader(IndexedData(data),args.multi_proto_batch_size,shuffle=False,drop_last=False,num_workers=0)
    sums=[];counts=None;rows=[]
    for column,source in enumerate(client.model_set):
        model=copy.deepcopy(source).eval().requires_grad_(False)
        c,d=model.fc3.out_features,model.fc3.in_features
        total=torch.zeros(c,d,dtype=torch.float64);n=torch.zeros(c,dtype=torch.int64)
        before=tensor_fingerprint(model.state_dict());seen=[];batches=zero=0
        sync(client.device);started=time.perf_counter()
        for images,labels,indices in loader:
            with Features(model) as capture:
                output=model(images.to(client.device));h=capture.get().detach().double().cpu()
            if output.shape!=(len(labels),c) or not torch.isfinite(output).all():raise FloatingPointError('Invalid prototype collector logits')
            norm=h.norm(dim=1,keepdim=True);u=h/norm.clamp_min(args.multi_proto_eps)
            total.index_add_(0,labels,u);n.add_(torch.bincount(labels,minlength=c))
            seen.extend(indices.tolist());batches+=1;zero+=int((norm<=args.multi_proto_eps).sum())
        sync(client.device)
        if seen!=list(range(len(data))) or n.sum()!=len(data):raise RuntimeError('Collection must visit each original sample exactly once per column')
        if before!=tensor_fingerprint(model.state_dict()):raise RuntimeError('Collector changed frozen parameters/buffers')
        if counts is not None and not torch.equal(n,counts):raise ValueError('Class counts differ across model columns')
        counts=n;sums.append(total)
        rows.append(dict(loop_round=client.download_round,completed_train_count=client.download_round+1,client_id=client.id,
            model_id=column,assigned_model_id=binding['assigned_model_id'],capacity_level=client.capacity_level,rank_rate=client.rank_rate,
            collected_samples=len(seen),forward_batches=batches,zero_feature_samples=zero,
            forward_seconds=time.perf_counter()-started,feature_dim=d,class_count=c,download_state_sha256=states[column],
            full_server_sha256=client.download_hashes[column],group_fingerprint=binding['group_fingerprint']))
        del model
    if states!=[tensor_fingerprint(m.state_dict()) for m in client.model_set]:raise RuntimeError('Collector modified actual low-rank models')
    metadata=dict(binding,client_id=client.id,capacity_level=client.capacity_level,rank_rate=client.rank_rate,
        model_state_sha256=states,full_server_sha256=client.download_hashes)
    packet=encode_packet('statistics',dict(sums=torch.stack(sums),counts=counts),metadata)
    upload=dict(loop_round=client.download_round,completed_train_count=client.download_round+1,client_id=client.id,
        model_ids=list(range(client.K)),assigned_model_id=binding['assigned_model_id'],capacity_level=client.capacity_level,
        rank_rate=client.rank_rate,collection_seconds=time.perf_counter()-tick,collected_sample_forwards=len(data)*client.K,
        forward_batches=sum(r['forward_batches'] for r in rows),statistics_upload_bytes=len(packet),packet_sha256=packet_sha(packet))
    return packet,rows,upload


def merge_statistics(statistics,groups,eps,loop,group_fingerprint,diagnostics=True):
    """Group presence is decided only by true-label COUNTS, not vector norms."""
    ids=sorted(statistics)
    if sorted(i for group in groups for i in group)!=ids:raise ValueError('Prototype groups must partition all contributors exactly once')
    first=statistics[ids[0]];k,c,d=first['sums'].shape
    if len(groups)!=k:raise ValueError('Prototype group/model count mismatch')
    for cid,st in statistics.items():
        if st['sums'].shape!=(k,c,d) or st['counts'].shape!=(c,) or st['counts'].dtype!=torch.int64 or (st['counts']<0).any():raise ValueError('Invalid class statistic shape/counts')
        if not torch.isfinite(st['sums']).all():raise FloatingPointError('Nonfinite class statistic')
    p=torch.zeros(k,c,d,dtype=torch.float64);valid=torch.zeros(k,c,dtype=torch.bool);sources=[];directions=[];capacities=[]
    for g,members in enumerate(groups):
        for label in range(c):
            inside=[i for i in members if statistics[i]['counts'][label]>0]
            contributors=inside or [i for i in ids if i not in members and statistics[i]['counts'][label]>0]
            kind='group' if inside else 'outside' if contributors else 'invalid'
            count=sum(int(statistics[i]['counts'][label]) for i in contributors)
            q=sum((statistics[i]['sums'][g,label].double() for i in contributors),torch.zeros(d,dtype=torch.float64))/count if count else torch.zeros(d,dtype=torch.float64)
            norm=float(q.norm());reason='no_training_samples' if not count else 'near_zero_merged_mean' if norm<=eps else None
            if not torch.isfinite(q).all() or not math.isfinite(norm):raise FloatingPointError('Nonfinite merged class statistic')
            if reason is None:p[g,label]=q/norm;valid[g,label]=True
            sources.append(dict(loop_round=loop,completed_train_count=loop+1,model_id=g,class_id=label,
                source=kind if reason is None else 'invalid',candidate_source=kind,invalid_reason=reason,
                source_client_ids=contributors,source_client_count=len(contributors),class_samples=count,
                pre_normalization_norm=norm,valid=bool(valid[g,label]),group_fingerprint=group_fingerprint))
            if diagnostics:
                for cid in ids:
                    st=statistics[cid];n=int(st['counts'][label])
                    if not n:continue
                    mean=st['sums'][g,label].double()/n;local_norm=float(mean.norm())
                    cosine=float(F.cosine_similarity(mean[None],p[g,label][None],dim=1,eps=eps)) if valid[g,label] and local_norm>eps else None
                    directions.append(dict(loop_round=loop,completed_train_count=loop+1,client_id=cid,model_id=g,class_id=label,
                        capacity_level=st['capacity_level'],rank_rate=st['rank_rate'],class_samples=n,local_mean_norm=local_norm,
                        cosine_to_prototype=cosine,within_group=cid in members,used_for_prototype=cid in contributors))
                for level in sorted(set(st['capacity_level'] for st in statistics.values())):
                    cohort=[i for i in ids if statistics[i]['capacity_level']==level and statistics[i]['counts'][label]>0]
                    n=sum(int(statistics[i]['counts'][label]) for i in cohort)
                    if not n:continue
                    mean=sum((statistics[i]['sums'][g,label].double() for i in cohort),torch.zeros(d,dtype=torch.float64))/n
                    local_norm=float(mean.norm())
                    capacities.append(dict(loop_round=loop,completed_train_count=loop+1,model_id=g,class_id=label,
                        capacity_level=level,class_samples=n,client_count=len(cohort),local_mean_norm=local_norm,
                        cosine_to_prototype=float(F.cosine_similarity(mean[None],p[g,label][None],dim=1,eps=eps)) if valid[g,label] and local_norm>eps else None,
                        scope='all_clients_of_capacity_including_outside'))
    if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite merged prototypes')
    return p,valid,sources,directions,capacities


def append_events(server,event,rows):
    with (Path(server.report_dir())/'class_prototypes.jsonl').open('a',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(jsonable(dict(event=event,**row)),allow_nan=False)+'\n')


def prepare_round(server):
    """Global barrier: ALL downloads -> ALL statistics -> libraries -> ANY SGD."""
    args=server.args;loop=server.loop_round
    ready=active_round(args,loop)
    if ready and (server.groups is None or server.group_payload is None):raise RuntimeError('Fixed groups required before prototype collection')
    for client in server.clients:
        if client.download_round!=loop or client.upload_round is not None:raise RuntimeError('All clients must download before prototype preparation')
        data=next(r for r in server.data_manifest['files'] if r['client_id']==client.id and r['split']=='train')
        assigned=int(server.group_payload['client_to_model'][str(client.id)]) if server.group_payload else None
        if assigned is not None and client.id not in server.groups[assigned]:raise ValueError('Invalid prototype client membership')
        client.proto_binding=dict(loop_round=loop,assigned_model_id=assigned,
            group_fingerprint=server.group_payload['group_fingerprint'] if server.group_payload else None,
            train_data_sha256=data['sha256'],decoded_x_sha256=data['decoded_x_sha256'],decoded_y_sha256=data['decoded_y_sha256'])
    if not ready:return
    tick=time.perf_counter();statistics={};packets={};rows=[];uploads=[]
    for client in server.clients:
        packet,collected,upload=collect_client(client,client.proto_binding)
        values,meta=decode_packet(packet,'statistics')
        if meta['loop_round']!=loop or meta['client_id']!=client.id or meta['model_state_sha256']!=client.proto_download_local_states:
            raise ValueError('Stale/incompatible prototype statistics packet')
        data=next(r for r in server.data_manifest['files'] if r['client_id']==client.id and r['split']=='train')
        actual={str(i):int(n) for i,n in enumerate(values['counts']) if n>0}
        if actual!={str(i):int(n) for i,n in data['labels'].items()}:raise ValueError('Class statistics disagree with audited true labels')
        statistics[client.id]=dict(**values,capacity_level=client.capacity_level,rank_rate=client.rank_rate)
        packets[str(client.id)]=packet_sha(packet);rows.extend(collected);uploads.append(upload)
    # Even a client later in the selection order cannot have been trained yet.
    for client in server.clients:
        if client.upload_round is not None or client.proto_download_local_states!=[tensor_fingerprint(m.state_dict()) for m in client.model_set]:
            raise RuntimeError('Collective prototype barrier violated')
    collection_seconds=time.perf_counter()-tick;merge_started=time.perf_counter()
    p,valid,sources,directions,capacities=merge_statistics(statistics,server.groups,args.multi_proto_eps,loop,
        server.group_payload['group_fingerprint'],bool(args.multi_proto_diagnostics))
    k,c,d=p.shape
    if k!=server.K or c!=server.num_classes:raise ValueError('Prototype library/model class mismatch')
    context=dict(loop_round=loop,completed_train_count=loop+1,K=k,class_count=c,feature_dim=d,epsilon=args.multi_proto_eps,
        group_fingerprint=server.group_payload['group_fingerprint'],statistics_sha256_by_client=packets,
        source_rows_sha256=digest_json(sources),prototype_sha256=tensor_digest(p),valid_sha256=tensor_digest(valid))
    library=encode_packet('library',dict(prototypes=p,valid=valid),context)
    fingerprint=packet_sha(library);merge_seconds=time.perf_counter()-merge_started
    for row in sources:row.update(library_sha256=fingerprint,collection_fingerprint=digest_json(packets))
    server.proto_source_records.extend(sources);server.proto_direction_records.extend(directions)
    server.proto_capacity_records.extend(capacities);server.proto_collection_records.extend(rows);server.proto_upload_records.extend(uploads)
    server.proto_latest_packet=library
    bind_started=time.perf_counter()
    for client in server.clients:
        values,metadata=decode_packet(library,'library')  # Actual packet boundary, not a hypothetical byte estimate.
        if metadata['statistics_sha256_by_client'][str(client.id)]!=packets[str(client.id)]:raise ValueError('Library receipt mismatch')
        client.proto_banks=[]
        for column,model in enumerate(client.model_set):
            if model.fc3.out_features!=c or model.fc3.in_features!=d:raise ValueError('Prototype classifier dimension mismatch')
            client.proto_banks.append(dict(prototypes=values['prototypes'][column].detach().clone().to(model.fc3.weight),
                valid=values['valid'][column].detach().clone().to(client.device)))
        client.proto_receipt=dict(metadata,library_sha256=fingerprint,local_download_states=list(client.proto_download_local_states))
    budget=dict(loop_round=loop,completed_train_count=loop+1,group_fingerprint=context['group_fingerprint'],library_sha256=fingerprint,
        collected_sample_forwards=sum(r['collected_samples'] for r in rows),forward_batches=sum(r['forward_batches'] for r in rows),
        collection_seconds=collection_seconds,merge_seconds=merge_seconds,distribution_seconds=time.perf_counter()-bind_started,
        statistics_upload_bytes=sum(r['statistics_upload_bytes'] for r in uploads),prototype_packet_bytes=len(library),
        prototype_download_bytes=len(library)*len(server.clients),recipient_count=len(server.clients),
        valid_class_prototypes=int(valid.sum()),invalid_class_prototypes=int((~valid).sum()),
        communication_unit='actual canonical in-process packet bytes, excludes physical network framing')
    server.proto_round_records.append(budget)
    for event,records in [('collection',rows),('upload',uploads),('source',sources),('round', [budget])]:append_events(server,event,records)
    if args.multi_proto_diagnostics:
        append_events(server,'client_direction',directions);append_events(server,'capacity_direction',capacities)
    print(f'[ClassPrototypes] completed={loop+1}, all-client barrier complete; valid={int(valid.sum())}/{k*c}; '
          f'forward_samples={budget["collected_sample_forwards"]}; upload={budget["statistics_upload_bytes"]}B download={budget["prototype_download_bytes"]}B')


class Session:
    def __init__(self,client):
        self.client=client;self.args=client.args;self.loop=client.download_round;self.active=active_round(self.args,self.loop)
        self.reason=('zero_coefficient' if self.args.multi_proto_lambda==0 else 'first_round_unchanged' if self.loop==0 else 'before_start_round') if not self.active else None
        binding=getattr(client,'proto_binding',None)
        if binding is None or binding['loop_round']!=self.loop:raise RuntimeError('Prototype round binding missing/stale')
        self.binding=copy.deepcopy(binding);self.receipt=copy.deepcopy(getattr(client,'proto_receipt',None))
        if self.active:
            if (self.receipt is None or self.receipt['loop_round']!=self.loop or
                    self.receipt['group_fingerprint']!=self.binding['group_fingerprint'] or len(client.proto_banks)!=client.K):
                raise RuntimeError('Prototype library is absent/stale/wrong group')
            if self.receipt['local_download_states']!=[tensor_fingerprint(m.state_dict()) for m in client.model_set]:
                raise RuntimeError('Prototype student changed after collective pre-SGD collection')
            self.bank_hashes=[tensor_fingerprint(b) for b in client.proto_banks]

    def objective(self,column,model,x,y):
        from flcore.clients.clientTargetProj import clientTargetProj
        if self.client.download_round!=self.loop or self.client.proto_receipt!=self.receipt:raise RuntimeError('Stale prototype round/receipt')
        with Features(model) as capture:
            ce,regular=clientTargetProj._objective(self.client,model,x,y)
            raw,valid=prototype_loss(capture.get(),y,self.client.proto_banks[column]['prototypes'],
                                     self.client.proto_banks[column]['valid'],self.args.multi_proto_eps)
        weighted=self.args.multi_proto_lambda*raw
        if not torch.isfinite(raw) or not torch.isfinite(weighted):raise FloatingPointError('Nonfinite prototype objective')
        return ce,regular,raw,weighted,valid

    def fields(self,column,raw,weighted,total,valid,visits):
        return dict(proto_requested_mode=self.args.multi_proto_mode,proto_enabled=self.active,
            proto_effective_mode=self.args.multi_proto_mode if self.active else 'none',proto_disabled_reason=self.reason,
            proto_requested_lambda=self.args.multi_proto_lambda,proto_effective_lambda=self.args.multi_proto_lambda if self.active else 0.,
            prototype_source_loop_round=self.loop if self.active else None,
            prototype_library_sha256=self.receipt['library_sha256'] if self.active else None,
            assigned_model_id=self.binding['assigned_model_id'],group_fingerprint=self.binding['group_fingerprint'],
            raw_prototype_loss=raw/visits,weighted_prototype_loss=weighted/visits,total_loss=total/visits,
            prototype_valid_samples=valid,prototype_valid_ratio=valid/visits,
            prototype_average_unit='batch objective weighted by actual SGD sample visits, not endpoint CE')

    def finish(self):
        if self.active and self.bank_hashes!=[tensor_fingerprint(b) for b in self.client.proto_banks]:raise RuntimeError('Detached prototypes mutated during local training')


def persist_training(server,client):
    rows=[dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,**r) for r in client.train_stats]
    server.proto_training_records.extend(rows);append_events(server,'training',rows)


def manifest_value(server):
    return dict(protocol=protocol(server.args),eligible_rounds=len(server.proto_round_records),
        collection_sample_forwards=sum(r['collected_sample_forwards'] for r in server.proto_round_records),
        collection_forward_batches=sum(r['forward_batches'] for r in server.proto_round_records),
        collection_seconds=sum(r['collection_seconds'] for r in server.proto_round_records),
        merge_seconds=sum(r['merge_seconds'] for r in server.proto_round_records),
        statistics_upload_bytes=sum(r['statistics_upload_bytes'] for r in server.proto_round_records),
        prototype_download_bytes=sum(r['prototype_download_bytes'] for r in server.proto_round_records),
        latest_library_sha256=packet_sha(server.proto_latest_packet) if server.proto_latest_packet else None,
        durable_log=str(Path(server.report_dir())/'class_prototypes.jsonl'),
        inference='ordinary low-rank client model; prototype library is training-only')


def save_latest(server):
    if server.proto_latest_packet:
        (Path(server.save_folder_name)/'Prototype_library.packet').write_bytes(server.proto_latest_packet)
        (Path(server.final_model_dir())/'Prototype_library.packet').write_bytes(server.proto_latest_packet)
