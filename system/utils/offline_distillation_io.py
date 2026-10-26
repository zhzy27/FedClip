"""Immutable checkpoint bundles and explicit original shards for offline KD."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import torch

from common.data_io import read_shard, decode_image_data, TENSOR_LIST
from common.experiment import sha256, tensor_digest
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_svd import load_checkpoint, model_capacity


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class Bundle:
    def __init__(self,path):
        self.path = Path(path).resolve() if path else None
        self.zip = zipfile.ZipFile(self.path) if self.path and self.path.is_file() else None

    def read(self,member,required=True):
        if not self.path:
            if required: raise ValueError('No bundle supplied')
            return None
        if Path(member).is_absolute() or '..' in Path(member).parts:
            raise ValueError('Bundle members must be explicit relative paths')
        if self.zip:
            if member not in self.zip.namelist():
                if required: raise FileNotFoundError(f'Bundle member missing: {member}')
                return None
            return self.zip.read(member)
        path = self.path/member
        if not path.is_file():
            if required: raise FileNotFoundError(path)
            return None
        return path.read_bytes()

    def json(self,member):
        value=self.read(member,False)
        return json.loads(value.decode('utf-8')) if value is not None else None

    def close(self):
        if self.zip: self.zip.close()


def load_sources(args,bundle):
    """Use the existing low-rank loader directly; never recover or SVD."""
    group = json.loads(Path(args.group_file).read_text()) if args.group_file else bundle.json('fixed_groups.json')
    info = bundle.json('snapshot_info.json')
    order = list(args.model_ids)
    if len(order)!=3 or len(set(order))!=3 or any(mid<0 for mid in order):
        raise ValueError('Exactly three distinct model IDs are required')
    entries=[]
    for pos,mid in enumerate(order):
        if args.checkpoints:
            path=Path(args.checkpoints[pos]).resolve(); raw=path.read_bytes(); location=str(path)
        else:
            member=f'models/Client_{args.client_id}_model_{mid}.pt'
            raw=bundle.read(member); location=f'{bundle.path}!{member}'
        model,payload=load_checkpoint(io.BytesIO(raw),'cpu')
        if (payload.get('client_id')!=args.client_id or payload.get('model_id')!=mid or
                payload.get('role')!='ordinary_post_local_low_rank' or not payload.get('prediction_checkpoint')):
            raise ValueError(f'Expected this client/model ordinary low-rank prediction checkpoint: {location}')
        completed=payload.get('completed_train_count')
        if type(completed) is not int or completed<1 or payload.get('loop_round')!=completed-1:
            raise ValueError('Checkpoint completed round / loop_round mismatch')
        if args.completed_round is not None and completed!=args.completed_round:
            raise ValueError('Checkpoint round differs from requested completed round')
        if not 0 < model.ratio_LR < 1 or not all(hasattr(getattr(model,n),'rank') for n in ('conv2','fc1','fc2')):
            raise ValueError('Expected original factorized prediction model, not recovered full upload')
        actual=model.state_dict()
        if actual.keys()!=payload['state_dict'].keys(): raise ValueError('Checkpoint state names differ')
        for name,p in actual.items():
            original=payload['state_dict'][name]
            if p.dtype!=original.dtype or not torch.equal(p,original) or not torch.isfinite(p).all():
                raise ValueError(f'Checkpoint dtype/nonfinite/alias inconsistency: {name}')
        schema={n:list(p.shape) for n,p in actual.items()}
        if schema!=payload.get('state_schema'): raise ValueError('Checkpoint schema mismatch')
        cap=model_capacity(model,{})
        for key in ('rank_rate','layers','trainable_parameter_count'):
            if cap[key]!=payload['capacity'][key]: raise ValueError(f'Checkpoint capacity mismatch: {key}')
        if info:
            rows=[r for r in info['models'] if r['model_id']==mid]
            if (info['client_id']!=args.client_id or len(rows)!=1 or
                    info['completed_round']!=completed or rows[0]['completed_round']!=completed):
                raise ValueError('Bundle snapshot_info contradicts checkpoint')
        entries.append(dict(model=model,payload=payload,record=dict(model_id=mid,location=location,
            checkpoint_sha256=hashlib.sha256(raw).hexdigest(),state_sha256=tensor_fingerprint(actual),
            parameter_count=sum(p.numel() for p in model.parameters()),completed_round=completed)))
    common=('client_id','completed_train_count','model_class','model_kwargs','model_family','capacity',
            'group_fingerprint','capacity_protocol','grouping_protocol','initial_parameter_sha256','svd_initialization')
    for key in common:
        if any(e['payload'].get(key)!=entries[0]['payload'].get(key) for e in entries):
            raise ValueError(f'Three checkpoints are incompatible: {key}')
    assigned=args.assigned_model_id
    declarations=[e['payload'].get('assigned_model_id') for e in entries]
    if group:
        from utils.multi_group_clustering import validate_group_file
        validate_group_file(group,group['context'])
        declared=int(group['client_to_model'][str(args.client_id)])
        declarations.append(declared)
        for e in entries:
            if e['payload'].get('group_fingerprint')!=group['group_fingerprint']:
                raise ValueError('Checkpoint/group fingerprint mismatch')
            if e['payload']['model_family']!=group['context']['model_family']:
                raise ValueError('Group model family mismatch')
            if e['payload']['capacity_protocol']!=group['context'].get('svd_capacity_protocol'):
                raise ValueError('Group capacity protocol mismatch')
    declarations=[x for x in declarations if x is not None]
    if assigned is None:
        if not declarations: raise ValueError('Provide assigned-model-id or a grouping/deployment declaration')
        assigned=declarations[0]
    if assigned not in order or any(x!=assigned for x in declarations):
        raise ValueError('Assigned model contradicts checkpoint/group membership')
    entries=sorted(entries,key=lambda e: (e['record']['model_id']!=assigned,order.index(e['record']['model_id'])))
    classes=entries[0]['payload']['model_kwargs']['num_classes']
    expected=json.loads(Path(args.class_order).read_text()) if args.class_order else list(range(classes))
    if len(expected)!=classes or len(set(expected))!=classes: raise ValueError('Class order must contain C unique entries')
    explicit=[]
    for e in entries:
        value=e['payload'].get('class_order')
        if value is not None and value!=expected: raise ValueError('Checkpoint class order mismatch')
        explicit.append(value is not None)
    return entries,dict(client_id=args.client_id,assigned_model_id=assigned,completed_round=entries[0]['record']['completed_round'],
        role_order=['A0','B0','C0'],model_id_order=[e['record']['model_id'] for e in entries],
        class_order=expected,class_order_sha256=digest_json(expected),
        class_order_verification='explicit checkpoint metadata' if all(explicit) else
            'legacy checkpoints lack class_order; original integer label indices, no remapping; semantic class names not independently verified',
        group=group,checkpoints=[e['record'] for e in entries])


def read_data(args,bundle,source,train_required=True):
    result={}; records={}
    directory=Path(args.data_root).expanduser().resolve()/args.dataset/args.data_subdir
    config_path=Path(args.data_config) if args.data_config else directory/'config.json'
    config=json.loads(config_path.read_text()) if config_path.is_file() else bundle.json('data/config.json')
    bundled=bundle.json('data/config.json')
    if config is None: raise ValueError('Original dataset config.json is required')
    if bundled and config!=bundled: raise ValueError('Selected config differs from bundle original data configuration')
    classes=len(source['class_order'])
    if config.get('num_classes')!=classes or not 0<=args.client_id<config['num_clients']:
        raise ValueError('Data class/client metadata contradicts checkpoints')
    if config.get('class_order') is not None and config['class_order']!=source['class_order']:
        raise ValueError('Dataset class order contradicts checkpoints/declaration')
    if source['group'] and config['num_clients']!=source['group']['context']['num_clients']:
        raise ValueError('Dataset client count contradicts original grouping')
    if config.get('partition')!='pat' or args.dataset!='Cifar100' or classes!=100:
        raise ValueError('Offline v1 supports original RGB32 CIFAR100 pat shards only')
    manifest=json.loads(Path(args.data_manifest).read_text()) if args.data_manifest else None
    if manifest is None and source['group'] and train_required:
        # The grouping artifact stores a whole-dataset fingerprint. Verify it
        # using the SAME public audit used at source training, not a guessed
        # fingerprint from the shard directory name or C0's label histogram.
        from common.client_data import audit_image_clients
        manifest=audit_image_clients(SimpleNamespace(algorithm='OfflineDistillation',data_root=args.data_root,
            dataset=args.dataset,data_subdir=args.data_subdir,data_formats='npz_only',
            num_clients=config['num_clients'],num_classes=classes))
    verified_fingerprint=None
    if manifest is not None:
        if manifest.get('config')!=config:
            raise ValueError('Supplied original training data manifest config mismatch')
        if source['group']:
            keys=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')
            verified_fingerprint=digest_json(dict(config_sha256=manifest['config_sha256'],
                files=[{k:m[k] for k in keys} for m in manifest['files']]))
            if verified_fingerprint!=source['group']['context']['data_fingerprint']:
                raise ValueError('Original full data manifest differs from grouping source fingerprint')
    for split in (('train','test') if train_required else ('test',)):
        value=getattr(args,f'{split}_file')
        path=Path(value).resolve() if value else directory/split/f'{args.client_id}.npz'
        if split=='test' and args.test_from_bundle:
            raw=bundle.read(f'data/test/{args.client_id}.npz')
            import numpy as np
            decoded=np.load(io.BytesIO(raw),allow_pickle=True)['data'].tolist()
            location=f'{bundle.path}!data/test/{args.client_id}.npz'
        else:
            if not path.is_file(): raise FileNotFoundError(f'Original {split} shard required, no synthetic fallback: {path}')
            if path.suffix!='.npz': raise ValueError('Offline experiment uses explicit npz_only policy')
            raw=path.read_bytes(); decoded=read_shard(path); location=str(path)
        if len(decoded['x'])!=len(decoded['y']): raise ValueError('Image/label counts differ')
        data=decode_image_data(decoded,list_mode=TENSOR_LIST)
        if not data: raise ValueError('Empty original shard')
        x=torch.stack([p[0] for p in data]);y=torch.stack([p[1] for p in data])
        if x.shape[1:]!=(3,32,32) or not torch.isfinite(x).all() or y.min()<0 or y.max()>=classes:
            raise ValueError('Invalid original image/label content')
        r=dict(client_id=args.client_id,split=split,path=location,sha256=hashlib.sha256(raw).hexdigest(),count=len(data),
            decoded_x_sha256=tensor_digest(x),decoded_y_sha256=tensor_digest(y),
            labels={str(int(k)):int(v) for k,v in zip(*y.unique(return_counts=True))})
        if split=='test':
            original=bundle.read(f'data/test/{args.client_id}.npz',False)
            if original is not None and hashlib.sha256(original).hexdigest()!=r['sha256']:
                raise ValueError('Selected test shard differs from original bundled test file')
        if manifest is not None:
            matches=[m for m in manifest['files'] if m['client_id']==args.client_id and m['split']==split]
            if len(matches)!=1 or any(matches[0][k]!=r[k] for k in ('sha256','count','decoded_x_sha256','decoded_y_sha256')):
                raise ValueError(f'{split} fingerprint differs from supplied original training data manifest')
        result[split]=data;records[split]=r
    return result,dict(config=config,config_path=str(config_path.resolve()) if config_path.is_file() else f'{bundle.path}!data/config.json',
        config_sha256=sha256(config_path) if config_path.is_file() else hashlib.sha256(bundle.read('data/config.json')).hexdigest(),
        files=records,format_policy='npz_only',
        preprocessing='common.data_io TENSOR_LIST float32/int64; original order; no normalization or augmentation added',
        original_data_manifest=manifest,verified_source_data_fingerprint=verified_fingerprint,
        source_group_data_fingerprint=source['group']['context'].get('data_fingerprint') if source['group'] else None,
        source_per_shard_verification='original manifest verified against selected shards' if manifest is not None else
            'original manifest not supplied; bundle config/test checked, training provenance cannot be independently established')
