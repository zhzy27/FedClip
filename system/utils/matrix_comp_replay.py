"""Offline upload-based replay only; no federated module imports this file."""
import copy
import json
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
from utils import svd_r6_replay as replay,multi_group_svd as svd,matrix_tail_compensation as comp
from utils import fc1_tail_diagnostic as old,multi_svd_spectral as spectral,single_avg_proto as proto
from utils.cnn512_reporting import model_digest
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_clustering import digest_json
from utils.multi_rank_partition import protected_rng
from utils.whole_network_spectrum_diagnostic import error_metrics,truncated_model

BRANCHES=('H','Hplus','matrix')
LAYERS=comp.LAYERS


def resolve_source(cli):
    return replay.resolve_source(SimpleNamespace(h_server_file=cli.server_file,h_run=cli.source_run),'h')


def check_client(model,p,cid,args,source_payload,fingerprint,*,legacy_export=None):
    rate=replay.client_rates(args)[cid]
    if (p.get('role')!='ordinary_post_local_low_rank' or not p.get('prediction_checkpoint')
        or p.get('completed_train_count')!=5 or p.get('loop_round')!=4 or p.get('client_id')!=cid or p.get('model_id')!=0
        or model.ratio_LR!=rate or p.get('assigned_model_id',0)!=0 or p.get('svd_aggregation',{}).get('method')!='full_avg'):
        raise ValueError(f'Client {cid}: wrong R5 ordinary post-local role/ID/capacity/aggregation')
    if p.get('model_kwargs')!=svd.model_spec(args,rate,model)['model_kwargs'] or p.get('model_family')!=args.model_family:
        raise ValueError(f'Client {cid}: model structure mismatch')
    if p.get('svd_initialization')!=source_payload['svd_initialization']:raise ValueError(f'Client {cid}: initialization protocol mismatch')
    if 'initial_parameter_sha256' in p and p['initial_parameter_sha256']!=source_payload['initial_parameter_sha256']:
        raise ValueError(f'Client {cid}: initial fingerprint mismatch')
    if 'capacity_protocol' in p and p['capacity_protocol']!=svd.capacity_protocol(args):raise ValueError(f'Client {cid}: capacity protocol mismatch')
    for name,t in model.state_dict().items():
        value=p['state_dict'][name]
        if t.dtype!=value.dtype or not torch.isfinite(t).all() or not torch.equal(t,value):raise ValueError(f'Client {cid}: nonfinite/dtype/alias mismatch {name}')
    if p.get('state_schema')!={name:list(t.shape) for name,t in model.state_dict().items()}:raise ValueError(f'Client {cid}: state schema mismatch')
    capacity=svd.model_capacity(model,{n:t for n,t in source_payload['state_dict'].items() if not n.startswith(('base.','head.'))})
    declared=p.get('capacity',{})
    for key in ('rank_rate','layers','trainable_parameter_count'):
        if declared.get(key)!=capacity[key]:raise ValueError(f'Client {cid}: actual/declaration capacity {key} mismatch')
    usage=p.get('single_avg_prototype_usage') or {}
    if not isinstance(usage,dict):raise ValueError(f'Client {cid}: invalid prototype usage metadata')
    downloaded=usage.get('downloaded_model')
    if downloaded is not None and not isinstance(downloaded,dict):raise ValueError(f'Client {cid}: invalid download context')
    context=downloaded or {}
    direct=[]
    for location,fields in [('checkpoint',p),('download context',context)]:
        if fields.get('data_fingerprint') is not None:
            if fields['data_fingerprint']!=fingerprint:raise ValueError(f'Client {cid}: mismatched {location} data fingerprint')
            direct.append(location)
    # Missing is not mismatched. Legacy acceptance needs the checked SAME final
    # export envelope AND a later full upload-set Avg proof, never invented data.
    if not direct:
        if not legacy_export or not legacy_export.get('same_final_export_directory'):
            raise ValueError(f'Client {cid}: missing direct data fingerprint requires the same verified H final export directory')
        for key in ('initial_parameter_sha256','capacity_protocol','group_fingerprint','grouping_protocol','recovered_parameter_schema'):
            if key not in p or p[key]!=source_payload.get(key):raise ValueError(f'Client {cid}: legacy same-export {key} mismatch/missing')
        if p['group_fingerprint']!=legacy_export['group_fingerprint']:raise ValueError(f'Client {cid}: legacy grouping fingerprint mismatch')
    # Explicit provenance may never contradict the same server, even with a
    # correct direct data fingerprint (both bindings are checked independently).
    for key in ('group_fingerprint','grouping_protocol','recovered_parameter_schema'):
        if key in p and key in source_payload and p[key]!=source_payload[key]:raise ValueError(f'Client {cid}: same-export {key} mismatch')
    if 'training_loop_round' in usage and usage['training_loop_round']!=4:raise ValueError(f'Client {cid}: source usage round mismatch')
    if context and (context.get('initial_sha256')!=source_payload['initial_parameter_sha256'] or usage.get('training_loop_round')!=4):
        raise ValueError(f'Client {cid}: source/download context mismatch')
    if usage.get('mse_enabled',False) or usage.get('actual_used_library') is not None:raise ValueError(f'Client {cid}: source prototype training is unsupported')
    groups=p.get('optimizer',{}).get('param_groups',[])
    if len(groups)!=1 or any(groups[0].get(k)!=v for k,v in dict(lr=.005,momentum=0.,weight_decay=0.).items()):
        raise ValueError(f'Client {cid}: optimizer/source training mismatch')
    if len(groups[0]['params'])!=len(list(model.parameters())):raise ValueError(f'Client {cid}: optimizer parameter count mismatch')
    return dict(client_id=cid,rank_rate=rate,actual_ranks={n:getattr(model,n).rank for n in LAYERS},
        source_download_server_sha256=context.get('server_sha256'),data_fingerprint_binding='direct' if direct else 'legacy_same_export',
        direct_data_binding_locations=direct,legacy_missing_direct_fingerprint=not bool(direct),
        checkpoint_fingerprint_fields=dict(top_level_present='data_fingerprint' in p,top_level_value=p.get('data_fingerprint'),
            downloaded_model_present='downloaded_model' in usage,downloaded_model_is_null=downloaded is None,
            download_context_fingerprint_present='data_fingerprint' in context,download_context_fingerprint_value=context.get('data_fingerprint')),
        provenance_conclusion=('Direct checkpoint data binding verified' if direct else
            'Pending: same-export upload-set Avg proof; data independently verified against server run manifest'),
        trainable_parameter_count=sum(t.numel() for t in model.parameters()))


def legacy_export_context(payload,manifest,args,fingerprint,server_path,client_directory):
    grouping=manifest.get('grouping')
    if grouping is None:return None  # Directly bound older offline fixtures remain supported.
    ids=list(range(args.num_clients));context=grouping.get('context',{})
    expected_fp=digest_json(dict(protocol='fixed_all_client_Avg',clients=ids))
    if (grouping.get('groups')!=[ids] or grouping.get('client_to_model')!={str(i):0 for i in ids}
        or grouping.get('group_fingerprint')!=expected_fp or payload.get('group_fingerprint')!=expected_fp
        or payload.get('grouping_protocol')!=manifest.get('grouping_protocol')):
        raise ValueError('H server/manifest fixed K1 grouping mismatch')
    expected=dict(num_clients=args.num_clients,K=1,initial_parameter_sha256=payload['initial_parameter_sha256'],
        data_fingerprint=fingerprint,svd_capacity_protocol=svd.capacity_protocol(args),svd_initialization=svd.initialization_protocol(args))
    if any(context.get(k)!=v for k,v in expected.items()):raise ValueError('H manifest grouping context data/initialization/capacity mismatch')
    if payload.get('recovered_parameter_schema')!=manifest.get('parameter_schema'):raise ValueError('H server/manifest recovered schema mismatch')
    return dict(group_fingerprint=expected_fp,server_manifest_grouping_verified=True,
        same_final_export_directory=Path(client_directory).resolve()==Path(server_path).resolve().parent,
        direct_checkpoint_data_binding_assumed=False)


def load_source(cli):
    path=resolve_source(cli);m=json.loads((path.parent/'manifest.json').read_text(encoding='utf-8'))
    a=replay.validate_manifest(m,'h',5);mother,p=svd.load_checkpoint(path,'cpu')
    if (p.get('role')!='post_aggregation_full_server' or p.get('completed_train_count')!=5 or p.get('model_id')!=0
        or p.get('svd_aggregation',{}).get('method')!='full_avg' or mother.ratio_LR!=1.):raise ValueError('Need H R5 post-Avg FULL server')
    for key,mkey in [('model_kwargs','model_kwargs'),('model_family','model_family'),('svd_initialization','svd_initialization'),
        ('capacity_protocol','svd_capacity_protocol'),('initial_parameter_sha256','initial_parameter_sha256')]:
        if p.get(key)!=m.get(mkey):raise ValueError('Server/manifest mismatch: '+key)
    if p['capacity_protocol']!=svd.capacity_protocol(a) or p['model_kwargs']!=dict(in_features=3,num_classes=100,n_kernels=16,ratio_LR=1.,input_size=32):
        raise ValueError('Need original full-width CIFAR100 Hyper_CNN_512 and five tiers')
    for n,t in mother.state_dict().items():
        if t.dtype!=p['state_dict'][n].dtype or not torch.isfinite(t).all() or not torch.equal(t,p['state_dict'][n]):raise ValueError('Invalid full source tensor '+n)
    if model_digest(svd.initial_model(a))!=p['initial_parameter_sha256']:raise ValueError('Source S0 initialization fingerprint cannot be reproduced')
    source_effective=copy.deepcopy(vars(a));a.data_root=cli.data_root;a.dataset=cli.dataset;a.data_subdir=cli.data_subdir
    if cli.dataset!='Cifar100':raise ValueError('Need original Cifar100 data')
    a.data_formats='npz_only';a.data_expected_partition='pat';data=audit_image_clients(a)
    files=[{k:r[k] for k in replay.FILE_FIELDS} for r in data['files']]
    expected=[{k:r[k] for k in replay.FILE_FIELDS} for r in m['data_manifest']['files']]
    if data['config_sha256']!=m['data_manifest']['config_sha256'] or files!=expected:raise ValueError('Actual config/files/decoded data fingerprints differ from H R5')
    fingerprint=digest_json(dict(config_sha256=data['config_sha256'],files=files))
    backend={k:m['environment'][k] for k in replay.BACKEND}
    torch.use_deterministic_algorithms(backend['deterministic']);torch.backends.cudnn.deterministic=backend['cudnn_deterministic']
    torch.backends.cudnn.benchmark=backend['cudnn_benchmark'];torch.backends.cuda.matmul.allow_tf32=backend['matmul_tf32'];torch.backends.cudnn.allow_tf32=backend['cudnn_tf32']
    a.device='cuda:0' if cli.device=='cuda' else 'cpu'
    if cli.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Need exactly the requested visible GPU or --device cpu')
        torch.cuda.set_device(0)
    a.initial_sha256=p['initial_parameter_sha256'];a.data_fingerprint=fingerprint;a.offline_loop_round=5
    a.multi_svd_spectral_mode='none';a.multi_svd_five_diagnostics=0;a.multi_rank_spectrum=0;a.multi_svd_diagnostics=0
    a.multi_svd_avg_train_samples=[next(r['count'] for r in data['files'] if r['client_id']==i and r['split']=='train') for i in range(a.num_clients)]
    folder=Path(cli.clients_dir).expanduser().resolve() if cli.clients_dir else path.parent
    clients={cid:folder/f'Client_{cid}_model_0.pt' for cid in range(a.num_clients)}
    if any(not file.is_file() for file in clients.values()):raise FileNotFoundError('Need exactly the same-run 20 ordinary R5 Client_i_model_0.pt inputs; no fallback to other rounds')
    export=legacy_export_context(p,m,a,fingerprint,path,folder)
    source=dict(server_file=str(path),server_file_sha256=sha256(path),manifest_file=str(path.parent/'manifest.json'),
        manifest_file_sha256=sha256(path.parent/'manifest.json'),source_server_state_sha256=model_digest(mother),
        initial_parameter_sha256=a.initial_sha256,data_fingerprint=fingerprint,source_effective=source_effective,
        source_environment=m['environment'],source_fixture_note=m.get('synthetic_fixture'),client_directory=str(folder),legacy_export_context=export)
    return mother.to(a.device),p,a,data,clients,source


def cohorts(args,target):
    result={'all':list(range(args.num_clients)),'C0' if target==0 else f'C{target}':[target]}
    for level in range(len(args.multi_svd_effective_ratios)):
        ids=[cid for cid,i in enumerate(args.multi_svd_effective_assignment) if i==level]
        if ids:result[f'capacity_{args.multi_svd_effective_ratios[level]:g}']=ids
    return result


def summarize(rows,groups):
    lookup={};output=[]
    for r in rows:lookup.setdefault((r['round'],r['branch'],r['phase'],r['epoch'],r['split']),[]).append(r)
    for (round_id,branch,phase,epoch,split),members in sorted(lookup.items()):
        if len(members)!=len(groups['all']) or {r['client_id'] for r in members}!=set(groups['all']):raise ValueError('Missing/duplicate client endpoints')
        for scope,ids in groups.items():
            chosen=[r for r in members if r['client_id'] in ids];n=sum(r['samples'] for r in chosen)
            output.append(dict(round=round_id,branch=branch,phase=phase,epoch=epoch,split=split,scope=scope,member_ids=ids,
                correct=sum(r['correct'] for r in chosen),samples=n,accuracy=sum(r['correct'] for r in chosen)/n,
                ce=sum(r['ce']*r['samples'] for r in chosen)/n,weighted_by='split sample count',accuracy_unit='fraction'))
    return output


def contrasts(metrics):
    lookup={(r['round'],r['branch'],r['phase'],r['epoch'],r['split'],r['scope']):r for r in metrics};output=[]
    for key,a in lookup.items():
        round_id,branch,phase,epoch,split,scope=key
        if branch!='matrix':continue
        for other in ('Hplus','H'):
            b=lookup[round_id,other,phase,epoch,split,scope]
            row=dict(round=round_id,contrast='matrix_minus_'+other,phase=phase,epoch=epoch,split=split,scope=scope,
                accuracy_difference=a['accuracy']-b['accuracy'],CE_difference=a['ce']-b['ce'],accuracy_unit='fraction')
            if phase=='local':
                x=lookup[round_id,branch,phase,0,split,scope];y=lookup[round_id,other,phase,0,split,scope]
                row.update(accuracy_gap_change_since_epoch0=row['accuracy_difference']-(x['accuracy']-y['accuracy']),
                    CE_gap_change_since_epoch0=row['CE_difference']-(x['ce']-y['ce']))
            output.append(row)
    # Each branch's R7 actual intervention versus its OWN untreated full-Avg mother.
    for r in metrics:
        if r['round']==7 and r['phase']=='local' and r['epoch']==0:
            raw=lookup[7,r['branch'],'ordinary_counterfactual',0,r['split'],r['scope']]
            output.append(dict(round=7,contrast=r['branch']+'_actual_minus_own_ordinary',phase='instant_download',epoch=0,
                split=r['split'],scope=r['scope'],accuracy_difference=r['accuracy']-raw['accuracy'],CE_difference=r['ce']-raw['ce'],accuracy_unit='fraction'))
    return output


def prepare_round(averages,upload_sets,ranks,samples,order,args,cli,round_id,costs,segments,boundaries,spectra,downloads):
    """One common native download SVD per mother; capacities take its prefixes."""
    templates={};mothers={};ordinary={};highest=max(replay.client_rates(args))
    for branch in BRANCHES:
        avg=averages[branch];tick=time.perf_counter()
        if branch=='matrix':
            effective,records=comp.compensate_model(avg,upload_sets[branch],ranks,samples,order,
                rank_rtol=cli.matrix_rank_rtol,rank_atol=cli.matrix_rank_atol)
            for r in records:
                identity=dict(round=round_id,layer=r['layer'])
                segments.extend(dict(identity,**s) for s in r['segments'])
                boundaries.extend(dict(identity,**b) for b in r['boundaries'])
                spectra.extend(dict(identity,rank=i+1,before=a,after=b) for i,(a,b) in enumerate(zip(r['singular_before'],r['singular_after'])))
                costs.append(dict(kind='matrix_compensation_and_boundary_observations',**identity,seconds=r['total_seconds']))
            raw,_,_=replay.paired_download(effective,highest,args,0,False)
            templates[branch]=raw;mothers[branch]=effective
            if round_id==7:ordinary[branch]=replay.paired_download(avg,highest,args,0,False)[0]
        elif branch=='Hplus':
            raw,repaired,receipt=replay.paired_download(avg,highest,args,0,True)
            templates[branch]=repaired;mothers[branch]=avg
            if round_id==7:ordinary[branch]=raw
            for row in receipt:
                detail=row['rank_corrections'];base={k:v for k,v in row.items() if k!='rank_corrections'}
                downloads.extend(dict(round=round_id,branch=branch,**base,**item) for item in detail)
        else:
            templates[branch]=replay.paired_download(avg,highest,args,0,False)[0];mothers[branch]=avg
        costs.append(dict(kind='prepare_download_templates',round=round_id,branch=branch,seconds=time.perf_counter()-tick,
            note='includes matrix compensation/observation costs where applicable; do not sum overlapping costs'))
    return templates,mothers,ordinary


def render(root):
    import subprocess
    import sys
    with (root/'render.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve().parents[1]/'plot_matrix_comp_replay.py'),
            '--output-root',str(root)],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Plotting failed; tables retained. See '+str(root/'render.log'))


def write_tables(root,tables):
    for name,rows in tables.items():
        write_json(root/(name+'.json'),rows);old.write_csv(root/(name+'.csv'),rows)


def check_source_Avg(average,source,atol,rtol,*,check=True):
    expected=dict(source.named_parameters());actual=dict(average.named_parameters())
    if actual.keys()!=expected.keys():raise ValueError('Recomputed Avg parameter schema differs from saved server')
    return [dict(parameter=name,**error_metrics(value,expected[name],atol,rtol,'R5 Avg '+name,check=check)) for name,value in actual.items()]


def run(cli):
    if cli.threads<1:raise ValueError('threads must be positive')
    for value in (cli.source_atol,cli.source_rtol,cli.matrix_rank_atol):
        if not math.isfinite(value) or value<0:raise ValueError('Tolerances must be finite nonnegative')
    if cli.matrix_rank_rtol is not None and (not math.isfinite(cli.matrix_rank_rtol) or cli.matrix_rank_rtol<0):raise ValueError('Invalid matrix-rank-rtol')
    root=Path(cli.output_root).expanduser().resolve();source_path=resolve_source(cli)
    protected=[source_path.parent,(Path(cli.data_root)/cli.dataset/cli.data_subdir).expanduser().resolve()]
    if cli.clients_dir:protected.append(Path(cli.clients_dir).expanduser().resolve())
    scratch=Path(cli.temp_root).expanduser().resolve() if cli.temp_root else root
    for path in [root]+([scratch] if cli.temp_root else []):
        if any(path==p or p in path.parents or path in p.parents for p in protected):raise ValueError('Output/scratch must not overlap source or data')
    if root.exists():raise FileExistsError('Use a new output root: '+str(root))
    replay.preflight_curves();torch.set_num_threads(cli.threads)
    source,p,args,data,files,provenance=load_source(cli)
    if not 0<=cli.target_client_id<args.num_clients:raise ValueError('Target client out of range')
    samples={r['client_id']:r['count'] for r in data['files'] if r['split']=='train'}
    selector=np.random.RandomState(args.seed);orders={}
    for count in range(1,7):
        order=selector.choice(np.arange(args.num_clients),args.num_clients,replace=False).tolist()
        if count in (5,6):orders[count]=order
    root.mkdir(parents=True,exist_ok=False)
    for branch in BRANCHES:(root/branch).mkdir()
    manifest=dict(status='running',role='offline_S5_upload_replay_R6_R7',requested=vars(cli),source=provenance,
        environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),branches=list(BRANCHES),
        source_completed_aggregations=5,training_rounds=[6,7],epochs=list(range(6)),effective=vars(args),
        ordinary_Avg='original aggregate_group_updates/group_avg_only, sample mass, original ordered full recovered uploads',
        matrix_rule='original W_avg + sum((1-q)*residual_SVD_component); only projector prefixes advance, subgroup matrices never modified',
        matrix_rank_tolerance='float64 arithmetic; default max(shape)*eps(float64)*sigma_max(original subgroup B); upload-precision threshold is observation-only',
        unified_download='one native SVD per branch/download intervention, then all client capacities truncate its common ordered factors',
        R7_ordinary_counterfactual='own branch ordinary full Avg BEFORE any current intervention; inference only; H reuses epoch0',
        batch_protocol='existing single_avg_proto.loader_seed(seed,client,loop_round=5 or6)',
        training_RNG='stable isolated seed per round/client, restored across branches; NOT resumption of source-process RNG',
        aggregation_order=orders,order_protocol='isolated original NumPy RandomState(seed) full-participation choice, fifth/sixth calls; not recovered source process RNG',
        source_Avg_anchor='saved S5 serves only as original delta-Avg arithmetic anchor when rechecking R5 uploads; roundoff against saved result is measured explicitly',
        outputs='diagnostic tables and fixed-epoch curves only; transient checkpoints and restored uploads cleaned',
        accuracy_unit='fraction',CE_unit='nats per sample; endpoint CE excludes F',R8_aggregation=False)
    write_json(root/'manifest.json',manifest);write_json(root/'data_manifest.json',data)
    hashes={source_path:sha256(source_path),source_path.parent/'manifest.json':sha256(source_path.parent/'manifest.json')}
    hashes.update({file:sha256(file) for file in files.values()})
    hashes.update({Path(r['path']):r['sha256'] for r in data['files']});hashes[Path(data['config_path'])]=data['config_sha256']
    endpoints=[];epochs=[];training=[];batches=[];aggregations=[];costs=[];segments=[];boundaries=[];spectra=[];downloads=[]
    receipts=[];ranks={n:{} for n in LAYERS};groups=cohorts(args,cli.target_client_id)
    if cli.temp_root:scratch.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='matrix_comp_',dir=scratch) as temporary:
            temp=Path(temporary);pool={}
            # Validate every ordinary prediction checkpoint before training any replay task.
            for cid,file in files.items():
                local,cp=svd.load_checkpoint(file,args.device)
                row=check_client(local,cp,cid,args,p,args.data_fingerprint,legacy_export=provenance['legacy_export_context'])
                restored=svd.recover_parameters(local)
                if set(restored)!=set(dict(source.named_parameters())):raise ValueError('Restored upload schema mismatch')
                upload=temp/f'R5_upload_{cid}.pt';torch.save(restored,upload);pool[cid]=upload
                receipts.append(dict(row,file=str(file),file_sha256=hashes[file],checkpoint_state_sha256=tensor_fingerprint(local.state_dict()),
                    restored_upload_sha256=tensor_fingerprint(restored)))
                for name in LAYERS:ranks[name][cid]=row['actual_ranks'][name]
                del local,cp,restored
            bound={r['source_download_server_sha256'] for r in receipts if r['source_download_server_sha256'] is not None}
            if len(bound)>1:raise ValueError('R5 clients were downloaded from different source servers')
            tick=time.perf_counter();average,receipt=replay.aggregate_once(source,pool,samples,orders[5])
            checks=check_source_Avg(average,source,cli.source_atol,cli.source_rtol,check=False)
            passed=all(r['passed'] for r in checks)
            direct_ids=[r['client_id'] for r in receipts if r['data_fingerprint_binding']=='direct']
            legacy_ids=[r['client_id'] for r in receipts if r['data_fingerprint_binding']=='legacy_same_export']
            if passed:
                for row in receipts:
                    if row['data_fingerprint_binding']=='legacy_same_export':
                        row['provenance_conclusion']='Same-export upload set reproduced saved Avg; data independently verified against server run manifest. Checkpoint has no direct data fingerprint.'
            write_json(root/'source_validation.json',dict(clients=receipts,Avg_checks=checks,all_passed=passed,
                direct_data_bound_client_ids=direct_ids,legacy_compatible_client_ids=legacy_ids,
                server_data_manifest_and_actual_shards_verified=True,upload_set_recomputed_Avg_verified=passed,
                shared_R5_download_server_sha256=next(iter(bound),None),source_server_sha256=model_digest(source),recomputed_Avg_sha256=model_digest(average)))
            if not passed:raise ValueError('R5 upload-set Avg differs from saved server; see source_validation.json: '+str([r for r in checks if not r['passed']]))
            aggregations.append(dict(round=5,branch='shared_input',seconds=time.perf_counter()-tick,**receipt))
            averages={b:average for b in BRANCHES};pools={b:pool for b in BRANCHES}
            for round_id in (6,7):
                args.offline_loop_round=round_id-1
                templates,mothers,ordinary=prepare_round(averages,pools,ranks,samples,orders[round_id-1],args,cli,round_id,
                    costs,segments,boundaries,spectra,downloads)
                next_uploads={b:{} for b in BRANCHES}
                for cid in range(args.num_clients):
                    print(f'========== offline R{round_id}: client {cid}, three paired local5 tasks ==========',flush=True)
                    own=read_image_client(args,cid,True);test=read_image_client(args,cid,False);rate=replay.client_rates(args)[cid]
                    models={b:truncated_model(templates[b],rate) for b in BRANCHES}
                    seed=int(digest_json(dict(purpose='offline_matrix_comp_replay',seed=args.seed,round=round_id,client_id=cid))[:16],16)%(2**63)
                    random.seed(seed);np.random.seed(seed%(2**32));torch.random.default_generator.manual_seed(seed)
                    if args.device.startswith('cuda'):torch.cuda.default_generators[0].manual_seed(seed)
                    base_rng=old.rng_snapshot(models['H']);canonical=None;ending=None
                    for branch in BRANCHES:
                        local=models.pop(branch);identity=dict(round=round_id,branch=branch,client_id=cid,rank_rate=rate,
                            actual_layer_ranks={n:getattr(local,n).rank for n in LAYERS})
                        if identity['actual_layer_ranks']!={n:ranks[n][cid] for n in LAYERS}:raise ValueError('Replay changed actual capacity')
                        if round_id==7 and branch!='H':
                            counter=truncated_model(ordinary[branch],rate)
                            for split,rows in [('train',own),('test',test)]:
                                row,_=old.evaluate(counter,rows,args.test_batch_size,args.device)
                                endpoints.append(dict(identity,phase='ordinary_counterfactual',epoch=0,split=split,**row))
                            del counter
                        folder=temp/f'R{round_id}_{branch}';folder.mkdir(exist_ok=True)
                        initial=tensor_fingerprint(local.state_dict());old.restore_rng(base_rng)
                        result,client=replay.train_branch(local,mothers[branch],args,cid,own,test,folder)
                        if canonical is None:canonical=result['batches'];ending=result['final_rng_sha256']
                        if result['batches']!=canonical or result['initial_rng_sha256']!=old.rng_fingerprint(base_rng) or result['final_rng_sha256']!=ending:
                            raise RuntimeError('Three-branch batch or training RNG pairing failed')
                        endpoints.extend(dict(identity,phase='local',**row) for row in result['endpoints'])
                        if round_id==7 and branch=='H':
                            endpoints.extend(dict(identity,phase='ordinary_counterfactual',**row) for row in result['endpoints'] if row['epoch']==0)
                        epochs.extend(dict(identity,**row) for row in result['epochs'])
                        training.append(dict(result['training'],**identity,initial_model_sha256=initial,final_model_sha256=result['model_sha256'],
                            upload_sha256=tensor_fingerprint(client.full_uploads[0])))
                        batches.append(dict(identity,loader_seed=proto.loader_seed(args,cid,round_id-1),training_seed=seed,
                            batch_sha256=result['batch_sha256'],initial_rng_sha256=result['initial_rng_sha256'],final_rng_sha256=result['final_rng_sha256'],
                            epoch_batch_sha256={str(e):digest_json([r for r in result['batches'] if r['epoch']==e]) for e in range(1,6)},actual_steps=len(result['batches'])))
                        if round_id==6:
                            upload=folder/f'upload_{cid}.pt';torch.save(client.full_uploads[0],upload);next_uploads[branch][cid]=upload
                        (folder/f'Client_{cid}_model_0.pt').unlink()
                        costs.append(dict(kind='local5_training_with_endpoints',round=round_id,branch=branch,client_id=cid,
                            seconds=result['wall_seconds'],endpoint_seconds=result['endpoint_seconds']))
                        del client,local,result
                del templates,ordinary,models
                if round_id==6:
                    averages={}
                    for branch in BRANCHES:
                        tick=time.perf_counter();averages[branch],receipt=replay.aggregate_once(mothers[branch],next_uploads[branch],samples,orders[6])
                        aggregations.append(dict(round=6,branch=branch,seconds=time.perf_counter()-tick,**receipt))
                    pools=next_uploads
                del mothers
            # No R8 aggregation or extra training.
        metrics=summarize(endpoints,groups);pairs=contrasts(metrics)
        tables=dict(endpoints=endpoints,cohort_endpoints=metrics,epoch_training=epochs,local_training=training,batch_fingerprints=batches,
            comparisons=pairs,aggregations=aggregations,costs=costs,compensation_segments=segments,capacity_boundaries=boundaries,
            compensation_spectrum=spectra,all_tail_downloads=downloads)
        write_tables(root,tables)
        for branch in BRANCHES:
            write_tables(root/branch,{n:[r for r in rows if r.get('branch', 'matrix' if n.startswith(('compensation_','capacity_')) else None)==branch]
                for n,rows in tables.items()})
        render(root)
        lines=['# S5 upload replay: R6 and R7 local training','',
            'Main contrast: matrix compensation minus existing all_tail. Accuracy gaps are percentage points. CE is pure endpoint CE.',
            'Both R6 and R7 execute local5. R7 ordinary counterfactual uses each branch\'s own unmodified full Avg before current intervention. No counterfactual training.','',
            '| round | epoch | all-client test accuracy gap (pp) | test CE gap |','| --- | ---: | ---: | ---: |']
        for r in pairs:
            if r['contrast']=='matrix_minus_Hplus' and r['phase']=='local' and r['scope']=='all' and r['split']=='test':
                lines.append(f"| R{r['round']} | {r['epoch']} | {100*r['accuracy_difference']:.6f} | {r['CE_difference']:.8g} |")
        lines+=['','Subspace overlaps are normalized squared Frobenius overlaps, sign/rotation invariant; nearly degenerate boundary flags are reported.',
            'Residual components can reorder across capacity boundaries after addition. Numerical-rank shortfalls are not filled or carried to later bands.',
            'This is an offline replay, not a resumed source RNG trajectory or evidence of real accuracy improvement.']
        (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        if any(sha256(file)!=value for file,value in hashes.items()):raise RuntimeError('Input source/data modified during replay')
        if model_digest(source)!=provenance['source_server_state_sha256']:raise RuntimeError('Source server modified')
        manifest.update(status='complete',actual_training_tasks=len(training),actual_SGD_steps=sum(r['train_batches'] for r in training),
            actual_SGD_sample_visits=sum(r['processed_train_samples'] for r in training),all_batches_and_RNG_paired=True,
            source_artifacts_unchanged=True,source_Avg_validated=True,transient_files_cleaned=True,model_exports=0,
            original_download_SVD_calls=21,extra_R7_ordinary_training_steps=0,
            per_branch_directories={b:str(root/b) for b in BRANCHES})
        write_json(root/'manifest.json',manifest)
        for branch in BRANCHES:write_json(root/branch/'manifest.json',dict(manifest,branch=branch))
        print('Completed paired R6/R7 tables and curves:',root,flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(root/'manifest.json',manifest);raise
