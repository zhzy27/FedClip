"""Actual-main head-role smoke; synthetic fixtures only, serial workers.

--width 16 uses the formal convolution width. --width 2 explicitly scales only
the synthetic fixture's convolution width; 512-D features/100-way fc3 remain.
No production model factory or real shards are modified by this test hook.
"""
import argparse
import csv
from contextlib import ExitStack
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


def verify_tables(root):
    """Check exported cells without loading models or consuming training RNG."""
    counts={}
    for stem in ('head_aggregation','last_epoch_training','client_metrics'):
        rows=json.loads((root/f'{stem}.json').read_text())
        with (root/f'{stem}.csv').open(newline='',encoding='utf-8') as stream:cells=list(csv.DictReader(stream))
        assert len(rows)==len(cells)
        for row,cell in zip(rows,cells):
            for name,value in row.items():
                if isinstance(value,(dict,list)):assert json.loads(cell[name])==value,(stem,name)
                else:assert cell[name]==('' if value is None else str(value)),(stem,name,value,cell[name])
        counts[stem]=len(rows)
    return counts


def width_patch(width):
    from utils import multi_group_svd as svd
    original=svd.model_spec
    def spec(args,ratio=1.,model=None):
        value=original(args,ratio,model)
        if model is None:value['model_kwargs']['n_kernels']=width
        return value
    return patch.object(svd,'model_spec',spec)


def source(out,width):
    import torch
    from common.client_data import audit_image_clients
    from utils.multi_group_svd import initial_model,DEFAULT_RATIOS
    from utils.cnn512_reporting import model_digest
    from utils.multi_group_clustering import grouping_context,digest_json
    from utils.multi_group_protocol import grouping_protocol
    torch.set_num_threads(2)
    a=SimpleNamespace(algorithm='FedMultiGroupProj',multi_svd=True,model_family='Decom_CNN-5-512',num_classes=100,
        seed=0,num_clients=20,num_server_models=3,multi_capacities=[7,7,6],local_learning_rate=.005,batch_size=16,
        local_epochs=5,local_grad_clip=10,multi_group_basis='full_weights',multi_warmup_rounds=0,multi_history_window=None,
        multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=.15,multi_svd_effective_ratios=list(DEFAULT_RATIOS),
        multi_svd_effective_assignment=[i%5 for i in range(20)],is_regular=1,regular_lamda=.001,
        data_root=str(out/'data'),data_subdir='dir_fixture',dataset='Cifar100',cnn512_control=True,data_formats='npz_only',
        partition='dir',dir_alpha=.5,class_per_client=20,data_expected_partition='dir',data_expected_alpha=.5)
    with width_patch(width):context=grouping_context(a,audit_image_clients(a),model_digest(initial_model(a)))
    groups=[list(range(7)),list(range(7,14)),list(range(14,20))];matrix=torch.eye(20).tolist();proto=grouping_protocol(a)
    row=dict(loop_round=0,completed_train_count=1,matrix=matrix,matrix_sha256=digest_json(matrix),diagnostics=dict(per_model_cosine_matrices=[matrix]*3))
    evidence=dict(protocol=proto,rounds=[row],final_similarity_sha256=digest_json(matrix));evidence['evidence_sha256']=digest_json(evidence)
    payload=dict(schema_version=2,context=context,context_sha256=digest_json(context),groups=groups,
        group_fingerprint=digest_json(dict(context=context,groups=groups)),client_to_model={str(i):k for k,g in enumerate(groups) for i in g},
        grouping_protocol=proto,evidence=evidence,similarity_matrix=matrix,completed_train_count=1,diagnostics={})
    (out/'pair.json').write_text(json.dumps(payload))


def worker(out,label,width):
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from common.experiment import sha256
    from utils import multi_group_head as heads
    from utils.group_projection import aggregate_group_updates
    from utils.head_snapshot import tensor_fingerprint
    from utils.multi_group_proto import packet_sha,decode_packet
    from utils import multi_last_epoch_proto as post
    from utils.multi_group_svd import load_checkpoint
    from fedfews_compat.observe import rng_digest
    from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
    from flcore.trainmodel.models import Hyper_CNN_512
    import flcore.clients.clientMultiGroupProj as client_module
    jobs=json.loads((out/'jobs.json').read_text());mode=jobs[label]['head'];run=out/label
    helper_update=jobs[label].get('helper_head_update','train')
    def backed(value,path):
        import numpy as np
        assert value.device.type=='cpu'
        array=value.detach().numpy()
        assert array.flags.c_contiguous or array.flags.f_contiguous
        path.parent.mkdir(parents=True,exist_ok=True)
        mapped=np.lib.format.open_memmap(path,mode='w+',dtype=array.dtype,shape=array.shape,
            fortran_order=array.flags.f_contiguous and not array.flags.c_contiguous)
        mapped[:]=array;mapped.flush();result=torch.from_numpy(mapped)
        assert result.stride()==value.stride() and torch.equal(result,value)
        return result
    def back_uploads(client):
        # Test-only, lossless CPU storage backing to avoid exhausting this host's
        # commit limit. Production downloads/SGD/recovery/kernels are unchanged.
        if jobs[label].get('upload_backing','memory')!='mmap':return
        directory=run/'upload_backing'/str(client.download_round)/str(client.id)
        directory.mkdir(parents=True,exist_ok=True)
        for k,values in enumerate(client.full_uploads):
            for name,value in list(values.items()):
                assert value.device.type=='cpu' and not value.requires_grad
                values[name]=backed(value,directory/f'{k}_{name}.npy')
    context=dict(training=False);events=[];batches=[];created=[];libraries=[];forward_counts={}
    class TraceLoader(DataLoader):
        def __iter__(self):
            for x,y in super().__iter__():
                if context['training']:batches.append([context['loop'],context['cid'],y.tolist()])
                yield x,y
    original_forward=Hyper_CNN_512.forward
    def forward(model,*args,**kwargs):
        if context['training']:
            key=(context['loop'],context['cid']);forward_counts[key]=forward_counts.get(key,0)+1
        return original_forward(model,*args,**kwargs)
    original_init=FedMultiGroupProj.__init__
    def initialize(server,*args,**kwargs):
        original_init(server,*args,**kwargs);created.append(server)
        assert all(m.conv1.out_channels==width for m in server.global_model_set)
        events.append(dict(kind='initial',model=server.initial_hash,rng=rng_digest()))
        for c in server.clients:
            native_download,native_train=c.set_parameters,c.train
            def download(servers,loop,c=c,native=native_download):
                assert c.multi_role_round==loop  # Native server bound roles before this call.
                previous=[heads.clone_head(m) for m in c.model_set]
                native(servers,loop)
                for k,m in enumerate(c.model_set):
                    private=mode=='group_only_local' and not c.multi_head_roles[k]
                    expected=previous[k] if private and loop else c.multi_initial_head if private else heads.clone_head(servers[k])
                    assert heads.fingerprint(heads.clone_head(m))==heads.fingerprint(expected)
                    assert m.ratio_LR==(.9,.37,.35,.25,.15)[c.id%5]
                    assert [id(p) for p in m.parameters() if p.requires_grad]==[id(p) for g in c.optimizer_set[k].param_groups for p in g['params']]
            def train(c=c,native=native_train):
                context.update(training=True,cid=c.id,loop=server.loop_round)
                try:result=native()
                finally:context['training']=False
                assert forward_counts[server.loop_round,c.id]==15
                for k,row in enumerate(c.train_stats):
                    assert row['local_epochs']==5 and row['train_batches']==5 and row['prototype_collected_samples']==c.train_samples
                    assert row['processed_train_samples']==5*c.train_samples and row['effective_grad_clip']==10
                    assert row['role']==('group' if c.multi_head_roles[k] else 'helper')
                    assert ('fc3.weight' in c.full_uploads[k])==bool(row['head_upload_enabled'])
                    frozen=helper_update=='frozen' and not c.multi_head_roles[k]
                    assert (row['downloaded_head_sha256']==row['post_local_head_sha256'])==frozen
                    if helper_update=='frozen':
                        assert row['head_optimizer_parameter_count']==(0 if frozen else 2)
                        assert row['head_gradient_produced']==(not frozen)
                        assert all(p.requires_grad for p in c.model_set[k].parameters())
                        if frozen:
                            assert row['head_update_norm']==0. and row['head_unchanged_bitwise']
                            assert all(p.grad is None for p in c.model_set[k].fc3.parameters())
                            assert row['backbone_pre_clip_grad_norm_mean_per_step']==row['total_pre_clip_grad_norm_mean_per_step']
                    assert row['diagnostic_parameter_scope']=='backbone_excluding_fc3'
                    payload=torch.load(Path(c.save_folder_name)/f'Client_{c.id}_model_{k}.pt',weights_only=False)
                    assert payload['multi_head_policy']['post_local_head_sha256']==row['post_local_head_sha256']
                    assert 'fc3.weight' in payload['state_dict']  # Actual private head kept in prediction model.
                events.append(dict(kind='local',loop=server.loop_round,cid=c.id,
                    model=[tensor_fingerprint(m.state_dict()) for m in c.model_set],
                    uploads=[tensor_fingerprint(p) for p in c.full_uploads],rng=rng_digest()))
                back_uploads(c)
                return result
            c.set_parameters=download;c.train=train
        native_aggregate=server.aggregate_parameters
        def aggregate():
            expected=[];sample_total=sum(c.train_samples for c in server.clients)
            samples={c.id:c.train_samples/sample_total for c in server.clients}
            before=[tensor_fingerprint(p) for c in server.clients for p in c.full_uploads]
            for k,model in enumerate(server.global_model_set):
                bb,head=heads.split(model);group=server.groups[k]
                factory=lambda:iter((c.id,samples[c.id],{n:c.full_uploads[k][n] for n in bb}) for c in server.selected_clients)
                values,metrics,rows,_=aggregate_group_updates(bb,factory,samples,group,'group_projection_softmax',.5)
                weights={r['client_id']:r['aggregation_weight'] for r in rows} if mode=='shared' else {c.id:c.train_samples/sum(server.clients[i].train_samples for i in group) if c.id in group else 0. for c in server.clients}
                h={n:torch.zeros_like(p) for n,p in head.items()}
                for c in server.selected_clients:
                    if mode!='shared' and c.id not in group:continue
                    for n in h:h[n].add_(c.full_uploads[k][n].to(h[n])*weights[c.id])
                expected.append(dict(values,**h))
            before_rng=rng_digest();result=native_aggregate();assert rng_digest()==before_rng
            for k,model in enumerate(server.global_model_set):assert all(torch.equal(p,expected[k][n]) for n,p in model.named_parameters())
            assert before==[tensor_fingerprint(p) for c in server.clients for p in c.full_uploads]
            assert all(abs(r['head_weight_sum']-1.)<=1e-12 for r in server.multi_head_records)
            events.append(dict(kind='aggregate',loop=server.loop_round,model=[tensor_fingerprint(m.state_dict()) for m in server.global_model_set],rng=rng_digest()))
            return result
        server.aggregate_parameters=aggregate
    original_finish=post.finish_round
    def finish(server):
        original_finish(server);libraries.append(packet_sha(server.post_proto_packet))
        values,meta=decode_packet(server.post_proto_packet,'last_epoch_columns_library')
        assert values['prototypes'].shape==(3,100,512)
    sys.argv=['main.py','-did','0','-algo','FedMultiGroupProj','-m','Decom_CNN-5-512','-data','Cifar100','-dev','cpu',
        '-nc','20','-ncl','100','-nsm','3','-t','1','-pv','0','-jr','1','--seed','0','-lr','.005','-lbs','16','-ls','5',
        '-gr','1','-eg','1','-is_regular','1','-regular_lamda','.001','--local_grad_clip','10','-clip_cpu_threads','2',
        '-niid','1','-pt','dir','-dir_alpha','.5','-cpc','20','--data-root',str(out/'data'),'--data-subdir','dir_fixture',
        '--data-formats','npz_only','--data-expected-partition','dir','--data-expected-alpha','.5','--control_test_batch_size','16',
        '--multi_svd','--multi_svd_init','low_rank_recovered','--multi_svd_init_rank_rate','.15','--multi_group_basis','full_weights',
        '--multi_svd_diagnostics',str(jobs[label].get('svd_diagnostics',1)),
        '--multi_group_sizes','7,7,6','--multi_warmup_rounds','0','--multi_group_mode','group_projection_softmax',
        '--multi_group_weight','.5','--group_softmax_tau','.2','--multi_last_epoch_protocol','--multi_group_reuse_policy','members_only',
        '--multi_group_file',str(out/'pair.json'),'--multi_proto_mode','last_epoch','--multi_proto_loss','mse',
        '--multi_proto_feature','base_output','--multi_proto_target_type','raw_mean','--multi_proto_lambda','1',
        '--multi_proto_start_round','2','--multi_proto_bank_mode','per_model','--multi_proto_scope','group_priority',
        '--multi_head_mode',mode,'--multi_projection_scope','backbone','-sfn',str(run/'checkpoints'),
        '--multi_helper_head_update',helper_update,
        '--run_log_dir',str(run/'logs'),'--h5_result_root',str(run/'h5'),'--final-model-root',str(run/'final_models'),'-exp_name',label]
    original_copy=shutil.copy2
    def compact(src,dst,*a,**kw):
        if str(src).endswith('.pt') and Path(src).is_file() and not Path(dst).exists():
            try:os.link(src,dst);return dst
            except OSError:pass  # e.g. external exFAT SSD: use ordinary copy.
        return original_copy(src,dst,*a,**kw)
    with ExitStack() as stack:
        for obj,key,value in [(FedMultiGroupProj,'__init__',initialize),(client_module,'DataLoader',TraceLoader),
            (Hyper_CNN_512,'forward',forward),(post,'finish_round',finish),(shutil,'copy2',compact)]:stack.enter_context(patch.object(obj,key,value))
        stack.enter_context(width_patch(width));runpy.run_path(str(ROOT/'system/main.py'),run_name='__main__')
    server=created[0];root=Path(server.report_dir());final=Path(server.final_model_dir());final_rng=rng_digest()
    rows=json.loads((root/'last_epoch_training.json').read_text());assert len(rows)==120
    table_counts=verify_tables(root)
    assert len(batches)==600 and sum(r['train_batches'] for r in rows)==600
    assert len(server.multi_head_records)==120
    assert all(r['multi_projection_scope']=='backbone' for r in rows)
    summary=json.loads((root/'primary_summary.json').read_text());curve=[r['post_local_accuracy'] for r in server.primary_records]
    assert summary['best']==max(curve) and summary['best_loop_round']==curve.index(max(curve))
    for point in server.primary_records:
        selected=[r for r in server.endpoint_records if r['loop_round']==point['loop_round'] and r['stage']=='post' and r['selected_by_membership']]
        assert point['post_local_accuracy']==sum(r['correct'] for r in selected)/sum(r['test_samples'] for r in selected)
    with h5py.File(server.args.save_file_paths[-1]) as hf:
        assert [json.loads(r) for r in hf['head_aggregation_json'][:]]==server.multi_head_records
        assert hf['rs_test_acc'][:].tolist()==curve
        assert json.loads(hf.attrs['multi_head_policy_json'])==server.multi_head_policy
    manifest=json.loads((final/'manifest.json').read_text());assert manifest['multi_head_policy']==server.multi_head_policy
    with torch.no_grad():
        for k,m in enumerate(server.global_model_set):
            restored,payload=load_checkpoint(final/f'Server_model_{k}.pt');assert tensor_fingerprint(restored.state_dict())==tensor_fingerprint(m.state_dict())
        for c in server.clients:
            for k,m in enumerate(c.model_set):
                restored,payload=load_checkpoint(final/f'Client_{c.id}_model_{k}.pt')
                assert tensor_fingerprint(restored.state_dict())==tensor_fingerprint(m.state_dict())
                assert torch.equal(restored.eval()(torch.zeros(1,3,32,32)),m.eval()(torch.zeros(1,3,32,32)))
                assert payload['multi_head_policy']['post_local_head_sha256']==heads.fingerprint(heads.clone_head(restored))
    (run/'verification.json').write_text(json.dumps(dict(status='PASS',mode=mode,actual_width=width,events=events,batches=batches,
        helper_head_update=helper_update,
        test_only_upload_storage_backing=jobs[label].get('upload_backing','memory'),
        csv_json_table_counts=table_counts,
        final_rng=final_rng,prototype_packets=libraries,summary=summary,train_tasks=120,SGD_steps=600,models_reloaded=63,
        backbone_and_head_oracle_max_error=0.,head_upload_mask_verified=True),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--width',type=int,default=16)
    p.add_argument('--worker');p.add_argument('--make-source',action='store_true');p.add_argument('--keep-models',action='store_true')
    p.add_argument('--svd-diagnostics',type=int,choices=[0,1],default=1,help='Optional read-only recovered-start distance cache; 0 reduces fixture peak memory')
    p.add_argument('--upload-backing',choices=['memory','mmap'],default='memory',help='Test-only lossless CPU recovered-upload storage backing; production code unchanged')
    p.add_argument('--helper-head-update',choices=['train','frozen'],default='train',help='frozen runs one group_only_reset case, with original backbone/group head oracles')
    args=p.parse_args();out=Path(args.output).resolve()
    if args.worker:return worker(out,args.worker,args.width)
    if args.make_source:return source(out,args.width)
    out.mkdir(parents=True,exist_ok=False)
    import numpy as np
    rng=np.random.default_rng(80);data=out/'data/Cifar100/dir_fixture'
    for stage in ('train','test'):
        (data/stage).mkdir(parents=True)
        for i in range(20):
            n=2+i%2 if stage=='train' else 1+i%4
            np.savez(data/stage/f'{i}.npz',data=dict(x=(rng.normal(size=(n,3,32,32))*.1).astype('float32'),y=(np.arange(n)+i)%8))
    (data/'config.json').write_text(json.dumps(dict(num_clients=20,num_classes=100,partition='dir',alpha=.5)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable,__file__,'--output',str(out),'--width',str(args.width),'--make-source'],check=True,env=env)
    jobs={name:dict(head=mode,svd_diagnostics=args.svd_diagnostics,upload_backing=args.upload_backing) for name,mode in [('shared','shared'),('reset','group_only_reset'),('local','group_only_local')]}
    if args.helper_head_update=='frozen':
        jobs={'frozen':dict(head='group_only_reset',svd_diagnostics=args.svd_diagnostics,upload_backing=args.upload_backing,helper_head_update='frozen')}
    (out/'jobs.json').write_text(json.dumps(jobs))
    for label in jobs:
        print('RUN SERIAL',label,'width',args.width,flush=True)
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,__file__,'--output',str(out),'--width',str(args.width),'--worker',label],cwd=ROOT/'system',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Head actual-entry smoke failed: {out}/{label}.log')
        if not args.keep_models:
            for name in ('checkpoints','final_models','upload_backing'):
                owned=(out/label/name).resolve()
                if owned==out or not owned.is_relative_to(out):raise RuntimeError('Unsafe synthetic cleanup target')
                if owned.exists():shutil.rmtree(owned)
        print('PASS',label,flush=True)
    reports={k:json.loads((out/k/'verification.json').read_text()) for k in jobs}
    if args.helper_head_update=='train':
        for key in ('batches','final_rng'):assert reports['reset'][key]==reports['local'][key]
        first=lambda r:[e for e in r['events'] if e['kind']=='initial' or e.get('loop')==0]
        assert first(reports['reset'])==first(reports['local'])
        assert reports['reset']['prototype_packets'][0]==reports['local']['prototype_packets'][0]
        assert reports['reset']['events'][-1]!=reports['local']['events'][-1]
    (out/'acceptance_report.json').write_text(json.dumps(dict(status='PASS',synthetic_only=True,clients=20,K=3,rounds=2,
        actual_width=args.width,formal_width=16,scaled_fixture=args.width!=16,feature_dim=512,classes=100,
        svd_distance_observer=args.svd_diagnostics,
        test_only_upload_storage_backing=args.upload_backing,
        helper_head_update=args.helper_head_update,
        first_round_reset_local_exact=True if args.helper_head_update=='train' else None,
        first_round_prototypes_exact=True if args.helper_head_update=='train' else None,
        batches_RNG_exact=True if args.helper_head_update=='train' else None,
        cases={k:{name:r[name] for name in ('SGD_steps','train_tasks','models_reloaded','backbone_and_head_oracle_max_error')} for k,r in reports.items()}),indent=2))
    print('PASS',out/'acceptance_report.json')


if __name__=='__main__':main()
