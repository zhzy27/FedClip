"""Opt-in raw-feature Z2 for ordinary single-target low-rank training only.

CLI validation imports no Torch. The shared feature hook/math are imported only
by the active client branch; no prototype protocol or collection is involved.
"""
import json
import math
import os
from pathlib import Path


MODES=('avg','softmax_only','projection_softmax')


def add_options(parser):
    parser.add_argument('--target_z2_lambda',type=float,default=0.)
    parser.add_argument('--target_z2_start_round',type=int,default=2,help='Completed training count; first round always unchanged')
    parser.add_argument('--target_z2_feature',choices=['base_output','classifier_input'],default='base_output')
    parser.add_argument('--target_z2_diagnostics',type=int,choices=[0,1],default=1,help='Client training log rows; never changes the objective or SGD')


def enabled(args):return getattr(args,'target_z2_lambda',0.)>0


def active(args,loop_round):
    return enabled(args) and loop_round+1>=getattr(args,'target_z2_start_round',2)


def validate(args):
    coefficient=getattr(args,'target_z2_lambda',0.)
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('target_z2_lambda must be finite and nonnegative')
    start=getattr(args,'target_z2_start_round',2)
    if type(start) is not int or start<2:raise ValueError('target_z2_start_round must be >=2; no extra warmup or first-round Z2')
    if getattr(args,'target_z2_feature','base_output') not in ('base_output','classifier_input'):raise ValueError('Invalid target_z2_feature')
    if getattr(args,'target_z2_diagnostics',1) not in (0,1):raise ValueError('target_z2_diagnostics must be 0/1')
    if not enabled(args):return  # Zero must not restrict old algorithms/combinations.
    for key,default in dict(target_z2_start_round=2,target_z2_feature='base_output',target_z2_diagnostics=1).items():
        if not hasattr(args,key):setattr(args,key,default)
    if (getattr(args,'algorithm','')!='FedTargetProj' or getattr(args,'target_proj_mode','') not in MODES
            or getattr(args,'model_family','')!='Decom_CNN-5-512'
            or getattr(args,'target_head_mode','shared')!='shared' or getattr(args,'target_local_schedule','joint')!='joint'):
        raise ValueError('Nonzero target_z2_lambda supports only FedTargetProj avg/softmax_only/projection_softmax, Decom_CNN-5-512, shared head, joint')
    if (getattr(args,'target_prox_mu',0.) or getattr(args,'target_head_l2_lambda',0.)
            or getattr(args,'cnn512_control',False) or getattr(args,'meta_c0_split','')
            or getattr(args,'meta_collect_snapshots',False) or getattr(args,'meta_weight_file','')
            or getattr(args,'head_diag_collect',False) or getattr(args,'models_folder_name','')
            or getattr(args,'resume',False) or getattr(args,'learning_rate_decay',False)):
        raise ValueError('Target Z2 v1 forbids Prox/head L2/control/meta/head_diag/pretrained/resume/decay combinations')
    if (getattr(args,'join_ratio',1.)!=1. or getattr(args,'random_join_ratio',False)
            or getattr(args,'client_drop_rate',0.) or getattr(args,'train_slow_rate',0.)
            or getattr(args,'send_slow_rate',0.) or getattr(args,'few_shot',0) or getattr(args,'auto_break',False)):
        raise ValueError('Target Z2 v1 requires full participation/data/epochs and no slow/drop/early-stop protocol')


def tag(args):
    return (f'{args.target_proj_mode}_z2_lam{args.target_z2_lambda:g}_start{args.target_z2_start_round}_'
            f'{args.target_z2_feature}').replace('.','p')


def ensure_names(args):
    if not enabled(args):return
    value=tag(args)
    if value not in getattr(args,'exp_name',''):args.exp_name=getattr(args,'exp_name','')+'_'+value
    for key,default in dict(save_folder_name='temp',final_model_root='final_models',h5_result_root='h5_results',
                            training_log_root='training_logs',run_log_dir='',training_log_dir='').items():
        path=getattr(args,key,default)
        if path and value not in path:setattr(args,key,os.path.join(path,value))


def policy(server,initial_model):
    from utils.head_snapshot import tensor_fingerprint
    from utils.cnn512_control import effective_grad_clip
    args=server.args
    return dict(requested_lambda=args.target_z2_lambda,start_completed_train_count=args.target_z2_start_round,
        feature_position=args.target_z2_feature,diagnostics=bool(getattr(args,'target_z2_diagnostics',1)),
        formula='CE + original weighted F + effective_lambda * features.square().mean()',
        scope='all clients; one ordinary model per client; no prototypes/teachers/extra auxiliary communication',
        applies_to_client_ids=list(range(server.num_clients)),prototype_communication_bytes=0,
        initialization='unchanged legacy Model_Distribe and original server recovery; no initialization/seed/SVD replacement',
        initial_server_state_sha256=tensor_fingerprint(initial_model.state_dict()),
        client_model_formulas=list(getattr(args,'models',[])),server_model_formula=getattr(args,'global_model',None),
        seed=server.seed,dataset=args.dataset,partition=getattr(args,'partition',None),class_per_client=getattr(args,'class_per_client',None),
        optimizer=dict(name='SGD',lr=server.learning_rate,momentum=0.,weight_decay=0.,grad_norm_cap=effective_grad_clip(args)),
        regularization_enabled=args.is_regular,regularization_lambda=args.regular_lamda,
        models_trained_per_client_per_round=1,local_epochs=server.local_epochs,
        planned_train_count=server.global_rounds+1,accuracy_unit='fraction',tail_unit='ordinary post-local evaluation points',
        primary_metric='target_post_local_acc',aggregate_metric='target_client_test_acc (normal download after aggregation, no extra training)',
        loss_mean_unit='training process batch objectives weighted by actual SGD sample visits, not endpoint CE',
        historical_alignment='initialization and random trajectory not claimed aligned with historical ~55% results')


def initialize(server,model):
    if not enabled(server.args):return
    server.target_z2_policy=policy(server,model);server.target_z2_clients=[]
    from utils.target_head import preserved_head_rng
    from flcore.clients.clientbase import load_item
    with preserved_head_rng():
        capacities=[]
        for client in server.clients:
            local=load_item(client.role,'model',client.save_folder_name)
            capacities.append(dict(client_id=client.id,rank_rate=local.ratio_LR,
                trainable_parameters=sum(p.numel() for p in local.parameters() if p.requires_grad),
                train_samples=client.train_samples,test_samples=client.test_samples))
            del local
    server.target_z2_policy['actual_client_capacities']=capacities
    from common.experiment import sha256
    from utils.data_utils import get_dataset_sub_dir
    explicit=getattr(server.args,'data_root',None)
    if explicit:
        from common.client_data import data_directory,client_file,image_policy
        directory=data_directory(server.args);formats,_=image_policy(server.args)
    else:
        directory=(Path('../dataset')/server.args.dataset/get_dataset_sub_dir(server.args)).resolve();formats=('.npz',)
    files=[]
    for client in server.clients:
        for train in (True,False):
            path=(client_file(server.args,client.id,train) if explicit else directory/('train' if train else 'test')/f'{client.id}.npz')
            files.append(dict(client_id=client.id,split='train' if train else 'test',path=str(path.resolve()),
                sha256=sha256(path) if path.is_file() else None,
                samples=client.train_samples if train else client.test_samples))
    server.target_z2_policy.update(data_directory=str(directory),format_policy=list(formats),data_files=files,
        data_fingerprint_scope='actual selected file SHA256; no extra decode, sampler traversal or preprocessing',
        data_config_sha256=sha256(directory/'config.json') if (directory/'config.json').is_file() else None)
    print('[TargetZ2Policy] '+json.dumps(server.target_z2_policy,sort_keys=True))


def record_client(server,client):
    if not enabled(server.args) or not getattr(server.args,'target_z2_diagnostics',1):return
    row=dict(client.last_z2_train_stats,mode=server.target_proj_mode,seed=server.seed)
    if row['loop_round']!=server.cur_ground or row['client_id']!=client.id:raise RuntimeError('Stale target Z2 client training record')
    server.target_z2_clients.append(row);server._append_csv('target_z2_clients.csv',[row])
    with (Path(server.report_dir())/'target_z2.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(event='local_training',**row),allow_nan=False)+'\n')


def primary_rows(server):
    return [dict(loop_round=r['loop_round'],completed_train_count=r['round'],client_id=server.target_client_id,
        target_post_local_acc=r['target_post_local_acc'],target_aggregated_download_acc=r['target_client_test_acc'])
        for r in server.target_proj_history]


def save_json(server):
    if not enabled(server.args):return
    from common.experiment import write_json
    root=Path(server.report_dir());summary=server._local_accuracy_summary()
    write_json(root/'target_z2_clients.json',dict(policy=server.target_z2_policy,history=server.target_z2_clients))
    write_json(root/'target_z2_primary.json',dict(policy=server.target_z2_policy,history=primary_rows(server),summary=summary))
    write_json(root/'target_z2_summary.json',dict(primary_metric='target_post_local_acc',accuracy_unit='fraction',**summary))
    import csv
    rows=primary_rows(server)
    if rows:
        with (root/'target_z2_primary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save_h5(server,parent):
    if not enabled(server.args):return
    import h5py
    group=parent.create_group('target_z2');group.attrs['policy_json']=json.dumps(server.target_z2_policy)
    for name,rows in [('clients',server.target_z2_clients),('primary',primary_rows(server))]:
        group.create_dataset(name+'_json',data=[json.dumps(r,allow_nan=False) for r in rows],dtype=h5py.string_dtype('utf-8'))
    group.attrs['summary_json']=json.dumps(server._local_accuracy_summary())
    rows=primary_rows(server)
    for key in ('loop_round','completed_train_count','client_id','target_post_local_acc','target_aggregated_download_acc'):
        group.create_dataset('primary/'+key,data=[float('nan') if r[key] is None else r[key] for r in rows])
    # Only the NEW positive-Z2 exports declare the root curve as C0 primary.
    # Preserve the original server diagnostic rather than silently relabel it.
    root=parent.file
    if 'rs_test_acc' in root:
        root.create_dataset('legacy_server_diagnostic_test_acc',data=root['rs_test_acc'][:]);del root['rs_test_acc']
    root.create_dataset('rs_test_acc',data=[r['target_post_local_acc'] for r in rows])
    root.attrs['rs_test_acc_scope']='ordinary_target_post_local_before_aggregation'
    root.attrs['legacy_server_diagnostic_test_acc_scope']='original all-client evaluate() series at pre-download timing; not current target post-local primary'
    root.attrs['rs_train_loss_scope']='original evaluation CE plus F; not Z2 training total'


def save_manifest(server):
    if not enabled(server.args):return
    path=Path(server.final_model_dir())/'manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
    manifest['target_z2']=dict(policy=server.target_z2_policy,primary_summary=server._local_accuracy_summary(),
        actual_train_count=len(server.target_proj_history),ordinary_target_model=f'Client_{server.target_client_id}_model.pt',
        ordinary_target_scope='saved ordinary low-rank post-local model, not aggregation download',
        metrics_dir=str(Path(server.report_dir()).resolve()))
    manifest['legacy_server_diagnostic_best_accuracy']=manifest.get('best_accuracy')
    manifest['best_accuracy']=server._local_accuracy_summary()['best_target_local_acc']
    manifest['primary_accuracy_metric']='target_post_local_acc'
    path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
