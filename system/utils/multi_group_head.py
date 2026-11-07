"""Opt-in client x model classifier roles; legacy shared/full is a no-op.

Only fc3.weight/bias are classifier parameters (resolved by object identity).
SVD downloads remain full downloads; private restoration does not save downlink.
"""
import math
import time


def add_options(parser):
    parser.add_argument('--multi_head_mode',choices=['shared','group_only_reset','group_only_local'],default='shared')
    parser.add_argument('--multi_projection_scope',choices=['full','backbone'],default='full')
    parser.add_argument('--multi_helper_head_update',choices=['train','frozen'],default='train')


def enabled(args):
    return getattr(args,'multi_head_mode','shared')!='shared' or getattr(args,'multi_projection_scope','full')!='full'


def validate(args):
    mode=getattr(args,'multi_head_mode','shared');scope=getattr(args,'multi_projection_scope','full')
    helper=getattr(args,'multi_helper_head_update','train')
    if helper not in ('train','frozen'):raise ValueError('multi_helper_head_update must be train or frozen')
    if helper=='frozen' and (mode!='group_only_reset' or scope!='backbone'):
        raise ValueError('Frozen helper heads require group_only_reset + backbone; local/private heads are unsupported')
    if mode not in ('shared','group_only_reset','group_only_local') or scope not in ('full','backbone'):
        raise ValueError('Invalid multi head mode/projection scope')
    if not enabled(args):return
    if scope!='backbone':raise ValueError('group_only heads require backbone projection/scoring')
    if (getattr(args,'algorithm','')!='FedMultiGroupProj' or not getattr(args,'multi_svd',False)
            or getattr(args,'model_family','')!='Decom_CNN-5-512' or not getattr(args,'multi_last_epoch_protocol',False)
            or getattr(args,'multi_group_mode','')!='group_projection_softmax'
            or getattr(args,'multi_proto_bank_mode','per_model')!='per_model'
            or getattr(args,'multi_proto_scope','group_priority')!='group_priority'):
        raise ValueError('Head ablation v1 requires SVD FedMultiGroupProj, fixed last-epoch per_model/group_priority protocol and group_projection_softmax')
    if (not getattr(args,'multi_group_file','') or getattr(args,'multi_warmup_rounds',0)
            or getattr(args,'target_head_mode','shared')!='shared' or getattr(args,'target_local_schedule','joint')!='joint'
            or any(getattr(args,k,False) for k in ('target_prox_mu','target_head_l2_lambda','target_z2_lambda','multi_avg_proto_ablation'))
            or getattr(args,'multi_kd_mode','none')!='none' or getattr(args,'target_c0_replay_mode','none')!='none'):
        raise ValueError('Multi head roles require known fixed groups before download, shared/joint local training and no KD/Prox/Z2/head L2/replay')


def ensure_names(args):
    if enabled(args):
        tag=f"head_{args.multi_head_mode}_proj_{args.multi_projection_scope}"
        if tag not in args.exp_name:args.exp_name+='-'+tag
        if getattr(args,'multi_helper_head_update','train')=='frozen' and 'helper_head_frozen' not in args.exp_name:
            args.exp_name+='-helper_head_frozen'


def capacity_metadata(args):
    """Runtime report; leave immutable source-group capacity context untouched."""
    from utils.multi_group_svd import capacity_protocol
    value=capacity_protocol(args)
    if enabled(args):
        value.update(shared_classifier=args.multi_head_mode=='shared',classifier_full_rank=True,
            classifier_sharing_policy=args.multi_head_mode,projection_parameter_scope='backbone_excluding_fc3',
            global_delta_scope='recovered_backbone_post_local_minus_download_server_backbone_including_SVD_truncation',
            source_group_context='original immutable capacity/initialization schema retained for members_only validation')
        if getattr(args,'multi_helper_head_update','train')=='frozen':
            value.update(helper_head_update='frozen_only_for_outside_column_tasks',
                classifier_storage='full fc3 retained for prediction; freezing reduces trainable count, not stored parameter count')
    return value


def split(model):
    from utils.target_head import classifier_names
    names=classifier_names(model,require_trainable=not getattr(model,'_multi_helper_head_frozen',False));all_params=dict(model.named_parameters())
    return {n:p for n,p in all_params.items() if n not in names},{n:all_params[n] for n in names}


def clone_head(model):
    return {n:p.detach().cpu().clone() for n,p in split(model)[1].items()}


def fingerprint(values):
    from utils.head_snapshot import tensor_fingerprint
    return tensor_fingerprint(values)


def norm(values):
    return math.sqrt(math.fsum(float(p.detach().double().square().sum()) for p in values.values()))


def initialize(server,initial):
    if not enabled(server.args):return
    bb,head=split(initial);common=clone_head(initial)
    server.multi_head_records=[]
    server.multi_head_policy=dict(head_mode=server.args.multi_head_mode,projection_scope='backbone',
        head_parameter_names=list(head),backbone_parameter_names=list(bb),K=server.K,
        classifier='only fc3.weight/bias by identity; fc1/fc2 remain backbone',
        initial_head_sha256=fingerprint(common),initial_head_norm=norm(common),
        group_file_fingerprint=server.group_payload['group_fingerprint'],
        head_aggregation=('all uploaded heads with SAME backbone similarity weights, without head projection' if server.args.multi_head_mode=='shared'
                          else 'only group members, normalized original train samples; head mass=1'),
        download='FULL recovered mother incl head downloaded, then helper-local head restored after SVD; no claimed downlink saving',
        upload='recovered backbone from every client; group_only helpers have NO head entries in upload view',
        diagnostic_scope='all SVD truncation/local recovered/global-delta distances refer to backbone only; head change logged separately',
        initialization='helper-local first head copied from original common mother; no new classifier construction',
        training='unchanged K joint tasks/local_epochs; CE + original F + configured previous-round prototypes; original whole-model clip')
    for c in server.clients:c.multi_initial_head=common
    if getattr(server.args,'multi_helper_head_update','train')=='frozen':
        server.multi_head_policy.update(helper_head_update='frozen',
            training='unchanged K joint tasks/local_epochs and original CE/F/previous-round MSE; ONE clip over actual trainable parameters (backbone only for frozen helpers)',
            helper_head_training='download current column server head EACH round; freeze only helper fc3 before SGD creation; CE still backpropagates through fixed head',
            gradient_clipping='one original clip on actual trainable parameters; frozen helper total norm equals backbone norm',
            requires_grad_lifecycle='restore temporary classifier flags after success/error; fresh role-bound download required before next training')
    print('[MultiHead] '+str(server.multi_head_policy))


def bind_roles(client,server):
    if not enabled(server.args):return
    assigned=int(server.group_payload['client_to_model'][str(client.id)])
    roles=[client.id in g for g in server.groups]
    if sum(roles)!=1 or not roles[assigned]:raise RuntimeError('Head role requires a complete fixed partition')
    previous=getattr(client,'multi_head_roles',None)
    if previous is not None and roles!=previous:raise RuntimeError('Classifier roles cannot change across rounds')
    client.multi_head_roles=roles;client.multi_assigned_model_id=assigned;client.multi_role_round=server.loop_round


def begin_download(client,loop):
    if getattr(client,'multi_role_round',None)!=loop:raise RuntimeError('Bind client x column roles BEFORE SVD download')
    saved={}
    if client.args.multi_head_mode=='group_only_local':
        if loop and (client.upload_round!=loop-1 or client.download_round!=loop-1):
            raise RuntimeError('Private helper head must come from immediately previous ordinary training')
        for k,inside in enumerate(client.multi_head_roles):
            if not inside:saved[k]=clone_head(client.model_set[k]) if loop else client.multi_initial_head
    client.multi_head_download=[]
    client.multi_head_metadata=[None]*client.K
    return saved


def finish_download(client,model,column,server,saved,loop):
    import torch
    _,head=split(model);_,server_head=split(server)
    inside=client.multi_head_roles[column];private=client.args.multi_head_mode=='group_only_local' and not inside
    if private:
        for n,p in head.items():p.copy_(saved[column][n].to(p))
    for n,p in head.items():
        expected=saved[column][n] if private else server_head[n]
        if not torch.equal(p,expected.to(p)):raise RuntimeError('Classifier restoration/download mismatch')
    values=clone_head(model)
    client.multi_head_download.append(dict(client_id=client.id,model_id=column,assigned_model_id=client.multi_assigned_model_id,
        role='group' if inside else 'helper',multi_head_mode=client.args.multi_head_mode,multi_projection_scope='backbone',
        head_download_overwritten=int(not private),head_restored_after_full_download=int(private),
        head_upload_enabled=int(client.args.multi_head_mode=='shared' or inside),
        private_head_source_completed_train_count=(loop if private else None),
        head_initialization_source='common_initial' if private and loop==0 else 'previous_local_same_client_column' if private else 'current_server_column',
        evaluation_head_kind=('all_client_shared' if client.args.multi_head_mode=='shared' else 'group_shared' if inside else 'private_helper' if private else 'reset_helper'),
        downloaded_head_sha256=fingerprint(values),downloaded_head_norm=norm(values),
        full_download_head_parameters=sum(p.numel() for p in server_head.values()),
        actual_head_download_bytes=sum(p.numel()*p.element_size() for p in server_head.values()),
        actual_full_model_download_bytes=sum(p.numel()*p.element_size() for p in server.parameters())))
    from utils.multi_helper_head import configure_download
    configure_download(client,model,column,loop)


def finalize_upload(client,column,full,row):
    """Called after original recovery, before checkpoint save. Never mutate it."""
    from utils.multi_group_svd import distance_squared
    model=client.model_set[column];bb,head=split(model)
    names=set(head);view={n:p for n,p in full.items() if n not in names}
    base={n:p for n,p in client.download_servers[column].named_parameters() if n not in names}
    if set(view)!=set(base) or any(view[n].shape!=base[n].shape for n in view):raise ValueError('Backbone upload schema mismatch')
    if client.args.multi_svd_diagnostics:
        start=client.download_starts[column]
        for key,left,right in [('svd_truncation',start,base),('local_training_recovered',view,start),('global_delta',view,base)]:
            squared=distance_squared(left,right);row[key+'_squared_norm']=squared;row[key+'_norm']=squared**.5
    info=client.multi_head_download[column]
    uploaded=bool(info['head_upload_enabled']);actual=full if uploaded else view
    current=clone_head(model)
    row.update(info,post_local_head_sha256=fingerprint(current),post_local_head_norm=norm(current),
        diagnostic_parameter_scope='backbone_excluding_fc3',
        uploaded_backbone_parameters=sum(p.numel() for p in view.values()),
        uploaded_head_parameters=sum(full[n].numel() for n in head) if uploaded else 0,
        actual_parameter_upload_bytes=sum(p.numel()*p.element_size() for p in actual.values()),
        full_recovered_parameter_count=sum(p.numel() for p in full.values()))
    client.multi_head_metadata[column]=dict(row)
    return actual


def client_metadata(client,column):
    return dict(multi_head_policy=dict(client.multi_head_metadata[column]))


def endpoint_metadata(client,column):
    info=client.multi_head_download[column]
    result={k:info[k] for k in ('role','multi_head_mode','multi_projection_scope','evaluation_head_kind','head_initialization_source','private_head_source_completed_train_count')}
    if 'effective_head_update' in info:result.update(helper_head_update=info['helper_head_update'],effective_head_update=info['effective_head_update'])
    return result


def aggregate_column(model,clients,order,samples,group,column,args):
    import torch
    with torch.no_grad():return _aggregate_column(model,clients,order,samples,group,column,args)


def _aggregate_column(model,clients,order,samples,group,column,args):
    """Untouched old kernel on backbone; independently averaged classifier."""
    import torch
    from utils.group_projection import aggregate_group_updates
    bb,head=split(model);members=set(group)
    def uploads():
        for cid in order:
            post=clients[cid].full_upload_parameters(column,next(iter(bb.values())).device)
            expected=set(bb)|set(head) if args.multi_head_mode=='shared' or cid in members else set(bb)
            if set(post)!=expected:raise ValueError('Helper private head must be absent, not a placeholder, in upload view')
            yield cid,samples[cid],{n:post[n] for n in bb}
    result,metrics,rows,matrices=aggregate_group_updates(bb,uploads,samples,group,args.multi_group_mode,
        args.multi_group_weight,softmax_tau=args.group_softmax_tau)
    backbone_weights={r['client_id']:r['aggregation_weight'] for r in rows}
    total=sum(clients[i].train_samples for i in group)
    weights=backbone_weights if args.multi_head_mode=='shared' else {i:clients[i].train_samples/total if i in members else 0. for i in samples}
    head_result={n:torch.zeros_like(p) for n,p in head.items()}
    for cid in order:
        if args.multi_head_mode!='shared' and cid not in members:continue  # Never read a helper classifier.
        post=clients[cid].full_upload_parameters(column,next(iter(head.values())).device)
        for n in head_result:
            if post[n].shape!=head_result[n].shape or not torch.isfinite(post[n]).all():raise ValueError('Invalid uploaded group/shared classifier')
            head_result[n].add_(post[n]*weights[cid])
    mass=math.fsum(weights.values())
    if not math.isclose(mass,1.,abs_tol=1e-12):raise ValueError('Head aggregation mass must be 1')
    matrices.update(parameter_scope='recovered_backbone_excluding_fc3_no_buffers',projection_scope='backbone',
        head_parameter_scope=list(head),head_aggregation_mass=mass,
        head_contributor_ids=sorted(samples) if args.multi_head_mode=='shared' else sorted(group))
    metrics.update(multi_head_mode=args.multi_head_mode,multi_projection_scope='backbone',head_aggregation_mass=mass,
        diagnostic_parameter_scope='backbone_excluding_fc3')
    for row in rows:row.update(head_aggregation_weight=weights[row['client_id']],head_aggregation_mass=mass)
    return dict(result,**head_result),metrics,rows,matrices


def aggregate(server):
    import torch
    if server.groups is None or any(c.upload_round!=server.loop_round for c in server.clients):raise RuntimeError('Current full participation uploads/roles required')
    order=[c.id for c in server.selected_clients]
    if sorted(order)!=list(range(server.num_clients)):raise RuntimeError('Each client must upload exactly once')
    server.uploaded_ids=order;total=sum(c.train_samples for c in server.clients)
    samples={c.id:c.train_samples/total for c in server.clients};next_values=[];diagnostics=[]
    for k,model in enumerate(server.global_model_set):
        tick=time.perf_counter();result,metrics,rows,matrices=aggregate_column(model,server.clients,order,samples,server.groups[k],k,server.args)
        timing=dict(loop_round=server.loop_round,completed_train_count=server.loop_round+1,model_id=k,phase='grouped',
            aggregation_mode=server.args.multi_group_mode,aggregation_seconds=time.perf_counter()-tick)
        server.aggregation_records.append(dict(**timing,**metrics,matrices=matrices))
        server.weight_records.extend(dict(**timing,**row) for row in rows)
        server.multi_head_records.extend(dict(**timing,client_id=row['client_id'],in_group=row['in_target_group'],
            head_weight=row['head_aggregation_weight'],head_weight_sum=metrics['head_aggregation_mass'],
            backbone_weight=row['aggregation_weight'],head_mode=server.args.multi_head_mode,projection_scope='backbone') for row in rows)
        next_values.append(result);diagnostics.append(dict(metrics=metrics,rows=rows,matrices=matrices))
    with torch.no_grad():
        for model,result in zip(server.global_model_set,next_values):
            for n,p in model.named_parameters():p.copy_(result[n])
    server.last_aggregation=diagnostics
    return diagnostics
