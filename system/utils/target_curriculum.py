"""Opt-in classwise soft CE curriculum for ordinary single-target helpers.

CLI utilities are Torch-free. Scoring uses TRAIN data and protects only CUDA
devices actually holding this model. No changes to server aggregation weights.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time


def add_options(parser):
    parser.add_argument('--target_curriculum_mode',choices=['none','loss_rank','random_rank'],default='none')
    parser.add_argument('--target_curriculum_strength',type=float,default=.5)
    parser.add_argument('--target_curriculum_warmup_rounds',type=int,default=10)
    parser.add_argument('--target_curriculum_diagnostics',type=int,choices=[0,1],default=1)


def enabled(args):
    return getattr(args,'target_curriculum_mode','none')!='none' and getattr(args,'target_curriculum_strength',.5)>0


def active(args,client_id,loop_round):
    return enabled(args) and client_id!=getattr(args,'target_client_id',0) and loop_round>=getattr(args,'target_curriculum_warmup_rounds',10)


def conflicts(args):
    return (any(getattr(args,k,False) for k in ('target_prox_mu','target_head_l2_lambda','target_z2_lambda',
        'meta_c0_split','meta_collect_snapshots','meta_snapshot_dir','meta_weight_file','models_folder_name',
        'resume','cnn512_control','head_diag_collect','learning_rate_decay','auto_break','few_shot',
        'random_join_ratio','client_drop_rate','train_slow_rate','send_slow_rate')) or
        getattr(args,'target_c0_replay_mode','none')!='none' or getattr(args,'multi_proto_mode','none')!='none' or
        getattr(args,'multi_kd_mode','none')!='none')


def compatible(args):
    return (getattr(args,'algorithm','')=='FedTargetProj' and getattr(args,'target_proj_mode','')=='projection_softmax'
        and getattr(args,'model_family','')=='Decom_CNN-5-512' and getattr(args,'target_head_mode','shared')=='shared'
        and getattr(args,'target_local_schedule','joint')=='joint' and getattr(args,'join_ratio',1.)==1.
        and not conflicts(args))


def validate(args):
    if getattr(args,'target_curriculum_mode','none') not in ('none','loss_rank','random_rank'):
        raise ValueError('Invalid target_curriculum_mode')
    strength=getattr(args,'target_curriculum_strength',.5);warmup=getattr(args,'target_curriculum_warmup_rounds',10)
    if not math.isfinite(strength) or not 0<=strength<1:raise ValueError('Curriculum strength must be finite in [0,1)')
    if type(warmup) is not int or warmup<0:raise ValueError('Curriculum warmup must be a nonnegative integer')
    if getattr(args,'target_curriculum_diagnostics',1) not in (0,1):raise ValueError('Curriculum diagnostics must be 0/1')
    if not enabled(args):return
    if not compatible(args):
        raise ValueError('Nonzero curriculum supports only full-participation FedTargetProj projection_softmax, Decom_CNN-5-512/shared/joint, without replay/Prox/Z2/head L2/meta/prototypes/KD/feedback/decay/slow/drop')
    target=getattr(args,'target_client_id',0);n=getattr(args,'num_clients',20)
    if type(target) is not int or not 0<=target<n:raise ValueError('Curriculum target client ID out of range')


def reporting(args):
    return compatible(args) and (getattr(args,'_target_curriculum_explicit',False) or getattr(args,'target_curriculum_mode','none')!='none')


def tag(args):
    return f"curriculum_{args.target_curriculum_mode}_s{args.target_curriculum_strength:g}_w{args.target_curriculum_warmup_rounds}".replace('.','p')


def ensure_names(args):
    if reporting(args):
        value=tag(args)
        if value not in getattr(args,'exp_name',''):args.exp_name=getattr(args,'exp_name','')+'_'+value


def stable_seed(seed,loop,client_id,purpose,label=None):
    value=json.dumps([int(seed),int(loop),int(client_id),purpose,label],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8],'big')%(2**63-1)


def class_ranks(scores,labels):
    import torch
    scores=scores.detach().to(device='cpu',dtype=torch.float64);labels=labels.detach().to(device='cpu',dtype=torch.int64)
    if scores.ndim!=1 or labels.shape!=scores.shape or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError('Require complete finite per-sample scores and labels')
    result=torch.empty_like(scores)
    for label in sorted(labels.unique().tolist()):
        indices=(labels==label).nonzero(as_tuple=True)[0];values,order=torch.sort(scores[indices],stable=True)
        _,counts=torch.unique_consecutive(values,return_counts=True);start=0
        ranks=torch.empty_like(values)
        for count in counts.tolist():
            ranks[start:start+count]=(start+count/2)/len(indices);start+=count
        result[indices[order]]=ranks
    return result


def permute_class_ranks(ranks,labels,seed,loop,client_id):
    import torch
    result=ranks.clone()
    for label in sorted(labels.unique().tolist()):
        ids=(labels==label).nonzero(as_tuple=True)[0]
        generator=torch.Generator().manual_seed(stable_seed(seed,loop,client_id,'target_curriculum_random_rank_v1',label))
        result[ids]=ranks[ids][torch.randperm(len(ids),generator=generator)]
    return result


def epoch_beta(strength,epoch,epochs):
    if not 1<=epoch<=epochs:raise ValueError('Invalid local epoch')
    return strength*(epochs-epoch)/(epochs-1) if epochs>1 else 0.


def sample_weights(ranks,beta):return 1.+beta*(1.-2.*ranks)


@contextmanager
def preserved_scoring_state(model):
    import random
    import numpy as np
    import torch
    cpu=torch.random.get_rng_state();python=random.getstate();numpy=np.random.get_state()
    devices=sorted({p.device.index for p in list(model.parameters())+list(model.buffers()) if p.device.type=='cuda'})
    cuda={device:torch.cuda.get_rng_state(device) for device in devices}  # NOT all visible cards.
    state={n:p.detach().clone() for n,p in model.state_dict().items()};modes=[(m,m.training) for m in model.modules()]
    try:yield
    finally:
        model.load_state_dict(state)
        for module,mode in modes:module.training=mode
        random.setstate(python);np.random.set_state(numpy);torch.random.set_rng_state(cpu)
        for device,value in cuda.items():torch.cuda.set_rng_state(value,device)


class IndexedDataset:
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        x,y=self.dataset[index];return x,y,index


class IndexedCollate:
    def __init__(self,collate):self.collate=collate
    def __call__(self,rows):
        import torch
        x,y=self.collate([(x,y) for x,y,_ in rows])
        return x,y,torch.tensor([index for _,_,index in rows],dtype=torch.int64)


def indexed_training_loader(loader):
    from common.data_loading import make_data_loader
    if loader.num_workers!=0 or loader.drop_last:raise ValueError('Curriculum v1 requires workers=0 and drop_last=False')
    # Reuse the ORIGINAL sampler object, generator and collate; do not iterate it.
    return make_data_loader(IndexedDataset(loader.dataset),1,shuffle=False,drop_last=False,num_workers=0,
        batch_sampler=loader.batch_sampler,generator=loader.generator,collate_fn=IndexedCollate(loader.collate_fn),pin_memory=loader.pin_memory)


def score_download(client,model,loader,loop):
    import torch
    from common.data_loading import make_data_loader
    from utils.head_snapshot import tensor_fingerprint
    start=time.perf_counter();n=len(loader.dataset)
    if not n or n!=client.train_samples:raise ValueError('Curriculum scoring requires the complete own TRAIN dataset')
    with preserved_scoring_state(model):
        source=tensor_fingerprint(model.state_dict())
        generator=torch.Generator().manual_seed(stable_seed(getattr(client.args,'seed',0),loop,client.id,'target_curriculum_score_loader_v1'))
        scoring=make_data_loader(IndexedDataset(loader.dataset),client.batch_size,shuffle=False,drop_last=False,
            num_workers=0,generator=generator,collate_fn=IndexedCollate(loader.collate_fn))
        scores=torch.empty(n,dtype=torch.float64);labels=torch.empty(n,dtype=torch.int64);seen=torch.zeros(n,dtype=torch.bool);batches=0
        model.eval()
        with torch.no_grad():
            for x,y,indices in scoring:
                x,y=client._move_batch(x,y);logits=model(x)
                loss=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                if not torch.isfinite(loss).all() or seen[indices].any():raise ValueError('Nonfinite or duplicate curriculum scores')
                scores[indices]=loss.detach().cpu().double();labels[indices]=y.detach().cpu();seen[indices]=True;batches+=1
        if not seen.all():raise ValueError('Incomplete curriculum scores')
        if str(client.device).startswith('cuda'):torch.cuda.synchronize(client.device)
    ranks=class_ranks(scores,labels)
    if client.args.target_curriculum_mode=='random_rank':
        ranks=permute_class_ranks(ranks,labels,getattr(client.args,'seed',0),loop,client.id)
    quantiles=torch.quantile(scores,torch.tensor([0.,.25,.5,.75,1.],dtype=torch.float64)).tolist()
    return ranks,dict(score_samples=n,score_batches=batches,score_seconds=time.perf_counter()-start,
        score_model_sha256=source,score_quantiles=quantiles,score_source='own TRAIN; actual downloaded low-rank checkpoint, BEFORE any SGD',
        score_source_loop_round=loop)


def row_template(client,loop):
    args=client.args
    return dict(loop_round=loop,completed_train_count=loop+1,client_id=client.id,epoch=None,
        configured_mode=getattr(args,'target_curriculum_mode','none'),configured_strength=getattr(args,'target_curriculum_strength',.5),
        warmup_rounds=getattr(args,'target_curriculum_warmup_rounds',10),active=False,beta=0.,record_scope='client_train_summary',
        disabled_reason=None,unweighted_ce_sample_mean=None,weighted_ce_sample_mean=None,
        weighted_F_sample_mean=None,weighted_F_step_mean=None,total_loss_sample_mean=None,total_loss_step_mean=None,
        train_sample_visits=0,train_batches=0,optimizer_steps=0,local_epochs=client.local_epochs,
        weight_min=1.,weight_max=1.,weight_mean=1.,score_samples=0,score_batches=0,score_seconds=0.,
        score_dataset_samples=None,score_ce_q0=None,score_ce_q25=None,score_ce_q50=None,score_ce_q75=None,score_ce_q100=None,
        score_model_sha256=None,score_source='not_used',score_source_loop_round=None,
        sgd_seconds=0.,total_seconds=0.,pre_clip_grad_norm_mean_per_step=None,clip_trigger_fraction=None,
        aggregation_train_samples=client.train_samples,rank_rate=None,trainable_parameter_count=None)


def train(client,loop):
    import torch
    from flcore.clients.clientTargetProj import save_item
    from utils.cnn512_control import effective_grad_clip
    validate(client.args);args=client.args;model=client._load_model()
    if not any(n.endswith(('conv_v','weight_v')) for n,_ in model.named_parameters()):
        raise ValueError('Curriculum requires the actual ordinary low-rank prediction model')
    original=client.load_train_data();ranks,score=score_download(client,model,original,loop)
    loader=indexed_training_loader(original);params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.SGD(params,lr=client.learning_rate);model.train();rows=[]
    total_ce=total_f=0.;visits=steps=0;sgd_seconds=0.
    for epoch in range(1,client.local_epochs+1):
        tick=time.perf_counter();beta=epoch_beta(args.target_curriculum_strength,epoch,client.local_epochs)
        ce_sum=weighted_sum=f_sum=objective_sum=f_steps=objective_steps=weight_sum=norm_sum=0.
        count=batches=triggered=0;minimum=float('inf');maximum=0.
        for x,y,indices in loader:
            x,y=client._move_batch(x,y);optimizer.zero_grad()
            logits=model(x);ce=client.loss(logits,y)
            if beta:
                losses=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                weights=sample_weights(ranks[indices],beta).to(losses)
                weighted=(weights*losses).mean()  # Denominator B, never sum(weights).
            else:weights=torch.ones_like(y,dtype=ce.dtype);weighted=ce
            f=args.regular_lamda*model.frobenius_decay() if args.is_regular==1 else ce.new_zeros(())
            loss=weighted+f
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite curriculum CE/F objective')
            loss.backward();cap=effective_grad_clip(args)
            if cap is not None:gradient_norm=torch.nn.utils.clip_grad_norm_(params,cap)
            elif args.target_curriculum_diagnostics:
                gradient_norm=torch.stack([p.grad.detach().norm(2) for p in params if p.grad is not None]).norm(2)
            if args.target_curriculum_diagnostics:
                value=float(gradient_norm)
                if not math.isfinite(value):raise FloatingPointError('Nonfinite curriculum gradient norm')
                coefficient=min(1.,cap/(value+1e-6)) if cap is not None else 1.
                norm_sum+=value;triggered+=int(coefficient<1.)
            optimizer.step();b=len(y)
            ce_sum+=float(ce.detach())*b;weighted_sum+=float(weighted.detach())*b
            f_value=float(f.detach());objective=float(loss.detach())
            f_sum+=f_value*b;objective_sum+=objective*b;f_steps+=f_value;objective_steps+=objective
            weight_sum+=float(weights.detach().sum());minimum=min(minimum,float(weights.min()));maximum=max(maximum,float(weights.max()))
            count+=b;batches+=1
        if not count:raise ValueError('Empty curriculum local epoch')
        if str(client.device).startswith('cuda'):torch.cuda.synchronize(client.device)
        elapsed=time.perf_counter()-tick;sgd_seconds+=elapsed;row=row_template(client,loop)
        row.update(epoch=epoch,record_scope='training_epoch',active=True,beta=beta,
            unweighted_ce_sample_mean=ce_sum/count,weighted_ce_sample_mean=weighted_sum/count,
            weighted_F_sample_mean=f_sum/count,weighted_F_step_mean=f_steps/batches,
            total_loss_sample_mean=objective_sum/count,total_loss_step_mean=objective_steps/batches,
            train_sample_visits=count,train_batches=batches,optimizer_steps=batches,weight_min=minimum,weight_max=maximum,weight_mean=weight_sum/count,
            score_samples=score['score_samples'] if epoch==1 else 0,score_batches=score['score_batches'] if epoch==1 else 0,
            score_seconds=score['score_seconds'] if epoch==1 else 0.,score_dataset_samples=score['score_samples'],
            score_model_sha256=score['score_model_sha256'],score_source=score['score_source'],score_source_loop_round=loop,
            sgd_seconds=elapsed,total_seconds=elapsed+(score['score_seconds'] if epoch==1 else 0.),
            pre_clip_grad_norm_mean_per_step=norm_sum/batches if args.target_curriculum_diagnostics else None,
            clip_trigger_fraction=triggered/batches if args.target_curriculum_diagnostics else None,
            rank_rate=getattr(model,'ratio_LR',None),trainable_parameter_count=sum(p.numel() for p in params))
        for key,value in zip(('score_ce_q0','score_ce_q25','score_ce_q50','score_ce_q75','score_ce_q100'),score['score_quantiles']):row[key]=value
        rows.append(row);total_ce+=ce_sum;total_f+=f_sum;visits+=count;steps+=batches
        if args.target_curriculum_diagnostics:print('[CurriculumLocal] '+str(row))
    save_item(model,client.role,'model',client.save_folder_name)
    elapsed=score['score_seconds']+sgd_seconds
    client.train_time_cost['num_rounds']+=1;client.train_time_cost['total_cost']+=elapsed;client.last_train_time_cost=elapsed
    client.last_normal_train_stats=dict(ce_loss=total_ce/visits,regularization_loss=total_f/visits,proximal_loss=0.,
        train_seconds=sgd_seconds,local_epochs=client.local_epochs,processed_train_samples=visits,train_batches=steps)
    client.last_curriculum_rows=rows
    return elapsed


def initialize(server,model):
    if not reporting(server.args):return
    from utils.head_snapshot import tensor_fingerprint
    from common.experiment import sha256
    from utils.data_utils import get_dataset_sub_dir
    from utils.cnn512_control import effective_grad_clip
    import torch
    args=server.args
    if getattr(args,'data_root',None):
        from common.client_data import client_file,data_directory
        directory=data_directory(args)
        paths=[client_file(args,c.id,train) for c in server.clients for train in (True,False)]
    else:
        directory=(Path('../dataset')/args.dataset/get_dataset_sub_dir(args)).resolve()
        paths=[directory/split/f'{c.id}.npz' for c in server.clients for split in ('train','test')]
    capacities=[]
    # Read existing prediction shells on CPU; never construct another random model.
    with torch.random.fork_rng(devices=[]):
        for client in server.clients:
            local=torch.load(Path(client.save_folder_name)/(client.role+'_model.pt'),map_location='cpu',weights_only=False)
            capacities.append(dict(client_id=client.id,rank_rate=getattr(local,'ratio_LR',None),
                trainable_parameter_count=sum(p.numel() for p in local.parameters() if p.requires_grad)))
            del local
    server.curriculum_rows=[]
    server.curriculum_policy=dict(mode=args.target_curriculum_mode,strength=args.target_curriculum_strength,warmup_rounds=args.target_curriculum_warmup_rounds,
        diagnostics=bool(args.target_curriculum_diagnostics),target_client_id=server.target_client_id,
        applies_to_client_ids=[c.id for c in server.clients if c.id!=server.target_client_id],
        initial_server_sha256=tensor_fingerprint(model.state_dict()),seed=server.seed,
        client_sample_counts=[dict(client_id=c.id,train_samples=c.train_samples,test_samples=c.test_samples) for c in server.clients],
        optimizer=dict(name='SGD',lr=server.learning_rate,momentum=0.,weight_decay=0.,grad_norm_cap=effective_grad_clip(args)),
        batch_size=server.batch_size,actual_client_capacities=capacities,
        is_regular=args.is_regular,regular_lamda=args.regular_lamda,
        model_formulas=list(getattr(args,'models',[])),data_directory=str(directory),
        data_files=[dict(path=str(p.resolve()),sha256=sha256(p) if p.is_file() else None) for p in paths],
        aggregation='unchanged ProjectionSoftmax, original train_samples, raw pre-projection cosine; no sample weights uploaded',
        self_weight=getattr(args,'projection_self_weight',None),softmax_tau=getattr(args,'projection_softmax_tau',.2),
        formula='mean_B((1+beta_e*(1-2*r_i))*CE_i) + original_weighted_F, F once',
        ranking='within each true class, ascending own-download TRAIN CE with average ties; (rank-.5)/class_count',
        random_assignment='SHA256(experiment seed, loop, client, class, fixed purpose); independent CPU Generator; classwise rank multiset preserved',
        scoring='eval/no_grad complete own TRAIN once per active helper/round BEFORE SGD; same for loss_rank/random_rank',
        state_protection='parameters, buffers, module modes, Python/NumPy/Torch CPU and only actual model CUDA devices',
        cost_units='score_* charged ONCE on epoch 1; SGD accesses/steps separate; no per-sample persistence',
        logging='active helpers: real per-epoch records; target/none/zero/warmup: legacy client summary, not fabricated epoch losses',
        losses='CE means sample-visit weighted; F and total BOTH sample-weighted and SGD-step means when active; inactive legacy only sample means available',
        primary='ordinary target post-local BEFORE aggregation; aggregate-download accuracy is a distinct later observation',
        local_epochs=server.local_epochs,planned_train_count=server.global_rounds+1)
    print('[CurriculumPolicy] '+json.dumps(server.curriculum_policy,sort_keys=True))


def record_client(server,client):
    if not reporting(server.args):return
    loop=server.cur_ground
    if active(server.args,client.id,loop):rows=client.last_curriculum_rows
    else:
        row=row_template(client,loop);stats=client.last_normal_train_stats
        reason=('target_client' if client.id==server.target_client_id else 'mode_none' if server.args.target_curriculum_mode=='none'
                else 'zero_strength' if server.args.target_curriculum_strength==0 else 'warmup')
        row.update(disabled_reason=reason,unweighted_ce_sample_mean=stats['ce_loss'],weighted_ce_sample_mean=stats['ce_loss'],
            weighted_F_sample_mean=stats['regularization_loss'],total_loss_sample_mean=stats['ce_loss']+stats['regularization_loss'],
            train_sample_visits=stats['processed_train_samples'],train_batches=stats['train_batches'],optimizer_steps=stats['train_batches'],
            sgd_seconds=stats['train_seconds'],total_seconds=stats['train_seconds'])
        capacity=next(r for r in server.curriculum_policy['actual_client_capacities'] if r['client_id']==client.id)
        row.update(rank_rate=capacity['rank_rate'],trainable_parameter_count=capacity['trainable_parameter_count'])
        rows=[row]
    if any(r['loop_round']!=loop or r['client_id']!=client.id for r in rows):raise RuntimeError('Stale curriculum rows')
    server.curriculum_rows.extend(rows);server._append_csv('target_curriculum_clients.csv',rows)


def save_json(server):
    if not reporting(server.args):return
    from common.experiment import write_json
    import csv
    rows=[dict(loop_round=r['loop_round'],completed_train_count=r['round'],target_post_local_acc=r['target_post_local_acc'],
        aggregated_download_acc=r['target_client_test_acc']) for r in server.target_proj_history]
    root=Path(server.report_dir())
    write_json(root/'target_curriculum_clients.json',dict(policy=server.curriculum_policy,history=server.curriculum_rows))
    write_json(root/'target_curriculum_primary.json',dict(history=rows,summary=server._local_accuracy_summary(),accuracy_unit='fraction'))
    if rows:
        with (root/'target_curriculum_primary.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save_h5(server,parent):
    if not reporting(server.args):return
    import h5py
    group=parent.create_group('target_curriculum');group.attrs['policy_json']=json.dumps(server.curriculum_policy)
    group.create_dataset('clients_json',data=[json.dumps(r,allow_nan=False) for r in server.curriculum_rows],dtype=h5py.string_dtype('utf-8'))
    group.attrs['summary_json']=json.dumps(server._local_accuracy_summary())
    group.create_dataset('primary_json',data=[json.dumps(dict(loop_round=r['loop_round'],completed_train_count=r['round'],
        target_post_local_acc=r['target_post_local_acc'],aggregated_download_acc=r['target_client_test_acc']))
        for r in server.target_proj_history],dtype=h5py.string_dtype('utf-8'))
    root=parent.file
    if 'rs_test_acc' in root:
        root.create_dataset('legacy_server_diagnostic_test_acc',data=root['rs_test_acc'][:]);del root['rs_test_acc']
    root.create_dataset('rs_test_acc',data=[r['target_post_local_acc'] for r in server.target_proj_history])
    root.attrs['rs_test_acc_scope']='ordinary_target_post_local_before_aggregation'
    root.attrs['rs_train_loss_scope']='original evaluation CE plus F, not weighted curriculum training objective'


def save_manifest(server):
    if not reporting(server.args):return
    path=Path(server.final_model_dir())/'manifest.json';value=json.loads(path.read_text(encoding='utf-8'))
    value['target_curriculum']=dict(policy=server.curriculum_policy,summary=server._local_accuracy_summary(),
        actual_score_samples=sum(r['score_samples'] for r in server.curriculum_rows),
        actual_SGD_sample_visits=sum(r['train_sample_visits'] for r in server.curriculum_rows),
        actual_optimizer_steps=sum(r['optimizer_steps'] for r in server.curriculum_rows),
        metrics_dir=str(Path(server.report_dir()).resolve()))
    value['legacy_server_diagnostic_best_accuracy']=value.get('best_accuracy')
    value['best_accuracy']=server._local_accuracy_summary()['best_target_local_acc'];value['primary_accuracy_metric']='target_post_local_acc'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
