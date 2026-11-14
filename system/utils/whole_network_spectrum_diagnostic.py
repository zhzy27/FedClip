"""Known attenuation experiment. No federated module imports this file.

Native initialization, SVD factor construction/recovery, full parameter Avg and
local CE+F trainer are the repository implementations. Only the oracle E2
singular-value replacement and offline observations are defined here.
"""
import copy
import math
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch

from common.client_data import audit_image_clients,read_image_client,data_directory
from common.experiment import write_json,sha256,environment,code_version
from utils import fc1_tail_diagnostic as old,multi_group_svd as svd,single_avg_proto as trainer
from utils.multi_group_clustering import digest_json
from utils.head_snapshot import tensor_fingerprint
from utils.multi_rank_partition import protected_rng,actual_matrix,factors
from utils.group_projection import aggregate_group_updates

LAYERS=('conv2','fc1','fc2')
CASES=('E1','E2','E3','E4')
NAMES=dict(E1='simulated_Avg_raw_spectrum_large',E2='simulated_Avg_oracle_repaired_spectrum_large',
    E3='native_random_small',E4='native_random_large_original_factors')


def rates(text):
    try:values=[float(v.strip()) for v in text.split(',')]
    except (ValueError,AttributeError):raise ValueError('capacity-ratios must be comma-separated rates')
    if len(values)<2 or len(set(values))!=len(values) or any(not math.isfinite(v) or not 0<v<1 for v in values):
        raise ValueError('Require at least two distinct finite capacity rates in (0,1)')
    return sorted(values)


def configuration(cli):
    levels=rates(cli.capacity_ratios)
    for name in ('epochs','batch_size','test_batch_size','threads','num_clients','class_per_client'):
        if type(getattr(cli,name)) is not int or getattr(cli,name)<1:raise ValueError(f'{name} must be a positive integer')
    if not 0<=cli.client_id<cli.num_clients or not 0<=cli.seed<2**32:raise ValueError('Invalid client ID or seed')
    if cli.dataset!='Cifar100' or cli.class_per_client>100:raise ValueError('Only original RGB32/100-class CIFAR100 pat data supported')
    for name in ('lr','f_lambda','grad_clip','atol','rtol'):
        value=getattr(cli,name)
        if not math.isfinite(value) or value<0 or name in ('lr','grad_clip') and value==0:raise ValueError(f'Invalid {name}')
    return SimpleNamespace(algorithm='FedMultiGroupProj',dataset=cli.dataset,num_classes=100,num_clients=cli.num_clients,
        data_root=cli.data_root,data_subdir=cli.data_subdir,data_formats='npz_only',data_expected_partition='pat',
        partition='pat',class_per_client=cli.class_per_client,cnn512_control=True,seed=cli.seed,
        device='cuda:0' if cli.device=='cuda' else 'cpu',num_server_models=1,join_ratio=1.,
        model_family='Decom_CNN-5-512',multi_svd=True,multi_avg_proto_ablation=True,
        multi_svd_diagnostics=0,multi_svd_effective_ratios=levels,multi_svd_effective_assignment=[0]*cli.num_clients,
        multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=max(levels),
        multi_group_mode='group_avg_only',multi_group_weight=1.,target_head_mode='shared',target_local_schedule='joint',
        multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_aggregation='client_equal',
        multi_proto_feature='base_output',multi_proto_diagnostics=1,multi_proto_start_round=2,
        is_regular=1,regular_lamda=cli.f_lambda,local_epochs=cli.epochs,batch_size=cli.batch_size,
        test_batch_size=cli.test_batch_size,local_learning_rate=cli.lr,local_grad_clip=cli.grad_clip,offline_loop_round=0)


def error_metrics(actual,reference,atol,rtol,label,check=True):
    a=old.finite(actual.detach().cpu().double());b=old.finite(reference.detach().cpu().double())
    if a.shape!=b.shape:raise ValueError(f'{label}: shape mismatch')
    delta=a-b;maximum=float(delta.abs().max());scale=float(b.abs().max());norm=float(b.norm());distance=float(delta.norm())
    passed=maximum<=atol+rtol*scale and distance<=atol+rtol*norm
    record=dict(label=label,max_absolute_error=maximum,relative_max_error=maximum/scale if scale else None,
        error_frobenius=distance,relative_frobenius_error=distance/norm if norm else None,
        reference_frobenius=norm,atol=atol,rtol=rtol,passed=passed,
        criterion='global max error <= atol+rtol*reference_max AND L2 error <= atol+rtol*reference_L2; not per-element allclose')
    if check and not passed:raise ValueError(f'{label}: construction equivalence failed: {record}')
    return record


def attenuation(integer_ranks,dimension):
    if not integer_ranks or any(type(r) is not int or not 1<=r<=dimension for r in integer_ranks):raise ValueError('Invalid actual rank population')
    positions=torch.arange(1,dimension+1)
    return torch.stack([positions<=r for r in integer_ranks]).double().mean(0)


def layer_factors(layer):
    factors(layer)  # Original schema/finite validation; keep Parameter identity.
    if hasattr(layer,'conv_u'):return layer.conv_u,layer.conv_v
    return layer.weight_u,layer.weight_v


def initial_norms(model,stage,**identity):
    records=[]
    with torch.no_grad():
        for name in LAYERS:
            layer=getattr(model,name);u,v=layer_factors(layer);matrix=u.detach().cpu().double()@v.detach().cpu().double()
            records.append(dict(identity,stage=stage,layer=name,actual_rank=layer.rank,
                factor_U_frobenius=float(u.detach().double().norm()),factor_V_frobenius=float(v.detach().double().norm()),
                restored_matrix_frobenius=float(matrix.norm()),
                original_float32_recovery_product_frobenius=float(actual_matrix(layer).detach().cpu().double().norm()),
                bias_l2=float(layer.bias.detach().double().norm()) if layer.bias is not None else 0.,
                layer_parameter_count=sum(p.numel() for p in layer.parameters()),model_parameter_count=sum(p.numel() for p in model.parameters()),
                norm_dtype='float64 CPU product of actual factors; no factor normalization'))
    return records


@torch.no_grad()
def common_alignment(model,reference):
    """All nonfactor parameters/buffers share the large native initialization."""
    excluded={id(p) for name in LAYERS for p in layer_factors(getattr(model,name))}
    target=dict(model.named_parameters());base=dict(reference.named_parameters());records=[]
    if target.keys()!=base.keys():raise ValueError('Common model parameter names differ')
    for name,p in target.items():
        if id(p) in excluded:continue
        if p.shape!=base[name].shape:raise ValueError(f'Common initial shape mismatch: {name}')
        change=float((p.detach().cpu().double()-base[name].detach().cpu().double()).abs().max())
        p.copy_(base[name]);records.append(dict(parameter=name,max_absolute_alignment_change=change))
        if not torch.equal(p,base[name]):raise RuntimeError('Initial common-parameter copy failed')
    buffers=dict(reference.named_buffers())
    if dict(model.named_buffers()).keys()!=buffers.keys():raise ValueError('Common buffer names differ')
    for name,p in model.named_buffers():p.copy_(buffers[name])
    return records


@torch.no_grad()
def decomposed_once(full,rate):
    """Call the ORIGINAL decomposer; capture its same three SVD triples."""
    model=copy.deepcopy(full);evidence={};original=torch.linalg.svd;order=[]
    expected={name:actual_matrix(getattr(full,name)) for name in LAYERS}
    def capture(matrix,*a,**kw):
        if len(order)>=len(LAYERS):raise RuntimeError('Unexpected extra download SVD')
        name=LAYERS[len(order)]
        if not torch.equal(matrix,expected[name]):raise RuntimeError(f'Wrong original {name} convolution/matrix layout')
        result=original(matrix,*a,**kw);order.append(name)
        evidence[name]={key:value.detach().cpu().clone() for key,value in zip(('P','S','Qh'),result)}
        return result
    with protected_rng([full]),old.temporary_attribute(torch.linalg,'svd',capture):model.decom_larger_model(rate)
    if order!=list(LAYERS):raise RuntimeError('Need exactly one original SVD per decomposed layer')
    model.ratio_LR=rate
    return model,evidence


@torch.no_grad()
def truncated_model(balanced,rate):
    from flcore.trainmodel.models import FactorizedConv,FactorizedLinear
    with protected_rng([balanced]):
        model=copy.deepcopy(balanced)
        for name in LAYERS:
            layer=getattr(balanced,name);u,v=layer_factors(layer)
            if isinstance(layer,FactorizedConv):
                new=FactorizedConv(layer.in_channels,layer.out_channels,rate,padding=layer.padding,stride=layer.stride,
                    kernel_size=layer.kernel_size,bias=layer.bias is not None).to(u)
            else:new=FactorizedLinear(layer.in_features,layer.out_features,rate,bias=layer.bias is not None).to(u)
            uu,vv=layer_factors(new)
            if new.rank>layer.rank:raise ValueError('Simulation rank exceeds native mother rank')
            uu.copy_(u[:,:new.rank]);vv.copy_(v[:new.rank,:])
            if new.bias is not None:new.bias.copy_(layer.bias)
            setattr(model,name,new)
        model._rebuild_base();model.ratio_LR=rate
    return model


@torch.no_grad()
def equal_full_avg(mother,uploads):
    n=len(uploads);weights={i:1./n for i in range(n)}
    source={name:p.detach().cpu() for name,p in mother.named_parameters()}
    def factory():return iter((i,weights[i],uploads[i]) for i in range(n))
    values,metrics,rows,_=aggregate_group_updates(source,factory,weights,list(range(n)),'group_avg_only')
    result=copy.deepcopy(mother)
    for name,p in result.named_parameters():p.copy_(values[name])
    return result,dict(kernel='utils.group_projection.aggregate_group_updates/group_avg_only',
        upload_weights=rows,metrics=metrics,upload_count=n,SGD_steps_before_aggregation=0,
        weighting='equal virtual capacity tiers; NOT real client sample counts; one full-parameter Avg')


@torch.no_grad()
def repaired_model(raw,post,before):
    result=copy.deepcopy(raw)
    for name in LAYERS:
        layer=getattr(result,name);u,v=layer_factors(layer);rank=layer.rank
        # Oracle target is measured BEFORE simulated aggregation, no gain cap.
        root=before[name]['S'][:rank].sqrt();p=post[name]['P'];qh=post[name]['Qh']
        u.copy_(p[:,:rank]@torch.diag(root));v.copy_(torch.diag(root)@qh[:rank,:])
    return result


def spectral_tables(stage,spectra,before,ranks):
    rows=[];boundaries=[];energies=[]
    for name in LAYERS:
        s=old.finite(spectra[name].detach().cpu().double());reference=before[name]['S'].double();assigned=ranks[name]
        gamma=attenuation(assigned,len(s));maximum=max(assigned)
        for i,value in enumerate(s):
            source=float(reference[i]);actual=float(value)
            rows.append(dict(stage=stage,layer=name,rank=i+1,singular_value=actual,
                pre_aggregation_singular_value=source,actual_over_pre=actual/source if source else None,
                theoretical_Avg_attenuation=float(gamma[i]),retained_in_large=i<maximum))
        previous=0
        for end in sorted(set(assigned)):
            count=sum(r>=end for r in assigned);mass=count/len(assigned)
            energies.append(dict(stage=stage,layer=name,rank_start=previous+1,rank_end=end,
                singular_value_squared_energy=float(s[previous:end].square().sum()),contributing_tiers=count,
                theoretical_Avg_attenuation=mass,ideal_compensation_gain=1./mass))
            if end<len(s):
                left,right=float(s[end-1]),float(s[end])
                boundaries.append(dict(stage=stage,layer=name,boundary_rank=end,left_sigma=left,right_sigma=right,
                    right_over_left=right/left if left else None,left_over_right=left/right if right else None,
                    right_is_retained_in_large=end<maximum))
            previous=end
        if maximum<len(s):energies.append(dict(stage=stage,layer=name,rank_start=maximum+1,rank_end=len(s),
            singular_value_squared_energy=float(s[maximum:].square().sum()),contributing_tiers=0,theoretical_Avg_attenuation=0.,ideal_compensation_gain=None))
    return rows,boundaries,energies


def build_experiments(seed,levels,*,n_kernels=16,atol=1e-5,rtol=1e-5,diagnostics=True):
    from flcore.trainmodel.models import Hyper_CNN_512
    low,high=min(levels),max(levels);norms=[];checks=[];tick=time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        native=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=n_kernels,ratio_LR=high,input_size=32)
    native_sha=tensor_fingerprint(native.state_dict())
    with protected_rng([native]):
        mother=copy.deepcopy(native);mother.recover_larger_model();mother.ratio_LR=1.
    balanced,before=decomposed_once(mother,high);uploads=[];ranks={name:[] for name in LAYERS}
    for index,rate in enumerate(levels):
        local=truncated_model(balanced,rate)
        norms.extend(initial_norms(local,'simulation_truncated_upload',tier_id=index,rank_rate=rate))
        for name in LAYERS:ranks[name].append(getattr(local,name).rank)
        uploads.append(svd.recover_parameters(local))
    aggregate,aggregation=equal_full_avg(mother,uploads)
    raw,post=decomposed_once(aggregate,high);repaired=repaired_model(raw,post,before)
    for name in LAYERS:
        gamma=attenuation(ranks[name],len(before[name]['S']));expected=before[name]['S'].double()*gamma
        checks.append(error_metrics(post[name]['S'],expected,atol,rtol,name+' expected segmented spectrum attenuation'))
        p,s,qh=before[name]['P'],before[name]['S'],before[name]['Qh'];rank=max(ranks[name])
        expected_matrix=(p[:,:rank].double()*(s[:rank].double()*gamma[:rank]))@qh[:rank].double()
        checks.append(error_metrics(actual_matrix(getattr(aggregate,name)),expected_matrix,atol,rtol,name+' independent segmented matrix formula'))
        checks.append(error_metrics(actual_matrix(getattr(repaired,name)),actual_matrix(getattr(native,name)),atol,rtol,name+' E2 repaired matrix vs native E4'))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        small=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=n_kernels,ratio_LR=low,input_size=32)
    small_before=tensor_fingerprint(small.state_dict());models=dict(E1=raw,E2=repaired,E3=small,E4=copy.deepcopy(native))
    alignment={case:common_alignment(model,native) for case,model in models.items()}
    for case,model in models.items():norms.extend(initial_norms(model,'student_initial',case_id=case,rank_rate=model.ratio_LR))
    if tensor_fingerprint(native.state_dict())!=native_sha:raise RuntimeError('Original native E4 initialization changed')
    stages={'pre_aggregation':{n:before[n]['S'] for n in LAYERS},'post_Avg_raw':{n:post[n]['S'] for n in LAYERS}}
    # Mandatory E2 actual-matrix spectrum, not merely the requested target.
    stages['E2_repaired_actual']={n:torch.linalg.svdvals(actual_matrix(getattr(models['E2'],n)).detach().cpu().double()) for n in LAYERS}
    stages['E1_initial_actual']={n:torch.linalg.svdvals(actual_matrix(getattr(models['E1'],n)).detach().cpu().double()) for n in LAYERS}
    if diagnostics:
        for case in ('E3','E4'):
            stages[case+'_initial_actual']={n:torch.linalg.svdvals(actual_matrix(getattr(models[case],n)).detach().cpu().double()) for n in LAYERS}
    rows=[];boundaries=[];energies=[]
    for stage,values in stages.items():
        a,b,c=spectral_tables(stage,values,before,ranks);rows+=a;boundaries+=b;energies+=c
    gains=[]
    gaps=[]
    for n in LAYERS:
        maximum=max(ranks[n]);source=before[n]['S'].double();actual=post[n]['S'].double();gamma=attenuation(ranks[n],len(source))
        for i in range(maximum):gains.append(dict(layer=n,rank=i+1,target_pre_sigma=float(source[i]),post_Avg_sigma=float(actual[i]),
            actual_repair_gain=float(source[i]/actual[i]) if actual[i]>0 else None,
            theoretical_gain=1./float(gamma[i]),
            gain_cap=None,target='measured pre-aggregation spectrum, NO 1.5 cap'))
        a=stages['E1_initial_actual'][n];b=stages['E2_repaired_actual'][n]
        for i in range(len(a)):gaps.append(dict(layer=n,rank=i+1,E1_actual_sigma=float(a[i]),E2_actual_sigma=float(b[i]),
            E2_minus_E1_sigma=float(b[i]-a[i]),E2_over_E1=float(b[i]/a[i]) if i<maximum and a[i]>0 else None,
            retained_in_large=i<maximum))
    ledger=dict(seed=seed,capacity_ratios=levels,actual_layer_ranks=ranks,native_large_rank_rate=high,native_small_rank_rate=low,
        native_large_state_sha256=native_sha,recovered_mother_sha256=tensor_fingerprint(mother.state_dict()),
        simulated_aggregate_sha256=tensor_fingerprint(aggregate.state_dict()),
        simulation_upload_sha256=[tensor_fingerprint(p) for p in uploads],
        student_initial_state_sha256={case:tensor_fingerprint(model.state_dict()) for case,model in models.items()},
        native_small_before_common_alignment_sha256=small_before,
        common_initial_alignment=alignment,simulation_aggregation=aggregation,construction_checks=checks,
        pre_aggregation_SVD_calls=3,shared_E1_E2_download_SVD_calls=3,
        E1_E2_directions='same post-Avg P/Qh; E2 replaces singular values with measured pre-Avg values',
        E4_representation='original actual native random factors, never overwritten by balanced SVD factors',
        construction_device='CPU, original float32 model/SVD/recovery; float64 only for observation/error comparisons',
        construction_seconds=time.perf_counter()-tick)
    return models,mother,ledger,dict(initial_norms=norms,spectra=rows,spectral_boundaries=boundaries,
        spectral_interval_energy=energies,repair_gains=gains,E1_E2_spectral_gap=gaps)


class Endpoints:
    def __init__(self,model,data,test,args,epoch0):
        self.model=model;self.data=data;self.test=test;self.args=args;self.rows=list(epoch0);self.steps=0;self.seconds=0.
    def end_epoch(self,epoch):
        if epoch not in {1,self.args.local_epochs}:return
        tick=time.perf_counter()
        for split,data in [('train',self.data),('test',self.test)]:
            values,_=old.evaluate(self.model,data,self.args.test_batch_size,self.args.device)
            self.rows.append(dict(epoch=epoch,split=split,**values))
        self.seconds+=time.perf_counter()-tick
    def attach(self,optimizer):self.optimizer=optimizer;return self
    def __enter__(self):
        self.step= self.optimizer.step;self.had='step' in self.optimizer.__dict__
        def counted(*a,**kw):result=self.step(*a,**kw);self.steps+=1;return result
        self.optimizer.step=counted;return self
    def __exit__(self,*exc):
        if self.had:self.optimizer.step=self.step
        else:del self.optimizer.step


def train_case(model,mother,args,cid,data,test,epoch0,folder):
    client=old.local_client(model,mother,args,cid,None,None,data,folder)
    observer=Endpoints(model,data,test,args,epoch0).attach(client.optimizer_set[0])
    loader=old.EpochLoader(data,args,cid,0,observer);client.load_train_data=lambda:loader
    initial_rng=old.rng_fingerprint(old.rng_snapshot(model));tick=time.perf_counter()
    with observer:trainer.train(client)
    expected=args.local_epochs*math.ceil(len(data)/args.batch_size)
    if observer.steps!=expected or len(loader.batches)!=expected or client.avg_proto_local is not None:raise RuntimeError('Offline training step/prototype budget mismatch')
    return dict(training=client.train_stats[0],endpoints=observer.rows,batches=loader.batches,
        initial_rng_sha256=initial_rng,final_rng_sha256=old.rng_fingerprint(old.rng_snapshot(model)),
        final_model_sha256=tensor_fingerprint(model.state_dict()),batch_order_sha256=digest_json(loader.batches),
        endpoint_observation_seconds=observer.seconds,wall_seconds_including_observation=time.perf_counter()-tick)


def comparisons(endpoints,epochs):
    points={}
    for row in endpoints:
        key=(row['case_id'],row['epoch'],row['split'])
        if key in points:raise ValueError('Duplicate experiment endpoint')
        points[key]=row
    required={(c,e,s) for c in CASES for e in {0,1,epochs} for s in ('train','test')}
    if points.keys()!=required:raise ValueError('Incomplete four-case endpoints')
    result=[]
    for high,low,role in [('E2','E1','primary_spectral_repair'),('E4','E3','primary_native_whole_network_capacity'),('E2','E4','auxiliary_factor_representation')]:
        for epoch in sorted({0,1,epochs}):
            row=dict(contrast=high+'_minus_'+low,role=role,epoch=epoch,accuracy_unit='fraction')
            for split in ('train','test'):
                a,b=points[high,epoch,split],points[low,epoch,split]
                if a['samples']!=b['samples']:raise ValueError('Unpaired experiment data')
                row[split+'_samples']=a['samples'];row[split+'_accuracy_difference']=a['accuracy']-b['accuracy'];row[split+'_CE_difference']=a['ce']-b['ce']
            result.append(row)
    return result


def summary_text(rows,label):
    lines=['# 整网容量与聚合谱修复：离线四组',label,'',
        '| 对照 | epoch | 训练准确率差(pp) | 测试准确率差(pp) | 训练CE差 | 测试CE差 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['contrast']} | {r['epoch']} | {100*r['train_accuracy_difference']:.4f} | {100*r['test_accuracy_difference']:.4f} | {r['train_CE_difference']:.6g} | {r['test_CE_difference']:.6g} |")
    lines+=['','测试仅作固定epoch报告；不选模型/epoch。E2−E1是谱修复主对照，E4−E3是原生整网容量主对照；E2/E4因子表示不同，SGD轨迹不要求一致。']
    return '\n'.join(lines)+'\n'


def run(cli):
    args=configuration(cli);output=Path(cli.output_dir).expanduser().resolve();directory=data_directory(args)
    if output.exists():raise FileExistsError(f'Output exists: {output}')
    if output==directory or output in directory.parents or directory in output.parents:raise ValueError('Output must be separate from original data')
    torch.set_num_threads(cli.threads)
    # Controlled precision for these NEW offline experiments only. No global
    # package/environment or federated defaults are changed.
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if cli.device=='cuda':
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Need the requested single visible GPU, or --device cpu')
        torch.cuda.set_device(0)
    data_manifest=audit_image_clients(args);train=read_image_client(args,cli.client_id,True);test=read_image_client(args,cli.client_id,False)
    fields=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')
    args.data_fingerprint=digest_json(dict(config_sha256=data_manifest['config_sha256'],files=[{k:r[k] for k in fields} for r in data_manifest['files']]))
    input_hashes={Path(r['path']):r['sha256'] for r in data_manifest['files']};input_hashes[Path(data_manifest['config_path'])]=data_manifest['config_sha256']
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(status='running',experiment='four offline local controls from random native initialization; ONE simulated Avg, ZERO preceding SGD',
        requested=vars(cli),effective=vars(args),environment=environment(),code=code_version(Path(__file__).resolve().parents[2]),
        data_fingerprint=args.data_fingerprint,model_class='flcore.trainmodel.models.Hyper_CNN_512',
        model_kwargs=dict(in_features=3,num_classes=100,n_kernels=16,input_size=32),no_checkpoint_input=True,trained_models_exported=False,
        prediction='original trained classifier; no prototype/KD/extra fine-tune or test-driven selection',units='accuracy fraction; CE nats/sample; spectrum singular values; energy sigma squared')
    write_json(output/'manifest.json',manifest);write_json(output/'data_manifest.json',data_manifest)
    random.seed(args.seed);np.random.seed(args.seed);torch.random.default_generator.manual_seed(args.seed)
    if args.device.startswith('cuda'):torch.cuda.default_generators[0].manual_seed(args.seed)
    try:
        models,mother,ledger,tables=build_experiments(args.seed,args.multi_svd_effective_ratios,atol=cli.atol,rtol=cli.rtol,diagnostics=bool(cli.diagnostics))
        manifest['construction']=ledger;args.initial_sha256=ledger['recovered_mother_sha256']
        for name,rows in tables.items():write_json(output/(name+'.json'),rows);old.write_csv(output/(name+'.csv'),rows)
        mother=mother.to(args.device);models={c:m.to(args.device) for c,m in models.items()}
        cache={};initial_logits={};checks=[]
        for case,model in models.items():
            cache[case]=[];initial_logits[case]={}
            for split,data in [('train',train),('test',test)]:
                metrics,logits=old.evaluate(model,data,args.test_batch_size,args.device)
                cache[case].append(dict(epoch=0,split=split,**metrics));initial_logits[case][split]=logits
        for split in ('train','test'):
            check=error_metrics(initial_logits['E2'][split],initial_logits['E4'][split],cli.atol,cli.rtol,'E2 vs E4 '+split+' initial logits')
            check['argmax_disagreements']=int((initial_logits['E2'][split].argmax(1)!=initial_logits['E4'][split].argmax(1)).sum());checks.append(check)
        del initial_logits
        manifest['initial_function_checks']=checks
        base_rng=old.rng_snapshot(models['E4']);expected_rng=old.rng_fingerprint(base_rng);endpoints=[];training=[];batches=[]
        batch_hash=None;final_rng=None
        for case in CASES:
            model=models[case];case_args=copy.deepcopy(args);level=args.multi_svd_effective_ratios.index(model.ratio_LR)
            case_args.multi_svd_effective_assignment=[level]*args.num_clients
            initial_sha=tensor_fingerprint(model.state_dict());old.restore_rng(base_rng)
            print(f'\n========== {case} {NAMES[case]}, client={cli.client_id}, rate={model.ratio_LR} ==========',flush=True)
            with tempfile.TemporaryDirectory(prefix='transient_training_',dir=output) as temp:
                result=train_case(model,mother,case_args,cli.client_id,train,test,cache[case],temp)
            if result['initial_rng_sha256']!=expected_rng:raise RuntimeError('Training initialization RNG changed')
            if batch_hash is None:batch_hash=result['batch_order_sha256'];final_rng=result['final_rng_sha256']
            if batch_hash!=result['batch_order_sha256'] or final_rng!=result['final_rng_sha256']:raise RuntimeError('Four cases did not use paired batches/RNG')
            identity=dict(case_id=case,description=NAMES[case],client_id=cli.client_id,rank_rate=model.ratio_LR,
                actual_layer_ranks={n:getattr(model,n).rank for n in LAYERS},model_parameter_count=sum(p.numel() for p in model.parameters()))
            endpoints.extend(dict(identity,**row) for row in result['endpoints'])
            training.append(dict(identity,**result['training'],initial_model_sha256=initial_sha,
                initial_rng_sha256=result['initial_rng_sha256'],final_rng_sha256=result['final_rng_sha256'],final_model_sha256=result['final_model_sha256'],
                batch_order_sha256=result['batch_order_sha256'],endpoint_observation_seconds=result['endpoint_observation_seconds'],
                wall_seconds_including_observation=result['wall_seconds_including_observation']))
            for row in result['batches']:
                item=dict(row);indices=item.pop('indices');batches.append(dict(identity,**item,samples=len(indices),sample_indices_sha256=digest_json(indices)))
            for name,rows in [('endpoints',endpoints),('local_training',training),('batch_order',batches)]:write_json(output/(name+'.json'),rows);old.write_csv(output/(name+'.csv'),rows)
        pairs=comparisons(endpoints,args.local_epochs)
        write_json(output/'comparisons.json',pairs);old.write_csv(output/'comparisons.csv',pairs)
        (output/'summary.md').write_text(summary_text(pairs,'Random initialization, client '+str(cli.client_id)+', seed '+str(args.seed)),encoding='utf-8')
        if any(sha256(p)!=s for p,s in input_hashes.items()):raise RuntimeError('Original data changed during diagnostics')
        manifest.update(status='complete',data_unchanged=True,case_count=4,
            actual_SGD_steps=sum(r['train_batches'] for r in training),actual_training_sample_visits=sum(r['processed_train_samples'] for r in training),
            endpoint_sample_visits=sum(r['samples'] for r in endpoints),paired_batch_and_training_RNG_exact=True)
        write_json(output/'manifest.json',manifest);print('Offline results:',output,flush=True)
        return manifest
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(output/'manifest.json',manifest);raise
