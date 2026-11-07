"""Opt-in fixed downloaded helper classifiers; no forward or loss replacement."""
from contextlib import contextmanager
import math


def requested(args):
    return getattr(args,'multi_helper_head_update','train')=='frozen'


def configure_download(client,model,column,loop):
    """After real SVD/head download and BEFORE constructing this round's SGD."""
    if not requested(client.args):return
    from utils.multi_group_head import split
    _,head=split(model)
    frozen=not client.multi_head_roles[column]
    model._multi_helper_restore_flags={n:p.requires_grad for n,p in head.items()}
    for p in head.values():
        p.grad=None
        p.requires_grad_(not frozen)
    model._multi_helper_head_frozen=frozen
    client._multi_helper_training_failed=False
    client.multi_head_download[column].update(helper_head_update='frozen',
        effective_head_update='frozen' if frozen else 'train',
        head_source_completed_train_count=loop,
        head_requires_grad_during_train=not frozen,
        head_freeze_scope='only current client and model column; fc3.weight/bias')


@contextmanager
def training_state(client):
    """Real training persists; only temporary requires_grad/mode flags unwind."""
    if getattr(client,'_multi_helper_training_failed',False):
        raise RuntimeError('Failed partial local training requires a fresh role-bound SVD download')
    modes=[(m,m.training) for model in client.model_set for m in model.modules()]
    success=False
    try:
        yield
        success=True
    finally:
        from utils.target_head import classifier_names
        for model in client.model_set:
            names=classifier_names(model,require_trainable=False);params=dict(model.named_parameters())
            frozen=getattr(model,'_multi_helper_head_frozen',False)
            flags=getattr(model,'_multi_helper_restore_flags',{})
            for n in names:
                if frozen:params[n].grad=None
                params[n].requires_grad_(flags.get(n,True))
            model._multi_helper_head_frozen=False
        if not success:
            client._multi_helper_training_failed=True
            for module,mode in modes:module.training=mode


class Monitor:
    """Read existing gradients; no extra forward/backward or clipping call."""
    def __init__(self,client,column):
        from utils import multi_group_head as heads
        self.client=client;self.column=column;self.model=client.model_set[column]
        self.bb,self.head=heads.split(self.model)
        self.before=heads.clone_head(self.model)
        self.frozen=not client.multi_head_roles[column]
        self.enabled=bool(client.args.multi_proto_diagnostics)
        optimizer_ids={id(p) for g in client.optimizer_set[column].param_groups for p in g['params']}
        self.in_optimizer=sum(id(p) in optimizer_ids for p in self.head.values())
        if (self.in_optimizer!=(0 if self.frozen else len(self.head)) or
                any(p.requires_grad==self.frozen for p in self.head.values())):
            raise RuntimeError('Downloaded head freeze/optimizer binding inconsistent with column role')
        self.head_grad=False;self.bb_sum=self.total_sum=self.coef_sum=0.;self.steps=self.triggered=0

    def before_clip(self):
        import torch
        self.head_grad |= any(p.grad is not None for p in self.head.values())
        if self.frozen and self.head_grad:raise RuntimeError('Frozen helper classifier unexpectedly has gradients')
        if self.enabled:
            norms=[p.grad.detach().norm(2) for p in self.bb.values() if p.requires_grad and p.grad is not None]
            value=torch.stack(norms).norm(2) if norms else next(iter(self.bb.values())).new_zeros(())
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite backbone gradient norm')
            self.bb_sum+=float(value)

    def after_clip(self,total_norm,cap):
        if not self.enabled:return
        value=float(total_norm)
        if not math.isfinite(value):raise FloatingPointError('Nonfinite total gradient norm')
        coef=float((cap/(total_norm+1e-6)).clamp(max=1.)) if cap is not None and cap>0 else 1.
        self.total_sum+=value;self.coef_sum+=coef;self.triggered+=int(coef<1.);self.steps+=1

    def finish(self,row):
        import torch
        from utils import multi_group_head as heads
        current=heads.clone_head(self.model)
        squared=math.fsum(float((current[n].double()-self.before[n].double()).square().sum()) for n in current)
        identical=all(torch.equal(current[n],self.before[n]) for n in current)
        if self.frozen and not identical:raise RuntimeError('Frozen downloaded helper classifier changed during SGD')
        row.update(helper_head_update='frozen',effective_head_update='frozen' if self.frozen else 'train',
            head_optimizer_parameter_count=self.in_optimizer,head_gradient_produced=self.head_grad,
            head_update_squared_norm=squared,head_update_norm=math.sqrt(squared),head_unchanged_bitwise=identical,
            total_local_parameter_count=sum(p.numel() for p in self.model.parameters()),
            backbone_pre_clip_grad_norm_mean_per_step=self.bb_sum/self.steps if self.steps else None,
            total_pre_clip_grad_norm_mean_per_step=self.total_sum/self.steps if self.steps else None,
            helper_clip_coefficient_mean_per_step=self.coef_sum/self.steps if self.steps else None,
            helper_clip_trigger_fraction=self.triggered/self.steps if self.steps else None,
            gradient_norm_unit='arithmetic mean over SGD steps, before ONE original clip; actual trainable parameters only',
            prediction_requires_grad='factory defaults restored at load; next training requires fresh role-bound download')


def monitor(client,column):
    return Monitor(client,column) if requested(client.args) else None
