"""Independent opt-in group-priority class prototypes for the SVD protocol."""
import math

DEFAULTS=dict(multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_start_round=2,
              multi_proto_batch_size=16,multi_proto_eps=1e-12,multi_proto_diagnostics=1,
              multi_proto_loss='cosine',multi_proto_feature='classifier_input',multi_proto_target_norm=1.)


def add_proto_arguments(parser):
    parser.add_argument('--multi_proto_mode',choices=['none','group_priority'],default='none')
    parser.add_argument('--multi_proto_lambda',type=float,default=0.)
    parser.add_argument('--multi_proto_start_round',type=int,default=2,help='Completed-training numbering; 2 means before loop=1 SGD')
    parser.add_argument('--multi_proto_batch_size',type=int,default=16)
    parser.add_argument('--multi_proto_eps',type=float,default=1e-12)
    parser.add_argument('--multi_proto_diagnostics',type=int,choices=[0,1],default=1,help='Direction, training-feature and clipping observations; never changes training')
    parser.add_argument('--multi_proto_loss',choices=['cosine','mse','z2'],default='cosine')
    parser.add_argument('--multi_proto_feature',choices=['classifier_input','base_output'],default='classifier_input')
    parser.add_argument('--multi_proto_target_norm',type=float,default=1.,help='Fixed positive target norm, MSE only')


def requested(args):return getattr(args,'multi_proto_mode','none')!='none'


def active_round(args,loop):
    return requested(args) and args.multi_proto_lambda>0 and loop+1>=args.multi_proto_start_round


def uses_prototypes(args):return getattr(args,'multi_proto_loss','cosine')!='z2'


def validate_proto(args):
    for key,default in DEFAULTS.items():
        if not hasattr(args,key):setattr(args,key,default)
    if args.multi_proto_mode not in ('none','group_priority'):raise ValueError('Invalid multi_proto_mode')
    if args.multi_proto_loss not in ('cosine','mse','z2'):raise ValueError('Invalid multi_proto_loss')
    if args.multi_proto_feature not in ('classifier_input','base_output'):raise ValueError('Invalid multi_proto_feature')
    if not math.isfinite(args.multi_proto_target_norm) or args.multi_proto_target_norm<=0:raise ValueError('multi_proto_target_norm must be finite and positive')
    if args.multi_proto_loss!='mse' and args.multi_proto_target_norm!=1.:raise ValueError('Nondefault multi_proto_target_norm is meaningful only for MSE')
    if args.multi_proto_loss=='z2' and (args.multi_proto_batch_size!=16 or args.multi_proto_eps!=1e-12):
        raise ValueError('Z2 does not use collection batch size or prototype epsilon; leave these at defaults')
    if not math.isfinite(args.multi_proto_lambda) or args.multi_proto_lambda<0:raise ValueError('multi_proto_lambda must be finite and nonnegative')
    if not math.isfinite(args.multi_proto_eps) or args.multi_proto_eps<=0:raise ValueError('multi_proto_eps must be finite and positive')
    if type(args.multi_proto_start_round) is not int or args.multi_proto_start_round<2:raise ValueError('multi_proto_start_round must be >=2; first round is always unchanged')
    if type(args.multi_proto_batch_size) is not int or args.multi_proto_batch_size<1:raise ValueError('multi_proto_batch_size must be positive')
    if args.multi_proto_diagnostics not in (0,1):raise ValueError('multi_proto_diagnostics must be 0/1')
    if args.algorithm!='FedMultiGroupProj':
        if any(getattr(args,k)!=v for k,v in DEFAULTS.items()):raise ValueError('multi_proto options require FedMultiGroupProj')
        return
    if not requested(args):
        if args.multi_proto_lambda:raise ValueError('Nonzero multi_proto_lambda requires group_priority mode')
        return
    if not getattr(args,'multi_svd',False):raise ValueError('Class prototypes v1 require FedMultiGroupProj --multi_svd')
    if getattr(args,'multi_kd_mode','none')!='none':raise ValueError('Prototype constraints cannot be mixed with online KD, including zero-KD modes')
    if (getattr(args,'target_prox_mu',0) or getattr(args,'target_head_l2_lambda',0) or
            args.target_head_mode!='shared' or args.target_local_schedule!='joint'):
        raise ValueError('Prototypes require zero Prox/head L2, shared classifier and joint training')
    if getattr(args,'multi_warmup_rounds',0)!=0 or getattr(args,'multi_group_basis','')!='full_weights':
        raise ValueError('Prototypes require zero warmup and first-round full_weights grouping')
    if args.multi_proto_lambda>0 and args.multi_proto_start_round>args.global_rounds+1:
        raise ValueError('Training budget has no eligible prototype round')


def protocol(args):
    kind=getattr(args,'multi_proto_loss','cosine');feature=getattr(args,'multi_proto_feature','classifier_input')
    anchored=uses_prototypes(args)
    return dict(mode=args.multi_proto_mode,coefficient=args.multi_proto_lambda,start_completed_round=args.multi_proto_start_round,
        requested_loss=kind,feature_position=feature,prototype_participation=anchored,
        target_norm=args.multi_proto_target_norm if kind=='mse' else None,
        collection_batch_size=args.multi_proto_batch_size if anchored else None,epsilon=args.multi_proto_eps if anchored else None,
        direction_diagnostics=bool(args.multi_proto_diagnostics),training_scale_and_clip_diagnostics=bool(args.multi_proto_diagnostics),
        feature=('actual fc3 input via temporary pre-hook, including original post-base ReLU' if feature=='classifier_input'
                 else 'actual base output via temporary forward hook, cloned BEFORE the original in-place head ReLU'),
        statistics='per-sample normalized features; float64 class sums and int64 true-label counts' if anchored else 'not applicable; no statistics collected',
        source='group if any group labels; otherwise outside; zero merged mean invalid, never fall back from zero group features' if anchored else 'not applicable; Z2 uses no prototypes',
        loss=dict(cosine='CE + original weighted F + coefficient*mean_valid(1-cos(raw_feature,detached_prototype))',
                  mse='CE + original weighted F + coefficient*mean_valid_samples_and_dimensions((raw_feature-r*detached_unit_prototype)^2)',
                  z2='CE + original weighted F + coefficient*mean_samples_and_dimensions(raw_feature^2)')[kind],
        scope='all clients, all model columns; each column uses its own library' if anchored else 'all clients, all model columns; no library or labels used by Z2',
        timing='collect ALL N x K downloaded low-rank models before ANY client SGD; rebuild every eligible round' if anchored else 'no collection; first round unchanged; apply only in original SGD forwards from configured start',
        coordinate_limitation='approximate alignment under one mother per column; heterogeneous coordinates not claimed strictly identical' if anchored else None,
        communication='actual in-process packet bytes, canonical header+raw tensors; no claim of physical network traffic' if anchored else 'zero; no prototype packets',
        grouping_compatibility='prototype configuration excluded because first round always unchanged')


def ensure_proto_name(args):
    if args.algorithm=='FedMultiGroupProj' and requested(args):
        tag=f'proto_group_priority_lam{args.multi_proto_lambda:g}_start{args.multi_proto_start_round}'.replace('.','p')
        if args.multi_proto_loss!='cosine' or args.multi_proto_feature!='classifier_input':
            tag+=f'_{args.multi_proto_loss}_{args.multi_proto_feature}'
        if args.multi_proto_loss=='mse':tag+=f'_r{args.multi_proto_target_norm:g}'.replace('.','p')
        if tag not in args.exp_name:args.exp_name+='-'+tag
