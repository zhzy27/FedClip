"""Opt-in online cross-group KD, independent of aggregation/group compatibility."""
import math

DEFAULTS=dict(multi_kd_mode='none',multi_kd_lambda=.1,multi_kd_temperature=2.,
              multi_kd_teacher_batch_size=16,multi_kd_observe_gap=1,multi_kd_probe_samples=0)


def add_kd_arguments(parser):
    parser.add_argument('--multi_kd_mode',choices=['none','full','not_true'],default='none')
    parser.add_argument('--multi_kd_lambda',type=float,default=.1)
    parser.add_argument('--multi_kd_temperature',type=float,default=2.)
    parser.add_argument('--multi_kd_teacher_batch_size',type=int,default=16)
    parser.add_argument('--multi_kd_observe_gap',type=int,default=1,help='Completed-round interval for optional train-only probes; last round included')
    parser.add_argument('--multi_kd_probe_samples',type=int,default=0,help='Fixed independent train subset; zero disables differentiation probes')


def requested(args):
    return getattr(args,'multi_kd_mode','none')!='none'


def validate_kd(args):
    for key,value in DEFAULTS.items():
        if not hasattr(args,key):setattr(args,key,value)
    if args.multi_kd_mode not in ('none','full','not_true'):raise ValueError('Invalid multi_kd_mode')
    if not math.isfinite(args.multi_kd_lambda) or args.multi_kd_lambda<0:raise ValueError('multi_kd_lambda must be finite and nonnegative')
    if not math.isfinite(args.multi_kd_temperature) or args.multi_kd_temperature<=0:raise ValueError('multi_kd_temperature must be finite and positive')
    if args.multi_kd_temperature>math.sqrt(float.fromhex('0x1.fffffffffffffp+1023')):
        raise ValueError('multi_kd_temperature squared must be finite')
    if not math.isfinite(args.multi_kd_lambda*args.multi_kd_temperature**2):raise ValueError('KD coefficient*T^2 must be finite')
    for key in ('multi_kd_teacher_batch_size','multi_kd_observe_gap'):
        if type(getattr(args,key)) is not int or getattr(args,key)<1:raise ValueError(f'{key} must be a positive integer')
    if type(args.multi_kd_probe_samples) is not int or args.multi_kd_probe_samples<0:raise ValueError('multi_kd_probe_samples must be a nonnegative integer')
    if args.algorithm!='FedMultiGroupProj':
        if any(getattr(args,k)!=v for k,v in DEFAULTS.items()):raise ValueError('multi_kd options require FedMultiGroupProj')
        return
    if not requested(args):
        if args.multi_kd_probe_samples:raise ValueError('Online KD probes require full/not_true mode; disabled/zero KD never adds forwards')
        return
    if not getattr(args,'multi_svd',False):raise ValueError('Online KD v1 requires FedMultiGroupProj --multi_svd')
    if (args.target_head_mode!='shared' or args.target_local_schedule!='joint' or
            getattr(args,'target_prox_mu',0) or getattr(args,'target_head_l2_lambda',0)):
        raise ValueError('Online KD requires shared head, joint training, zero Prox and zero head L2')
    if getattr(args,'multi_warmup_rounds',0)!=0 or getattr(args,'multi_group_basis','')!='full_weights':
        raise ValueError('Online SVD KD requires zero warmup and first-round full_weights grouping')


def protocol(args):
    return dict(requested_mode=args.multi_kd_mode,coefficient=args.multi_kd_lambda,temperature=args.multi_kd_temperature,
        teacher_batch_size=args.multi_kd_teacher_batch_size,probe_samples=args.multi_kd_probe_samples,
        probe_completed_round_gap=args.multi_kd_observe_gap,first_eligible_loop_round=1,
        teacher_source='this_round_actual_downloaded_SVD_models_before_any_local_optimizer_step',
        assignment='validated_fixed_group_membership_not_client_id_mod_K',
        teacher_rule='all_K_minus_1_external_models_equal_probability_average',
        not_true_rule='delete_true_logit_per_teacher_and_student_then_softmax_then_average',
        loss='CE + original weighted F + coefficient*T^2*KL(teacher||student); sum_classes_mean_batch',
        cache='detached_CPU_raw_logits_original_sample_indices_fresh_each_round_memory_only',
        zero_rule='none_or_zero_coefficient_or_K1_or_first_round_uses_original_training_no_extra_forward',
        grouping_compatibility='KD excluded: first-round training and grouping unchanged',backend='existing_federated_settings_unchanged')


def ensure_kd_name(args):
    if args.algorithm=='FedMultiGroupProj' and requested(args):
        tag=f'onlineKD_{args.multi_kd_mode}_lam{args.multi_kd_lambda:g}_T{args.multi_kd_temperature:g}'.replace('.','p')
        if tag not in args.exp_name:args.exp_name+='-'+tag
