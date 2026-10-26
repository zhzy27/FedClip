"""Optional heterogeneous Hyper_CNN path; all SVD/F operations use legacy code."""
import copy
import math
from pathlib import Path

DEFAULT_RATIOS = (.9, .37, .35, .25, .15)


def enabled(args):
    return bool(getattr(args, 'multi_svd', False))


def add_svd_arguments(parser):
    parser.add_argument('--multi_svd', action='store_true', help='Heterogeneous Hyper_CNN SVD capacities; old full CNN default unchanged')
    parser.add_argument('--multi_svd_ratios', default=None, help='Comma-separated rank rates in (0,1]; default .9,.37,.35,.25,.15')
    parser.add_argument('--multi_svd_assignment', default=None, help='N capacity-level indices; default client_id modulo number of levels')
    parser.add_argument('--multi_svd_diagnostics', type=int, choices=[0, 1], default=1,
                        help='Observe truncation/local/global distances without changing training RNG')
    parser.add_argument('--multi_svd_init', choices=['dense', 'low_rank_recovered'], default='dense',
                        help='Mother initialization only; dense preserves existing experiments')
    parser.add_argument('--multi_svd_init_rank_rate', type=float, default=.15,
                        help='Random low-rank mother rank rate before original recovery; not inferred from client capacities')


def validate_svd(args):
    active = enabled(args)
    if not active:
        if (getattr(args, 'multi_svd_ratios', None) is not None or getattr(args, 'multi_svd_assignment', None) is not None
                or getattr(args, 'multi_svd_diagnostics', 1) != 1
                or getattr(args, 'multi_svd_init', 'dense') != 'dense'
                or getattr(args, 'multi_svd_init_rank_rate', .15) != .15):
            raise ValueError('SVD capacity/diagnostic options require --multi_svd')
        return
    if args.algorithm != 'FedMultiGroupProj' or args.model_family != 'Decom_CNN-5-512':
        raise ValueError('--multi_svd requires FedMultiGroupProj and Decom_CNN-5-512')
    if not hasattr(args, 'multi_svd_diagnostics'):
        args.multi_svd_diagnostics = 1
    if args.multi_svd_diagnostics not in (0, 1):
        raise ValueError('multi_svd_diagnostics must be 0 or 1')
    args.multi_svd_init = getattr(args, 'multi_svd_init', 'dense')
    args.multi_svd_init_rank_rate = getattr(args, 'multi_svd_init_rank_rate', .15)
    if args.multi_svd_init not in ('dense', 'low_rank_recovered'):
        raise ValueError('multi_svd_init must be dense or low_rank_recovered')
    if not math.isfinite(args.multi_svd_init_rank_rate) or not 0 < args.multi_svd_init_rank_rate <= 1:
        raise ValueError('multi_svd_init_rank_rate must be finite in (0,1]')
    if args.multi_svd_init == 'dense' and args.multi_svd_init_rank_rate != .15:
        raise ValueError('Nondefault multi_svd_init_rank_rate requires low_rank_recovered; refusing to ignore it')
    if getattr(args, 'multi_warmup_rounds', 0) != 0 or getattr(args, 'multi_group_basis', '') != 'full_weights':
        raise ValueError('SVD v1 requires no warmup and first post-local recovered full_weights grouping')
    if getattr(args, 'target_head_l2_lambda', 0.) != 0:
        raise ValueError('SVD multi-group does not support classifier-specific regularization')
    if args.is_regular not in (0, 1) or not math.isfinite(args.regular_lamda) or args.regular_lamda < 0:
        raise ValueError('SVD requires is_regular=0/1 and finite nonnegative regular_lamda')
    try:
        value = getattr(args, 'multi_svd_ratios', None)
        ratios = list(DEFAULT_RATIOS) if value is None else [float(x) for x in (value.split(',') if isinstance(value,str) else value)]
        value = getattr(args, 'multi_svd_assignment', None)
        assignment = ([i % len(ratios) for i in range(args.num_clients)] if value is None else
                      [int(x) for x in value.split(',')] if isinstance(value, str) else list(value))
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError('Invalid SVD ratio list or capacity-level assignment') from error
    if not ratios or any(not math.isfinite(r) or not 0 < r <= 1 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError('SVD ratios must be distinct finite values in (0,1]')
    if len(assignment) != args.num_clients or any(type(i) is not int or not 0 <= i < len(ratios) for i in assignment):
        raise ValueError('SVD assignment must contain N valid capacity-level indices')
    args.multi_svd_effective_ratios, args.multi_svd_effective_assignment = ratios, assignment


def capacity_protocol(args):
    return dict(kind='svd_heterogeneous_v1', rank_rates=args.multi_svd_effective_ratios,
        client_level_ids=args.multi_svd_effective_assignment,
        client_rank_rates=[args.multi_svd_effective_ratios[i] for i in args.multi_svd_effective_assignment],
        capacity_fixed_across_models_and_rounds=True, rank_rate_is_parameter_retention_fraction=False,
        factorized_layers=['conv2', 'fc1', 'fc2'], shared_classifier=True,
        is_regular=args.is_regular, regular_lamda=args.regular_lamda,
        objective='CE_plus_original_Frobenius' if args.is_regular else 'CE_only',
        grouping_parameter_scope='recovered_full_named_parameters_including_fc3_aliases_deduplicated',
        global_delta_scope='recovered_post_local_minus_full_download_server_including_SVD_truncation')


def initialization_protocol(args):
    mode = getattr(args, 'multi_svd_init', 'dense')
    rate = getattr(args, 'multi_svd_init_rank_rate', .15)
    return dict(mode=mode, configured_rank_rate=rate,
        effective_low_rank_rate=rate if mode == 'low_rank_recovered' else None,
        recovered_server_rank_rate=1., seed_scope='existing_CPU_seed_fork',
        procedure='random_low_rank_then_original_recovery' if mode == 'low_rank_recovered' else 'original_random_dense',
        client_capacity_rule_unchanged=True)


def model_spec(args, ratio=1., model=None):
    if enabled(args):
        return dict(model_class='flcore.trainmodel.models.Hyper_CNN_512',
            model_kwargs=dict(in_features=3, num_classes=args.num_classes,
                n_kernels=model.conv1.out_channels if model is not None else 16, ratio_LR=ratio, input_size=32))
    return dict(model_class='common.models.CNN_512',
        model_kwargs=dict(in_channels=3, n_kernels=16, out_dim=args.num_classes, input_size=32))


def initial_model(args):
    if not enabled(args):
        from common.models import initial_full_model
        return initial_full_model(args)
    import torch
    from flcore.trainmodel.models import Hyper_CNN_512
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        if getattr(args, 'multi_svd_init', 'dense') == 'dense':
            return Hyper_CNN_512(**model_spec(args)['model_kwargs'])
        model = Hyper_CNN_512(**model_spec(args, args.multi_svd_init_rank_rate)['model_kwargs'])
        # Recovery constructors fill their dense parameters from the factors.
        # Do not reset/reinitialize those layers after this original recovery.
        model.recover_larger_model()
        model.ratio_LR = 1.  # Full mother state; each client sets its actual rank after download.
        return model


def recover_parameters(model):
    """One protected recovery of a private copy; never SVD or modify local checkpoint."""
    import torch
    from common.experiment import rng_state, set_rng
    state = rng_state()
    try:
        with torch.no_grad():
            recovered = copy.deepcopy(model)
            recovered.recover_larger_model()
            return {n: p.detach().cpu().clone() for n, p in recovered.named_parameters()}
    finally:
        set_rng(state)


def distance_squared(left, right):
    import torch
    if left.keys() != right.keys():
        raise ValueError('Recovered full parameter names differ')
    result = 0.
    with torch.no_grad():
        for name, a in left.items():
            b = right[name].detach().to(a)
            if a.shape != b.shape:
                raise ValueError(f'Recovered parameter shape differs: {name}')
            value = float((a - b).double().square().sum())
            if not math.isfinite(value):
                raise FloatingPointError(f'Nonfinite recovered distance: {name}')
            result += value
    return result


def model_capacity(model, full):
    ranks = {name: dict(rank=getattr(getattr(model, name), 'rank', None),
             parameter_shapes={n: list(p.shape) for n, p in getattr(model, name).named_parameters()})
             for name in ('conv2', 'fc1', 'fc2')}
    return dict(rank_rate=model.ratio_LR, layers=ranks,
        trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
        recovered_parameter_count=sum(p.numel() for p in full.values()))


def client_payload(client, column, **metadata):
    model = client.model_set[column]
    from utils.multi_group_kd_options import requested,protocol
    if requested(client.args):metadata['online_kd']=protocol(client.args)
    return dict(metadata, **model_spec(client.args, client.rank_rate, model), model_family=client.args.model_family,
        state_dict=model.state_dict(), capacity=client.capacity_metadata[column], client_id=client.id,
        model_id=column, role='ordinary_post_local_low_rank', completed_train_count=client.download_round+1,
        loop_round=client.download_round,
        svd_initialization=initialization_protocol(client.args),
        optimizer=client.optimizer_set[column].state_dict(),
        state_schema={n: list(p.shape) for n, p in model.state_dict().items()},
        prediction_checkpoint=True, buffers_rule='client_local_not_aggregated')


def load_checkpoint(path, device='cpu'):
    """Reconstruct the actual complete mother or ordinary low-rank prediction model."""
    import torch
    from common.experiment import rng_state, set_rng
    from flcore.trainmodel.models import Hyper_CNN_512
    state = rng_state()
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload['model_class'] != 'flcore.trainmodel.models.Hyper_CNN_512':
            raise ValueError('Not a heterogeneous multi-group Hyper_CNN checkpoint')
        model = Hyper_CNN_512(**payload['model_kwargs']).to(device)
        model.load_state_dict(payload['state_dict'], strict=True)
        return model.eval(), payload
    finally:
        set_rng(state)


def save_models(server):
    import torch
    from common.experiment import rng_state, write_json
    root = Path(server.save_folder_name)
    common = dict(initial_parameter_sha256=server.initial_hash,
        recovered_parameter_schema=server.initial_schema, group_fingerprint=server.group_payload['group_fingerprint'],
        capacity_protocol=capacity_protocol(server.args), grouping_protocol=server.group_protocol)
    from utils.multi_group_kd_options import requested,protocol
    if requested(server.args):common['online_kd']=protocol(server.args)
    for column, model in enumerate(server.global_model_set):
        torch.save(dict(common, **model_spec(server.args, model=model), model_family=server.args.model_family,
            svd_initialization=initialization_protocol(server.args),
            state_dict=model.state_dict(), model_id=column, role='post_aggregation_full_server',
            completed_train_count=server.loop_round+1), root/f'Server_model_{column}.pt')
    for client in server.clients:
        for column in range(server.K):
            torch.save(client_payload(client, column, **common,
                assigned_model_id=int(server.group_payload['client_to_model'][str(client.id)])),
                root/f'Client_{client.id}_model_{column}.pt')
    torch.save(dict(rng=rng_state(), training_start_rng=server.training_start_rng), root/'MultiGroup_rng.pt')
    write_json(root/'fixed_groups.json', server.group_payload)
