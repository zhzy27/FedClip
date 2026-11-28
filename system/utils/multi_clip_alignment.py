"""Fixed legacy CIFAR-100 text anchors for the K1 CNN training loop."""
from functools import lru_cache
import hashlib
from io import BytesIO
import math
from pathlib import Path


DEFAULT_CACHE = Path(__file__).parent / 'clip_weights/text_features_cache/Cifar100_ViT-B-32_a_photo_of_OBJ.pt'


def add_arguments(parser):
    parser.add_argument('--multi_clip_mse_lambda', type=float, default=0.,
                        help='Weight of legacy CLIP text MSE on raw CNN base output; zero disables')
    parser.add_argument('--multi_clip_text_cache', default=None,
                        help='Path to the legacy CIFAR-100 ViT-B/32 text feature cache')


def validate(args):
    coefficient = getattr(args, 'multi_clip_mse_lambda', 0.)
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('multi_clip_mse_lambda must be finite and nonnegative')
    if not coefficient:
        return
    if (args.algorithm != 'FedMultiGroupProj' or not getattr(args, 'multi_avg_proto_ablation', False)
            or args.num_server_models != 1 or args.dataset != 'Cifar100' or args.num_classes != 100
            or args.model_family != 'Decom_CNN-5-512'):
        raise ValueError('CLIP text MSE requires K1 FedMultiGroupProj Avg ablation with the CIFAR-100 CNN')
    if args.multi_proto_mode != 'none' or args.multi_proto_lambda != 0:
        raise ValueError('CLIP alignment requires prototypes to be disabled')
    if getattr(args, 'use_asymmetric_lr', 0) or getattr(args, 'use_loss_specific_u_scaling', 0):
        raise ValueError('CLIP alignment requires symmetric SGD and no loss-specific U gradient scaling')
    if args.multi_proto_diagnostics and args.multi_proto_feature != 'base_output':
        raise ValueError('With CLIP and feature diagnostics enabled, set --multi_proto_feature base_output')


@lru_cache(maxsize=None)
def _read_cache(path):
    import torch

    try:
        content = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f'CLIP cache missing: {path}. First run the legacy '
                                'get_clip_class_embeddings("Cifar100", model_name="ViT-B/32", '
                                'prompt_template="a photo of {}") once, or pass --multi_clip_text_cache.') from error
    packet = torch.load(BytesIO(content), map_location='cpu', weights_only=True)
    table = packet.get('text_features') if isinstance(packet, dict) else None
    if not isinstance(table, torch.Tensor) or table.shape != (100, 512) or not table.is_floating_point():
        raise ValueError(f'CLIP cache {path} must contain floating text_features with shape [100, 512]')
    raw_dtype = str(table.dtype)
    table = table.detach().float()  # The legacy CNN client uses raw text_features.float().
    if not torch.isfinite(table).all():
        raise ValueError(f'CLIP cache {path} contains nonfinite text features')
    receipt = dict(model='ViT-B/32', prompt_template='a photo of {}', cache_path=str(path),
                   cache_sha256=hashlib.sha256(content).hexdigest(), cache_key='text_features',
                   cache_dtype=raw_dtype, training_dtype=str(table.dtype), shape=list(table.shape),
                   label_indexing='original cache row order indexed directly by the CIFAR-100 label',
                   feature_position='base_output', reduction='mean over batch and 512 features',
                   first_training_round=1)
    return table, receipt


def initialize(server, model):
    args = server.args
    validate(args)
    coefficient = getattr(args, 'multi_clip_mse_lambda', 0.)
    if not coefficient:
        return
    if (model.fc3.out_features, model.fc3.in_features) != (100, 512):
        raise ValueError('CLIP alignment requires a 512-feature, 100-class CNN')
    path = Path(getattr(args, 'multi_clip_text_cache', None) or DEFAULT_CACHE).expanduser().resolve()
    table, receipt = _read_cache(path)
    anchors = table.to(server.device)
    for client in server.clients:
        client.clip_text_anchors = anchors
    server.clip_alignment = dict(receipt, coefficient=coefficient)
    print(f'[CLIPTextAlignment] lambda={coefficient} cache={path} sha256={receipt["cache_sha256"]}')
