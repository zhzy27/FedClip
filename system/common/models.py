"""Original full FedAvgCNN; Torch-only, without the optional model imports.

Source: pgg3/FedFew, 1b29dec70b66d89b956d90e0da024897f698897a.
Copyright (c) 2025 Ping Guo; MIT, see LICENSE.fedfew.
FedClip historically used the same FedAvgCNN class body; legacy exports remain
aliases. BaseHeadCNN/CNN_512 below are the existing FedClip CNN implementations,
relocated from flcore.trainmodel.models without changing their construction.
"""
import torch
from torch import nn


class BaseHeadCNN(nn.Module):
    """Legacy FedClip base/head container; old import paths remain aliases."""
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        features = self.base(x)
        return self.head(features)


class CNN_512(BaseHeadCNN):
    """FedClip's original CNN-512, moved without changing construction order."""
    def __init__(self, in_channels=3, n_kernels=16, out_dim=10, input_size=32):
        s = (((input_size - 4) // 2) - 4) // 2
        fc_input_dim = int(2 * n_kernels * s * s)
        base = nn.Sequential(
            nn.Conv2d(in_channels, n_kernels, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(n_kernels, 2 * n_kernels, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(), nn.Linear(fc_input_dim, 2000), nn.ReLU(), nn.Linear(2000, 512))
        head = nn.Sequential(nn.ReLU(), nn.Linear(512, out_dim))
        super().__init__(base, head)


def create_full_model(args):
    """One construction entry for full CNN baselines; no capacity/SVD logic."""
    if args.model_family == 'CNN-512':
        return CNN_512(in_channels=3, n_kernels=16, out_dim=args.num_classes, input_size=32)
    if args.model_family == 'FedAvgCNN-plain':
        return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
    raise ValueError(f'Unsupported full model: {args.model_family}')


def initial_full_model(args):
    if not getattr(args, 'cnn512_control', False):
        return create_full_model(args)
    # A CPU-only fork avoids touching CUDA/lazy CUDA seeding or the training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        return create_full_model(args)


def checked_parameters(source, destination):
    """Validate every name/shape before copying or arithmetic (never silent zip)."""
    src, dst = dict(source.named_parameters()), dict(destination.named_parameters())
    if src.keys() != dst.keys():
        raise ValueError('Full model parameter names differ')
    for name, parameter in dst.items():
        if src[name].shape != parameter.shape:
            raise ValueError(f'Full model shape differs: {name}')
    return src, dst


@torch.no_grad()
def copy_full_parameters(source, destination):
    src, dst = checked_parameters(source, destination)
    for name, parameter in dst.items():
        parameter.copy_(src[name])


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out
