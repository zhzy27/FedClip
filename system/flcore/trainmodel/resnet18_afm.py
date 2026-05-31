import torch
import torch.nn as nn


__all__ = [
    "resnet18_afm",
]


class BasicBlockAFM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNet18AFMBase(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_width=64,
        input_size=32,
        cifar_stem=None,
    ):
        super().__init__()
        if input_size is None:
            input_size = 32
        if cifar_stem is None:
            cifar_stem = input_size <= 32

        self.in_planes = base_width
        self.out_dim = base_width * 8 * BasicBlockAFM.expansion

        if cifar_stem:
            self.conv1 = nn.Conv2d(
                in_channels,
                base_width,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                base_width,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(base_width, blocks=2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlockAFM(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlockAFM.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockAFM(self.in_planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet18_AFM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        base_width=64,
        input_size=32,
        cifar_stem=None,
        feature_dim=None,
    ):
        super().__init__()
        self.base = ResNet18AFMBase(
            in_channels=in_channels,
            base_width=base_width,
            input_size=input_size,
            cifar_stem=cifar_stem,
        )
        self.feature_dim = feature_dim or self.base.out_dim

        if self.feature_dim == self.base.out_dim:
            self.neck = nn.Identity()
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.base.out_dim, self.feature_dim),
                nn.ReLU(inplace=True),
            )

        self.head = nn.Linear(self.feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, x):
        x = self.base(x)
        x = self.neck(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def resnet18_afm(
    in_channels=3,
    num_classes=10,
    base_width=64,
    input_size=32,
    cifar_stem=None,
    feature_dim=None,
):
    return ResNet18_AFM(
        in_channels=in_channels,
        num_classes=num_classes,
        base_width=base_width,
        input_size=input_size,
        cifar_stem=cifar_stem,
        feature_dim=feature_dim,
    )


def resnet18_cifar_afm(
    in_channels=3,
    num_classes=10,
    base_width=64,
    feature_dim=None,
):
    return resnet18_afm(
        in_channels=in_channels,
        num_classes=num_classes,
        base_width=base_width,
        input_size=32,
        cifar_stem=True,
        feature_dim=feature_dim,
    )
