import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import math
import torch.nn.functional as F
import torch.nn.utils as utils

#对应不同channel的分组
GROUP_NORM_LOOKUP = {
    16: 2,  # -> channels per group: 8
    32: 4,  # -> channels per group: 8
    64: 8,  # -> channels per group: 8
    128: 8,  # -> channels per group: 16
    256: 16,  # -> channels per group: 16
    512: 32,  # -> channels per group: 16
    1024: 32,  # -> channels per group: 32
    2048: 32,  # -> channels per group: 64
}

import torch
import torch.nn as nn

# 组归一化
def group_norm(num_channels):
    return nn.GroupNorm(GROUP_NORM_LOOKUP[num_channels], num_channels)

# 实例归一化
def instance_norm(num_channels, affine=True):
    return nn.InstanceNorm2d(num_channels, affine=affine)


def channel_norm(num_channels, eps=1e-6):
    return ChannelNorm(num_channels, eps, affine=True)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank_rate, padding=None, stride=1, kernel_size=3, bias=False):
        super(FactorizedConv, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding if padding is not None else 0
        self.groups = 1  # 分组卷积支持，默认为1

        # 计算低秩分解的秩
        # self.rank = max(1, round(rank_rate * min(in_channels, out_channels)))
        self.rank = max(1, round(rank_rate * min(out_channels * kernel_size, in_channels * kernel_size)))
        # 使用二维矩阵存储分解参数
        # 通用处理任意kernel_size
        self.dim1 = out_channels * kernel_size
        self.dim2 = in_channels * kernel_size

        # 低秩参数矩阵 (二维存储)

        self.conv_v = nn.Parameter(torch.Tensor(self.rank, self.dim2))
        self.conv_u = nn.Parameter(torch.Tensor(self.dim1, self.rank))

        # 偏置参数
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        #初始化模型参数
        self.reset_parameters()

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.conv_u, a=math.sqrt(0))
    #     nn.init.kaiming_uniform_(self.conv_v, a=math.sqrt(0))
    #     if self.bias is not None:
    #         fan_in = self.in_channels * self.kernel_size * self.kernel_size
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)
#     # 稳定的低秩初始化方式
    def reset_parameters(self):
        """
        稳定初始化低秩卷积参数：
        - conv_u 和 conv_v 使用小方差正态分布，防止矩阵乘法后输出过大
        - bias 按照 fan_in 范围均匀初始化
        """
        # 小方差正态初始化，保证输出幅度不爆炸
        nn.init.normal_(self.conv_u, mean=0.0, std=0.01)
        nn.init.normal_(self.conv_v, mean=0.0, std=0.01)

        # 偏置初始化
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # 空间分解: 1xK + Kx1
        # 垂直卷积 (1xK)
        weight_v = self.conv_v.T.reshape(self.in_channels, self.kernel_size, 1, self.rank).permute(3, 0, 2, 1)
        out = F.conv2d(
            x, weight_v, None,
            stride=(1, self.stride),
            padding=(0, self.padding),
            dilation=(1, 1),
            groups=self.groups
        )

        # 水平卷积 (Kx1)  不显示reshpe权重
        weight_u = self.conv_u.reshape(self.out_channels, self.kernel_size, self.rank, 1).permute(0, 2, 1, 3)
        out = F.conv2d(
            out, weight_u, self.bias,
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            dilation=(1, 1),
            groups=self.groups
        )
        return out

    def frobenius_loss(self):
        return torch.sum((self.conv_u @ self.conv_v) ** 2)

    def reconstruct_full_weight(self):
        """重建完整的卷积核权重 (用于聚合)"""
        # 直接使用矩阵乘法
        A = self.conv_u @ self.conv_v  # [out*K, in*K]
        W = A.reshape(self.out_channels, self.kernel_size, self.in_channels, self.kernel_size)
        W = W.permute(0, 2, 1, 3)  # [out, in, K, K]
        return W

    def L2_loss(self):
        """分解参数的L2范数平方和"""
        return torch.norm(self.conv_u, p='fro') ** 2 + torch.norm(self.conv_v, p='fro') ** 2

    def kronecker_loss(self):
        """Kronecker乘积损失"""
        return (torch.norm(self.conv_u, p='fro') ** 2) * (torch.norm(self.conv_v, p='fro') ** 2)


# 卷积层分解函数 (FedHM兼容)
def Decom_COV(conv_model, ratio_LR=0.5):
    # 自动从卷积层获取参数
    in_planes = conv_model.in_channels
    out_planes = conv_model.out_channels
    kernel_size = conv_model.kernel_size[0]
    stride = conv_model.stride[0]
    padding = conv_model.padding[0]
    bias = conv_model.bias is not None

    # 创建分解层 (使用二维矩阵存储)
    factorized_cov = FactorizedConv(
        in_planes,
        out_planes,
        rank_rate=ratio_LR,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        bias=bias
    )

    # 获取原始权重并重塑
    W = conv_model.weight.data

    #重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)

    # SVD分解
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    # 计算截断秩
    rank = factorized_cov.rank
    S_sqrt = torch.sqrt(S[:rank])
    # 分配奇异值
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]

    # 加载参数
    with torch.no_grad():
        factorized_cov.conv_u.copy_(U_weight)
        factorized_cov.conv_v.copy_(V_weight)

        # 复制偏置
        if bias:
            factorized_cov.bias.copy_(conv_model.bias.data)

    return factorized_cov

# 卷积层恢复函数
def Recover_COV(decom_conv):
    # 获取分解层参数
    in_planes = decom_conv.in_channels
    out_planes = decom_conv.out_channels
    kernel_size = decom_conv.kernel_size
    stride = decom_conv.stride
    padding = decom_conv.padding
    bias = decom_conv.bias is not None

    # 重建完整权重
    W = decom_conv.reconstruct_full_weight()

    # 创建原始卷积层
    recovered_conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=kernel_size,
        stride=stride, padding=padding, bias=bias
    )

    # 加载权重
    with torch.no_grad():
        recovered_conv.weight.copy_(W)
        if bias:
            recovered_conv.bias.copy_(decom_conv.bias)

    return recovered_conv


# 正常残差块
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_norm:
            self.bn1 = norm_layer(planes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_norm:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 低秩分解残差块
class Low_RANK_BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            rank_rate=1.0
    ) -> None:
        super(Low_RANK_BasicBlock, self).__init__()
        self.rank_rate = rank_rate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if self.rank_rate < 1.0:
            self.conv1 = FactorizedConv(in_channels=inplanes, out_channels=planes, rank_rate=rank_rate, padding=1,
                                        stride=stride, kernel_size=3, bias=False)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if has_norm:
            self.bn1 = norm_layer(planes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        if self.rank_rate < 1.0:
            self.conv2 = FactorizedConv(in_channels=planes, out_channels=planes, rank_rate=rank_rate, padding=1,
                                        stride=1, kernel_size=3, bias=False)
        else:
            self.conv2 = conv3x3(planes, planes)
        if has_norm:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def recover(self):
        if self.rank_rate >= 1.0:
            return
        else:
            self.conv1 = Recover_COV(self.conv1)
            self.conv2 = Recover_COV(self.conv2)
            print(f"实现恢复函数")

    def decom(self, rank_rate):
        if self.rank_rate >= 1.0:
            return
        else:
            self.conv1 = Decom_COV(self.conv1, ratio_LR=rank_rate)
            self.conv2 = Decom_COV(self.conv2, ratio_LR=rank_rate)
            print(f"实现分解函数，分解比例为{rank_rate}")

    # 正则化函数
    def frobenius_loss(self):
        if self.rank_rate >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.conv1.frobenius_loss() + self.conv2.frobenius_loss()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if has_norm:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if has_norm:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion)
        if has_norm:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out




class LOW_RANK_Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            rank_rate=1.0
    ) -> None:
        super().__init__()
        self.rank_rate = rank_rate
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        if self.rank_rate >= 1.0:
            self.conv1 = conv1x1(inplanes, width)
        else:
            self.conv1 = FactorizedConv(in_channels=inplanes, out_channels=width, rank_rate=rank_rate, padding=0,
                                        stride=1,
                                        kernel_size=1, bias=False)
        if has_norm:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()

        if self.rank_rate >= 1.0:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        else:
            self.conv2 = FactorizedConv(in_channels=width, out_channels=width, rank_rate=rank_rate, padding=1,
                                        stride=stride,
                                        kernel_size=3, bias=False)
        if has_norm:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        if self.rank_rate >= 1.0:
            self.conv3 = conv1x1(width, planes * self.expansion)
        else:
            self.conv3 = FactorizedConv(in_channels=width, out_channels=planes * self.expansion, rank_rate=rank_rate,
                                        padding=0, stride=1,
                                        kernel_size=1, bias=False)

        if has_norm:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def recover(self):
        print(f"实现恢复函数")
        if self.rank_rate >= 1.0:
            return
        else:
            self.conv1 = Recover_COV(self.conv1)
            self.conv2 = Recover_COV(self.conv2)
            self.conv3 = Recover_COV(self.conv3)

    def decom(self, rank_rate):
        print(f"实现分解函数，分解比例为{self.rank_rate}")
        if self.rank_rate >= 1.0:
            return
        else:
            self.conv1 = Decom_COV(self.conv1, ratio_LR=rank_rate)
            self.conv2 = Decom_COV(self.conv2, ratio_LR=rank_rate)
            self.conv3 = Decom_COV(self.conv3, ratio_LR=rank_rate)

    # 正则化函数
    def frobenius_loss(self):
        if self.rank_rate >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.conv1.frobenius_loss() + self.conv2.frobenius_loss() + self.conv3.frobenius_loss()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_norm:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0)))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_norm=has_norm and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_norm=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_norm))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_norm=has_norm))

        return layers

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class LOW_RANK_ResNet(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet, self).__init__()
        self.ratio_LR = ratio_LR
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 第一个卷积层不进行操作
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_norm:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0)))

        for num in range(1, len(layers)):
            #这样设计超过四层的网络会不使用bn层
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_norm=has_norm and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        #分类器层
        self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 一般不启用这个函数
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_norm=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_norm, self.ratio_LR))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_norm, rank_rate=self.ratio_LR))

        return layers

    def recover_larger_model(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            for block in self.layers:
                block.recover()

    def decom_larger_model(self, rank_rate):
        if self.ratio_LR >= 1.0:
            return
        else:
            for block in self.layers:
                block.decom(rank_rate)

    def frobenius_decay(self):
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.ratio_LR >= 1.0:
            return loss
        else:
            for block in self.layers:
                loss += block.frobenius_loss()
            return loss

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

#改变第一个卷积层的输入防止丢失太多细节
# class LOW_RANK_ResNet_CIFAR(nn.Module):
#
#     def __init__(
#             self,
#             block: Low_RANK_BasicBlock,
#             layers: List[int],
#             features: List[int] = [64, 128, 256, 512],
#             num_classes: int = 1000,
#             zero_init_residual: bool = False,
#             groups: int = 1,
#             width_per_group: int = 64,
#             replace_stride_with_dilation: Optional[List[bool]] = None,
#             norm_layer: Optional[Callable[..., nn.Module]] = None,
#             has_norm=True,
#             bn_block_num=4,
#             ratio_LR=1.0
#     ) -> None:
#         super(LOW_RANK_ResNet_CIFAR, self).__init__()
#         self.ratio_LR = ratio_LR
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         # 第一个卷积层不进行操作
#
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         if has_norm:
#             self.bn1 = norm_layer(self.inplanes)
#         else:
#             self.bn1 = nn.Identity()
#         self.relu = nn.ReLU(inplace=True)
#
#         self.layers = []
#         self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_norm and (bn_block_num > 0)))
#
#         for num in range(1, len(layers)):
#             self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
#                                                 dilate=replace_stride_with_dilation[num - 1],
#                                                 has_bn=has_norm and (num < bn_block_num)))
#
#         for i, layer in enumerate(self.layers):
#             setattr(self, f'layer_{i}', layer)
#
#         self.avgpool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )
#         self.fc = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)
#
#         # self.fc = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d((1, 1)),
#         #     nn.Flatten(),
#         #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
#         # )
#         # 模型参数初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#         self.base = nn.Sequential()
#         self.base.add_module('conv1', self.conv1)
#         self.base.add_module('bn1', self.bn1)
#         self.base.add_module('relu', self.relu)
#         for i in range(len(self.layers)):
#             self.base.add_module(f'layer_{i}', getattr(self, f'layer_{i}'))
#         self.base.add_module('avgpool', self.avgpool)
#         self.base.add_module('flatten', nn.Flatten())
#
#         self.head = self.fc
#
#     def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
#                     stride: int = 1, dilate: bool = False, has_bn=True) -> List:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if has_bn:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     norm_layer(planes * block.expansion),
#                 )
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     nn.Identity(),
#                 )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, has_bn, self.ratio_LR))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, has_bn=has_bn, rank_rate=self.ratio_LR))
#
#         return layers
#
#     def recover_larger_model(self):
#         if self.ratio_LR >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.recover()
#
#     def decom_larger_model(self, rank_rate):
#         if rank_rate >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.decom(rank_rate)
#
#     def frobenius_decay(self):
#         loss = torch.tensor(0.0, device=self.conv1.weight.device)
#         if self.ratio_LR >= 1.0:
#             return loss
#         else:
#             for block in self.layers:
#                 loss += block.frobenius_loss()
#             return loss
#
#     # def _forward_impl(self, x: Tensor) -> Tensor:
#     #     x = self.conv1(x)
#     #     x = self.bn1(x)
#     #     x = self.relu(x)
#
#     #     for i in range(len(self.layers)):
#     #         layer = getattr(self, f'layer_{i}')
#     #         x = layer(x)
#
#     #     x = self.avgpool(x)
#     #     x = self.fc(x)
#
#     #     return x
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.base(x)
#         x = self.head(x)
#         return x
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)



class LOW_RANK_ResNet_Base_CIFAR(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_Base_CIFAR, self).__init__()
        self.ratio_LR = ratio_LR
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 第一个卷积层不进行操作

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if has_norm:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0)))

        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_norm=has_norm and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 模型参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_norm=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_norm, self.ratio_LR))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_norm=has_norm, rank_rate=self.ratio_LR))

        return layers

    def recover_larger_model(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            for block in self.layers:
                block.recover()

    def decom_larger_model(self, rank_rate):
        if rank_rate >= 1.0:
            return
        else:
            for block in self.layers:
                block.decom(rank_rate)

    def frobenius_decay(self):
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.ratio_LR >= 1.0:
            return loss
        else:
            for block in self.layers:
                loss += block.frobenius_loss()
            return loss
    # 返回编码器的特征
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class LOW_RANK_ResNet_CIFAR(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_CIFAR, self).__init__()
        self.ratio_LR = ratio_LR
        self.base = LOW_RANK_ResNet_Base_CIFAR(block=block,layers=layers,features=features,zero_init_residual=zero_init_residual,groups = groups,
            width_per_group= width_per_group,
            replace_stride_with_dilation =replace_stride_with_dilation,
            norm_layer= norm_layer,
            has_norm=has_norm,
            bn_block_num=bn_block_num,
            ratio_LR=ratio_LR)
        self.head = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)
    def recover_larger_model(self):
        self.base.recover_larger_model()

    def decom_larger_model(self, rank_rate):
        self.base.decom_larger_model(rank_rate)

    def frobenius_decay(self):
        return self.base.frobenius_decay()

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.base(x)
        x = self.head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# 添加了投影层的resnet,保证不同尺寸的resnet特征输出维度一致
# class LOW_RANK_ResNet_CIFAR_512(nn.Module):
#     def __init__(
#             self,
#             block: Low_RANK_BasicBlock,
#             layers: List[int],
#             features: List[int] = [64, 128, 256, 512],
#             num_classes: int = 1000,
#             zero_init_residual: bool = False,
#             groups: int = 1,
#             width_per_group: int = 64,
#             replace_stride_with_dilation: Optional[List[bool]] = None,
#             norm_layer: Optional[Callable[..., nn.Module]] = None,
#             has_bn=True,
#             bn_block_num=4,
#             ratio_LR=1.0
#     ) -> None:
#         super(LOW_RANK_ResNet_CIFAR_512, self).__init__()
#         self.ratio_LR = ratio_LR
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer
#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         # 第一个卷积层不进行操作
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         if has_bn:
#             self.bn1 = norm_layer(self.inplanes)
#         else:
#             self.bn1 = nn.Identity()
#         self.relu = nn.ReLU(inplace=True)
#         self.layers = []
#         self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0)))
#         for num in range(1, len(layers)):
#             self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
#                                                 dilate=replace_stride_with_dilation[num - 1],
#                                                 has_bn=has_bn and (num < bn_block_num)))
#         for i, layer in enumerate(self.layers):
#             setattr(self, f'layer_{i}', layer)
#         self.avgpool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )
#         # # 添加投影层，将特征维度映射到512
#         self.projection = nn.Sequential(
#             nn.Linear(features[len(layers) - 1] * block.expansion, 512),
#             norm_layer(512),
#             nn.ReLU(inplace=True)
#         )
#         self.fc = nn.Linear(512, num_classes)
#         # self.fc = nn.Sequential(
#         #     nn.AdaptiveAvgPool2d((1, 1)),
#         #     nn.Flatten(),
#         #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
#         # )
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
#         self.base = nn.Sequential()
#         self.base.add_module('conv1', self.conv1)
#         self.base.add_module('bn1', self.bn1)
#         self.base.add_module('relu', self.relu)
#         for i in range(len(self.layers)):
#             self.base.add_module(f'layer_{i}', getattr(self, f'layer_{i}'))
#         self.base.add_module('avgpool', self.avgpool)
#         self.base.add_module('flatten', nn.Flatten())
#         self.base.add_module('projection', self.projection)
#         self.head = self.fc
#
#     def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
#                     stride: int = 1, dilate: bool = False, has_bn=True) -> List:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if has_bn:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     norm_layer(planes * block.expansion),
#                 )
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     nn.Identity(),
#                 )
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, has_bn, self.ratio_LR))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, has_bn=has_bn, rank_rate=self.ratio_LR))
#         return layers
#
#     def recover_larger_model(self):
#         if self.ratio_LR >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.recover()
#
#     def decom_larger_model(self, rank_rate):
#         if rank_rate >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.decom(rank_rate)
#
#     def frobenius_decay(self):
#         loss = torch.tensor(0.0, device=self.conv1.weight.device)
#         if self.ratio_LR >= 1.0:
#             return loss
#         else:
#             for block in self.layers:
#                 loss += block.frobenius_loss()
#             return loss
#
#     # def _forward_impl(self, x: Tensor) -> Tensor:
#     #     x = self.conv1(x)
#     #     x = self.bn1(x)
#     #     x = self.relu(x)
#     #     for i in range(len(self.layers)):
#     #         layer = getattr(self, f'layer_{i}')
#     #         x = layer(x)
#     #     x = self.avgpool(x)
#     #     x = self.fc(x)
#     #     return x
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.base(x)
#         x = self.head(x)
#         return x
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)


class LOW_RANK_ResNet_Base_CIFAR_512(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_Base_CIFAR_512, self).__init__()
        self.ratio_LR = ratio_LR
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 第一个卷积层不进行操作

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if has_norm:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0)))

        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                                dilate=replace_stride_with_dilation[num - 1],
                                                has_norm=has_norm and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 添加投影层，将特征维度映射到统一的512维度
        self.projection = nn.Sequential(
            nn.Linear(features[len(layers) - 1] * block.expansion, 512),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_norm=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_norm, self.ratio_LR))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_norm, rank_rate=self.ratio_LR))

        return layers

    def recover_larger_model(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            for block in self.layers:
                block.recover()

    def decom_larger_model(self, rank_rate):
        if rank_rate >= 1.0:
            return
        else:
            for block in self.layers:
                block.decom(rank_rate)

    def frobenius_decay(self):
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.ratio_LR >= 1.0:
            return loss
        else:
            for block in self.layers:
                loss += block.frobenius_loss()
            return loss
    # 返回编码器的特征
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.projection(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class LOW_RANK_ResNet_CIFAR_512(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_CIFAR_512, self).__init__()
        self.ratio_LR = ratio_LR
        self.base = LOW_RANK_ResNet_Base_CIFAR_512(block=block,layers=layers,features=features,zero_init_residual=zero_init_residual,groups = groups,
            width_per_group= width_per_group,
            replace_stride_with_dilation =replace_stride_with_dilation,
            norm_layer= norm_layer,
            has_norm=has_norm,
            bn_block_num=bn_block_num,
            ratio_LR=ratio_LR)
        self.head = nn.Linear(512, num_classes)
    def recover_larger_model(self):
        self.base.recover_larger_model()

    def decom_larger_model(self, rank_rate):
        self.base.decom_larger_model(rank_rate)

    def frobenius_decay(self):
        return self.base.frobenius_decay()

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.base(x)
        x = self.head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

# 实现resnet多层投影输出
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        if x.ndim == 4:
            # 对卷积特征做全局池化
            x = torch.mean(x, dim=[2,3])
        return self.proj(x)
# 每个残差快为一个layer没有分stage 
# class LOW_RANK_ResNet_Base_CIFAR_MUTILPRO(nn.Module):

#     def __init__(
#             self,
#             block: Low_RANK_BasicBlock,
#             layers: List[int],
#             features: List[int] = [64, 128, 256, 512],
#             zero_init_residual: bool = False,
#             groups: int = 1,
#             width_per_group: int = 64,
#             replace_stride_with_dilation: Optional[List[bool]] = None,
#             norm_layer: Optional[Callable[..., nn.Module]] = None,
#             has_norm=True,
#             bn_block_num=4,
#             ratio_LR=1.0
#     ) -> None:
#         super(LOW_RANK_ResNet_Base_CIFAR_MUTILPRO, self).__init__()
#         self.ratio_LR = ratio_LR
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         self._norm_layer = norm_layer

#         self.inplanes = 64
#         self.dilation = 1
#         if replace_stride_with_dilation is None:
#             replace_stride_with_dilation = [False, False, False]
#         if len(replace_stride_with_dilation) != 3:
#             raise ValueError("replace_stride_with_dilation should be None "
#                              "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
#         self.groups = groups
#         self.base_width = width_per_group
#         # 第一个卷积层不进行操作

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         if has_norm:
#             self.bn1 = norm_layer(self.inplanes)
#         else:
#             self.bn1 = nn.Identity()
#         self.relu = nn.ReLU(inplace=True)

#         self.layers = []
#         self.layers.extend(self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0)))

#         for num in range(1, len(layers)):
#             self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
#                                                 dilate=replace_stride_with_dilation[num - 1],
#                                                 has_norm=has_norm and (num < bn_block_num)))

#         for i, layer in enumerate(self.layers):
#             setattr(self, f'layer_{i}', layer)

#         # 新增投影头，用于 MSE 对齐，每个 layer 一个,最后一个不需要投影头
#         self.proj_heads = nn.ModuleList()
#         for i, layer in enumerate(self.layers[:-1]):  # 最后一个 layer 不需要投影头
#             # 假设每个 block 输出通道数为 planes * block.expansion
#             out_ch = layer.conv2.out_channels if isinstance(layer, Low_RANK_BasicBlock) else layer.conv3.out_channels
#             self.proj_heads.append(ProjectionHead(out_ch, out_dim=128))
        
#         self.avgpool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten()
#         )

#         # 模型参数初始化
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
#                     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
#                 elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
#                     nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


#     def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
#                     stride: int = 1, dilate: bool = False, has_norm=True) -> List:
#         norm_layer = self._norm_layer
#         downsample = None
#         previous_dilation = self.dilation
#         if dilate:
#             self.dilation *= stride
#             stride = 1
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             if has_norm:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     norm_layer(planes * block.expansion),
#                 )
#             else:
#                 downsample = nn.Sequential(
#                     conv1x1(self.inplanes, planes * block.expansion, stride),
#                     nn.Identity(),
#                 )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
#                             self.base_width, previous_dilation, norm_layer, has_norm, self.ratio_LR))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=self.groups,
#                                 base_width=self.base_width, dilation=self.dilation,
#                                 norm_layer=norm_layer, has_norm=has_norm, rank_rate=self.ratio_LR))

#         return layers

#     def recover_larger_model(self):
#         if self.ratio_LR >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.recover()

#     def decom_larger_model(self, rank_rate):
#         if rank_rate >= 1.0:
#             return
#         else:
#             for block in self.layers:
#                 block.decom(rank_rate)

#     def frobenius_decay(self):
#         loss = torch.tensor(0.0, device=self.conv1.weight.device)
#         if self.ratio_LR >= 1.0:
#             return loss
#         else:
#             for block in self.layers:
#                 loss += block.frobenius_loss()
#             return loss
#     # 返回编码器的特征
#     def _forward_impl(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         features = []  # 初始化列表
#         for i in range(len(self.layers[:-1])):  # 最后一个 layer 不需要投影头
#             layer = getattr(self, f'layer_{i}')
#             x = layer(x)
#             # 提取投影特征
#             proj_feat = self.proj_heads[i](x)
#             features.append(proj_feat)
#         #单独处理最后一个 layer
#         layer = getattr(self, f'layer_{len(self.layers)-1}')
#         x = layer(x)
#         x = self.avgpool(x)

#         return x, features

#     def forward(self, x: Tensor) -> Tensor:
#         return self._forward_impl(x)



class LOW_RANK_ResNet_Base_CIFAR_MUTILPRO(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_Base_CIFAR_MUTILPRO, self).__init__()
        self.ratio_LR = ratio_LR
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if has_norm:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 构建 4 个 stage，每个 stage 是一个 Sequential 包含多个 block
        self.stages = nn.ModuleList()
        # stage 1
        blocks = self._make_layer(block, 64, layers[0], has_norm=has_norm and (bn_block_num > 0))
        self.stages.append(nn.Sequential(*blocks))
        # stage 2,3,4
        for num in range(1, len(layers)):
            blocks = self._make_layer(block, features[num], layers[num], stride=2,
                                      dilate=replace_stride_with_dilation[num - 1],
                                      has_norm=has_norm and (num < bn_block_num))
            self.stages.append(nn.Sequential(*blocks))

        # 为每个 stage 添加投影头最后一个不添加
        self.proj_heads = nn.ModuleList()
        for stage in self.stages[:-1]:
            last_block = stage[-1]  # 获取该 stage 的最后一个 block
            # 根据 block 类型获取输出通道数
            if isinstance(last_block, Low_RANK_BasicBlock):
                out_ch = last_block.conv2.out_channels
            else:
                # 其他可能的 block 类型（如 Bottleneck）
                out_ch = last_block.out_channels
            self.proj_heads.append(ProjectionHead(out_ch, out_dim=128))

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 模型参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, LOW_RANK_Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, Low_RANK_BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_norm=True) -> List:
        """构建单个 stage 的 block 列表（未封装为 Sequential）"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_norm:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_norm, self.ratio_LR))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_norm=has_norm, rank_rate=self.ratio_LR))
        return layers

    def recover_larger_model(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            for stage in self.stages:
                for block in stage:
                    block.recover()

    def decom_larger_model(self, rank_rate):
        if rank_rate >= 1.0:
            return
        else:
            for stage in self.stages:
                for block in stage:
                    block.decom(rank_rate)

    def frobenius_decay(self):
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        if self.ratio_LR >= 1.0:
            return loss
        else:
            for stage in self.stages:
                for block in stage:
                    loss += block.frobenius_loss()
            return loss

    def _forward_impl(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features = []
        for stage, proj_head in zip(self.stages[:-1], self.proj_heads):
            x = stage(x)                 # 经过当前 stage
            proj_feat = proj_head(x)     # 提取投影特征
            features.append(proj_feat)
        x = self.stages[-1](x)  # 最后一个 stage
        x = self.avgpool(x)
        return x, features

    def forward(self, x: Tensor):
        return self._forward_impl(x)

class LOW_RANK_ResNet_CIFAR_MUTILPRO(nn.Module):

    def __init__(
            self,
            block: Low_RANK_BasicBlock,
            layers: List[int],
            features: List[int] = [64, 128, 256, 512],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=1.0
    ) -> None:
        super(LOW_RANK_ResNet_CIFAR_MUTILPRO, self).__init__()
        self.ratio_LR = ratio_LR
        self.base = LOW_RANK_ResNet_Base_CIFAR_MUTILPRO(block=block,layers=layers,features=features,zero_init_residual=zero_init_residual,groups = groups,
            width_per_group= width_per_group,
            replace_stride_with_dilation =replace_stride_with_dilation,
            norm_layer= norm_layer,
            has_norm=has_norm,
            bn_block_num=bn_block_num,
            ratio_LR=ratio_LR)
        self.head = nn.Linear(features[len(layers) - 1] * block.expansion, num_classes)
    def recover_larger_model(self):
        self.base.recover_larger_model()

    def decom_larger_model(self, rank_rate):
        self.base.decom_larger_model(rank_rate)

    def frobenius_decay(self):
        return self.base.frobenius_decay()

    def _forward_impl(self, x: Tensor) -> Tensor:
        x,_ = self.base(x)
        x = self.head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)








# def resnet152(**kwargs: Any) -> ResNet:
#     return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

# def resnet101(**kwargs: Any) -> ResNet:
#     return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

# def resnet50(**kwargs: Any) -> ResNet:
#     return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

# def resnet34(**kwargs: Any) -> ResNet:
#     return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

# def resnet18(**kwargs: Any) -> ResNet: # 18 = 2 + 2 * (2 + 2 + 2 + 2)
#     return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

# def resnet10(**kwargs: Any) -> ResNet: # 10 = 2 + 2 * (1 + 1 + 1 + 1)
#     return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)

# def resnet8(**kwargs: Any) -> ResNet: # 8 = 2 + 2 * (1 + 1 + 1)
#     return ResNet(BasicBlock, [1, 1, 1], **kwargs)

# def resnet6(**kwargs: Any) -> ResNet: # 6 = 2 + 2 * (1 + 1)
#     return ResNet(BasicBlock, [1, 1], **kwargs)

# def resnet4(**kwargs: Any) -> ResNet: # 4 = 2 + 2 * (1)
#     return ResNet(BasicBlock, [1], **kwargs)


# --------------------------------------------------------------LOW_RANK

def low_rank_resnet152(**kwargs: Any) -> LOW_RANK_ResNet:
    return LOW_RANK_ResNet(LOW_RANK_Bottleneck, [3, 8, 36, 3], **kwargs)


def low_rank_resnet101(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet(LOW_RANK_Bottleneck, [3, 4, 23, 3], **kwargs)


def low_rank_resnet50(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet(LOW_RANK_Bottleneck, [3, 4, 6, 3], **kwargs)


def low_rank_resnet34(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [3, 4, 6, 3], **kwargs)


def low_rank_resnet18(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [2, 2, 2, 2], **kwargs)


def low_rank_resnet10(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [1, 1, 1, 1], **kwargs)


def low_rank_resnet8(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [1, 1, 1], **kwargs)


def low_rank_resnet6(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [1, 1], **kwargs)


def low_rank_resnet4(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return LOW_RANK_ResNet(Low_RANK_BasicBlock, [1], **kwargs)


def low_rank_resnet152_cifar(**kwargs: Any) -> LOW_RANK_ResNet:
    return LOW_RANK_ResNet_CIFAR(LOW_RANK_Bottleneck, [3, 8, 36, 3], **kwargs)


def low_rank_resnet101_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(LOW_RANK_Bottleneck, [3, 4, 23, 3], **kwargs)


def low_rank_resnet50_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(LOW_RANK_Bottleneck, [3, 4, 6, 3], **kwargs)


def low_rank_resnet34_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [3, 4, 6, 3], **kwargs)


def low_rank_resnet18_cifar(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [2, 2, 2, 2], **kwargs)


def low_rank_resnet16_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 2, 2, 2], **kwargs)


def low_rank_resnet14_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 1, 2, 2], **kwargs)


def low_rank_resnet12_cifar(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 1, 1, 2], **kwargs)


def low_rank_resnet10_cifar(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 1, 1, 1], **kwargs)


def low_rank_resnet8_cifar(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 1, 1], **kwargs)


def low_rank_resnet6_cifar(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1, 1], **kwargs)


def low_rank_resnet4_cifar(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return LOW_RANK_ResNet_CIFAR(Low_RANK_BasicBlock, [1], **kwargs)


def low_rank_resnet152_cifar_512(**kwargs: Any) -> LOW_RANK_ResNet:
    return LOW_RANK_ResNet_CIFAR_512(LOW_RANK_Bottleneck, [3, 8, 36, 3], **kwargs)


def low_rank_resnet101_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(LOW_RANK_Bottleneck, [3, 4, 23, 3], **kwargs)


def low_rank_resnet50_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(LOW_RANK_Bottleneck, [3, 4, 6, 3], **kwargs)


def low_rank_resnet34_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [3, 4, 6, 3], **kwargs)


def low_rank_resnet18_cifar_512(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [2, 2, 2, 2], **kwargs)


def low_rank_resnet16_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 2, 2, 2], **kwargs)


def low_rank_resnet14_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 1, 2, 2], **kwargs)


def low_rank_resnet12_cifar_512(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 1, 1, 2], **kwargs)


def low_rank_resnet10_cifar_512(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 1, 1, 1], **kwargs)


def low_rank_resnet8_cifar_512(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 1, 1], **kwargs)


def low_rank_resnet6_cifar_512(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1, 1], **kwargs)


def low_rank_resnet4_cifar_512(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return LOW_RANK_ResNet_CIFAR_512(Low_RANK_BasicBlock, [1], **kwargs)



def low_rank_resnet152_cifar_MUTIL(**kwargs: Any) -> LOW_RANK_ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(LOW_RANK_Bottleneck, [3, 8, 36, 3], **kwargs)


def low_rank_resnet101_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(LOW_RANK_Bottleneck, [3, 4, 23, 3], **kwargs)


def low_rank_resnet50_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(LOW_RANK_Bottleneck, [3, 4, 6, 3], **kwargs)


def low_rank_resnet34_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [3, 4, 6, 3], **kwargs)

def low_rank_resnet18_cifar_MUTIL(**kwargs: Any) -> ResNet:  # 18 = 2 + 2 * (2 + 2 + 2 + 2)
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [2, 2, 2, 2], **kwargs)


def low_rank_resnet16_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 2, 2, 2], **kwargs)

def low_rank_resnet14_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 1, 2, 2], **kwargs)


def low_rank_resnet12_cifar_MUTIL(**kwargs: Any) -> ResNet:
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 1, 1, 2], **kwargs)

def low_rank_resnet10_cifar_MUTIL(**kwargs: Any) -> ResNet:  # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 1, 1, 1], **kwargs)


def low_rank_resnet8_cifar_MUTIL(**kwargs: Any) -> ResNet:  # 8 = 2 + 2 * (1 + 1 + 1)
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 1, 1], **kwargs)

def low_rank_resnet6_cifar_MUTIL(**kwargs: Any) -> ResNet:  # 6 = 2 + 2 * (1 + 1)
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1, 1], **kwargs)


def low_rank_resnet4_cifar_MUTIL(**kwargs: Any) -> ResNet:  # 4 = 2 + 2 * (1)
    return LOW_RANK_ResNet_CIFAR_MUTILPRO(Low_RANK_BasicBlock, [1], **kwargs)






if __name__ == "__main__":
    model = low_rank_resnet18_cifar_MUTIL(
        features=[64, 128, 256, 512],
        num_classes=100,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=group_norm,
        has_norm=True,
        bn_block_num=4,
        ratio_LR=0.05)
    model.eval()
    # 2. 准备示例输入
    input_tensor = torch.randn(1, 3, 32, 32)

    # 3. 计算FLOPs 前向传播
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total()}")

    # 计算总参数数量和可训练参数数量
    # 生成详细的参数统计表
    print(parameter_count_table(model, max_depth=5))
    for param in model.parameters():
        print(param.shape)
    # freeze_bn_stats_for_finetune(model)
    print("\n模型结构:", model)
    print("model f_loss", model.frobenius_decay())
    # 创建随机输入数据
    batch_size = 1
    random_input = torch.randn(batch_size, 3, 32, 32)  # 模拟4张32×32的RGB图像

    print(f"\n输入数据形状: {random_input.shape}")
    print(f"输入数据范围: [{random_input.min():.3f}, {random_input.max():.3f}]")
    print(f"输入数据均值: {random_input.mean():.3f}, 标准差: {random_input.std():.3f}")

    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)

    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
    print(output)

    model.recover_larger_model()
    for name,param in model.named_parameters():
        print(f"参数名{name},权重形状{param.shape}")
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)
    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
    print("\n模型结构:", model)
    print(output)
    # 2. 准备示例输入
    input_tensor = torch.randn(1, 3, 32, 32)

    # 3. 计算FLOPs 前向传播
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total()}")

    # 计算总参数数量和可训练参数数量
    # 生成详细的参数统计表
    print(parameter_count_table(model, max_depth=4))

    model.decom_larger_model(0.05)
    for param in model.parameters():
        print(param.shape)
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)
    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
    print("\n模型结构:", model)
    print("model f_loss", model.frobenius_decay())
    print(output)
    # 2. 准备示例输入
    input_tensor = torch.randn(1, 3, 32, 32)

    # 3. 计算FLOPs 前向传播
    flops = FlopCountAnalysis(model, input_tensor)
    print(f"Total FLOPs: {flops.total()}")

    # 计算总参数数量和可训练参数数量
    # 生成详细的参数统计表
    print(parameter_count_table(model, max_depth=4))
    # with torch.no_grad():  # 不计算梯度
    #     base_output = model.base(random_input)

    # print(f"\nbase输出数据形状: {base_output.shape}")
    # print(f"bese输出数据范围: [{base_output.min():.3f}, {base_output.max():.3f}]")
    # print(f"输出数据均值: {base_output.mean():.3f}, 标准差: {base_output.std():.3f}")