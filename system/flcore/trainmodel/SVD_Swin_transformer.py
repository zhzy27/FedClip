# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import math
import torch.nn.functional as F
# from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 分解后的全连接层 (二维矩阵存储)
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate, bias=True):
        super(FactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 中间rank值
        self.rank = max(1, round(rank_rate * min(in_features, out_features)))

        # 二维矩阵参数
        # 第一个全连接层的参数（维度为 r*in）
        self.weight_v = nn.Parameter(torch.Tensor(self.rank, in_features))
        # 第二个全连接层的参数 (维度为 out*r)
        self.weight_u = nn.Parameter(torch.Tensor(out_features, self.rank))

        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.weight_v, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight_u.size(1))  # rank
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        return self.weight_u @ self.weight_v

    def forward(self, x):
        """
        前向传播分为两步：
        1. 降维投影：x -> (batch,  rank)
        2. 升维投影：x -> (batch, out_features)
        """
        # 第一步：降维投影 (输入特征空间 -> 低秩空间)  传入 F.linear() 的权重存储形式与正常线性层 (nn.Linear) 完全一致必须是（out*in）
        x = F.linear(x, self.weight_v, None)  # 形状: (batch,  rank)

        # 第二步：升维投影 (低秩空间 -> 输出特征空间)
        x = F.linear(x, self.weight_u, self.bias)  # 形状: (batch, out_features)

        return x

    def frobenius_loss(self):
        W = self.weight_u @ self.weight_v
        return torch.sum(W ** 2)

    def L2_loss(self):
        return torch.norm(self.weight_v) ** 2 + torch.norm(self.weight_u) ** 2

    def kronecker_loss(self):
        return (torch.norm(self.weight_v) ** 2) * (torch.norm(self.weight_u) ** 2)


# 全连接层分解函数(将全连接权重  W（out*in）分解为 out*r（第二个全连接权重） r*in （第一个权全连接重）)
def Decom_LINEAR(linear_model, ratio_LR=0.5):
    in_features = linear_model.in_features
    out_features = linear_model.out_features
    has_bias = linear_model.bias is not None

    # 创建分解层
    factorized_linear = FactorizedLinear(in_features, out_features, ratio_LR, has_bias)

    # SVD分解（属注意与torch.svd函数的区别  主要是第三个矩阵）  Vh是V矩阵的转置（就是第三个矩阵）
    U, S, Vh = torch.linalg.svd(linear_model.weight.data, full_matrices=False)

    # 计算截断秩
    rank = factorized_linear.rank

    # 分配奇异值  第一个矩阵切列   第三个矩阵切行
    S_sqrt = torch.sqrt(S[:rank])
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)  # shape out*r
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]  # shape r*in

    # 加载参数
    with torch.no_grad():
        factorized_linear.weight_v.copy_(V_weight)
        factorized_linear.weight_u.copy_(U_weight)
        if has_bias:
            factorized_linear.bias.copy_(linear_model.bias.data)

    return factorized_linear


# 全连接层恢复函数
def Recover_LINEAR(factorized_linear):
    in_features = factorized_linear.in_features
    out_features = factorized_linear.out_features
    has_bias = factorized_linear.bias is not None

    # # 重建权重
    # weight = factorized_linear.weight_u @ factorized_linear.weight_v
    weight = factorized_linear.reconstruct_full_weight()

    # 创建原始线性层
    recovered_linear = nn.Linear(in_features, out_features, bias=has_bias)

    # 加载参数
    with torch.no_grad():
        recovered_linear.weight.copy_(weight)
        if has_bias:
            recovered_linear.bias.copy_(factorized_linear.bias)

    return recovered_linear


try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from ..kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class Low_Rank_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,ratio_LR=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.ratio_LR = ratio_LR
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(in_features, hidden_features)
        else:
            self.fc1 = FactorizedLinear(in_features, hidden_features, ratio_LR, bias=True)
        self.act = act_layer()
        if ratio_LR >= 1.0:
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            self.fc2 =FactorizedLinear(hidden_features, out_features, ratio_LR, bias=True)
        self.drop = nn.Dropout(drop)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            print("Recovering Low-Rank FeedForward Layer...")
            # 恢复并更新fc1
            self.fc1 = Recover_LINEAR(self.fc1)
            # 恢复并更新fc2
            self.fc2 = Recover_LINEAR(self.fc2)

    # 这个是按照传递的分解比例进行分解
    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank FeedForward Layer...")
            # 分解并更新fc1
            self.fc1 = Decom_LINEAR(self.fc1, ratio_LR)
            # 分解并更新fc2
            self.fc2 = Decom_LINEAR(self.fc2, ratio_LR)


    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR < 1.0:
            loss += self.fc1.frobenius_loss()
            loss += self.fc2.frobenius_loss()
        return loss

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class  Low_Rank_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,ratio_LR=1.0):

        super().__init__()
        self.ratio_LR=ratio_LR
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if ratio_LR >= 1.0:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv =  FactorizedLinear(dim, dim * 3, ratio_LR, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if ratio_LR >= 1.0:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = FactorizedLinear(dim, dim, ratio_LR, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def recover(self):
        """恢复为完整秩的线性层"""
        if self.ratio_LR >= 1.0:
            return
        print("Recovering Low-Rank Attention Layer...")
        self.qkv = Recover_LINEAR(self.qkv)
        self.proj = Recover_LINEAR(self.proj)

    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank Attention Layer...")
            # 分解QKV投影
            self.qkv = Decom_LINEAR(self.qkv, ratio_LR)
            self.proj = Decom_LINEAR(self.proj, ratio_LR)


    def frobenius_loss(self):
        """计算低秩近似与原始权重的Frobenius损失"""
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR < 1.0:
            loss += self.qkv.frobenius_loss()
            loss += self.proj.frobenius_loss()
        return loss

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Low_Rank_SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,ratio_LR=1.0):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = Low_Rank_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,ratio_LR=ratio_LR)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Low_Rank_Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,ratio_LR=ratio_LR)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process
    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        print("Decomposing Low-Rank Transformer Layers...")
        self.attn.decom(ratio_LR=ratio_LR)
        self.mlp.decom(ratio_LR=ratio_LR)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        print("Recovering Low-Rank Transformer Layers...")
        self.attn.recover()
        self.mlp.recover()

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR >= 1.0:
            return loss
        loss += self.attn.frobenius_loss()
        loss += self.mlp.frobenius_loss()
        return loss


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class Low_Rank_PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm,ratio_LR=1.0):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.input_resolution = input_resolution
        self.dim = dim
        if self.ratio_LR<=1.0:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        else:
            self.reduction = FactorizedLinear(4 * dim, 2 * dim, ratio_LR, bias=False)
        self.norm = norm_layer(4 * dim)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            self.reduction = Recover_LINEAR(self.reduction)

    # 这个是按照传递的分解比例进行分解
    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        else:
            self.reduction = Decom_LINEAR(self.reduction, ratio_LR)


    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR < 1.0:
            loss += self.reduction.frobenius_loss()
        return loss

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class Low_Rank_BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False,ratio_LR=1.0):

        super().__init__()
        self.ratio_LR=ratio_LR
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            Low_Rank_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,ratio_LR=ratio_LR)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        for block in self.blocks:
            block.decom(ratio_LR)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        for block in self.blocks:
            block.recover()

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR >= 1.0:
            return loss
        for block in self.blocks:
            loss += block.frobenius_loss()
        return loss


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class Low_Rank_SwinTransformer_Base(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,ratio_LR=1.0, **kwargs):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Low_Rank_BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=Low_Rank_PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,ratio_LR=ratio_LR)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        for l in self.layers:
            l.decom(ratio_LR)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        for l in self.layers:
            l.recover()

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR >= 1.0:
            return loss
        for l in self.layers:
            loss += l.frobenius_loss()
        return loss

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops


class Low_Rank_SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,ratio_LR=1.0, **kwargs):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.base = Low_Rank_SwinTransformer_Base(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                 embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                 window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                 drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, fused_window_process= fused_window_process,ratio_LR=ratio_LR)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()



    def decom_larger_model(self, ratio_LR):
        self.base.decom(ratio_LR)

    def recover_larger_model(self):
        self.base.recover()

    def frobenius_decay(self):
        return self.base.frobenius_loss()

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.base.flops()
        flops += self.num_features * self.num_classes
        return flops
# if __name__ == '__main__':

#     # # 适用于CIFAR100的轻量级ViT模型  四层VIT
#     model = Low_Rank_SwinTransformer(
#         img_size=32,
#         patch_size=2,  # 关键
#         in_chans=3,
#         num_classes=100,  # CIFAR-100
#         embed_dim=64,  # 不要太大
#         depths=[2, 2, 2],  # 足够
#         num_heads=[2, 4, 8],
#         window_size=4,  # CIFAR 合理窗口
#         mlp_ratio=4.0,
#         drop_rate=0.0,
#         attn_drop_rate=0.0,
#         drop_path_rate=0.1,
#         patch_norm=True,
#         ratio_LR=0.1
#     )


#     model.eval()
#     # 2. 准备示例输入
#     input_tensor = torch.randn(1, 3, 32, 32)

#     # 3. 计算FLOPs 前向传播
#     flops = FlopCountAnalysis(model, input_tensor)
#     print(f"Total FLOPs: {flops.total()}")

#     # 计算总参数数量和可训练参数数量
#     # 生成详细的参数统计表
#     print(parameter_count_table(model, max_depth=4))

#     print("\n模型结构:", model)
#     print("model f_loss", model.frobenius_decay())
#     # 创建随机输入数据
#     batch_size = 1
#     random_input = torch.randn(batch_size, 3, 32, 32)  # 模拟4张32×32的RGB图像

#     print(f"\n输入数据形状: {random_input.shape}")
#     print(f"输入数据范围: [{random_input.min():.3f}, {random_input.max():.3f}]")
#     print(f"输入数据均值: {random_input.mean():.3f}, 标准差: {random_input.std():.3f}")

#     # 前向传播
#     model.eval()  # 设置为评估模式
#     with torch.no_grad():  # 不计算梯度
#         output = model(random_input)

#     print(f"\n输出数据形状: {output.shape}")
#     print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
#     print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
#     print(output)

#     model.recover_larger_model()

#     # 前向传播
#     model.eval()  # 设置为评估模式
#     with torch.no_grad():  # 不计算梯度
#         output = model(random_input)
#     print(f"\n输出数据形状: {output.shape}")
#     print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
#     print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
#     print("\n模型结构:", model)
#     print(output)
#     # 2. 准备示例输入
#     input_tensor = torch.randn(1, 3, 32, 32)

#     # 3. 计算FLOPs 前向传播
#     flops = FlopCountAnalysis(model, input_tensor)
#     print(f"Total FLOPs: {flops.total()}")

#     # 计算总参数数量和可训练参数数量
#     # 生成详细的参数统计表
#     print(parameter_count_table(model, max_depth=4))

#     model.decom_larger_model(0.5)

#     # 前向传播
#     model.eval()  # 设置为评估模式
#     with torch.no_grad():  # 不计算梯度
#         output = model(random_input)
#     print(f"\n输出数据形状: {output.shape}")
#     print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
#     print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
#     print("\n模型结构:", model)
#     print("model f_loss", model.frobenius_decay())
#     print(output)
#     # 2. 准备示例输入
#     input_tensor = torch.randn(1, 3, 32, 32)

#     # 3. 计算FLOPs 前向传播
#     flops = FlopCountAnalysis(model, input_tensor)
#     print(f"Total FLOPs: {flops.total()}")

#     # 计算总参数数量和可训练参数数量
#     # 生成详细的参数统计表
#     print(parameter_count_table(model, max_depth=4))
#     # with torch.no_grad():  # 不计算梯度
#     #     base_output = model.base(random_input)

#     # print(f"\nbase输出数据形状: {base_output.shape}")
#     # print(f"bese输出数据范围: [{base_output.min():.3f}, {base_output.max():.3f}]")
#     # print(f"输出数据均值: {base_output.mean():.3f}, 标准差: {base_output.std():.3f}")