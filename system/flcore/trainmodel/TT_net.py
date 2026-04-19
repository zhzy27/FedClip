import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
# from flcore.trainmodel.bilstm import *
# from flcore.trainmodel.resnet import *
# from flcore.trainmodel.alexnet import *
# from flcore.trainmodel.mobilenet_v2 import *
# from flcore.trainmodel.transformer import *
import copy
import math
import numpy as np
import string
# --------------------------------------------SVD分解---------------------------------------------------------------
class FactorizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, rank_rate, padding=None, stride=1, kernel_size=3, bias=True):
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
        # 初始化模型参数
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_u, a=math.sqrt(0))
        nn.init.kaiming_uniform_(self.conv_v, a=math.sqrt(0))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
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

    # 重塑: [out, in, K, K] -> [out*K, in*K]
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


class Hyper_CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16, ratio_LR=0.7):
        super(Hyper_CNN, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        if ratio_LR >= 1.0:
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        else:
            self.conv2 = FactorizedConv(in_channels=n_kernels, out_channels=2 * n_kernels, padding=0,
                                        rank_rate=ratio_LR, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 2 * n_kernels * 5 * 5
        # #
        # print(f"全连接层的输入维度为：",self.fc_input_dim)
        # 全连接层（可能低秩分解）
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = FactorizedLinear(in_features=self.fc_input_dim, out_features=2000, rank_rate=ratio_LR, bias=True)
            self.fc2 = FactorizedLinear(2000, 500, rank_rate=ratio_LR, bias=True)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def recover_larger_model(self):
        """将低秩层恢复为完整秩"""
        if self.ratio_LR >= 1.0:
            return
        self.conv2 = Recover_COV(self.conv2)
        # 恢复两个全连接层
        self.fc1 = Recover_LINEAR(self.fc1)
        self.fc2 = Recover_LINEAR(self.fc2)
        # 更新base索引
        self._rebuild_base()
        print("(卷积)恢复低秩模型为完整模型，fc1和fc2已恢复")

    def decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩"""
        if rank_rate >= 1.0:
            return

        if isinstance(self.conv2, nn.Conv2d):
            self.conv2 = Decom_COV(self.conv2, rank_rate)
        if isinstance(self.fc1, nn.Linear):
            self.fc1 = Decom_LINEAR(self.fc1, rank_rate)
        if isinstance(self.fc2, nn.Linear):
            self.fc2 = Decom_LINEAR(self.fc2, rank_rate)

        self._rebuild_base()
        print(f"将完整模型分解(卷积也分解)为低秩模型(rank_rate={rank_rate})")

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.frobenius_loss() + self.fc2.frobenius_loss() + self.conv2.frobenius_loss()

    def kronecker_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.kronecker_loss() + self.fc2.kronecker_loss() + self.conv2.kronecker_loss()

    def L2_decay(self):
        if self.ratio_LR >= 1.0:
            return torch.tensor(0.0, device=self.conv1.weight.device)
        return self.fc1.L2_loss() + self.fc2.L2_loss() + self.conv2.L2_loss()

    def forward(self, x):
        features = self.base(x)  # 提取特征
        output = self.head(features)  # 分类输出
        return output


# ----------------------------------------------TT分解---------------------------------------------------------------------
# ---------- 辅助函数 ----------
def Factorize(num):
    """原来的最接近两个整数因子分解（保留）"""
    if num <= 1:
        return [1, num]
    root = int(math.isqrt(num))
    for i in range(root, 0, -1):
        if num % i == 0:
            return [i, num // i]
    return [1, num]

# 卷积层分解后的
# 卷积层分解函数
def Decom_TT_COV(conv_model, ratio_LR=0.5):
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

    # 重塑: [out, in, K, K] -> [out*K, in*K]
    A = W.permute(0, 2, 1, 3).reshape(out_planes * kernel_size, in_planes * kernel_size)

    # SVD分解
    # 添加小的正则化到对角线
    A_reg = A + 1e-8 * torch.eye(A.shape[0], A.shape[1], device=A.device)
    U, S, Vh = torch.linalg.svd(A_reg, full_matrices=False)
    # U, S, Vh = torch.linalg.svd(A, full_matrices=False)

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




# ---------- TT-SVD ----------
def tt_svd(W_reshaped, tt_ranks_rate, device=None, dtype=None):
    """
    通用的 TT-SVD（逐模分解）实现。
    - W_reshaped: tensor，shape = (d1, d2, ..., dk)
    - tt_ranks: list of length k-1，表示每一段的秩 (r1, r2, ..., r_{k-1})
    返回 cores 列表，cores[i] 形状为 (r_{i-1}, d_i, r_i) （r_0 = 1, r_k = 1）
    """
    if device is None:
        device = W_reshaped.device
    if dtype is None:
        dtype = W_reshaped.dtype

    shape = list(W_reshaped.shape)
    k = len(shape)

    cores = []
    # C 用于保存当前剩余张量（矩阵化）
    C = W_reshaped.clone().to(device=device, dtype=dtype)
    # 初始把 C reshape 成 (d1, d2*d3*...*dk)
    C = C.reshape(shape[0], -1)
    r_pre = 1
    for i in range(k - 1):
        # SVD
        # torch.linalg.svd 返回 U, S, Vh (Vh 的形状是 (m, n) 并已共轭转置)
        C_reg = C + 1e-8 * torch.eye(C.shape[0], C.shape[1], device=C.device)
        U, S, Vh = torch.linalg.svd(C_reg, full_matrices=False)
        # U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        # 使用矩阵的两个维度确定rank
        max_rank = round(min(C.size(0),C.size(1)) * tt_ranks_rate)
        r = max(1, max_rank)
        print(f"核心截断的rank为{r}")
        U_trunc = U[:, :r]  # (left_dim, r)
        S_trunc = S[:r]     # (r,)
        Vh_trunc = Vh[:r, :]  # (r, right_dim)


        core = U_trunc.reshape(r_pre, shape[i], r).contiguous()
        cores.append(core.to(device=device, dtype=dtype))
        #迭代更新上一个r
        r_pre = r
        # 更新 C = S_trunc @ Vh_trunc  -> (r, right_dim)
        C = torch.diag(S_trunc) @ Vh_trunc
        # 如果后续还有维度，要 reshape 为 (r * d_{i+1}, rest)
        if i < k - 2:
            next_block = shape[i + 1]
            C = C.reshape(r * next_block, -1)

    # 最后一个 core: r_{k-2} × d_k × 1
    last_core = C.reshape(r_pre, shape[-1], 1).contiguous()
    cores.append(last_core.to(device=device, dtype=dtype))
    return cores

# ---------- 通用 TTLinear 实现 ----------
class TTLinear(nn.Module):
    """
    通用的 TT-分解的全连接层（支持任意数量的输入/输出因子）
    W ∈ R^{out_features × in_features}，内部以 TT 核表示。
    参数说明：
        in_features, out_features：原始的输入/输出维度
        in_dims: 列表，in_features 的分解，例如 [16,8,8]（乘积= in_features）
        out_dims: 列表，out_features 的分解，例如 [10,10]（乘积= out_features）
        tt_ranks: 可选列表，长度为 total_factors-1（即 len(in_dims)+len(out_dims)-1）。
                  若提供则直接使用。也可用 rank_rate 自动推断。
        rank_rate: float（0, +inf），用于按比例计算 rank（当 tt_ranks 未给出时）。
        bias: 是否使用偏置
    """
    def __init__(self, in_features, out_features,
                 in_dims=None, out_dims=None,
                tt_rank_rate=0.5, bias=True,
                 device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 维度分解
        if in_dims is None:
            # 默认把输入分成两个因子（原来行为）
            in_dims = Factorize(in_features)
        if out_dims is None:
            out_dims = Factorize(out_features)

        assert math.prod(in_dims) == in_features, f"in_dims 的乘积应等于 in_features ({math.prod(in_dims)} != {in_features})"
        assert math.prod(out_dims) == out_features, f"out_dims 的乘积应等于 out_features ({math.prod(out_dims)} != {out_features})"

        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        print(f"输入分解核心为{self.in_dims},输出分解核心为{self.out_dims}")

        # 因子总数 k
        self.k = len(self.in_dims) + len(self.out_dims)


        # 创建 tt cores：长度 k
        # core i 的形状： (r_{i-1}, d_i, r_i)，r_0 = r_k = 1
        self.tt_cores = nn.ParameterList()
        dims = self.in_dims + self.out_dims  # d_1 ... d_k
        # r_0, r_1, ..., r_{k-1}, r_k
        ranks = [1]

        for i in range(len(dims)-1):
            rank = round(min(dims[i]*ranks[-1],math.prod(dims[i+1:]))*tt_rank_rate)
            ranks.append(rank)
        ranks.append(1)
        print(f"设置的rank值为{ranks}")

        for i in range(self.k):
            r_prev = ranks[i]
            d_i = dims[i]
            r_i = ranks[i + 1]
            core = nn.Parameter(torch.empty(r_prev, d_i, r_i, device=self.device, dtype=self.dtype))
            print(f"创建的核心shape为:{core.shape}")
            self.tt_cores.append(core)

        # bias
        self.bias_flag = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=self.device, dtype=self.dtype))
        else:
            self.register_parameter('bias', None)

        # 初始化
        self.reset_parameters()

    def reset_parameters(self):
        """初始化 TT 核心和 bias"""
        for core in self.tt_cores:
            # Kaiming 标准初始化（对每个核心按其第二维 d_i）
            nn.init.kaiming_uniform_(core, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1.0 / math.sqrt(max(1, fan_in))
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        """
        精确模拟TT-SVD分解逆过程的重建函数。
        按照分解时的相反顺序重建完整权重。
        """
        cores = self.tt_cores
        k = len(cores)
        dims = self.in_dims + self.out_dims

        # 步骤1: 处理最后一个core (形状: r_{k-1}, d_k, 1)
        # 在TT-SVD中，最后一个core是 S_{k-1} * Vh_{k-1} 的reshape
        print(f"最后一个核心维度为{cores[-1].shape}")
        # 重建时，我们从这里开始
        current_matrix = cores[-1].squeeze(-1)  # 形状: (r_{k-1}, d_k)
        print(f"最后一个核心reshape后的温度为{current_matrix.shape}")

        # 步骤2: 从后向前处理每个core
        # 在TT-SVD中，每个中间步骤是: C = diag(S_i) @ Vh_i
        # 然后被reshape并与下一个core合并
        for i in range(k - 2, -1, -1):
            core_i = cores[i]  # 形状: (r_{i-1}, d_i, r_i)
            r_prev, d_i, r_i = core_i.shape

            # 在TT-SVD正向过程中，这一步是:
            # 1. 将当前矩阵reshape为 (r_i * d_{i+1}, 剩余维度)
            # 2. 进行SVD得到 U_i, S_i, Vh_i
            # 3. U_i reshape为 (r_prev, d_i, r_i) 成为core_i
            # 4. C = diag(S_i) @ Vh_i 成为下一轮迭代的矩阵

            # 逆向过程:
            # 1. core_i是U_i，形状为(r_prev, d_i, r_i)
            # 2. 我们需要将它reshape为矩阵并乘以当前矩阵

            # 将core_i reshape为矩阵: (r_prev * d_i, r_i)
            U_matrix = core_i.reshape(-1, r_i)  # 形状: (r_prev * d_i, r_i)
            print(f"当前要合并的核心维度为{core_i.shape},该核心reshape后的维度为{U_matrix.shape},之前合并的核心维度为{current_matrix.shape}")
            # 当前矩阵是上一轮的 diag(S_i) @ Vh_i
            # 我们需要计算: U_matrix @ current_matrix
            # 这相当于重建 SVD 前的矩阵: U @ (diag(S) @ Vh) = U @ diag(S) @ Vh
            current_matrix = torch.matmul(U_matrix, current_matrix)  # 形状: (r_prev * d_i, 剩余维度)
            # 重塑以便下一步迭代
            current_matrix = current_matrix.reshape(r_prev, -1)  # 形状: (r_prev, d_i * 剩余维度)

        # 此时current_matrix形状应为: (1, d_1 * d_2 * ... * d_k)
        current_matrix = current_matrix.squeeze(0)  # 形状: (d_1 * d_2 * ... * d_k,)

        # 重塑为原始维度
        full_tensor = current_matrix.reshape(*dims)

        # 转换为权重矩阵格式
        in_size = math.prod(self.in_dims)
        out_size = math.prod(self.out_dims)
        weight_matrix = full_tensor.reshape(in_size, out_size).T

        return weight_matrix.contiguous()

    def forward(self, x, debug=False):
        B = x.size(0)
        cores = self.tt_cores
        p = len(self.in_dims)

        # 安全的标签集：去掉'b'（batch）, 'r','s'（我们将显式使用）
        all_lower = list(string.ascii_lowercase)
        reserved_lower = set(['b', 'r', 's'])  # 不分配给 data letters
        data_letters = [ch for ch in all_lower if ch not in reserved_lower]

        # Uppercase 用于为每个 output core 分配独立输出维标签（不会与小写冲突）
        out_letters = list(string.ascii_uppercase)

        # reshape input and append trivial rank dim r0=1
        x = x.view(B, *self.in_dims)  # (B, d1, d2, ..., dp)
        state = x.unsqueeze(-1)  # (B, d1, d2, ..., dp, r0=1)

        # --------- contract input cores (0..p-1) ----------
        for i in range(p):
            core = cores[i]  # (r_prev, d_i, r_next)
            r_prev = core.size(0)
            assert state.size(
                -1) == r_prev, f"rank mismatch at input core {i}: state last dim {state.size(-1)} != {r_prev}"

            ndims = state.dim() - 2  # number of data dims currently in state (excluding batch & rank)
            if ndims > len(data_letters):
                raise RuntimeError("Too many data dims for available einsum letters.")

            # labels for current data dims: 保证第0个是当前待收缩的 d_i
            cur_letters = data_letters[:ndims]  # e.g. ['a','c','d',...], note 'b','r','s' excluded

            # build einsum strings
            # state: b + cur_letters + r
            lhs_state = 'b' + ''.join(cur_letters) + 'r'

            # core: r + current_d + s  (use 's' as new rank char)
            current_d = cur_letters[0] if ndims >= 1 else None
            if current_d is None:
                # 没有 data dims —— state is (B, r) 但这种情况通常不在输入阶段出现
                lhs_core = 'rs'  # core must be (r, d, s) but d absent -> malformed; keep for safety
                rhs_out = 'bs'  # fallback
            else:
                lhs_core = 'r' + current_d + 's'
                # output removes the current_d (index 0) and keeps the rest of cur_letters
                out_letters_list = cur_letters[1:]
                rhs_out = 'b' + ''.join(out_letters_list) + 's'

            eins = f"{lhs_state},{lhs_core}->{rhs_out}"
            if debug:
                print(f"[in {i}] state={tuple(state.shape)}, core={tuple(core.shape)}, eins='{eins}'")

            state = torch.einsum(eins, state, core)
            # after contraction: dims (B, d_{i+1}, ..., dp, r_next)

        # after input cores, state usually becomes (B, r_p) or (B, some_output_dims..., r_p)
        if debug:
            print("after inputs:", tuple(state.shape))

        # --------- contract output cores (p..k-1) ----------
        out_count = 0
        for j in range(p, len(cores)):
            core = cores[j]  # (r_prev, out_d_j, r_next)
            r_prev = core.size(0)
            assert state.size(
                -1) == r_prev, f"rank mismatch at output core {j}: state last dim {state.size(-1)} != {r_prev}"

            ndims = state.dim() - 2
            if ndims > len(data_letters):
                raise RuntimeError("Too many data dims for available einsum letters.")

            cur_letters = data_letters[:ndims]  # existing accumulated output dim labels (lowercase, no 'b')

            # choose a unique uppercase letter for this new output dimension
            if out_count >= len(out_letters):
                raise RuntimeError("Too many output cores for available uppercase letters.")
            new_out_label = out_letters[out_count]
            out_count += 1

            # build einsum
            lhs_state = 'b' + ''.join(cur_letters) + 'r'
            lhs_core = 'r' + new_out_label + 's'  # core (r, OUT, s)
            rhs_out = 'b' + ''.join(cur_letters) + new_out_label + 's'

            eins = f"{lhs_state},{lhs_core}->{rhs_out}"
            if debug:
                print(f"[out {j}] state={tuple(state.shape)}, core={tuple(core.shape)}, eins='{eins}'")

            state = torch.einsum(eins, state, core)
            # after: appended one output dim (new_out_label) before new rank

        # 最后：如果末尾是 rank dim = 1，去掉
        if state.dim() > 2 and state.size(-1) == 1:
            state = state.squeeze(-1)

        # flatten
        state = state.reshape(B, -1)

        if self.bias_flag and self.bias is not None:
            state = state + self.bias

        return state
    def frobenius_loss(self):
        W = self.reconstruct_full_weight()
        return torch.sum(W ** 2)


# ---------- 分解与恢复函数 ----------
def Decom_TTLinear(linear_model, in_dims=None, out_dims=None,  tt_rank_rate=None):
    """
    将普通 nn.Linear 分解为 TTLinear。
    支持用户传入 in_dims/out_dims 或使用 Factorize / FactorizeN。
    - linear_model: nn.Linear（weight shape: out_features x in_features）
    - rank_rate: 可选，若 tt_ranks 未给出则使用此值推断
    """
    device = linear_model.weight.device
    dtype = linear_model.weight.dtype

    in_features = linear_model.in_features
    out_features = linear_model.out_features
    bias = linear_model.bias is not None

    # 默认分解：输入分为两个因子，输出分为1个因子（和原版兼容）
    if in_dims is None:
        in_dims = Factorize(in_features)
    if out_dims is None:
        out_dims = Factorize(out_features)

    assert math.prod(in_dims) == in_features, "in_dims 乘积必须等于 in_features"
    assert math.prod(out_dims) == out_features, "out_dims 乘积必须等于 out_features"

    # 如果用户给了 rank_rate 而没给 tt_ranks，我们将在 TTLinear 中推断
    tt_linear = TTLinear(in_features, out_features,
                         in_dims=in_dims, out_dims=out_dims,
                        tt_rank_rate=tt_rank_rate,
                         bias=bias, device=device, dtype=dtype)
    tt_linear = tt_linear.to(device=device, dtype=dtype)

    # 取得原始权重并 reshape 为 (d1, d2, ..., dk)
    W_original = linear_model.weight.data.clone()  # (out_features, in_features)
    W = W_original.T  # (in_features, out_features)
    all_dims = in_dims + out_dims
    W_reshaped = W.reshape(*all_dims).to(device=device, dtype=dtype)

    # 如果 tt_ranks 未显式给出，这里我们可以用默认推断出的 tt_linear.tt_ranks
    cores = tt_svd(W_reshaped, tt_rank_rate, device=device, dtype=dtype)

    # 赋值
    for i, core in enumerate(cores):
        tt_linear.tt_cores[i].data.copy_(core)

    # 拷贝 bias
    if bias:
        tt_linear.bias.data.copy_(linear_model.bias.data)

    return tt_linear

def Recover_TTLinear(tt_linear):
    """将 TTLinear 恢复为普通 Linear"""
    in_features = tt_linear.in_features
    out_features = tt_linear.out_features
    bias = tt_linear.bias is not None

    W_full = tt_linear.reconstruct_full_weight()  # (out_features, in_features)
    linear_layer = nn.Linear(in_features, out_features, bias=bias).to(W_full.device, dtype=W_full.dtype)
    with torch.no_grad():
        linear_layer.weight.data.copy_(W_full)
        if bias:
            linear_layer.bias.data.copy_(tt_linear.bias.data)
    return linear_layer




# 修改Hyper_CNN类以支持TT分解
class Hyper_CNN_TT(nn.Module):
    def __init__(self, in_features=3, num_classes=10, n_kernels=16,
                 ratio_LR=0.7):
        super(Hyper_CNN_TT, self).__init__()
        self.ratio_LR = ratio_LR

        # 卷积层和池化层
        self.conv1 = nn.Conv2d(in_features, n_kernels, 5)
        if ratio_LR >= 1.0:
            self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        else:
            self.conv2 = FactorizedConv(in_channels=n_kernels, out_channels=2 * n_kernels,
                                        padding=0, rank_rate=ratio_LR, kernel_size=5, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 计算全连接层输入维度
        self.fc_input_dim = 2 * n_kernels * 5 * 5

        # 全连接层
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(self.fc_input_dim, 2000)
            self.fc2 = nn.Linear(2000, 500)
        else:
            self.fc1 = TTLinear(self.fc_input_dim, 2000, tt_rank_rate=ratio_LR)
            self.fc2 = TTLinear(2000, 500, tt_rank_rate=ratio_LR)

        # 激活函数和输出层
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(500, num_classes)

        # 构建共享基础部分
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

        # 个性化头部
        self.head = self.fc3

    def TT_recover_larger_model(self):
        """将低秩/TT层恢复为完整秩"""
        if self.ratio_LR < 1:
            # 恢复TT层
            if isinstance(self.conv2, FactorizedConv):
                self.conv2 = Recover_COV(self.conv2)
            self.fc1 = Recover_TTLinear(self.fc1)
            self.fc2 = Recover_TTLinear(self.fc2)
            # 更新base索引
            self._rebuild_base()
            print("恢复低秩/TT模型为完整模型")
        else:
            return

    def TT_decom_larger_model(self, rank_rate):
        """将完整秩层分解为低秩或TT"""
        if rank_rate < 1.0:
            if isinstance(self.conv2, nn.Conv2d):
                self.conv2 = Decom_TT_COV(self.conv2, rank_rate)
            if isinstance(self.fc1, nn.Linear):
                self.fc1 = Decom_TTLinear(self.fc1, tt_rank_rate=rank_rate)
            if isinstance(self.fc2, nn.Linear):
                self.fc2 = Decom_TTLinear(self.fc2, tt_rank_rate=rank_rate)
            self._rebuild_base()
            print(f"将完整模型进行TT分解分解比例为{rank_rate}")
        else:
            # 不需要分解
            return

    def _rebuild_base(self):
        """重构基础网络部分"""
        self.base = nn.Sequential(
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
            self.relu
        )

    def frobenius_decay(self):
        """计算Frobenius衰减"""
        total_loss = torch.tensor(0.0, device=self.conv1.weight.device)

        if self.ratio_LR < 1:
            total_loss += self.fc1.frobenius_loss() + self.fc2.frobenius_loss() + self.conv2.frobenius_loss()

        return total_loss

    def forward(self, x):
        features = self.base(x)
        output = self.head(features)
        return output


# def test_ttlinear_comprehensive():
#     """
#     全面的TTLinear验证函数
#     测试所有主要功能，确保实现正确
#     """
#     print("=" * 60)
#     print("开始全面的TTLinear验证测试")
#     print("=" * 60)
#
#     torch.manual_seed(42)
#     np.random.seed(42)
#
#     test_results = {}
#
#     # 测试1：基本功能测试
#     print("\n1. 基本功能测试")
#     print("-" * 40)
#
#     # 创建不同配置的TTLinear
#     test_cases = [
#         # (in_features, out_features, in_dims, out_dims, tt_rank_rate)
#         (24, 30, [2, 3, 4], [5, 6], 0.5),
#         (100, 50, [10, 10], [5, 10], 0.3),
#         (64, 32, [8, 8], [4, 8], 0.7),
#         (128, 128, [8, 8, 2], [8, 8, 2], 0.4),
#     ]
#
#     for i, (in_feat, out_feat, in_dims, out_dims, rate) in enumerate(test_cases):
#         print(f"\n测试用例 {i + 1}: in_features={in_feat}, out_features={out_feat}")
#         print(f"  in_dims={in_dims}, out_dims={out_dims}, tt_rank_rate={rate}")
#
#         # 创建TTLinear层
#         tt_layer = TTLinear(in_feat, out_feat,
#                             in_dims=in_dims, out_dims=out_dims,
#                             tt_rank_rate=rate, bias=False)
#
#         # 测试输入
#         B = 5
#         x = torch.randn(B, in_feat)
#
#         # 方法1: 使用TT前向传播
#         y_tt = tt_layer(x)
#
#         # 方法2: 使用重建权重计算
#         W_full = tt_layer.reconstruct_full_weight()
#         y_direct = torch.matmul(x, W_full.T)
#
#         # 计算相对误差
#         error = torch.norm(y_tt - y_direct) / torch.norm(y_direct)
#
#         test_name = f"test_basic_{i + 1}"
#         test_results[test_name] = error.item() < 1e-5
#
#         print(f"  TT前向传播形状: {y_tt.shape}")
#         print(f"  直接计算形状: {y_direct.shape}")
#         print(f"  相对误差: {error.item():.6e}")
#         print(f"  测试结果: {'✓ 通过' if error.item() < 1e-5 else '✗ 失败'}")
#
#     # 测试2：偏置测试
#     print("\n2. 偏置功能测试")
#     print("-" * 40)
#
#     in_feat, out_feat = 24, 30
#     tt_layer_with_bias = TTLinear(in_feat, out_feat,
#                                   in_dims=[2, 3, 4], out_dims=[5, 6],
#                                   tt_rank_rate=0.5, bias=True)
#
#     # 设置一个非零偏置
#     with torch.no_grad():
#         tt_layer_with_bias.bias.data = torch.ones(out_feat)
#
#     x = torch.randn(3, in_feat)
#     y_with_bias = tt_layer_with_bias(x)
#
#     # 使用无偏置的相同权重计算
#     tt_layer_no_bias = TTLinear(in_feat, out_feat,
#                                 in_dims=[2, 3, 4], out_dims=[5, 6],
#                                 tt_rank_rate=0.5, bias=False)
#
#     # 复制权重
#     for i in range(len(tt_layer_with_bias.tt_cores)):
#         tt_layer_no_bias.tt_cores[i].data.copy_(tt_layer_with_bias.tt_cores[i].data)
#
#     y_no_bias = tt_layer_no_bias(x)
#
#     # 检查偏置是否被正确添加
#     expected_y_with_bias = y_no_bias + tt_layer_with_bias.bias
#
#     bias_error = torch.norm(y_with_bias - expected_y_with_bias) / torch.norm(expected_y_with_bias)
#     test_results["test_bias"] = bias_error.item() < 1e-5
#
#     print(f"  带偏置输出: {y_with_bias[0, :5].tolist()}")
#     print(f"  预期输出: {expected_y_with_bias[0, :5].tolist()}")
#     print(f"  偏置测试误差: {bias_error.item():.6e}")
#     print(f"  测试结果: {'✓ 通过' if bias_error.item() < 1e-5 else '✗ 失败'}")
#
#     # 测试3：Frobenius范数测试
#     print("\n3. Frobenius范数测试")
#     print("-" * 40)
#
#     tt_layer = TTLinear(24, 30, in_dims=[2, 3, 4], out_dims=[5, 6], tt_rank_rate=0.5)
#
#     # 方法1: 使用重建权重计算Frobenius范数
#     W_full = tt_layer.reconstruct_full_weight()
#     frob_direct = torch.sum(W_full ** 2)
#
#     # 方法2: 使用TT层的frobenius_loss方法
#     frob_tt = tt_layer.frobenius_loss()
#
#     frob_error = torch.abs(frob_direct - frob_tt) / torch.abs(frob_direct)
#     test_results["test_frobenius"] = frob_error.item() < 1e-5
#
#     print(f"  直接计算Frobenius范数: {frob_direct.item():.6f}")
#     print(f"  TT方法计算Frobenius范数: {frob_tt.item():.6f}")
#     print(f"  相对误差: {frob_error.item():.6e}")
#     print(f"  测试结果: {'✓ 通过' if frob_error.item() < 1e-5 else '✗ 失败'}")
#
#     # 测试4：TT-SVD分解与重建测试
#     print("\n4. TT-SVD分解与重建测试")
#     print("-" * 40)
#
#     # 创建一个随机权重矩阵
#     in_dims = [3, 4]
#     out_dims = [5, 6]
#     in_features = math.prod(in_dims)
#     out_features = math.prod(out_dims)
#
#     # 随机权重矩阵
#     W_original = torch.randn(out_features, in_features)
#
#     # 重塑为TT-SVD需要的形状
#     W_reshaped = W_original.T.reshape(*in_dims, *out_dims)
#
#     # 进行TT-SVD分解
#     cores = tt_svd(W_reshaped, tt_ranks_rate=0.99)
#
#     # 重建权重
#     # 创建一个临时的TTLinear来使用其重建方法
#     temp_tt = TTLinear(in_features, out_features,
#                        in_dims=in_dims, out_dims=out_dims,
#                        tt_rank_rate=0.99, bias=False)
#
#     # 替换核心
#     for i, core in enumerate(cores):
#         temp_tt.tt_cores[i].data.copy_(core)
#
#     # 重建完整权重
#     W_reconstructed = temp_tt.reconstruct_full_weight()
#
#     # 计算重建误差
#     svd_error = torch.norm(W_original - W_reconstructed) / torch.norm(W_original)
#     test_results["test_tt_svd"] = svd_error.item() < 1e-5
#
#     print(f"  原始权重形状: {W_original.shape}")
#     print(f"  重建权重形状: {W_reconstructed.shape}")
#     print(f"  TT-SVD重建相对误差: {svd_error.item():.6e}")
#     print(f"  测试结果: {'✓ 通过' if svd_error.item() < 1e-5 else '✗ 失败'}")
#
#     # 测试5：Decom_TTLinear和Recover_TTLinear测试
#     print("\n5. Decom_TTLinear和Recover_TTLinear测试")
#     print("-" * 40)
#
#     # 创建一个普通Linear层
#     linear_layer = nn.Linear(24, 30, bias=True)
#
#     # 分解为TTLinear
#     tt_decomposed = Decom_TTLinear(linear_layer,
#                                    in_dims=[2, 3, 4],
#                                    out_dims=[5, 6],
#                                    tt_rank_rate=1.0)
#
#     # 恢复为普通Linear
#     linear_recovered = Recover_TTLinear(tt_decomposed)
#
#     # 测试输入
#     x = torch.randn(3, 24)
#
#     # 原始Linear输出
#     y_original = linear_layer(x)
#
#     # 恢复的Linear输出
#     y_recovered = linear_recovered(x)
#
#     decom_error = torch.norm(y_original - y_recovered) / torch.norm(y_original)
#     test_results["test_decom_recover"] = decom_error.item() < 1e-5
#
#     print(f"  原始Linear输出形状: {y_original.shape}")
#     print(f"  恢复Linear输出形状: {y_recovered.shape}")
#     print(f"  分解恢复相对误差: {decom_error.item():.6e}")
#     print(f"  测试结果: {'✓ 通过' if decom_error.item() < 1e-5 else '✗ 失败'}")
#
#     # 测试6：批量大小变化测试
#     print("\n6. 批量大小变化测试")
#     print("-" * 40)
#
#     tt_layer = TTLinear(24, 30, in_dims=[2, 3, 4], out_dims=[5, 6], tt_rank_rate=0.5)
#
#     batch_sizes = [1, 3, 10, 100]
#     batch_test_passed = True
#
#     for i, B in enumerate(batch_sizes):
#         x = torch.randn(B, 24)
#         y = tt_layer(x)
#
#         if y.shape[0] == B and y.shape[1] == 30:
#             print(f"  批量大小 {B}: 输出形状 {y.shape} ✓")
#         else:
#             print(f"  批量大小 {B}: 输出形状 {y.shape} ✗ (期望 ({B}, 30))")
#             batch_test_passed = False
#
#     test_results["test_batch_sizes"] = batch_test_passed
#
#     # 测试7：梯度测试
#     print("\n7. 梯度计算测试")
#     print("-" * 40)
#
#     tt_layer = TTLinear(24, 30, in_dims=[2, 3, 4], out_dims=[5, 6],
#                         tt_rank_rate=0.5, bias=True)
#
#     # 启用梯度
#     for param in tt_layer.parameters():
#         param.requires_grad_(True)
#
#     x = torch.randn(3, 24, requires_grad=True)
#     y = tt_layer(x)
#
#     # 创建目标值
#     target = torch.randn_like(y)
#
#     # 计算损失和梯度
#     loss = torch.nn.functional.mse_loss(y, target)
#     loss.backward()
#
#     # 检查梯度是否存在
#     has_gradients = True
#     for name, param in tt_layer.named_parameters():
#         if param.grad is None:
#             print(f"  参数 {name} 没有梯度 ✗")
#             has_gradients = False
#         elif torch.all(param.grad == 0):
#             print(f"  参数 {name} 梯度全为零 ✗")
#             has_gradients = False
#
#     if has_gradients:
#         print(f"  所有参数都有非零梯度 ✓")
#
#     test_results["test_gradients"] = has_gradients
#
#     # 测试8：压缩率测试
#     print("\n8. 压缩率计算测试")
#     print("-" * 40)
#
#     # 原始参数数量
#     in_features, out_features = 100, 100
#     original_params = in_features * out_features
#
#     # TTLinear参数数量
#     tt_layer = TTLinear(in_features, out_features,
#                         in_dims=[10, 10], out_dims=[10, 10],
#                         tt_rank_rate=0.2)
#
#     # 计算TT参数数量
#     tt_params = 0
#     for core in tt_layer.tt_cores:
#         tt_params += core.numel()
#     if tt_layer.bias is not None:
#         tt_params += tt_layer.bias.numel()
#
#     compression_ratio = original_params / tt_params
#
#     print(f"  原始参数数量: {original_params}")
#     print(f"  TT参数数量: {tt_params}")
#     print(f"  压缩率: {compression_ratio:.2f}x")
#     print(f"  参数减少: {(1 - tt_params / original_params) * 100:.1f}%")
#
#     # 测试9：不同秩比例的影响
#     print("\n9. 不同秩比例的影响测试")
#     print("-" * 40)
#
#     in_features, out_features = 64, 64
#     x = torch.randn(2, in_features)
#
#     rank_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
#     prev_output = None
#
#     for rate in rank_rates:
#         tt_layer = TTLinear(in_features, out_features,
#                             in_dims=[8, 8], out_dims=[8, 8],
#                             tt_rank_rate=rate, bias=False)
#
#         y = tt_layer(x)
#
#         # 计算参数数量
#         tt_params = sum(core.numel() for core in tt_layer.tt_cores)
#
#         print(f"  秩比例 {rate:.1f}: 参数数量 {tt_params}, 输出范数 {torch.norm(y):.4f}")
#
#         if prev_output is not None:
#             diff = torch.norm(y - prev_output) / torch.norm(prev_output)
#             print(f"    与前一个输出的差异: {diff.item():.6e}")
#
#         prev_output = y
#
#     print("\n" + "=" * 60)
#     print("测试结果汇总")
#     print("=" * 60)
#
#     passed_count = sum(test_results.values())
#     total_count = len(test_results)
#
#     for test_name, passed in test_results.items():
#         status = "✓ 通过" if passed else "✗ 失败"
#         print(f"  {test_name:30} {status}")
#
#     print(f"\n总计: {passed_count}/{total_count} 个测试通过")
#
#     if passed_count == total_count:
#         print("\n🎉 所有测试通过！TTLinear实现正确。")
#     else:
#         print(f"\n⚠️  {total_count - passed_count} 个测试失败，请检查实现。")
#
#     return test_results
#
#
# def test_edge_cases():
#     """
#     测试边界情况和极端条件
#     """
#     print("\n" + "=" * 60)
#     print("边界情况测试")
#     print("=" * 60)
#
#     torch.manual_seed(42)
#
#     edge_results = {}
#
#     # 测试1：极小维度
#     print("\n1. 极小维度测试")
#     try:
#         tt = TTLinear(2, 3, in_dims=[2], out_dims=[3], tt_rank_rate=0.5)
#         x = torch.randn(1, 2)
#         y = tt(x)
#         print(f"  in_features=2, out_features=3: 输出形状 {y.shape} ✓")
#         edge_results["tiny_dims"] = True
#     except Exception as e:
#         print(f"  in_features=2, out_features=3: 失败 - {e}")
#         edge_results["tiny_dims"] = False
#
#     # 测试2：大维度
#     print("\n2. 大维度测试")
#     try:
#         tt = TTLinear(1000, 800, in_dims=[10, 10, 10], out_dims=[10, 8, 10], tt_rank_rate=0.1)
#         x = torch.randn(2, 1000)
#         y = tt(x)
#         print(f"  in_features=1000, out_features=800: 输出形状 {y.shape} ✓")
#         edge_results["large_dims"] = True
#     except Exception as e:
#         print(f"  in_features=1000, out_features=800: 失败 - {e}")
#         edge_results["large_dims"] = False
#
#     # 测试3：秩为1的情况
#     print("\n3. 极小秩测试")
#     try:
#         tt = TTLinear(24, 30, in_dims=[2, 3, 4], out_dims=[5, 6], tt_rank_rate=0.01)
#         # 确保所有秩至少为1
#         for i, core in enumerate(tt.tt_cores):
#             print(f"  Core {i} 形状: {core.shape}")
#         edge_results["tiny_rank"] = True
#     except Exception as e:
#         print(f"  极小秩测试失败: {e}")
#         edge_results["tiny_rank"] = False
#
#     # 测试4：秩比例大于1
#     print("\n4. 大秩比例测试")
#     try:
#         tt = TTLinear(24, 30, in_dims=[2, 3, 4], out_dims=[5, 6], tt_rank_rate=2.0)
#         for i, core in enumerate(tt.tt_cores):
#             print(f"  Core {i} 形状: {core.shape}")
#         edge_results["large_rank_rate"] = True
#     except Exception as e:
#         print(f"  大秩比例测试失败: {e}")
#         edge_results["large_rank_rate"] = False
#
#     # 测试5：自定义维度分解
#     print("\n5. 自定义维度分解测试")
#     try:
#         # 使用质数维度
#         tt = TTLinear(2 * 3 * 5, 7 * 11, in_dims=[2, 3, 5], out_dims=[7, 11], tt_rank_rate=0.5)
#         x = torch.randn(3, 30)
#         y = tt(x)
#         print(f"  质数维度分解: 输出形状 {y.shape} ✓")
#         edge_results["prime_dims"] = True
#     except Exception as e:
#         print(f"  质数维度分解失败: {e}")
#         edge_results["prime_dims"] = False
#
#     return edge_results
#
#
# def test_performance():
#     """
#     性能测试：比较TTLinear和普通Linear的速度和内存使用
#     """
#     print("\n" + "=" * 60)
#     print("性能测试")
#     print("=" * 60)
#
#     import time
#
#     # 设置测试参数
#     in_features, out_features = 1024, 1024
#     batch_size = 32
#
#     # 创建普通Linear层
#     linear = nn.Linear(in_features, out_features, bias=False)
#
#     # 创建TTLinear层（使用分解）
#     in_dims = Factorize(in_features)
#     out_dims = Factorize(out_features)
#     tt_linear = TTLinear(in_features, out_features,
#                          in_dims=in_dims, out_dims=out_dims,
#                          tt_rank_rate=0.2, bias=False)
#
#     # 复制权重
#     with torch.no_grad():
#         tt_weight = tt_linear.reconstruct_full_weight()
#         linear.weight.data.copy_(tt_weight)
#
#     # 创建测试数据
#     x = torch.randn(batch_size, in_features)
#
#     # 预热
#     for _ in range(10):
#         _ = linear(x)
#         _ = tt_linear(x)
#
#     # 测试普通Linear
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     start = time.time()
#     for _ in range(100):
#         _ = linear(x)
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     linear_time = time.time() - start
#
#     # 测试TTLinear
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     start = time.time()
#     for _ in range(100):
#         _ = tt_linear(x)
#     torch.cuda.synchronize() if torch.cuda.is_available() else None
#     tt_time = time.time() - start
#
#     # 计算参数数量
#     linear_params = linear.weight.numel()
#     tt_params = sum(core.numel() for core in tt_linear.tt_cores)
#
#     print(f"普通Linear:")
#     print(f"  参数数量: {linear_params}")
#     print(f"  100次前向传播时间: {linear_time:.4f}秒")
#     print(f"  平均每次: {linear_time / 100 * 1000:.2f}毫秒")
#
#     print(f"\nTTLinear:")
#     print(f"  参数数量: {tt_params} (压缩率: {linear_params / tt_params:.2f}x)")
#     print(f"  100次前向传播时间: {tt_time:.4f}秒")
#     print(f"  平均每次: {tt_time / 100 * 1000:.2f}毫秒")
#
#     print(f"\n速度比: {linear_time / tt_time:.2f}x")
#     print(f"内存节省: {(1 - tt_params / linear_params) * 100:.1f}%")
#
#
# # 运行所有测试
# if __name__ == "__main__":
#     # 运行全面测试
#     results = test_ttlinear_comprehensive()
#
#     # 运行边界情况测试
#     edge_results = test_edge_cases()
#
#     # 运行性能测试
#     test_performance()
#
#     # 最终总结
#     print("\n" + "=" * 60)
#     print("所有测试完成")
#     print("=" * 60)




if __name__ == "__main__":
    torch.set_printoptions(precision=10)
    TT_model = Hyper_CNN_TT(in_features=3, num_classes=10, n_kernels=16,
                            ratio_LR=1.0)
    x = torch.randn((1, 3, 32, 32))
    out = TT_model(x)
    print(TT_model)
    print(f"模型输出维度{out.shape},{out}")

    # TT_model.TT_recover_larger_model()
    # TT_recover_model = copy.deepcopy(TT_model)
    # print(TT_recover_model)
    # recover_out = TT_recover_model(x)
    # print(f"模型输出维度{recover_out.shape},{recover_out}")

    TT_model.TT_decom_larger_model(1.0)
    TT_recover_TT_model = copy.deepcopy(TT_model)
    recover_TT_out = TT_recover_TT_model(x)
    print(TT_recover_TT_model)
    print(f"模型输出维度{recover_TT_out.shape},{recover_TT_out}")


