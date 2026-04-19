import torch
from torch import nn, Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import math
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def Factorize(num):
    """原来的最接近两个因子分解（保留）"""
    if num <= 1:
        return [1, num]
    root = int(math.isqrt(num))
    for i in range(root, 0, -1):
        if num % i == 0:
            return [i, num // i]
    return [1, num]


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
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        # 使用矩阵的两个维度确定rank
        max_rank = round(min(C.size(0), C.size(1)) * tt_ranks_rate)
        r = max(1, max_rank)
        # print(f"该层中间设置的秩为{r}")
        # print(f"核心截断的rank为{r}")
        U_trunc = U[:, :r]  # (left_dim, r)
        S_trunc = S[:r]  # (r,)
        Vh_trunc = Vh[:r, :]  # (r, right_dim)

        # 将奇异值分给分给U和V（第一种分配方式）
        S_trunc_sqrt = torch.sqrt(S_trunc)
        U_weight = U_trunc @ torch.diag(S_trunc_sqrt)
        core = U_weight.reshape(r_pre, shape[i], r).contiguous()
        cores.append(core.to(device=device, dtype=dtype))
        # 迭代更新上一个r
        r_pre = r
        # 更新 C = S_trunc_sqrt @ Vh_trunc  -> (r, right_dim)
        C = torch.diag(S_trunc_sqrt) @ Vh_trunc

        # # 第二种分配方式
        # core = U_trunc.reshape(r_pre, shape[i], r).contiguous()
        # cores.append(core.to(device=device, dtype=dtype))
        # # 迭代更新上一个r
        # r_pre = r
        # # 更新 C = S_trunc @ Vh_trunc  -> (r, right_dim)
        # C = torch.diag(S_trunc) @ Vh_trunc
        # 如果后续还有维度，要 reshape 为 (r * d_{i+1}, rest)
        if i < k - 2:
            next_block = shape[i + 1]
            C = C.reshape(r * next_block, -1)

    # 最后一个 core: r_{k-2} × d_k × 1
    last_core = C.reshape(r_pre, shape[-1], 1).contiguous()
    cores.append(last_core.to(device=device, dtype=dtype))
    return cores


class TTLinear(nn.Module):
    """
    通用的 TT-分解的全连接层，支持任意数量的核心
    当只有两个核心时，自动等价于矩阵分解 W = U @ V
    """

    def __init__(self, in_features, out_features,
                 tt_rank_rate=0.5, bias=True,
                 device=None, dtype=None,core=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        if core ==2:
            # 默认把输入分成两个因子（原来行为）
            in_dims = [in_features]
            out_dims = [out_features]
        elif core ==3:
            if in_features > out_features:
                in_dims = Factorize(in_features)
                out_dims = [out_features]
            else:
                in_dims = [in_features]
                out_dims = Factorize(out_features)
        elif core==4:
            # 默认把输入分成四个因子（原来行为）
            in_dims = Factorize(in_features)
            out_dims = Factorize(out_features)
        else:
            raise ValueError("分解核太多了会影响性能")


        assert math.prod(in_dims) == in_features, f"in_dims乘积应等于in_features"
        assert math.prod(out_dims) == out_features, f"out_dims乘积应等于out_features"

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.p = len(in_dims)  # 输入核心数
        self.q = len(out_dims)  # 输出核心数
        self.k = self.p + self.q  # 总核心数
        # print(f"总核心数为{self.k}")
        # 创建TT核心
        self.tt_cores = nn.ParameterList()
        dims = self.out_dims + self.in_dims  # 注意：输出维度在前，输入维度在后

        # 计算秩
        ranks = [1]
        for i in range(self.k - 1):
            # 计算最大可能秩（基于矩阵秩的上界）
            # max_rank = min(
            #     math.prod(dims[:i+1]),  # 左边所有维度的乘积
            #     math.prod(dims[i+1:])   # 右边所有维度的乘积
            # )
            rank_1 = round(min(dims[i] * ranks[-1], math.prod(dims[i + 1:])) * tt_rank_rate)
            # 使用rank_rate计算实际秩
            rank = max(1, rank_1)
            ranks.append(rank)
        ranks.append(1)

        # 创建核心
        for i in range(self.k):
            r_prev = ranks[i]
            d_i = dims[i]
            r_next = ranks[i + 1]
            core = nn.Parameter(torch.empty(r_prev, d_i, r_next,
                                            device=device, dtype=dtype))
            self.tt_cores.append(core)

        # 偏置
        self.bias_flag = bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                 device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

        # 初始化
        self.reset_parameters()

    # def reset_parameters(self):
    #     """初始化TT核心和偏置"""
    #     for i, core in enumerate(self.tt_cores):
    #         # 使用正交初始化以获得更好的数值稳定性
    #         if i < self.q:  # 输出核心
    #             # 对应U矩阵的部分，使用正交初始化
    #             with torch.no_grad():
    #                 if core.size(0) == 1:  # 第一个核心
    #                     nn.init.orthogonal_(core.squeeze(0))
    #                 else:
    #                     # 对于非第一个核心，使用Kaiming初始化
    #                     nn.init.kaiming_uniform_(core, a=math.sqrt(5))
    #         else:  # 输入核心
    #             # 对应V矩阵的部分，使用正交初始化
    #             with torch.no_grad():
    #                 if core.size(-1) == 1:  # 最后一个核心
    #                     nn.init.orthogonal_(core.squeeze(-1))
    #                 else:
    #                     nn.init.kaiming_uniform_(core, a=math.sqrt(5))
    #
    #     if self.bias is not None:
    #         fan_in = self.in_features
    #         bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         nn.init.uniform_(self.bias, -bound, bound)
    def reset_parameters(self):
        """初始化TT核心和偏置"""
        for i, core in enumerate(self.tt_cores):
            if i < self.q:  # 输出核心
                with torch.no_grad():
                    if core.size(0) == 1:  # 第一个核心
                        nn.init.kaiming_uniform_(core.squeeze(0), a=math.sqrt(0))
                    else:
                        nn.init.kaiming_uniform_(core, a=math.sqrt(0))
            else:  # 输入核心
                with torch.no_grad():
                    if core.size(-1) == 1:  # 最后一个核心
                        nn.init.kaiming_uniform_(core.squeeze(-1), a=math.sqrt(0))
                    else:
                        nn.init.kaiming_uniform_(core, a=math.sqrt(0))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.tt_cores[0].size(2))
            nn.init.uniform_(self.bias, -bound, bound)

    def reconstruct_full_weight(self):
        """
        精确模拟TT-SVD分解逆过程的重建函数。
        按照分解时的相反顺序重建完整权重。
        """
        cores = self.tt_cores
        k = len(cores)
        dims = self.out_dims + self.in_dims

        # 步骤1: 处理最后一个core (形状: r_{k-1}, d_k, 1)
        # 在TT-SVD中，最后一个core是 S_{k-1} * Vh_{k-1} 的reshape
        # print(f"最后一个核心维度为{cores[-1].shape}")
        # 重建时，我们从这里开始
        current_matrix = cores[-1].squeeze(-1)  # 形状: (r_{k-1}, d_k)
        # print(f"最后一个核心reshape后的温度为{current_matrix.shape}")

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
            # print(f"当前要合并的核心维度为{core_i.shape},该核心reshape后的维度为{U_matrix.shape},之前合并的核心维度为{current_matrix.shape}")
            # 当前矩阵是上一轮的 diag(S_i) @ Vh_i
            # 我们需要计算: U_matrix @ current_matrix
            # 这相当于重建 SVD 前的矩阵: U @ (diag(S) @ Vh) = U @ diag(S) @ Vh
            current_matrix = torch.matmul(U_matrix, current_matrix)  # 形状: (r_prev * d_i, 剩余维度)
            # 重塑以便下一步迭代
            if i > 0:  # 如果不是第一个核心
                current_matrix = current_matrix.reshape(r_prev, -1)  # 形状: (r_prev, d_i * 剩余维度)

        # 4. 重塑为完整形状
        dims = self.out_dims + self.in_dims
        full_tensor = current_matrix.reshape(*dims)

        # 转换为权重矩阵格式
        in_size = math.prod(self.in_dims)
        out_size = math.prod(self.out_dims)
        weight_matrix = full_tensor.reshape(out_size, in_size)

        return weight_matrix.contiguous()


    # 这个代码有问题，二核心的时候没问题，多核心的时候重构后的前向传播行为和分解时候不同
    # def forward(self, x):
    #     """
    #     支持输入:
    #     - (B, D)
    #     - (B, N, D)  ← ViT
    #
    #     仅对最后一个维度做 TT 线性变换
    #     """
    #     original_shape = x.shape
    #
    #     # === 统一 reshape 成 (B*, D) ===
    #     if x.dim() == 3:
    #         B, N, D = x.shape
    #         x = x.reshape(B * N, D)
    #     elif x.dim() == 2:
    #         B, D = x.shape
    #     else:
    #         raise ValueError(f"不支持的输入维度: {x.shape}")
    #
    #     batch_size = x.size(0)
    #
    #     # === TT forward（和你原来一样） ===
    #     x_reshaped = x.view(batch_size, *self.in_dims)
    #
    #     state = x_reshaped.unsqueeze(-1)  # (B*, i1,...,ip,1)
    #
    #     # === 输入核心收缩 ===
    #     for i in range(self.k - 1, self.q - 1, -1):
    #         core = self.tt_cores[i]  # (r_prev, d_i, r_next)
    #
    #         if state.size(-2) != core.size(1):
    #             raise ValueError(
    #                 f"维度不匹配: state[...,-2]={state.size(-2)}, core d_i={core.size(1)}"
    #             )
    #
    #         state = torch.einsum('...ir, kir -> ...k', state, core)
    #
    #     # === 输出核心扩展 ===
    #     for i in range(self.q - 1, -1, -1):
    #         core = self.tt_cores[i]  # (r_out, d_o, r_in)
    #
    #         if state.size(-1) != core.size(2):
    #             raise ValueError(
    #                 f"秩不匹配: state[...,-1]={state.size(-1)}, core r_in={core.size(2)}"
    #             )
    #
    #         state = torch.einsum('...r, sdr -> ...ds', state, core)
    #
    #     # === reshape 输出 ===
    #     if state.size(-1) == 1:
    #         state = state.squeeze(-1)
    #
    #     state = state.reshape(batch_size, -1)
    #
    #     # === bias ===
    #     if self.bias_flag and self.bias is not None:
    #         state = state + self.bias
    #
    #     # === 恢复 ViT 形状 ===
    #     if len(original_shape) == 3:
    #         state = state.view(B, N, -1)
    #
    #     return state

    def forward(self, x):
        """
        支持输入:
        - (B, D)
        - (B, N, D)  ← ViT
        仅对最后一个维度做 TT 线性变换
        """
        original_shape = x.shape
        # === 统一 reshape 成 (B*, D) ===
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        elif x.dim() == 2:
            B, D = x.shape
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, *self.in_dims)
        state = x_reshaped.unsqueeze(-1)  # (B, i1, ..., ip, 1)

        # 收缩输入核心（反向）
        for i in range(self.k - 1, self.q - 1, -1):
            core = self.tt_cores[i]
            r_prev, d_i, r_next = core.shape
            if state.size(-2) != d_i:
                raise ValueError(...)
            state = torch.einsum('...ir, kir -> ...k', state, core)

        # 扩展输出核心（反向）
        for i in range(self.q - 1, -1, -1):
            core = self.tt_cores[i]
            r_out, d_o, r_in = core.shape
            if state.size(-1) != r_in:
                raise ValueError(...)
            state = torch.einsum('...r,sdr->...ds', state, core)

        # 此时 state 形状: (B, out_{q-1}, out_{q-2}, ..., out_0, rank)
        # 反转输出维度顺序，使其与 out_dims 一致: (B, out_0, out_1, ..., out_{q-1}, rank)
        if self.q > 1:
            # 维度索引说明：
            # 0: batch
            # 1..q: 输出维度（当前顺序是反向的）
            # -1: 秩维度（通常为1）
            perm = [0] + list(range(state.dim() - 2, 0, -1)) + [state.dim() - 1]
            state = state.permute(*perm)

        # 移除秩维度（如果为1）
        if state.size(-1) == 1:
            state = state.squeeze(-1)

        # 展平输出
        state = state.reshape(batch_size, -1)

        if self.bias_flag and self.bias is not None:
            state = state + self.bias
        # === 恢复 ViT 形状 ===
        if len(original_shape) == 3:
            state = state.view(B, N, -1)
        return state

    def frobenius_loss(self):
        """计算权重矩阵的F-范数平方"""
        # 多核心情况：重建完整矩阵计算(reshap成多阶进行正则化效果一样)
        W = self.reconstruct_full_weight()
        return torch.sum(W ** 2)

    # def frobenius_loss(self):
    #     """计算权重矩阵的F-范数平方"""
    #     # 多核心情况：重建完整矩阵计算(reshap成多阶进行正则化效果一样)
    #     W = self.reconstruct_full_weight()
    #     dims = self.out_dims + self.in_dims
    #     full_tensor = W.reshape(*dims)
    #     return torch.sum(full_tensor ** 2)
    # def frobenius_loss(self):
    #     """计算所有权重核心的二范数平方之和"""
    #     total_loss = 0.0

    #     # 遍历所有TT核心（三阶张量）
    #     for i, core in enumerate(self.tt_cores):
    #         # 计算每个核心的二范数平方
    #         core_norm_sq = torch.sum(core ** 2)
    #         total_loss += core_norm_sq

    #     return total_loss

    def __repr__(self):
        return (f"TTLinear({self.in_features}, {self.out_features}, "
                f"in_dims={self.in_dims}, out_dims={self.out_dims}, "
                f"cores={self.k}, bias={self.bias_flag})")


# ---------- 分解与恢复函数 ----------
def Decom_TTLinear(linear_model,tt_rank_rate=None,core=2):
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
    if core==2:
        # # 二核心
        in_dims = [in_features]
        out_dims = [out_features]
    elif core==3:
        if in_features > out_features:
            in_dims = Factorize(in_features)
            out_dims = [out_features]
        else:
            in_dims = [in_features]
            out_dims = Factorize(out_features)
    elif core==4:
        # 默认分解：输入分为两个因子，输出分为1个因子（和原版兼容） 四核
        in_dims = Factorize(in_features)
        out_dims = Factorize(out_features)
    else:
        raise ValueError("分解核心数太多会影响性能")

    assert math.prod(in_dims) == in_features, "in_dims 乘积必须等于 in_features"
    assert math.prod(out_dims) == out_features, "out_dims 乘积必须等于 out_features"

    # 如果用户给了 rank_rate 而没给 tt_ranks，我们将在 TTLinear 中推断
    tt_linear = TTLinear(in_features, out_features,
                         tt_rank_rate=tt_rank_rate,
                         bias=bias, device=device, dtype=dtype,core=core)
    tt_linear = tt_linear.to(device=device, dtype=dtype)

    # 取得原始权重并 reshape 为 (d1, d2, ..., dk)
    W_original = linear_model.weight.data.clone()  # (out_features, in_features)
    all_dims = out_dims + in_dims
    W_reshaped = W_original.reshape(*all_dims).to(device=device, dtype=dtype)

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




def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# --------------------------------------LOW_RANK VIT---------------------------------------------------------------------------
class LOW_RANK_FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0., ratio_LR=1.0,core=2):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.ln = nn.LayerNorm(dim)
        if ratio_LR >= 1.0:
            self.fc1 = nn.Linear(dim, hidden_dim)
        else:
            self.fc1 = TTLinear(dim, hidden_dim, ratio_LR, bias=True,core=core)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        if ratio_LR >= 1.0:
            self.fc2 = nn.Linear(hidden_dim, dim)
        else:
            self.fc2 = TTLinear(hidden_dim, dim, ratio_LR, bias=True,core=core)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.ln,
            self.fc1,
            self.gelu,
            self.dropout1,
            self.fc2,
            self.dropout2
        )


    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            print("Recovering Low-Rank FeedForward Layer...")
            # 恢复并更新fc1
            recovered_fc1 = Recover_TTLinear(self.fc1)
            # 恢复并更新fc2
            recovered_fc2 = Recover_TTLinear(self.fc2)

            # 重要：更新整个网络序列
            self.net = nn.Sequential(
                self.ln,
                recovered_fc1,
                self.gelu,
                self.dropout1,
                recovered_fc2,
                self.dropout2
            )
            # 更新引用
            self.fc1 = recovered_fc1
            self.fc2 = recovered_fc2

    # 这个是按照传递的分解比例进行分解
    def decom(self, ratio_LR,core=2):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank FeedForward Layer...")
            # 分解并更新fc1
            decomposed_fc1 = Decom_TTLinear(self.fc1, ratio_LR,core=core)
            # 分解并更新fc2
            decomposed_fc2 = Decom_TTLinear(self.fc2, ratio_LR,core=core)

            # 更新整个网络序列
            self.net = nn.Sequential(
                self.ln,
                decomposed_fc1,
                self.gelu,
                self.dropout1,
                decomposed_fc2,
                self.dropout2
            )
            # 更新引用
            self.fc1 = decomposed_fc1
            self.fc2 = decomposed_fc2

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR < 1.0:
            loss += self.fc1.frobenius_loss()
            loss += self.fc2.frobenius_loss()
        return loss

    def forward(self, x):
        return self.net(x)


class LOW_RANK_Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., ratio_LR=1.0,core=2):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # Layer normalization
        self.norm = nn.LayerNorm(dim)

        # 低秩化的QKV投影
        if ratio_LR >= 1.0:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            # 将QKV投影分解为低秩形式
            self.to_qkv = TTLinear(dim, inner_dim * 3, ratio_LR, bias=False,core=core)

        # Softmax和Dropout
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 低秩化的输出投影
        if project_out:
            if ratio_LR >= 1.0:
                self.out_linear = nn.Linear(inner_dim, dim)
                self.to_out = nn.Sequential(
                    self.out_linear,
                    self.dropout,
                )
            else:
                # 创建低秩的线性层
                self.out_linear = TTLinear(inner_dim, dim, ratio_LR, bias=True,core=core)
                self.to_out = nn.Sequential(
                    self.out_linear,
                    self.dropout,
                )
        else:
            self.to_out = nn.Identity()

        self.inner_dim = inner_dim
        self.project_out = project_out
        self.dropout_rate = dropout

    def recover(self):
        """恢复为完整秩的线性层"""
        if self.ratio_LR >= 1.0:
            return

        print("Recovering Low-Rank Attention Layer...")

        # 恢复QKV投影
        self.to_qkv = Recover_TTLinear(self.to_qkv)

        # 恢复输出投影
        if self.project_out:
            # 恢复线性层
            recovered_linear = Recover_TTLinear(self.out_linear)

            # 重要：更新to_out序列中的层
            self.to_out = nn.Sequential(
                recovered_linear,
                self.dropout,
            )
            # 更新引用
            self.out_linear = recovered_linear

    def decom(self, ratio_LR,core=2):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank Attention Layer...")
            # 分解QKV投影
            decomposed_qkv = Decom_TTLinear(self.to_qkv, ratio_LR,core=core)

            # 更新QKV投影
            self.to_qkv = decomposed_qkv

            # 分解输出投影
            if self.project_out:
                decomposed_out = Decom_TTLinear(self.out_linear, ratio_LR,core=core)

                # 更新to_out序列
                self.to_out = nn.Sequential(
                    decomposed_out,
                    self.dropout,
                )
                # 更新引用
                self.out_linear = decomposed_out

    def frobenius_loss(self):
        """计算低秩近似与原始权重的Frobenius损失"""
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR < 1.0:
            loss += self.to_qkv.frobenius_loss()
            # 输出投影的损失（如果存在）
            if self.project_out:
                loss += self.out_linear.frobenius_loss()

        return loss

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LOW_RANK_Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., ratio_LR=1.0,core=2):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for i in range(depth):
            self.layers.append(ModuleList([
                LOW_RANK_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, ratio_LR=ratio_LR,core=core),
                LOW_RANK_FeedForward(dim, mlp_dim, dropout=dropout, ratio_LR=ratio_LR,core=core)
            ]))

    def decom(self, ratio_LR,core=2):
        if ratio_LR >= 1.0:
            return
        print("Decomposing Low-Rank Transformer Layers...")
        for attn, ff in self.layers:
            attn.decom(ratio_LR,core=core)
            ff.decom(ratio_LR,core=core)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        print("Recovering Low-Rank Transformer Layers...")
        for attn, ff in self.layers:
            attn.recover()
            ff.recover()

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR >= 1.0:
            return loss
        for attn, ff in self.layers:
            loss += attn.frobenius_loss()
            loss += ff.frobenius_loss()
        return loss

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class LOW_RANK_ViTBase_TT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            ratio_LR=1.0,
            core = 2
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0
        self.core = core
        self.pool = pool

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height,
                p2=patch_width
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches + num_cls_tokens, dim)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = LOW_RANK_Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, ratio_LR=ratio_LR,core=core
        )

        # self.to_latent = nn.Identity()
        self.to_latent = nn.Linear(dim, 512)

    def forward(self, img):
        b = img.size(0)

        x = self.to_patch_embedding(img)

        if self.cls_token.numel() > 0:
            cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x


class LOW_RANK_ViT_TT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            ratio_LR=1.0,
            core = 2
    ):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.core = core
        self.base = LOW_RANK_ViTBase_TT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            ratio_LR=ratio_LR,
            core = core
        )

        self.head = (
            nn.Linear(512, num_classes)
            if num_classes > 0 else nn.Identity()
        )

    def decom_larger_model(self, ratio_LR,core=2):
        self.base.transformer.decom(ratio_LR,core=core)

    def recover_larger_model(self):
        self.base.transformer.recover()

    def frobenius_decay(self):
        return self.base.transformer.frobenius_loss()

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x



class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)





# 低秩化Transformer  2
class LOW_RANK_Transformer_Select(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., ratio_LR=1.0, decom_start_layer=-1,core=2):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        self.core = core
        # 开始分解的层数 -1表示不分解
        self.decom_start_layer = decom_start_layer
        for i in range(depth):
            # 完全不分解
            if decom_start_layer == -1:
                self.layers.append(ModuleList([
                    Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]))
            else:
                if i + 1 < decom_start_layer:
                    ML = ModuleList([
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ])
                else:
                    ML = ModuleList([
                        LOW_RANK_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, ratio_LR=ratio_LR),
                        LOW_RANK_FeedForward(dim, mlp_dim, dropout=dropout, ratio_LR=ratio_LR)
                    ])
                self.layers.append(ML)

    def decom(self, ratio_LR,core):
        if ratio_LR >= 1.0:
            return
        print("Decomposing Low-Rank Transformer Layers...")
        i = 0
        for attn, ff in self.layers:
            # 不到分解层不分解
            i += 1
            if i < self.decom_start_layer:
                continue
            attn.decom(ratio_LR,core)
            ff.decom(ratio_LR,core)

    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        i = 0
        print("Recovering Low-Rank Transformer Layers...")
        for attn, ff in self.layers:
            i += 1
            if i < self.decom_start_layer:
                continue
            attn.recover()
            ff.recover()

    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR >= 1.0:
            return loss
        i = 0
        for attn, ff in self.layers:
            i += 1
            if i < self.decom_start_layer:
                continue
            loss += attn.frobenius_loss()
            loss += ff.frobenius_loss()
        return loss

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class LOW_RANK_ViTBase_Select_TT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            ratio_LR=1.0,
            decom_start_layer=-1,
            core = 2
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

        self.pool = pool
        self.core = core
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height,
                p2=patch_width
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(num_patches + num_cls_tokens, dim)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = LOW_RANK_Transformer_Select(
            dim, depth, heads, dim_head, mlp_dim, dropout, ratio_LR=ratio_LR, decom_start_layer=decom_start_layer,core=core
        )

        # self.to_latent = nn.Identity()
        self.to_latent = nn.Linear(dim, 512)

    def forward(self, img):
        b = img.size(0)

        x = self.to_patch_embedding(img)

        if self.cls_token.numel() > 0:
            cls_tokens = repeat(self.cls_token, 'n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)

        return x


class LOW_RANK_ViT_Select_TT(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads,
            mlp_dim,
            pool='cls',
            channels=3,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            ratio_LR=1.0,
            decom_start_layer=-1,
            core = 2
    ):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.core = core
        self.base = LOW_RANK_ViTBase_Select_TT(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            pool=pool,
            channels=channels,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            ratio_LR=ratio_LR,
            decom_start_layer=decom_start_layer,
            core = core
        )

        self.head = (
            nn.Linear(512, num_classes)
            if num_classes > 0 else nn.Identity()
        )

    def decom_larger_model(self, ratio_LR):
        self.base.transformer.decom(ratio_LR)

    def recover_larger_model(self):
        self.base.transformer.recover()

    def frobenius_decay(self):
        return self.base.transformer.frobenius_loss()

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x



if __name__ == '__main__':
    # 标准VIT设置 2核心
    model = LOW_RANK_ViT_TT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dim_head=64,  # 768/12=64，与 heads 匹配
        dropout=0.2,
        emb_dropout=0.2,
        pool='cls',
        channels=3,
        ratio_LR=0.1,
        core =4
    )
    # model = LOW_RANK_ViT_Select(
    #     image_size=32,
    #     patch_size=4,  # 小图像用小patch
    #     num_classes=10,
    #     dim=384,  # Base维度
    #     depth=4,  # 12层
    #     heads=6,  # 12个头
    #     mlp_dim=1536,  # dim×4
    #     dim_head=64,
    #     dropout=0.3,  # 更高的dropout防止过拟合
    #     emb_dropout=0.3,
    #     pool='cls',
    #     channels=3,
    #     ratio_LR=0.5  # 低秩比例
    # )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"Patch数量: {(32 // 4) * (32 // 4)} = {32 // 4}×{32 // 4}")

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
    print(parameter_count_table(model, max_depth=5))
    for param in model.parameters():
        print(param.shape)
    model.recover_larger_model()

    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
    print("\n模型结构:", model)
    print(output)
    print(parameter_count_table(model, max_depth=5))
    for param in model.parameters():
        print(param.shape)
    model.decom_larger_model(0.1,core=4)

    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")
    print("\n模型结构:", model)
    print("model f_loss", model.frobenius_decay())
    print(output)
    print(parameter_count_table(model, max_depth=5))
    for param in model.parameters():
        print(param.shape)
    # with torch.no_grad():  # 不计算梯度
    #     base_output = model.base(random_input)

    # print(f"\nbase输出数据形状: {base_output.shape}")
    # print(f"bese输出数据范围: [{base_output.min():.3f}, {base_output.max():.3f}]")
    # print(f"输出数据均值: {base_output.mean():.3f}, 标准差: {base_output.std():.3f}")