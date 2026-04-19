import torch
import torch.nn as nn
import psutil
import time
import math
import torch
import os
import torch.nn.functional as F
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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
# -------------------------------
# 示例模型（可替换成你的 CNN/ViT）
# -------------------------------
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


# -------------------------------
# CPU load 测试函数
# -------------------------------
def measure_cpu_load(model, batch_size=32, input_shape=(3, 32, 32), repeats=10):
    """
    测量模型在 CPU 上推理的平均占用率
    """
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    # psutil 进程
    proc = psutil.Process()
    
    # 构造输入
    x = torch.randn(batch_size, *input_shape)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
    
    # 测量
    cpu_usages = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.time()
            _ = model(x)
            end = time.time()
            
            # interval=0.1 测量本次 CPU 使用率
            cpu_percent = proc.cpu_percent(interval=0.1) / psutil.cpu_count()
            cpu_usages.append(cpu_percent)
    
    avg_cpu = sum(cpu_usages) / len(cpu_usages)
    return avg_cpu

def measure_latency(model, batch_size=16, repeats=100, warmup=20):
    device = torch.device("cpu")
    model.to(device).eval()

    x = torch.randn(batch_size, 3, 32, 32, device=device)

    # -------- 预热 --------
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    # -------- 正式计时 --------
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)
    end = time.perf_counter()

    total_time = end - start
    avg_latency = total_time / repeats          # 单次前向时间（秒）
    throughput = batch_size / avg_latency       # samples/s

    return avg_latency * 1000, throughput       # ms, samples/s


# -------------------------------
# 测试不同 Rank_ratio 模型
# -------------------------------
if __name__ == "__main__":
    rank_ratios = [1.0, 0.5, 0.35, 0.25, 0.15, 0.05, 0.01]
    batch_size = 16

    print(f"{'Rank':<8}{'Latency(ms)':<15}{'Throughput(samples/s)':<25}")
    print("-" * 45)

    for r in rank_ratios:
        model = Hyper_CNN(in_features=3, num_classes=100, n_kernels=16, ratio_LR=r)

        latency, th = measure_latency(model, batch_size=batch_size)

        print(f"{r:<8.2f}{latency:<15.3f}{th:<25.2f}")