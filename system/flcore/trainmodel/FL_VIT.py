import torch
from torch import nn,Tensor
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import math
import torch.nn.functional as F



# 分解后的全连接层 (二维矩阵存储)
class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_rate, bias=True):
        super(FactorizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #中间rank值
        self.rank = max(1, round(rank_rate * min(in_features, out_features)))

        # 二维矩阵参数
        #第一个全连接层的参数（维度为 r*in）
        self.weight_v = nn.Parameter(torch.Tensor(self.rank, in_features))
        #第二个全连接层的参数 (维度为 out*r)
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
    U_weight = U[:, :rank] @ torch.diag(S_sqrt)  #shape out*r
    V_weight = torch.diag(S_sqrt) @ Vh[:rank, :]  #shape r*in

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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class ViTBase(nn.Module):
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
        emb_dropout=0.
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

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

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.to_latent = nn.Identity()
        

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


class ViT(nn.Module):
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
        emb_dropout=0.
    ):
        super().__init__()

        self.base = ViTBase(
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
        )

        self.head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.Linear(512, num_classes)
            if num_classes > 0 else nn.Identity()
        )

    def forward(self, x):
        x = self.base(x)
        x = self.head(x)
        return x


# --------------------------------------LOW_RANK VIT---------------------------------------------------------------------------
class LOW_RANK_FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.,ratio_LR=1.0):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.ln = nn.LayerNorm(dim)
        if ratio_LR>=1.0:
            self.fc1 = nn.Linear(dim, hidden_dim)
        else:
            self.fc1 =FactorizedLinear(dim, hidden_dim, ratio_LR,bias=True)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        if ratio_LR>=1.0:
            self.fc2 = nn.Linear(hidden_dim, dim)
        else:
            self.fc2 = FactorizedLinear(hidden_dim, dim, ratio_LR,bias=True)
        self.dropout2 = nn.Dropout(dropout)
        
        
        self.net = nn.Sequential(
            self.ln,
            self.fc1,
            self.gelu,
            self.dropout1,
            self.fc2,
            self.dropout2
        )
    # def recover(self):
    #     if self.ratio_LR>=1.0:
    #         return
    #     else:
    #         print("Recovering Low-Rank FeedForward Layer...")
    #         self.fc1=Recover_LINEAR(self.fc1)
    #         self.fc2=Recover_LINEAR(self.fc2)
    def recover(self):
        if self.ratio_LR >= 1.0:
            return
        else:
            print("Recovering Low-Rank FeedForward Layer...")
            # 恢复并更新fc1
            recovered_fc1 = Recover_LINEAR(self.fc1)
            # 恢复并更新fc2
            recovered_fc2 = Recover_LINEAR(self.fc2)

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
    #这个是按照传递的分解比例进行分解
    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank FeedForward Layer...")
            # 分解并更新fc1
            decomposed_fc1 = Decom_LINEAR(self.fc1, ratio_LR)
            # 分解并更新fc2
            decomposed_fc2 = Decom_LINEAR(self.fc2, ratio_LR)
            
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
        if self.ratio_LR<1.0:
            loss+=self.fc1.frobenius_loss()
            loss+=self.fc2.frobenius_loss()
        return loss
    def forward(self, x):
        return self.net(x)



class LOW_RANK_Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., ratio_LR=1.0):
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
            self.to_qkv = FactorizedLinear(dim, inner_dim * 3, ratio_LR, bias=False)
        
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
                self.out_linear = FactorizedLinear(inner_dim, dim, ratio_LR, bias=True)
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
        self.to_qkv = Recover_LINEAR(self.to_qkv)

        # 恢复输出投影
        if self.project_out:
            # 恢复线性层
            recovered_linear = Recover_LINEAR(self.out_linear)

            # 重要：更新to_out序列中的层
            self.to_out = nn.Sequential(
                recovered_linear,
                self.dropout,
            )
            # 更新引用
            self.out_linear = recovered_linear


    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        else:
            print("Decomposing Low-Rank Attention Layer...")
            # 分解QKV投影
            decomposed_qkv = Decom_LINEAR(self.to_qkv, ratio_LR)
            
            # 更新QKV投影
            self.to_qkv = decomposed_qkv
            
            # 分解输出投影
            if self.project_out:
                decomposed_out = Decom_LINEAR(self.out_linear, ratio_LR)
                
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
            loss +=self.to_qkv.frobenius_loss()
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., ratio_LR=1.0):
        super().__init__()
        self.ratio_LR=ratio_LR
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        for i in range(depth):
            self.layers.append(ModuleList([
                LOW_RANK_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,ratio_LR=ratio_LR),
                LOW_RANK_FeedForward(dim, mlp_dim, dropout = dropout,ratio_LR=ratio_LR)
                ]))

    def decom(self, ratio_LR):
        if ratio_LR>=1.0:
            return
        print("Decomposing Low-Rank Transformer Layers...")
        for attn, ff in self.layers:
            attn.decom(ratio_LR)
            ff.decom(ratio_LR)
    def recover(self):
        if self.ratio_LR>=1.0:
            return
        print("Recovering Low-Rank Transformer Layers...")
        for attn, ff in self.layers:
            attn.recover()
            ff.recover()
    def frobenius_loss(self):
        device = next(self.parameters()).device
        loss = torch.tensor(0.0, device=device)
        if self.ratio_LR>=1.0:
            return loss
        for attn, ff in self.layers:
            loss+=attn.frobenius_loss()
            loss+=ff.frobenius_loss()
        return loss
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)



class LOW_RANK_ViTBase(nn.Module):
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
            ratio_LR=1.0
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

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
            dim, depth, heads, dim_head, mlp_dim, dropout, ratio_LR=ratio_LR
        )

        self.to_latent = nn.Identity()
    
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


class LOW_RANK_ViT(nn.Module):
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
            ratio_LR=1.0
    ):
        super().__init__()
        self.ratio_LR =ratio_LR
        self.base = LOW_RANK_ViTBase(
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
            ratio_LR=ratio_LR
        )

        self.head = nn.Sequential(
            nn.Linear(dim, 512),
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

# 低秩化Transformer  2 
class LOW_RANK_Transformer_Select(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., ratio_LR=1.0, decom_start_layer=-1):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
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

    def decom(self, ratio_LR):
        if ratio_LR >= 1.0:
            return
        print("Decomposing Low-Rank Transformer Layers...")
        i = 0
        for attn, ff in self.layers:
            # 不到分解层不分解
            i += 1
            if i < self.decom_start_layer:
                continue
            attn.decom(ratio_LR)
            ff.decom(ratio_LR)

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


class LOW_RANK_ViTBase_Select(nn.Module):
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
            decom_start_layer=-1
    ):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_cls_tokens = 1 if pool == 'cls' else 0

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

        self.transformer = LOW_RANK_Transformer_Select(
            dim, depth, heads, dim_head, mlp_dim, dropout, ratio_LR=ratio_LR, decom_start_layer=decom_start_layer
        )

        self.to_latent = nn.Identity()

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


class LOW_RANK_ViT_Select(nn.Module):
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
            decom_start_layer=-1
    ):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.base = LOW_RANK_ViTBase_Select(
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
            decom_start_layer = decom_start_layer
        )

        self.head = nn.Sequential(
            nn.Linear(dim, 512),
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





# class ViT(Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width

#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         num_cls_tokens = 1 if pool == 'cls' else 0

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.LayerNorm(patch_dim),
#             nn.Linear(patch_dim, dim),
#             nn.LayerNorm(dim),
#         )

#         self.cls_token = nn.Parameter(torch.randn(num_cls_tokens, dim))
#         self.pos_embedding = nn.Parameter(torch.randn(num_patches + num_cls_tokens, dim))

#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Linear(dim, num_classes) if num_classes > 0 else None

#     def forward(self, img):
#         batch = img.shape[0]
#         x = self.to_patch_embedding(img)

#         cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
#         x = torch.cat((cls_tokens, x), dim = 1)

#         seq = x.shape[1]

#         x = x + self.pos_embedding[:seq]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         if self.mlp_head is None:
#             return x

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
#         return self.mlp_head(x)

if __name__ == '__main__':
    # 参数数量：~22M（但patch数量不同）
    # Patch数量：32/4 = 8 → 8×8=64个patches
#     model = ViT(
#     image_size=32,
#     patch_size=4,       # 更小的patch适应小图像
#     num_classes=10,
#     dim=384,            # 保持Small维度
#     depth=12,           # 保持12层
#     heads=6,            # 保持6个头
#     mlp_dim=1536,       # dim×4
#     dim_head=64,        # 384/6 = 64
#     dropout=0.1,
#     emb_dropout=0.1,
#     pool='cls',
#     channels=3
# )
#     model = ViT(
#     image_size=32,
#     patch_size=4,       # 小图像用小patch
#     num_classes=10,
#     dim=768,            # Base维度
#     depth=12,           # 12层
#     heads=12,           # 12个头
#     mlp_dim=3072,       # dim×4
#     dim_head=64,        # 768/12 = 64
#     dropout=0.2,        # 更高的dropout防止过拟合
#     emb_dropout=0.2,
#     pool='cls',
#     channels=3
# )

# # 适用于CIFAR100的轻量级ViT模型  四层VIT
# model = ViT(
#     image_size=32,           # CIFAR100图像尺寸
#     patch_size=4,            # 4×4的patch，适合32×32的小图像
#     num_classes=100,         # CIFAR100有100个类别
#     dim=384,                 # 适当减小维度，降低计算量
#     depth=4,                 # 4层Transformer，如您所要求
#     heads=6,                 # 6个注意力头
#     mlp_dim=1536,            # MLP维度为dim×4
#     dim_head=64,             # 每个头的维度
#     dropout=0.3,             # 增加dropout防止CIFAR100过拟合
#     emb_dropout=0.3,         # 嵌入层dropout
#     pool='cls',              # 使用CLS token进行分类
#     channels=3               # RGB三通道
# )
#     model = LOW_RANK_ViT(
#     image_size=32,
#     patch_size=4,       # 小图像用小patch
#     num_classes=10,
#     dim=768,            # Base维度
#     depth=12,           # 12层
#     heads=12,           # 12个头
#     mlp_dim=3072,       # dim×4
#     dim_head=64,        # 768/12 = 64
#     dropout=0.2,        # 更高的dropout防止过拟合
#     emb_dropout=0.2,
#     pool='cls',
#     channels=3,
#     ratio_LR=0.5       # 低秩比例
# )
    
    model = LOW_RANK_ViT(
    image_size=32,
    patch_size=4,       # 小图像用小patch
    num_classes=10,
    dim=384,            # Base维度
    depth=4,           # 12层
    heads=6,           # 12个头
    mlp_dim=1536,       # dim×4
    dim_head=64,       
    dropout=0.3,        # 更高的dropout防止过拟合
    emb_dropout=0.3,
    pool='cls',
    channels=3,
    ratio_LR=0.5       # 低秩比例
)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"Patch数量: {(32//4) * (32//4)} = {32//4}×{32//4}")
    
    print("\n模型结构:", model)
    print("model f_loss",model.frobenius_decay()  )
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
    
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)
    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")   
    print("\n模型结构:", model)
    print(output)
    
    model.decom_larger_model(0.5)
    
    # 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        output = model(random_input)
    print(f"\n输出数据形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"输出数据均值: {output.mean():.3f}, 标准差: {output.std():.3f}")   
    print("\n模型结构:", model)
    print("model f_loss",model.frobenius_decay()  )
    print(output)
    # with torch.no_grad():  # 不计算梯度
    #     base_output = model.base(random_input)
    
    # print(f"\nbase输出数据形状: {base_output.shape}")
    # print(f"bese输出数据范围: [{base_output.min():.3f}, {base_output.max():.3f}]")
    # print(f"输出数据均值: {base_output.mean():.3f}, 标准差: {base_output.std():.3f}")   