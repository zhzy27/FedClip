import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from utils.flops_utils import estimate_forward_flops


def _finite_denominator(loss_a, loss_b, eps=1e-8):
    return torch.clamp(loss_a + loss_b, min=eps)


def _sanitize_gradients_(module, tag):
    fixed = False
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
            fixed = True
            print(f"⚠️ FedKD {tag}.{name} 梯度出现 NaN/Inf，已置零异常梯度。")
    return fixed


def _sanitize_parameters_(module, tag):
    fixed = False
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not torch.isfinite(param).all():
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=0.0, neginf=0.0)
                fixed = True
                print(f"⚠️ FedKD {tag}.{name} 参数出现 NaN/Inf，已置零异常参数。")
    return fixed


class clientKD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.mentee_learning_rate = args.mentee_learning_rate
        self.energy = args.T_start

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            W_h = nn.Linear(args.feature_dim, args.feature_dim, bias=False).to(self.device)
            save_item(W_h, self.role, 'W_h', self.save_folder_name)
            global_model = load_item('Server', 'global_model', self.save_folder_name)
            save_item(global_model, self.role, 'global_model', self.save_folder_name)

        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()

    def estimate_local_train_flops(self):
        trainloader = self.load_train_data()
        try:
            x, _ = next(iter(trainloader))
        except StopIteration:
            return {
                "client_id": self.id,
                "train_samples": self.train_samples,
                "local_epochs": self.local_epochs,
                "forward_flops_per_sample": 0.0,
                "forward_flops_per_epoch": 0.0,
                "local_train_flops": 0.0,
                "module_flops": {},
                "note": "empty_trainloader",
            }

        if isinstance(x, (list, tuple)):
            model_input = x[0].to(self.device)
            batch_size = x[0].shape[0]
        else:
            model_input = x.to(self.device)
            batch_size = x.shape[0]

        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)
        module_flops = {}
        total_forward_flops = 0

        if model is not None:
            model = model.to(self.device)
            model_flops, model_module_flops = estimate_forward_flops(model, model_input, self.device)
            total_forward_flops += model_flops
            module_flops["local_model"] = model_module_flops
        if global_model is not None:
            global_model = global_model.to(self.device)
            global_flops, global_module_flops = estimate_forward_flops(global_model, model_input, self.device)
            total_forward_flops += global_flops
            module_flops["global_model"] = global_module_flops
        if W_h is not None and global_model is not None and hasattr(global_model, "base"):
            W_h = W_h.to(self.device)
            was_training = global_model.training
            global_model.eval()
            with torch.no_grad():
                rep_g = global_model.base(model_input)
            global_model.train(was_training)
            wh_flops, wh_module_flops = estimate_forward_flops(W_h, rep_g, self.device)
            total_forward_flops += wh_flops
            module_flops["W_h"] = wh_module_flops

        forward_flops_per_sample = total_forward_flops / max(1, batch_size)
        forward_flops_per_epoch = forward_flops_per_sample * self.train_samples
        train_multiplier = getattr(self.args, "local_flops_train_multiplier", 3.0)
        local_train_flops = forward_flops_per_epoch * self.local_epochs * train_multiplier

        return {
            "client_id": self.id,
            "train_samples": int(self.train_samples),
            "local_epochs": int(self.local_epochs),
            "forward_flops_per_sample": float(forward_flops_per_sample),
            "forward_flops_per_epoch": float(forward_flops_per_epoch),
            "local_train_flops": float(local_train_flops),
            "module_flops": module_flops,
            "note": "fedkd_local_model_plus_global_model_plus_projection_x_train_multiplier",
        }


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.mentee_learning_rate)
        optimizer_W = torch.optim.SGD(W_h.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        _sanitize_parameters_(model, f"Client_{self.id}.model")
        _sanitize_parameters_(global_model, f"Client_{self.id}.global_model")
        _sanitize_parameters_(W_h, f"Client_{self.id}.W_h")
        model.train()
        global_model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                rep_g = global_model.base(x)
                output = model.head(rep)
                output_g = global_model.head(rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                denom = _finite_denominator(CE_loss, CE_loss_g)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / denom
                L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / denom
                L_h = self.MSE(rep, W_h(rep_g)) / denom
                L_h_g = self.MSE(rep, W_h(rep_g)) / denom

                loss = CE_loss + L_d + L_h
                loss_g = CE_loss_g + L_d_g + L_h_g

                if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                    print(f"⚠️ FedKD Client_{self.id} 第 {step} 轮第 {i} 个 batch loss 非有限，跳过该 batch。")
                    optimizer.zero_grad()
                    optimizer_g.zero_grad()
                    optimizer_W.zero_grad()
                    _sanitize_parameters_(model, f"Client_{self.id}.model")
                    _sanitize_parameters_(global_model, f"Client_{self.id}.global_model")
                    _sanitize_parameters_(W_h, f"Client_{self.id}.W_h")
                    continue

                optimizer.zero_grad()
                optimizer_g.zero_grad()
                optimizer_W.zero_grad()
                (loss + loss_g).backward()
                _sanitize_gradients_(model, f"Client_{self.id}.model")
                _sanitize_gradients_(global_model, f"Client_{self.id}.global_model")
                _sanitize_gradients_(W_h, f"Client_{self.id}.W_h")
                # prevent divergency on specifical tasks
                if 'Cifar10' in self.args.dataset:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(W_h.parameters(), 10)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    torch.nn.utils.clip_grad_norm_(global_model.parameters(), 5)
                    torch.nn.utils.clip_grad_norm_(W_h.parameters(), 5)
                _sanitize_gradients_(model, f"Client_{self.id}.model")
                _sanitize_gradients_(global_model, f"Client_{self.id}.global_model")
                _sanitize_gradients_(W_h, f"Client_{self.id}.W_h")
                optimizer.step()
                optimizer_g.step()
                optimizer_W.step()
                _sanitize_parameters_(model, f"Client_{self.id}.model")
                _sanitize_parameters_(global_model, f"Client_{self.id}.global_model")
                _sanitize_parameters_(W_h, f"Client_{self.id}.W_h")

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        save_item(W_h, self.role, 'W_h', self.save_folder_name)
        compressed_param = decomposition(global_model.named_parameters(), self.energy)
        save_item(compressed_param, self.role, 'compressed_param', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self):
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        compressed_param = load_item('Server', 'compressed_param', self.save_folder_name)
        param = recover(compressed_param)
        for name, old_param in global_model.named_parameters():
            if name in param:
                old_param.data = torch.tensor(param[name], device=self.device).data.clone()
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        W_h = load_item(self.role, 'W_h', self.save_folder_name)
        # model.to(self.device)
        model.eval()
        global_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = model.base(x)
                rep_g = global_model.base(x)
                output = model.head(rep)
                output_g = global_model.head(rep_g)

                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                denom = _finite_denominator(CE_loss, CE_loss_g)
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / denom
                L_h = self.MSE(rep, W_h(rep_g)) / denom

                loss = CE_loss + L_d + L_h
                train_num += y.shape[0]
                if torch.isfinite(loss):
                    losses += loss.item() * y.shape[0]
                else:
                    print(f"⚠️ FedKD Client_{self.id} train_metrics loss 非有限，本 batch 不计入训练损失。")

        return losses, train_num
            

# def recover(compressed_param):
#     for k in compressed_param.keys():
#         if len(compressed_param[k]) == 3:
#             # use np.matmul to support high-dimensional CNN param
#             compressed_param[k] = np.matmul(
#                 compressed_param[k][0] * compressed_param[k][1][..., None, :], 
#                     compressed_param[k][2])
#     return compressed_param

    
# def decomposition(param_iter, energy):
#     compressed_param = {}
#     for name, param in param_iter:
#         try:
#             param_cpu = param.detach().cpu().numpy()
#         except:
#             param_cpu = param
#         # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
#         if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
#             u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
#             # support high-dimensional CNN param
#             if len(u.shape)==4:
#                 u = np.transpose(u, (2, 3, 0, 1))
#                 sigma = np.transpose(sigma, (2, 0, 1))
#                 v = np.transpose(v, (2, 3, 0, 1))
#             threshold=0
#             if np.sum(np.square(sigma))==0:
#                 compressed_param_cpu=param_cpu
#             else:
#                 for singular_value_num in range(len(sigma)):
#                     if np.sum(np.square(sigma[:singular_value_num]))>energy*np.sum(np.square(sigma)):
#                         threshold=singular_value_num
#                         break
#                 u=u[:, :threshold]
#                 sigma=sigma[:threshold]
#                 v=v[:threshold, :]
#                 # support high-dimensional CNN param
#                 if len(u.shape)==4:
#                     u = np.transpose(u, (2, 3, 0, 1))
#                     sigma = np.transpose(sigma, (1, 2, 0))
#                     v = np.transpose(v, (2, 3, 0, 1))
#                 compressed_param_cpu=[u,sigma,v]
#         elif 'embeddings' not in name:
#             compressed_param_cpu=param_cpu

#         compressed_param[name] = compressed_param_cpu
        
#     return compressed_param

def _safe_svd(name, param_tensor):
    try:
        return torch.linalg.svd(param_tensor, full_matrices=False)
    except Exception as gpu_error:
        print(f"⚠️ FedKD SVD 在 {name} 上使用 {param_tensor.device} 失败，尝试 CPU SVD。错误: {gpu_error}")

    param_cpu = param_tensor.detach().cpu()
    cpu_svd_input = param_cpu.double() if param_cpu.is_floating_point() else param_cpu
    try:
        u, sigma, v = torch.linalg.svd(cpu_svd_input, full_matrices=False)
        return u.to(param_cpu.dtype), sigma.to(param_cpu.dtype), v.to(param_cpu.dtype)
    except Exception as cpu_error:
        print(f"⚠️ FedKD SVD 在 {name} 上使用 CPU 仍失败，尝试 NumPy SVD。错误: {cpu_error}")

    try:
        param_np = param_cpu.numpy()
        svd_np = param_np.astype(np.float64, copy=False) if np.issubdtype(param_np.dtype, np.floating) else param_np
        u, sigma, v = np.linalg.svd(svd_np, full_matrices=False)
        return (
            torch.from_numpy(u).to(param_cpu.dtype),
            torch.from_numpy(sigma).to(param_cpu.dtype),
            torch.from_numpy(v).to(param_cpu.dtype),
        )
    except Exception as numpy_error:
        print(f"⚠️ FedKD SVD 在 {name} 上彻底失败，该参数本轮不压缩。错误: {numpy_error}")
        return None


def decomposition(param_items, energy):
    compressed_param = {}
    
    for name, param in param_items:
        # 1. 统一设备处理：优雅且安全地确保参数在 GPU 上
        if isinstance(param, torch.Tensor):
            param_gpu = param.detach()
            if not param_gpu.is_cuda and torch.cuda.is_available():
                param_gpu = param_gpu.cuda()
        else:
            param_gpu = torch.tensor(param, device='cuda' if torch.cuda.is_available() else 'cpu')

        # 2. 判断是否符合压缩条件 (跳过1维的偏置等)
        if param_gpu.shape[0] > 1 and len(param_gpu.shape) > 1 and 'embeddings' not in name:
            if not torch.isfinite(param_gpu).all():
                print(f"⚠️ FedKD 参数 {name} 出现 NaN/Inf，先做 nan_to_num 后再尝试 SVD。")
                param_gpu = torch.nan_to_num(param_gpu, nan=0.0, posinf=1e4, neginf=-1e4)

            svd_result = _safe_svd(name, param_gpu)
            if svd_result is None:
                compressed_param[name] = param_gpu.detach().cpu().numpy()
                continue

            u, sigma, v = svd_result
            
            # CNN 4维参数转置
            if len(u.shape) == 4:
                u = u.permute(2, 3, 0, 1)      # (k_h, k_w, out_c, k)
                sigma = sigma.permute(2, 0, 1) # (k_h, k_w, k)
                v = v.permute(2, 3, 0, 1)      # (k_h, k_w, k, in_c)
                
            # 计算能量平方
            sigma_sq = torch.square(sigma)
            total_energy = torch.sum(sigma_sq)
            
            if total_energy == 0:
                compressed_param_cpu = param_gpu.cpu().numpy()
            else:
                target_energy = energy * total_energy
                
                # 【修复 AI 的 Bug】：针对高维 sigma 先压缩为 1D，再做 cumsum
                # 将除第0维以外的维度拉平并求和，确保送入 searchsorted 的是 1D 张量
                sigma_sq_1d = sigma_sq.view(sigma_sq.shape[0], -1).sum(dim=1) 
                
                cum_energy = torch.cumsum(sigma_sq_1d, dim=0)
                # 直接在 searchsorted 结果后 + 1，这是理论上最完美的映射
                threshold = torch.searchsorted(cum_energy, target_energy).item() + 1
                threshold = min(threshold, len(sigma))
                # 截断矩阵
                u = u[:, :threshold]
                sigma = sigma[:threshold]
                v = v[:threshold, :]
                
                # CNN 4维参数恢复维度
                if len(u.shape) == 4:
                    u = u.permute(2, 3, 0, 1)      
                    sigma = sigma.permute(1, 2, 0) 
                    v = v.permute(2, 3, 0, 1)      
                
                # 仅将极小的数据传回 CPU 用于网络传输
                compressed_param_cpu = [u.cpu().numpy(), sigma.cpu().numpy(), v.cpu().numpy()]
        else:
            compressed_param_cpu = param_gpu.cpu().numpy()
            
        compressed_param[name] = compressed_param_cpu
        
    return compressed_param


def recover(compressed_param):
    for k in compressed_param.keys():
        if isinstance(compressed_param[k], list) and len(compressed_param[k]) == 3:
            u, sigma, v = compressed_param[k]
            
            # 直接使用广播乘法重构矩阵，这一步输出的 recon 就已经是正确的原始形状了！
            recon = np.matmul(u * sigma[..., None, :], v)
            
            # 【千万不要在这里再做 np.transpose！！！】
            
            compressed_param[k] = recon
    return compressed_param
