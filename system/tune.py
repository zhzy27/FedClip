import os
import sys
import argparse

# ===================================================================
# 🚨 致命修正：必须在导入 torch 之前，提前截获 GPU ID 并设置环境变量！
# ===================================================================
opt_parser = argparse.ArgumentParser()
opt_parser.add_argument('--gpu', type=int, required=True, help="要使用的 GPU ID")
# 使用 parse_known_args 防止未知参数报错
opt_args, _ = opt_parser.parse_known_args()

# 提前锁定显卡！这时候 PyTorch 还没醒，系统只会给它看这张卡！
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt_args.gpu)

# ---------------- 只有在设置完环境变量后，才能导入深度学习相关库 ----------------
import optuna
import warnings
from datetime import datetime
import torch
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from flcore.servers.serverCLIP import FedCLIP

warnings.simplefilter("ignore")

def set_seed(seed=0):
    """固定所有随机种子，确保贝叶斯优化的严格控制变量"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial, opt_args):
    # 1. 固定全局随机种子
    set_seed(0)

    # 2. 手动构建精准的 args 对象
    args = argparse.Namespace()
    
    args.goal = "test"
    args.device = "cuda"
    
    # ⚠️ 既然外面已经锁定了显卡，对 PyTorch 来说唯一可见的卡就是 0 号！
    args.device_id = "0" 
    
    args.dataset = "Cifar100"
    args.num_classes = 100
    args.model_family = "Decom_CNN-5-512"
    args.batch_size = 16
    args.local_learning_rate = 0.005
    args.learning_rate_decay = False
    args.learning_rate_decay_gamma = 0.99
    args.global_rounds = 100
    args.top_cnt = 100
    args.local_epochs = 5
    args.algorithm = "FedCLIP"
    args.join_ratio = 1.0
    args.random_join_ratio = False
    args.num_clients = 20
    args.prev = 0
    args.times = 1
    args.eval_gap = 1
    
    # 把模型等碎片文件关进小黑屋，防止硬盘污染
    args.save_folder_name = os.path.join("Optuna_Temp", f"trial_{trial.number}")
    args.auto_break = False
    args.feature_dim = 512
    args.vocab_size = 80
    args.max_len = 200
    args.models_folder_name = ''
    args.few_shot = 0
    args.client_drop_rate = 0.0
    args.train_slow_rate = 0.0
    args.send_slow_rate = 0.0
    args.time_select = False
    args.time_threthold = 10000
    args.lamda = 1.0
    args.noise_dim = 512
    args.generator_learning_rate = 0.005
    args.hidden_dim = 512
    args.server_epochs = 100
    args.alpha = 1.0
    args.beta = 1.0
    args.mentee_learning_rate = 0.01
    args.T_start = 0.95
    args.T_end = 0.98
    args.server_learning_rate = 0.01
    args.margin_threthold = 100.0
    args.generator_path = 'stylegan/stylegan-xl-models/imagenet64.pkl'
    args.stable_diffusion_prompt = 'a cat'
    args.server_batch_size = 100
    args.gen_batch_size = 4
    args.mu = 50.0
    args.sub_feature_dim = 128
    args.align_epoch = 1
    args.is_regular = 1
    args.regular_lamda = 0.001
    args.align_lr = 0.01
    args.align_method = 1
    args.kl_Tim = 1
    args.kl_lamda = 0.1
    args.alpha_lr = 0.01
    args.wo_local = 1
    args.rank = 3
    args.layer_idx = 30
    args.gap = 5
    args.sce_lamda = 0.1
    args.Con_lamda = 1.0
    args.Rel_lamda = 1.0
    args.which_con = 'sim'
    args.hard_negative_mining = False
    args.is_TT_Decom = False
    args.topk = 5
    args.Con_T = 0.1
    args.Cos_lamda = 0.0
    args.temperature = 0.1
    args.struct_lamda = 0.1
    args.rel_lamda = 0.1
    args.niid = 1
    args.partition = 'dir'
    args.dir_alpha = 0.1
    args.class_per_client = 6
    args.resume = False
    
    # 动态记录试错编号，防止 h5 文件互相踩踏
    args.trial_id = trial.number
    args.exp_name = f"Tune_FedCLIP_Cifar100_Trial{trial.number}"

    # 3. 动态构建模型架构
    if args.model_family == "Decom_CNN-5-512":
        args.models = [
            'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=1.0)'
        ]
        args.global_model = 'Hyper_CNN_512(in_features=3,  num_classes=args.num_classes,n_kernels=16, ratio_LR=0.15)'
    else:
        raise NotImplementedError(f"Model family {args.model_family} not implemented in tuner.")

    # ================= 4. 贝叶斯优化接管的 5 个核心超参数 =================
    args.mse_lamda = trial.suggest_float('mse_lamda', 0.001, 4.0, log=True)
    args.v_mse_lamda = trial.suggest_float('v_mse_lahimda_log', 0.001, 4.0, log=True)
    
    args.aggregate_tau = trial.suggest_float('aggregate_tau', 0.05, 5.0)
    args.aggregate_power = trial.suggest_float('aggregate_power', 1.5, 7.0)
    args.aggregate_gamma = trial.suggest_float('aggregate_gamma', 0.1, 6.0)

    # ================= 5. 启动本轮训练 =================
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚀 GPU {opt_args.gpu} 开始 Trial {trial.number}...")
    print(f"超参: mse={args.mse_lamda:.3f}, v_mse={args.v_mse_lamda:.3f}, tau={args.aggregate_tau:.3f}, power={args.aggregate_power:.3f}, gamma={args.aggregate_gamma:.3f}")
    
    server = FedCLIP(args, 0)
    server.train()
    
    # 返回最高准确率
    final_acc = max(server.rs_test_acc) if len(server.rs_test_acc) > 0 else 0.0
    
    print(f"✅ Trial {trial.number} 结束 (GPU {opt_args.gpu})，最高准确率: {final_acc:.4f}")
    return final_acc


if __name__ == "__main__":
    db_name = "fedclip_tuning_tensor_db"
    storage_name = f"sqlite:///{db_name}.db"
    
    storage = optuna.storages.RDBStorage(
        url=storage_name,
        engine_kwargs={"connect_args": {"timeout": 60}} # 让排队的进程最多等 60 秒
    )


    study = optuna.create_study(
        study_name=db_name, 
        storage=storage, 
        direction="maximize", 
        load_if_exists=True 
    )
    
    # 每个打工人跑 8 次任务
    study.optimize(lambda trial: objective(trial, opt_args), n_trials=10)