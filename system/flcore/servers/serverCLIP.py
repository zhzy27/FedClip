import copy
import random
import time
from flcore.clients.clientCLIP import clientCLIP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import  Model_Distribe
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.get_clip_text_encoder import get_clip_class_embeddings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import math

class FedCLIP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCLIP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_model = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_model.recover_larger_model()
        self.global_acc=[]
        save_item(global_model, self.role, 'model', self.save_folder_name)
        clip_text_features,clip_text_features_norm = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
        self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()


    def train(self):
        for i in range(self.global_rounds+1):
            self.cur_ground = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # 下发就测试
            # self.send_parameters()
            if i%self.eval_gap == 0: # 测试间隔
                print(f"\n-------------Round number: {i} 的先测-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=i)
            if i%self.eval_gap == 0: # 再测一次看看到底那一次又问题
                print(f"\n-------------Round number: {i} 的后测-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        
        print("\nBest Global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        if len(self.global_acc) > 0:
            print(max(self.global_acc))
        else:
            print("未记录 Global accuracy")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_json_file()


    #从客户顿接受id信息和样本数信息
    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_parameters(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)



    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        
        self.uploaded_base_model = []
        delta_params_per_client = [] 
        
        # 1. 提取 Delta W
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
            
            old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            if old_start_model is not None:
                old_start_model = old_start_model.to(self.device)
            else:
                old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            
            client_layer_deltas = []
            for p_new, p_old in zip(model.base.parameters(), old_start_model.base.parameters()):
                delta_l = (p_new.data - p_old.data).view(-1)
                client_layer_deltas.append(delta_l)
                
            delta_params_per_client.append(client_layer_deltas)

        num_layers = len(delta_params_per_client[0])

        # ================== 修复：兜底聚合的安全防护 ==================
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
            
        if not hasattr(self, 'uploaded_weights') or len(self.uploaded_weights) != len(self.uploaded_ids):
            print("⚠️ 未检测到 uploaded_weights，使用均匀权重进行兜底聚合")
            fallback_weights = [1.0 / len(self.uploaded_ids)] * len(self.uploaded_ids)
        else:
            fallback_weights = self.uploaded_weights

        for w, base_model in zip(fallback_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)
        # ==============================================================

        # ================= 核心升级 1：计算数据量相对规模系数 =================
        num_participants = len(self.uploaded_ids)
        # 如果前面 fallback_weights 是均匀的，这里全是 1.0；
        # 如果是按样本量计算的真实 weights，这里就是相对规模！
        data_scales = [w * num_participants for w in fallback_weights]
        # ====================================================================

        print(f"执行基于按层(Layer-wise)与数据量先验(Data Prior)的个性化聚合...")
        
        tau = 0.25
        power = 2.0
        # beta = 0.3
        gamma = 2.0
        
        num_total_clients = len(self.clients) 
        global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_layers)]

        uploaded_params_per_client = [list(m.parameters()) for m in self.uploaded_base_model]

        for i, target_cid in enumerate(self.uploaded_ids):
            # # 获取目标客户端自己的数据规模系数
            scale_i = data_scales[i]
            
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()
                
            pers_params = list(personalized_global_model.base.parameters())

           # ... 保持前面的提取逻辑不变 ...
            for layer_idx in range(num_layers):
                logits = [] 
                
                delta_i = delta_params_per_client[i][layer_idx]
                pure_norm_i = torch.norm(delta_i)
                
                # ================= 核心修复：剔除参数尺寸霸权 (RMS) =================
                # 用总参数量的平方根进行规范化，让 Weight 和 Bias 处于同一量级
                rms_norm_i = pure_norm_i / math.sqrt(delta_i.numel())
                
                # 乘以 100 仅仅是为了让微小的梯度(比如 0.02)放大到 log1p 敏感的区间(比如 2.0)
                scaled_norm_i = rms_norm_i * 100.0
                # ====================================================================
                
                for j in range(len(self.uploaded_ids)):
                    delta_j = delta_params_per_client[j][layer_idx]
                    
                    cos_sim = torch.nn.functional.cosine_similarity(delta_i, delta_j, dim=0)
                    
                    
                    # ⚠️ 严重提醒：一定要检查这里你是不是又写回 math.log 了！
                    # 必须用 log1p 才能防止学渣拿高分的大锅饭！
                    # scale_j = data_scales[j]
                    # data_factor = math.log1p(scale_j) 
                    
                    # logit_j = (cos_sim * data_factor) / tau
                    # logits.append(logit_j)
                    # 2. 获取安全的数据规模 (加 1e-4 防止除以 0 的奇点崩溃)
                    safe_scale_j = max(data_scales[j], 1e-4)
                    
                    # 3. 终极优美数学门控：符号指数变换
                    # 正向梯度：乘 scale 奖励；负向梯度：除以 scale 严惩！
                    data_factor = safe_scale_j ** (torch.sign(cos_sim).item() * 0.5)
                    
                    # 4. 计算 Logit
                    logit_j = (cos_sim * data_factor) / tau
                    # if target_cid = 
                    # print(f"data_factor:{data_factor} logit_j {logit_j}")
                    logits.append(logit_j)
                
                logits = torch.stack(logits) 
                
                # ================= 护盾公式升级 =================
                depth_ratio = ((layer_idx + 1) / num_layers) ** power
                
                # 直接将自己的数据相对规模 scale_i 乘上去！
                # 散户(如 scale_i=0.03)：护盾被彻底粉碎，趋近于0，绝不盲目自信。
                # 大户(如 scale_i=2.0)：护盾被双倍强化，坚守本地防线。
                self_bias = depth_ratio * (gamma + torch.log1p(scaled_norm_i)) * (scale_i ** 0.5)
                
                logits[i] += self_bias
                # ===============================================
                
                layer_weights = torch.nn.functional.softmax(logits, dim=0)
                
                # 将该层的权重记录到对应的矩阵中
                # ... (后续写回矩阵和聚合的代码保持不变) ...
                
                # 将该层的权重记录到对应的矩阵中
                aligned_weights = np.zeros(num_total_clients)
                for j, upload_cid in enumerate(self.uploaded_ids):
                    aligned_weights[upload_cid] = layer_weights[j].item()
                    
                global_weight_matrices[layer_idx][target_cid] = aligned_weights

                # 对当前特定层执行物理加权
                for j in range(len(self.uploaded_ids)):
                    if layer_weights[j].item() > 0:
                        client_j_layer_data = uploaded_params_per_client[j][layer_idx].data
                        pers_params[layer_idx].data += client_j_layer_data.clone() * layer_weights[j].item()

            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)

        # 4. 遍历打印每一层的权重，并保存热力图
        for layer_idx in range(num_layers):
            self.print_row_weights(global_weight_matrices[layer_idx], layer_idx=layer_idx)



    def print_aligned_weights(self, global_weight_matrix):
        """
        专属视图层打印函数：不改变任何底层聚合逻辑，仅为了人类可读性
        强制将参与的客户端按 ID 升序排列打印，且权重数组的下标绝对对齐全局 ID
        """
        print("\n" + "="*20 + " 本轮个性化聚合权重分配 (绝对对齐版) " + "="*20)
        
        # 1. 对本轮实际参与的客户端 ID 进行升序排序
        sorted_upload_ids = sorted(self.uploaded_ids)
        
        # 2. 临时设置 numpy 的打印格式，保留3位小数，防止科学计数法，加宽单行防止折行
        original_printoptions = np.get_printoptions()
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        
        # 3. 按顺序打印
        for cid in sorted_upload_ids:
            # 从全局矩阵中取出属于该客户端的那一行
            aligned_weights = global_weight_matrix[cid]
            # 这里的 aligned_weights 长度已经是全网总人数了，且第 i 位就是给第 i 个人的权重
            print(f"  -> 客户端 {cid:2d} 的聚合权重: {aligned_weights}")
            
        # 4. 恢复原来的打印格式，防止影响其他地方
        np.set_printoptions(**original_printoptions)
        
        print("="*78 + "\n")

    def print_row_weights(self, raw_weight_matrix, layer_idx=None):
        """
        专属视图层保存函数：将单次启动的所有轮次、所有层重定向到同一个按时间排序的本地日志文件中
        """
        import os
        from datetime import datetime
        import numpy as np
        
        # 1. 创建专门存放权重的日志目录
        log_dir = "./Weight_Logs"
        os.makedirs(log_dir, exist_ok=True)
        
        current_round = getattr(self, 'global_round', None)
        if current_round is None:
            current_round = getattr(self, 'cur_ground', 0)
            
        # ================= 核心修改：一次启动，只生成一个文件 =================
        if not hasattr(self, 'weight_log_filepath'):
            # 只有在第一轮、第一次调用时才会执行这里
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{start_time}_All_Rounds_Weights.txt"
            self.weight_log_filepath = os.path.join(log_dir, log_filename)
            
            # 创建文件并写入一个总标题
            with open(self.weight_log_filepath, "w", encoding="utf-8") as f:
                f.write(f"========== 联邦学习权重聚合全局日志 (启动时间: {start_time}) ==========\n")
        # ====================================================================
        
        # 开始构建要写入文件的内容格式
        log_lines = []
        
        # 增加超级分割线：如果是一轮的开始（第 0 层），打一个巨无霸醒目标志
        if layer_idx == 0 or layer_idx is None:
            log_lines.append("\n\n" + "★"*25 + f" 🟢 第 {current_round:03d} 轮个性化聚合开始 " + "★"*25)
            
        title_suffix = f"(第 {layer_idx} 层参数 Tensor)" if layer_idx is not None else ""
        log_lines.append("\n" + "="*15 + f" 权重分配 {title_suffix} " + "="*15)
        
        sorted_upload_ids = sorted(self.uploaded_ids)
        original_printoptions = np.get_printoptions()
        np.set_printoptions(precision=6, suppress=True, linewidth=200) 
        
        for cid in sorted_upload_ids:
            raw_weights = raw_weight_matrix[cid]
            weights_str = np.array_str(raw_weights, max_line_width=200)
            log_lines.append(f"  -> 客户端 {cid:2d} 的导入权重: {weights_str}")
            
        log_lines.append("="*65)
        np.set_printoptions(**original_printoptions)
        
        # 写入同一个文件 (模式为 "a" 追加)
        with open(self.weight_log_filepath, "a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
            
        # 终端降噪：仅在第 0 层时报个平安
        if layer_idx == 0 or layer_idx is None:
            print(f"📄 第 {current_round} 轮权重矩阵已追加写入日志: {self.weight_log_filepath}")
        
        # 热力图生成保持不变
        prefix = f"raw_weight_heatmap_layer_{layer_idx}" if layer_idx is not None else "raw_weight_heatmap"
        self.save_weight_heatmap(raw_weight_matrix, filename_prefix=prefix)

    def save_weight_heatmap(self, weight_matrix, filename_prefix="weight_heatmap"):
        """
        专属视图层画图函数：根据传入的矩阵生成热力图并保存
        """
        import os
        import matplotlib.pyplot as plt
        import seaborn as sns
        from datetime import datetime

        current_round = getattr(self, 'global_round', None) 
        if current_round is None:
            current_round = getattr(self, 'cur_ground', 0)
            
        if current_round > 0 and current_round % 10 == 0:
            base_dir = "./Heatmap_Results"
            algo_name = getattr(self.args, 'algorithm', 'FedCLIP')
            dataset_name = getattr(self.args, 'dataset', 'UnknownData')
            alpha = getattr(self.args, 'dir_alpha', 'UnknownAlpha')
            sub1_name = f"{algo_name}_{dataset_name}_dir{alpha}_Similarity"
            
            if not hasattr(self, 'heatmap_run_time'):
                self.heatmap_run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub2_name = self.heatmap_run_time
            
            save_folder = os.path.join(base_dir, sub1_name, sub2_name)
            os.makedirs(save_folder, exist_ok=True)
            
            plt.figure(figsize=(10, 8))
            num_total_clients = len(self.clients)
            labels_abs = list(range(num_total_clients))
            
            sns.heatmap(weight_matrix, annot=False, cmap="YlGnBu", 
                        xticklabels=labels_abs, yticklabels=labels_abs)
            
            plt.title(f"Client Aggregation Weight Matrix ({filename_prefix} - Round {current_round})")
            
            # X轴提供知识，Y轴接收知识
            plt.xlabel("Source Client (Others)", fontsize=14)
            plt.ylabel("Target Client (Self)", fontsize=14)
            
            save_path = os.path.join(save_folder, f"{filename_prefix}_round_{current_round}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 考虑到层数较多，避免刷屏，可以选择注释掉这行打印
            # print(f"📊 第 {current_round} 轮 [{filename_prefix}] 热力图已保存至: {save_path}")
    def aggregate_val(self):
        print("--- 🔮 使用验真聚合函数 (Loading Oracle Weights from offline file) ---")
        assert (len(self.uploaded_ids) > 0)
        
        # 1. 加载上传的模型
        self.uploaded_base_model = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
            
        # 兜底：更新并保存全局通用模型
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
        for w, base_model in zip(self.uploaded_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)

        # 2. 动态拼接并读取离线权重文件名
        algo_name = getattr(self.args, 'algorithm', 'FedCLIP')
        dataset_name = getattr(self.args, 'dataset', 'Cifar10')
        partition = getattr(self.args, 'partition', 'dir')
        alpha_data = getattr(self.args, 'dir_alpha', 0.1) 
        
        weight_filename = f"{dataset_name}_{partition}_{alpha_data}_subset_noself_norm_weights.txt"
        weight_filepath = os.path.join("./Oracle_Weights", weight_filename)
        
        if not os.path.exists(weight_filepath):
            error_msg = (
                f"\n{'='*60}\n"
                f"❌ 严重错误：找不到离线权重文件！\n"
                f"试图加载的路径：{weight_filepath}\n"
                f"请检查:\n"
                f"1. 是否忘了在 dataset 目录下运行 plot_dataset_similarity.py 导出权重？\n"
                f"2. 训练脚本的 alpha ({alpha_data}) 是否与导出时的 alpha 一致？\n"
                f"{'='*60}"
            )
            raise FileNotFoundError(error_msg)
            
        oracle_weight_matrix = np.loadtxt(weight_filepath)
        
        # 3. 初始化对齐矩阵与保留系数
        num_total_clients = len(self.clients)
        current_round_weight_matrix = np.zeros((num_total_clients, num_total_clients))
        raw_round_weight_matrix = np.zeros((num_total_clients, num_total_clients))
        alpha_retention = 0.3 # 自己保留30%

        # 4. 执行个性化聚合
        for target_cid in self.uploaded_ids:
            target_weights = oracle_weight_matrix[target_cid]
            
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            # 提取本轮上线客户端的权重并重新归一化
            active_weights = []
            for upload_cid in self.uploaded_ids:
                active_weights.append(target_weights[upload_cid])
                
            active_weights = np.array(active_weights)
            
            if active_weights.sum() == 0:
                print(f"⚠️ 客户端 {target_cid} 匹配不到任何非0权重的上线节点，回退为全自身保留")
                active_weights = np.zeros_like(active_weights)
            else:
                active_weights = active_weights / active_weights.sum()

            # --- 分流记录纯净版与混合版打印矩阵 ---
            aligned_weights = np.zeros(num_total_clients)
            raw_aligned_weights = np.zeros(num_total_clients)
            
            for j, upload_cid in enumerate(self.uploaded_ids):
                raw_aligned_weights[upload_cid] = active_weights[j]
                aligned_weights[upload_cid] = active_weights[j] * (1.0 - alpha_retention)
                
            raw_round_weight_matrix[target_cid] = raw_aligned_weights
            aligned_weights[target_cid] += alpha_retention
            current_round_weight_matrix[target_cid] = aligned_weights

            # --- 物理双轨聚合 (他人 + 自己) ---
            for w, base_model in zip(active_weights, self.uploaded_base_model):
                if w > 0: 
                    for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                        server_param.data += client_param.data.clone() * w * (1.0 - alpha_retention)

            my_idx = self.uploaded_ids.index(target_cid)
            my_own_model = self.uploaded_base_model[my_idx]
            
            for server_param, my_param in zip(personalized_global_model.base.parameters(), my_own_model.parameters()):
                server_param.data += my_param.data.clone() * alpha_retention

            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
            
        # 5. 统一调用对齐打印函数 (严格区分变量)
        self.print_row_weights(raw_round_weight_matrix)
        self.print_aligned_weights(current_round_weight_matrix)
        print("✅ 验真聚合完成，模型已基于数据集真实特征更新。")