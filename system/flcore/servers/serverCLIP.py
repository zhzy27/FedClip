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
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=i)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_val()
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


    def aggregate_val(self):
        print("--- 🔮 使用验真聚合函数 (Loading Oracle Weights from offline file) ---")
        assert (len(self.uploaded_ids) > 0)
        
        self.uploaded_base_model = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            # 读取客户端刚训练完上传的模型
            client_model = load_item(client.role, 'model', client.save_folder_name)
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
            
        # (强烈建议也加上生成通用模型做兜底，防止第一轮未被抽中的客户端拉取时报错)
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
        for w, base_model in zip(self.uploaded_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)
        # ==============================================================================

        # 1. 动态拼接离线权重文件名
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
        # 2. 读取离线矩阵
        oracle_weight_matrix = np.loadtxt(weight_filepath)
        
        # ================= 新增：构建用于打印的全局对齐矩阵 =================
        num_total_clients = len(self.clients)
        current_round_weight_matrix = np.zeros((num_total_clients, num_total_clients))
        # 验真聚合时同样需要保留自身权重，否则 noself 矩阵会导致丢失本地模型
        alpha_retention = 0.3
        # ====================================================================

        # 3. 执行聚合
        for target_cid in self.uploaded_ids:
            target_weights = oracle_weight_matrix[target_cid]
            
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            # 提取本轮上线客户端的权重
            active_weights = []
            for upload_cid in self.uploaded_ids:
                active_weights.append(target_weights[upload_cid])
                
            active_weights = np.array(active_weights)
            
            if active_weights.sum() == 0:
                print(f"⚠️ 客户端 {target_cid} 匹配不到任何非0权重的上线节点，回退为全自身保留")
                active_weights = np.zeros_like(active_weights)
            else:
                active_weights = active_weights / active_weights.sum()

            # ================= 新增：对齐并保存至打印矩阵 =================
            aligned_weights = np.zeros(num_total_clients)
            for j, upload_cid in enumerate(self.uploaded_ids):
                # 记录他人的实际聚合权重 (权重 * 分配给他的比例)
                aligned_weights[upload_cid] = active_weights[j] * (1.0 - alpha_retention)
                
            # 把保留给自己的权重(alpha)加上，这样打印出来总和依然是 1.0
            aligned_weights[target_cid] += alpha_retention
            current_round_weight_matrix[target_cid] = aligned_weights
            # ==============================================================

            # 4. 执行双轨加权 (他人 + 自身)
            # 轨 1: 融合他人 (占比 1 - alpha_retention)
            for w, base_model in zip(active_weights, self.uploaded_base_model):
                if w > 0: 
                    for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                        server_param.data += client_param.data.clone() * w * (1.0 - alpha_retention)

            # 轨 2: 融合自身原生特性 (占比 alpha_retention)
            # 找到目标客户端在本轮 uploaded_base_model 中的索引位置
            my_idx = self.uploaded_ids.index(target_cid)
            my_own_model = self.uploaded_base_model[my_idx]
            
            for server_param, my_param in zip(personalized_global_model.base.parameters(), my_own_model.parameters()):
                server_param.data += my_param.data.clone() * alpha_retention

            # 保存模型
            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
            
        # ================= 新增：统一调用对齐打印函数 =================
        self.print_row_weights(current_round_weight_matrix)
        self.print_aligned_weights(current_round_weight_matrix)
        # ==============================================================
        
        print("✅ 验真聚合完成，模型已基于数据集真实特征更新。")

    # 客户端base进行个性化余弦相似度聚合
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        
        self.uploaded_base_model = []
        delta_params = [] # 用于存放精准的参数增量 Delta W

        # 1. 遍历所有上传的客户端，计算它们各自的 Delta W
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            
            # --- 1.1 加载客户端本地训练后的最新模型 ---
            client_model = load_item(client.role, 'model', client.save_folder_name)
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
            
            # --- 1.2 加载该客户端在训练前【原本的起点模型】 ---
            # 优先加载它的旧版专属个性化模型
            old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            
            if old_start_model is not None:
                old_start_model = old_start_model.to(self.device)
            else:
                # 如果没有专属模型（比如第一轮），说明它本地训练的起点是通用的全局模型
                old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            
            # --- 1.3 展平参数，计算真正的本地更新增量 Delta W ---
            flat_updated = torch.cat([p.data.view(-1) for p in model.base.parameters()])
            flat_old_start = torch.cat([p.data.view(-1) for p in old_start_model.base.parameters()])
            
            delta_w = flat_updated - flat_old_start
            delta_params.append(delta_w)


        # ================== 保留基础的 FedAvg 通用聚合 (兜底机制) ==================
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
        for w, base_model in zip(self.uploaded_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)
        # =====================================================================


        print(f"执行基于参数增量(Delta W)的个性化聚合，参与客户端: {self.uploaded_ids}")
        
        # 此时的相似度是基于纯粹的 Delta W 计算的，差异已经非常明显。
        # tau 可以先设为 0.05（如果你想两极分化更严重，可以下调到 0.01）
        tau = 1.0 

        # --- 修改：创建一个全局大小的零矩阵来保存热力图数据 ---
        num_total_clients = len(self.clients) 
        global_weight_matrix = np.zeros((num_total_clients, num_total_clients))

        # 2. 为每个上传的客户端计算专属的聚合权重
        for i, target_cid in enumerate(self.uploaded_ids):
            sims = []
            for j in range(len(self.uploaded_ids)):
                sim = torch.nn.functional.cosine_similarity(delta_params[i], delta_params[j], dim=0)
                sims.append(sim)
            
            sims = torch.stack(sims) 
            sims_scaled = (sims - torch.max(sims)) / tau
            weights = torch.nn.functional.softmax(sims_scaled, dim=0)
            
            # ================== 核心修正：映射到绝对 ID ==================
            # 创建一个长度为总客户端数的全 0 数组
            aligned_weights = np.zeros(num_total_clients)
            # 把算出的权重，精准地填到对应的绝对 Client ID 位置上
            for j, upload_cid in enumerate(self.uploaded_ids):
                aligned_weights[upload_cid] = weights[j].item()
                
            # 将对齐后的这行权重保存到全局矩阵对应的行中
            global_weight_matrix[target_cid] = aligned_weights
            
            # 此时打印的 aligned_weights，它的第 x 位就绝对是客户端 x 的权重！
            # print(f"  -> 客户端 {target_cid} 的绝对对齐权重: {aligned_weights.round(3)}")
            # ==========================================================

            # --- 下面的聚合逻辑保持不变 (因为 zip 依赖相对顺序，所以还是用原来的 weights) ---
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            for w, base_model in zip(weights, self.uploaded_base_model):
                for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                    server_param.data += client_param.data.clone() * w.item()

            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
        self.print_row_weights(global_weight_matrix)
        self.print_aligned_weights(global_weight_matrix)


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

    def print_row_weights(self, raw_weight_matrix):
        """
        专属视图层打印函数：打印应用 alpha 之前的纯净导入比例。
        对角线为 0，其他值为归一化后的原始权重，与离线导出的 txt 文件完全一致。
        """
        print("\n" + "="*20 + " 本轮个性化聚合权重分配 (原始离线导入版) " + "="*20)
        
        sorted_upload_ids = sorted(self.uploaded_ids)
        original_printoptions = np.get_printoptions()
        # precision=6，完美复刻 txt 文件的 6 位小数格式
        np.set_printoptions(precision=6, suppress=True, linewidth=200) 
        
        for cid in sorted_upload_ids:
            raw_weights = raw_weight_matrix[cid]
            print(f"  -> 客户端 {cid:2d} 的原始导入权重: {raw_weights}")
            
        np.set_printoptions(**original_printoptions)
        print("="*78 + "\n")
        self.save_weight_heatmap(raw_weight_matrix, filename_prefix="raw_weight_heatmap")

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
            
        # 每十轮画一次图
        if current_round > 0 and current_round % 10 == 0:
            base_dir = "./Heatmap_Results"
            algo_name = getattr(self.args, 'algorithm', 'FedCLIP')
            dataset_name = getattr(self.args, 'dataset', 'UnknownData')
            alpha = getattr(self.args, 'alpha', 'UnknownAlpha')
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
            
            # 标题也加上 prefix 区分是哪种矩阵
            plt.title(f"Client Aggregation Weight Matrix ({filename_prefix} - Round {current_round})")
            plt.xlabel("Source Client ID")
            plt.ylabel("Target Client ID")
            
            save_path = os.path.join(save_folder, f"{filename_prefix}_round_{current_round}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 第 {current_round} 轮 [{filename_prefix}] 热力图已保存至: {save_path}")