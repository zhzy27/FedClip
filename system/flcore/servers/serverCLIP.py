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
            if i > 0 and i % self.eval_gap == 0: 
                print(f"\n-------------Round number: {i} 聚合前-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            # if i%self.eval_gap == 0: # 再测一次看看到底那一次又问题
            #     print(f"\n-------------Round number: {i} 聚合后-------------")
            #     print("\nEvaluate heterogeneous models")
            #     self.evaluate(epoch=i)
                # self.
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            local_train_wall_start = time.time()
            client_train_times = []
            for client in self.selected_clients:
                client_train_time = client.train(current_round=i)
                if client_train_time is None:
                    client_train_time = getattr(client, "last_train_time_cost", 0.0)
                client_train_times.append((client.id, float(client_train_time)))
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            local_train_wall_time = time.time() - local_train_wall_start
            local_train_sum_time = sum(train_time for _, train_time in client_train_times)
            print(
                f"⏱️ [Round {i:03d}] 本地训练总耗时: "
                f"sum_client={local_train_sum_time:.3f}s | wall={local_train_wall_time:.3f}s | "
                f"clients={len(client_train_times)}"
            )
            print(
                "⏱️ [Round {:03d}] 客户端训练耗时明细: {}".format(
                    i,
                    ", ".join(
                        f"Client_{client_id}:{train_time:.3f}s"
                        for client_id, train_time in client_train_times
                    )
                )
            )
            

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_start = time.time()
            if "resnet" in getattr(self.args, "model_family", "").lower():
                self.aggregate_parameters_full_w_res()
            else:
                self.aggregate_parameters_full_w()
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_time = time.time() - aggregation_wall_start
            print(f"⏱️ [Round {i:03d}] 聚合总墙钟耗时: {aggregation_wall_time:.3f}s")
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

    def _has_low_rank_params(self, model):
        # 用参数名判断模型当前是否处在低秩形态；ResNet 的低秩卷积参数名是 conv_u/conv_v。
        return any(
            # 只要存在 V 矩阵，就说明这个模型还没有完全恢复成全秩卷积。
            name.endswith('conv_v') or name.endswith('weight_v')
            # 遍历模型的所有命名参数；这里不需要参数值，只需要名字。
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        # 聚合最终在全秩空间执行，所以只有低秩模型需要先恢复。
        if self._has_low_rank_params(model):
            # 调用模型自己的恢复接口；低秩 ResNet 和低秩 CNN 都使用这个接口名。
            model.recover_larger_model()
        # 返回同一个 model，方便调用处链式理解。
        return model

    def _low_rank_start_folder(self):
        # 客户端接收参数时已经分解过一次；该缓存恢复成满秩后就是本轮真实的 W 起点。
        return os.path.join(self.save_folder_name, 'low_rank_start')

    def _build_resnet18_layer_groups(self, named_params):
        # 保存全秩模型的参数名，后续只根据名字做 ResNet18 的逻辑层划分。
        param_names = [name for name, _ in named_params]
        # clientCLIP 使用的低秩 ResNet 外层通常是 base/head，因此 ResNet 主干参数以 base. 开头。
        base_prefix = "base." if any(name.startswith("base.") for name in param_names) else ""
        # 分类器可能叫 head，也可能叫 fc；这里先置空，再根据真实参数名判断。
        classifier_prefix = None
        # 当前 low_rank_resnet18_cifar 外层分类器叫 head。
        if any(name.startswith("head.") for name in param_names):
            classifier_prefix = "head."
        # 兼容 torchvision / 其他 ResNet 写法里的 fc。
        elif any(name.startswith("fc.") for name in param_names):
            classifier_prefix = "fc."

        # 每个 group 对应一个个性化权重单元；ResNet18 预期一共 18 个。
        groups = []

        def add_group(group_name, prefixes, primary_prefix=None):
            # 只有当至少一个参数名匹配当前 group 的前缀时，才真正加入这个 group。
            if any(any(name.startswith(prefix) for prefix in prefixes) for name in param_names):
                # name 用于日志打印，prefixes 用于把多个参数归到同一个权重单元。
                groups.append({
                    # 逻辑层名，例如 base.layer_3.conv2。
                    "name": group_name,
                    # 该逻辑层包含的参数名前缀，例如 conv/bn/downsample。
                    "prefixes": prefixes,
                    # full fallback 时优先用哪个前缀找代表参数，一般优先用 conv 或 head。
                    "primary_prefix": primary_prefix or prefixes[0],
                })

        # 第 1 个权重单元：CIFAR ResNet 的首层卷积，同时把对应 bn1 归到同一个权重。
        add_group(
            f"{base_prefix}conv1".rstrip("."),
            [f"{base_prefix}conv1.", f"{base_prefix}bn1."],
            f"{base_prefix}conv1.",
        )

        # 低秩 ResNet18 的 8 个 BasicBlock 在 SVD_resnet.py 中命名为 layer_0 到 layer_7。
        block_prefix = f"{base_prefix}layer_"
        # 用列表保存 block id，并保持模型定义里的顺序。
        block_ids = []
        # 从真实参数名里解析有哪些 layer_i，避免硬编码在模型结构变化时直接失效。
        for name in param_names:
            # 只处理 BasicBlock 的参数，跳过 conv1/head 等其他参数。
            if not name.startswith(block_prefix):
                continue
            # 去掉 base.layer_ 前缀，剩下形如 "0.conv1.weight" 或 "0.conv1.conv_v"。
            rest = name[len(block_prefix):]
            # block_id 是第一个点号前的数字。
            block_id = rest.split(".", 1)[0]
            # 只接受纯数字 id，并避免重复加入。
            if block_id.isdigit() and int(block_id) not in block_ids:
                block_ids.append(int(block_id))
        # 排序后保证 layer_0, layer_1, ... 的深度顺序稳定。
        block_ids.sort()

        # 每个 BasicBlock 有两个主卷积，因此每个 block 拆成 conv1 和 conv2 两个权重单元。
        for block_id in block_ids:
            # 当前 block 的公共前缀，例如 base.layer_3。
            block = f"{base_prefix}layer_{block_id}"
            # block 的 conv1 权重单元；downsample 是这一步的残差投影，跟 conv1 同步聚合更合理。
            add_group(
                f"{block}.conv1",
                [f"{block}.conv1.", f"{block}.bn1.", f"{block}.downsample."],
                f"{block}.conv1.",
            )
            # block 的 conv2 权重单元；bn2 跟随 conv2 使用同一套个性化权重。
            add_group(
                f"{block}.conv2",
                [f"{block}.conv2.", f"{block}.bn2."],
                f"{block}.conv2.",
            )

        # 最后 1 个权重单元：分类器 head/fc。
        if classifier_prefix is not None:
            add_group(
                classifier_prefix.rstrip("."),
                [classifier_prefix],
                classifier_prefix,
            )

        # ResNet18 的主层数应为 1 + 8*2 + 1 = 18；不等于 18 时直接打印，方便查命名问题。
        if len(groups) != 18:
            print(f"⚠️ ResNet18 聚合层数解析为 {len(groups)}，预期为 18。请检查模型结构或命名。")
            print("解析到的聚合层:", [group["name"] for group in groups])
        # 返回固定顺序的 18 个聚合单元，后面 depth_ratio 就按这个顺序计算。
        return groups

    def _prepare_full_w_aggregation_inputs(self):
        """Recover uploaded and actual-start models, then build full-W deltas."""
        self.uploaded_base_model = []
        uploaded_full_param_dicts = []
        full_delta_params_per_client = []
        cache_hits = 0
        cache_misses = 0

        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            if client_model is None:
                raise RuntimeError(f"Client_{cid} uploaded model is missing.")
            low_rank_end = copy.deepcopy(client_model).to(self.device)
            self.uploaded_base_model.append(low_rank_end)

            actual_start = load_item(self.role, f'model_{cid}', self._low_rank_start_folder())
            if actual_start is not None:
                cache_hits += 1
            else:
                cache_misses += 1
                print(
                    f"⚠️ Client_{cid} 缺少本轮真实低秩训练起点；"
                    "将退回客户端专属服务器模型，当前轮 delta 可能存在截断误差。"
                )
                actual_start = load_item(self.role, f'model_{cid}', self.save_folder_name)
            if actual_start is None:
                actual_start = load_item(self.role, 'model', self.save_folder_name)
            if actual_start is None:
                raise RuntimeError(f"Client_{cid} has no model available as its training start.")

            full_end = copy.deepcopy(low_rank_end).to(self.device)
            self._recover_if_needed(full_end)
            full_end = full_end.to(self.device)

            full_start = copy.deepcopy(actual_start).to(self.device)
            self._recover_if_needed(full_start)
            full_start = full_start.to(self.device)

            end_params = dict(full_end.named_parameters())
            start_params = dict(full_start.named_parameters())
            if end_params.keys() != start_params.keys():
                missing_at_start = sorted(end_params.keys() - start_params.keys())
                missing_at_end = sorted(start_params.keys() - end_params.keys())
                raise RuntimeError(
                    f"Client_{cid} full-rank start/end parameter names differ: "
                    f"missing_at_start={missing_at_start}, missing_at_end={missing_at_end}"
                )

            full_deltas = {}
            for name, end_param in end_params.items():
                start_param = start_params[name]
                if end_param.shape != start_param.shape:
                    raise RuntimeError(
                        f"Client_{cid} full-rank delta shape mismatch for {name}: "
                        f"end={tuple(end_param.shape)}, start={tuple(start_param.shape)}"
                    )
                full_deltas[name] = end_param.detach().clone() - start_param.detach().clone()

            uploaded_full_param_dicts.append(end_params)
            full_delta_params_per_client.append(full_deltas)

        return uploaded_full_param_dicts, full_delta_params_per_client, cache_hits, cache_misses

    def _full_w_delta_cosine(self, delta_i, delta_j, layer_name):
        if delta_i.shape != delta_j.shape:
            raise RuntimeError(
                f"Full-W delta shape mismatch in {layer_name}: "
                f"{tuple(delta_i.shape)} vs {tuple(delta_j.shape)}"
            )
        flat_i = delta_i.reshape(-1)
        flat_j = delta_j.reshape(-1)
        if flat_i.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        similarity = torch.nn.functional.cosine_similarity(flat_i, flat_j, dim=0)
        if not torch.isfinite(similarity):
            raise RuntimeError(f"Non-finite full-W delta similarity in {layer_name}.")
        return similarity

    def _full_w_similarity_matrix(self, anchor_name, layer_name, full_deltas):
        num_participants = len(full_deltas)
        sim_matrix = torch.zeros((num_participants, num_participants), device=self.device)
        for i in range(num_participants):
            if anchor_name not in full_deltas[i]:
                raise RuntimeError(f"Client_{self.uploaded_ids[i]} is missing full-W delta {anchor_name}.")
            for j in range(i, num_participants):
                if anchor_name not in full_deltas[j]:
                    raise RuntimeError(f"Client_{self.uploaded_ids[j]} is missing full-W delta {anchor_name}.")
                similarity = self._full_w_delta_cosine(
                    full_deltas[i][anchor_name],
                    full_deltas[j][anchor_name],
                    layer_name,
                )
                sim_matrix[i, j] = similarity
                sim_matrix[j, i] = similarity
        return sim_matrix

    def _mixed_personalized_weights(self, similarities, depth_ratio, tau):
        personal_weights = torch.nn.functional.softmax(similarities / tau, dim=0).detach().cpu().numpy()
        fallback_weights = np.asarray(self.uploaded_weights, dtype=np.float64)
        return (1.0 - depth_ratio) * fallback_weights + depth_ratio * personal_weights

    def aggregate_parameters_full_w_res(self):
        assert len(self.uploaded_ids) > 0
        print("🚀 开始 ResNet18 聚合：使用恢复后的满秩 W 变化量计算层级相似度")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_start = time.time()

        model_prepare_start = time.time()
        full_param_dicts, full_deltas, cache_hits, cache_misses = self._prepare_full_w_aggregation_inputs()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        model_prepare_time = time.time() - model_prepare_start
        print(
            f"⏱️ ResNet18 full-W model_prepare: cache_hit={cache_hits} | "
            f"cache_miss={cache_misses} | total={model_prepare_time:.3f}s"
        )

        reference_named_params = list(full_param_dicts[0].items())
        layer_groups = self._build_resnet18_layer_groups(reference_named_params)
        num_layers = len(layer_groups)
        if num_layers == 0:
            raise RuntimeError("No ResNet18 aggregation layer was found in the recovered full-rank model.")
        tau = self.args.aggregate_tau if self.args.aggregate_tau > 0 else 1.0
        num_participants = len(self.uploaded_ids)
        num_total_clients = len(self.clients)

        def matches_group(name, group):
            return any(name.startswith(prefix) for prefix in group["prefixes"])

        sim_matrix_start = time.time()
        sim_matrices = {}
        print("🧮 正在计算 ResNet18 满秩 W 变化量相似度矩阵...")
        for layer_idx, group in enumerate(layer_groups):
            search_prefixes = [group["primary_prefix"]] + [
                prefix for prefix in group["prefixes"] if prefix != group["primary_prefix"]
            ]
            anchor_name = next(
                (
                    name
                    for prefix in search_prefixes
                    for name in full_param_dicts[0]
                    if name.startswith(prefix) and name.endswith('.weight')
                ),
                None,
            )
            if anchor_name is None:
                raise RuntimeError(f"No full-rank weight anchor found for ResNet layer {group['name']}.")
            sim_matrices[group["name"]] = self._full_w_similarity_matrix(
                anchor_name, group["name"], full_deltas
            )
            print(f"  -> 第 {layer_idx + 1:02d} 层 {group['name']}: 参数={anchor_name}")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        sim_matrix_time = time.time() - sim_matrix_start

        weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_layers)]
        personal_weight_time = 0.0
        param_aggregate_time = 0.0
        for target_idx, target_cid in enumerate(self.uploaded_ids):
            personalized_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            self._recover_if_needed(personalized_model)
            personalized_model = personalized_model.to(self.device)
            for param in personalized_model.parameters():
                param.data.zero_()

            target_params = dict(personalized_model.named_parameters())
            covered_names = set()
            for layer_idx, group in enumerate(layer_groups):
                depth_ratio = 0.7 * (layer_idx + 1) / num_layers
                weight_start = time.time()
                mixed_weights = self._mixed_personalized_weights(
                    sim_matrices[group["name"]][target_idx], depth_ratio, tau
                )
                aligned_weights = np.zeros(num_total_clients)
                for source_idx, source_cid in enumerate(self.uploaded_ids):
                    aligned_weights[source_cid] = mixed_weights[source_idx]
                personal_weight_time += time.time() - weight_start

                aggregate_start = time.time()
                group_param_names = [name for name in target_params if matches_group(name, group)]
                for param_name in group_param_names:
                    covered_names.add(param_name)
                    target_param = target_params[param_name]
                    for source_idx in range(num_participants):
                        target_param.data += full_param_dicts[source_idx][param_name].data * mixed_weights[source_idx]
                param_aggregate_time += time.time() - aggregate_start
                weight_matrices[layer_idx][target_cid] = aligned_weights

            uncovered_names = [name for name in target_params if name not in covered_names]
            if uncovered_names:
                aggregate_start = time.time()
                for param_name in uncovered_names:
                    for source_idx, fallback_weight in enumerate(self.uploaded_weights):
                        target_params[param_name].data += (
                            full_param_dicts[source_idx][param_name].data * fallback_weight
                        )
                param_aggregate_time += time.time() - aggregate_start
                print(
                    f"⚠️ 目标客户端 {target_cid} 有 {len(uncovered_names)} 个参数未归入18层，"
                    "使用样本量权重聚合。"
                )

            save_start = time.time()
            save_item(personalized_model, self.role, f'model_{target_cid}', self.save_folder_name)
            param_aggregate_time += time.time() - save_start

        weight_print_start = time.time()
        for layer_idx, matrix in enumerate(weight_matrices):
            self.print_row_weights(matrix, layer_idx=layer_idx)
        weight_print_time = time.time() - weight_print_start
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        total_time = time.time() - aggregate_total_start
        print(
            f"⏱️ ResNet18 聚合耗时拆分: model_prepare={model_prepare_time:.3f}s | "
            f"sim_matrix={sim_matrix_time:.3f}s | personal_weight={personal_weight_time:.3f}s | "
            f"param_aggregate_save={param_aggregate_time:.3f}s | weight_print={weight_print_time:.3f}s | "
            f"total_inside={total_time:.3f}s"
        )

    def aggregate_parameters_full_w(self):
        assert len(self.uploaded_ids) > 0
        print("🚀 开始 CNN 聚合：使用恢复后的满秩 W 变化量计算层级相似度")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_start = time.time()

        model_prepare_start = time.time()
        full_param_dicts, full_deltas, cache_hits, cache_misses = self._prepare_full_w_aggregation_inputs()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        model_prepare_time = time.time() - model_prepare_start
        print(
            f"⏱️ CNN full-W model_prepare: cache_hit={cache_hits} | "
            f"cache_miss={cache_misses} | total={model_prepare_time:.3f}s"
        )

        full_param_names = list(full_param_dicts[0].keys())
        logical_layers = []
        for name in full_param_names:
            parent_name = name.rsplit('.', 1)[0]
            if parent_name not in logical_layers:
                logical_layers.append(parent_name)
        if not logical_layers:
            raise RuntimeError("No CNN aggregation layer was found in the recovered full-rank model.")

        layer_param_names = {
            layer_name: [name for name in full_param_names if name.rsplit('.', 1)[0] == layer_name]
            for layer_name in logical_layers
        }
        layer_anchors = {}
        for layer_name, param_names in layer_param_names.items():
            anchor_name = next((name for name in param_names if name.endswith('.weight')), None)
            if anchor_name is None:
                raise RuntimeError(f"No full-rank weight anchor found for CNN layer {layer_name}.")
            layer_anchors[layer_name] = anchor_name

        tau = self.args.aggregate_tau if self.args.aggregate_tau > 0 else 1.0
        num_participants = len(self.uploaded_ids)
        num_total_clients = len(self.clients)
        sim_matrix_start = time.time()
        sim_matrices = {}
        print("🧮 正在计算 CNN 满秩 W 变化量相似度矩阵...")
        for layer_name, anchor_name in layer_anchors.items():
            sim_matrices[layer_name] = self._full_w_similarity_matrix(
                anchor_name, layer_name, full_deltas
            )
            print(f"  -> {layer_name}: 参数={anchor_name}")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        sim_matrix_time = time.time() - sim_matrix_start

        param_indices = {name: idx for idx, name in enumerate(full_param_names)}
        weight_matrices = [
            np.zeros((num_total_clients, num_total_clients)) for _ in full_param_names
        ]
        personal_weight_time = 0.0
        param_aggregate_time = 0.0
        for target_idx, target_cid in enumerate(self.uploaded_ids):
            personalized_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            self._recover_if_needed(personalized_model)
            personalized_model = personalized_model.to(self.device)
            for param in personalized_model.parameters():
                param.data.zero_()
            target_params = dict(personalized_model.named_parameters())

            for layer_idx, layer_name in enumerate(logical_layers):
                depth_ratio = 0.7 * (layer_idx + 1) / len(logical_layers)
                weight_start = time.time()
                mixed_weights = self._mixed_personalized_weights(
                    sim_matrices[layer_name][target_idx], depth_ratio, tau
                )
                aligned_weights = np.zeros(num_total_clients)
                for source_idx, source_cid in enumerate(self.uploaded_ids):
                    aligned_weights[source_cid] = mixed_weights[source_idx]
                personal_weight_time += time.time() - weight_start

                aggregate_start = time.time()
                for param_name in layer_param_names[layer_name]:
                    target_param = target_params[param_name]
                    for source_idx in range(num_participants):
                        target_param.data += full_param_dicts[source_idx][param_name].data * mixed_weights[source_idx]
                    weight_matrices[param_indices[param_name]][target_cid] = aligned_weights
                param_aggregate_time += time.time() - aggregate_start

            save_start = time.time()
            personalized_model.decom_larger_model(self.uploaded_base_model[target_idx].ratio_LR)
            personalized_model = personalized_model.to(self.device)
            save_item(personalized_model, self.role, f'model_{target_cid}', self.save_folder_name)
            param_aggregate_time += time.time() - save_start

        weight_print_start = time.time()
        for tensor_idx, matrix in enumerate(weight_matrices):
            self.print_row_weights(matrix, layer_idx=tensor_idx)
        weight_print_time = time.time() - weight_print_start
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        total_time = time.time() - aggregate_total_start
        print(
            f"⏱️ CNN 聚合耗时拆分: model_prepare={model_prepare_time:.3f}s | "
            f"sim_matrix={sim_matrix_time:.3f}s | personal_weight={personal_weight_time:.3f}s | "
            f"param_aggregate_save={param_aggregate_time:.3f}s | weight_print={weight_print_time:.3f}s | "
            f"total_inside={total_time:.3f}s"
        )

    def aggregate_parameters_v_svd_res(self):
        # Backward-compatible entry point; active similarity is full-W delta based.
        return self.aggregate_parameters_full_w_res()

        # 没有客户端上传时不能聚合。
        assert (len(self.uploaded_ids) > 0)
        print("🚀 开始 ResNet18 聚合 (18层权重：低秩层优先用 V，相似度缺失时退回全秩)")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_start = time.time()
        model_prepare_start = time.time()

        def sync_prepare_step():
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)

        prepare_times = {
            "load_client": 0.0,
            "copy_client_to_device": 0.0,
            "load_old": 0.0,
            "old_to_device": 0.0,
            "old_decompose": 0.0,
            "low_rank_delta": 0.0,
            "full_current_copy": 0.0,
            "full_current_recover": 0.0,
            "full_current_to_device": 0.0,
            "full_current_dict": 0.0,
            "full_old_copy": 0.0,
            "full_old_recover": 0.0,
            "full_old_to_device": 0.0,
            "full_old_dict": 0.0,
            "full_delta": 0.0,
        }
        low_rank_cache_hits = 0
        low_rank_cache_misses = 0

        # 保存客户端上传的低秩模型；后面按目标客户端的 ratio_LR 再分解回对应秩。
        self.uploaded_base_model = []
        # 保存低秩空间里的 delta；低秩 V 相似度优先从这里取。
        delta_params_per_client = []
        # 保存恢复到全秩后的 delta；没有低秩 V 的层会退回这里计算相似度。
        full_delta_params_per_client = []
        # 保存每个客户端恢复到全秩后的模型；最终参数聚合从这些模型取值。
        uploaded_full_models = []
        # 保存全秩模型参数字典，避免内层循环反复 dict(model.named_parameters())。
        uploaded_full_param_dicts = []

        # 第一阶段：读取每个上传客户端的模型，并构造低秩 delta 与全秩 delta。
        for cid in self.uploaded_ids:
            # 根据客户端 id 找到客户端对象。
            client = self.clients[cid]
            # 读取该客户端本轮本地训练后的模型。
            step_start = time.time()
            client_model = load_item(client.role, 'model', client.save_folder_name)
            prepare_times["load_client"] += time.time() - step_start
            # 深拷贝后放到服务器设备，避免修改客户端缓存对象。
            step_start = time.time()
            model = copy.deepcopy(client_model).to(self.device)
            sync_prepare_step()
            prepare_times["copy_client_to_device"] += time.time() - step_start

            # 读取该客户端上一轮下发前保存的专属模型，用作 delta 的起点。
            step_start = time.time()
            old_start_model = load_item(self.role, f'model_{cid}', self._low_rank_start_folder())
            if old_start_model is not None:
                low_rank_cache_hits += 1
            else:
                low_rank_cache_misses += 1
                old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            # 如果是第一轮、没有专属模型、也没有低秩缓存，就用服务器通用模型作为起点。
            if old_start_model is None:
                # 服务器通用模型通常是全秩，后面会按当前客户端 rank 临时分解。
                old_start_model = load_item(self.role, 'model', self.save_folder_name)
            prepare_times["load_old"] += time.time() - step_start
            # 确保起点模型在同一设备上，避免 delta 计算时 device mismatch。
            step_start = time.time()
            old_start_model = old_start_model.to(self.device)
            sync_prepare_step()
            prepare_times["old_to_device"] += time.time() - step_start
            # server 保存给客户端的是全秩模型；计算低秩 delta 前，临时分解到当前客户端的 rank。
            if not self._has_low_rank_params(old_start_model):
                step_start = time.time()
                old_start_model.decom_larger_model(model.ratio_LR)
                old_start_model = old_start_model.to(self.device)
                sync_prepare_step()
                prepare_times["old_decompose"] += time.time() - step_start

            # 低秩 delta 字典：name -> 当前低秩参数 - 起点低秩参数。
            step_start = time.time()
            client_raw_deltas = {}
            # 这里要求 model 与 old_start_model 的参数顺序一致；同一个模型类下成立。
            for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
                # clone 防止后续原参数变化影响 delta。
                client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
            sync_prepare_step()
            prepare_times["low_rank_delta"] += time.time() - step_start
            # 保存当前客户端低秩 delta。
            delta_params_per_client.append(client_raw_deltas)
            # 保存当前客户端低秩模型本体。
            self.uploaded_base_model.append(model)

            # 准备全秩版本用于最终参数聚合。
            step_start = time.time()
            full_m = copy.deepcopy(model).to(self.device)
            sync_prepare_step()
            prepare_times["full_current_copy"] += time.time() - step_start
            # 如果模型仍是低秩形态，则恢复成全秩卷积。
            step_start = time.time()
            self._recover_if_needed(full_m)
            sync_prepare_step()
            prepare_times["full_current_recover"] += time.time() - step_start
            # recover 会替换模块，重新 to 一次确保新模块也在正确设备。
            step_start = time.time()
            full_m = full_m.to(self.device)
            sync_prepare_step()
            prepare_times["full_current_to_device"] += time.time() - step_start
            # 保存全秩模型。
            uploaded_full_models.append(full_m)
            # 缓存全秩参数字典，后面按参数名直接索引。
            step_start = time.time()
            uploaded_full_param_dicts.append(dict(full_m.named_parameters()))
            prepare_times["full_current_dict"] += time.time() - step_start

            # 起点模型也恢复成全秩，用于计算没有 V 的层的 full delta。
            step_start = time.time()
            old_full_m = copy.deepcopy(old_start_model).to(self.device)
            sync_prepare_step()
            prepare_times["full_old_copy"] += time.time() - step_start
            # 如果起点是低秩形态，则先恢复。
            step_start = time.time()
            self._recover_if_needed(old_full_m)
            sync_prepare_step()
            prepare_times["full_old_recover"] += time.time() - step_start
            # recover 会新建全秩卷积层，新模块默认在 CPU；必须重新搬到当前设备。
            step_start = time.time()
            old_full_m = old_full_m.to(self.device)
            sync_prepare_step()
            prepare_times["full_old_to_device"] += time.time() - step_start
            # 起点全秩参数字典。
            step_start = time.time()
            old_full_param_dict = dict(old_full_m.named_parameters())
            prepare_times["full_old_dict"] += time.time() - step_start
            # 全秩 delta 字典：name -> 当前全秩参数 - 起点全秩参数。
            step_start = time.time()
            full_delta_params = {}
            # 遍历当前全秩模型参数，按同名参数找旧值。
            for name, p_new in full_m.named_parameters():
                # 全秩 delta 用于 base.conv1、head，以及任何未低秩化的层。
                full_delta_params[name] = p_new.data.clone() - old_full_param_dict[name].data.clone()
            sync_prepare_step()
            prepare_times["full_delta"] += time.time() - step_start
            # 保存当前客户端全秩 delta。
            full_delta_params_per_client.append(full_delta_params)

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        model_prepare_time = time.time() - model_prepare_start
        prepare_accounted_time = sum(prepare_times.values())
        print(
            f"⏱️ ResNet18 model_prepare 细分: "
            f"low_rank_cache_hit={low_rank_cache_hits} | "
            f"low_rank_cache_miss={low_rank_cache_misses} | "
            f"load_client={prepare_times['load_client']:.3f}s | "
            f"copy_client_to_device={prepare_times['copy_client_to_device']:.3f}s | "
            f"load_old={prepare_times['load_old']:.3f}s | "
            f"old_to_device={prepare_times['old_to_device']:.3f}s | "
            f"old_decompose={prepare_times['old_decompose']:.3f}s | "
            f"low_rank_delta={prepare_times['low_rank_delta']:.3f}s | "
            f"full_current_copy={prepare_times['full_current_copy']:.3f}s | "
            f"full_current_recover={prepare_times['full_current_recover']:.3f}s | "
            f"full_current_to_device={prepare_times['full_current_to_device']:.3f}s | "
            f"full_current_dict={prepare_times['full_current_dict']:.3f}s | "
            f"full_old_copy={prepare_times['full_old_copy']:.3f}s | "
            f"full_old_recover={prepare_times['full_old_recover']:.3f}s | "
            f"full_old_to_device={prepare_times['full_old_to_device']:.3f}s | "
            f"full_old_dict={prepare_times['full_old_dict']:.3f}s | "
            f"full_delta={prepare_times['full_delta']:.3f}s | "
            f"unaccounted={max(model_prepare_time - prepare_accounted_time, 0.0):.3f}s | "
            f"total={model_prepare_time:.3f}s"
        )

        # 样本量权重作为全局兜底权重，也用于和个性化权重按 depth_ratio 融合。
        fallback_weights = self.uploaded_weights
        # 本轮实际参与上传的客户端数。
        num_participants = len(self.uploaded_ids)
        # 将归一化样本量权重放缩到均值约为 1，用作相似度 logit 的数据可靠性因子。
        data_scales = [w * num_participants for w in fallback_weights]
        # 总客户端数用于构造完整 N x N 打印矩阵，未参与客户端保持 0。
        num_total_clients = len(self.clients)

        # 用第一个全秩模型的参数名解析 ResNet18 的 18 个逻辑聚合层。
        full_named_params = list(uploaded_full_models[0].named_parameters())
        # 每个元素包含 name/prefixes/primary_prefix。
        res_layers = self._build_resnet18_layer_groups(full_named_params)
        # 实际解析出的层数，正常应为 18。
        num_res_layers = len(res_layers)
        print(f"🚀 ResNet18 聚合层数: {num_res_layers} | 全秩参数张量数: {len(full_named_params)}")

        # softmax 温度，接口与原 aggregate_parameters_v_svd 保持一致。
        tau = self.args.aggregate_tau if self.args.aggregate_tau > 0 else 1.0

        def match_group(name, group):
            # 判断某个参数名是否属于当前逻辑层 group。
            return any(name.startswith(prefix) for prefix in group["prefixes"])

        def select_v_delta(client_idx, group):
            # 取指定客户端的低秩 delta 字典。
            delta_dict = delta_params_per_client[client_idx]
            # 在当前逻辑层的所有前缀里查找 V 矩阵。
            for prefix in group["prefixes"]:
                # 遍历该客户端所有低秩参数。
                for name, delta in delta_dict.items():
                    # ResNet 低秩卷积的 V 叫 conv_v；Linear 低秩时兼容 weight_v。
                    if name.startswith(prefix) and (name.endswith('conv_v') or name.endswith('weight_v')):
                        # 返回参数名和对应 delta，参数名用于确认 i/j 是否同一层。
                        return name, delta
            # 当前逻辑层没有低秩 V，比如 base.conv1 或 head。
            return None, None

        def select_full_delta(client_idx, group):
            # 取指定客户端的全秩 delta 字典。
            delta_dict = full_delta_params_per_client[client_idx]
            # full fallback 优先在 primary_prefix 中找代表参数，再找同 group 的其他参数。
            search_prefixes = [group["primary_prefix"]] + [
                # 保留其他前缀作为兜底，比如 bn 或 downsample。
                prefix for prefix in group["prefixes"] if prefix != group["primary_prefix"]
            ]
            # 第一轮优先找 weight，因为 weight 比 bias/BN 参数更适合作相似度锚点。
            for prefix in search_prefixes:
                # 遍历全秩 delta 参数。
                for name, delta in delta_dict.items():
                    # 优先返回当前前缀下的 weight。
                    if name.startswith(prefix) and name.endswith('.weight'):
                        return name, delta
            # 如果没有 weight，就退而求其次找任意属于该前缀的参数。
            for prefix in search_prefixes:
                # 遍历全秩 delta 参数。
                for name, delta in delta_dict.items():
                    # 找到第一个匹配参数即可。
                    if name.startswith(prefix):
                        return name, delta
            # 当前逻辑层没有可用全秩参数；理论上不该发生，发生时后面会记 missing。
            return None, None

        def cosine_by_common_prefix(delta_i, delta_j):
            # 不同客户端的低秩 rank 可能不同，因此只取每个维度的公共前缀比较。
            slices = tuple(
                # 每个维度截到 min(dim_i, dim_j)，这就是有序 dropout 的前缀对齐。
                slice(0, min(dim_i, dim_j))
                # zip 会逐维比较两个 tensor 的 shape。
                for dim_i, dim_j in zip(delta_i.shape, delta_j.shape)
            )
            # 截断后展平成向量，准备计算 cosine。
            trunc_i = delta_i[slices].contiguous().view(-1)
            # 第二个客户端同样截断到公共前缀。
            trunc_j = delta_j[slices].contiguous().view(-1)
            # 如果公共部分为空，就返回 0 相似度，避免 cosine 报错。
            if trunc_i.numel() == 0:
                return torch.tensor(0.0, device=self.device)
            # 计算两个 delta 向量的余弦相似度。
            return torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0)

        # 保存每个逻辑层的参与客户端 x 参与客户端相似度矩阵。
        sim_matrices = {}
        # 保存每层相似度来源统计，主要用于调试确认是否真的使用 V。
        sim_sources = {}
        # 记录多少层实际使用了低秩 V。
        sim_debug_printed = set()
        print("🧮 正在计算 ResNet18 的层级相似度矩阵...")
        sim_matrix_start = time.time()
        # 逐层计算相似度矩阵。
        for layer_idx, group in enumerate(res_layers):
            # 当前逻辑层的相似度矩阵，大小为本轮参与客户端数 x 本轮参与客户端数。
            sim_mat = torch.zeros((num_participants, num_participants), device=self.device)
            # 统计当前层有多少 pair 使用 V、full fallback 或 missing。
            source_counter = {"v": 0, "full": 0, "missing": 0}
            # 保存一个示例 V 参数名，日志里打印出来方便检查命名是否对。
            example_v_name = None
            # 保存一个示例全秩参数名，说明 fallback 具体用了哪个参数。
            example_full_name = None

            # 利用相似度对称性，只计算上三角。
            for i in range(num_participants):
                # j 从 i 开始，避免重复计算 i-j 和 j-i。
                for j in range(i, num_participants):
                    # 尝试拿客户端 i 当前层的低秩 V delta。
                    v_name_i, v_delta_i = select_v_delta(i, group)
                    # 尝试拿客户端 j 当前层的低秩 V delta。
                    v_name_j, v_delta_j = select_v_delta(j, group)

                    # 只有两个客户端都找到 V，且 V 名字完全一致，才在低秩 V 空间算相似度。
                    if v_name_i is not None and v_name_i == v_name_j:
                        # 对不同 rank 的 V 做公共前缀截断后计算 cosine。
                        cos_sim = cosine_by_common_prefix(v_delta_i, v_delta_j)
                        # 记录该 pair 使用了 V。
                        source_counter["v"] += 1
                        # 第一次命中 V 时记录参数名用于日志。
                        if example_v_name is None:
                            example_v_name = v_name_i
                    # 如果没有可对齐的 V，就退回全秩 delta。
                    else:
                        # 取客户端 i 当前层的全秩代表参数。
                        full_name_i, full_delta_i = select_full_delta(i, group)
                        # 取客户端 j 当前层的全秩代表参数。
                        full_name_j, full_delta_j = select_full_delta(j, group)
                        # 如果任一客户端没有对应参数，标记为不可用。
                        if full_delta_i is None or full_delta_j is None:
                            cos_sim = torch.tensor(-9999.0, device=self.device)
                            source_counter["missing"] += 1
                        # 两边都有全秩代表参数时，用全秩 delta 算 cosine。
                        else:
                            cos_sim = cosine_by_common_prefix(full_delta_i, full_delta_j)
                            source_counter["full"] += 1
                            # 保存一个 fallback 参数名，方便日志审计。
                            if example_full_name is None:
                                example_full_name = full_name_i if full_name_i == full_name_j else f"{full_name_i} / {full_name_j}"

                    # 写入上三角。
                    sim_mat[i, j] = cos_sim
                    # 同步写入下三角，保证矩阵对称。
                    sim_mat[j, i] = cos_sim

            # 当前逻辑层相似度矩阵计算完毕，按层名保存。
            sim_matrices[group["name"]] = sim_mat
            # 保存来源统计，后续如果需要调试也可以读取。
            sim_sources[group["name"]] = source_counter
            # 当前层至少一个 pair 使用低秩 V，就明确打印 V 参数名。
            if source_counter["v"] > 0:
                print(
                    f"  -> 第 {layer_idx + 1:02d} 层 {group['name']}: "
                    f"使用低秩 V 相似度，V参数={example_v_name} | "
                    f"V_pairs={source_counter['v']} full_fallback_pairs={source_counter['full']} missing={source_counter['missing']}"
                )
                # 记录这个逻辑层确实用到了 V。
                sim_debug_printed.add(group["name"])
            # 当前层没有 V，但有 full fallback，就打印全秩代表参数名。
            elif source_counter["full"] > 0:
                print(
                    f"  -> 第 {layer_idx + 1:02d} 层 {group['name']}: "
                    f"未找到低秩 V，退回全秩相似度，参数={example_full_name} | "
                    f"full_pairs={source_counter['full']} missing={source_counter['missing']}"
                )
            # 连 full fallback 都没有，说明当前层命名解析有问题。
            else:
                print(
                    f"  -> 第 {layer_idx + 1:02d} 层 {group['name']}: "
                    f"未找到可用于相似度的参数 | missing={source_counter['missing']}"
                )

        # 正常低秩 ResNet18 应该是 16/18：16 个 block conv 用 V，首层和 head 用全秩。
        print(f"✅ ResNet18 低秩 V 相似度层数: {len(sim_debug_printed)} / {num_res_layers}")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        sim_matrix_time = time.time() - sim_matrix_start

        # 每个逻辑层保存一个完整客户端数 x 完整客户端数的权重矩阵，用于后续打印。
        global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_res_layers)]
        personal_weight_time = 0.0
        param_aggregate_time = 0.0

        # 第三阶段：为每个目标客户端生成一个专属全秩模型，再分解回该客户端 rank。
        for i, target_cid in enumerate(self.uploaded_ids):
            # 从服务器通用全秩模型拿一个干净壳子作为个性化模型。
            personalized_full_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            # 如果壳子意外处于低秩形态，先恢复成全秩。
            self._recover_if_needed(personalized_full_model)
            # recover 后重新移动到设备。
            personalized_full_model = personalized_full_model.to(self.device)

            # 目标模型参数先清零，随后按聚合权重加权写入。
            for param in personalized_full_model.parameters():
                param.data.zero_()

            # 目标模型全秩参数字典，便于按名字聚合。
            target_full_param_dict = dict(personalized_full_model.named_parameters())
            # 记录哪些参数已经被 18 个逻辑层覆盖。
            covered_param_names = set()

            # 按 ResNet18 逻辑深度逐层计算个性化聚合权重。
            for layer_idx, group in enumerate(res_layers):
                # 越深层越偏向相似度个性化权重，但最深层也固定保留 30% AVG fallback。
                depth_ratio = 0.7 * (layer_idx + 1) / num_res_layers
                # 取当前目标客户端 i 与所有参与客户端的当前层相似度。
                layer_sims = sim_matrices[group["name"]][i]

                personal_weight_start = time.time()
                # logits 将被 softmax 成当前层的个性化权重。
                logits = []
                # 遍历所有上传客户端 j，计算 j 对目标客户端 i 的贡献 logit。
                for j in range(num_participants):
                    # 当前层 i-j 的 cosine 相似度。
                    cos_sim = layer_sims[j]
                    # -9999 表示该 pair 没有可用相似度，softmax 后基本为 0。
                    if cos_sim.item() == -9999.0:
                        logits.append(torch.tensor(-9999.0, device=self.device))
                        continue

                    # 个性化分支只负责客户端关系，不再混入样本量缩放或 self-bias。
                    logit_j = cos_sim / tau
                    # 收集当前上传客户端的 logit。
                    logits.append(logit_j)

                # 变成 tensor 后才能 softmax。
                logits_tensor = torch.stack(logits)
                # softmax 得到参与客户端维度上的层级个性化权重。
                layer_weights = torch.nn.functional.softmax(logits_tensor, dim=0).cpu().numpy()

                # aligned_weights 对齐到全体客户端 id，未参与客户端位置保持 0。
                aligned_weights = np.zeros(num_total_clients)
                # 将参与客户端顺序的权重写到真实 client id 位置。
                for j, upload_cid in enumerate(self.uploaded_ids):
                    # 和原 CNN 聚合保持一致：浅层偏样本量全局权重，深层偏相似度个性化权重。
                    final_w = (1.0 - depth_ratio) * fallback_weights[j] + depth_ratio * layer_weights[j]
                    # 写入完整客户端矩阵对应列。
                    aligned_weights[upload_cid] = final_w
                personal_weight_time += time.time() - personal_weight_start

                param_aggregate_start = time.time()
                # 找到当前逻辑层包含的所有全秩参数，比如 conv/bn/downsample 或 head。
                param_names_in_group = [
                    # 这里遍历目标模型参数名。
                    name for name in target_full_param_dict.keys()
                    # 只保留属于当前 group 的参数。
                    if match_group(name, group)
                ]

                # 对当前逻辑层内所有参数复用同一套层级聚合权重。
                for param_name in param_names_in_group:
                    # 目标参数引用，后续直接累加到它的 data。
                    target_param = target_full_param_dict[param_name]
                    # 标记该参数已经被 18 层规则覆盖。
                    covered_param_names.add(param_name)
                    # 对每个上传客户端按当前层权重累加全秩参数。
                    for j, upload_cid in enumerate(self.uploaded_ids):
                        # 取真实 client id 对应的最终权重。
                        final_w = aligned_weights[upload_cid]
                        # 权重大于 0 时才累加，减少无意义操作。
                        if final_w > 0:
                            # 从缓存的全秩参数字典读取客户端 j 的同名参数。
                            target_param.data += uploaded_full_param_dicts[j][param_name].data * final_w
                param_aggregate_time += time.time() - param_aggregate_start

                # 保存当前目标客户端在当前层的完整权重行，供 print_row_weights 打印。
                global_weight_matrices[layer_idx][target_cid] = aligned_weights

            # 找出没有被 18 层规则覆盖的参数，正常情况下应为空或很少。
            uncovered_param_names = [
                # 遍历目标模型所有参数名。
                name for name in target_full_param_dict.keys()
                # 未被 covered_param_names 标记的参数需要兜底处理。
                if name not in covered_param_names
            ]
            # 如果有未覆盖参数，就用样本量权重进行普通 FedAvg 兜底，避免参数保持 0。
            if uncovered_param_names:
                print(f"⚠️ 目标客户端 {target_cid} 有 {len(uncovered_param_names)} 个参数未归入18层，使用样本量权重兜底聚合。")
                param_aggregate_start = time.time()
                # 逐个未覆盖参数做兜底聚合。
                for param_name in uncovered_param_names:
                    # 目标参数引用。
                    target_param = target_full_param_dict[param_name]
                    # 遍历参与客户端，用 fallback_weights 聚合。
                    for j in range(num_participants):
                        # 这里不做个性化，只做普通样本量加权。
                        target_param.data += uploaded_full_param_dicts[j][param_name].data * fallback_weights[j]
                param_aggregate_time += time.time() - param_aggregate_start

            # 专属模型已经在全秩空间聚合完毕；保存时保持全秩，兼容 clientCLIP.set_parameters() 的原接口。
            personalized_full_model = personalized_full_model.to(self.device)
            # 保存给目标客户端下轮接收，客户端会自己调用 decom_larger_model(model.ratio_LR)。
            save_start = time.time()
            save_item(personalized_full_model, self.role, f'model_{target_cid}', self.save_folder_name)
            param_aggregate_time += time.time() - save_start

        # 打印每个 ResNet 逻辑层的个性化聚合权重矩阵。
        weight_print_start = time.time()
        for layer_idx in range(num_res_layers):
            self.print_row_weights(global_weight_matrices[layer_idx], layer_idx=layer_idx)
        weight_print_time = time.time() - weight_print_start
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_time = time.time() - aggregate_total_start
        print(
            f"⏱️ ResNet18 聚合耗时拆分: "
            f"model_prepare={model_prepare_time:.3f}s | "
            f"sim_matrix={sim_matrix_time:.3f}s | "
            f"personal_weight={personal_weight_time:.3f}s | "
            f"param_aggregate_save={param_aggregate_time:.3f}s | "
            f"weight_print={weight_print_time:.3f}s | "
            f"total_inside={aggregate_total_time:.3f}s"
        )

    def aggregate_parameters_v_svd_drop(self):
        # The simplified branch no longer exposes the legacy V/drop aggregation.
        return self.aggregate_parameters_full_w()

        assert (len(self.uploaded_ids) > 0)
        print("🚀 开始聚合 (极速优化版：相似度矩阵预计算 + 对称性优化)")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_start = time.time()
        model_prepare_start = time.time()
        
        self.uploaded_base_model = []   # 保存低秩分解后的原始版本
        delta_params_per_client = []    # 保存客户端在低秩空间内的参数变化量
        
        # ============================================================================
        # 🟢 第一阶段：提取低秩 Delta 用于计算相似度，并准备全秩模型用于最终聚合
        # ============================================================================
        uploaded_full_models = []       
        uploaded_full_param_dicts = []  # 提速：缓存全秩字典，避免后续循环中重复 dict()
        
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name) 
            model = copy.deepcopy(client_model).to(self.device)                     
            
            # 1. 提取用于计算相似度的低秩 Delta
            old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)   
            if old_start_model is None:         
                old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
                old_start_model.decom_larger_model(model.ratio_LR)
            old_start_model = old_start_model.to(self.device)
            
            client_raw_deltas = {}              
            for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
                client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
                
            delta_params_per_client.append(client_raw_deltas)   
            self.uploaded_base_model.append(model)
            
            # 2. 将当前客户端模型在内存中还原为全秩大矩阵备用
            full_m = copy.deepcopy(model).to(self.device)
            full_m.recover_larger_model()
            full_m = full_m.to(self.device)
            uploaded_full_models.append(full_m)
            uploaded_full_param_dicts.append(dict(full_m.named_parameters())) # 缓存参数字典

        # 兜底权重与数据规模放缩计算
        fallback_weights = self.uploaded_weights            
        num_participants = len(self.uploaded_ids)
        data_scales = [w * num_participants for w in fallback_weights]

        # ============================================================================
        # 🟡 第二阶段：网络架构解析与【对称相似度矩阵】预计算
        # ============================================================================
        target_named_params = list(self.uploaded_base_model[0].named_parameters())
        
        # 提取真实的物理逻辑层名称前缀
        logical_layers = [] 
        for name, _ in target_named_params:
            parent_name = name.rsplit('.', 1)[0]
            if parent_name not in logical_layers:
                logical_layers.append(parent_name)
                
        num_logical_layers = len(logical_layers) 
        num_total_tensors_full_rank = len(list(uploaded_full_models[0].named_parameters()))
        print(f"🚀 执行全秩重构聚合 | 逻辑层数: {num_logical_layers} | 全秩总张量数: {num_total_tensors_full_rank}")
        
        tau = self.args.aggregate_tau if self.args.aggregate_tau > 0 else 1.0
        
        num_total_clients = len(self.clients) 
        
        # 建立 逻辑层 -> 锚点名称 的映射
        layer_anchors = {}
        for logical_layer_name in logical_layers:
            tensors_in_layer_low_rank = [name for name, _ in target_named_params if name.rsplit('.', 1)[0] == logical_layer_name]
            anchor_name = None
            for name in tensors_in_layer_low_rank:
                if name.endswith('conv_v') or name.endswith('weight_v'):
                    anchor_name = name
                    break
            if anchor_name is None:
                for name in tensors_in_layer_low_rank:
                    if name.endswith('.weight'): 
                        anchor_name = name
                        break
            layer_anchors[logical_layer_name] = anchor_name if anchor_name else tensors_in_layer_low_rank[0]

        # 🚀 核心优化：预计算对称相似度矩阵
        # sim_matrices[逻辑层名] = N x N 的相似度张量
        sim_matrices = {}
        print("🧮 正在利用对称性计算相似度矩阵...")
        sim_matrix_start = time.time()
        for logical_layer_name, anchor_name in layer_anchors.items():
            sim_mat = torch.zeros((num_participants, num_participants), device=self.device)
            for i in range(num_participants):
                # 利用对称性，j 直接从 i 开始，计算量减半！
                for j in range(i, num_participants):
                    if anchor_name not in delta_params_per_client[i] or anchor_name not in delta_params_per_client[j]:
                        sim_mat[i, j] = sim_mat[j, i] = -9999.0
                        continue
                        
                    raw_i = delta_params_per_client[i][anchor_name]
                    raw_j = delta_params_per_client[j][anchor_name]
                        
                    slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(raw_i.shape, raw_j.shape)) 
                    trunc_i = raw_i[slices].contiguous().view(-1)
                    trunc_j = raw_j[slices].contiguous().view(-1)
                    
                    cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0) if trunc_i.numel() > 0 else torch.tensor(0.0).to(self.device)
                    
                    sim_mat[i, j] = cos_sim
                    sim_mat[j, i] = cos_sim  # A对B的相似度 == B对A的相似度
            sim_matrices[logical_layer_name] = sim_mat
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        sim_matrix_time = time.time() - sim_matrix_start

        # ============================================================================
        # 🔴 第三阶段：基于相似度矩阵计算权重，并在全秩空间内直接相加
        # ============================================================================
        global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_total_tensors_full_rank)]
        personal_weight_time = 0.0
        param_aggregate_time = 0.0
        
        for i, target_cid in enumerate(self.uploaded_ids):
            personalized_full_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            personalized_full_model.recover_larger_model() 
            personalized_full_model = personalized_full_model.to(self.device)
                
            for param in personalized_full_model.parameters():    
                param.data.zero_()
                
            tensor_idx = 0 
            target_full_param_dict = dict(personalized_full_model.named_parameters())

            for logical_layer_idx, logical_layer_name in enumerate(logical_layers):
                depth_ratio = 0.7 * (logical_layer_idx + 1) / num_logical_layers

                # 🚀 极速获取权重：直接从预计算矩阵中读取该层的相似度，偏置 (bias) 会自然复用这一权重
                layer_sims = sim_matrices[logical_layer_name][i]
                
                personal_weight_start = time.time()
                logits = []
                for j in range(num_participants):
                    cos_sim = layer_sims[j].item()
                    if cos_sim == -9999.0:
                        logits.append(torch.tensor(-9999.0).to(self.device))
                        continue
                        
                    logit_j = cos_sim / tau
                    logits.append(logit_j)
                    
                logits_tensor = torch.tensor(logits, device=self.device)
                layer_weights = torch.nn.functional.softmax(logits_tensor, dim=0).cpu().numpy()
                
                aligned_weights = np.zeros(num_total_clients)
                for j, upload_cid in enumerate(self.uploaded_ids):
                    final_w = (1.0 - depth_ratio) * fallback_weights[j] + depth_ratio * layer_weights[j] 
                    aligned_weights[upload_cid] = final_w
                personal_weight_time += time.time() - personal_weight_start

                # ================= 暴力相加：全层所有组件复用一套聚合权重 =================
                param_aggregate_start = time.time()
                tensors_in_layer_full_rank = [name for name, _ in personalized_full_model.named_parameters() if name.rsplit('.', 1)[0] == logical_layer_name]
                
                for param_name in tensors_in_layer_full_rank:
                    target_param = target_full_param_dict[param_name]
                    
                    is_u_matrix = param_name.endswith('conv_u') or param_name.endswith('weight_u')

                    for j, upload_cid in enumerate(self.uploaded_ids):
                        if is_u_matrix:
                            final_w = fallback_weights[j] 
                        # 否则 (V矩阵、Bias等)，使用计算出的个性化相似度权重，保留个性化特征
                        else:
                            final_w = aligned_weights[upload_cid]

                        if final_w > 0:
                            # 🚀 提速：从外部预处理好的字典中直接读取，消除内循环创建字典的开销
                            client_j_data = uploaded_full_param_dicts[j][param_name].data  
                            target_param.data += client_j_data * final_w
                    
                    global_weight_matrices[tensor_idx][target_cid] = aligned_weights
                    tensor_idx += 1  
                param_aggregate_time += time.time() - param_aggregate_start

            # SVD 降维，然后再下发保存
            param_aggregate_start = time.time()
            personalized_full_model.decom_larger_model(self.uploaded_base_model[i].ratio_LR)
            personalized_full_model = personalized_full_model.to(self.device)
            save_item(personalized_full_model, self.role, f'model_{target_cid}', self.save_folder_name)
            param_aggregate_time += time.time() - param_aggregate_start
                    
        weight_print_start = time.time()
        for idx in range(num_total_tensors_full_rank):
            self.print_row_weights(global_weight_matrices[idx], layer_idx=idx)
        weight_print_time = time.time() - weight_print_start
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_time = time.time() - aggregate_total_start
        print(
            f"⏱️ CNN 聚合耗时拆分: "
            f"model_prepare={model_prepare_time:.3f}s | "
            f"sim_matrix={sim_matrix_time:.3f}s | "
            f"personal_weight={personal_weight_time:.3f}s | "
            f"param_aggregate_save={param_aggregate_time:.3f}s | "
            f"weight_print={weight_print_time:.3f}s | "
            f"total_inside={aggregate_total_time:.3f}s"
        )



    def aggregate_parameters_v_svd(self):
        # Backward-compatible entry point; active similarity is full-W delta based.
        return self.aggregate_parameters_full_w()

        assert (len(self.uploaded_ids) > 0)
        print("🚀 开始聚合 (极速优化版：相似度矩阵预计算 + 对称性优化)")
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_start = time.time()
        model_prepare_start = time.time()

        def sync_prepare_step():
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)

        prepare_times = {
            "load_client": 0.0,
            "copy_client_to_device": 0.0,
            "load_old": 0.0,
            "old_to_device": 0.0,
            "old_decompose": 0.0,
            "low_rank_delta": 0.0,
            "full_current_copy": 0.0,
            "full_current_recover": 0.0,
            "full_current_to_device": 0.0,
            "full_current_dict": 0.0,
        }
        low_rank_cache_hits = 0
        low_rank_cache_misses = 0
        
        self.uploaded_base_model = []   # 保存低秩分解后的原始版本
        delta_params_per_client = []    # 保存客户端在低秩空间内的参数变化量
        
        # ============================================================================
        # 🟢 第一阶段：提取低秩 Delta 用于计算相似度，并准备全秩模型用于最终聚合
        # ============================================================================
        uploaded_full_models = []       
        uploaded_full_param_dicts = []  # 提速：缓存全秩字典，避免后续循环中重复 dict()
        
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            step_start = time.time()
            client_model = load_item(client.role, 'model', client.save_folder_name) 
            prepare_times["load_client"] += time.time() - step_start
            step_start = time.time()
            model = copy.deepcopy(client_model).to(self.device)                     
            sync_prepare_step()
            prepare_times["copy_client_to_device"] += time.time() - step_start
            
            # 1. 提取用于计算相似度的低秩 Delta
            step_start = time.time()
            old_start_model = load_item(self.role, f'model_{cid}', self._low_rank_start_folder())
            if old_start_model is not None:
                low_rank_cache_hits += 1
            else:
                low_rank_cache_misses += 1
                old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            if old_start_model is None:
                old_start_model = load_item(self.role, 'model', self.save_folder_name)
            prepare_times["load_old"] += time.time() - step_start
            step_start = time.time()
            old_start_model = old_start_model.to(self.device)
            sync_prepare_step()
            prepare_times["old_to_device"] += time.time() - step_start
            if not self._has_low_rank_params(old_start_model):
                step_start = time.time()
                old_start_model.decom_larger_model(model.ratio_LR)
                old_start_model = old_start_model.to(self.device)
                sync_prepare_step()
                prepare_times["old_decompose"] += time.time() - step_start
            
            step_start = time.time()
            client_raw_deltas = {}              
            for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
                client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
            sync_prepare_step()
            prepare_times["low_rank_delta"] += time.time() - step_start
                
            delta_params_per_client.append(client_raw_deltas)   
            self.uploaded_base_model.append(model)
            
            # 2. 将当前客户端模型在内存中还原为全秩大矩阵备用
            step_start = time.time()
            full_m = copy.deepcopy(model).to(self.device)
            sync_prepare_step()
            prepare_times["full_current_copy"] += time.time() - step_start
            step_start = time.time()
            full_m.recover_larger_model()
            sync_prepare_step()
            prepare_times["full_current_recover"] += time.time() - step_start
            step_start = time.time()
            full_m = full_m.to(self.device)
            sync_prepare_step()
            prepare_times["full_current_to_device"] += time.time() - step_start
            uploaded_full_models.append(full_m)
            step_start = time.time()
            uploaded_full_param_dicts.append(dict(full_m.named_parameters())) # 缓存参数字典
            prepare_times["full_current_dict"] += time.time() - step_start

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        model_prepare_time = time.time() - model_prepare_start
        prepare_accounted_time = sum(prepare_times.values())
        print(
            f"⏱️ CNN model_prepare 细分: "
            f"low_rank_cache_hit={low_rank_cache_hits} | "
            f"low_rank_cache_miss={low_rank_cache_misses} | "
            f"load_client={prepare_times['load_client']:.3f}s | "
            f"copy_client_to_device={prepare_times['copy_client_to_device']:.3f}s | "
            f"load_old={prepare_times['load_old']:.3f}s | "
            f"old_to_device={prepare_times['old_to_device']:.3f}s | "
            f"old_decompose={prepare_times['old_decompose']:.3f}s | "
            f"low_rank_delta={prepare_times['low_rank_delta']:.3f}s | "
            f"full_current_copy={prepare_times['full_current_copy']:.3f}s | "
            f"full_current_recover={prepare_times['full_current_recover']:.3f}s | "
            f"full_current_to_device={prepare_times['full_current_to_device']:.3f}s | "
            f"full_current_dict={prepare_times['full_current_dict']:.3f}s | "
            f"unaccounted={max(model_prepare_time - prepare_accounted_time, 0.0):.3f}s | "
            f"total={model_prepare_time:.3f}s"
        )

        # 兜底权重与数据规模放缩计算
        fallback_weights = self.uploaded_weights            
        num_participants = len(self.uploaded_ids)
        data_scales = [w * num_participants for w in fallback_weights]

        # ============================================================================
        # 🟡 第二阶段：网络架构解析与【对称相似度矩阵】预计算
        # ============================================================================
        target_named_params = list(self.uploaded_base_model[0].named_parameters())
        
        # 提取真实的物理逻辑层名称前缀
        logical_layers = [] 
        for name, _ in target_named_params:
            parent_name = name.rsplit('.', 1)[0]
            if parent_name not in logical_layers:
                logical_layers.append(parent_name)
                
        num_logical_layers = len(logical_layers) 
        num_total_tensors_full_rank = len(list(uploaded_full_models[0].named_parameters()))
        print(f"🚀 执行全秩重构聚合 | 逻辑层数: {num_logical_layers} | 全秩总张量数: {num_total_tensors_full_rank}")
        
        tau = self.args.aggregate_tau if self.args.aggregate_tau > 0 else 1.0
        
        num_total_clients = len(self.clients) 
        
        # 建立 逻辑层 -> 锚点名称 的映射
        layer_anchors = {}
        for logical_layer_name in logical_layers:
            tensors_in_layer_low_rank = [name for name, _ in target_named_params if name.rsplit('.', 1)[0] == logical_layer_name]
            anchor_name = None
            for name in tensors_in_layer_low_rank:
                if name.endswith('conv_v') or name.endswith('weight_v'):
                    anchor_name = name
                    break
            if anchor_name is None:
                for name in tensors_in_layer_low_rank:
                    if name.endswith('.weight'): 
                        anchor_name = name
                        break
            layer_anchors[logical_layer_name] = anchor_name if anchor_name else tensors_in_layer_low_rank[0]

        # 🚀 核心优化：预计算对称相似度矩阵
        # sim_matrices[逻辑层名] = N x N 的相似度张量
        sim_matrices = {}
        print("🧮 正在利用对称性计算相似度矩阵...")
        sim_matrix_start = time.time()
        for logical_layer_name, anchor_name in layer_anchors.items():
            sim_mat = torch.zeros((num_participants, num_participants), device=self.device)
            for i in range(num_participants):
                # 利用对称性，j 直接从 i 开始，计算量减半！
                for j in range(i, num_participants):
                    if anchor_name not in delta_params_per_client[i] or anchor_name not in delta_params_per_client[j]:
                        sim_mat[i, j] = sim_mat[j, i] = -9999.0
                        continue
                        
                    raw_i = delta_params_per_client[i][anchor_name]
                    raw_j = delta_params_per_client[j][anchor_name]
                        
                    slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(raw_i.shape, raw_j.shape)) 
                    trunc_i = raw_i[slices].contiguous().view(-1)
                    trunc_j = raw_j[slices].contiguous().view(-1)
                    
                    cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0) if trunc_i.numel() > 0 else torch.tensor(0.0).to(self.device)
                    
                    sim_mat[i, j] = cos_sim
                    sim_mat[j, i] = cos_sim  # A对B的相似度 == B对A的相似度
            sim_matrices[logical_layer_name] = sim_mat
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        sim_matrix_time = time.time() - sim_matrix_start

        # ============================================================================
        # 🔴 第三阶段：基于相似度矩阵计算权重，并在全秩空间内直接相加
        # ============================================================================
        global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_total_tensors_full_rank)]
        personal_weight_time = 0.0
        param_aggregate_time = 0.0
        
        for i, target_cid in enumerate(self.uploaded_ids):
            personalized_full_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            personalized_full_model.recover_larger_model() 
            personalized_full_model = personalized_full_model.to(self.device)
                
            for param in personalized_full_model.parameters():    
                param.data.zero_()
                
            tensor_idx = 0 
            target_full_param_dict = dict(personalized_full_model.named_parameters())

            for logical_layer_idx, logical_layer_name in enumerate(logical_layers):
                depth_ratio = 0.7 * (logical_layer_idx + 1) / num_logical_layers

                # 🚀 极速获取权重：直接从预计算矩阵中读取该层的相似度，偏置 (bias) 会自然复用这一权重
                layer_sims = sim_matrices[logical_layer_name][i]
                
                personal_weight_start = time.time()
                logits = []
                for j in range(num_participants):
                    cos_sim = layer_sims[j].item()
                    if cos_sim == -9999.0:
                        logits.append(torch.tensor(-9999.0).to(self.device))
                        continue
                        
                    logit_j = cos_sim / tau
                    logits.append(logit_j)
                    
                logits_tensor = torch.tensor(logits, device=self.device)
                layer_weights = torch.nn.functional.softmax(logits_tensor, dim=0).cpu().numpy()
                
                aligned_weights = np.zeros(num_total_clients)
                for j, upload_cid in enumerate(self.uploaded_ids):
                    final_w = (1.0 - depth_ratio) * fallback_weights[j] + depth_ratio * layer_weights[j] 
                    aligned_weights[upload_cid] = final_w
                personal_weight_time += time.time() - personal_weight_start

                # ================= 暴力相加：全层所有组件复用一套聚合权重 =================
                param_aggregate_start = time.time()
                tensors_in_layer_full_rank = [name for name, _ in personalized_full_model.named_parameters() if name.rsplit('.', 1)[0] == logical_layer_name]
                
                for param_name in tensors_in_layer_full_rank:
                    target_param = target_full_param_dict[param_name]
                    
                    for j, upload_cid in enumerate(self.uploaded_ids):
                        final_w = aligned_weights[upload_cid]
                        if final_w > 0:
                            # 🚀 提速：从外部预处理好的字典中直接读取，消除内循环创建字典的开销
                            client_j_data = uploaded_full_param_dicts[j][param_name].data  
                            target_param.data += client_j_data * final_w
                    
                    global_weight_matrices[tensor_idx][target_cid] = aligned_weights
                    tensor_idx += 1  
                param_aggregate_time += time.time() - param_aggregate_start

            # SVD 降维，然后再下发保存
            param_aggregate_start = time.time()
            personalized_full_model.decom_larger_model(self.uploaded_base_model[i].ratio_LR)
            personalized_full_model = personalized_full_model.to(self.device)
            save_item(personalized_full_model, self.role, f'model_{target_cid}', self.save_folder_name)
            param_aggregate_time += time.time() - param_aggregate_start
                    
        weight_print_start = time.time()
        for idx in range(num_total_tensors_full_rank):
            self.print_row_weights(global_weight_matrices[idx], layer_idx=idx)
        weight_print_time = time.time() - weight_print_start
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        aggregate_total_time = time.time() - aggregate_total_start
        print(
            f"⏱️ CNN 聚合耗时拆分: "
            f"model_prepare={model_prepare_time:.3f}s | "
            f"sim_matrix={sim_matrix_time:.3f}s | "
            f"personal_weight={personal_weight_time:.3f}s | "
            f"param_aggregate_save={param_aggregate_time:.3f}s | "
            f"weight_print={weight_print_time:.3f}s | "
            f"total_inside={aggregate_total_time:.3f}s"
        )

    # def aggregate_parameters_v(self):
    #     assert (len(self.uploaded_ids) > 0)
    #     print("开始聚合")
    #     self.uploaded_base_model = []   # 保存低秩分解后的版本
    #     delta_params_per_client = []    # 列表，每个元素是一个字典 {参数名: Δ张量}，保存客户端在低秩空间内的参数变化量
        
    #     # ============================================================================
    #     # 🟢 第一阶段：参数提取与 SVD 状态对齐
    #     # ============================================================================
    #     for cid in self.uploaded_ids:
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name) # 加载模型参数（低秩），本地训练完成的模型
    #         model = copy.deepcopy(client_model).to(self.device)                     # 深拷贝模型参数（低秩）
            
    #         old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)   # 上一轮该客户端的模型，server开头的文件中，低秩
            
    #         # 冷启动兜底：当场分解全局模型，对齐目标架构与维度
    #         if old_start_model is None:         # 如果上一轮为空，说明是第一轮，则需要将全局模型分解到低秩状态
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #             old_start_model.decom_larger_model(model.ratio_LR)
    #         old_start_model = old_start_model.to(self.device)
            
    #         # 计算每一个张量的原始更新量 (Delta)
    #         client_raw_deltas = {}              
    #         for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
    #             client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
                
    #         delta_params_per_client.append(client_raw_deltas)   # 将其放到大表里面
    #         self.uploaded_base_model.append(model)

    #     # 兜底权重与数据规模放缩计算
    #     fallback_weights = self.uploaded_weights            

    #     num_participants = len(self.uploaded_ids)
    #     data_scales = [w * num_participants for w in fallback_weights]  # 自身数据集在参与客户端中所占比重，乘以参与数量，起到放大和缩小的作用，而非仅仅缩小

    #     # ============================================================================
    #     # 🟡 第二阶段：网络架构解析 (解决 Bias 与 分解层深度扭曲问题)
    #     # ============================================================================
    #     target_named_params = list(self.uploaded_base_model[0].named_parameters())  # 这里的层是将UVbias看成三层的
    #     num_total_tensors = len(target_named_params)
        
    #     # 提取真实的物理逻辑层名称前缀 (实际为['conv1', 'conv2', 'fc1', 'fc2', 'fc3'])
    #     logical_layers = [] 
    #     for name, _ in target_named_params:
    #         parent_name = name.rsplit('.', 1)[0]
    #         if parent_name not in logical_layers:
    #             logical_layers.append(parent_name)
                
    #     num_logical_layers = len(logical_layers) 
    #     print(f"🚀 执行子空间截断聚合 | 逻辑层数: {num_logical_layers} | 总张量数: {num_total_tensors}")
        
    #     tau = self.args.aggregate_tau
    #     power = self.args.aggregate_power
    #     gamma = self.args.aggregate_gamma
        
    #     num_total_clients = len(self.clients) 
    #     # 热力图矩阵依然按张量总数保留，确保精细化打印
    #     global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_total_tensors)]

    #     # ============================================================================
    #     # 🔴 第三阶段：按“物理逻辑层”逐层深入聚合
    #     # ============================================================================
    #     for i, target_cid in enumerate(self.uploaded_ids):
    #         scale_i = data_scales[i]
            
    #         # 1. 获取目标模型空壳并清零
    #         personalized_global_model = load_item(self.role, f'model_{target_cid}', self.save_folder_name)  
    #         if personalized_global_model is None:
    #             personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #             personalized_global_model.decom_larger_model(self.uploaded_base_model[i].ratio_LR)
    #         personalized_global_model = personalized_global_model.to(self.device)
                
    #         for param in personalized_global_model.parameters():    
    #             param.data.zero_()
                
    #         tensor_idx = 0  # 全局张量计数器，用于保存热力图矩阵
    #         target_model_param_dict = dict(personalized_global_model.named_parameters()) # 包含13层

    #         # 🌟 外层循环：遍历真实的物理层 (如 conv1, conv2, fc1...)
    #         for logical_layer_idx, logical_layer_name in enumerate(logical_layers):
                
    #             depth_ratio = ((logical_layer_idx + 1) / num_logical_layers) ** power
    #             self_bias = depth_ratio * gamma * (scale_i ** 1)

    #             tensors_in_layer = [name for name, _ in target_named_params if name.rsplit('.', 1)[0] == logical_layer_name]

    #             # ================= 🚀 终极修复：寻找全层唯一的特征锚点 (Anchor) =================
    #             layer_anchor_name = None
                
    #             # 优先级 1: 如果是 SVD 分解层，锚点绝对是 V 矩阵 (特征子空间基底)
    #             for name in tensors_in_layer:
    #                 if name.endswith('conv_v') or name.endswith('weight_v'):
    #                     layer_anchor_name = name
    #                     break
                        
    #             # 优先级 2: 如果是全秩层 (未分解)，锚点则是原生的 Weight 矩阵
    #             if layer_anchor_name is None:
    #                 for name in tensors_in_layer:
    #                     if name.endswith('.weight'): # 涵盖 conv1.weight, fc3.weight 等
    #                         layer_anchor_name = name
    #                         break
    #             # ==============================================================================

    #             # 🌟 内层循环：遍历该物理层内部的具体组件 (U, V, Weight, Bias...)
    #             for param_name in tensors_in_layer:
    #                 target_param = target_model_param_dict[param_name]
    #                 logits = [] 
                    
    #                 is_v_matrix = param_name.endswith('conv_v') or param_name.endswith('weight_v')
    #                 is_u_matrix = param_name.endswith('conv_u') or param_name.endswith('weight_u')
                    
    #                 # 当前组件统一认祖归宗，使用该层的“大哥”作为相似度计算的锚点
    #                 # 如果由于某种极其罕见的结构连 Weight 都没有，才退回自身
    #                 target_anchor_name = layer_anchor_name if layer_anchor_name else param_name
                    
    #                 for j in range(len(self.uploaded_ids)):
    #                     if param_name not in delta_params_per_client[j]:    # 架构不一样才会执行，这里跳过
    #                         logits.append(torch.tensor(-9999.0).to(self.device))
    #                         continue
                            
    #                     # =================  安全且公平的相似度计算 =================
    #                     # 统统使用锚点的 Delta 提取特征余弦相似度！
    #                     # (废弃了 is_v_anchor 判断，因为全秩 Weight 也可以作为完美锚点)
    #                     if target_anchor_name in delta_params_per_client[i] and target_anchor_name in delta_params_per_client[j]:
    #                         raw_i = delta_params_per_client[i][target_anchor_name]
    #                         raw_j = delta_params_per_client[j][target_anchor_name]
    #                     else:
    #                         # 极少见的兜底
    #                         raw_i = delta_params_per_client[i][param_name]
    #                         raw_j = delta_params_per_client[j][param_name]
                            
    #                     # 找到两者的公共最小维度进行对齐并截断
    #                     # 神奇之处：如果锚点是全秩 Weight，这里的 min() 会自然取满全尺寸，等价于普通切片！
    #                     slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(raw_i.shape, raw_j.shape)) # 取较小截断
    #                     trunc_i = raw_i[slices].contiguous().view(-1)
    #                     trunc_j = raw_j[slices].contiguous().view(-1)
                        
    #                     cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0) if trunc_i.numel() > 0 else torch.tensor(0.0).to(self.device)


    #                     # ===============================================================
                        
    #                     safe_scale_j = max(data_scales[j], 1e-4)


    #                     data_factor = safe_scale_j
    #                     logit_j = (cos_sim * data_factor) / tau

    #                     print(f"客户端{self.uploaded_ids[j]}对客户端{target_cid} : self_bias:{self_bias},safe_scale_j:{safe_scale_j},cos_sim:{cos_sim},data_factor:{data_factor},logit_j:{logit_j}")

    #                     if i == j:
    #                         logit_j += self_bias 
                            
    #                     logits.append(logit_j)
                        
    #                 # ... 后续的 Softmax、以及按照 final_w 执行参数切片拼接 (拼接逻辑不变，依然认 U/V 切片) ...
                    
    #                 logits = torch.stack(logits) 
    #                 layer_weights = torch.nn.functional.softmax(logits, dim=0)
    #                 aligned_weights = np.zeros(num_total_clients)
                    
    #                 # 最终的参数物理切片拼接
    #                 for j, upload_cid in enumerate(self.uploaded_ids):
                            
    #                     final_w = (1.0 - depth_ratio) * fallback_weights[j] + depth_ratio * layer_weights[j].item() # 最终聚合权重
    #                     aligned_weights[upload_cid] = final_w
                        
    #                     if final_w > 0:
    #                         client_j_data = dict(self.uploaded_base_model[j].named_parameters())[param_name].data  # 取出j模型对应层的参数
                            
    #                         if is_v_matrix:                                                                     # 如果是低秩部分则要截断（U，V）
    #                             min_r = min(target_param.shape[0], client_j_data.shape[0])
    #                             target_param.data[:min_r, ...] += client_j_data[:min_r, ...] * final_w
    #                         elif is_u_matrix:
    #                             min_r = min(target_param.shape[1], client_j_data.shape[1])
    #                             target_param.data[:, :min_r, ...] += client_j_data[:, :min_r, ...] * final_w
    #                         else:
    #                             slices = tuple(slice(0, min(dim_t, dim_j)) for dim_t, dim_j in zip(target_param.shape, client_j_data.shape))
    #                             target_param.data[slices] += client_j_data[slices] * final_w
                    
    #                 global_weight_matrices[tensor_idx][target_cid] = aligned_weights
    #                 tensor_idx += 1  # 推进全局张量计数器

    #         save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
                    
    #     for idx in range(num_total_tensors):
    #         self.print_row_weights(global_weight_matrices[idx], layer_idx=idx)

    # def aggregate_parameters_v_svd(self):
    #     assert (len(self.uploaded_ids) > 0)
    #     print("🚀 开始聚合 (SVD全秩重构版：低秩算权重，全秩做相加)")
        
    #     self.uploaded_base_model = []   # 保存低秩分解后的原始版本
    #     delta_params_per_client = []    # 保存客户端在低秩空间内的参数变化量
        
    #     # ============================================================================
    #     # 🟢 第一阶段：提取低秩 Delta 用于计算相似度，并准备全秩模型用于最终聚合
    #     # ============================================================================
    #     uploaded_full_models = []       # ★ 新增：保存恢复成全秩后的客户端模型
        
    #     for cid in self.uploaded_ids:
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name) 
    #         model = copy.deepcopy(client_model).to(self.device)                     
            
    #         # 1. 提取用于计算相似度的低秩 Delta
    #         old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)   
    #         if old_start_model is None:         
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #             old_start_model.decom_larger_model(model.ratio_LR)
    #         old_start_model = old_start_model.to(self.device)
            
    #         client_raw_deltas = {}              
    #         for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
    #             client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
                
    #         delta_params_per_client.append(client_raw_deltas)   
    #         self.uploaded_base_model.append(model)
            
    #         # 2. ★ 新增：将当前客户端模型在内存中还原为全秩大矩阵，备用
    #         full_m = copy.deepcopy(model).to(self.device)
    #         full_m.recover_larger_model()
    #         full_m = full_m.to(self.device)
    #         uploaded_full_models.append(full_m)

    #     # 兜底权重与数据规模放缩计算
    #     fallback_weights = self.uploaded_weights            
    #     num_participants = len(self.uploaded_ids)
    #     data_scales = [w * num_participants for w in fallback_weights]

    #     # ============================================================================
    #     # 🟡 第二阶段：网络架构解析 (基于低秩提取逻辑层)
    #     # ============================================================================
    #     target_named_params = list(self.uploaded_base_model[0].named_parameters())
        
    #     # 提取真实的物理逻辑层名称前缀 (实际为['conv1', 'conv2', 'fc1', 'fc2', 'fc3'])
    #     logical_layers = [] 
    #     for name, _ in target_named_params:
    #         parent_name = name.rsplit('.', 1)[0]
    #         if parent_name not in logical_layers:
    #             logical_layers.append(parent_name)
                
    #     num_logical_layers = len(logical_layers) 
        
    #     # ★ 新增：热力图现在基于全秩的张量数量来构建
    #     num_total_tensors_full_rank = len(list(uploaded_full_models[0].named_parameters()))
    #     print(f"🚀 执行全秩重构聚合 | 逻辑层数: {num_logical_layers} | 全秩总张量数: {num_total_tensors_full_rank}")
        
    #     tau = self.args.aggregate_tau
    #     power = self.args.aggregate_power
    #     gamma = self.args.aggregate_gamma
        
    #     num_total_clients = len(self.clients) 
    #     global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_total_tensors_full_rank)]

    #     # ============================================================================
    #     # 🔴 第三阶段：按“物理逻辑层”计算相似度，并在全秩空间内相加
    #     # ============================================================================
    #     for i, target_cid in enumerate(self.uploaded_ids):
    #         scale_i = data_scales[i]
            
    #         # ★ 核心改变：我们拿一个完整的全局大模型（全秩）作为目标空壳
    #         personalized_full_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #         personalized_full_model.recover_larger_model() # 确保彻底是全秩
    #         personalized_full_model = personalized_full_model.to(self.device)
                
    #         for param in personalized_full_model.parameters():    
    #             param.data.zero_()
                
    #         tensor_idx = 0 
    #         target_full_param_dict = dict(personalized_full_model.named_parameters())

    #         # 🌟 外层循环：遍历真实的物理层 (如 conv1, conv2, fc1...)
    #         for logical_layer_idx, logical_layer_name in enumerate(logical_layers):
                
    #             depth_ratio = ((logical_layer_idx + 1) / num_logical_layers) ** power
    #             self_bias = depth_ratio * gamma * (scale_i ** 1)

    #             # 提取【低秩】结构下的张量名，用于寻找锚点算相似度
    #             tensors_in_layer_low_rank = [name for name, _ in target_named_params if name.rsplit('.', 1)[0] == logical_layer_name]

    #             # ================= 🚀 寻找全层唯一的特征锚点 (Anchor) =================
    #             layer_anchor_name = None
                
    #             # 优先级 1: 如果是 SVD 分解层，锚点绝对是 V 矩阵
    #             for name in tensors_in_layer_low_rank:
    #                 if name.endswith('conv_v') or name.endswith('weight_v'):
    #                     layer_anchor_name = name
    #                     break
                        
    #             # 优先级 2: 如果是全秩层，锚点是原生的 Weight
    #             if layer_anchor_name is None:
    #                 for name in tensors_in_layer_low_rank:
    #                     if name.endswith('.weight'): 
    #                         layer_anchor_name = name
    #                         break
                
    #             # 同一层共享同一个锚点的相似度计算结果
    #             target_anchor_name = layer_anchor_name if layer_anchor_name else tensors_in_layer_low_rank[0]
    #             # ==============================================================================

    #             logits = [] 
    #             for j in range(len(self.uploaded_ids)):
    #                 if target_anchor_name not in delta_params_per_client[j]:
    #                     logits.append(torch.tensor(-9999.0).to(self.device))
    #                     continue
                        
    #                 # 统统使用锚点的 Delta 提取特征余弦相似度
    #                 raw_i = delta_params_per_client[i][target_anchor_name]
    #                 raw_j = delta_params_per_client[j][target_anchor_name]
                        
    #                 # 找到两者的公共最小维度进行对齐并截断
    #                 slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(raw_i.shape, raw_j.shape)) 
    #                 trunc_i = raw_i[slices].contiguous().view(-1)
    #                 trunc_j = raw_j[slices].contiguous().view(-1)
                    
    #                 cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0) if trunc_i.numel() > 0 else torch.tensor(0.0).to(self.device)

    #                 safe_scale_j = max(data_scales[j], 1e-4)
    #                 data_factor = safe_scale_j ** (torch.sign(cos_sim).item() * 1)
    #                 logit_j = (cos_sim * data_factor) / tau

    #                 if i == j:
    #                     logit_j += self_bias 
                        
    #                 logits.append(logit_j)
                    
    #             logits = torch.stack(logits) 
    #             layer_weights = torch.nn.functional.softmax(logits, dim=0)
    #             aligned_weights = np.zeros(num_total_clients)
                
    #             # 算出本层的终极权重 final_w
    #             for j, upload_cid in enumerate(self.uploaded_ids):
    #                 final_w = (1.0 - depth_ratio) * fallback_weights[j] + depth_ratio * layer_weights[j].item() 
    #                 aligned_weights[upload_cid] = final_w

    #             # ================= ★ 核心改变：全秩矩阵无损相加 =================
    #             # 提取【全秩】结构下该层的所有张量名（此时只有 .weight 和 .bias）
    #             tensors_in_layer_full_rank = [name for name, _ in personalized_full_model.named_parameters() if name.rsplit('.', 1)[0] == logical_layer_name]
                
    #             for param_name in tensors_in_layer_full_rank:
    #                 target_param = target_full_param_dict[param_name]
                    
    #                 for j, upload_cid in enumerate(self.uploaded_ids):
    #                     final_w = aligned_weights[upload_cid]
    #                     if final_w > 0:
    #                         # 从之前准备好的全秩模型列表中提取数据
    #                         client_j_data = dict(uploaded_full_models[j].named_parameters())[param_name].data  
                            
    #                         # 因为都是重构好的全秩矩阵，形状绝对一模一样，直接暴力无缝相加！
    #                         target_param.data += client_j_data * final_w
                    
    #                 global_weight_matrices[tensor_idx][target_cid] = aligned_weights
    #                 tensor_idx += 1  

    #         # ★ 聚合完成后，这依然是一个庞大的全秩模型。
    #         # 为了适配客户端 i 本地的真实算力（容量），将其在服务端当场 SVD 降维，然后再下发保存！
    #         personalized_full_model.decom_larger_model(self.uploaded_base_model[i].ratio_LR)
    #         personalized_full_model = personalized_full_model.to(self.device)
    #         save_item(personalized_full_model, self.role, f'model_{target_cid}', self.save_folder_name)
                    
    #     for idx in range(num_total_tensors_full_rank):
    #         self.print_row_weights(global_weight_matrices[idx], layer_idx=idx)

    # def aggregate_parameters(self):
    #     assert (len(self.uploaded_ids) > 0)
        
    #     self.uploaded_base_model = []
    #     delta_params_per_client = [] 
        
    #     # 1. 提取 Delta W
    #     for cid in self.uploaded_ids:
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name)
    #         model = copy.deepcopy(client_model)
    #         model.recover_larger_model()
    #         model.to(self.device)
    #         self.uploaded_base_model.append(model)
            
    #         old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
    #         if old_start_model is not None:
    #             old_start_model = old_start_model.to(self.device)
    #         else:
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            
    #         client_layer_deltas = []
    #         for p_new, p_old in zip(model.parameters(), old_start_model.parameters()):
    #             delta_l = (p_new.data - p_old.data).view(-1)
    #             client_layer_deltas.append(delta_l)
                
    #         delta_params_per_client.append(client_layer_deltas)

    #     num_layers = len(delta_params_per_client[0])

    #     # ================== 修复：兜底聚合的安全防护 ==================
    #     general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #     for param in general_global_model.parameters():
    #         param.data.zero_()
            
    #     if not hasattr(self, 'uploaded_weights') or len(self.uploaded_weights) != len(self.uploaded_ids):
    #         print("⚠️ 未检测到 uploaded_weights，使用均匀权重进行兜底聚合")
    #         fallback_weights = [1.0 / len(self.uploaded_ids)] * len(self.uploaded_ids)
    #     else:
    #         fallback_weights = self.uploaded_weights

    #     for w, base_model in zip(fallback_weights, self.uploaded_base_model):
    #         for server_param, client_param in zip(general_global_model.parameters(), base_model.parameters()):
    #             w_tensor = torch.tensor(w).to(self.device)
    #             server_param.data += client_param.data.clone() * w_tensor
    #     save_item(general_global_model, self.role, 'model', self.save_folder_name)
    #     # ==============================================================

    #     # ================= 计算数据量相对规模系数 =================
    #     num_participants = len(self.uploaded_ids)
    #     # 如果前面 fallback_weights 是均匀的，这里全是 1.0；
    #     # 如果是按样本量计算的真实 weights，这里就是相对规模！
    #     data_scales = [w * num_participants for w in fallback_weights]
    #     # ====================================================================

    #     print(f"执行基于按层(Layer-wise)与数据量先验(Data Prior)的个性化聚合...")
        
    #     # tau = 0.25
    #     # power = 3.0
    #     # gamma = 1.0
        
    #     tau = self.args.aggregate_tau
    #     power = self.args.aggregate_power
    #     gamma = self.args.aggregate_gamma


    #     num_total_clients = len(self.clients) 
    #     global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_layers)]

    #     uploaded_params_per_client = [list(m.parameters()) for m in self.uploaded_base_model]

    #     for i, target_cid in enumerate(self.uploaded_ids):
    #         # # 获取目标客户端自己的数据规模系数
    #         scale_i = data_scales[i]
            
    #         personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #         for param in personalized_global_model.parameters():
    #             param.data.zero_()
                
    #         pers_params = list(personalized_global_model.parameters())

    #         for layer_idx in range(num_layers):
    #             logits = [] 
    #             delta_i = delta_params_per_client[i][layer_idx]
                
    #             pure_norm_i = torch.norm(delta_i)
    #             # 剔除参数尺寸霸权 (RMS Norm)
    #             rms_norm_i = pure_norm_i / math.sqrt(delta_i.numel())
    #             scaled_norm_i = rms_norm_i * 100.0
                
    #             depth_ratio = ((layer_idx + 1) / num_layers) ** power
                
    #             # 集大成者的自适应自我偏置
    #             # self_bias = depth_ratio * (gamma + torch.log1p(scaled_norm_i)) * (scale_i ** 0.5)
    #             self_bias = depth_ratio * gamma * (scale_i ** 0.5)
                
    #             for j in range(len(self.uploaded_ids)):
    #                 delta_j = delta_params_per_client[j][layer_idx]
                    
    #                 # 1. 纯粹几何对齐度
    #                 cos_sim = torch.nn.functional.cosine_similarity(delta_i, delta_j, dim=0)
                    
    #                 # 2. 数据规模可靠性缩放：同向增强，反向加重惩罚。
    #                 safe_scale_j = max(data_scales[j], 1e-4)
    #                 data_factor = safe_scale_j ** 0.5
                    
    #                 # 3. 基础 Logit 计算
    #                 logit_j = (cos_sim * data_factor) / tau
                    
    #                 # ================= 护盾 =================
                    
    #                 if i == j:
    #                     logit_j += self_bias 
    #                 # ============================================
                        
    #                 logits.append(logit_j)
                
    #             # 算出纯粹的个性化注意力权重
    #             logits = torch.stack(logits) 
    #             layer_weights = torch.nn.functional.softmax(logits, dim=0)
                
    #             # 算出当前的深度比例
    #             depth_ratio = ((layer_idx + 1) / num_layers) ** power
                
    #             aligned_weights = np.zeros(num_total_clients)
                
    #             # ================= 终极闭环：深度残差融合 =================
    #             for j, upload_cid in enumerate(self.uploaded_ids):
    #                 global_w = fallback_weights[j]           # 大锅饭权重
    #                 pers_w = layer_weights[j].item()         # 个性化权重
                    
    #                 # 浅层 depth_ratio 近乎 0，强制使用 global_w
    #                 # 深层 depth_ratio 近乎 1，放权给 pers_w
    #                 final_w = (1.0 - depth_ratio) * global_w + depth_ratio * pers_w
                    
    #                 aligned_weights[upload_cid] = final_w
                    
    #                 # 物理参数加权
    #                 if final_w > 0:
    #                     client_j_layer_data = uploaded_params_per_client[j][layer_idx].data
    #                     pers_params[layer_idx].data += client_j_layer_data.clone() * final_w
    #             # ==========================================================
                
    #             global_weight_matrices[layer_idx][target_cid] = aligned_weights

    #         save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
                    
    #     # 4. 遍历打印每一层的权重，并保存热力图
    #     for layer_idx in range(num_layers):
    #         self.print_row_weights(global_weight_matrices[layer_idx], layer_idx=layer_idx)



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
        按 数据集/异构程度 建立层级文件夹
        """
        import os
        from datetime import datetime
        import numpy as np
        
        # 1. 动态获取当前实验的环境配置
        dataset_name = getattr(self.args, 'dataset', 'UnknownData')
        partition = getattr(self.args, 'partition', 'dir')
        # 兼容不同的参数命名习惯 (dir_alpha 或 alpha)
        alpha = getattr(self.args, 'dir_alpha', getattr(self.args, 'alpha', 'UnknownAlpha'))
        algo_name = getattr(self.args, 'algorithm', 'FedCLIP')
        
        # 2. 构建多级目录: ./Weight_Logs/数据集名称/划分方式_异构参数/
        # 举例: ./Weight_Logs/Cifar100/dir_0.1/
        log_dir = os.path.join(".", "Weight_Logs", dataset_name, f"{partition}_{alpha}")
        os.makedirs(log_dir, exist_ok=True)
        
        current_round = getattr(self, 'global_round', None)
        if current_round is None:
            current_round = getattr(self, 'cur_ground', 0)
            
        # ================= 核心修改：一次启动，只生成一个文件 =================
        if not hasattr(self, 'weight_log_filepath'):
            # 只有在第一轮、第一次调用时才会执行这里
            start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 文件名加入算法名，方便同分布下对比不同算法 (如 FedCLIP_20240514_xxx_Weights.txt)
            log_filename = f"{algo_name}_{start_time}_Weights.txt"
            self.weight_log_filepath = os.path.join(log_dir, log_filename)
            
            # 创建文件并写入一个清晰的实验信息头
            with open(self.weight_log_filepath, "w", encoding="utf-8") as f:
                f.write(f"========== 联邦学习权重聚合全局日志 ==========\n")
                f.write(f"算法名称: {algo_name}\n")
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"异构设置: {partition} (Alpha/分布参数: {alpha})\n")
                f.write(f"启动时间: {start_time}\n")
                f.write("="*46 + "\n")
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
            print(f"📄 第 {current_round} 轮权重日志已存入: {self.weight_log_filepath}")
        
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
        
        # ⚠️ 修改点 1：去掉了文件名里的 _noself
        weight_filename = f"{dataset_name}_{partition}_{alpha_data}_subset_norm_weights.txt"
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
        
        # 3. 初始化对齐矩阵
        num_total_clients = len(self.clients)
        current_round_weight_matrix = np.zeros((num_total_clients, num_total_clients))

        # 4. 执行完全依赖离线矩阵的个性化聚合
        for target_cid in self.uploaded_ids:
            target_weights = oracle_weight_matrix[target_cid]
            
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            # 提取本轮上线客户端的权重并重新归一化 (防止有人掉线导致权重和不为1)
            active_weights = []
            for upload_cid in self.uploaded_ids:
                active_weights.append(target_weights[upload_cid])
                
            active_weights = np.array(active_weights)
            
            if active_weights.sum() == 0:
                print(f"⚠️ 客户端 {target_cid} 匹配不到任何非0权重的上线节点，回退为全自身保留")
                active_weights = np.zeros_like(active_weights)
                # 如果自己在线，则100%保留自己的模型
                if target_cid in self.uploaded_ids:
                    my_idx = self.uploaded_ids.index(target_cid)
                    active_weights[my_idx] = 1.0
            else:
                active_weights = active_weights / active_weights.sum()

            # 记录对齐后的真实聚合权重用于打印
            aligned_weights = np.zeros(num_total_clients)
            for j, upload_cid in enumerate(self.uploaded_ids):
                aligned_weights[upload_cid] = active_weights[j]
            current_round_weight_matrix[target_cid] = aligned_weights

            # ⚠️ 修改点 2：单轨直接聚合！(不再切分 alpha_retention，因为 active_weights 里已经包含了对自己模型的正确权重)
            for w, base_model in zip(active_weights, self.uploaded_base_model):
                if w > 0: 
                    for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                        server_param.data += client_param.data.clone() * w

            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
            
        # 5. 打印本轮实际生效的聚合矩阵
        self.print_aligned_weights(current_round_weight_matrix)
        print("✅ 验真聚合完成，模型已完全基于上帝视角子集权重 (Subset Oracle) 更新。")

    def aggregate_avg(self):
        assert (len(self.uploaded_ids) > 0)
        #载入全局模型,全局模型是完整模型状态
        global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in global_model.parameters():
            param.data.zero_()
        #记录客户端恢复形状后的base模型
        self.uploaded_base_model = []

        for cid in  self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            #创建临时模型用于模型参数恢复
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
        print(f"执行权重聚合，聚合权重为{self.uploaded_weights}")
        for w,base_model in zip(self.uploaded_weights,self.uploaded_base_model):
            #将模型参数聚合
            for server_param, client_param in zip(global_model.base.parameters(), base_model.parameters()):
                w = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'model', self.save_folder_name)

    # def aggregate_parameters_delta(self):
    #     assert (len(self.uploaded_ids) > 0)
        
    #     self.uploaded_base_model = []
    #     delta_params_per_client = [] 
        
    #     # 1. 提取 Delta W (与原来保持一致)
    #     for cid in self.uploaded_ids:
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name)
    #         model = copy.deepcopy(client_model)
    #         model.recover_larger_model()
    #         model.to(self.device)
    #         self.uploaded_base_model.append(model.base)
            
    #         # 获取每个客户端的旧起点模型
    #         old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
    #         if old_start_model is not None:
    #             old_start_model = old_start_model.to(self.device)
    #         else:
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            
    #         client_layer_deltas = []
    #         for p_new, p_old in zip(model.base.parameters(), old_start_model.base.parameters()):
    #             # 这里为了算余弦相似度压平了 tensor
    #             delta_l = (p_new.data - p_old.data).view(-1) 
    #             client_layer_deltas.append(delta_l)
                
    #         delta_params_per_client.append(client_layer_deltas)

    #     num_layers = len(delta_params_per_client[0])

    #     # ================== 兜底全局聚合 (用于生成 fallback weights) ==================
    #     general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #     for param in general_global_model.parameters():
    #         param.data.zero_()
            
    #     if not hasattr(self, 'uploaded_weights') or len(self.uploaded_weights) != len(self.uploaded_ids):
    #         print("⚠️ 未检测到 uploaded_weights，使用均匀权重进行兜底聚合")
    #         fallback_weights = [1.0 / len(self.uploaded_ids)] * len(self.uploaded_ids)
    #     else:
    #         fallback_weights = self.uploaded_weights

    #     for w, base_model in zip(fallback_weights, self.uploaded_base_model):
    #         for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
    #             w_tensor = torch.tensor(w).to(self.device)
    #             server_param.data += client_param.data.clone() * w_tensor
    #     save_item(general_global_model, self.role, 'model', self.save_folder_name)
    #     # ==============================================================

    #     # 计算数据量相对规模系数
    #     num_participants = len(self.uploaded_ids)
    #     data_scales = [w * num_participants for w in fallback_weights]

    #     print(f"🚀 执行 PFL Delta 聚合 (加权变化量: 彻底抛弃自护盾) ...")
        
    #     tau = 0.25
    #     power = 3.0
        
    #     num_total_clients = len(self.clients) 
    #     global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_layers)]

    #     for i, target_cid in enumerate(self.uploaded_ids):
            
    #         # ================= 核心突破 1：加载“旧的我”作为绝对基底 =================
    #         old_start_model = load_item(self.role, f'model_{target_cid}', self.save_folder_name)
    #         if old_start_model is None:
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name)
            
    #         # 注意：不再是 zero_()！我们保留旧参数的所有知识！
    #         target_model = copy.deepcopy(old_start_model).to(self.device)
    #         pers_params = list(target_model.base.parameters())
    #         # ========================================================================

    #         for layer_idx in range(num_layers):
    #             logits = [] 
    #             delta_i = delta_params_per_client[i][layer_idx]
                
    #             # 彻底删除了 pure_norm_i, rms_norm_i, scaled_norm_i 和 self_bias 的运算！
                
    #             for j in range(len(self.uploaded_ids)):
    #                 delta_j = delta_params_per_client[j][layer_idx]
                    
    #                 # 1. 纯粹几何对齐度
    #                 cos_sim = torch.nn.functional.cosine_similarity(delta_i, delta_j, dim=0)
                    
    #                 # 2. 数据规模可靠性缩放：同向增强，反向加重惩罚。
    #                 safe_scale_j = max(data_scales[j], 1e-4)
    #                 data_factor = safe_scale_j ** 0.5
                    
    #                 # 3. 基础 Logit 计算 (不再强行给自己加护盾了！)
    #                 logit_j = (cos_sim * data_factor) / tau
                        
    #                 logits.append(logit_j)
                
    #             # 算出纯粹的个性化增量注意力权重
    #             logits = torch.stack(logits) 
    #             layer_weights = torch.nn.functional.softmax(logits, dim=0)
                
    #             # 算出当前的深度比例
    #             depth_ratio = ((layer_idx + 1) / num_layers) ** power
                
    #             aligned_weights = np.zeros(num_total_clients)
                
    #             # ================= 核心突破 2：Delta 残差融合加权 =================
    #             for j, upload_cid in enumerate(self.uploaded_ids):
    #                 global_w = fallback_weights[j] 
    #                 pers_w = layer_weights[j].item() 
                    
    #                 final_w = (1.0 - depth_ratio) * global_w + depth_ratio * pers_w
    #                 aligned_weights[upload_cid] = final_w
                    
    #                 if final_w > 0:
    #                     # 获取平铺的 delta_j
    #                     delta_j_flat = delta_params_per_client[j][layer_idx].to(self.device)
                        
    #                     # ⚠️ 关键修复：还原 delta_j 的维度，使其与物理参数维度一致
    #                     delta_j_reshaped = delta_j_flat.view_as(pers_params[layer_idx].data)
                        
    #                     # 在我的旧基底上，仅仅加上吸收来的“经验增量(Delta)”
    #                     pers_params[layer_idx].data += delta_j_reshaped * final_w
    #             # ==================================================================
                
    #             global_weight_matrices[layer_idx][target_cid] = aligned_weights

    #         # 保存基于增量更新后的目标模型
    #         save_item(target_model, self.role, f'model_{target_cid}', self.save_folder_name)
                    
    #     # 遍历打印每一层的权重，并保存热力图
    #     for layer_idx in range(num_layers):
    #         self.print_row_weights(global_weight_matrices[layer_idx], layer_idx=layer_idx)

    # def aggregate_parameters_v_old(self):
    #     assert (len(self.uploaded_ids) > 0)
        
    #     self.uploaded_base_model = []
    #     delta_params_per_client = [] 
        
    #     # 1. 🚀 提取原汁原味的低秩矩阵，并精准还原 SVD 初始状态
    #     for cid in self.uploaded_ids:
    #         client = self.clients[cid]
    #         client_model = load_item(client.role, 'model', client.save_folder_name)
    #         model = copy.deepcopy(client_model).to(self.device)
            
    #         old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            
    #         if old_start_model is None:
    #             # ================= 🚀 核心修复：现场还原 SVD 初始状态 =================
    #             print(f"⚠️ 未找到客户端 {cid} 的旧模型，加载全局模型并执行 SVD 分解对齐初始状态...")
    #             old_start_model = load_item(self.role, 'model', self.save_folder_name)
    #             old_start_model = old_start_model.to(self.device)
                
    #             # 严格模拟客户端拉取模型时的初始化动作，利用 SVD 将全秩分解为低秩
    #             old_start_model.decom_larger_model(model.ratio_LR)
                
    #             # SVD 分解会生成新的张量，必须再次推到 GPU 上防止 Device Mismatch
    #             old_start_model.to(self.device)
    #             # =======================================================================
    #         else:
    #             old_start_model = old_start_model.to(self.device)
            
    #         client_raw_deltas = {}
    #         for (name, p_new), (_, p_old) in zip(model.named_parameters(), old_start_model.named_parameters()):
    #             # 此时，无论是冷启动还是后续轮次，p_old 和 p_new 的形状已经 100% 完美对齐！
    #             # 算出来的就是真正的、纯粹的本地 SVD 子空间内的变化量！
    #             client_raw_deltas[name] = p_new.data.clone() - p_old.data.clone()
                
    #         delta_params_per_client.append(client_raw_deltas)
    #         self.uploaded_base_model.append(model)

    #     # 兜底：处理 uploaded_weights
    #     if not hasattr(self, 'uploaded_weights') or len(self.uploaded_weights) != len(self.uploaded_ids):
    #         fallback_weights = [1.0 / len(self.uploaded_ids)] * len(self.uploaded_ids)
    #     else:
    #         fallback_weights = self.uploaded_weights

    #     num_participants = len(self.uploaded_ids)
    #     data_scales = [w * num_participants for w in fallback_weights]

    #     print(f"🚀 执行真正的低秩子空间截断聚合 (无重构、无分解，安全切片拼接) ...")
        
    #     tau = getattr(self.args, 'aggregate_tau', 0.25)
    #     power = getattr(self.args, 'aggregate_power', 3.0)
    #     gamma = getattr(self.args, 'aggregate_gamma', 1.0)
        
    #     num_layers = len(list(self.uploaded_base_model[0].named_parameters()))
    #     num_total_clients = len(self.clients) 
    #     global_weight_matrices = [np.zeros((num_total_clients, num_total_clients)) for _ in range(num_layers)]

    #     for i, target_cid in enumerate(self.uploaded_ids):
    #         scale_i = data_scales[i]
            
    #         # 拿到目标客户端的模型壳子，并将里面清零
    #         # 这一步非常重要：清零后，小客户端加进来的数据不够的地方，天然就是补 0！
    #         personalized_global_model = load_item(self.role, f'model_{target_cid}', self.save_folder_name)
    #         if personalized_global_model is None:
    #             personalized_global_model = load_item(self.role, 'model', self.save_folder_name)
    #             personalized_global_model = personalized_global_model.to(self.device)
    #             personalized_global_model.decom_larger_model(self.uploaded_base_model[i].ratio_LR)
    #             personalized_global_model = personalized_global_model.to(self.device)
    #         else:
    #             personalized_global_model = personalized_global_model.to(self.device)
    #         for param in personalized_global_model.parameters():
    #             param.data.zero_()
                
    #         target_named_params = list(personalized_global_model.named_parameters())

    #         for layer_idx, (param_name, target_param) in enumerate(target_named_params):
    #             logits = [] 
                
    #             # 判断当前张量到底是什么属性
    #             is_v_matrix = param_name.endswith('conv_v') or param_name.endswith('weight_v')
    #             is_u_matrix = param_name.endswith('conv_u') or param_name.endswith('weight_u')
                
    #             # 寻找对照组 V (为了计算相似度护盾)
    #             target_v_name = param_name
    #             if param_name.endswith('conv_u'):
    #                 target_v_name = param_name[:-6] + 'conv_v'
    #             elif param_name.endswith('weight_u'):
    #                 target_v_name = param_name[:-8] + 'weight_v'
    #             is_v_anchor = target_v_name.endswith('conv_v') or target_v_name.endswith('weight_v')
                
    #             delta_i = delta_params_per_client[i][param_name]
                
    #             # 计算自护盾
    #             pure_norm_i = torch.norm(delta_i)
    #             rms_norm_i = pure_norm_i / math.sqrt(max(1, delta_i.numel()))
                
    #             depth_ratio = ((layer_idx + 1) / num_layers) ** power
    #             self_bias = depth_ratio * gamma * (scale_i ** 0.5)

    #             for j in range(len(self.uploaded_ids)):
    #                 delta_j = delta_params_per_client[j][param_name]
                    
    #                 # =============== 2. 安全的截断相似度计算 ===============
    #                 if is_v_anchor and target_v_name in delta_params_per_client[i]:
    #                     raw_i = delta_params_per_client[i][target_v_name]
    #                     raw_j = delta_params_per_client[j][target_v_name]
                        
    #                     # 找到两者的公共最小维度(交集)进行对齐
    #                     slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(raw_i.shape, raw_j.shape))
    #                     trunc_i = raw_i[slices].contiguous().view(-1)
    #                     trunc_j = raw_j[slices].contiguous().view(-1)
                        
    #                     if trunc_i.numel() > 0:
    #                         cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0)
    #                     else:
    #                         cos_sim = torch.tensor(0.0).to(self.device)
    #                 else:
    #                     # 全秩层的退化处理
    #                     slices = tuple(slice(0, min(dim_i, dim_j)) for dim_i, dim_j in zip(delta_i.shape, delta_j.shape))
    #                     trunc_i = delta_i[slices].contiguous().view(-1)
    #                     trunc_j = delta_j[slices].contiguous().view(-1)
    #                     cos_sim = torch.nn.functional.cosine_similarity(trunc_i, trunc_j, dim=0) if trunc_i.numel() > 0 else torch.tensor(0.0).to(self.device)
                    
    #                 safe_scale_j = max(data_scales[j], 1e-4)
    #                 data_factor = safe_scale_j ** 0.5
    #                 logit_j = (cos_sim * data_factor) / tau
                    
    #                 if i == j:
    #                     logit_j += self_bias 
                        
    #                 logits.append(logit_j)
                
    #             logits = torch.stack(logits) 
    #             layer_weights = torch.nn.functional.softmax(logits, dim=0)
                
    #             aligned_weights = np.zeros(num_total_clients)
                
    #             # =============== 3. 终极逻辑：参数直接切片拼接 ===============
    #             for j, upload_cid in enumerate(self.uploaded_ids):
    #                 global_w = fallback_weights[j]           
    #                 pers_w = layer_weights[j].item()         
                    
    #                 final_w = (1.0 - depth_ratio) * global_w + depth_ratio * pers_w
    #                 aligned_weights[upload_cid] = final_w
                    
    #                 if final_w > 0:
    #                     client_j_named_params = dict(self.uploaded_base_model[j].named_parameters())
    #                     client_j_data = client_j_named_params[param_name].data
                        
    #                     if is_v_matrix:
    #                         # V 矩阵: rank 控制行数 (第 0 维)
    #                         min_r = min(target_param.shape[0], client_j_data.shape[0])
    #                         # 截取前 min_r 行相加 (因为 target 已经被清零了，大尺寸接收小尺寸天然就是补0！)
    #                         target_param.data[:min_r, ...] += client_j_data[:min_r, ...] * final_w
                            
    #                     elif is_u_matrix:
    #                         # U 矩阵: rank 控制列数 (第 1 维)
    #                         min_r = min(target_param.shape[1], client_j_data.shape[1])
    #                         # 截取前 min_r 列相加
    #                         target_param.data[:, :min_r, ...] += client_j_data[:, :min_r, ...] * final_w
                            
    #                     else:
    #                         # 普通全秩矩阵 (Bias等) 理论上维度都一样，但也加上保护
    #                         slices = tuple(slice(0, min(dim_t, dim_j)) for dim_t, dim_j in zip(target_param.shape, client_j_data.shape))
    #                         target_param.data[slices] += client_j_data[slices] * final_w
    #             # ==============================================================
                
    #             global_weight_matrices[layer_idx][target_cid] = aligned_weights

    #         save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)
                    
    #     for layer_idx in range(num_layers):
    #         self.print_row_weights(global_weight_matrices[layer_idx], layer_idx=layer_idx)
