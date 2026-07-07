import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
import re
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item
# from torch.utils.tensorboard import SummaryWriter
import json
class Server(object):
    # Core server compute only. Client selection may run local FLOPs estimation, and
    # parameter sending calls client-side loading/decomposition, so both stay out.
    _TIMED_SERVER_METHODS = {
        "receive_ids",
        "receive_protos",
        "receive_models",
        "receive_logits",
        "aggregate_parameters",
        "aggregate_parameters_v_svd",
        "aggregate_parameters_v_svd_res",
        "aggregate_parameters_v_svd_drop",
        "train_head",
        "update_TGP",
        "train_generator",
        "aggregate_protos",
        "aggregate_logits",
        "get_filters",
        "set_filters",
    }

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if name.startswith("_") or name not in Server._TIMED_SERVER_METHODS or not callable(attr):
            return attr

        def timed_attr(*args, **kwargs):
            args_obj = object.__getattribute__(self, "args") if "args" in self.__dict__ else None
            if args_obj is None or not getattr(args_obj, "measure_server_compute", 0):
                return attr(*args, **kwargs)

            start_time = time.time()
            try:
                return attr(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                object.__getattribute__(self, "_record_server_compute_event")(name, elapsed)

        return timed_attr

    def __init__(self, args, times):
        # Set up the main attributes
        #参数
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        #总的通信轮次
        self.global_rounds = args.global_rounds
        #本地训练轮次
        self.local_epochs = args.local_epochs
        #数据批次数
        self.batch_size = args.batch_size
        #客户端本地训练学习率
        self.learning_rate = args.local_learning_rate
        #客户端个数
        self.num_clients = args.num_clients
        #客户端参与率
        self.join_ratio = args.join_ratio
        #客户端是否随机参与
        self.random_join_ratio = args.random_join_ratio
        #激活客户端个数
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        #当前参与客户端个数（）
        self.current_num_join_clients = self.num_join_clients
        #是否进行小样本学习
        self.few_shot = args.few_shot
        #算法
        self.algorithm = args.algorithm
        #是否根据训练时间选择客户端
        self.time_select = args.time_select
        #实验目标  是测试还是训练
        self.goal = args.goal
        #设置丢弃的客户端
        self.time_threthold = args.time_threthold
        #设置早停的判断步数
        self.top_cnt = args.top_cnt
        #是否设置早停
        self.auto_break = args.auto_break
        #用于标记客户端
        self.role = 'Server'
        #权重参数保存的位置
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full
        #记录所有客户端对象
        self.clients = []
        #被选择的客户端
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []
        #记录上传的客户端参数的加权值
        self.uploaded_weights = []
        #记录上传模型的客户端id
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.local_flops_records = []
        self.server_compute_records = []
        self._local_flops_select_count = 0
        self._current_server_round_idx = 0

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate
        # #tensor_board日志文件
        # log_path = os.path.join("./hylogs", self.args.exp_name)
        # self.writer = SummaryWriter(log_dir=log_path)
    #设置客户端参数，创建客户端对象
    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot, args=self.args)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot, args=self.args)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)
    #设一般不使用
    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self): # 选择慢客户端
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)
    #选择激活的客户顿
    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self._current_server_round_idx = int(getattr(self, "cur_ground", self._local_flops_select_count))
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        self._maybe_report_selected_local_flops(selected_clients)
        self._local_flops_select_count += 1

        return selected_clients

    def _record_server_compute_event(self, event_name, elapsed_seconds):
        if not getattr(self.args, "measure_server_compute", 0):
            return
        round_idx = int(getattr(self, "_current_server_round_idx", 0))
        self.server_compute_records.append({
            "round": round_idx,
            "event": event_name,
            "seconds": float(elapsed_seconds),
        })
        if getattr(self.args, "server_compute_detail", 0):
            print(f"🖥️ [Round {round_idx:03d}] Server event {event_name}: {elapsed_seconds:.6f}s")

    def _format_flops(self, flops):
        flops = float(flops)
        if flops >= 1e12:
            return f"{flops / 1e12:.3f} TFLOPs"
        if flops >= 1e9:
            return f"{flops / 1e9:.3f} GFLOPs"
        if flops >= 1e6:
            return f"{flops / 1e6:.3f} MFLOPs"
        return f"{flops:.0f} FLOPs"

    def _maybe_report_selected_local_flops(self, selected_clients):
        if not getattr(self.args, "measure_local_flops", 0):
            return
        round_idx = getattr(self, "cur_ground", self._local_flops_select_count)
        target_round = getattr(self.args, "local_flops_round", 0)
        if target_round >= 0 and round_idx != target_round:
            return

        details = []
        failed_clients = []
        for client in selected_clients:
            try:
                details.append(client.estimate_local_train_flops())
            except Exception as exc:
                failed_clients.append((client.id, repr(exc)))

        total_flops = sum(item["local_train_flops"] for item in details)
        total_forward_epoch = sum(item["forward_flops_per_epoch"] for item in details)
        record = {
            "round": int(round_idx),
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "model_family": getattr(self.args, "model_family", None),
            "join_ratio": float(self.join_ratio),
            "num_selected_clients": len(selected_clients),
            "train_multiplier": float(getattr(self.args, "local_flops_train_multiplier", 3.0)),
            "total_forward_flops_per_epoch": float(total_forward_epoch),
            "total_local_train_flops": float(total_flops),
            "client_details": details,
            "failed_clients": failed_clients,
        }
        self.local_flops_records.append(record)

        if getattr(self.args, "local_flops_detail", 1):
            print(
                f"🧮 [Round {round_idx:03d}] 本地训练计算量估计: "
                f"total={self._format_flops(total_flops)} | "
                f"forward/epoch={self._format_flops(total_forward_epoch)} | "
                f"clients={len(details)}/{len(selected_clients)} | "
                f"train_multiplier={record['train_multiplier']:.2f}"
            )
            detail_text = ", ".join(
                f"Client_{item['client_id']}:{self._format_flops(item['local_train_flops'])}"
                for item in details
            )
            print(f"🧮 [Round {round_idx:03d}] 本地训练计算量明细: {detail_text}")
        if failed_clients and getattr(self.args, "local_flops_detail", 1):
            print(f"⚠️ [Round {round_idx:03d}] FLOPs 估计失败客户端: {failed_clients}")
    #发送模型参数
    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
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
    #服务器聚合模型参数
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, 'model', client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
    #保存训练结果
    def save_results(self):
        import os
        from datetime import datetime
        import h5py 
        
        result_path = self.h5_result_dir()
        os.makedirs(result_path, exist_ok=True)

        if len(self.rs_test_acc) > 0:
            algo_prefix = f"{self.dataset}_{self.algorithm}_{self.goal}_{self.times}"
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if hasattr(self.args, 'trial_id'):
                file_name = f"{algo_prefix}_Trial{self.args.trial_id}_{time_str}.h5"
            else:
                file_name = f"{algo_prefix}_{time_str}.h5"
                
            file_path = os.path.join(result_path, file_name)
            print(f"💾 实验结果已安全保存至: {file_path}")

            # 🌟 核心修改：将精确生成的物理路径保存到 args 中！
            if hasattr(self.args, 'save_file_paths'):
                self.args.save_file_paths.append(file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.attrs['dataset'] = str(self.dataset)
                hf.attrs['algorithm'] = str(self.algorithm)
                hf.attrs['model_family'] = str(getattr(self.args, 'model_family', 'unknown_model'))
                hf.attrs['partition'] = str(getattr(self.args, 'partition', 'unknown_partition'))
                hf.attrs['args_json'] = json.dumps(
                    self._to_json_serializable(vars(self.args)),
                    ensure_ascii=False,
                    sort_keys=True
                )
        self.export_final_models()
        # # 训练完成整个存储的模型都被删除了
        # if 'temp' in self.save_folder_name:
        #     try:
        #         shutil.rmtree(self.save_folder_name)
        #         print('Deleted.')
        #     except:
        #         print('Already deleted.')

    def _safe_path_component(self, value):
        text = str(value)
        text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
        return text.strip("_") or "default"

    def _float_path_component(self, value):
        return str(value).replace(".", "p")

    def h5_result_dir(self):
        root = getattr(self.args, "h5_result_root", "./h5_results")
        model_family = getattr(self.args, "model_family", "unknown_model")
        data_tag = f"ncl{self.num_classes}_niid{getattr(self.args, 'niid', 'default')}"
        join_tag = f"clients{self.num_clients}_jr{self._float_path_component(self.join_ratio)}"
        return os.path.join(
            root,
            self._safe_path_component(self.dataset),
            self._safe_path_component(self.algorithm),
            self._safe_path_component(model_family),
            self._partition_path_component(),
            data_tag,
            join_tag,
        )

    def _partition_path_component(self):
        partition = getattr(self.args, "partition", "iid")
        if partition == "dir":
            alpha = getattr(self.args, "dir_alpha", "default")
            return f"dir_alpha{self._float_path_component(alpha)}"
        if partition == "pat":
            cpc = getattr(self.args, "class_per_client", "default")
            return f"pat_cpc{cpc}"
        if partition == "exdir":
            alpha = getattr(self.args, "dir_alpha", "default")
            return f"exdir_alpha{self._float_path_component(alpha)}"
        return self._safe_path_component(partition)

    def final_model_dir(self):
        root = getattr(self.args, "final_model_root", "./final_models")
        model_family = getattr(self.args, "model_family", "unknown_model")
        data_tag = f"ncl{self.num_classes}_niid{getattr(self.args, 'niid', 'default')}"
        join_tag = f"clients{self.num_clients}_jr{self._float_path_component(self.join_ratio)}"
        return os.path.join(
            root,
            self._safe_path_component(self.dataset),
            self._safe_path_component(self.algorithm),
            self._safe_path_component(model_family),
            self._partition_path_component(),
            data_tag,
            join_tag,
        )

    def _is_final_model_file(self, filename):
        return filename.endswith(".pt")

    def export_final_models(self):
        source_dir = self.save_folder_name
        if not source_dir or not os.path.isdir(source_dir):
            print(f"⚠️ 最终模型导出失败，源目录不存在: {source_dir}")
            return

        target_dir = self.final_model_dir()
        if os.path.abspath(source_dir) == os.path.abspath(target_dir):
            print(f"⚠️ 最终模型导出跳过，源目录和目标目录相同: {target_dir}")
            return
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        copied_files = []
        for filename in sorted(os.listdir(source_dir)):
            if not self._is_final_model_file(filename):
                continue
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
                copied_files.append(filename)

        manifest = {
            "source_dir": source_dir,
            "target_dir": target_dir,
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "model_family": getattr(self.args, "model_family", None),
            "partition": getattr(self.args, "partition", None),
            "dir_alpha": getattr(self.args, "dir_alpha", None),
            "class_per_client": getattr(self.args, "class_per_client", None),
            "num_classes": self.num_classes,
            "niid": getattr(self.args, "niid", None),
            "num_clients": self.num_clients,
            "join_ratio": self.join_ratio,
            "global_rounds": self.global_rounds,
            "local_epochs": self.local_epochs,
            "best_accuracy": max(self.rs_test_acc) if len(self.rs_test_acc) > 0 else None,
            "exported_files": copied_files,
            "args": vars(self.args),
        }
        manifest_path = os.path.join(target_dir, "manifest.json")
        self.save_json(file_path=manifest_path, dict=manifest, indent=4)
        print(f"✅ 最终模型已覆盖导出到: {target_dir}")
        print(f"✅ 导出模型文件数: {len(copied_files)}")
    #记录客户端的平均测试精度
    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            print(f'Client {c.id}: Acc: {ct*1.0/ns}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc
    #所有客户端的平均训练指标
    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
            print(f'Client {c.id}: Loss: {cl*1.0/ns}')

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses
    #对所有客户端测试性能
    # evaluate all clients
    def evaluate(self, acc=None, loss=None,epoch=0):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)
        # self.writer.add_scalar("Train/train_loss", train_loss,epoch)
        # self.writer.add_scalar("Test/Test_acc",test_acc,epoch)
        # self.writer.add_scalar("Test_std/test_acc",np.std(accs),epoch)
        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))
    #检查早停设置（精度不增长/达到目标精度等）
    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
    def _to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {self._to_json_serializable(k): self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return obj

    def save_json(self,file_path="./json",dict={},indent=4):
        """
        将数据保存为JSON文件

        参数:
            data: 要保存的数据，可以是字典或列表
            file_path: 保存的文件路径
            indent: JSON缩进空格数，默认为4
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._to_json_serializable(dict), f, ensure_ascii=False, indent=indent)
            print(f"JSON文件已成功保存到: {file_path}")
        except Exception as e:
            print(f"保存JSON文件时出错: {e}")
    def save_json_file(self):
        dict = {
            "train_loss": self.rs_train_loss,
            "test_acc": self.rs_test_acc,
            "local_flops_records": getattr(self, "local_flops_records", []),
            "server_compute_records": getattr(self, "server_compute_records", []),
            "args": vars(self.args)  # 新增这一行，将所有启动参数转换为字典保存
        }
        filename = self.args.exp_name + ".json"
        filepath = os.path.join("./json", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save_json(file_path=filepath, dict=dict)
