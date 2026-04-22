import numpy as np
import os
import torch
from collections import defaultdict

def get_dataset_sub_dir(args):
    """根据传入参数推导正确的数据子文件夹名称"""
    if getattr(args, 'niid', 1) == 0:
        return "iid"
    partition = getattr(args, 'partition', 'dir')
    if partition == "dir":
        return f"dir_{getattr(args, 'dir_alpha', 0.1)}"
    elif partition == "pat":
        return f"pat_{getattr(args, 'class_per_client', 2)}"
    elif partition == "exdir":
        return f"exdir_{getattr(args, 'class_per_client', 2)}_{getattr(args, 'dir_alpha', 0.1)}"
    return partition


#读客户端数据
def read_data(dataset, idx, args, is_train=True):
    sub_dir = get_dataset_sub_dir(args)
    
    if is_train:
        data_dir = os.path.join('../dataset', dataset, sub_dir, 'train/')
    else:
        data_dir = os.path.join('../dataset', dataset, sub_dir, 'test/')

    file_path = os.path.join(data_dir, f"{idx}.npz")
    
    # 【核心安全校验】: 阻断错误加载
    if not os.path.exists(file_path):
        cmd = f"python generate_{dataset}.py --niid {args.niid} --partition {args.partition} --alpha {args.dir_alpha} --cpc {args.class_per_client}"
        raise FileNotFoundError(
            f"\n\n❌ 【致命错误】找不到匹配的数据切片: {file_path}\n"
            f"系统拒绝加载其他目录的默认数据！\n"
            f"请先退回 dataset 目录并执行以下命令生成对应数据:\n👉 {cmd}\n"
        )

    with open(file_path, 'rb') as f:
        train_data = np.load(f, allow_pickle=True)['data'].tolist()
    
    if idx == 1:
        phase = "训练" if is_train else "测试"
        print(f"✅ 成功加载 [{phase}] 数据，路径: {data_dir}*.npz")

    return train_data

#读客户端数据  few_shot 参数的作用是控制在训练阶段读取每个类别的样本数量，从而实现小样本学习（Few-Shot Learning）的效果
def read_client_data(dataset, idx, args, is_train=True, few_shot=0):
    # ⚠️ 注意这里：将 args 传给 read_data
    data = read_data(dataset, idx, args, is_train)
    
    # 完全保留你的数据预处理逻辑
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        from collections import defaultdict # 确保 defaultdict 被导入
        shot_cnt_dict = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
        
    return data_list

def process_image(data):
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data):
    X, X_lens = list(zip(*data['x']))
    y = data['y']
    X = torch.Tensor(X).type(torch.int64)
    X_lens = torch.Tensor(X_lens).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [((x, lens), y) for x, lens, y in zip(X, X_lens, y)]


def process_Shakespeare(data):
    X = torch.Tensor(data['x']).type(torch.int64)
    y = torch.Tensor(data['y']).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

