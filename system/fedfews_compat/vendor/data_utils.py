import numpy as np
import os
import torch
from collections import defaultdict
from common.data_io import select_client_shard, read_shard, decode_image_data, PKL_FIRST, NUMPY_STACK_LIST

SHARD_FORMATS = PKL_FIRST
IMAGE_LIST_MODE = NUMPY_STACK_LIST


def _get_dataset_root():
    """
    智能查找 dataset 目录
    支持从 system/ 目录或项目根目录运行
    """
    # 首先检查环境变量
    if "PFLLIB_DATA_DIR" in os.environ:
        return os.environ["PFLLIB_DATA_DIR"]

    # 方法1: 从当前文件位置向上查找 (适用于从 system/ 运行)
    current_file = os.path.abspath(__file__)
    system_dir = os.path.dirname(os.path.dirname(current_file))  # utils -> system
    dataset_dir_v1 = os.path.join(os.path.dirname(system_dir), 'dataset')  # system -> PFLlib -> dataset

    if os.path.exists(dataset_dir_v1):
        return dataset_dir_v1

    # 方法2: 从工作目录查找 (适用于从项目根目录运行)
    cwd = os.getcwd()
    # 情况1: cwd 是 code/，dataset 在 code/external/PFLlib/dataset/
    dataset_dir_v2 = os.path.join(cwd, 'external', 'PFLlib', 'dataset')
    if os.path.exists(dataset_dir_v2):
        return dataset_dir_v2

    # 情况2: cwd 是 system/，dataset 在 ../dataset/
    dataset_dir_v3 = os.path.join(cwd, '..', 'dataset')
    if os.path.exists(dataset_dir_v3):
        return os.path.abspath(dataset_dir_v3)

    # 如果都找不到，返回默认相对路径（保持向后兼容）
    return os.path.join('..', 'dataset')


def client_shard_path(dataset, idx, is_train=True, *, data_root=None):
    """Keep the FedFew root/layout and explicit pkl-first policy in this adapter."""
    dataset_root = _get_dataset_root() if data_root is None else data_root
    if is_train:
        data_dir = os.path.join(dataset_root, dataset, 'train/')
    else:
        data_dir = os.path.join(dataset_root, dataset, 'test/')
    try:
        return select_client_shard(data_dir, idx, formats=SHARD_FORMATS)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found for client {idx} in {data_dir} (tried .pkl and .npz)") from None


def read_data(dataset, idx, is_train=True):
    return read_shard(client_shard_path(dataset, idx, is_train))


def read_client_data(dataset, idx, is_train=True, few_shot=0):
    data = read_data(dataset, idx, is_train)
    return _process_client_data(data, dataset, is_train, few_shot)


def read_client_data_from_file(path, dataset, is_train=True, few_shot=0):
    """Audit a selected file through the SAME decoding/filter path as training."""
    return _process_client_data(read_shard(path), dataset, is_train, few_shot)


def _process_client_data(data, dataset, is_train, few_shot):
    dataset_lower = dataset.lower()
    if "news" in dataset_lower:
        data_list = process_text(data)
    elif "shakespeare" in dataset_lower:
        data_list = process_Shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
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
    return decode_image_data(data, list_mode=IMAGE_LIST_MODE)


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
