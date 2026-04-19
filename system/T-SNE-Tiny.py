import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_utils import read_client_data
import random
import numpy as np

# 固定所有随机种子
RANDOM_SEED = 0


def set_random_seed(seed=RANDOM_SEED):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_random_seed(RANDOM_SEED)

# TinyImageNet 类别名称（200个类别）
# 首先尝试从文件中读取类别名称，如果不存在则创建
def load_tinyimagenet_classes(class_file_path="../dataset/TinyImagenet/rawdata/tiny-imagenet-200/wnids.txt"):
    """
    加载TinyImageNet类别名称
    Args:
        class_file_path: wnids.txt文件路径，包含200个WordNet ID
    """
    if os.path.exists(class_file_path):
        with open(class_file_path, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return wnids
    else:
        # 如果文件不存在，生成200个占位类别名称
        return [f"class_{i:03d}" for i in range(200)]

# 加载类别名称
tinyimagenet_classes = load_tinyimagenet_classes()


def unified_tsne_across_clients(model_path_dr,
                                num_clients=20,
                                excel_result_dir="./T-SNE/DAT/CIFAR100/Hetero/Total",
                                dataset_type="TinyImagenet"):
    """
    所有客户端在统一表征空间进行t-SNE降维
    """

    # 1. 收集所有客户端的特征
    print("收集所有客户端的特征...")
    all_features = []
    all_labels = []
    all_client_ids = []

    for client_id in range(num_clients):
        print(f"处理客户端 {client_id}...")

        # 加载模型
        model_path = os.path.join(model_path_dr, f"Client_{client_id}_model.pt")
        print(model_path)
        if not os.path.exists(model_path):
            print(f"警告: 客户端 {client_id} 的模型文件不存在，跳过")
            continue

        try:
            model = torch.load(model_path, map_location='cpu')
            model.eval()
        except Exception as e:
            print(f"加载客户端 {client_id} 模型失败: {e}")
            continue

        # 加载数据并提取特征
        try:
            client_dataloader = load_train_data(str(client_id), dataset_type)

            client_features = []
            client_labels = []

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(client_dataloader):
                    if batch_idx >= 40:  # 限制每个客户端的数据量，避免内存不足
                        break
                    features = model.base(images)
                    client_features.append(features.numpy())
                    client_labels.append(labels.numpy())

            if client_features:
                client_features_flat = np.concatenate(client_features, axis=0)
                client_labels_flat = np.concatenate(client_labels, axis=0)

                # 添加到总集合
                all_features.append(client_features_flat)
                all_labels.append(client_labels_flat)
                all_client_ids.extend([client_id] * len(client_labels_flat))

                print(f"客户端 {client_id}: 收集了 {len(client_labels_flat)} 个样本")

        except Exception as e:
            print(f"处理客户端 {client_id} 数据失败: {e}")
            continue

    if not all_features:
        print("没有成功收集到任何客户端的特征")
        return

    # 2. 合并所有客户端的特征
    print("合并所有客户端的特征...")
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_client_ids = np.array(all_client_ids)

    print(f"总特征形状: {combined_features.shape}")
    print(f"总标签形状: {combined_labels.shape}")
    print(f"客户端ID分布: {np.bincount(combined_client_ids)}")

    # 3. 统一进行t-SNE降维
    print("进行统一的t-SNE降维...")
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(combined_features) - 1),
            learning_rate=200,
            random_state=RANDOM_SEED,  # 使用相同的随机种子
            max_iter=1000,
            verbose=1
        )
        unified_features_2d = tsne.fit_transform(combined_features)
        print(f"统一t-SNE降维后形状: {unified_features_2d.shape}")
    except Exception as e:
        print(f"统一t-SNE降维失败: {e}")
        return

    # 4. 创建结果DataFrame
    unified_df = pd.DataFrame({
        'client_id': combined_client_ids,
        'label': combined_labels,
        'class_name': [tinyimagenet_classes[label] for label in combined_labels],
        't-SNE_dim1': unified_features_2d[:, 0],
        't-SNE_dim2': unified_features_2d[:, 1],
    })

    # 5. 保存结果
    os.makedirs(excel_result_dir, exist_ok=True)

    # 保存整体结果
    unified_excel_path = os.path.join(excel_result_dir, "unified_tsne_all_clients.xlsx")
    unified_csv_path = os.path.join(excel_result_dir, "unified_tsne_all_clients.csv")

    unified_df.to_excel(unified_excel_path, index=False)
    unified_df.to_csv(unified_csv_path, index=False)

    print(f"统一t-SNE结果已保存:")
    print(f"Excel: {unified_excel_path}")
    print(f"CSV: {unified_csv_path}")

    # 6. 为每个客户端单独保存结果（基于统一降维）
    for client_id in range(num_clients):
        client_mask = unified_df['client_id'] == client_id
        client_data = unified_df[client_mask]

        if len(client_data) > 0:
            client_excel_path = os.path.join(excel_result_dir, f"client_{client_id}_unified_tsne.xlsx")
            client_data.to_excel(client_excel_path, index=False)

    # 7. 可视化
    visualize_unified_tsne(unified_df, excel_result_dir)

    return unified_df




def visualize_unified_tsne(tsne_df, save_dir):

    # 创建单个图表
    plt.figure(figsize=(15, 10))

    # 获取所有唯一的类别
    unique_classes = tsne_df['class_name'].unique()
    num_classes = len(unique_classes)

    # 使用丰富的颜色映射（适合100个类别）
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))

    # 为每个类别绘制点
    for i, class_name in enumerate(unique_classes):
        class_data = tsne_df[tsne_df['class_name'] == class_name]
        if len(class_data) > 0:
            color = colors[i]
            plt.scatter(class_data['t-SNE_dim1'], class_data['t-SNE_dim2'],
                        color=color, alpha=0.7, s=20)

    plt.title('T-SNE Visualization of tinyimagenet Features')
    plt.xlabel('T-SNE Dimension 1')
    plt.ylabel('T-SNE Dimension 2')

    # 保存图像
    plot_path = os.path.join(save_dir, "tinyimagenet_tsne_visualization.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_path = os.path.join(save_dir, "tinyimagenet_tsne_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"CIFAR-100 t-SNE可视化图已保存: {plot_path}")


def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 使用统一表征空间的t-SNE
    print("开始统一t-SNE降维...")
    unified_results = unified_tsne_across_clients(model_path_dr="./temp/TinyImagenet/FedHAS/1765263319.2276285",
                                num_clients=20,
                                excel_result_dir="./T-SNE/ARA/TinyImagenet/Hetero/Total",)

    if unified_results is not None:
        print("\n统一t-SNE处理完成!")
        print(f"总共处理了 {len(unified_results)} 个样本")
        print(f"涉及客户端: {sorted(unified_results['client_id'].unique())}")
    else:
        print("\n统一t-SNE处理失败")