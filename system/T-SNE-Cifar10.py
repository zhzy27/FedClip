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

# CIFAR-10 类别名称
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def unified_tsne_across_clients(model_path_dr="./temp/Cifar10/FedDAR/1764939710.5983057",
                                num_clients=20,
                                excel_result_dir="./T-SNE/DAT/CIFAR10/Hetero/Total",
                                dataset_type="Cifar10"):
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
        'class_name': [cifar10_classes[label] for label in combined_labels],
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
    """可视化统一的t-SNE结果"""

    # 按客户端可视化
    plt.figure(figsize=(15, 10))

    # 子图1: 按类别着色
    plt.subplot(2, 2, 1)
    for class_name in cifar10_classes:
        class_data = tsne_df[tsne_df['class_name'] == class_name]
        if len(class_data) > 0:
            plt.scatter(class_data['t-SNE_dim1'], class_data['t-SNE_dim2'],
                        label=class_name, alpha=0.6, s=20)
    plt.title('Unified t-SNE by Class')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "unified_tsne_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"统一t-SNE可视化图已保存: {plot_path}")


def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 使用统一表征空间的t-SNE /temp/Cifar10/FedDAR/1765270697.6636217
    print("开始统一t-SNE降维...")
    unified_results = unified_tsne_across_clients(model_path_dr="./temp/Cifar10/FedDAR/1775566461.1170385",
                                num_clients=20,
                                excel_result_dir="./T-SNE/DAT/CIFAR10/Hetero/total/CE_CON_MSE")

    if unified_results is not None:
        print("\n统一t-SNE处理完成!")
        print(f"总共处理了 {len(unified_results)} 个样本")
        print(f"涉及客户端: {sorted(unified_results['client_id'].unique())}")
    else:
        print("\n统一t-SNE处理失败")