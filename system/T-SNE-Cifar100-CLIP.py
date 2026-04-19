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
import torch.nn.functional as F
from utils.get_clip_text_encoder import get_clip_class_embeddings
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

clip_text_features,clip_text_features_norm = get_clip_class_embeddings('Cifar100',model_name= "ViT-B/32",prompt_template="a photo of {}")
clip_text_features,clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()

# CIFAR-100 类别名称
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


# 1762577601.8180041 1762577550.5284958   /FedTGP/1762826980.5112686 /FedProto/1762826991.56914 system/temp/Cifar100/FedHAS/1764478460.493608
def unified_tsne_across_clients(model_path_dr="./temp/Cifar100/FedDAR/1764775858.727305",
                                num_clients=20,
                                excel_result_dir="./T-SNE/DAT/CIFAR100/Hetero/Total",
                                dataset_type="Cifar100"):
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
            print(model)
            for param in model.parameters():
                print(param.shape)
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
                    features_norm = F.normalize(features, dim=-1)
                    client_features.append(features_norm.numpy())
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
    #clip_text_features_norm 将clip文本特征合并和模型生成的特征一起显示，文本特征用星号表示    
    text_features_list = []
    text_labels_list = []
    text_client_ids_list = []
    
    for class_id, text_feature in enumerate(clip_text_features_norm):
        text_features_list.append(text_feature.cpu().numpy().flatten())
        text_labels_list.append(class_id)
        text_client_ids_list.append(-1)  # 用 -1 标记为文本特征

    # 将文本特征加入总集合
    all_features.append(np.stack(text_features_list, axis=0))
    all_labels.append(np.array(text_labels_list))
    all_client_ids.extend(text_client_ids_list)

    # 2. 合并所有客户端的特征
    print("合并所有客户端的特征...")
    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_client_ids = np.array(all_client_ids)



    print(f"总特征形状: {combined_features.shape}")
    print(f"总标签形状: {combined_labels.shape}")
    print(f"客户端ID分布: {(combined_client_ids)}")

    # 3. 统一进行t-SNE降维
    print("进行统一的t-SNE降维...")
    try:
        tsne = TSNE(
            metric='cosine',
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

    # 5. 创建结果 DataFrame，增加一列 'type' 用于区分图像和文本
    unified_df = pd.DataFrame({
        'client_id': combined_client_ids,
        'label': combined_labels,
        'class_name': [cifar100_classes[label] for label in combined_labels],
        't-SNE_dim1': unified_features_2d[:, 0],
        't-SNE_dim2': unified_features_2d[:, 1],
        'type': ['text' if cid == -1 else 'image' for cid in combined_client_ids]
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


# def visualize_unified_tsne(tsne_df, save_dir):
#     """可视化统一的t-SNE结果"""

#     # 按客户端可视化
#     plt.figure(figsize=(15, 10))

#     # 子图1: 按类别着色
#     plt.subplot(2, 2, 1)
#     for class_name in cifar100_classes:
#         class_data = tsne_df[tsne_df['class_name'] == class_name]
#         if len(class_data) > 0:
#             plt.scatter(class_data['t-SNE_dim1'], class_data['t-SNE_dim2'],
#                        label=class_name, alpha=0.6, s=20)
#     plt.title(' T-SNE by Class')
#     plt.xlabel('T-SNE Dimension 1')
#     plt.ylabel('T-SNE Dimension 2')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


#     plt.tight_layout()
#     plot_path = os.path.join(save_dir, "unified_tsne_visualization.png")
#     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"统一t-SNE可视化图已保存: {plot_path}")


def visualize_unified_tsne(tsne_df, save_dir):
    """可视化 t-SNE 结果，用星号标记 CLIP 文本特征点"""
    plt.figure(figsize=(15, 10))

    # 分离图像点和文本点
    image_df = tsne_df[tsne_df['type'] == 'image']
    text_df = tsne_df[tsne_df['type'] == 'text']

    # 绘制图像点（按类别着色）
    unique_classes = image_df['class_name'].unique()
    num_classes = len(unique_classes)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_classes))
    
    for i, class_name in enumerate(unique_classes):
        class_data = image_df[image_df['class_name'] == class_name]
        if len(class_data) > 0:
            plt.scatter(class_data['t-SNE_dim1'], class_data['t-SNE_dim2'],
                        color=colors[i], alpha=0.7, s=20, label=class_name if i == 0 else "")

    # 绘制文本点（用星号，黑色边框，半透明填充）
    if len(text_df) > 0:
        # 按类别为文本点着色，以便与图像点对应
        for i, class_name in enumerate(unique_classes):
            class_text = text_df[text_df['class_name'] == class_name]
            if len(class_text) > 0:
                plt.scatter(class_text['t-SNE_dim1'], class_text['t-SNE_dim2'],
                            color=colors[i], marker='*', s=200, edgecolors='black',
                            linewidth=1, alpha=0.9, label=f'text-{class_name}' if i == 0 else "")

    plt.title('t-SNE Visualization with CLIP Text Features (stars)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # 添加图例（可选择性关闭，此处仅显示少量示例）
    # 如果不想显示所有类别，可以只显示几个代表性的图例
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles[:10], labels[:10], loc='upper right', fontsize=8)  # 只显示前10个

    # 保存图像
    plot_path = os.path.join(save_dir, "tsne_with_clip_text.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_path = os.path.join(save_dir, "tsne_with_clip_text.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"包含 CLIP 文本特征的 t-SNE 图已保存: {plot_path}")

def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 使用统一表征空间的t-SNE  system/temp/Cifar100/FedDAR/1765289055.04626
    print("开始统一t-SNE降维...")
    unified_results = unified_tsne_across_clients(model_path_dr  ="./temp/Cifar100/FedCLIP/1775566683.1429236",
                                num_clients=20,  
                                excel_result_dir="./resnet/FedCLIP/Cifar100",)

    if unified_results is not None:
        print("\n统一t-SNE处理完成!")
        print(f"总共处理了 {len(unified_results)} 个样本")
        print(f"涉及客户端: {sorted(unified_results['client_id'].unique())}")
    else:
        print("\n统一t-SNE处理失败")