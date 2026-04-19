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
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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

clip_text_features,clip_text_features_norm = get_clip_class_embeddings('Cifar100',model_name= "ViT-B/32",prompt_template= "{}")
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
                    # if batch_idx >= 15:  # 限制每个客户端的数据量，避免内存不足
                    #     break
                    features = model.base(images)
                    features_norm = F.normalize(features, dim=-1)
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


def compute_alignment_similarity(model_path_dr, num_clients, dataset_type, save_dir):
    """
    计算所有客户端：
    - 图像特征与对应文本特征的余弦相似度
    - 分类准确率（基于与所有文本特征的相似度）
    - 每个样本的最高相似度与最低相似度之差
    """
    # 所有类别的文本特征矩阵 (num_classes, feature_dim)，已在 CPU
    text_feat_all = clip_text_features_norm.cpu()  # 确保是 torch tensor

    all_records = []  # 存储每个样本的统计信息

    for client_id in range(num_clients):
        model_path = os.path.join(model_path_dr, f"Client_{client_id}_model.pt")
        if not os.path.exists(model_path):
            print(f"客户端 {client_id} 模型不存在，跳过")
            continue

        try:
            model = torch.load(model_path, map_location='cpu')
            model.eval()
        except Exception as e:
            print(f"加载客户端 {client_id} 模型失败: {e}")
            continue

        # 加载该客户端的训练数据
        dataloader = load_train_data(str(client_id), dataset_type, batch_size=64)

        client_correct = 0
        client_total = 0
        client_true_sims = []
        client_max_min_diffs = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to('cpu')
                features = model.base(images)
                features_norm = F.normalize(features, dim=-1)  # [B, D]

                # 计算与所有类别的相似度矩阵 [B, num_classes]
                sim_matrix = torch.mm(features_norm, text_feat_all.T)  # 点积即余弦相似度
                sim_matrix_np = sim_matrix.cpu().numpy()

                labels_np = labels.cpu().numpy()

                for i in range(len(labels_np)):
                    label = labels_np[i]
                    sim_i = sim_matrix_np[i]

                    true_sim = sim_i[label]
                    max_sim = np.max(sim_i)
                    min_sim = np.min(sim_i)
                    pred_label = np.argmax(sim_i)
                    correct = (pred_label == label)
                    max_min_diff = max_sim - min_sim

                    # 记录样本信息
                    all_records.append({
                        'client_id': client_id,
                        'label': label,
                        'class_name': cifar100_classes[label],
                        'true_similarity': true_sim,
                        'max_similarity': max_sim,
                        'min_similarity': min_sim,
                        'pred_label': pred_label,
                        'correct': correct,
                        'max_min_diff': max_min_diff
                    })

                    # 用于客户端实时统计
                    client_correct += correct
                    client_total += 1
                    client_true_sims.append(true_sim)
                    client_max_min_diffs.append(max_min_diff)

        if client_total > 0:
            print(f"客户端 {client_id}: 样本数={client_total}, 准确率={client_correct/client_total:.4f}, "
                  f"平均真实相似度={np.mean(client_true_sims):.4f}, "
                  f"平均max-min差={np.mean(client_max_min_diffs):.4f}")

    if not all_records:
        print("未收集到任何数据")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(all_records)

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始数据
    csv_path = os.path.join(save_dir, 'alignment_analysis.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n详细数据已保存至: {csv_path}")

    # ========== 整体统计 ==========
    print("\n=== 整体统计 ===")
    overall_acc = df['correct'].mean()
    print(f"整体准确率: {overall_acc:.4f}")
    print(f"平均真实相似度: {df['true_similarity'].mean():.4f}")
    print(f"平均max-min差: {df['max_min_diff'].mean():.4f}")

    # ========== 按客户端统计 ==========
    client_stats = df.groupby('client_id').agg(
        accuracy=('correct', 'mean'),
        mean_true_sim=('true_similarity', 'mean'),
        mean_max_min_diff=('max_min_diff', 'mean'),
        count=('correct', 'count')
    ).round(4)
    print("\n=== 各客户端统计 ===")
    print(client_stats)

    # ========== 按类别统计 ==========
    class_stats = df.groupby('class_name').agg(
        accuracy=('correct', 'mean'),
        mean_true_sim=('true_similarity', 'mean'),
        mean_max_min_diff=('max_min_diff', 'mean'),
        count=('correct', 'count')
    ).round(4)
    print("\n=== 各类别统计（前10类）===")
    print(class_stats.sort_values('count', ascending=False).head(10))

    # ========== 可视化 ==========
    plt.figure(figsize=(15, 10))

    # 1. 准确率分布（按客户端）
    plt.subplot(2, 3, 1)
    client_acc = client_stats['accuracy'].sort_index()
    plt.bar(client_acc.index.astype(str), client_acc.values)
    plt.xlabel('Client ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Client')
    plt.xticks(rotation=45)

    # 2. 真实相似度分布直方图
    plt.subplot(2, 3, 2)
    plt.hist(df['true_similarity'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('True Class Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of True Similarity')

    # 3. max-min差分布直方图
    plt.subplot(2, 3, 3)
    plt.hist(df['max_min_diff'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Max-Min Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Max-Min Difference')

    # 4. 按客户端的max-min差箱线图
    plt.subplot(2, 3, 4)
    client_diff_data = [df[df['client_id'] == cid]['max_min_diff'].values 
                        for cid in sorted(df['client_id'].unique())]
    plt.boxplot(client_diff_data, labels=sorted(df['client_id'].unique()))
    plt.xlabel('Client ID')
    plt.ylabel('Max-Min Difference')
    plt.title('Max-Min Difference per Client')
    plt.xticks(rotation=45)

    # 5. 真实相似度 vs max-min差散点图（按预测是否正确着色）
    plt.subplot(2, 3, 5)
    colors = df['correct'].map({True: 'green', False: 'red'})
    plt.scatter(df['true_similarity'], df['max_min_diff'], c=colors, alpha=0.3, s=10)
    plt.xlabel('True Similarity')
    plt.ylabel('Max-Min Difference')
    plt.title('True Sim vs Max-Min Diff (green=correct)')

    # 6. 各类别平均准确率（仅显示样本数前15的类别）
    plt.subplot(2, 3, 6)
    top_classes = class_stats.sort_values('count', ascending=False).head(15).index
    top_acc = class_stats.loc[top_classes, 'accuracy']
    plt.barh(range(len(top_acc)), top_acc.values[::-1])
    plt.yticks(range(len(top_acc)), top_acc.index[::-1])
    plt.xlabel('Accuracy')
    plt.title('Top 15 Classes by Sample Count')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'alignment_analysis_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化图已保存: {plot_path}")

    return df

def compute_similarity(model_path_dr, num_clients, dataset_type, save_dir):
    """
    计算所有客户端图像特征与对应 CLIP 文本特征的余弦相似度
    """
    all_sims = []
    all_labels = []
    all_client_ids = []

    for client_id in range(num_clients):
        model_path = os.path.join(model_path_dr, f"Client_{client_id}_model.pt")
        if not os.path.exists(model_path):
            print(f"客户端 {client_id} 模型不存在，跳过")
            continue

        try:
            model = torch.load(model_path, map_location='cpu')
            model.eval()
        except Exception as e:
            print(f"加载客户端 {client_id} 模型失败: {e}")
            continue

        # 加载该客户端的数据（训练集）
        dataloader = load_train_data(str(client_id), dataset_type, batch_size=64)

        client_sims = []
        client_labels = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to('cpu')  # 模型已在 cpu，数据也放在 cpu
                features = model.base(images)
                features_norm = F.normalize(features, dim=-1)  # [B, D]

                # 获取对应类别的文本特征（形状 [B, D]）
                text_feat = clip_text_features_norm[labels].cpu().numpy()  # 已在 cpu

                # 余弦相似度（点积）
                cos_sim = (features_norm * text_feat).sum(dim=-1)  # [B]
                client_sims.append(cos_sim.cpu().numpy())
                client_labels.append(labels.cpu().numpy())

        if client_sims:
            client_sims = np.concatenate(client_sims)
            client_labels = np.concatenate(client_labels)
            all_sims.append(client_sims)
            all_labels.append(client_labels)
            all_client_ids.extend([client_id] * len(client_labels))
            print(f"客户端 {client_id}: {len(client_labels)} 样本，平均相似度 = {client_sims.mean():.4f}")

    if not all_sims:
        print("未收集到任何相似度数据")
        return

    # 合并所有数据
    all_sims = np.concatenate(all_sims)
    all_labels = np.concatenate(all_labels)
    all_client_ids = np.array(all_client_ids)

    # 转换为 DataFrame 便于分析
    df = pd.DataFrame({
        'client_id': all_client_ids,
        'label': all_labels,
        'class_name': [cifar100_classes[l] for l in all_labels],
        'cosine_similarity': all_sims
    })

    # 保存原始数据
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'cosine_similarities.csv'), index=False)

    # 1. 整体统计
    print("\n=== 整体统计 ===")
    print(f"平均余弦相似度: {df['cosine_similarity'].mean():.4f}")
    print(f"标准差: {df['cosine_similarity'].std():.4f}")
    print(f"中位数: {df['cosine_similarity'].median():.4f}")

    # 2. 按客户端统计
    client_stats = df.groupby('client_id')['cosine_similarity'].agg(['mean', 'std', 'count'])
    print("\n=== 各客户端统计 ===")
    print(client_stats)

    # 3. 按类别统计
    class_stats = df.groupby('class_name')['cosine_similarity'].agg(['mean', 'std', 'count'])
    print("\n=== 各类别统计 ===")
    print(class_stats)

    # 4. 可视化分布
    plt.figure(figsize=(12, 5))

    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(df['cosine_similarity'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Cosine Similarity')

    # 按客户端箱线图
    plt.subplot(1, 2, 2)
    # 按 client_id 分组准备数据
    client_data = [df[df['client_id'] == cid]['cosine_similarity'].values for cid in sorted(df['client_id'].unique())]
    plt.boxplot(client_data, labels=sorted(df['client_id'].unique()))
    plt.xlabel('Client ID')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity per Client')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_similarity_distribution.png'), dpi=150)
    plt.close()

    # 5. 按类别箱线图（可选，类别太多可只选部分）
    # 选取样本数较多的前 20 个类别
    top_classes = class_stats.sort_values('count', ascending=False).head(20).index
    top_df = df[df['class_name'].isin(top_classes)]
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=top_df, x='class_name', y='cosine_similarity')
    plt.xticks(rotation=90)
    plt.title('Cosine Similarity per Class (Top 20 classes)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cosine_similarity_by_class.png'), dpi=150)
    plt.close()

    print(f"\n结果已保存至: {save_dir}")
    return df

def load_train_data(id, dataset, batch_size=64):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=64):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 使用统一表征空间的t-SNE  system/temp/Cifar100/FedDAR/1765289055.04626
    # print("开始统一t-SNE降维...")
    # unified_results = unified_tsne_across_clients(model_path_dr="./temp/Cifar100/FedCLIP/1773662794.820246",
    #                             num_clients=20,  
    #                             excel_result_dir="./resnet/FedCLIP/Freezen_classfer",)

    # if unified_results is not None:
    #     print("\n统一t-SNE处理完成!")
    #     print(f"总共处理了 {len(unified_results)} 个样本")
    #     print(f"涉及客户端: {sorted(unified_results['client_id'].unique())}")
    # else:
    #     print("\n统一t-SNE处理失败")
        
    # 计算余弦相似度
    similarity_df = compute_similarity(
        model_path_dr="./temp/Cifar100/FedCLIP/1773662794.820246",
        num_clients=20,
        dataset_type="Cifar100",
        save_dir="./resnet/FedCLIP/Freezen_classfer/cosine_stats"
    )
    df = compute_alignment_similarity(
        model_path_dr="./temp/Cifar100/FedCLIP/1773662794.820246",
        num_clients=20,
        dataset_type="Cifar100",
        save_dir="./resnet/FedCLIP/Freezen_classfer/alignment_stats"
    )