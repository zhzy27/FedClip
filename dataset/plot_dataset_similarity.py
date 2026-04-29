import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def calculate_iou_similarity(matrix):
    num_clients = matrix.shape[0]
    sim_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(num_clients):
            intersection = np.sum(np.minimum(matrix[i], matrix[j]))
            union = np.sum(np.maximum(matrix[i], matrix[j]))
            if union == 0:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = intersection / union
    return sim_matrix
# ================= 新增：非对称子集包含度计算 =================
def calculate_subset_similarity(matrix):
    """
    计算非对称的子集包含度 (Intersection over Target Size)
    matrix shape: (num_clients, num_classes)
    """
    num_clients = matrix.shape[0]
    sim_matrix = np.zeros((num_clients, num_clients))
    
    for i in range(num_clients): # i 是 Target (需要拉取参数的客户端)
        for j in range(num_clients): # j 是 Source (提供参数的客户端)
            intersection = np.sum(np.minimum(matrix[i], matrix[j]))
            target_size = np.sum(matrix[i]) # 核心修改：分母变成了目标自己原本的数据量
            
            if target_size == 0:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = intersection / target_size
    return sim_matrix
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Plot dataset distribution similarity heatmap")
    parser.add_argument('--dataset', type=str, default='Cifar10', help='Name of the dataset')
    parser.add_argument('--partition', type=str, default='dir', help='Partition strategy')
    parser.add_argument('--alpha', type=float, default=0.1, help='Heterogeneity parameter')
    parser.add_argument('--method', type=str, default='iou', choices=['iou', 'cosine', 'subset'], 
                        help='Similarity calculation method: "iou", "cosine", or "subset"')
    
    # ================= 新增：视图增强控制参数 =================
    parser.add_argument('--exclude_self', action='store_true', 
                        help='Set self-similarity (diagonal) to 0 to prevent self-dominance')
    parser.add_argument('--normalize', action='store_true', 
                        help='Normalize each row to sum to 1 (relative weight distribution among peers)')
    # ==========================================================
    args = parser.parse_args()

    # 1. 拼接 config.json 的路径
    folder_name = f"{args.partition}_{args.alpha}"
    config_path = os.path.join(args.dataset, folder_name, "config.json")

    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    print(f"✅ 正在读取配置文件: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    num_clients = config['num_clients']
    num_classes = config['num_classes']
    client_label_sizes = config['Size of samples for labels in clients']

    dist_matrix = np.zeros((num_clients, num_classes))
    for client_id, label_counts in enumerate(client_label_sizes):
        for item in label_counts:
            dist_matrix[client_id, item[0]] = item[1]

    # 2. 计算基础相似度
    if args.method == 'iou':
        sim_matrix = calculate_iou_similarity(dist_matrix)
        cmap_color = "YlOrRd"
        title_method = "IoU (Count-Aware)"
    elif args.method == 'subset':
        # 调用新增的非对称计算函数
        sim_matrix = calculate_subset_similarity(dist_matrix)
        cmap_color = "Purples" # 使用紫色系区分非对称模式
        title_method = "Subset (Asymmetric)"
    else:
        sim_matrix = cosine_similarity(dist_matrix)
        cmap_color = "YlGnBu"
        title_method = "Cosine (Ratio-Aware)"

    # ================= 核心逻辑：剔除自身与归一化 =================
    if args.exclude_self:
        np.fill_diagonal(sim_matrix, 0.0)
        title_method += " | Exclude Self"
        print("[-] 已将自身相似度(对角线)剔除(设为0)")

    if args.normalize:
        # 对每一行求和
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        # 防止除以0（如果有某行全为0）
        row_sums[row_sums == 0] = 1.0 
        # 归一化：将数值转化为在该行中的占比
        sim_matrix = sim_matrix / row_sums
        title_method += " | Row-Normalized"
        print("[-] 已对每一行进行归一化(使其余客户端相似度占比之和为1)")
    # ============================================================

    # 打印最终矩阵
    print(f"\n================ 最终计算的 {args.method.upper()} 相似度矩阵 ================")
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print(sim_matrix)
    print("======================================================================\n")

    # 3. 绘制热力图
    plt.figure(figsize=(12, 10))
    
    # 动态设定最大值：如果不归一化且不剔除自己，最大值强制设为1.0
    # 如果开启了增强视图，则设为 None，让 Seaborn 自动根据当前矩阵的实际最大值拉伸色彩对比度！
    vmax_val = 1.0 if not (args.exclude_self or args.normalize) else None
    
    sns.heatmap(sim_matrix, annot=False, cmap=cmap_color, vmin=0.0, vmax=vmax_val,
                xticklabels=range(num_clients), 
                yticklabels=range(num_clients))

    plt.title(f"Dataset {title_method} Similarity\n({args.dataset} - Partition: {args.partition}, Alpha: {args.alpha})", fontsize=16)
    plt.xlabel("Target Client (Others)", fontsize=14)
    plt.ylabel("Source Client (Self)", fontsize=14)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # 保存图片时的名字也加上标识，避免覆盖原始图
    save_tag = ""
    if args.exclude_self: save_tag += "_noself"
    if args.normalize: save_tag += "_norm"
    
    save_filename = f"{args.dataset}_{args.partition}_{args.alpha}_{args.method}{save_tag}_similarity.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 增强版热力图已成功生成并保存为: {os.path.abspath(save_filename)}")
    # ================= 新增：导出上帝视角权重给联邦系统使用 =================
    # 联邦训练代码一般在 system 目录下运行，所以我们导出到 ../system/Oracle_Weights 
    system_weight_dir = os.path.join("..", "system", "Oracle_Weights")
    os.makedirs(system_weight_dir, exist_ok=True)
    
    # 文件名与热力图名对齐
    weight_filename = f"{args.dataset}_{args.partition}_{args.alpha}_{args.method}{save_tag}_weights.txt"
    weight_filepath = os.path.join(system_weight_dir, weight_filename)
    
    # 保存矩阵 (保留 6 位小数，使用制表符或空格分隔)
    np.savetxt(weight_filepath, sim_matrix, fmt='%.6f')
    print(f"💾 上帝视角聚合权重已导出至: {os.path.abspath(weight_filepath)}")
    # ========================================================================

if __name__ == '__main__':
    main()


# 两种启动方式
# python plot_dataset_similarity.py --dataset Cifar10 --alpha 0.1 --method iou --exclude_self --normalize
# python plot_dataset_similarity.py --dataset Cifar10 --alpha 0.1 --method cosine --exclude_self --normalize
# python plot_dataset_similarity.py --dataset Cifar100 --alpha 0.1 --method iou --exclude_self --normalize
# python plot_dataset_similarity.py --dataset Cifar100 --alpha 0.1 --method cosine --exclude_self --normalize