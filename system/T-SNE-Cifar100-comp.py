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
from pathlib import Path
from matplotlib.cm import get_cmap

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

# 算法配置
ALGORITHM_CONFIGS = {
    'FedGen': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedGen/CIFAR100/Hetero/test"
    },
    'FD': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FD/CIFAR100/Hetero/test"
    },
    'FedGH': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedGH/CIFAR100/Hetero/test"
    },
    'FedKD': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedKD/CIFAR100/Hetero/test"
    },
    'FedMRL': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedMRL/CIFAR100/Hetero/test"
    },
    'FedProto': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedProto/CIFAR100/Hetero/test"
    },
    'FedTGP': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedTGP/CIFAR100/Hetero/test"
    },
    'PFedAFM': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/PFedAFM/CIFAR100/Hetero/test"
    },
    'LG-FedAvg': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/LG-FedAvg/CIFAR100/Hetero/test"
    },
    # 'FedSPU': {
    #     'color': '#1f77b4',
    #     'marker': 'o',
    #     'result_dir': "./T-SNE/FedSPU/CIFAR100/Hetero/test"
    # }, 
    'FedDAR': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/test"
    },       
}

# ALGORITHM_CONFIGS = {
#     'CE': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/CE"
#     },
#     'CE_CON': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/CE_CON"
#     }, 
#     'CE_MSE': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/CE_MSE"
#     },  
#     'CE_MSE_CON': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/CE_CON_MSE"
#     },          
# }



# # 算法配置
# ALGORITHM_CONFIGS = {


#     'FedProto': {
#         'color': '#1f77b4',
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedProto/CIFAR100/Hetero/test"
#     },
#     'FedDAR': {
#         'color': '#1f77b4',
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR100/Hetero/test"
#     },       
# }


# def visualize_multiple_algorithms(algorithm_configs, 
#                                   save_path="./T-SNE/DAT/CIFAR100/Hetero/all_algorithms_tsne.png",
#                                   num_clients_per_algorithm=20,
#                                   max_clients_per_row=10,
#                                   figsize_per_subplot=(4, 4)):
#     """
#     绘制多个算法的客户端T-SNE图，每个算法一行
    
#     Args:
#         algorithm_configs: 算法配置字典
#         save_path: 保存合并图的路径
#         num_clients_per_algorithm: 每个算法显示的客户端数量
#         max_clients_per_row: 每行最多显示的客户端数量
#         figsize_per_subplot: 每个子图的尺寸
#     """
    
#     # 获取算法名称列表
#     algorithm_names = list(algorithm_configs.keys())
#     num_algorithms = len(algorithm_names)
    
#     # 确定每行显示的客户端数量
#     num_clients_to_display = min(num_clients_per_algorithm, max_clients_per_row)
    
#     # 创建图形，行数为算法数量，列数为客户端数量
#     fig, axes = plt.subplots(num_algorithms, num_clients_to_display, 
#                             figsize=(figsize_per_subplot[0] * num_clients_to_display,
#                                     figsize_per_subplot[1] * num_algorithms))
    
#     # 如果只有一个算法，确保axes是二维数组
#     if num_algorithms == 1:
#         axes = axes.reshape(1, -1)
#     if num_clients_to_display == 1:
#         axes = axes.reshape(-1, 1)
    
#     # 为所有算法设置一致的颜色映射（按类别）
#     class_colors = {}
#     # 使用tab20颜色映射，它可以提供20种不同的颜色
#     # 对于100个类别，我们可以重复使用这20种颜色
#     color_palette = plt.cm.tab20
    
#     for i, class_name in enumerate(cifar100_classes):
#         # 使用模运算来循环使用20种颜色
#         class_colors[class_name] = color_palette(i % 20)
    
#     # 遍历所有算法
#     for algo_idx, algo_name in enumerate(algorithm_names):
#         algo_config = algorithm_configs[algo_name]
#         result_dir = algo_config.get('result_dir')
        
#         if not result_dir or not os.path.exists(result_dir):
#             print(f"警告: 算法 {algo_name} 的结果目录不存在: {result_dir}")
#             continue
        
#         # 获取该算法的所有客户端CSV文件
#         client_files = [f for f in os.listdir(result_dir) if f.endswith('_T-SNE.csv')]
#         client_ids = sorted([int(f.split('_')[0]) for f in client_files if f.split('_')[0].isdigit()])
        
#         # 只取前num_clients_per_algorithm个客户端
#         client_ids = [i for i in client_ids if i < num_clients_per_algorithm]
#         client_ids = client_ids[:num_clients_to_display]  # 限制每行显示的客户端数量
        
#         print(f"算法 {algo_name}: 找到 {len(client_ids)} 个客户端的数据")
        
#         # 遍历该算法的所有客户端
#         for col_idx, client_id in enumerate(client_ids):
#             try:
#                 # 读取客户端数据
#                 csv_path = os.path.join(result_dir, f"{client_id}_T-SNE.csv")
#                 df = pd.read_csv(csv_path)
                
#                 # 获取当前子图
#                 ax = axes[algo_idx, col_idx]
                
#                 # 按类别绘制散点图
#                 # 由于类别太多（100个），我们可以只绘制存在数据的类别
#                 unique_classes = df['class_name'].unique()
#                 for class_name in unique_classes:
#                     class_data = df[df['class_name'] == class_name]
#                     if len(class_data) > 0:
#                         ax.scatter(class_data['t-SNE_dim1'], 
#                                   class_data['t-SNE_dim2'],
#                                   c=[class_colors[class_name]],
#                                   label=class_name,
#                                   alpha=0.6, 
#                                   s=8,
#                                   edgecolors='w', 
#                                   linewidth=0.3)
                
#                 # 设置标题和坐标轴
#                 if col_idx == 0:  # 每行的第一个子图
#                     ax.set_ylabel(f'{algo_name}\nDimension 2', fontsize=10)
#                 else:
#                     ax.set_ylabel('')
                
#                 if algo_idx == num_algorithms - 1:  # 最后一行的子图
#                     ax.set_xlabel(f'Client {client_id}\nDimension 1', fontsize=10)
#                 else:
#                     ax.set_xlabel('')
                
#                 # 移除坐标轴刻度
#                 ax.set_xticks([])
#                 ax.set_yticks([])
                
#                 # 添加网格
#                 ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
#                 # 设置子图边界
#                 ax.spines['top'].set_visible(False)
#                 ax.spines['right'].set_visible(False)
                
#             except Exception as e:
#                 print(f"处理算法 {algo_name} 客户端 {client_id} 失败: {e}")
#                 ax = axes[algo_idx, col_idx]
#                 ax.text(0.5, 0.5, f'Data\nMissing', 
#                        horizontalalignment='center', verticalalignment='center',
#                        transform=ax.transAxes, fontsize=8, color='red')
#                 ax.set_xticks([])
#                 ax.set_yticks([])
        
#         # 如果该算法的客户端数量不足，填充空白子图
#         for col_idx in range(len(client_ids), num_clients_to_display):
#             ax = axes[algo_idx, col_idx]
#             ax.axis('off')
    
#     # 调整布局
#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为总标题留出空间
    
#     # 添加总标题（修复：从CIFAR-10改为CIFAR-100）
#     fig.suptitle('T-SNE Visualization Across Algorithms and Clients (CIFAR-100)', 
#                 fontsize=14, fontweight='bold', y=0.99)
    
#     # 由于有100个类别，图例会非常庞大，所以不添加图例
#     # 如果需要图例，可以单独保存或只显示部分类别
    
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"\n多个算法的T-SNE可视化图已保存: {save_path}")
    
#     return fig

def visualize_multiple_algorithms(algorithm_configs, 
                                  save_path="./T-SNE/DAT/CIFAR100/Hetero/all_algorithms_tsne.png",
                                  num_clients_per_algorithm=20,
                                  max_clients_per_row=10,
                                  figsize_per_subplot=(4, 4)):
    """
    绘制多个算法的客户端T-SNE图，每个算法一行
    
    Args:
        algorithm_configs: 算法配置字典
        save_path: 保存合并图的路径
        num_clients_per_algorithm: 每个算法显示的客户端数量
        max_clients_per_row: 每行最多显示的客户端数量
        figsize_per_subplot: 每个子图的尺寸
    """
    
    # 获取算法名称列表
    algorithm_names = list(algorithm_configs.keys())
    num_algorithms = len(algorithm_names)
    
    # 确定每行显示的客户端数量
    num_clients_to_display = min(num_clients_per_algorithm, max_clients_per_row)
    
    # 创建图形，行数为算法数量，列数为客户端数量
    fig, axes = plt.subplots(num_algorithms, num_clients_to_display, 
                            figsize=(figsize_per_subplot[0] * num_clients_to_display,
                                    figsize_per_subplot[1] * num_algorithms))
# 如果没有数据，使用默认范围
    x_min, x_max = -50, 50
    y_min, y_max = -50, 50
    # 如果只有一个算法，确保axes是二维数组
    if num_algorithms == 1:
        axes = axes.reshape(1, -1)
    if num_clients_to_display == 1:
        axes = axes.reshape(-1, 1)
    
    # 为所有算法设置一致的颜色映射（按类别）
    class_colors = {}
    # 使用tab20颜色映射，它可以提供20种不同的颜色
    # 对于100个类别，我们可以重复使用这20种颜色
    color_palette = plt.cm.tab20
    
    for i, class_name in enumerate(cifar100_classes):
        # 使用模运算来循环使用20种颜色
        class_colors[class_name] = color_palette(i % 20)
    
    # 遍历所有算法
    for algo_idx, algo_name in enumerate(algorithm_names):
        algo_config = algorithm_configs[algo_name]
        result_dir = algo_config.get('result_dir')
        
        if not result_dir or not os.path.exists(result_dir):
            print(f"警告: 算法 {algo_name} 的结果目录不存在: {result_dir}")
            continue
        
        # 获取该算法的所有客户端CSV文件
        client_files = [f for f in os.listdir(result_dir) if f.endswith('_T-SNE.csv')]
        client_ids = sorted([int(f.split('_')[0]) for f in client_files if f.split('_')[0].isdigit()])
        
        # 只取前num_clients_per_algorithm个客户端
        client_ids = [i for i in client_ids if i < num_clients_per_algorithm]
        client_ids = client_ids[:num_clients_to_display]  # 限制每行显示的客户端数量
        
        print(f"算法 {algo_name}: 找到 {len(client_ids)} 个客户端的数据")
        
        # 遍历该算法的所有客户端
        for col_idx, client_id in enumerate(client_ids):
            try:
                # 读取客户端数据
                csv_path = os.path.join(result_dir, f"{client_id}_T-SNE.csv")
                df = pd.read_csv(csv_path)
                
                # 获取当前子图
                ax = axes[algo_idx, col_idx]
                
                # 按类别绘制散点图
                # 由于类别太多（100个），我们可以只绘制存在数据的类别
                unique_classes = df['class_name'].unique()
                for class_name in unique_classes:
                    class_data = df[df['class_name'] == class_name]
                    if len(class_data) > 0:
                        ax.scatter(class_data['t-SNE_dim1'], 
                                  class_data['t-SNE_dim2'],
                                  c=[class_colors[class_name]],
                                  label=class_name,
                                  alpha=0.6, 
                                  s=8,
                                  edgecolors='w', 
                                  linewidth=0.3)

                # 设置统一的坐标轴范围
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)               
                # 设置标题和坐标轴
                if col_idx == 0:  # 每行的第一个子图
                    ax.set_ylabel(f'{algo_name}\nDimension 2', fontsize=10)
                else:
                    ax.set_ylabel('')
                
                if algo_idx == num_algorithms - 1:  # 最后一行的子图
                    ax.set_xlabel(f'Client {client_id}\nDimension 1', fontsize=10)
                else:
                    ax.set_xlabel('')
                
                # 移除坐标轴刻度
                ax.set_xticks([])
                ax.set_yticks([])
                
                # 添加网格
                ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
                
                # 设置子图边界
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
            except Exception as e:
                print(f"处理算法 {algo_name} 客户端 {client_id} 失败: {e}")
                ax = axes[algo_idx, col_idx]
                ax.text(0.5, 0.5, f'Data\nMissing', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=8, color='red')
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 如果该算法的客户端数量不足，填充空白子图
        for col_idx in range(len(client_ids), num_clients_to_display):
            ax = axes[algo_idx, col_idx]
            ax.axis('off')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # 为总标题留出空间
    
    # 添加总标题（修复：从CIFAR-10改为CIFAR-100）
    fig.suptitle('T-SNE Visualization Across Algorithms and Clients (CIFAR-100)', 
                fontsize=14, fontweight='bold', y=0.99)
    
    # 由于有100个类别，图例会非常庞大，所以不添加图例
    # 如果需要图例，可以单独保存或只显示部分类别
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n多个算法的T-SNE可视化图已保存: {save_path}")
    
    return fig

def create_algorithm_comparison_figure(algorithm_configs, 
                                      save_path="./T-SNE/DAT/CIFAR100/Hetero/algorithm_comparison.pdf",
                                      sample_client_id=0,
                                      num_samples_per_class=50):
    """
    创建算法对比图，显示同一客户端在不同算法下的表现
    
    Args:
        algorithm_configs: 算法配置字典
        save_path: 保存对比图的路径
        sample_client_id: 采样的客户端ID
        num_samples_per_class: 每类采样的样本数量
    """
    
    num_algorithms = len(algorithm_configs)
    
    # 创建图形，每个算法一列
    fig, axes = plt.subplots(1, num_algorithms, figsize=(5*num_algorithms, 5))
    
    if num_algorithms == 1:
        axes = [axes]
    
    # 为所有算法设置一致的颜色映射
    class_colors = {}
    color_palette = plt.cm.tab20
    
    for i, class_name in enumerate(cifar100_classes):
        class_colors[class_name] = color_palette(i % 20)
    
    # 遍历所有算法
    for idx, (algo_name, algo_config) in enumerate(algorithm_configs.items()):
        result_dir = algo_config.get('result_dir')
        
        if not result_dir or not os.path.exists(result_dir):
            print(f"警告: 算法 {algo_name} 的结果目录不存在: {result_dir}")
            continue
        
        try:
            # 读取指定客户端的数据
            csv_path = os.path.join(result_dir, f"{sample_client_id}_T-SNE.csv")
            df = pd.read_csv(csv_path)
            
            # 获取当前子图
            ax = axes[idx]
            
            # 按类别绘制散点图
            # 只绘制存在数据的类别
            unique_classes = df['class_name'].unique()
            for class_name in unique_classes:
                class_data = df[df['class_name'] == class_name]
                if len(class_data) > 0:
                    # 如果数据太多，可以采样显示
                    if len(class_data) > num_samples_per_class:
                        class_data = class_data.sample(n=num_samples_per_class, random_state=RANDOM_SEED)
                    
                    ax.scatter(class_data['t-SNE_dim1'], 
                              class_data['t-SNE_dim2'],
                              c=[class_colors[class_name]],
                              label=class_name if idx == 0 else "",
                              alpha=0.7, 
                              s=15,
                              edgecolors='w', 
                              linewidth=0.5)
            
            # 设置标题和坐标轴
            ax.set_title(f'{algo_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
            if idx == 0:
                ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
        except Exception as e:
            print(f"处理算法 {algo_name} 失败: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f'{algo_name}\nData Missing', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为总标题留出空间
    
    # 添加总标题（修复：从CIFAR-10改为CIFAR-100）
    fig.suptitle(f'T-SNE Comparison for Client {sample_client_id} Across Algorithms (CIFAR-100)', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # 由于有100个类别，图例会非常庞大，所以不添加图例
    # 如果需要图例，可以单独保存或只显示部分类别
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n算法对比图已保存: {save_path}")
    
    return fig


if __name__ == "__main__":
    
    # 基本路径配置
    base_result_dir = "./T-SNE/"
    
    # 确保输出目录存在
    os.makedirs(base_result_dir, exist_ok=True)
    
    # 步骤1: 绘制多个算法的客户端T-SNE图（每个算法一行）
    print("\n生成多个算法的T-SNE合并图...")
    save_path_multi = os.path.join(base_result_dir, "CIFAR100-test-all_algorithms_clients_tsne.png")
    fig_multi = visualize_multiple_algorithms(
        algorithm_configs=ALGORITHM_CONFIGS,
        save_path=save_path_multi,
        num_clients_per_algorithm=20,  # 每个算法显示20个客户端
        max_clients_per_row=20,  # 每行最多显示20个客户端
        figsize_per_subplot=(3, 3)  # 每个子图的尺寸
    )
    
    print("\n处理完成!")