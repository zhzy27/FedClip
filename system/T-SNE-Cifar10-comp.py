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

# 算法配置 - 可以在这里添加更多算法
# ALGORITHM_CONFIGS = {
#     'FedGen': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedGen/CIFAR10/Hetero/Each"
#     },
#     'FD': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FD/CIFAR10/Hetero/Each"
#     },
#     'FedGH': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedGH/CIFAR10/Hetero/Each"
#     },
#     'FedKD': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedKD/CIFAR10/Hetero/Each"
#     },
#     'FedMRL': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedMRL/CIFAR10/Hetero/Each"
#     },
#     'FedProto': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedProto/CIFAR10/Hetero/Each"
#     },
#     'FedTGP': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedTGP/CIFAR10/Hetero/Each"
#     },
#     'FML': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FML/CIFAR10/Hetero/Each"
#     },
#     'PFedAFM': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/PFedAFM/CIFAR10/Hetero/Each"
#     },
#     'LG-FedAvg': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/LG-FedAvg/CIFAR10/Hetero/Each"
#     },
#     'FedSPU': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedSPU/CIFAR10/Hetero/each"
#     }, 
#     'FedDAR': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/Each"
#     },       
# }
# 算法配置
ALGORITHM_CONFIGS = {


    'FedProto': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedProto/CIFAR10/Hetero/Test"
    },
    'FedDAR': {
        'color': '#1f77b4',
        'marker': 'o',
        'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/Test"
    },       
}

# ALGORITHM_CONFIGS = {
#     'CE': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/CE"
#     },
#     'CE_CON': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/CE_CON"
#     }, 
#     'CE_MSE': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/CE_MSE"
#     },  
#     'CE_MSE_CON': {
#         'color': '#1f77b4',  # 蓝色
#         'marker': 'o',
#         'result_dir': "./T-SNE/FedDAR/CIFAR10/Hetero/CE_CON_MSE"
#     },          
# }
# def visualize_multiple_algorithms(algorithm_configs, 
#                                   save_path="./T-SNE/DAT/CIFAR10/Hetero/all_algorithms_tsne.png",
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
#     color_palette = plt.cm.tab10  # 使用tab10颜色映射
    
#     for i, class_name in enumerate(cifar10_classes):
#         class_colors[class_name] = color_palette(i/len(cifar10_classes))
    
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
#                 for class_name in cifar10_classes:
#                     class_data = df[df['class_name'] == class_name]
#                     if len(class_data) > 0:
#                         ax.scatter(class_data['t-SNE_dim1'], 
#                                   class_data['t-SNE_dim2'],
#                                   c=[class_colors[class_name]],
#                                   label=class_name if (algo_idx == 0 and col_idx == 0) else "",
#                                   alpha=0.6, 
#                                   s=8,  # 稍微减小点大小
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
    
#     # 添加总标题
#     fig.suptitle('T-SNE Visualization Across Algorithms and Clients (CIFAR-10)', 
#                 fontsize=14, fontweight='bold', y=0.99)
    
#     # 添加图例
#     if num_algorithms > 0:
#         # 从第一个子图获取图例句柄
#         handles, labels = axes[0, 0].get_legend_handles_labels()
        
#         # 创建分类图例
#         if handles and labels:
#             # 创建类别图例（放在图上方）
#             fig.legend(handles, labels, 
#                       loc='upper center', 
#                       bbox_to_anchor=(0.5, 0.02),  # 放在图下方
#                       ncol=5,  # 5列布局
#                       fontsize=9,
#                       frameon=True,
#                       fancybox=True,
#                       shadow=False,
#                       title='Classes',
#                       title_fontsize=10)
    
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"\n多个算法的T-SNE可视化图已保存: {save_path}")
    
#     return fig
def visualize_multiple_algorithms(algorithm_configs, 
                                  save_path="./T-SNE/DAT/CIFAR10/Hetero/all_algorithms_tsne.png",
                                  num_clients_per_algorithm=20,
                                  max_clients_per_row=10,
                                  figsize_per_subplot=(4, 4),
                                  fixed_axis_range=(-100, 100)):
    """
    绘制多个算法的客户端T-SNE图，每个算法一行，使用统一的坐标轴刻度
    
    Args:
        algorithm_configs: 算法配置字典
        save_path: 保存合并图的路径
        num_clients_per_algorithm: 每个算法显示的客户端数量
        max_clients_per_row: 每行最多显示的客户端数量
        figsize_per_subplot: 每个子图的尺寸
        fixed_axis_range: 固定坐标轴范围，如(-100, 100)
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
    
    # 如果只有一个算法，确保axes是二维数组
    if num_algorithms == 1:
        axes = axes.reshape(1, -1)
    if num_clients_to_display == 1:
        axes = axes.reshape(-1, 1)
    
    # 为所有算法设置一致的颜色映射（按类别）
    class_colors = {}
    color_palette = plt.cm.tab10
    
    for i, class_name in enumerate(cifar10_classes):
        class_colors[class_name] = color_palette(i/len(cifar10_classes))

# 如果没有数据，使用默认范围
    x_min, x_max = -100, 100
    y_min, y_max = -100, 100
    
    # 遍历所有算法进行绘图
    for algo_idx, algo_name in enumerate(algorithm_names):
        algo_config = algorithm_configs[algo_name]
        result_dir = algo_config.get('result_dir')
        
        if not result_dir or not os.path.exists(result_dir):
            print(f"警告: 算法 {algo_name} 的结果目录不存在: {result_dir}")
            # 填充空白子图
            for col_idx in range(num_clients_to_display):
                ax = axes[algo_idx, col_idx]
                ax.axis('off')
            continue
        
        # 获取该算法的所有客户端CSV文件
        client_files = [f for f in os.listdir(result_dir) if f.endswith('_T-SNE.csv')]
        client_ids = sorted([int(f.split('_')[0]) for f in client_files if f.split('_')[0].isdigit()])
        
        # 只取前num_clients_per_algorithm个客户端
        client_ids = [i for i in client_ids if i < num_clients_per_algorithm]
        client_ids = client_ids[:num_clients_to_display]
        
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
                for class_name in cifar10_classes:
                    class_data = df[df['class_name'] == class_name]
                    if len(class_data) > 0:
                        ax.scatter(class_data['t-SNE_dim1'], 
                                  class_data['t-SNE_dim2'],
                                  c=[class_colors[class_name]],
                                  label=class_name if (algo_idx == 0 and col_idx == 0) else "",
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
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 如果该算法的客户端数量不足，填充空白子图
        for col_idx in range(len(client_ids), num_clients_to_display):
            ax = axes[algo_idx, col_idx]
            ax.axis('off')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # 添加总标题
    fig.suptitle('T-SNE Visualization Across Algorithms and Clients (CIFAR-10)', 
                fontsize=14, fontweight='bold', y=0.99)
    
    # 添加图例
    if num_algorithms > 0:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles and labels:
            fig.legend(handles, labels, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02),
                      ncol=5,
                      fontsize=9,
                      frameon=True,
                      fancybox=True,
                      shadow=False,
                      title='Classes',
                      title_fontsize=10)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n多个算法的T-SNE可视化图已保存: {save_path}")
    
    return fig

def create_algorithm_comparison_figure(algorithm_configs, 
                                      save_path="./T-SNE/DAT/CIFAR10/Hetero/algorithm_comparison.pdf",
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
    color_palette = plt.cm.tab10
    
    for i, class_name in enumerate(cifar10_classes):
        class_colors[class_name] = color_palette(i/len(cifar10_classes))
    
    # 遍历所有算法
    for idx, (algo_name, algo_config) in enumerate(algorithm_configs.items()):
        result_dir = algo_config.get('result_dir')
        algo_color = algo_config.get('color', color_palette(idx/num_algorithms))
        
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
            for class_name in cifar10_classes:
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
    
    # 添加总标题
    fig.suptitle(f'T-SNE Comparison for Client {sample_client_id} Across Algorithms', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # 添加图例
    if num_algorithms > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles and labels:
            fig.legend(handles, labels, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 0.02),  # 放在图下方
                      ncol=5,
                      fontsize=10,
                      frameon=True)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n算法对比图已保存: {save_path}")
    
    return fig


if __name__ == "__main__":
    
    # 基本路径配置
    base_result_dir = "./T-SNE/"
    
    
    # 步骤1: 绘制多个算法的客户端T-SNE图（每个算法一行）
    print("\n生成多个算法的T-SNE合并图...")
    save_path_multi = os.path.join(base_result_dir, "CIFAR10-Test-all_algorithms_clients_tsne.png")
    fig_multi = visualize_multiple_algorithms(
        algorithm_configs=ALGORITHM_CONFIGS,
        save_path=save_path_multi,
        num_clients_per_algorithm=20,  # 每个算法显示20个客户端
        max_clients_per_row=20,  # 每行最多显示20个客户端
        figsize_per_subplot=(3, 3)  # 每个子图的尺寸
    )
    
    # # 步骤2: 绘制算法对比图（同一客户端在不同算法下的表现）
    # print("\n生成算法对比图...")
    # save_path_comparison = os.path.join(base_result_dir, "algorithm_comparison_client0.pdf")
    # fig_comparison = create_algorithm_comparison_figure(
    #     algorithm_configs=ALGORITHM_CONFIGS,
    #     save_path=save_path_comparison,
    #     sample_client_id=0,  # 对比客户端0在所有算法下的表现
    #     num_samples_per_class=30  # 每类显示30个样本
    # )
    
    # # 步骤3: 可选 - 对每个算法单独生成一行图（如果算法太多，可以分开处理）
    # print("\n为每个算法单独生成一行T-SNE图...")
    # for algo_name, algo_config in ALGORITHM_CONFIGS.items():
    #     single_algo_config = {algo_name: algo_config}
    #     save_path_single = os.path.join(base_result_dir, f"{algo_name}_clients_tsne.pdf")
        
    #     try:
    #         fig_single = visualize_multiple_algorithms(
    #             algorithm_configs=single_algo_config,
    #             save_path=save_path_single,
    #             num_clients_per_algorithm=20,
    #             max_clients_per_row=10,
    #             figsize_per_subplot=(3, 3)
    #         )
    #         print(f"  算法 {algo_name} 的T-SNE图已保存: {save_path_single}")
    #     except Exception as e:
    #         print(f"  生成算法 {algo_name} 的T-SNE图失败: {e}")
    
    # print("\n所有处理完成!")