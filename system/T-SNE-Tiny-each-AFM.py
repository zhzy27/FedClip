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
import pickle

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

def unified_tsne_across_clients(client_id,
                                model_path_dir="./temp/TinyImagenet/FedSPU/0.1",
                                excel_result_dir="./T-SNE/SPU/TinyImagenet/Hetero/Each",
                                dataset_type="TinyImagenet"):
    """
    所有客户端在统一表征空间进行t-SNE降维
    """
    model_path = os.path.join(model_path_dir,f"Client_{client_id}_model.pt")
    global_model_path = os.path.join(model_path_dir,f"Server_global_model.pt")
    alpha_model_path =  os.path.join(model_path_dir,f"Client_{client_id}_alpha.pt")
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在，跳过")
    try:
        model = torch.load(model_path, map_location='cpu')
        global_model = torch.load(global_model_path, map_location='cpu')
        alpha_model = torch.load(alpha_model_path, map_location='cpu')
        model.eval()
        global_model.eval()
        alpha_model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")

    try:
        client_dataloader = load_train_data(str(client_id), dataset_type)
        # client_dataloader = load_test_data(str(client_id), dataset_type)
        client_features = []
        client_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(client_dataloader):
                # if batch_idx >= 40:  # 限制每个客户端的数据量，避免内存不足
                #     break
                _,rep_g = global_model(images)
                alpha = alpha_model.alpha
                w_rep_g = alpha_model(rep_g)
                _,features = model(images,w_rep_g,alpha)
                client_features.append(features.numpy())
                client_labels.append(labels.numpy())
    except Exception as e:
        print(f"处理客户端 {client_id} 数据失败: {e}")

    combined_features = np.concatenate(client_features, axis=0)
    combined_labels = np.concatenate(client_labels, axis=0)
    print(f"客户端 {client_id} 总特征形状: {combined_features.shape}")
    print(f"客户端 {client_id} 总标签形状: {combined_labels.shape}")

    # 3. 统一进行t-SNE降维
    print(f"客户端 {client_id} 进行t-SNE降维...")
    try:
        # 调整t-SNE参数以适应TinyImageNet
        perplexity_value = min(30, len(combined_features) - 1)
        if perplexity_value < 5:
            perplexity_value = 5
            
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity_value,
            learning_rate='auto',
            random_state=RANDOM_SEED,
            max_iter=1500,
            verbose=1
        )
        unified_features_2d = tsne.fit_transform(combined_features)
        print(f"客户端 {client_id} t-SNE降维后形状: {unified_features_2d.shape}")
    except Exception as e:
        print(f"客户端 {client_id} t-SNE降维失败: {e}")
        return None

    # 4. 创建结果DataFrame
    # TinyImageNet标签是0-199
    unified_df = pd.DataFrame({
        'label': combined_labels,
        'class_name': [tinyimagenet_classes[int(label)] for label in combined_labels],
        't-SNE_dim1': unified_features_2d[:, 0],
        't-SNE_dim2': unified_features_2d[:, 1],
    })

    # 5. 保存结果
    os.makedirs(excel_result_dir, exist_ok=True)

    # 保存整体结果
    unified_excel_path = os.path.join(excel_result_dir, f"{client_id}_T-SNE.xlsx")
    unified_csv_path = os.path.join(excel_result_dir, f"{client_id}_T-SNE.csv")

    unified_df.to_excel(unified_excel_path, index=False)
    unified_df.to_csv(unified_csv_path, index=False)

    print(f"客户端 {client_id} t-SNE结果已保存:")
    print(f"Excel: {unified_excel_path}")
    print(f"CSV: {unified_csv_path}")

    # 7. 可视化
    visualize_unified_tsne(unified_df, excel_result_dir, client_id)

    return unified_df

def visualize_unified_tsne(tsne_df, save_dir,client_id):
    """可视化统一的t-SNE结果"""

    # 按客户端可视化
    plt.figure(figsize=(20, 16))

    # 子图1: 按类别着色
    plt.subplot(2, 2, 1)
    for class_name in tinyimagenet_classes:
        class_data = tsne_df[tsne_df['class_name'] == class_name]
        if len(class_data) > 0:
            plt.scatter(class_data['t-SNE_dim1'], class_data['t-SNE_dim2'],
                        label=class_name, alpha=0.6, s=20)
    plt.title('Unified t-SNE by Class')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{client_id}_tsne_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"统一t-SNE可视化图已保存: {plot_path}")

def visualize_all_clients_in_one_row(results_dir="./T-SNE/DAT/CIFAR10/Hetero/Each/CE_MSE", 
                                     save_path="./T-SNE/DAT/CIFAR10/Hetero/Each/CE/all_clients_row.pdf",
                                     num_clients=20):
    """
    将所有客户端的T-SNE可视化图排成一行
    
    Args:
        results_dir: 保存各客户端结果的目录
        save_path: 保存合并图的路径
        num_clients: 客户端数量
    """
    
    # 获取所有客户端的CSV文件
    client_files = [f for f in os.listdir(results_dir) if f.endswith('_T-SNE.csv')]
    client_ids = sorted([int(f.split('_')[0]) for f in client_files])
    
    # 只取前num_clients个客户端
    client_ids = [i for i in client_ids if i < num_clients]
    
    print(f"找到 {len(client_ids)} 个客户端的数据")
    
    # 创建一个大图，将所有客户端排成一行
    fig, axes = plt.subplots(1, len(client_ids), figsize=(5*len(client_ids), 5))
    
    # 如果只有一个客户端，确保axes是数组
    if len(client_ids) == 1:
        axes = [axes]
    
    # 为所有客户端设置一致的颜色映射
    color_palette = plt.cm.gist_ncar  # 或者使用 plt.cm.nipy_spectral
    
    # 创建类别到颜色的映射
    class_colors = {}
    for i in range(200):  
        class_colors[str(i)] = color_palette(i/200)  # 均匀分布颜色
    
    for i, class_name in enumerate(tinyimagenet_classes):
        class_colors[class_name] = color_palette(i/len(tinyimagenet_classes))
    
    # 遍历所有客户端并绘制
    for idx, client_id in enumerate(client_ids):
        try:
            # 读取客户端数据
            csv_path = os.path.join(results_dir, f"{client_id}_T-SNE.csv")
            df = pd.read_csv(csv_path)
            
            # 获取当前子图
            ax = axes[idx]
            
            # 按类别绘制散点图
            for class_name in tinyimagenet_classes:
                class_data = df[df['class_name'] == class_name]
                if len(class_data) > 0:
                    ax.scatter(class_data['t-SNE_dim1'], 
                              class_data['t-SNE_dim2'],
                              c=[class_colors[class_name]],  # 使用统一颜色
                              label=class_name if idx == 0 else "",  # 只在第一个图中显示图例
                              alpha=0.6, 
                              s=10,
                              edgecolors='w', 
                              linewidth=0.5)
            
            # 设置标题和坐标轴
            ax.set_title(f'Client {client_id}', fontsize=14, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1' if idx == len(client_ids)//2 else '', fontsize=10)
            ax.set_ylabel('t-SNE Dimension 2' if idx == 0 else '', fontsize=10)
            
            # 移除坐标轴刻度
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
        except Exception as e:
            print(f"处理客户端 {client_id} 失败: {e}")
            # 创建空的子图
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Client {client_id}\nData Missing', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Client {client_id}', fontsize=14)
    
    # 添加全局图例（放在第一张图的位置）
    if len(client_ids) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        
        # 创建图例的子图
        if len(client_ids) < 20:  # 如果客户端较少，可以将图例放在右侧
            fig.legend(handles, labels, 
                      loc='upper center', 
                      bbox_to_anchor=(0.5, 0.05),  # 放在底部中间
                      ncol=5,  # 5列布局
                      fontsize=10,
                      frameon=True,
                      fancybox=True,
                      shadow=True)
        else:  # 如果客户端很多，创建一个单独的图例图
            # 创建一个单独的图例文件
            legend_fig, legend_ax = plt.subplots(figsize=(10, 1))
            legend_ax.axis('off')
            legend_fig.legend(handles, labels, 
                            loc='center', 
                            ncol=5,
                            fontsize=12)
            legend_fig.savefig(os.path.join(results_dir, "color_legend.pdf"), 
                             dpi=300, bbox_inches='tight')
            plt.close(legend_fig)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加总标题
    plt.suptitle('T-SNE Visualization for All Clients (TinyImageNet-100)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有客户端T-SNE可视化图已保存: {save_path}")
    
    return fig

def create_class_mapping_file(data_dir="./data/tiny-imagenet-200"):
    """
    创建TinyImageNet类别映射文件
    Args:
        data_dir: TinyImageNet数据目录
    Returns:
        dict: 包含WordNet ID到类别名称的映射
    """
    wnids_file = os.path.join(data_dir, "wnids.txt")
    words_file = os.path.join(data_dir, "words.txt")
    
    if not os.path.exists(wnids_file):
        print(f"警告: {wnids_file} 不存在")
        return {}
    
    # 读取WordNet IDs
    with open(wnids_file, 'r') as f:
        wnids = [line.strip() for line in f]
    
    # 读取类别名称
    wordnet_to_name = {}
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid = parts[0]
                    name = parts[1].split(',')[0]  # 取第一个名称
                    wordnet_to_name[wnid] = name
    
    # 创建映射文件
    class_mapping = {}
    for i, wnid in enumerate(wnids):
        if wnid in wordnet_to_name:
            class_mapping[i] = wordnet_to_name[wnid]
        else:
            class_mapping[i] = wnid
    
    # 保存映射文件
    mapping_file = os.path.join(data_dir, "class_mapping.pkl")
    with open(mapping_file, 'wb') as f:
        pickle.dump(class_mapping, f)
    
    print(f"类别映射文件已保存: {mapping_file}")
    return class_mapping

def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 配置参数
    model_path_dir="./temp/TinyImagenet/PFedAFM/pat"
    excel_result_dir="./T-SNE/PFedAFM/TINY/Hetero/p_train"
    
    # 创建类别映射（如果需要）
    # create_class_mapping_file()
    
    # 步骤1: 对每个客户端进行t-SNE降维
    print("开始统一t-SNE降维...")
    for i in range(20):
        print(f"\n处理客户端 {i}...")
        unified_results = unified_tsne_across_clients(i, model_path_dir, excel_result_dir, "TinyImagenet")
        
        if unified_results is not None:
            print(f"客户端 {i} t-SNE处理完成! 处理了 {len(unified_results)} 个样本")
        else:
            print(f"客户端 {i} t-SNE处理失败")
    
    # 步骤2: 将所有客户端的T-SNE图排成一行
    print("\n生成所有客户端的T-SNE合并图...")
    save_path = os.path.join(excel_result_dir, "all_clients_one_row.png")
    fig = visualize_all_clients_in_one_row(
        results_dir=excel_result_dir,
        save_path=save_path,
        num_clients=20
    )
    
    print("\n所有处理完成!")