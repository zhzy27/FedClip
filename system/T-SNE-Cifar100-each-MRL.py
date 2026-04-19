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


def unified_tsne_across_clients(client_id,
                                model_path_dir="./temp/Cifar100/FedDAR/1764775858.727305",
                                excel_result_dir="./T-SNE/DAT/CIFAR100/Hetero/Each",
                                dataset_type="Cifar100"):
    """
    所有客户端在统一表征空间进行t-SNE降维
    """
    model_path = os.path.join(model_path_dir,f"Client_{client_id}_model.pt")
    global_model_path = os.path.join(model_path_dir,f"Server_global_model.pt")
    pro_model_path =  os.path.join(model_path_dir,f"Client_{client_id}_proj.pt")
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在，跳过")
    try:
        model = torch.load(model_path, map_location='cpu')
        global_model = torch.load(global_model_path, map_location='cpu')
        pro_model = torch.load(pro_model_path, map_location='cpu')
        model.eval()
        global_model.eval()
        pro_model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")

    try:
        # client_dataloader = load_train_data(str(client_id), dataset_type)
        client_dataloader = load_test_data(str(client_id), dataset_type)
        client_features = []
        client_labels = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(client_dataloader):
                # if batch_idx >= 40:  # 限制每个客户端的数据量，避免内存不足
                #     break
                rep_g = global_model.base(images)
                rep = model.base(images)
                rep_concat = torch.concat((rep_g, rep), dim=1)
                features= pro_model(rep_concat)
                client_features.append(features.numpy())
                client_labels.append(labels.numpy())
    except Exception as e:
        print(f"处理客户端 {client_id} 数据失败: {e}")

    combined_features = np.concatenate(client_features, axis=0)
    combined_labels = np.concatenate(client_labels, axis=0)
    print(f"总特征形状: {combined_features.shape}")
    print(f"总标签形状: {combined_labels.shape}")

    # 3. 统一进行t-SNE降维
    print("进行t-SNE降维...")
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
        'label': combined_labels,
        'class_name': [cifar100_classes[label] for label in combined_labels],
        't-SNE_dim1': unified_features_2d[:, 0],
        't-SNE_dim2': unified_features_2d[:, 1],
    })

    # 5. 保存结果
    os.makedirs(excel_result_dir, exist_ok=True)

    # 保存整体结果
    unified_excel_path = os.path.join(excel_result_dir, f"{client_id}_T-SNE.xlsx")
    unified_csv_path = os.path.join(excel_result_dir,  f"{client_id}_T-SNE.csv")

    unified_df.to_excel(unified_excel_path, index=False)
    unified_df.to_csv(unified_csv_path, index=False)

    print(f"统一t-SNE结果已保存:")
    print(f"Excel: {unified_excel_path}")
    print(f"CSV: {unified_csv_path}")

    # 7. 可视化
    visualize_unified_tsne(unified_df, excel_result_dir,client_id)

    return unified_df


def visualize_unified_tsne(tsne_df, save_dir,client_id):
    """可视化统一的t-SNE结果"""

    # 按客户端可视化
    plt.figure(figsize=(20, 16))

    # 子图1: 按类别着色
    plt.subplot(2, 2, 1)
    for class_name in cifar100_classes:
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
                                     save_path="./T-SNE/DAT/CIFAR10/Hetero/Each/CE/all_clients_row.png",
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
    # 对于CIFAR-100，使用gist_ncar颜色映射，它可以产生100种不同的颜色
    color_palette = plt.cm.gist_ncar  # 或者使用 plt.cm.nipy_spectral
    
    # 创建类别到颜色的映射
    class_colors = {}
    # 假设CIFAR-100有100个类别（0-99）
    for i in range(100):  # CIFAR-100有100个类别
        class_colors[str(i)] = color_palette(i/100)  # 均匀分布颜色
    
    for i, class_name in enumerate(cifar100_classes):
        class_colors[class_name] = color_palette(i/len(cifar100_classes))
    
    # 遍历所有客户端并绘制
    for idx, client_id in enumerate(client_ids):
        try:
            # 读取客户端数据
            csv_path = os.path.join(results_dir, f"{client_id}_T-SNE.csv")
            df = pd.read_csv(csv_path)
            
            # 获取当前子图
            ax = axes[idx]
            
            # 按类别绘制散点图
            for class_name in cifar100_classes:
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
            legend_fig.savefig(os.path.join(results_dir, "color_legend.png"), 
                             dpi=300, bbox_inches='tight')
            plt.close(legend_fig)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加总标题
    plt.suptitle('T-SNE Visualization for All Clients (CIFAR-100)', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有客户端T-SNE可视化图已保存: {save_path}")
    
    return fig

def create_detailed_legend(results_dir="./T-SNE/DAT/CIFAR100/Hetero/Each/CE_MSE"):
    """
    创建详细的图例说明图
    """
    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    
    # 为每个类别创建颜色块
    color_palette = plt.cm.tab10
    legend_elements = []
    
    for i, class_name in enumerate(cifar100_classes):
        color = color_palette(i/len(cifar100_classes))
        legend_elements.append(plt.Line2D([0], [0], 
                                         marker='o', 
                                         color='w', 
                                         label=class_name,
                                         markerfacecolor=color, 
                                         markersize=10))
    
    # 创建图例
    ax.legend(handles=legend_elements, 
              loc='center', 
              ncol=5,  # 分成两行，每行5个
              fontsize=12,
              frameon=True,
              fancybox=True,
              shadow=True)
    
    # 保存图例
    legend_path = os.path.join(results_dir, "detailed_legend.png")
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"详细图例已保存: {legend_path}")

def load_train_data(id, dataset, batch_size=16):
    """加载训练数据"""
    train_data = read_client_data(dataset, id, is_train=True, few_shot=0)
    return DataLoader(train_data, batch_size, drop_last=False, shuffle=True)


def load_test_data(id, dataset, batch_size=16):
    """加载测试数据"""
    test_data = read_client_data(dataset, id, is_train=False, few_shot=0)
    return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)


if __name__ == "__main__":
    # 1764939719.6478398  ce
    # 1764939721.7231762  ce+mse
    # 1764939726.5198965 ce+con
    #1764939734.2981603  ce+con+mse  /CIFAR100/Hetero/Each/CE_CON_MSE
    model_path_dir="./temp/TinyImagenet/FedMRL/pat"
    excel_result_dir="./T-SNE/FedMRL/CIFAR100/Hetero/test"
    # 使用统一表征空间的t-SNE
    for i in range(20):
        print("开始统一t-SNE降维...")
        unified_results = unified_tsne_across_clients(i,model_path_dir,excel_result_dir)

        if unified_results is not None:
            print("\n统一t-SNE处理完成!")
            print(f"总共处理了 {len(unified_results)} 个样本")
        else:
            print("\n统一t-SNE处理失败")
        # 步骤2: 将所有客户端的T-SNE图排成一行
    print("\n生成所有客户端的T-SNE合并图...")
    save_path = os.path.join(excel_result_dir, "all_clients_one_row.png")
    fig = visualize_all_clients_in_one_row(
        results_dir=excel_result_dir,
        save_path=save_path,
        num_clients=20
    )
    
    # # 步骤3: 创建详细的图例
    # print("\n创建详细图例...")
    # create_detailed_legend(results_dir=excel_result_dir)
    
    print("\n所有处理完成!")