
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import torch
import torch
import clip
import os

# 常见数据集的类别名称（英文）
DATASET_CLASSES = {
    "cifar10": [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    "cifar100": [
        "beaver", "dolphin", "otter", "seal", "whale",
        "aquarium fish", "flatfish", "ray", "shark", "trout",
        "orchids", "poppies", "roses", "sunflowers", "tulips",
        "bottles", "bowls", "cans", "cups", "plates",
        "apples", "mushrooms", "oranges", "pears", "sweet peppers",
        "clock", "computer keyboard", "lamp", "telephone", "television",
        "bed", "chair", "couch", "table", "wardrobe",
        "bee", "beetle", "butterfly", "caterpillar", "cockroach",
        "bear", "leopard", "lion", "tiger", "wolf",
        "bridge", "castle", "house", "road", "skyscraper",
        "cloud", "forest", "mountain", "plain", "sea",
        "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
        "fox", "porcupine", "possum", "raccoon", "skunk",
        "crab", "lobster", "snail", "spider", "worm",
        "baby", "boy", "girl", "man", "woman",
        "crocodile", "dinosaur", "lizard", "snake", "turtle",
        "hamster", "mouse", "rabbit", "shrew", "squirrel",
        "maple", "oak", "palm", "pine", "willow",
        "bicycle", "bus", "motorcycle", "pickup truck", "train",
        "lawn-mower", "rocket", "streetcar", "tank", "tractor"
    ]
}

def get_clip_v_encoder(
    model_name: str, 
    device: str = None, 
    download_root: str = "./utils/clip_weights"
) -> torch.nn.Module:
    """
    获取指定 CLIP 模型的图像编码器 (Visual Encoder)。
    如果本地 download_root 目录没有权重文件，clip 库会自动联网下载。

    Args:
        model_name (str): 模型名称，例如 'RN50', 'ViT-B-32' 或 'ViT-B/32'。
        device (str): 运行设备，默认自动选择 "cuda" 或 "cpu"。
        download_root (str): 模型权重下载/读取的缓存路径。

    Returns:
        torch.nn.Module: 对应的图像编码器网络。
    """
    # 1. 映射模型名称 (处理斜杠和横杠的区别)
    name_map = {
        "RN50": "RN50",
        "ViT-B-32": "ViT-B/32",  # 用户习惯用横杠
        "ViT-B/32": "ViT-B/32"   # CLIP 官方的名称
    }
    
    if model_name not in name_map:
        raise ValueError(f"model_name 参数错误，请使用 'RN50' 或 'ViT-B/32'，当前输入: {model_name}")
        
    official_name = name_map[model_name]

    # 2. 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. 直接通过 CLIP 官方 API 加载
    print(f"正在从 CLIP 库加载 {official_name} 的视觉编码器...")
    model, _ = clip.load(official_name, device=device, download_root=download_root)
    
    # 4. 剥离并返回视觉编码器
    visual_encoder = model.visual
    
    return visual_encoder

import os
import torch
import clip

# ... (上面的 get_clip_v_encoder 保持不变) ...

def get_clip_class_embeddings(
    dataset_name: str,
    model_name: str = "RN50",
    prompt_template: str = "a photo of {}",
    device: str = None,
    download_root: str = "./utils/clip_weights"
) -> tuple:
    """
    获取指定数据集类别在 CLIP 下的文本嵌入（支持本地缓存加速）。

    Args:
        dataset_name (str): 数据集名称，例如 "cifar10"。
        model_name (str): CLIP 模型名称，如 "RN50", "ViT-B/32" 等。
        prompt_template (str): 提示模板，必须包含一个 `{}` 用于插入类别名。
        device (str): 运行设备，默认自动选择 "cuda" 或 "cpu"。
        download_root (str): 模型权重和缓存文件的根目录。

    Returns:
        tuple: (text_features, text_features_norm) 均在指定的 device 上
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================= 1. 构建缓存路径与文件名 =================
    # 在下载目录下单独建一个存特征的文件夹
    cache_dir = os.path.join(download_root, "text_features_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 替换特殊字符以生成合法的文件名
    safe_model = model_name.replace("/", "-")
    safe_prompt = prompt_template.replace(" ", "_").replace("{}", "OBJ")
    cache_filename = f"{dataset_name}_{safe_model}_{safe_prompt}.pt"
    cache_filepath = os.path.join(cache_dir, cache_filename)
    # ==========================================================

    # ================= 2. 检查并加载缓存 =====================
    if os.path.exists(cache_filepath):
        print(f"✅ 命中缓存！直接从本地加载文本特征: {cache_filepath}")
        # map_location 确保加载时直接放到正确的设备上
        cached_data = torch.load(cache_filepath, map_location=device)
        return cached_data['text_features'], cached_data['text_features_norm']
    # ==========================================================

    print(f"⚠️ 本地无缓存，正在初始化 CLIP 模型提取 {dataset_name} 文本特征...")

    # 3. 获取类别列表
    if dataset_name.lower() not in DATASET_CLASSES:
        raise ValueError(f"数据集 '{dataset_name}' 未在预定义列表中，请手动提供类别名称。")
    classes = DATASET_CLASSES[dataset_name.lower()]
    if classes is None:
        raise NotImplementedError(f"数据集 '{dataset_name}' 需要单独加载类别列表，当前未实现。")

    # 4. 加载 CLIP 模型 (极其耗时的步骤)
    model, _ = clip.load(model_name, device=device, download_root=download_root)

    # 5. 构造提示文本
    print("构造文本特征.........")
    texts = [prompt_template.format(cls) for cls in classes]

    # 6. 编码并归一化
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # ================= 7. 保存到本地缓存 =====================
    print(f"💾 提取完成，保存文本特征缓存至: {cache_filepath}")
    # 推荐保存到 CPU，这样以后无论是切到其他 GPU 还是本机调试都不会报设备不匹配错
    torch.save({
        'text_features': text_features.cpu(),
        'text_features_norm': text_features_norm.cpu()
    }, cache_filepath)
    # ==========================================================

    return text_features, text_features_norm



def plot_tsne(
    text_embeddings,
    class_names,
    perplexity=3,
    n_iter=1000,
    random_state=42,
    figsize=(10, 8),
    title="t-SNE visualization of class embeddings",
    save_path="./Vit_clip_weights"
):
    """
    绘制文本嵌入的 t-SNE 散点图。

    Args:
        text_embeddings (torch.Tensor or np.ndarray): 形状为 (n_samples, n_features) 的嵌入矩阵。
        class_names (list of str): 每个样本对应的类别名称，长度与 n_samples 相同。
        perplexity (int): t-SNE 的困惑度参数，通常取值 5-50。
        n_iter (int): 优化迭代次数。
        random_state (int): 随机种子，用于结果可复现。
        figsize (tuple): 图像大小 (宽, 高)。
        title (str): 图像标题。
        save_path (str, optional): 若提供，将图像保存到该路径。
    """
    # 转换为 numpy 数组（如果是 torch 张量）
    if isinstance(text_embeddings, torch.Tensor):
        embeddings = text_embeddings.cpu().numpy()
    else:
        embeddings = np.array(text_embeddings)

    # 执行 t-SNE 降维到 2 维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init='pca'  # 使用 PCA 初始化，更稳定
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # 绘制散点图
    plt.figure(figsize=figsize)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

    # 为每个点添加类别标签
    for i, name in enumerate(class_names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    text_features,text_features_norm = get_clip_class_embeddings('cifar100',"RN50")
    print("没有归一化的文本特征为:",text_features.shape,text_features)
    print("没有归一化的文本特征为:",text_features_norm)
    plot_tsne(
        text_embeddings=text_features,
        class_names=DATASET_CLASSES["cifar100"],
        title="CIFAR-100 class text_features (t-SNE)",
        save_path = "./Vit_clip_weights/text_features.png"
    )
    plot_tsne(
        text_embeddings=text_features_norm,
        class_names=DATASET_CLASSES["cifar100"],
        title="CIFAR-100 class text_features_norm (t-SNE)",
        save_path = "./Vit_clip_weights/text_features_norm.png"
    )
