
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

def get_clip_class_embeddings(
    dataset_name: str,
    model_name: str = "RN50",
    prompt_template: str = "a photo of {}",
    device: str = None,
    download_root: str = "./utils/clip_weights"
) -> torch.Tensor:
    """
    获取指定数据集类别在 CLIP 下的文本嵌入（已归一化）。

    Args:
        dataset_name (str): 数据集名称，例如 "cifar10"。
        model_name (str): CLIP 模型名称，如 "RN50", "ViT-B/32" 等。
        prompt_template (str): 提示模板，必须包含一个 `{}` 用于插入类别名。
        device (str): 运行设备，默认自动选择 "cuda" 或 "cpu"。
        download_root (str): 模型权重下载路径。

    Returns:
        torch.Tensor: 形状为 [num_classes, embedding_dim] 的归一化文本嵌入。
    """
    # 1. 获取类别列表
    if dataset_name.lower() not in DATASET_CLASSES:
        raise ValueError(f"数据集 '{dataset_name}' 未在预定义列表中，请手动提供类别名称。")
    classes = DATASET_CLASSES[dataset_name.lower()]
    if classes is None:
        raise NotImplementedError(f"数据集 '{dataset_name}' 需要单独加载类别列表，当前未实现。")

    # 2. 设置设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. 加载 CLIP 模型
    model, _ = clip.load(model_name, device=device, download_root=download_root)

    # 4. 构造提示文本
    print("构造文本特征.........")
    texts = [prompt_template.format(cls) for cls in classes]

    # 5. 编码并归一化
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features,text_features_norm




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
