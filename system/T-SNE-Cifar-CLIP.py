import argparse
import os
import random
import glob
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data
from utils.get_clip_text_encoder import get_clip_class_embeddings


RANDOM_SEED = 0

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm"
]


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def class_names_for(dataset, num_classes):
    if dataset == "Cifar10":
        return CIFAR10_CLASSES
    if dataset == "Cifar100":
        return CIFAR100_CLASSES
    return [f"class_{idx}" for idx in range(num_classes)]


def build_data_args(args):
    return SimpleNamespace(
        niid=args.niid,
        partition=args.partition,
        dir_alpha=args.dir_alpha,
        class_per_client=args.class_per_client,
    )


def sanitize_path_component(text):
    text = str(text)
    safe_chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "default"


def format_float_for_path(value):
    return str(value).replace(".", "p")


def partition_tag(args):
    if args.partition == "dir":
        return f"dir_alpha{format_float_for_path(args.dir_alpha)}"
    if args.partition == "pat":
        return f"pat_cpc{args.class_per_client}"
    if args.partition == "exdir":
        return f"exdir_alpha{format_float_for_path(args.dir_alpha)}"
    return sanitize_path_component(args.partition)


def data_tag(args):
    return f"ncl{args.num_classes}_niid{args.niid}"


def join_tag(args):
    return f"clients{args.num_clients}_jr{format_float_for_path(args.join_ratio)}"


def find_final_model_dir(args):
    base_dir = os.path.join(
        args.final_model_root,
        sanitize_path_component(args.dataset),
        sanitize_path_component(args.algorithm),
    )
    tail_parts = [partition_tag(args), data_tag(args), join_tag(args)]

    if args.model_family:
        candidate = os.path.join(base_dir, sanitize_path_component(args.model_family), *tail_parts)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(
                f"没有找到最终模型目录: {candidate}\n"
                f"如果你想使用旧的 temp 目录，请显式传入 --model-dir。"
            )
        return candidate

    pattern = os.path.join(base_dir, "*", *tail_parts)
    candidates = sorted(path for path in glob.glob(pattern) if os.path.isdir(path))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"没有找到最终模型目录，匹配规则: {pattern}\n"
            f"请检查训练是否已经结束并导出 final_models，或显式传入 --model-dir。"
        )
    raise RuntimeError(
        "匹配到多个模型目录，请用 --model-family 或 --model-dir 指定一个:\n"
        + "\n".join(f"  - {path}" for path in candidates)
    )


def resolve_model_dir(args):
    if args.model_dir:
        if not os.path.isdir(args.model_dir):
            raise FileNotFoundError(f"--model-dir 指定的目录不存在: {args.model_dir}")
        return args.model_dir
    return find_final_model_dir(args)


def validate_model_files(args, model_dir):
    target_client_ids = parse_client_ids(args.client_ids, args.num_clients)
    missing_files = [
        os.path.join(model_dir, model_file_name(client_id, args.model_source))
        for client_id in target_client_ids
        if not os.path.exists(os.path.join(model_dir, model_file_name(client_id, args.model_source)))
    ]
    if missing_files:
        preview = "\n".join(f"  - {path}" for path in missing_files[:10])
        more = "" if len(missing_files) <= 10 else f"\n  ... 还有 {len(missing_files) - 10} 个"
        raise FileNotFoundError(
            f"模型目录存在，但缺少要画的模型文件:\n{preview}{more}\n"
            f"当前 model_source={args.model_source}。client 需要 Client_i_model.pt；"
            f"server 需要 Server_model_i.pt。"
        )


def client_tag(args):
    if not args.client_ids:
        return f"clients_all{args.num_clients}"
    return "clients_" + sanitize_path_component(args.client_ids.replace(",", "_"))


def build_output_dir(args, model_dir):
    if args.output_dir:
        return args.output_dir

    model_run_tag = sanitize_path_component(os.path.basename(os.path.normpath(model_dir)))
    text_tag = "with_text" if args.include_text else "no_text"
    split_tag = f"split_{args.split}"
    split_view_tag = "separate_splits" if args.separate_splits else "merged_splits"

    return os.path.join(
        "./T-SNE",
        sanitize_path_component(args.dataset),
        sanitize_path_component(args.algorithm),
        sanitize_path_component(args.model_source),
        partition_tag(args),
        client_tag(args),
        split_tag,
        split_view_tag,
        text_tag,
        model_run_tag,
    )


def model_file_name(client_id, model_source):
    if model_source == "server":
        return f"Server_model_{client_id}.pt"
    if model_source == "client":
        return f"Client_{client_id}_model.pt"
    raise ValueError(f"未知 model_source: {model_source}")


def torch_load_model(model_path, device):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def load_client_data(client_id, dataset, data_args, split, batch_size):
    is_train = split == "train"
    data = read_client_data(dataset, client_id, args=data_args, is_train=is_train, few_shot=0)
    return DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)


def parse_client_ids(client_ids_text, num_clients):
    if not client_ids_text:
        return list(range(num_clients))

    client_ids = []
    for part in client_ids_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start_id = int(start_text)
            end_id = int(end_text)
            if end_id < start_id:
                raise ValueError(f"客户端范围不合法: {part}")
            client_ids.extend(range(start_id, end_id + 1))
        else:
            client_ids.append(int(part))

    client_ids = list(dict.fromkeys(client_ids))
    invalid_ids = [cid for cid in client_ids if cid < 0 or cid >= num_clients]
    if invalid_ids:
        raise ValueError(f"客户端编号超出范围: {invalid_ids}, 合法范围是 0 到 {num_clients - 1}")
    return client_ids


def extract_model_features(model, images):
    if hasattr(model, "base"):
        features = model.base(images)
    else:
        features = model(images)
        if isinstance(features, (tuple, list)):
            features = features[0]

    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return F.normalize(features, dim=-1)


def collect_client_features(args, model_dir, data_args, device):
    all_features = []
    all_labels = []
    all_client_ids = []
    all_splits = []
    target_client_ids = parse_client_ids(args.client_ids, args.num_clients)
    target_splits = ["train", "test"] if args.split == "both" else [args.split]

    for client_id in target_client_ids:
        model_path = os.path.join(model_dir, model_file_name(client_id, args.model_source))
        if not os.path.exists(model_path):
            print(f"警告: {model_path} 不存在，跳过 Client_{client_id}")
            continue

        print(f"处理 Client_{client_id}: {model_path}")
        model = torch_load_model(model_path, device)

        for split in target_splits:
            dataloader = load_client_data(
                client_id=client_id,
                dataset=args.dataset,
                data_args=data_args,
                split=split,
                batch_size=args.batch_size,
            )

            split_features = []
            split_labels = []
            seen = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(dataloader):
                    if args.max_batches > 0 and batch_idx >= args.max_batches:
                        break
                    if args.max_samples_per_client > 0 and seen >= args.max_samples_per_client:
                        break

                    images = images.to(device)
                    features = extract_model_features(model, images)
                    labels_np = labels.numpy()

                    if args.max_samples_per_client > 0:
                        remaining = args.max_samples_per_client - seen
                        features = features[:remaining]
                        labels_np = labels_np[:remaining]

                    split_features.append(features.detach().cpu().numpy())
                    split_labels.append(labels_np)
                    seen += len(labels_np)

            if not split_features:
                print(f"警告: Client_{client_id} 的 {split} 集没有成功提取任何特征。")
                continue

            features_np = np.concatenate(split_features, axis=0)
            labels_np = np.concatenate(split_labels, axis=0)
            all_features.append(features_np)
            all_labels.append(labels_np)
            all_client_ids.extend([client_id] * len(labels_np))
            all_splits.extend([split] * len(labels_np))
            print(
                f"Client_{client_id} [{split}]: 收集 {len(labels_np)} 个样本，"
                f"特征维度 {features_np.shape[1]}"
            )

    return all_features, all_labels, all_client_ids, all_splits


def append_clip_text_features(args, all_features, all_labels, all_client_ids, all_splits, device):
    if not args.include_text:
        return

    clip_text_features, clip_text_features_norm = get_clip_class_embeddings(
        args.dataset,
        model_name=args.clip_model,
        prompt_template=args.prompt_template,
        device=device,
    )
    text_features = clip_text_features_norm.float().detach().cpu().numpy()
    all_features.append(text_features)
    all_labels.append(np.arange(text_features.shape[0]))
    all_client_ids.extend([-1] * text_features.shape[0])
    all_splits.extend(["text"] * text_features.shape[0])
    print(f"已加入 CLIP 文本锚点: {text_features.shape}")


def run_tsne(args, model_dir, device):
    data_args = build_data_args(args)
    all_features, all_labels, all_client_ids, all_splits = collect_client_features(
        args=args,
        model_dir=model_dir,
        data_args=data_args,
        device=device,
    )

    if not all_features:
        raise RuntimeError("没有成功收集任何客户端特征，无法画 t-SNE。")

    append_clip_text_features(args, all_features, all_labels, all_client_ids, all_splits, device)

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_client_ids = np.array(all_client_ids)
    combined_splits = np.array(all_splits)

    print(f"总特征形状: {combined_features.shape}")
    print(f"总标签形状: {combined_labels.shape}")
    print("开始统一 t-SNE 降维...")

    tsne = TSNE(
        metric="cosine",
        n_components=2,
        perplexity=min(args.perplexity, len(combined_features) - 1),
        learning_rate=args.tsne_lr,
        random_state=args.seed,
        max_iter=args.max_iter,
        verbose=1,
    )
    features_2d = tsne.fit_transform(combined_features)

    class_names = class_names_for(args.dataset, args.num_classes)
    class_name_column = [
        class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
        for label in combined_labels
    ]

    return pd.DataFrame({
        "client_id": combined_client_ids,
        "split": combined_splits,
        "label": combined_labels,
        "class_name": class_name_column,
        "t-SNE_dim1": features_2d[:, 0],
        "t-SNE_dim2": features_2d[:, 1],
        "type": ["text" if cid == -1 else "image" for cid in combined_client_ids],
    })


def save_dataframe(tsne_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "unified_tsne_all_clients.csv")
    xlsx_path = os.path.join(output_dir, "unified_tsne_all_clients.xlsx")
    tsne_df.to_csv(csv_path, index=False)
    try:
        tsne_df.to_excel(xlsx_path, index=False)
        print(f"Excel: {xlsx_path}")
    except Exception as exc:
        print(f"写入 Excel 失败，仅保存 CSV。错误: {exc}")
    print(f"CSV: {csv_path}")


def save_plot(fig, output_dir, name):
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图像: {pdf_path}")
    print(f"保存图像: {png_path}")


def plot_by_class(tsne_df, output_dir, separate_splits=False):
    fig, ax = plt.subplots(figsize=(12, 9))
    image_df = tsne_df[tsne_df["type"] == "image"]
    text_df = tsne_df[tsne_df["type"] == "text"]
    has_text_anchors = len(text_df) > 0

    labels = sorted(image_df["label"].unique())
    cmap = plt.cm.nipy_spectral(np.linspace(0, 1, max(len(labels), 1)))
    label_to_color = {label: cmap[idx] for idx, label in enumerate(labels)}

    if not separate_splits:
        for label in labels:
            class_data = image_df[image_df["label"] == label]
            ax.scatter(
                class_data["t-SNE_dim1"],
                class_data["t-SNE_dim2"],
                color=label_to_color[label],
                marker="o",
                alpha=0.68,
                s=18,
                linewidths=0,
            )
    else:
        split_markers = {"train": "o", "test": "^"}
        for label in labels:
            for split, marker in split_markers.items():
                class_data = image_df[(image_df["label"] == label) & (image_df["split"] == split)]
                if len(class_data) == 0:
                    continue
                ax.scatter(
                    class_data["t-SNE_dim1"],
                    class_data["t-SNE_dim2"],
                    color=label_to_color[label],
                    marker=marker,
                    alpha=0.68 if split == "train" else 0.9,
                    s=18 if split == "train" else 28,
                    linewidths=0.25 if split == "test" else 0,
                    edgecolors="black" if split == "test" else "none",
                )

    if has_text_anchors:
        for label in labels:
            class_text = text_df[text_df["label"] == label]
            if len(class_text) == 0:
                continue
            ax.scatter(
                class_text["t-SNE_dim1"],
                class_text["t-SNE_dim2"],
                color=label_to_color[label],
                marker="*",
                s=180,
                edgecolors="black",
                linewidth=0.8,
                alpha=0.95,
            )

    ax.set_title("t-SNE by Class" if not has_text_anchors else "t-SNE by Class with CLIP Text Anchors")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(alpha=0.15)
    if separate_splits and "test" in set(image_df["split"]):
        train_proxy = plt.Line2D([0], [0], marker="o", color="gray", linestyle="", label="train")
        test_proxy = plt.Line2D([0], [0], marker="^", color="gray", linestyle="", label="test")
        ax.legend(handles=[train_proxy, test_proxy], loc="best", fontsize=9)
    save_plot(fig, output_dir, "tsne_by_class_with_clip_text" if has_text_anchors else "tsne_by_class")


def plot_by_client(tsne_df, output_dir):
    fig, ax = plt.subplots(figsize=(12, 9))
    image_df = tsne_df[tsne_df["type"] == "image"]
    text_df = tsne_df[tsne_df["type"] == "text"]

    client_ids = sorted(image_df["client_id"].unique())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(len(client_ids), 1)))
    for idx, client_id in enumerate(client_ids):
        client_data = image_df[image_df["client_id"] == client_id]
        ax.scatter(
            client_data["t-SNE_dim1"],
            client_data["t-SNE_dim2"],
            color=cmap[idx],
            alpha=0.65,
            s=14,
            linewidths=0,
            label=f"Client {client_id}",
        )

    if len(text_df) > 0:
        ax.scatter(
            text_df["t-SNE_dim1"],
            text_df["t-SNE_dim2"],
            color="black",
            marker="*",
            s=130,
            alpha=0.8,
            label="CLIP text",
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:21], labels[:21], loc="best", fontsize=8)
    ax.set_title("t-SNE by Client")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(alpha=0.15)
    save_plot(fig, output_dir, "tsne_by_client")


def parse_args():
    parser = argparse.ArgumentParser(description="FedCLIP t-SNE feature visualization.")
    parser.add_argument("--model-dir", type=str, default="",
                        help="实验权重目录，例如 ./temp/Cifar10/FedCLIP/1780475838.368152。指定后优先使用该目录。")
    parser.add_argument("--final-model-root", type=str, default="./final_models",
                        help="未指定 --model-dir 时，从这个最终模型根目录按实验配置查找。")
    parser.add_argument("--model-family", "-m", type=str, default="",
                        help="未指定 --model-dir 时用于定位最终模型目录，例如 Decom_CNN-5-512。为空则尝试自动唯一匹配。")
    parser.add_argument("--dataset", type=str, default="Cifar100")
    parser.add_argument("--algorithm", type=str, default="FedCLIP")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--join-ratio", "-jr", type=float, default=1.0)
    parser.add_argument("--client-ids", type=str, default="",
                        help="只画指定客户端，例如 3、3,7,18 或 0-4。为空则画全部客户端。")
    parser.add_argument("--model-source", choices=["server", "client"], default="server",
                        help="server 使用 Server_model_i.pt，client 使用 Client_i_model.pt。")
    parser.add_argument("--output-dir", type=str, default="",
                        help="输出目录。为空则写到 ./T-SNE/{dataset}/{algorithm}/{model_source}/。")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="train/test/both。both 会把训练集和测试集特征放到同一张 t-SNE 图中。")
    parser.add_argument("--separate-splits", action="store_true",
                        help="画图时区分 train/test 的 marker。默认不区分，只按类别着色。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=40)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--niid", type=int, default=1)
    parser.add_argument("--partition", "-pt", type=str, default="dir")
    parser.add_argument("--dir-alpha", "-dir_alpha", type=float, default=0.3)
    parser.add_argument("--class-per-client", "-cpc", type=int, default=2)

    parser.add_argument("--include-text", action=argparse.BooleanOptionalAction, default=False,
                        help="是否把 CLIP 文本锚点一起加入 t-SNE。默认不加入。")
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--prompt-template", type=str, default="a photo of {}")

    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--tsne-lr", type=float, default=200)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    model_dir = resolve_model_dir(args)
    validate_model_files(args, model_dir)
    output_dir = build_output_dir(args, model_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print("开始统一 t-SNE 降维...")
    print(f"模型目录: {model_dir}")
    print(f"模型来源: {args.model_source}")
    print(f"数据集: {args.dataset} | partition={args.partition} | alpha={args.dir_alpha} | cpc={args.class_per_client}")
    print(f"输出目录: {output_dir}")

    results = run_tsne(args, model_dir, device)
    save_dataframe(results, output_dir)
    plot_by_class(results, output_dir, separate_splits=args.separate_splits)
    plot_by_client(results, output_dir)

    print("\n统一 t-SNE 处理完成")
    print(f"总样本数: {len(results)}")
    print(f"涉及客户端: {sorted(results['client_id'].unique())}")
