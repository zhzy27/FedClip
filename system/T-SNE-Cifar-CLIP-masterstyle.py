import argparse
import glob
import importlib.util
import os
import random
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data
from utils.get_clip_text_encoder import get_clip_class_embeddings


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_TSNE_PATH = SCRIPT_DIR / "T-SNE-Cifar-CLIP.py"


def load_base_tsne_module():
    spec = importlib.util.spec_from_file_location("fedclip_base_tsne", BASE_TSNE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_tsne = load_base_tsne_module()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_name(value):
    return base_tsne.sanitize_path_component(value)


def float_tag(value):
    return str(value).replace(".", "p")


def build_data_args(args):
    return SimpleNamespace(
        niid=args.niid,
        partition=args.partition,
        dir_alpha=args.dir_alpha,
        class_per_client=args.class_per_client,
    )


def partition_tag(args):
    if args.partition == "dir":
        return f"dir_alpha{float_tag(args.dir_alpha)}"
    if args.partition == "pat":
        return f"pat_cpc{args.class_per_client}"
    if args.partition == "exdir":
        return f"exdir_alpha{float_tag(args.dir_alpha)}"
    return safe_name(args.partition)


def data_tag(args):
    return f"ncl{args.num_classes}_niid{args.niid}"


def join_tag(args):
    return f"clients{args.num_clients}_jr{float_tag(args.join_ratio)}"


def find_final_model_dir(args):
    base_dir = os.path.join(
        args.final_model_root,
        safe_name(args.dataset),
        safe_name(args.algorithm),
    )
    tail_parts = [partition_tag(args), data_tag(args), join_tag(args)]

    if args.model_family:
        candidate = os.path.join(base_dir, safe_name(args.model_family), *tail_parts)
        if not os.path.isdir(candidate):
            raise FileNotFoundError(
                f"没有找到最终模型目录: {candidate}\n"
                "如果你想使用旧 temp 目录，请显式传入 --model-dir。"
            )
        return candidate

    pattern = os.path.join(base_dir, "*", *tail_parts)
    candidates = sorted(path for path in glob.glob(pattern) if os.path.isdir(path))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"没有找到最终模型目录，匹配规则: {pattern}")
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


def parse_client_ids(client_ids_text, num_clients):
    return base_tsne.parse_client_ids(client_ids_text, num_clients)


def candidate_model_paths(model_dir, client_id, model_source):
    if model_source == "client":
        return [os.path.join(model_dir, f"Client_{client_id}_model.pt")]
    if model_source == "server":
        return [
            os.path.join(model_dir, f"Server_model_{client_id}.pt"),
            os.path.join(model_dir, "Server_model.pt"),
        ]
    raise ValueError(f"未知 model_source: {model_source}")


def resolve_model_path(model_dir, client_id, model_source):
    for path in candidate_model_paths(model_dir, client_id, model_source):
        if os.path.exists(path):
            return path
    candidates = "\n".join(f"  - {path}" for path in candidate_model_paths(model_dir, client_id, model_source))
    raise FileNotFoundError(f"找不到 Client_{client_id} 对应模型文件:\n{candidates}")


def torch_load_model(model_path, device):
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model


def load_client_data(client_id, dataset, data_args, split, batch_size):
    data = read_client_data(dataset, client_id, args=data_args, is_train=(split == "train"), few_shot=0)
    return DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)


def row_normalize(array, eps=1e-8):
    return array / (np.linalg.norm(array, axis=1, keepdims=True) + eps)


def extract_model_features(model, images, feature_source):
    if feature_source in ("auto", "base") and hasattr(model, "base"):
        features = model.base(images)
    else:
        output = model(images)
        if isinstance(output, (tuple, list)):
            features = output[0]
        else:
            features = output

    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return features


def collect_image_features(args, model_dir, device):
    data_args = build_data_args(args)
    target_client_ids = parse_client_ids(args.client_ids, args.num_clients)
    target_splits = ["train", "test"] if args.split == "both" else [args.split]

    features_all = []
    labels_all = []
    client_ids_all = []
    splits_all = []

    for client_id in target_client_ids:
        model_path = resolve_model_path(model_dir, client_id, args.model_source)
        print(f"处理 Client_{client_id}: {model_path}")
        model = torch_load_model(model_path, device)

        for split in target_splits:
            loader = load_client_data(client_id, args.dataset, data_args, split, args.batch_size)
            split_features = []
            split_labels = []
            seen = 0

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(loader):
                    if args.max_batches > 0 and batch_idx >= args.max_batches:
                        break
                    if args.max_samples_per_client > 0 and seen >= args.max_samples_per_client:
                        break

                    images = images.to(device)
                    features = extract_model_features(model, images, args.feature_source)
                    labels_np = labels.numpy()

                    if args.max_samples_per_client > 0:
                        remaining = args.max_samples_per_client - seen
                        features = features[:remaining]
                        labels_np = labels_np[:remaining]

                    split_features.append(features.detach().cpu().numpy())
                    split_labels.append(labels_np)
                    seen += len(labels_np)

            if not split_features:
                print(f"警告: Client_{client_id} [{split}] 没有收集到样本。")
                continue

            features_np = np.concatenate(split_features, axis=0)
            labels_np = np.concatenate(split_labels, axis=0)
            features_all.append(features_np)
            labels_all.append(labels_np)
            client_ids_all.extend([client_id] * len(labels_np))
            splits_all.extend([split] * len(labels_np))
            print(f"Client_{client_id} [{split}]: {len(labels_np)} samples, feature dim={features_np.shape[1]}")

    if not features_all:
        raise RuntimeError("没有成功收集任何特征。")

    return (
        np.concatenate(features_all, axis=0),
        np.concatenate(labels_all, axis=0),
        np.asarray(client_ids_all),
        np.asarray(splits_all),
    )


def select_labels_for_auxiliary_points(args, image_labels):
    present = np.array(sorted(np.unique(image_labels).astype(int)))
    if args.anchor_scope == "all":
        return np.arange(args.num_classes, dtype=int)
    return present


def get_text_anchors(args, device, selected_labels, feature_dim):
    if not args.include_text:
        return None

    _, text_features_norm = get_clip_class_embeddings(
        args.dataset,
        model_name=args.clip_model,
        prompt_template=args.prompt_template,
        device=device,
    )
    text_features = text_features_norm.float().detach().cpu().numpy()
    text_features = text_features[selected_labels]
    if text_features.shape[1] != feature_dim:
        raise ValueError(
            f"图像特征维度 {feature_dim} 与文本锚点维度 {text_features.shape[1]} 不一致。"
            "可以使用 --no-include-text 跳过文本锚点。"
        )
    return text_features


def preprocess_like_master(image_features, text_anchors, args):
    image_features = image_features.astype(np.float32, copy=True)
    text_anchors = None if text_anchors is None else text_anchors.astype(np.float32, copy=True)

    if args.center_modalities:
        image_features = image_features - np.mean(image_features, axis=0, keepdims=True)
        if text_anchors is not None:
            text_anchors = text_anchors - np.mean(text_anchors, axis=0, keepdims=True)

    if args.normalize_features:
        image_features = row_normalize(image_features)
        if text_anchors is not None:
            text_anchors = row_normalize(text_anchors)

    return image_features, text_anchors


def compute_centroids(image_features, image_labels, selected_labels):
    centroids = []
    centroid_labels = []
    for label in selected_labels:
        class_features = image_features[image_labels == label]
        if len(class_features) == 0:
            continue
        centroid = np.mean(class_features, axis=0, keepdims=True)
        centroids.append(row_normalize(centroid)[0])
        centroid_labels.append(label)
    if not centroids:
        return np.empty((0, image_features.shape[1]), dtype=image_features.dtype), np.array([], dtype=int)
    return np.stack(centroids, axis=0), np.asarray(centroid_labels, dtype=int)


def tsne_learning_rate(value):
    if str(value).lower() == "auto":
        return "auto"
    return float(value)


def fit_tsne(features, args):
    perplexity = min(args.perplexity, max(1, len(features) - 1))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        random_state=args.seed,
        metric=args.metric,
        init=args.init,
        learning_rate=tsne_learning_rate(args.tsne_lr),
        verbose=1,
    )
    try:
        tsne = TSNE(max_iter=args.max_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=args.max_iter, **kwargs)
    return tsne.fit_transform(features)


def run_masterstyle_tsne(args, model_dir, device):
    image_features, image_labels, client_ids, splits = collect_image_features(args, model_dir, device)
    selected_labels = select_labels_for_auxiliary_points(args, image_labels)
    text_anchors = get_text_anchors(args, device, selected_labels, image_features.shape[1])

    image_features, text_anchors = preprocess_like_master(image_features, text_anchors, args)
    centroids, centroid_labels = (
        compute_centroids(image_features, image_labels, selected_labels)
        if args.include_centroids else
        (np.empty((0, image_features.shape[1]), dtype=image_features.dtype), np.array([], dtype=int))
    )

    combined = [image_features]
    type_blocks = [np.array(["image"] * len(image_features))]
    label_blocks = [image_labels.astype(int)]
    client_blocks = [client_ids]
    split_blocks = [splits]

    if len(centroids) > 0:
        combined.append(centroids)
        type_blocks.append(np.array(["centroid"] * len(centroids)))
        label_blocks.append(centroid_labels)
        client_blocks.append(np.array([-2] * len(centroids)))
        split_blocks.append(np.array(["centroid"] * len(centroids)))

    if text_anchors is not None:
        combined.append(text_anchors)
        type_blocks.append(np.array(["text"] * len(text_anchors)))
        label_blocks.append(selected_labels.astype(int))
        client_blocks.append(np.array([-1] * len(text_anchors)))
        split_blocks.append(np.array(["text"] * len(text_anchors)))

    combined_features = np.concatenate(combined, axis=0)
    print(
        f"Running master-style t-SNE on {len(image_features)} images, "
        f"{len(centroids)} centroids, "
        f"{0 if text_anchors is None else len(text_anchors)} text anchors."
    )
    coords = fit_tsne(combined_features, args)

    return pd.DataFrame({
        "client_id": np.concatenate(client_blocks),
        "split": np.concatenate(split_blocks),
        "label": np.concatenate(label_blocks),
        "type": np.concatenate(type_blocks),
        "t-SNE_dim1": coords[:, 0],
        "t-SNE_dim2": coords[:, 1],
    })


def color_map_for_labels(labels):
    labels = sorted(int(label) for label in labels)
    if len(labels) <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(labels), 1)))
    elif len(labels) <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(labels), 1)))
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, max(len(labels), 1), endpoint=False))
    return {label: colors[idx] for idx, label in enumerate(labels)}


def save_plot(fig, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{name}.png")
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"保存图像: {png_path}")
    print(f"保存图像: {pdf_path}")


def plot_features_only(df, output_dir, args):
    image_df = df[df["type"] == "image"]
    labels = sorted(image_df["label"].unique().astype(int))
    colors = color_map_for_labels(labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    for label in labels:
        class_df = image_df[image_df["label"] == label]
        ax.scatter(
            class_df["t-SNE_dim1"],
            class_df["t-SNE_dim2"],
            color=colors[int(label)],
            alpha=0.5,
            s=args.feature_point_size,
            linewidths=0,
            label=str(label),
            zorder=1,
        )

    ax.set_title("t-SNE: Visual Features Only", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(alpha=0.15)
    if args.show_legend and len(labels) <= args.max_legend_classes:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=9)
    fig.tight_layout()
    save_plot(fig, output_dir, "masterstyle_tsne_features_only")


def plot_alignment(df, output_dir, args):
    image_df = df[df["type"] == "image"]
    centroid_df = df[df["type"] == "centroid"]
    text_df = df[df["type"] == "text"]
    labels = sorted(image_df["label"].unique().astype(int))
    colors = color_map_for_labels(labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    for label in labels:
        class_df = image_df[image_df["label"] == label]
        ax.scatter(
            class_df["t-SNE_dim1"],
            class_df["t-SNE_dim2"],
            color=colors[int(label)],
            alpha=0.15,
            s=10,
            linewidths=0,
            zorder=1,
        )

    for label in labels:
        centroid = centroid_df[centroid_df["label"] == label]
        if len(centroid) == 0:
            continue
        ax.scatter(
            centroid["t-SNE_dim1"],
            centroid["t-SNE_dim2"],
            color=colors[int(label)],
            marker="o",
            s=250,
            edgecolors="white",
            linewidths=2,
            zorder=5,
            label="Image Centroids" if label == labels[0] else None,
        )

    for label in labels:
        text = text_df[text_df["label"] == label]
        if len(text) == 0:
            continue
        ax.scatter(
            text["t-SNE_dim1"],
            text["t-SNE_dim2"],
            color=colors[int(label)],
            marker="*",
            s=500,
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            label="Text Anchors" if label == labels[0] else None,
        )

    for label in labels:
        centroid = centroid_df[centroid_df["label"] == label]
        text = text_df[text_df["label"] == label]
        if len(centroid) == 0 or len(text) == 0:
            continue
        ax.plot(
            [float(centroid["t-SNE_dim1"].iloc[0]), float(text["t-SNE_dim1"].iloc[0])],
            [float(centroid["t-SNE_dim2"].iloc[0]), float(text["t-SNE_dim2"].iloc[0])],
            color=colors[int(label)],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            zorder=4,
        )

    ax.set_title("t-SNE: Visual-Text Feature Alignment", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.grid(alpha=0.15)
    handles, labels_text = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels_text, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=10)
    fig.tight_layout()
    save_plot(fig, output_dir, "masterstyle_tsne_alignment")


def save_dataframe(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "masterstyle_tsne_points.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")


def build_output_dir(args, model_dir):
    if args.output_dir:
        return args.output_dir
    model_run = safe_name(os.path.basename(os.path.normpath(model_dir)))
    client_tag = "clients_" + safe_name(args.client_ids.replace(",", "_")) if args.client_ids else f"clients_all{args.num_clients}"
    return os.path.join(
        "./T-SNE-masterstyle",
        safe_name(args.dataset),
        safe_name(args.algorithm),
        safe_name(args.model_source),
        partition_tag(args),
        client_tag,
        f"split_{args.split}",
        model_run,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Master-style FedCLIP t-SNE visualization.")
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--final-model-root", type=str, default="./final_models")
    parser.add_argument("--model-family", "-m", type=str, default="")
    parser.add_argument("--dataset", type=str, default="Cifar100")
    parser.add_argument("--algorithm", type=str, default="FedCLIP")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--join-ratio", "-jr", type=float, default=1.0)
    parser.add_argument("--client-ids", type=str, default="")
    parser.add_argument("--model-source", choices=["server", "client"], default="client")
    parser.add_argument("--output-dir", type=str, default="")

    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--niid", type=int, default=1)
    parser.add_argument("--partition", "-pt", type=str, default="dir")
    parser.add_argument("--dir-alpha", "-dir_alpha", type=float, default=0.3)
    parser.add_argument("--class-per-client", "-cpc", type=int, default=2)

    parser.add_argument("--feature-source", choices=["auto", "base", "forward"], default="auto")
    parser.add_argument("--include-text", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-centroids", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--center-modalities", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anchor-scope", choices=["present", "all"], default="present")

    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--prompt-template", type=str, default="a photo of {}")

    parser.add_argument("--perplexity", type=int, default=50)
    parser.add_argument("--tsne-lr", type=str, default="auto")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--init", type=str, default="random")
    parser.add_argument("--feature-point-size", type=float, default=36)
    parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-legend-classes", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    model_dir = resolve_model_dir(args)
    output_dir = build_output_dir(args, model_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    print("开始 master-style t-SNE...")
    print(f"模型目录: {model_dir}")
    print(f"模型来源: {args.model_source}")
    print(f"数据集: {args.dataset} | partition={args.partition} | alpha={args.dir_alpha} | cpc={args.class_per_client}")
    print(f"输出目录: {output_dir}")

    df = run_masterstyle_tsne(args, model_dir, device)
    save_dataframe(df, output_dir)
    plot_features_only(df, output_dir, args)
    if args.include_centroids or args.include_text:
        plot_alignment(df, output_dir, args)

    print("\nmaster-style t-SNE 完成")
    print(f"总点数: {len(df)}")
    print(f"图像样本数: {len(df[df['type'] == 'image'])}")


if __name__ == "__main__":
    main()
