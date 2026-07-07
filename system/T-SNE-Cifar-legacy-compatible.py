import argparse
import csv
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

from utils.data_utils import read_client_data


RANDOM_SEED = 0

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
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
    "worm",
]


def set_random_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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


def safe_name(value):
    value = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value).strip("_") or "default"


def float_tag(value):
    return str(value).replace(".", "p")


def partition_tag(args):
    if args.partition == "dir":
        return f"dir_alpha{float_tag(args.dir_alpha)}"
    if args.partition == "pat":
        return f"pat_cpc{args.class_per_client}"
    if args.partition == "exdir":
        return f"exdir_cpc{args.class_per_client}_alpha{float_tag(args.dir_alpha)}"
    return safe_name(args.partition)


def data_tag(args):
    return f"ncl{args.num_classes}_niid{args.niid}"


def join_tag(args):
    return f"clients{args.num_clients}_jr{float_tag(args.join_ratio)}"


def find_final_model_dir(args):
    base_dir = Path(args.final_model_root) / safe_name(args.dataset) / safe_name(args.algorithm)
    tail_parts = [partition_tag(args), data_tag(args), join_tag(args)]
    if args.model_family:
        candidate = base_dir / safe_name(args.model_family) / Path(*tail_parts)
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"没有找到最终模型目录: {candidate}\n"
                "如果你要用旧 temp 目录，请显式传入 --model-dir。"
            )
        return str(candidate)

    matches = []
    if base_dir.is_dir():
        for model_family_dir in base_dir.iterdir():
            candidate = model_family_dir / Path(*tail_parts)
            if candidate.is_dir():
                matches.append(candidate)
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(
            f"没有在 {base_dir} 下找到匹配 {tail_parts} 的模型目录。\n"
            "请传入 --model-family 或 --model-dir。"
        )
    raise RuntimeError("匹配到多个模型目录，请用 --model-family 指定一个:\n" + "\n".join(map(str, matches)))


def resolve_model_dir(args):
    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"--model-dir 指定目录不存在: {model_dir}")
        return str(model_dir)
    return find_final_model_dir(args)


def parse_client_ids(client_ids_text, num_clients):
    if not client_ids_text:
        return list(range(num_clients))
    client_ids = []
    for part in client_ids_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            client_ids.extend(range(int(start), int(end) + 1))
        else:
            client_ids.append(int(part))
    return sorted(dict.fromkeys(client_ids))


def candidate_model_paths(model_dir, client_id, model_source):
    if model_source == "client":
        return [Path(model_dir) / f"Client_{client_id}_model.pt"]
    if model_source == "server":
        return [
            Path(model_dir) / f"Server_model_{client_id}.pt",
            Path(model_dir) / "Server_model.pt",
        ]
    raise ValueError(f"未知 model_source: {model_source}")


def resolve_model_path(model_dir, client_id, model_source):
    for path in candidate_model_paths(model_dir, client_id, model_source):
        if path.exists():
            return str(path)
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


def build_data_args(args):
    return SimpleNamespace(
        niid=args.niid,
        partition=args.partition,
        dir_alpha=args.dir_alpha,
        class_per_client=args.class_per_client,
    )


def load_split_data(client_id, args):
    data_args = build_data_args(args)
    data = read_client_data(
        args.dataset,
        client_id,
        args=data_args,
        is_train=(args.split == "train"),
        few_shot=0,
    )
    return DataLoader(data, batch_size=args.batch_size, drop_last=False, shuffle=False)


def extract_legacy_base_features(model, images):
    if hasattr(model, "base"):
        features = model.base(images)
    else:
        features = model(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = torch.flatten(features, start_dim=1)
    return features


def legacy_max_batches(args, auto_all_clients):
    max_batches = 40 if args.max_batches == -1 and auto_all_clients else args.max_batches
    if max_batches == -1:
        max_batches = 0
    return max_batches


def collect_one_client_features(args, model_dir, device, client_id, max_batches, verbose=True):
    model_path = resolve_model_path(model_dir, client_id, args.model_source)
    if verbose:
        print(f"处理客户端 {client_id}: {model_path}")
    model = torch_load_model(model_path, device)
    loader = load_split_data(client_id, args)

    client_features = []
    client_labels = []
    seen = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if args.max_samples_per_client > 0 and seen >= args.max_samples_per_client:
                break

            images = images.to(device)
            features = extract_legacy_base_features(model, images)
            labels_np = labels.numpy()

            if args.max_samples_per_client > 0:
                remaining = args.max_samples_per_client - seen
                features = features[:remaining]
                labels_np = labels_np[:remaining]

            client_features.append(features.detach().cpu().numpy())
            client_labels.append(labels_np)
            seen += len(labels_np)

    if not client_features:
        return None, None
    return np.concatenate(client_features, axis=0), np.concatenate(client_labels, axis=0).astype(int)


def collect_legacy_features(args, model_dir, device):
    client_ids = parse_client_ids(args.client_ids, args.num_clients)
    auto_all_clients = not args.client_ids
    max_batches = legacy_max_batches(args, auto_all_clients)

    all_features = []
    all_labels = []
    all_client_ids = []
    class_names = class_names_for(args.dataset, args.num_classes)

    print("收集客户端特征...")
    print(f"客户端: {client_ids}")
    print(f"split={args.split} | max_batches={max_batches if max_batches > 0 else 'all'} | raw model.base features")

    for client_id in client_ids:
        features_np, labels_np = collect_one_client_features(
            args, model_dir, device, client_id, max_batches=max_batches, verbose=True
        )
        if features_np is None:
            print(f"警告: 客户端 {client_id} 没有收集到样本，跳过。")
            continue

        all_features.append(features_np)
        all_labels.append(labels_np)
        all_client_ids.extend([client_id] * len(labels_np))

        labels_unique, label_counts = np.unique(labels_np, return_counts=True)
        label_text = ", ".join(
            f"{label}:{count}" if label >= len(class_names) else f"{label}({class_names[label]}):{count}"
            for label, count in zip(labels_unique, label_counts)
        )
        print(f"客户端 {client_id}: 收集 {len(labels_np)} 个样本，特征维度 {features_np.shape[1]} | {label_text}")

    if not all_features:
        raise RuntimeError("没有成功收集任何特征。")

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    combined_client_ids = np.asarray(all_client_ids)
    print(f"总特征形状: {combined_features.shape}")
    print(f"总标签形状: {combined_labels.shape}")
    return combined_features, combined_labels, combined_client_ids


def deterministic_sample(features, labels, max_samples, seed):
    if max_samples <= 0 or len(labels) <= max_samples:
        return features, labels
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(labels), size=max_samples, replace=False)
    return features[indices], labels[indices]


def compute_client_quality(features, labels, args, client_id):
    unique_labels, counts = np.unique(labels, return_counts=True)
    row = {
        "client_id": int(client_id),
        "num_samples": int(len(labels)),
        "num_classes": int(len(unique_labels)),
        "min_class_samples": int(counts.min()) if len(counts) else 0,
        "max_class_samples": int(counts.max()) if len(counts) else 0,
        "silhouette": float("nan"),
        "mean_intra": float("nan"),
        "min_inter_centroid": float("nan"),
        "separation": float("nan"),
        "score": float("-inf"),
    }
    if len(unique_labels) < 2:
        return row

    centroids = []
    intra_values = []
    for label in unique_labels:
        class_features = features[labels == label]
        centroid = class_features.mean(axis=0)
        centroids.append(centroid)
        intra_values.append(np.linalg.norm(class_features - centroid, axis=1).mean())
    centroids = np.stack(centroids, axis=0)
    mean_intra = float(np.mean(intra_values))

    inter_values = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            inter_values.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    min_inter = min(inter_values) if inter_values else float("nan")
    separation = min_inter / (mean_intra + 1e-8) if inter_values else float("nan")

    row["mean_intra"] = mean_intra
    row["min_inter_centroid"] = min_inter
    row["separation"] = float(separation)

    sample_features, sample_labels = deterministic_sample(
        features,
        labels,
        args.selection_max_samples,
        seed=args.seed + int(client_id),
    )
    if len(np.unique(sample_labels)) >= 2 and len(sample_labels) > len(np.unique(sample_labels)):
        try:
            from sklearn.metrics import silhouette_score
            row["silhouette"] = float(
                silhouette_score(sample_features, sample_labels, metric=args.selection_metric)
            )
        except Exception as exc:
            row["silhouette_error"] = repr(exc)

    if args.selection_score == "silhouette":
        row["score"] = row["silhouette"] if np.isfinite(row["silhouette"]) else float("-inf")
    elif args.selection_score == "separation":
        row["score"] = row["separation"] if np.isfinite(row["separation"]) else float("-inf")
    else:
        raise ValueError(f"未知 selection_score: {args.selection_score}")
    return row


def save_selection_metrics(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "client_selection_metrics.csv"
    fieldnames = [
        "client_id",
        "num_samples",
        "num_classes",
        "min_class_samples",
        "max_class_samples",
        "silhouette",
        "mean_intra",
        "min_inter_centroid",
        "separation",
        "score",
        "silhouette_error",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"客户端选择指标已保存: {path}")


def auto_select_best_client(args, model_dir, device):
    candidate_ids = parse_client_ids(args.client_ids, args.num_clients)
    max_batches = args.selection_max_batches
    if max_batches < 0:
        max_batches = 40
    rows = []

    print("自动评估最适合画 t-SNE 的客户端...")
    print(
        f"候选客户端: {candidate_ids} | selection_score={args.selection_score} | "
        f"silhouette_metric={args.selection_metric} | max_batches={max_batches} | "
        f"max_samples={args.selection_max_samples}"
    )
    for client_id in candidate_ids:
        try:
            features, labels = collect_one_client_features(
                args, model_dir, device, client_id, max_batches=max_batches, verbose=False
            )
            if features is None:
                rows.append({"client_id": client_id, "score": float("-inf"), "silhouette_error": "no samples"})
                continue
            row = compute_client_quality(features, labels, args, client_id)
            rows.append(row)
            print(
                f"Client_{client_id:02d}: score={row['score']:.6f} | "
                f"silhouette={row['silhouette']:.6f} | separation={row['separation']:.6f} | "
                f"samples={row['num_samples']} | classes={row['num_classes']}"
            )
        except Exception as exc:
            rows.append({"client_id": client_id, "score": float("-inf"), "silhouette_error": repr(exc)})
            print(f"Client_{client_id:02d}: 评估失败: {exc}")

    valid_rows = [row for row in rows if np.isfinite(row.get("score", float("-inf")))]
    if not valid_rows:
        raise RuntimeError("自动选客户端失败：没有任何客户端得到有效 score。")
    best = max(valid_rows, key=lambda row: row["score"])
    rows = sorted(rows, key=lambda row: row.get("score", float("-inf")), reverse=True)
    print(
        f"✅ 自动选择 Client_{best['client_id']} | "
        f"score={best['score']:.6f} | silhouette={best['silhouette']:.6f} | "
        f"separation={best['separation']:.6f}"
    )
    return int(best["client_id"]), rows


def run_tsne(features, args):
    try:
        from sklearn.manifold import TSNE
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行 t-SNE 需要 scikit-learn；请先在当前环境安装 scikit-learn。") from exc

    perplexity = min(args.perplexity, max(1, len(features) - 1))
    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.tsne_lr,
        random_state=args.seed,
        verbose=1,
    )
    try:
        tsne = TSNE(max_iter=args.max_iter, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=args.max_iter, **kwargs)
    return tsne.fit_transform(features)


def make_dataframe(coords, labels, client_ids, args):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("运行该脚本需要 pandas；请先在当前环境安装 pandas。") from exc

    class_names = class_names_for(args.dataset, args.num_classes)
    return pd.DataFrame({
        "client_id": client_ids,
        "label": labels,
        "class_name": [
            class_names[int(label)] if int(label) < len(class_names) else f"class_{int(label)}"
            for label in labels
        ],
        "t-SNE_dim1": coords[:, 0],
        "t-SNE_dim2": coords[:, 1],
    })


def save_outputs(df, output_dir, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "legacy_tsne_all_points.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")
    if args.save_excel:
        xlsx_path = output_dir / "legacy_tsne_all_points.xlsx"
        try:
            df.to_excel(xlsx_path, index=False)
            print(f"Excel: {xlsx_path}")
        except Exception as exc:
            print(f"Excel 保存失败，已跳过: {exc}")

    for client_id in sorted(df["client_id"].unique()):
        client_df = df[df["client_id"] == client_id]
        client_df.to_csv(output_dir / f"client_{client_id}_legacy_tsne.csv", index=False)


def plot_legacy_tsne(df, output_dir, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("绘制 t-SNE 图需要 matplotlib 和 seaborn；请先在当前环境安装。") from exc

    output_dir = Path(output_dir)
    labels = sorted(df["label"].unique().astype(int))
    palette = sns.color_palette("tab10", len(labels)) if len(labels) <= 10 else sns.color_palette("husl", len(labels))
    color_map = {label: palette[idx] for idx, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(12, 10))
    for label in labels:
        class_df = df[df["label"] == label]
        ax.scatter(
            class_df["t-SNE_dim1"],
            class_df["t-SNE_dim2"],
            color=color_map[label],
            alpha=args.alpha,
            s=args.point_size,
            linewidths=0,
            label=str(label),
        )

    ax.set_title(f"{args.dataset} {args.partition} {args.split} t-SNE", fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(alpha=0.15)
    if args.show_legend and len(labels) <= args.max_legend_classes:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=9)
    fig.tight_layout()

    png_path = output_dir / "legacy_tsne_visualization.png"
    pdf_path = output_dir / "legacy_tsne_visualization.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


def build_output_dir(args, model_dir):
    if args.output_dir:
        return args.output_dir
    client_tag = "clients_all" if not args.client_ids else "clients_" + safe_name(args.client_ids.replace(",", "_"))
    model_run = safe_name(Path(model_dir).name)
    return str(Path("./T-SNE-legacy") / safe_name(args.dataset) / safe_name(args.algorithm) /
               safe_name(args.model_source) / partition_tag(args) / client_tag /
               f"split_{args.split}" / model_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Legacy-compatible t-SNE: raw model.base features, train split by default, no anchors/normalization."
    )
    parser.add_argument("--model-dir", type=str, default="")
    parser.add_argument("--final-model-root", type=str, default="./final_models")
    parser.add_argument("--model-family", "-m", type=str, default="")
    parser.add_argument("--dataset", type=str, default="Cifar100")
    parser.add_argument("--algorithm", type=str, default="FedCLIP")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--join-ratio", "-jr", type=float, default=1.0)
    parser.add_argument("--client-ids", type=str, default="",
                        help="为空时按旧 all-client 脚本画全部客户端；例如 19 或 0,5,10,15。")
    parser.add_argument("--auto-best-client", action="store_true",
                        help="自动扫描候选客户端并选择最适合画 t-SNE 的客户端；候选集由 --client-ids 控制，为空则扫描全部客户端。")
    parser.add_argument("--selection-score", choices=["silhouette", "separation"], default="silhouette",
                        help="自动选客户端时使用的分数。silhouette 更贴近 t-SNE 簇分离，separation 是类中心最小间距/类内距离。")
    parser.add_argument("--selection-metric", choices=["euclidean", "cosine"], default="euclidean",
                        help="计算 silhouette 时使用的高维距离。旧 t-SNE 脚本更接近 euclidean。")
    parser.add_argument("--selection-max-batches", type=int, default=40,
                        help="自动选客户端时每个客户端最多读取多少 batch；设 0 表示不限制。")
    parser.add_argument("--selection-max-samples", type=int, default=1200,
                        help="自动选客户端时 silhouette 最多采样多少样本，避免 O(n^2) 太慢；设 0 表示不采样。")
    parser.add_argument("--model-source", choices=["client", "server"], default="client")
    parser.add_argument("--output-dir", type=str, default="")

    parser.add_argument("--split", "--data-split", dest="split", choices=["train", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int, default=-1,
                        help="旧 all-client 脚本默认每客户端 40 batch；指定 0 表示不限制。单客户端自动不限制。")
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--niid", type=int, default=1)
    parser.add_argument("--partition", "-pt", type=str, default="pat")
    parser.add_argument("--dir-alpha", "-dir_alpha", type=float, default=0.3)
    parser.add_argument("--class-per-client", "-cpc", type=int, default=10)

    parser.add_argument("--perplexity", type=int, default=30)
    parser.add_argument("--tsne-lr", type=float, default=200)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=18)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-legend-classes", type=int, default=20)
    parser.add_argument("--save-excel", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_dir = resolve_model_dir(args)

    selection_rows = None
    if args.auto_best_client:
        best_client_id, selection_rows = auto_select_best_client(args, model_dir, device)
        args.client_ids = str(best_client_id)

    output_dir = build_output_dir(args, model_dir)

    print("开始旧脚本兼容 t-SNE...")
    print(f"模型目录: {model_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型来源: {args.model_source}")
    print(f"数据集: {args.dataset} | partition={args.partition} | alpha={args.dir_alpha} | cpc={args.class_per_client}")
    print("协议: raw model.base features, no L2 normalize, no modality centering, no text anchors")
    if selection_rows is not None:
        save_selection_metrics(selection_rows, output_dir)

    features, labels, client_ids = collect_legacy_features(args, model_dir, device)
    coords = run_tsne(features, args)
    df = make_dataframe(coords, labels, client_ids, args)
    save_outputs(df, output_dir, args)
    plot_legacy_tsne(df, output_dir, args)
    print("完成。")


if __name__ == "__main__":
    main()
