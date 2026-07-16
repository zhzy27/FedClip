#!/usr/bin/env python

"""Evaluate whether learned low-rank components are ordered by importance.

The script loads the highest-capacity client model from a completed experiment,
keeps only the leading low-rank components at 100%, 95%, ..., 0%, and evaluates
every truncated model on that client's local training split.
"""

import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data_utils import read_client_data


CLIENT_MODEL_PATTERN = re.compile(r"^Client_(\d+)_model\.pt$")
SCRIPT_DIR = Path(__file__).resolve().parent


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(value):
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))
    return text.strip("_") or "default"


def float_tag(value):
    return str(value).replace(".", "p")


def partition_tag(args):
    if args.partition == "dir":
        return f"dir_alpha{float_tag(args.dir_alpha)}"
    if args.partition == "pat":
        return f"pat_cpc{args.class_per_client}"
    if args.partition == "exdir":
        return f"exdir_alpha{float_tag(args.dir_alpha)}"
    return safe_name(args.partition)


def resolve_model_dir(args):
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        if not args.model_family:
            raise ValueError("未指定 --model-dir 时必须提供 --model-family。")
        model_dir = (
            Path(args.final_model_root)
            / safe_name(args.dataset)
            / safe_name(args.algorithm)
            / safe_name(args.model_family)
            / partition_tag(args)
            / f"ncl{args.num_classes}_niid{args.niid}"
            / f"clients{args.num_clients}_jr{float_tag(args.join_ratio)}"
        ).resolve()

    if not model_dir.is_dir():
        raise FileNotFoundError(f"最终模型目录不存在: {model_dir}")
    return model_dir


def torch_load_model(model_path, map_location="cpu"):
    try:
        return torch.load(model_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=map_location)


def discover_factorized_layers(model):
    layers = []
    for name, module in model.named_modules():
        if hasattr(module, "conv_u") and hasattr(module, "conv_v"):
            u_param = module.conv_u
            v_param = module.conv_v
            kind = "conv"
        elif hasattr(module, "weight_u") and hasattr(module, "weight_v"):
            u_param = module.weight_u
            v_param = module.weight_v
            kind = "linear"
        else:
            continue

        if not isinstance(u_param, nn.Parameter) or not isinstance(v_param, nn.Parameter):
            continue
        if u_param.ndim != 2 or v_param.ndim != 2 or u_param.shape[1] != v_param.shape[0]:
            raise RuntimeError(
                f"低秩层 {name} 的 U/V 形状不匹配: U={tuple(u_param.shape)}, V={tuple(v_param.shape)}"
            )
        rank = int(v_param.shape[0])
        if rank <= 0:
            raise RuntimeError(f"低秩层 {name} 的秩非法: {rank}")
        layers.append(
            {
                "name": name,
                "module": module,
                "kind": kind,
                "rank": rank,
                "max_rank": int(getattr(module, "max_rank", rank)),
                "rank_rate": float(getattr(module, "rank_rate", rank)),
            }
        )
    return layers


def model_capacity(model):
    layers = discover_factorized_layers(model)
    if not layers:
        return None
    ratio = getattr(model, "ratio_LR", None)
    if ratio is None and hasattr(model, "base"):
        ratio = getattr(model.base, "ratio_LR", None)
    if ratio is None:
        ratio = float(np.mean([layer["rank"] / max(1, layer["max_rank"]) for layer in layers]))
    return {
        "ratio": float(ratio),
        "total_rank": int(sum(layer["rank"] for layer in layers)),
        "layer_count": len(layers),
    }


def client_model_files(model_dir):
    files = []
    for path in Path(model_dir).glob("Client_*_model.pt"):
        match = CLIENT_MODEL_PATTERN.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files)


def select_highest_rank_client(model_dir, requested_client_id=None):
    model_files = client_model_files(model_dir)
    if not model_files:
        raise FileNotFoundError(f"目录中没有 Client_<id>_model.pt: {model_dir}")

    if requested_client_id is not None:
        matching = [(cid, path) for cid, path in model_files if cid == requested_client_id]
        if not matching:
            raise FileNotFoundError(f"找不到 Client_{requested_client_id}_model.pt: {model_dir}")
        model = torch_load_model(matching[0][1])
        capacity = model_capacity(model)
        if capacity is None:
            raise RuntimeError(f"Client_{requested_client_id} 模型中没有可识别的低秩层。")
        return requested_client_id, matching[0][1], model, [dict(client_id=requested_client_id, **capacity)]

    candidates = []
    selected = None
    for client_id, model_path in model_files:
        model = torch_load_model(model_path)
        capacity = model_capacity(model)
        if capacity is None:
            del model
            continue
        record = dict(client_id=client_id, model_path=str(model_path), **capacity)
        candidates.append(record)
        score = (capacity["ratio"], capacity["total_rank"], -client_id)
        if selected is None or score > selected[0]:
            selected = (score, client_id, model_path, model)
        else:
            del model

    if selected is None:
        raise RuntimeError(f"所有客户端模型中都没有可识别的低秩层: {model_dir}")
    _, client_id, model_path, model = selected
    return client_id, model_path, model, candidates


def retention_percentages(start, stop, step):
    if not (0 <= stop <= start <= 100):
        raise ValueError("保留率必须满足 0 <= stop <= start <= 100。")
    if step <= 0:
        raise ValueError("--retention-step 必须大于 0。")
    values = []
    current = start
    while current >= stop:
        values.append(float(current))
        current -= step
    if not math.isclose(values[-1], float(stop)):
        values.append(float(stop))
    return values


def kept_rank(rank, retention_percent):
    if retention_percent <= 0:
        return 0
    if retention_percent >= 100:
        return rank
    return max(1, min(rank, math.floor(rank * retention_percent / 100.0)))


class RankPrefixController:
    def __init__(self, model):
        self.layers = discover_factorized_layers(model)
        if not self.layers:
            raise RuntimeError("模型中没有 conv_u/conv_v 或 weight_u/weight_v 低秩层。")
        for layer in self.layers:
            module = layer["module"]
            if layer["kind"] == "conv":
                layer["u_original"] = module.conv_u.detach().clone()
                layer["v_original"] = module.conv_v.detach().clone()
            else:
                layer["u_original"] = module.weight_u.detach().clone()
                layer["v_original"] = module.weight_v.detach().clone()

    def apply(self, retention_percent):
        layer_records = []
        total_rank = 0
        total_kept = 0
        with torch.no_grad():
            for layer in self.layers:
                module = layer["module"]
                if layer["kind"] == "conv":
                    u_param = module.conv_u
                    v_param = module.conv_v
                else:
                    u_param = module.weight_u
                    v_param = module.weight_v

                u_param.copy_(layer["u_original"])
                v_param.copy_(layer["v_original"])
                keep = kept_rank(layer["rank"], retention_percent)
                if keep < layer["rank"]:
                    u_param[:, keep:].zero_()
                    v_param[keep:, :].zero_()

                total_rank += layer["rank"]
                total_kept += keep
                layer_records.append(
                    {
                        "name": layer["name"],
                        "kind": layer["kind"],
                        "original_rank": layer["rank"],
                        "kept_rank": keep,
                    }
                )

        actual_percent = 100.0 * total_kept / max(1, total_rank)
        return total_kept, total_rank, actual_percent, layer_records

    def restore(self):
        self.apply(100.0)


def build_train_loader(args, client_id, device):
    data_args = SimpleNamespace(
        niid=args.niid,
        partition=args.partition,
        dir_alpha=args.dir_alpha,
        class_per_client=args.class_per_client,
    )
    # data_utils uses paths relative to system/, so make the script independent
    # of the directory from which it is launched.
    previous_cwd = Path.cwd()
    try:
        os.chdir(SCRIPT_DIR)
        train_data = read_client_data(args.dataset, client_id, data_args, is_train=True, few_shot=0)
    finally:
        os.chdir(previous_cwd)
    return DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )


def evaluate(model, loader, device):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for inputs, targets in loader:
            if isinstance(inputs, list):
                inputs[0] = inputs[0].to(device, non_blocking=True)
            else:
                inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            loss_sum += criterion(outputs, targets).item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
    if total == 0:
        raise RuntimeError("训练集为空，无法计算准确率。")
    return correct / total, loss_sum / total, correct, total


def resolve_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境中 torch.cuda.is_available() 为 False。")
    if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
        raise ValueError(
            f"--device-id={args.device_id} 超出范围，当前可见 GPU 数量为 {torch.cuda.device_count()}。"
        )
    return torch.device(f"cuda:{args.device_id}")


def default_output_dir(args, model_dir, client_id):
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return (
        Path("./figures/rank_retention")
        / safe_name(args.dataset)
        / safe_name(args.algorithm)
        / safe_name(args.model_family or Path(model_dir).parts[-5])
        / partition_tag(args)
        / f"client_{client_id}"
    ).resolve()


def write_csv(path, rows):
    fieldnames = [
        "client_id",
        "requested_retention_percent",
        "actual_component_retention_percent",
        "kept_rank_components",
        "total_rank_components",
        "train_accuracy",
        "train_accuracy_percent",
        "train_loss",
        "correct_samples",
        "total_samples",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(rows, output_dir, title, dpi):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("生成 PNG/PDF 需要 matplotlib，请先安装该依赖。") from error

    ordered = sorted(rows, key=lambda row: row["requested_retention_percent"])
    x = [row["requested_retention_percent"] for row in ordered]
    y = [row["train_accuracy_percent"] for row in ordered]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(x, y, marker="o", markersize=4.5, linewidth=1.8, color="#2878B5")
    ax.set_xlabel("Retained rank (%)")
    ax.set_ylabel("Training accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 10))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    fig.tight_layout()

    png_path = output_dir / "ordered_rank_retention_accuracy.png"
    pdf_path = output_dir / "ordered_rank_retention_accuracy.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="扫描低秩模型的前缀秩保留率，并在最高容量客户端的本地训练集上评估准确率。"
    )
    parser.add_argument("--model-dir", type=str, default="", help="最终模型目录；不指定时按实验配置自动查找。")
    parser.add_argument("--final-model-root", type=str, default="./final_models")
    parser.add_argument("-data", "--dataset", type=str, required=True)
    parser.add_argument("-algo", "--algorithm", type=str, default="FedCLIP")
    parser.add_argument("-m", "--model-family", type=str, default="Decom_CNN-5-512")
    parser.add_argument("-pt", "--partition", type=str, choices=["iid", "dir", "pat", "exdir"], default="dir")
    parser.add_argument("-dir_alpha", "--dir-alpha", "--dir_alpha", dest="dir_alpha", type=float, default=0.3)
    parser.add_argument(
        "-cpc", "--class-per-client", "--class_per_client", "--cpc", dest="class_per_client", type=int, default=2
    )
    parser.add_argument("-ncl", "--num-classes", "--num_classes", dest="num_classes", type=int, required=True)
    parser.add_argument("--niid", type=int, default=1)
    parser.add_argument("-nc", "--num-clients", "--num_clients", dest="num_clients", type=int, default=20)
    parser.add_argument("-jr", "--join-ratio", "--join_ratio", dest="join_ratio", type=float, default=1.0)
    parser.add_argument(
        "--client-id",
        "--client_id",
        dest="client_id",
        type=int,
        default=None,
        help="默认自动选择 ratio_LR 和总秩最高的客户端；指定后仅使用该客户端。",
    )
    parser.add_argument("--retention-start", type=float, default=100.0)
    parser.add_argument("--retention-stop", type=float, default=0.0)
    parser.add_argument("--retention-step", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("-did", "--device-id", "--device_id", dest="device_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = resolve_device(args)
    model_dir = resolve_model_dir(args)

    print(f"最终模型目录: {model_dir}")
    print("正在检查客户端模型容量...")
    client_id, model_path, model, candidates = select_highest_rank_client(model_dir, args.client_id)
    selected_capacity = model_capacity(model)
    print("检测到的低秩客户端:")
    for item in sorted(candidates, key=lambda value: value["client_id"]):
        marker = " <- selected" if item["client_id"] == client_id else ""
        print(
            f"  Client_{item['client_id']}: ratio_LR={item['ratio']:.6g}, "
            f"total_rank={item['total_rank']}, layers={item['layer_count']}{marker}"
        )
    print(f"选中模型: {model_path}")

    model = model.to(device)
    model.eval()
    controller = RankPrefixController(model)
    loader = build_train_loader(args, client_id, device)
    percentages = retention_percentages(args.retention_start, args.retention_stop, args.retention_step)

    print("低秩层信息:")
    for layer in controller.layers:
        print(
            f"  {layer['name']}: type={layer['kind']}, rank={layer['rank']}, "
            f"max_rank={layer['max_rank']}, rank_rate={layer['rank_rate']:.6g}"
        )
    print("评估使用 model.eval()；仅裁剪低秩层末尾分量，满秩层、bias 和归一化参数保持不变。")

    rows = []
    layer_details = {}
    try:
        for index, retention in enumerate(percentages, start=1):
            kept, total_rank, actual_percent, details = controller.apply(retention)
            accuracy, loss, correct, total = evaluate(model, loader, device)
            row = {
                "client_id": client_id,
                "requested_retention_percent": retention,
                "actual_component_retention_percent": actual_percent,
                "kept_rank_components": kept,
                "total_rank_components": total_rank,
                "train_accuracy": accuracy,
                "train_accuracy_percent": accuracy * 100.0,
                "train_loss": loss,
                "correct_samples": correct,
                "total_samples": total,
            }
            rows.append(row)
            layer_details[str(retention)] = details
            print(
                f"[{index:02d}/{len(percentages):02d}] requested={retention:6.1f}% | "
                f"actual={actual_percent:6.2f}% ({kept}/{total_rank}) | "
                f"train_acc={accuracy * 100.0:7.3f}% | loss={loss:.6f}"
            )
    finally:
        controller.restore()

    output_dir = default_output_dir(args, model_dir, client_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ordered_rank_retention_results.csv"
    write_csv(csv_path, rows)

    title = f"{args.dataset}: ordered-rank retention (Client {client_id})"
    png_path, pdf_path = plot_curve(rows, output_dir, title, args.dpi)
    metadata = {
        "model_dir": str(model_dir),
        "model_path": str(model_path),
        "selected_client_id": client_id,
        "selected_capacity": selected_capacity,
        "selection_candidates": candidates,
        "dataset": args.dataset,
        "partition": args.partition,
        "dir_alpha": args.dir_alpha,
        "class_per_client": args.class_per_client,
        "evaluation_split": "train",
        "rank_policy": "keep the leading prefix independently in every factorized layer",
        "rank_rounding": "floor, with minimum rank 1 for positive retention and rank 0 at 0%",
        "factorized_layers": [
            {
                "name": layer["name"],
                "kind": layer["kind"],
                "rank": layer["rank"],
                "max_rank": layer["max_rank"],
                "rank_rate": layer["rank_rate"],
            }
            for layer in controller.layers
        ],
        "layer_details_by_retention": layer_details,
    }
    metadata_path = output_dir / "ordered_rank_retention_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("完成。")
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
