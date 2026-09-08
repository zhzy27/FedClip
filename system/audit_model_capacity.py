"""Audit FedLAP model architectures and capacity configurations.

This script reads the active model-family definitions from ``system/main.py``,
instantiates the real model classes, and reports parameter counts and the
realized rank of every factorized layer. Run it from either the repository
root or ``system`` directory:

    python system/audit_model_capacity.py
    python audit_model_capacity.py

The generated files are intended to be the numerical source for the appendix
tables. No parameter count is inferred from a closed-form approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn


SYSTEM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SYSTEM_DIR.parent
MAIN_PATH = SYSTEM_DIR / "main.py"
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from flcore.trainmodel import models as model_defs  # noqa: E402
from flcore.trainmodel.SVD_resnet import (  # noqa: E402
    layer_norm,
    low_rank_resnet18_cifar,
)
from flcore.trainmodel.resnet18_family import resnet18_family  # noqa: E402


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    input_size: int


DATASETS = (
    DatasetSpec("CIFAR-10", 10, 32),
    DatasetSpec("CIFAR-100", 100, 32),
    DatasetSpec("Tiny-ImageNet", 200, 64),
)


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _branch_block(lines: Sequence[str], marker: str) -> List[str]:
    start = next((i for i, line in enumerate(lines) if marker in line), None)
    if start is None:
        raise RuntimeError(f"Could not find main.py branch marker: {marker}")
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("        elif ") or line.startswith("        else:"):
            break
        block.append(line)
    return block


def _models_list_lines(block: Sequence[str]) -> List[str]:
    start = next(
        (i for i, line in enumerate(block) if line.strip() == "args.models = ["),
        None,
    )
    if start is None:
        raise RuntimeError("Could not find args.models list in model-family branch")
    result = []
    for line in block[start + 1 :]:
        if line.strip() == "]":
            return result
        result.append(line)
    raise RuntimeError("Unterminated args.models list in main.py")


def _parse_main_configuration() -> Dict[str, Any]:
    lines = MAIN_PATH.read_text(encoding="utf-8").splitlines()

    cnn_low_block = _branch_block(lines, 'args.model_family == "Decom_CNN-5-512"')
    cnn_full_block = _branch_block(lines, 'args.model_family == "CNN-5-512"')
    cnn_tiny_block = _branch_block(lines, 'args.model_family == "CNN-5-512-tiny"')
    res_low_block = _branch_block(lines, 'args.model_family == "Decom_resnet18_5"')
    res_full_block = _branch_block(
        lines, 'args.model_family in ["ResNet18-5-AFM", "ResNet18-5"]'
    )

    def ratios(block: Sequence[str]) -> List[float]:
        model_lines = _models_list_lines(block)
        values = [
            float(value)
            for line in model_lines
            for value in re.findall(r"ratio_LR\s*=\s*([0-9.]+)", line)
        ]
        if len(values) != 5:
            raise RuntimeError(f"Expected five rank ratios, found {values}")
        return values

    def class_names(block: Sequence[str]) -> List[str]:
        names = []
        for line in _models_list_lines(block):
            match = re.search(r"['\"]([A-Za-z_][A-Za-z0-9_]*)\(", line)
            if match:
                names.append(match.group(1))
        if len(names) != 5:
            raise RuntimeError(f"Expected five model classes, found {names}")
        return names

    width_text = "\n".join(res_full_block)
    width_match = re.search(r"resnet18_widths\s*=\s*\[([^]]+)\]", width_text)
    if width_match is None:
        raise RuntimeError("Could not read ResNet-18 width list from main.py")
    resnet_widths = [int(value.strip()) for value in width_match.group(1).split(",")]

    main_text = "\n".join(lines)
    feature_match = re.search(
        r"add_argument\('-fd',\s*\"--feature_dim\".*?default\s*=\s*([0-9]+)",
        main_text,
    )
    if feature_match is None:
        raise RuntimeError("Could not read --feature_dim default from main.py")

    assignment_token = "args.models[cid % len(args.models)]"
    model_source = (SYSTEM_DIR / "flcore/trainmodel/models.py").read_text(
        encoding="utf-8"
    )
    if assignment_token not in model_source:
        raise RuntimeError(
            "Client assignment is no longer cid % len(args.models); update the audit script."
        )

    return {
        "cnn_rank_ratios_main_order": ratios(cnn_low_block),
        "resnet_rank_ratios_main_order": ratios(res_low_block),
        "cnn_full_classes_cifar_main_order": class_names(cnn_full_block),
        "cnn_full_classes_tiny_main_order": class_names(cnn_tiny_block),
        "resnet_full_widths_main_order": resnet_widths,
        "resnet_full_feature_dim": int(feature_match.group(1)),
        "client_assignment_expression": assignment_token,
    }


def _parameter_count(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _factorized_layer_records(model: nn.Module) -> List[Dict[str, Any]]:
    records = []
    for name, module in model.named_modules():
        if not hasattr(module, "rank") or not hasattr(module, "max_rank"):
            continue
        factor_names = [
            candidate
            for candidate in ("conv_u", "conv_v", "weight_u", "weight_v")
            if hasattr(module, candidate)
        ]
        if len(factor_names) != 2:
            continue
        factors = {
            factor_name: list(getattr(module, factor_name).shape)
            for factor_name in factor_names
        }
        records.append(
            {
                "layer": name,
                "module_class": type(module).__name__,
                "configured_rank_ratio": float(getattr(module, "rank_rate")),
                "actual_rank": int(getattr(module, "rank")),
                "maximum_unfolded_rank": int(getattr(module, "max_rank")),
                "actual_rank_fraction": float(
                    getattr(module, "rank") / getattr(module, "max_rank")
                ),
                "factor_shapes": factors,
                "layer_parameters": sum(
                    parameter.numel() for parameter in module.parameters(recurse=False)
                ),
                "layer_trainable_parameters": sum(
                    parameter.numel()
                    for parameter in module.parameters(recurse=False)
                    if parameter.requires_grad
                ),
            }
        )
    return records


def _build_cnn_low(dataset: DatasetSpec, ratio: float) -> nn.Module:
    return model_defs.Hyper_CNN_512(
        in_features=3,
        num_classes=dataset.num_classes,
        n_kernels=16,
        ratio_LR=ratio,
        input_size=dataset.input_size,
    )


def _build_resnet_low(dataset: DatasetSpec, ratio: float) -> nn.Module:
    return low_rank_resnet18_cifar(
        features=[64, 128, 256, 512],
        num_classes=dataset.num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=layer_norm,
        has_norm=True,
        bn_block_num=4,
        ratio_LR=ratio,
        input_size=dataset.input_size,
    )


def _build_cnn_full(dataset: DatasetSpec, class_name: str) -> nn.Module:
    model_class = getattr(model_defs, class_name)
    return model_class(in_channels=3, n_kernels=16, out_dim=dataset.num_classes)


def _build_resnet_full(
    dataset: DatasetSpec, base_width: int, feature_dim: int
) -> nn.Module:
    return resnet18_family(
        in_channels=3,
        num_classes=dataset.num_classes,
        base_width=base_width,
        input_size=dataset.input_size,
        feature_dim=feature_dim,
    )


def _capacity_labels(ratios: Sequence[float]) -> Dict[int, str]:
    sorted_indices = sorted(range(len(ratios)), key=lambda index: ratios[index])
    return {raw_index: f"L{level}" for level, raw_index in enumerate(sorted_indices, 1)}


def _client_ids_for_raw_index(raw_index: int, num_models: int, num_clients: int) -> List[int]:
    return [cid for cid in range(num_clients) if cid % num_models == raw_index]


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_millions(value: int) -> str:
    return f"{value / 1_000_000:.3f}"


def _make_capacity_tex(capacity_rows: Sequence[Dict[str, Any]]) -> str:
    lookup = {
        (row["architecture"], row["capacity"], row["dataset"]): row
        for row in capacity_rows
    }
    lines = []
    for architecture in ("CNN", "ResNet-18"):
        lines.append(f"% {architecture}")
        for level in range(1, 6):
            capacity = f"L{level}"
            c10 = lookup[(architecture, capacity, "CIFAR-10")]
            c100 = lookup[(architecture, capacity, "CIFAR-100")]
            tiny = lookup[(architecture, capacity, "Tiny-ImageNet")]
            lines.append(
                " & ".join(
                    [
                        capacity,
                        f"{c10['rank_ratio']:.2f}",
                        f"{_format_millions(c10['low_rank_total_parameters'])} / "
                        f"{_format_millions(c100['low_rank_total_parameters'])}",
                        _format_millions(tiny["low_rank_total_parameters"]),
                        f"{_format_millions(c10['full_rank_total_parameters'])} / "
                        f"{_format_millions(c100['full_rank_total_parameters'])}",
                        _format_millions(tiny["full_rank_total_parameters"]),
                        str(c10["number_of_clients"]),
                    ]
                )
                + r" \\"
            )
    return "\n".join(lines) + "\n"


def audit(num_clients: int) -> Dict[str, Any]:
    config = _parse_main_configuration()
    capacity_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    assignment_rows: List[Dict[str, Any]] = []

    architecture_configs = (
        (
            "CNN",
            config["cnn_rank_ratios_main_order"],
            _build_cnn_low,
        ),
        (
            "ResNet-18",
            config["resnet_rank_ratios_main_order"],
            _build_resnet_low,
        ),
    )

    for architecture, ratios, low_builder in architecture_configs:
        labels = _capacity_labels(ratios)
        for raw_index, ratio in enumerate(ratios):
            capacity = labels[raw_index]
            client_ids = _client_ids_for_raw_index(raw_index, len(ratios), num_clients)
            assignment_rows.append(
                {
                    "architecture": architecture,
                    "capacity": capacity,
                    "main_model_index": raw_index,
                    "rank_ratio": ratio,
                    "client_ids": ",".join(str(cid) for cid in client_ids),
                    "number_of_clients": len(client_ids),
                }
            )

            for dataset in DATASETS:
                low_model = low_builder(dataset, ratio)
                if architecture == "CNN":
                    class_names = (
                        config["cnn_full_classes_tiny_main_order"]
                        if dataset.input_size == 64
                        else config["cnn_full_classes_cifar_main_order"]
                    )
                    full_model = _build_cnn_full(dataset, class_names[raw_index])
                    full_descriptor = class_names[raw_index]
                else:
                    width = config["resnet_full_widths_main_order"][raw_index]
                    full_model = _build_resnet_full(
                        dataset, width, config["resnet_full_feature_dim"]
                    )
                    full_descriptor = f"ResNet18Family(base_width={width}, feature_dim=512)"

                low_total, low_trainable = _parameter_count(low_model)
                full_total, full_trainable = _parameter_count(full_model)
                capacity_rows.append(
                    {
                        "architecture": architecture,
                        "capacity": capacity,
                        "main_model_index": raw_index,
                        "rank_ratio": ratio,
                        "dataset": dataset.name,
                        "input_size": dataset.input_size,
                        "num_classes": dataset.num_classes,
                        "low_rank_model_class": type(low_model).__name__,
                        "low_rank_total_parameters": low_total,
                        "low_rank_trainable_parameters": low_trainable,
                        "full_rank_model": full_descriptor,
                        "full_rank_total_parameters": full_total,
                        "full_rank_trainable_parameters": full_trainable,
                        "number_of_clients": len(client_ids),
                        "client_ids": ",".join(str(cid) for cid in client_ids),
                    }
                )

                for layer in _factorized_layer_records(low_model):
                    layer_rows.append(
                        {
                            "architecture": architecture,
                            "capacity": capacity,
                            "dataset": dataset.name,
                            "model_class": type(low_model).__name__,
                            "rank_ratio": ratio,
                            **layer,
                        }
                    )

    aligners = nn.ModuleList(
        [nn.Linear(stage_dim, 512) for stage_dim in (64, 128, 256, 512)]
    )
    aligner_total, aligner_trainable = _parameter_count(aligners)

    capacity_rows.sort(
        key=lambda row: (
            0 if row["architecture"] == "CNN" else 1,
            int(row["capacity"][1:]),
            row["dataset"],
        )
    )
    layer_rows.sort(
        key=lambda row: (
            0 if row["architecture"] == "CNN" else 1,
            int(row["capacity"][1:]),
            row["dataset"],
            row["layer"],
        )
    )
    assignment_rows.sort(
        key=lambda row: (
            0 if row["architecture"] == "CNN" else 1,
            int(row["capacity"][1:]),
        )
    )

    return {
        "source": {
            "branch": _git_value("branch", "--show-current"),
            "commit": _git_value("rev-parse", "HEAD"),
            "main_path": str(MAIN_PATH.relative_to(REPO_ROOT)),
        },
        "configuration": config,
        "capacity_rows": capacity_rows,
        "factorized_layers": layer_rows,
        "client_assignments": assignment_rows,
        "resnet_clip_local_aligners": {
            "stage_dimensions": [64, 128, 256, 512],
            "target_dimension": 512,
            "total_parameters": aligner_total,
            "trainable_parameters": aligner_trainable,
            "uploaded_or_aggregated": False,
        },
    }


def _report_text(result: Dict[str, Any]) -> str:
    lines = [
        "FedLAP model-capacity audit",
        "=" * 80,
        f"branch: {result['source']['branch']}",
        f"commit: {result['source']['commit']}",
        f"source: {result['source']['main_path']}",
        "",
        "Client capacity assignment",
        "-" * 80,
    ]
    for row in result["client_assignments"]:
        lines.append(
            f"{row['architecture']:9s} {row['capacity']} ratio={row['rank_ratio']:.2f} "
            f"main_index={row['main_model_index']} clients=[{row['client_ids']}]"
        )

    lines.extend(["", "Instantiated model parameter counts", "-" * 80])
    layer_lookup: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in result["factorized_layers"]:
        layer_lookup.setdefault(
            (row["architecture"], row["capacity"], row["dataset"]), []
        ).append(row)

    for row in result["capacity_rows"]:
        lines.append(
            f"{row['architecture']} {row['capacity']} {row['dataset']} "
            f"ratio={row['rank_ratio']:.2f}"
        )
        lines.append(
            f"  low-rank class={row['low_rank_model_class']} "
            f"total={row['low_rank_total_parameters']} "
            f"trainable={row['low_rank_trainable_parameters']}"
        )
        lines.append(
            f"  corresponding full-rank={row['full_rank_model']} "
            f"total={row['full_rank_total_parameters']} "
            f"trainable={row['full_rank_trainable_parameters']}"
        )
        for layer in layer_lookup[
            (row["architecture"], row["capacity"], row["dataset"])
        ]:
            factor_text = ", ".join(
                f"{name}={shape}" for name, shape in layer["factor_shapes"].items()
            )
            lines.append(
                f"    {layer['layer']}: {layer['module_class']} "
                f"rank={layer['actual_rank']}/{layer['maximum_unfolded_rank']} "
                f"factors[{factor_text}]"
            )

    aligners = result["resnet_clip_local_aligners"]
    lines.extend(
        [
            "",
            "Client-local ResNet CLIP aligners (excluded from model-capacity table)",
            "-" * 80,
            f"stage dimensions: {aligners['stage_dimensions']} -> "
            f"{aligners['target_dimension']}",
            f"total={aligners['total_parameters']} "
            f"trainable={aligners['trainable_parameters']} "
            f"uploaded_or_aggregated={aligners['uploaded_or_aggregated']}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instantiate and audit the FedLAP CNN/ResNet-18 capacity families."
    )
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper/model_capacity_audit",
    )
    args = parser.parse_args()
    if args.num_clients <= 0:
        raise ValueError("--num-clients must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = audit(args.num_clients)

    report = _report_text(result)
    print(report, end="")
    (output_dir / "audit_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    capacity_fields = list(result["capacity_rows"][0].keys())
    _write_csv(
        output_dir / "capacity_parameters.csv",
        result["capacity_rows"],
        capacity_fields,
    )
    assignment_fields = list(result["client_assignments"][0].keys())
    _write_csv(
        output_dir / "client_capacity_assignments.csv",
        result["client_assignments"],
        assignment_fields,
    )

    layer_rows = []
    for row in result["factorized_layers"]:
        flattened = dict(row)
        flattened["factor_shapes"] = json.dumps(flattened["factor_shapes"])
        layer_rows.append(flattened)
    layer_fields = list(layer_rows[0].keys())
    _write_csv(
        output_dir / "factorized_layer_ranks.csv", layer_rows, layer_fields
    )
    (output_dir / "table_a2_rows.tex").write_text(
        _make_capacity_tex(result["capacity_rows"]), encoding="utf-8"
    )

    print(f"Audit files written to: {output_dir}")


if __name__ == "__main__":
    main()
