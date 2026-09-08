#!/usr/bin/env python3
"""Audit implementation details and experiment hyperparameters without importing main.py.

The script separates three sources that must not be conflated in a paper:

1. argparse defaults in ``system/main.py``;
2. commands recorded in ``system/.vscode/launch.json``;
3. runtime behavior hard-coded in the client/server/model implementations.

It writes a machine-readable JSON file, a command CSV, and a Chinese text
report under ``paper/implementation_hyperparameter_audit`` by default.
The runtime environment section describes the machine executing this script;
it is not automatically treated as the machine that produced paper results.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "system" / "main.py"
LAUNCH_PATH = ROOT / "system" / ".vscode" / "launch.json"
ENV_PATH = ROOT / "env_cuda_latest.yaml"


@dataclass
class ArgumentSpec:
    dest: str
    aliases: list[str]
    default: Any
    type_name: str | None
    choices: list[Any] | None
    action: str | None


def literal_or_repr(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def extract_argument_specs(path: Path) -> tuple[dict[str, ArgumentSpec], dict[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    specs: dict[str, ArgumentSpec] = {}
    alias_to_dest: dict[str, str] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue

        aliases = [
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value.startswith("-")
        ]
        if not aliases:
            continue

        long_alias = next((item for item in aliases if item.startswith("--")), None)
        dest = (long_alias or aliases[0]).lstrip("-").replace("-", "_")
        keywords = {item.arg: item.value for item in node.keywords if item.arg is not None}
        type_node = keywords.get("type")
        type_name = None
        if isinstance(type_node, ast.Name):
            type_name = type_node.id
        elif type_node is not None:
            type_name = ast.unparse(type_node)

        choices = literal_or_repr(keywords.get("choices"))
        if choices is not None and not isinstance(choices, list):
            choices = list(choices) if isinstance(choices, tuple) else [choices]

        spec = ArgumentSpec(
            dest=dest,
            aliases=aliases,
            default=literal_or_repr(keywords.get("default")),
            type_name=type_name,
            choices=choices,
            action=literal_or_repr(keywords.get("action")),
        )
        specs[dest] = spec
        for alias in aliases:
            alias_to_dest[alias] = dest

    return specs, alias_to_dest


def convert_value(value: str, spec: ArgumentSpec) -> Any:
    if spec.type_name == "int":
        return int(value)
    if spec.type_name == "float":
        return float(value)
    if spec.type_name == "bool":
        return value.lower() in {"1", "true", "yes", "on"}
    return value


def parse_cli_tokens(
    tokens: list[str],
    specs: dict[str, ArgumentSpec],
    alias_to_dest: dict[str, str],
) -> tuple[dict[str, Any], list[str], list[str]]:
    values = {dest: spec.default for dest, spec in specs.items()}
    explicit: list[str] = []
    unknown: list[str] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        if token not in alias_to_dest:
            if token.startswith("-"):
                unknown.append(token)
            index += 1
            continue

        dest = alias_to_dest[token]
        spec = specs[dest]
        explicit.append(dest)
        if spec.action == "store_true":
            values[dest] = True
            index += 1
            continue

        if index + 1 >= len(tokens):
            unknown.append(f"{token}=<missing>")
            index += 1
            continue
        raw_value = tokens[index + 1]
        try:
            values[dest] = convert_value(raw_value, spec)
        except ValueError:
            values[dest] = raw_value
            unknown.append(f"{token}=<invalid:{raw_value}>")
        index += 2

    return values, sorted(set(explicit)), unknown


def command_tokens(raw_command: str) -> list[str]:
    command = raw_command.strip().rstrip("\\").strip()
    command = command.strip('"').strip("'")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = shlex.split(command.rstrip('"').rstrip("'"), posix=True)
    if len(tokens) >= 2 and Path(tokens[1]).name == "main.py":
        return tokens[2:]
    return tokens


def extract_launch_records(
    specs: dict[str, ArgumentSpec], alias_to_dest: dict[str, str]
) -> list[dict[str, Any]]:
    text = LAUNCH_PATH.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []

    active_json_text = text.split("/*", 1)[0].strip()
    try:
        active_json = json.loads(active_json_text)
    except json.JSONDecodeError as exc:
        active_json = {"configurations": []}
        records.append({"source": "active_json_error", "error": str(exc)})

    for idx, config in enumerate(active_json.get("configurations", [])):
        tokens = [str(item) for item in config.get("args", [])]
        values, explicit, unknown = parse_cli_tokens(tokens, specs, alias_to_dest)
        records.append(
            {
                "source": "active_debug_configuration",
                "source_line": None,
                "configuration_name": config.get("name", f"configuration_{idx}"),
                "values": values,
                "explicit": explicit,
                "unknown": unknown,
                "raw_command": "python main.py " + " ".join(tokens),
            }
        )

    for line_no, line in enumerate(text.splitlines(), start=1):
        marker = "python main.py"
        if marker not in line:
            continue
        raw = line[line.index(marker) :].strip()
        try:
            tokens = command_tokens(raw)
        except ValueError as exc:
            records.append(
                {
                    "source": "commented_command_parse_error",
                    "source_line": line_no,
                    "error": str(exc),
                    "raw_command": raw,
                }
            )
            continue
        values, explicit, unknown = parse_cli_tokens(tokens, specs, alias_to_dest)
        records.append(
            {
                "source": "recorded_command",
                "source_line": line_no,
                "values": values,
                "explicit": explicit,
                "unknown": unknown,
                "raw_command": raw.rstrip("\\").strip().strip('"'),
            }
        )
    return records


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_environment_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    python_match = re.search(r"^\s*-\s*python=([^\s]+)", text, flags=re.MULTILINE)
    torch_match = re.search(r"^\s*-\s*torch==([^\s]+)", text, flags=re.MULTILINE)
    torchvision_match = re.search(r"^\s*-\s*torchvision(?:==([^\s]+))?", text, flags=re.MULTILINE)
    return {
        "environment_file": str(path.relative_to(ROOT)) if path.exists() else None,
        "declared_python": python_match.group(1) if python_match else None,
        "declared_torch": torch_match.group(1) if torch_match else None,
        "declared_torchvision": (
            torchvision_match.group(1) if torchvision_match and torchvision_match.group(1) else "unpinned"
        ) if torchvision_match else None,
        "declared_cuda": None,
        "declared_cudnn": None,
    }


def current_runtime_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "warning": "This is the machine running the audit, not necessarily the paper experiment server.",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "memory_gib": None,
        "torch": None,
        "torchvision": None,
        "cuda_runtime": None,
        "cudnn": None,
        "gpus": [],
    }
    try:
        import psutil  # type: ignore

        result["memory_gib"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except (ImportError, AttributeError):
        pass

    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_runtime"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            result["gpus"] = [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    try:
        import torchvision

        result["torchvision"] = torchvision.__version__
    except (ImportError, RuntimeError):
        pass
    return result


def active_set_seed_calls(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "set_seed":
            continue
        ancestor = parents.get(node)
        inside_definition = False
        while ancestor is not None:
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inside_definition = True
                break
            ancestor = parents.get(ancestor)
        if not inside_definition:
            count += 1
    return count


def source_audit() -> dict[str, Any]:
    paths = {
        "main": MAIN_PATH,
        "client_base": ROOT / "system/flcore/clients/clientbase.py",
        "client_clip": ROOT / "system/flcore/clients/clientCLIP.py",
        "server_base": ROOT / "system/flcore/servers/serverbase.py",
        "server_clip": ROOT / "system/flcore/servers/serverCLIP.py",
        "cnn": ROOT / "system/flcore/trainmodel/models.py",
        "resnet": ROOT / "system/flcore/trainmodel/SVD_resnet.py",
        "clip": ROOT / "system/utils/get_clip_text_encoder.py",
    }
    source = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}

    checks = {
        "fedlap_optimizer_is_sgd": "torch.optim.SGD" in source["client_clip"],
        "fedlap_optimizer_has_no_momentum_keyword": "momentum=" not in source["client_clip"],
        "fedlap_optimizer_has_no_weight_decay_keyword": "weight_decay=" not in source["client_clip"],
        "fedlap_has_no_lr_scheduler": "lr_scheduler" not in source["client_clip"],
        "fedlap_gradient_clip_10": "clip_grad_norm_(clip_params, 10.0)" in source["client_clip"],
        "dataloader_uses_default_zero_workers": "num_workers" not in source["client_base"],
        "train_loader_shuffles": "drop_last=False, shuffle=True" in source["client_base"],
        "test_loader_does_not_shuffle": "drop_last=False, shuffle=False" in source["client_base"],
        "torch_seed_zero_is_active": "torch.manual_seed(0)" in source["main"],
        "full_set_seed_is_called": active_set_seed_calls(MAIN_PATH) > 0,
        "clip_backbone_vit_b32": 'model_name="ViT-B/32"' in source["client_clip"],
        "clip_prompt_a_photo_of": 'prompt_template="a photo of {}"' in source["client_clip"],
        "clip_anchor_cache": "text_features_cache" in source["clip"],
        "clip_text_inference_no_grad": "with torch.no_grad():" in source["clip"],
        "resnet_four_depth_anchors": "num_depths=4" in source["client_clip"],
        "resnet_four_linear_aligners": "torch.nn.Linear(stage_dim, target_dim)" in source["client_clip"],
        "resnet_alignment_losses_are_averaged": "return sum(losses) / len(losses)" in source["client_clip"],
        "personal_ratio_max_point_seven": "depth_ratio = 0.7 *" in source["server_clip"],
        "personal_weights_use_softmax": "softmax(logits_tensor, dim=0)" in source["server_clip"],
        "personal_logits_are_cosine_over_tau": "logit_j = cos_sim / tau" in source["server_clip"],
        "sample_weights_are_train_sample_weighted": "self.uploaded_weights[i] = w / tot_samples" in source["server_clip"],
        "test_accuracy_is_sample_weighted": "sum(stats[2])*1.0 / sum(stats[1])" in source["server_base"],
        "evaluation_visits_all_clients": "for c in self.clients:" in source["server_base"],
        "fedclip_skips_round_zero_evaluation": "if i > 0 and i % self.eval_gap == 0" in source["server_clip"],
        "fedclip_loop_is_global_rounds_plus_one": "range(self.global_rounds+1)" in source["server_clip"],
        "cnn_minimum_rank_one": "self.rank = max(1, round(rank_rate * self.max_rank))" in source["cnn"],
        "resnet_minimum_rank_one": "self.rank = max(1, round(rank_rate * self.max_rank))" in source["resnet"],
        "cnn_has_no_rank_dropout": "rank_dropout" not in source["cnn"],
        "resnet_has_no_rank_dropout": "rank_dropout" not in source["resnet"],
    }

    failed = [name for name, passed in checks.items() if not passed and name != "full_set_seed_is_called"]
    return {
        "checks": checks,
        "failed_required_checks": failed,
        "derived": {
            "client_optimizer": "SGD",
            "momentum": 0.0,
            "weight_decay": 0.0,
            "learning_rate_scheduler": "none",
            "fedlap_gradient_clip_norm": 10.0,
            "dataloader_num_workers": 0,
            "maximum_personalization_ratio": 0.7,
            "clip_backbone": "ViT-B/32",
            "prompt_template": "a photo of {}",
            "anchor_dimension": 512,
            "resnet_alignment_stages": 4,
            "resnet_aligner_dimensions": ["64->512", "128->512", "256->512", "512->512"],
            "aligners_uploaded_or_aggregated": False,
            "minimum_retained_rank": 1,
            "accuracy_aggregation": "sum(correct_i) / sum(test_samples_i)",
        },
    }


def explicit_or_default(record: dict[str, Any], key: str) -> str:
    values = record.get("values", {})
    suffix = "explicit" if key in record.get("explicit", []) else "default"
    return f"{values.get(key)} [{suffix}]"


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in records if "values" in item]
    commands = [item for item in valid if item["source"] == "recorded_command"]
    fedlap = [item for item in commands if item["values"].get("algorithm") == "FedCLIP"]
    keys = [
        "dataset",
        "model_family",
        "join_ratio",
        "global_rounds",
        "local_epochs",
        "batch_size",
        "num_clients",
        "local_learning_rate",
        "mse_lamda",
        "regular_lamda",
        "aggregate_tau",
        "use_asymmetric_lr",
        "u_lr_ratio",
    ]
    fedlap_counts = {
        key: dict(Counter(explicit_or_default(record, key) for record in fedlap))
        for key in keys
    }

    participation = []
    for record in fedlap:
        ratio = record["values"].get("join_ratio")
        try:
            is_special = float(ratio) != 1.0
        except (TypeError, ValueError):
            is_special = False
        if is_special:
            participation.append(
                {
                    "source_line": record.get("source_line"),
                    "join_ratio": ratio,
                    "global_rounds": record["values"].get("global_rounds"),
                    "dataset": record["values"].get("dataset"),
                }
            )

    resnet = [
        {
            "source_line": item.get("source_line"),
            "dataset": item["values"].get("dataset"),
            "regular_lamda": explicit_or_default(item, "regular_lamda"),
            "aggregate_tau": explicit_or_default(item, "aggregate_tau"),
            "use_asymmetric_lr": explicit_or_default(item, "use_asymmetric_lr"),
            "raw_command": item.get("raw_command"),
        }
        for item in fedlap
        if "resnet" in str(item["values"].get("model_family", "")).lower()
    ]

    invalid_choices = []
    for record in valid:
        for dest in record.get("explicit", []):
            spec = ARG_SPECS.get(dest)
            if spec is None or not spec.choices:
                continue
            value = record["values"].get(dest)
            if value not in spec.choices:
                invalid_choices.append(
                    {
                        "source": record.get("source"),
                        "source_line": record.get("source_line"),
                        "argument": dest,
                        "value": value,
                        "choices": spec.choices,
                    }
                )

    baseline_fields = {
        "FedGH": ["server_learning_rate", "server_epochs"],
        "LG-FedAvg": [],
        "FedTGP": ["lamda", "server_epochs", "margin_threthold", "feature_dim"],
        "FedProto": ["lamda"],
        "FML": ["alpha", "beta"],
        "FD": ["lamda"],
        "FedKD": ["mentee_learning_rate", "T_start", "T_end", "feature_dim"],
        "FedGen": [
            "noise_dim",
            "generator_learning_rate",
            "hidden_dim",
            "server_epochs",
            "feature_dim",
        ],
        "FedMRL": ["feature_dim", "sub_feature_dim"],
        "PFedAFM": ["alpha_lr"],
        "FedSPU": [],
        "FedRE": ["server_learning_rate", "server_epochs", "re_samples", "head_batch_size"],
    }
    standard_commands = [
        item
        for item in commands
        if item["values"].get("local_learning_rate") == 0.005
        and item["values"].get("batch_size") == 16
        and item["values"].get("local_epochs") == 5
        and item["values"].get("num_clients") == 20
        and item["values"].get("global_rounds") == 100
    ]
    baseline_summary: dict[str, Any] = {}
    for method, fields in baseline_fields.items():
        method_records = [
            item for item in standard_commands if item["values"].get("algorithm") == method
        ]
        baseline_summary[method] = {
            "command_count": len(method_records),
            "parameters": {
                field: dict(Counter(explicit_or_default(item, field) for item in method_records))
                for field in fields
            },
        }

    return {
        "record_count": len(valid),
        "recorded_command_count": len(commands),
        "fedlap_command_count": len(fedlap),
        "fedlap_value_counts": fedlap_counts,
        "fedlap_resnet_records": resnet,
        "participation_records": participation,
        "invalid_choice_records": invalid_choices,
        "baseline_standard_command_summary": baseline_summary,
        "commands_with_unknown_arguments": [
            {
                "source": item.get("source"),
                "source_line": item.get("source_line"),
                "unknown": item.get("unknown"),
            }
            for item in valid
            if item.get("unknown")
        ],
    }


def build_conflicts(
    specs: dict[str, ArgumentSpec], summary: dict[str, Any], source_info: dict[str, Any]
) -> list[dict[str, str]]:
    return [
        {
            "item": "框架默认值与论文主实验命令",
            "finding": (
                f"解析器默认 global_rounds={specs['global_rounds'].default}, "
                f"local_epochs={specs['local_epochs'].default}, batch_size={specs['batch_size'].default}, "
                f"num_clients={specs['num_clients'].default}, lr={specs['local_learning_rate'].default}；"
                "launch 中的主实验记录显式覆盖为 100/5/16/20/0.005。"
            ),
            "action": "Table A3 应报告主实验命令值，而不是 argparse 默认值。",
        },
        {
            "item": "随机种子",
            "finding": (
                "main.py 和 Client 构造函数激活了 torch.manual_seed(0)，但完整 set_seed(0) 调用被注释；"
                "客户端抽样使用 NumPy，上传抽样使用 Python random，因此当前代码不是完整的 seed=0 确定性运行。"
            ),
            "action": "论文只能写 PyTorch seed 为 0；若需声明全局 seed=0，应先恢复并记录完整 set_seed 调用后重跑。",
        },
        {
            "item": "FedLAP 与 FedLAP* 学习率命名",
            "finding": (
                f"use_asymmetric_lr 的代码默认值为 {specs['use_asymmetric_lr'].default}，"
                "当前记录的 FedCLIP 命令大多未显式传入该开关，因此实际落入非对称学习率。"
            ),
            "action": "matched learning rates 必须显式使用 --use_asymmetric_lr 0；asymmetric learning rates 使用 1。",
        },
        {
            "item": "ResNet-18 正则系数与聚合温度",
            "finding": (
                "launch 记录同时出现 regular_lamda=1e-3/tau=1 与 regular_lamda=1e-4/tau=5；"
                "仅靠当前 launch 草稿无法判断最终表格使用哪一组。"
            ),
            "action": "从最终结果 JSON/manifest 的 args 字段逐项核对后再定稿。",
        },
        {
            "item": "参与率实验命令",
            "finding": (
                f"当前可解析的 FedCLIP launch 记录中，非 1.0 参与率命令数为 "
                f"{len(summary['participation_records'])}；0.2/0.4/0.8 与 500 轮不是 argparse 默认。"
            ),
            "action": "可按论文实验协议写入正文，但应由对应结果 JSON 再确认。",
        },
        {
            "item": "通信轮次与评估时点",
            "finding": (
                "FedCLIP 使用 range(global_rounds+1)，在 i>0 时先评估再训练；global_rounds=100 时，"
                "报告的是完成 1--100 次更新后的 100 个评估点，但循环还会执行一次未评估的第 101 次更新。"
            ),
            "action": "论文可将报告曲线描述为 100 个已评估通信轮次；最终导出的模型则比最后评估状态多一次更新。",
        },
        {
            "item": "运行环境",
            "finding": (
                "env_cuda_latest.yaml 只固定 Python 3.11 和 torch 2.0.1；GPU/CPU/RAM、Linux 发行版、"
                "CUDA 与 cuDNN 版本未写入仓库元数据。"
            ),
            "action": "必须在生成论文结果的主服务器上运行本脚本，并人工确认该输出对应最终实验服务器。",
        },
        {
            "item": "基线来源与调参协议",
            "finding": (
                "README 证明仓库为 HtFLlib/PFLlib-compatible，代码中只有部分方法标注官方来源；"
                "仓库没有验证集或统一超参数搜索日志。"
            ),
            "action": "不要声称所有基线均为官方代码或经统一验证集公平调参；应报告实际命令值。",
        },
        {
            "item": "结果提升的红色标注",
            "finding": "代码只保存准确率，不生成论文表格中的红色提升值，也未记录其参照方法。",
            "action": "需在论文表注中人工明确红色提升相对于同一设置下的最佳基线或指定基线。",
        },
        {
            "item": "launch 文件含有失效或跨分支命令",
            "finding": (
                f"当前 launch 记录中有 {len(summary['invalid_choice_records'])} 条参数取值不属于 main.py choices，"
                f"另有 {len(summary['commands_with_unknown_arguments'])} 条命令包含当前解析器不认识的参数；"
                "例如旧的 staged 模式或其他聚合分支参数。"
            ),
            "action": "launch.json 只能作为历史记录，不能整体视为可执行的最终实验清单。",
        },
    ]


def write_command_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = [
        "source",
        "source_line",
        "algorithm",
        "dataset",
        "model_family",
        "join_ratio",
        "global_rounds",
        "local_epochs",
        "batch_size",
        "num_clients",
        "local_learning_rate",
        "mse_lamda",
        "regular_lamda",
        "aggregate_tau",
        "use_asymmetric_lr",
        "u_lr_ratio",
        "explicit_arguments",
        "unknown_arguments",
        "raw_command",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            if "values" not in record:
                continue
            values = record["values"]
            row = {key: values.get(key, "") for key in columns if key in values}
            row.update(
                {
                    "source": record.get("source"),
                    "source_line": record.get("source_line"),
                    "explicit_arguments": ";".join(record.get("explicit", [])),
                    "unknown_arguments": ";".join(record.get("unknown", [])),
                    "raw_command": record.get("raw_command", ""),
                }
            )
            writer.writerow(row)


def format_counter(counter: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counter.items()) or "none"


def write_text_report(path: Path, audit: dict[str, Any]) -> None:
    specs = audit["argument_specs"]
    summary = audit["launch_summary"]
    source = audit["source_audit"]
    declared = audit["declared_environment"]
    runtime = audit["audit_runtime_environment"]

    lines = [
        "Implementation Details and Hyperparameters 代码审计",
        "=" * 72,
        f"Git branch: {audit['git']['branch']}",
        f"Git commit: {audit['git']['commit']}",
        "",
        "一、仓库声明环境",
        f"- environment file: {declared['environment_file']}",
        f"- Python: {declared['declared_python']}",
        f"- PyTorch: {declared['declared_torch']}",
        f"- torchvision: {declared['declared_torchvision']}",
        "- CUDA/cuDNN: 未固定",
        "",
        "二、执行审计脚本的当前机器（不自动等同论文实验服务器）",
        f"- platform: {runtime['platform']}",
        f"- Python: {runtime['python']}",
        f"- CPU: {runtime['cpu']}",
        f"- logical CPUs: {runtime['logical_cpu_count']}",
        f"- memory GiB: {runtime['memory_gib']}",
        f"- PyTorch/torchvision: {runtime['torch']} / {runtime['torchvision']}",
        f"- CUDA/cuDNN: {runtime['cuda_runtime']} / {runtime['cudnn']}",
        f"- GPUs: {runtime['gpus']}",
        "",
        "三、关键 argparse 默认值（不是论文主实验值）",
    ]
    for key in [
        "num_clients",
        "join_ratio",
        "global_rounds",
        "local_epochs",
        "batch_size",
        "local_learning_rate",
        "times",
        "eval_gap",
        "auto_break",
        "mse_lamda",
        "regular_lamda",
        "aggregate_tau",
        "u_lr_ratio",
        "use_asymmetric_lr",
    ]:
        lines.append(f"- {key}: {specs[key]['default']}")

    lines.extend(["", "四、launch 中 FedCLIP 命令摘要"])
    for key, counter in summary["fedlap_value_counts"].items():
        lines.append(f"- {key}: {format_counter(counter)}")
    lines.append(f"- ResNet command records: {len(summary['fedlap_resnet_records'])}")
    for item in summary["fedlap_resnet_records"]:
        lines.append(
            "  line {source_line}: dataset={dataset}, regular={regular_lamda}, "
            "tau={aggregate_tau}, asym={use_asymmetric_lr}".format(**item)
        )

    lines.extend(["", "五、标准主实验记录中的基线专属参数"])
    for method, item in summary["baseline_standard_command_summary"].items():
        params = []
        for name, values in item["parameters"].items():
            params.append(f"{name}=({format_counter(values)})")
        suffix = "; ".join(params) if params else "无额外记录参数"
        lines.append(f"- {method}: commands={item['command_count']}; {suffix}")

    lines.extend(["", "六、运行时源码检查"])
    for name, passed in source["checks"].items():
        lines.append(f"- {'PASS' if passed else 'INFO/FAIL'}: {name}")

    lines.extend(["", "七、必须人工处理的冲突"])
    for index, item in enumerate(audit["conflicts"], start=1):
        lines.append(f"{index}. {item['item']}")
        lines.append(f"   发现: {item['finding']}")
        lines.append(f"   建议: {item['action']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "paper" / "implementation_hyperparameter_audit",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global ARG_SPECS
    ARG_SPECS, alias_to_dest = extract_argument_specs(MAIN_PATH)
    records = extract_launch_records(ARG_SPECS, alias_to_dest)
    summary = summarize_records(records)
    source_info = source_audit()
    conflicts = build_conflicts(ARG_SPECS, summary, source_info)

    audit = {
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
        },
        "argument_specs": {key: asdict(value) for key, value in ARG_SPECS.items()},
        "launch_summary": summary,
        "source_audit": source_info,
        "declared_environment": parse_environment_file(ENV_PATH),
        "audit_runtime_environment": current_runtime_environment(),
        "conflicts": conflicts,
    }

    json_path = output_dir / "audit.json"
    report_path = output_dir / "audit_report.txt"
    csv_path = output_dir / "launch_commands.csv"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_report(report_path, audit)
    write_command_csv(csv_path, records)

    print(report_path.read_text(encoding="utf-8"))
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    if source_info["failed_required_checks"]:
        print("Required source checks failed:", source_info["failed_required_checks"], file=sys.stderr)
        return 1
    return 0


ARG_SPECS: dict[str, ArgumentSpec] = {}


if __name__ == "__main__":
    raise SystemExit(main())
