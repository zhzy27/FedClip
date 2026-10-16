"""Upstream parser defaults + upstream YAML merge, with a separate entry point."""
import argparse
import math
from pathlib import Path

import yaml

from .vendor.arguments import source_parser

CONFIGS = Path(__file__).parent / "configs"


def merge(base, override):
    result = base.copy()
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def load_yaml(path):
    path = Path(path).resolve()
    base = path.parent.parent / "base.yaml"
    if not base.exists():
        base = path.parent / "base.yaml"
    override = yaml.safe_load(path.read_text(encoding="utf-8"))
    return merge(yaml.safe_load(base.read_text(encoding="utf-8")), override) if base.exists() and base != path else override


def yaml_arguments(config):
    """Same mappings as scripts/run_pfllib.py for this supported protocol."""
    ds, tr, alg = config.get("dataset", {}), config.get("training", {}), config.get("algorithm", {})
    model, exp, dev = config.get("model", {}), config.get("experiment", {}), config.get("device", {})
    name = ds.get("name", "Cifar10")
    dataset = name + "/" + ds["config"] if ds.get("config") else name
    values = dict(dataset=dataset, num_classes=ds.get("num_classes", 10), num_clients=ds.get("num_clients", 20),
                  model=model.get("name", "CNN"), algorithm=alg.get("name", "FedAvg"),
                  global_rounds=tr.get("global_rounds", 100), local_epochs=tr.get("local_epochs", 5),
                  batch_size=tr.get("batch_size", 10), local_learning_rate=tr.get("learning_rate", .005),
                  join_ratio=tr.get("join_ratio", 1.), device=dev.get("type", "cuda"), device_id=str(dev.get("id", "0")),
                  goal=exp.get("goal", "test"), times=exp.get("times", 1))
    supported = {"global_rounds", "local_epochs", "batch_size", "learning_rate", "join_ratio", "eval_gap",
                 "num_threads", "use_cache", "learning_rate_decay", "learning_rate_decay_gamma", "auto_break", "top_cnt"}
    if set(tr) - supported:
        raise ValueError(f"Unsupported YAML training fields: {sorted(set(tr) - supported)}")
    for key in supported - {"learning_rate"}:
        if key in tr:
            values[key] = tr[key]
    for key, value in alg.get("params", {}).items():
        if key not in {"num_server_models", "smooth_mu", "use_rep_mode"}:
            raise ValueError(f"Unsupported FedFewS algorithm field: {key}")
        values[key] = value
    return values


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default=str(CONFIGS / "noniid_dir_20_a0p5/algorithms/fedfews.yaml"))
    opts, _ = preliminary.parse_known_args(argv)
    config = load_yaml(opts.config)
    parser = source_parser()
    parser.description = "Isolated FedFewS standard migration; preview does not train or generate data."
    parser.set_defaults(**yaml_arguments(config))
    parser.add_argument("--config", default=opts.config)
    parser.add_argument("--data-root", required=True, help="Existing FedFew dataset root (contains Cifar100/ or cifar100/)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--rng-mode", choices=["original", "controlled"], default="original")
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, help="Explicit intra-op override; source num_threads does not set this")
    parser.add_argument("--expected-alpha", type=float, help="Require config.json alpha; never infer it from a directory")
    parser.add_argument("--expected-partition", choices=["dir", "pat"], help="Require the actual config.json partition")
    parser.add_argument("--source-root", help="Optional actual FedFew checkout for provenance (does not replace migrated code)")
    parser.add_argument("--trace-batches", action="store_true", help="Save actual sample-index batches (large in long runs)")
    args = parser.parse_args(argv)
    args.config = str(Path(args.config).resolve())
    args.data_root = str(Path(args.data_root).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    # The original runner preserves dataset string case; do not case-normalize.
    args.test_batch_size = 1024 if args.dataset in ["Mnist", "MNIST", "Cifar10", "Cifar100", "kvasir", "Kvasir", "Digit5"] else 256
    validate(args)
    return args, config


def validate(args):
    if args.algorithm != "FedFewS" or args.model != "CNN" or "cifar100" not in args.dataset.lower():
        raise ValueError("This entry supports only FedFewS / CNN / CIFAR-100")
    if args.num_classes != 100 or args.num_server_models < 1 or args.num_clients < 1:
        raise ValueError("Require 100 classes, positive client count and K")
    if args.use_rep_mode or args.learning_rate_decay or args.auto_break or args.random_join_ratio:
        raise ValueError("Rep, decay, early stopping and random participation are outside standard protocol")
    if args.join_ratio != 1.0 or any(getattr(args, key) != 0 for key in ["few_shot", "client_drop_rate", "train_slow_rate", "send_slow_rate", "num_new_clients"]):
        raise ValueError("Require full participation, no few-shot/drop/slow/new clients")
    if args.dlg_eval or args.time_select or args.num_threads != 1:
        raise ValueError("Require no DLG/time selection and sequential num_threads=1")
    if args.global_rounds < 1 or args.local_epochs < 1 or args.batch_size < 1 or args.eval_gap < 1:
        raise ValueError("Require >=2 inclusive rounds, positive epochs/batch/eval_gap")
    if not math.isfinite(args.smooth_mu) or args.smooth_mu <= 0 or not math.isfinite(args.local_learning_rate) or args.local_learning_rate <= 0:
        raise ValueError("smooth_mu and lr must be finite and positive")
    if args.prev < 0 or args.times <= args.prev:
        raise ValueError("Require 0 <= prev < times")
    if args.torch_threads is not None and args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    if args.model_dir is not None or args.result_dir is not None or args.save_folder_name != "items":
        raise ValueError("Use --output-dir for isolated models/metrics/logs; source path overrides are not supported here")
    if args.expected_alpha is not None and (not math.isfinite(args.expected_alpha) or args.expected_alpha <= 0):
        raise ValueError("expected-alpha must be finite and positive")
