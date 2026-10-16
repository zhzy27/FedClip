"""Read-only data audit; migrated training and audit share explicit-file decoding."""
import json
import os
from pathlib import Path

import torch

from .metadata import sha256, tensor_digest
from .vendor import data_utils


def selected_file(root, dataset, client, train):
    path = data_utils.client_shard_path(dataset, client, train, data_root=root)
    return path, path.suffix == ".pkl" and path.with_suffix(".npz").exists()


def audit_data(args, reader=None):
    os.environ["PFLLIB_DATA_DIR"] = args.data_root
    config_path = Path(args.data_root) / args.dataset / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Need original data config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "num_clients" in config and config["num_clients"] != args.num_clients:
        raise ValueError("Dataset config num_clients differs from run")
    actual_alpha = config.get("alpha")
    if args.expected_alpha is not None and actual_alpha != args.expected_alpha:
        raise ValueError(f"config.json actual alpha={actual_alpha!r}, expected {args.expected_alpha}; directory name is not evidence")
    if args.expected_partition is not None and config.get("partition") != args.expected_partition:
        raise ValueError(f"config.json partition={config.get('partition')!r}, expected {args.expected_partition}")
    records = []
    for i in range(args.num_clients):
        for train in [True, False]:
            path, shadow = selected_file(args.data_root, args.dataset, i, train)
            if reader is None:
                decoded = data_utils.read_client_data_from_file(path, args.dataset, is_train=train, few_shot=0)
            else:
                # External reference MUST retain its own reader/decoder.
                decoded = reader(args.dataset, i, is_train=train, few_shot=0)
            if not decoded:
                raise ValueError(f"Empty shard: {path}")
            x, y = torch.stack([p[0] for p in decoded]), torch.stack([p[1] for p in decoded])
            if x.shape[1:] != (3, 32, 32) or x.dtype != torch.float32 or y.dtype != torch.int64:
                raise ValueError(f"Wrong decoded data shape/dtype: {path}: {x.shape}/{x.dtype}/{y.dtype}")
            if y.min() < 0 or y.max() >= args.num_classes or not torch.isfinite(x).all():
                raise ValueError(f"Invalid labels/nonfinite images: {path}")
            labels, counts = y.unique(return_counts=True)
            records.append(dict(client_id=i, split="train" if train else "test", path=str(path.resolve()),
                                format=path.suffix, sha256=sha256(path), npz_shadowed_by_pkl=shadow,
                                count=len(decoded), labels=dict(zip(labels.tolist(), counts.tolist())),
                                x_shape=list(x.shape), x_dtype=str(x.dtype), y_dtype=str(y.dtype),
                                decoded_x_sha256=tensor_digest(x), decoded_y_sha256=tensor_digest(y)))
    return dict(root=args.data_root, dataset=args.dataset, config_path=str(config_path),
                config_sha256=sha256(config_path), config=config, actual_alpha=actual_alpha,
                reader_rule="pkl preferred; npz data.tolist fallback; float32 x / int64 y; no transforms",
                format_policy=list(data_utils.SHARD_FORMATS), image_list_mode=data_utils.IMAGE_LIST_MODE, files=records)
