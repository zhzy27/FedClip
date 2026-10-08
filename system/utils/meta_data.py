"""Immutable stratified C0 holdout manifests, independent of training RNG/seed."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def data_fingerprint(data):
    digest = hashlib.sha256()
    for x, y in data:
        if not isinstance(x, torch.Tensor):
            raise ValueError("The C0 meta diagnostic currently supports image tensor data.")
        for value in (x, torch.as_tensor(y)):
            value = value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def create_split(data, fraction=.2, split_seed=1729, dataset="Cifar100", subdir="pat_20"):
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be in (0, 1).")
    labels = np.array([int(y) for _, y in data])
    rng = np.random.default_rng(split_seed)
    train, validation = [], []
    counts = {}
    for label in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two examples for an unseen holdout.")
        indices = rng.permutation(indices)
        size = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:size].tolist())
        train.extend(indices[size:].tolist())
        counts[str(label)] = dict(total=len(indices), train=len(indices) - size, validation=size)
    result = dict(schema=1, dataset=dataset, dataset_subdir=subdir, client_id=0,
        split_seed=int(split_seed), validation_fraction=fraction, sample_count=len(data),
        data_fingerprint=data_fingerprint(data), class_counts=counts,
        train_indices=sorted(train), validation_indices=sorted(validation))
    result["split_id"] = canonical_hash(result)
    return result


def validate_split(record, data, dataset=None, subdir=None):
    core = {k: v for k, v in record.items() if k != "split_id"}
    if record.get("schema") != 1 or record.get("client_id") != 0 or canonical_hash(core) != record.get("split_id"):
        raise ValueError("Invalid C0 split manifest/hash.")
    if dataset is not None and record["dataset"] != dataset or subdir is not None and record["dataset_subdir"] != subdir:
        raise ValueError("C0 split dataset/partition mismatch.")
    if record["sample_count"] != len(data) or record["data_fingerprint"] != data_fingerprint(data):
        raise ValueError("C0 training data fingerprint changed.")
    train, val = record["train_indices"], record["validation_indices"]
    if not train or not val or sorted(train + val) != list(range(len(data))) or set(train) & set(val):
        raise ValueError("C0 train/validation indices must be a disjoint complete partition.")
    return record


def save_split(path, record):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Refusing to overwrite a different C0 split.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def load_split(path, data, dataset=None, subdir=None):
    return validate_split(json.loads(Path(path).read_text(encoding="utf-8")), data, dataset, subdir)


def read_c0_training(dataset_root, dataset="Cifar100", subdir="pat_20"):
    path = Path(dataset_root) / dataset / subdir / "train" / "0.npz"
    with np.load(path, allow_pickle=True) as record:
        raw = record["data"].tolist()
    # Identical image processing to utils.data_utils.process_image.
    x, y = torch.tensor(raw["x"], dtype=torch.float32), torch.tensor(raw["y"], dtype=torch.int64)
    return list(zip(x, y))
