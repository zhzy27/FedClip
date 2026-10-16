"""Explicit shard formats and image decoding, without root discovery or caching."""
from pathlib import Path
import pickle

import numpy as np
import torch

NPZ_ONLY = (".npz",)
PKL_FIRST = (".pkl", ".npz")
TENSOR_LIST = "tensor_constructor"
NUMPY_STACK_LIST = "numpy_stack"


def select_client_shard(directory, client_id, *, formats):
    """Check only the supplied directory/ordered formats; never search elsewhere."""
    formats = tuple(formats)
    if not formats or len(set(formats)) != len(formats) or any(f not in {".pkl", ".npz"} for f in formats):
        raise ValueError(f"Invalid explicit shard format policy: {formats}")
    directory = Path(directory)
    for suffix in formats:
        path = directory / f"{client_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No client {client_id} shard in {directory}; allowed formats in order: {formats}")


def read_shard(path):
    """Read exactly this file. The caller, not this function, selects its format."""
    path = Path(path)
    if path.suffix not in {".npz", ".pkl"}:
        raise ValueError(f"Unsupported shard suffix: {path.suffix}")
    with path.open("rb") as stream:
        if path.suffix == ".pkl":
            return pickle.load(stream)
        return np.load(stream, allow_pickle=True)["data"].tolist()


def decode_image_data(data, *, list_mode):
    """Return the legacy list[(float32 image, int64 label)] in original order.

    FedClip uses Tensor(list); FedFew uses stack(from_numpy) for list images.
    Keeping this choice explicit preserves both input contracts/rounding. Label
    construction deliberately retains Tensor(...).type(int64), like both readers.
    This function neither transforms images nor caches or shuffles any samples.
    """
    if list_mode not in {TENSOR_LIST, NUMPY_STACK_LIST}:
        raise ValueError(f"Unknown list image decoding policy: {list_mode}")
    if isinstance(data["x"], list) and list_mode == NUMPY_STACK_LIST:
        x = torch.stack([torch.from_numpy(image) for image in data["x"]]).type(torch.float32)
    else:
        x = torch.Tensor(data["x"]).type(torch.float32)
    y = torch.Tensor(data["y"]).type(torch.int64)
    return [(image, label) for image, label in zip(x, y)]
