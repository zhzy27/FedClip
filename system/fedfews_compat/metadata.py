import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from .source_manifest import COMMIT, FILES, URL


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(tensor):
    a = tensor.detach().cpu().contiguous().numpy()
    h = hashlib.sha256(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def git_info(root):
    root = Path(root).resolve()
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.stdout.strip() if p.returncode == 0 else None
    return dict(path=str(root), commit=git("rev-parse", "HEAD"), status=git("status", "--short"), branch=git("branch", "--show-current"))


def source_info(root=None):
    result = dict(url=URL, vendored_commit=COMMIT, adaptations=FILES)
    if root:
        result["reference_checkout"] = git_info(root)
        result["actual_files"] = {rel: sha256(Path(root) / rel) for rel in FILES if (Path(root) / rel).is_file()}
        result["matches_recorded_files"] = all(result["actual_files"].get(p) == v["sha256"] for p, v in FILES.items())
    return result


def migration_info():
    directory = Path(__file__).parent
    paths = [p for p in directory.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".md"}]
    paths += [directory.parent / "run_fedfews.py", directory.parent / "verify_fedfews.py"]
    return dict(fedclip=git_info(directory.parents[1]),
                files={str(p.relative_to(directory.parent)): sha256(p) for p in sorted(paths)})


def environment():
    packages = {}
    for name in ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "h5py", "PyYAML"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages,
                cuda_build=torch.version.cuda, cuda_available=torch.cuda.is_available(),
                cuda_devices=([dict(index=i, name=torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
                              if torch.cuda.is_initialized() else "not initialized"),
                cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"))
