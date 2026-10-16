"""Read-only metadata and RNG snapshots; no training or seed policy."""
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    result = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    result.update(array.tobytes())
    return result.hexdigest()


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def environment():
    versions = {}
    for package in ['torch', 'numpy', 'scipy', 'h5py', 'torchvision']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'MISSING'
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=versions,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)


def code_version(root):
    def git(*args):
        result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, encoding='utf-8', errors='replace')
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(commit=git('rev-parse', 'HEAD'), status=git('status', '--short'))
