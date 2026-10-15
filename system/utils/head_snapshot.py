"""Immutable C0 post-local snapshots and original-data/public-head fingerprints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from flcore.clients.clientbase import load_item
from utils.data_utils import get_dataset_sub_dir, process_image
from utils.head_diagnostic_options import diagnostic_rounds
from utils.target_head import classifier_names, preserved_head_rng


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(named):
    digest = hashlib.sha256()
    for name, value in named.items():
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_image_shard(path):
    with np.load(path, allow_pickle=True) as record:
        return process_image(record["data"].tolist())


def sample_fingerprint(data):
    digest = hashlib.sha256()
    for x, y in data:
        digest.update(tensor_fingerprint(dict(x=x, y=y)).encode())
    return digest.hexdigest()


def shard_record(path):
    path = Path(path).resolve()
    data = read_image_shard(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                file_sha256=file_hash(path), sample_count=len(data), sample_sha256=sample_fingerprint(data))


def backend_settings():
    return dict(cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled())


def apply_backend(settings):
    torch.backends.cudnn.benchmark = settings["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = settings["cudnn_deterministic"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul_allow_tf32"]
    torch.use_deterministic_algorithms(settings["deterministic_algorithms"], warn_only=settings["deterministic_warn_only"])


class HeadDiagnosticCollector:
    def __init__(self, server, initial_server):
        self.rounds = diagnostic_rounds(server.args)
        self.server = server
        self.captured = []
        with preserved_head_rng():
            names = classifier_names(initial_server)
            self.initial_head = {n: dict(initial_server.named_parameters())[n].detach().cpu().clone() for n in names}
            subdir = get_dataset_sub_dir(server.args)
            data_root = Path("../dataset") / server.dataset / subdir
            self.data = {split: shard_record(data_root / split / "0.npz") for split in ("train", "test")}
            if self.data["train"]["sample_count"] != server.clients[0].train_samples:
                raise ValueError("Diagnostic training data must be the complete original C0 shard.")
            root = Path(getattr(server.args, "head_diag_dir", "") or "head_diag_snapshots").resolve()
            # Never archive model tensors inside the downloadable logs/checkpoint trees.
            for other in (Path(server.save_folder_name).resolve(), Path(server.report_dir()).resolve()):
                if root == other or root in other.parents or other in root.parents:
                    raise ValueError("head_diag_dir must be separate from checkpoints and logs.")
            self.folder = root / f"{server.target_proj_mode}_{getattr(server.args, 'target_head_mode', 'shared')}_seed{server.seed}" / server.run_id
            self.folder.mkdir(parents=True, exist_ok=False)
            self.public_head_sha256 = tensor_fingerprint(self.initial_head)
            self._write_index()
        print(f"[HeadDiag] collect completed rounds={self.rounds}; snapshots={self.folder}; offline head training only")

    def _write_index(self):
        record = dict(schema=1, stage="C0_post_local_before_aggregation", requested_rounds=self.rounds,
                      public_head_sha256=self.public_head_sha256, data=self.data, snapshots=self.captured)
        (self.folder / "index.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")

    def capture(self, correct, samples):
        server = self.server
        completed = server.cur_ground + 1
        if completed not in self.rounds:
            return
        if server._post_local_round != server.cur_ground or samples <= 0 or correct / samples != server._post_local_acc:
            raise RuntimeError("Head snapshot must match this round's ordinary post-local C0 evaluation.")
        with preserved_head_rng():
            client = server.clients[0]
            model = load_item(client.role, "model", client.save_folder_name).cpu()
            classifier_names(model)
            source = Path(client.save_folder_name) / f"{client.role}_model.pt"
            payload = dict(schema=1, stage="C0_post_local_before_aggregation", model=model,
                initial_public_head={n: p.clone() for n, p in self.initial_head.items()},
                metadata=dict(mode=server.target_proj_mode, head_mode=getattr(server.args, "target_head_mode", "shared"),
                    round=completed, loop_round=server.cur_ground, seed=server.seed, client_id=0, capacity=model.ratio_LR,
                    native_correct=int(correct), native_test_samples=int(samples), native_test_accuracy=correct / samples,
                    public_head_sha256=self.public_head_sha256, data=self.data,
                    dataset=server.dataset, dataset_subdir=get_dataset_sub_dir(server.args),
                    source_checkpoint=str(source.resolve()), source_checkpoint_sha256=file_hash(source),
                    model_state_sha256=tensor_fingerprint(model.state_dict()), model_class=type(model).__name__,
                    model_structure=repr(model), args=server._to_json_serializable(vars(server.args)),
                    ordinary_training=client.last_normal_train_stats, backend=backend_settings(),
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    parameter_spec={n: dict(shape=list(p.shape), dtype=str(p.dtype), requires_grad=p.requires_grad)
                                    for n, p in model.named_parameters()}))
            path = self.folder / f"R{completed}.pt"
            if path.exists():
                raise ValueError(f"Refusing to overwrite a head snapshot: {path}")
            torch.save(payload, path)
            self.captured.append(dict(round=completed, loop_round=server.cur_ground, file=path.name, sha256=file_hash(path)))
            self._write_index()
        print(f"[HeadDiag] saved ordinary C0 local5 R{completed} before aggregation: {path}")
