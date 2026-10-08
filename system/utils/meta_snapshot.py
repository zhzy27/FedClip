"""Pre-aggregation immutable full-W snapshots and compatible frozen weight files."""

import copy
import hashlib
import json
import random
from pathlib import Path

import torch

from flcore.clients.clientbase import load_item
from utils.meta_virtual import preserved_rng
from utils.target_projection import EPS, _delta, _dot


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(model):
    return dict(model_class=type(model).__module__ + "." + type(model).__name__,
                ratio_LR=float(getattr(model, "ratio_LR", 1.)),
                parameter_shapes={name: list(p.shape) for name, p in model.named_parameters()},
                buffer_shapes={name: list(p.shape) for name, p in model.named_buffers()})


def protocol_config(server):
    args, client = server.args, server.clients[0]
    return dict(dataset=server.dataset, dataset_subdir=server.meta_split_record["dataset_subdir"],
        model_family=getattr(args, "model_family", ""), num_clients=server.num_clients,
        num_classes=getattr(args, "num_classes", 100), local_epochs=client.local_epochs,
        batch_size=client.batch_size, local_lr=client.learning_rate,
        regularization=float(client.args.regular_lamda), is_regular=int(client.args.is_regular))


def capture_snapshot(server):
    completed = server.cur_ground + 1
    if not hasattr(server, "meta_split_record") or not hasattr(server.clients[0], "_meta_c0_train_data"):
        raise RuntimeError("C0 holdout must be active before collecting any baseline snapshot.")
    if server._meta_normal_rounds != {cid: server.cur_ground for cid in range(server.num_clients)}:
        raise RuntimeError("Snapshots require all ordinary uploads from this round.")
    root = Path(getattr(server.args, "meta_snapshot_dir", "") or Path(server.save_folder_name) / "meta_snapshots")
    folder = root / f"R{completed}"
    folder.mkdir(parents=True, exist_ok=False)
    config = protocol_config(server)
    files = {}
    with preserved_rng():
        import numpy as np
        numpy_state = np.random.get_state()
        torch.save(dict(python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                        torch_cpu=torch.get_rng_state(),
                        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
                   folder / "capture_rng.pt")
        files["capture_rng.pt"] = file_hash(folder / "capture_rng.pt")
        global_model = load_item(server.role, "model", server.save_folder_name).to(server.device)
        global_model = server._recover_if_needed(global_model).to(server.device)
        global_params = dict(global_model.named_parameters())
        target = server._load_full_parameters(0)
        target_delta = _delta(target, global_params)
        target_square = _dot(target_delta, target_delta)
        torch.save({n: p.detach().cpu().clone() for n, p in global_params.items()}, folder / "global.pt")
        files["global.pt"] = file_hash(folder / "global.pt")
        capacities, coefficients = [], {}
        for cid in range(server.num_clients):
            client = server.clients[cid]
            low = client._load_model()
            capacities.append(dict(client_id=cid, **model_signature(low)))
            if cid == 0:
                torch.save(copy.deepcopy(low).cpu(), folder / "c0_template.pt")
                files["c0_template.pt"] = file_hash(folder / "c0_template.pt")
            post = target if cid == 0 else server._load_full_parameters(cid)
            dot = _dot(_delta(post, global_params), target_delta)
            coefficients[str(cid)] = dot / (target_square + EPS) if cid != 0 and dot < 0 else 0.
            name = f"client_{cid}.pt"
            torch.save({n: p.detach().cpu().clone() for n, p in post.items()}, folder / name)
            files[name] = file_hash(folder / name)
            del low, post
    metadata = dict(schema=1, round=completed, loop_round=server.cur_ground, training_seed=server.seed,
        source_mode=server.target_proj_mode, client_ids=list(range(server.num_clients)),
        aggregation_order=list(server.uploaded_ids),
        config=config, capacities=capacities, split=server.meta_split_record,
        projection_epsilon=EPS, projection_coefficients=coefficients, files=files,
        torch_version=torch.__version__, capture_scope="ordinary_local_complete_before_aggregation",
        numpy_version=np.__version__, snapshot_rounds_requested=sorted(server._meta_snapshot_rounds),
        backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        split_active_before_first_local_training=True)
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["config"] != config or manifest["split"]["split_id"] != server.meta_split_record["split_id"]:
            raise ValueError("Cannot mix incompatible baseline snapshots.")
    else:
        manifest = dict(schema=1, config=config, capacities=capacities, split=server.meta_split_record, snapshots=[])
    manifest["snapshots"].append(dict(round=completed, folder=folder.name, metadata_sha256=file_hash(folder / "metadata.json")))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[MetaSnapshot] R{completed} saved: {folder.resolve()}")


class Snapshot:
    def __init__(self, folder, device="cpu", dtype=None):
        self.folder, self.device, self.dtype = Path(folder), device, dtype
        self.metadata = json.loads((self.folder / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["schema"] != 1 or not self.metadata.get("split_active_before_first_local_training"):
            raise ValueError("Snapshot does not certify a pre-training C0 holdout.")
        for name, digest in self.metadata["files"].items():
            if file_hash(self.folder / name) != digest:
                raise ValueError(f"Snapshot file fingerprint mismatch: {name}")
        self.client_ids = self.metadata["client_ids"]
        if self.client_ids != list(range(len(self.client_ids))) or self.metadata["projection_epsilon"] != EPS:
            raise ValueError("Incompatible snapshot IDs or projection epsilon.")
        self.coefficients = {int(cid): value for cid, value in self.metadata["projection_coefficients"].items()}

    def parameters(self, client_id=None):
        name = "global.pt" if client_id is None else f"client_{client_id}.pt"
        values = torch.load(self.folder / name, map_location=self.device, weights_only=True)
        return {n: p.to(dtype=self.dtype) if self.dtype else p for n, p in values.items()}

    def template(self):
        # This is the trusted ordinary C0 checkpoint captured by this repository.
        model = torch.load(self.folder / "c0_template.pt", map_location=self.device, weights_only=False)
        return model.to(dtype=self.dtype) if self.dtype else model


def load_snapshot_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != 1 or not manifest["snapshots"]:
        raise ValueError("No valid baseline snapshots.")
    for row in manifest["snapshots"]:
        if file_hash(root / row["folder"] / "metadata.json") != row["metadata_sha256"]:
            raise ValueError("Snapshot metadata fingerprint mismatch.")
    return manifest


def load_fixed_weights(path, server):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("schema") != 1 or artifact.get("kind") != "meta_projection_fixed":
        raise ValueError("Only a selected shared fixed-weight artifact can enter a federated run.")
    if artifact["client_ids"] != list(range(server.num_clients)) or artifact["config"] != protocol_config(server):
        raise ValueError("Fixed meta weights have incompatible client IDs/model/training configuration.")
    if artifact["split"]["split_id"] != server.meta_split_record["split_id"]:
        raise ValueError("Fixed meta weights require the identical C0 holdout split.")
    with preserved_rng():
        capacities = [dict(client_id=c.id, **model_signature(c._load_model())) for c in server.clients]
    if capacities != artifact["capacities"]:
        raise ValueError("Fixed meta weights have incompatible client capacities/parameter layouts.")
    weights = torch.tensor(artifact["weights"], dtype=torch.float64)
    if weights.shape != (server.num_clients,) or not torch.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum().item() - 1.) > 1e-12:
        raise ValueError("Invalid frozen meta aggregation simplex.")
    if artifact.get("selection", {}).get("test_used", True):
        raise ValueError("Meta weight artifact must certify validation-only selection.")
    return artifact, {cid: weights[cid].item() for cid in range(server.num_clients)}
