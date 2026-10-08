"""Measured full-CNN exact-unroll cost and random-direction finite differences."""

import copy
import json
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from utils.meta_data import create_split, read_c0_training
from utils.meta_learning import MetaProblem, batch_plan
from utils.meta_snapshot import Snapshot, file_hash, model_signature, load_snapshot_manifest
from utils.meta_virtual import (adapted_validation_loss, differentiable_decompose, differentiable_projection,
                                preserved_rng, projection_coefficients, sync, snapshot_backend)


def synthetic_collection(root, train_samples=40):
    from flcore.trainmodel.models import Hyper_CNN_512
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data_root = root / "dataset"
    shard = data_root / "Cifar100" / "pat_20" / "train"
    shard.mkdir(parents=True)
    with preserved_rng(29):
        x = torch.randn(train_samples, 3, 32, 32)
        y = torch.arange(train_samples) % 4
        import numpy as np
        np.savez(shard / "0.npz", data=dict(x=x.numpy(), y=y.numpy()))
        split = create_split(list(zip(x, y)))
        full = Hyper_CNN_512(num_classes=100, ratio_LR=1.)
        template = Hyper_CNN_512(num_classes=100, ratio_LR=.9)
        initial = {n: p.detach().clone() for n, p in full.named_parameters()}
        posts = []
        for cid in range(20):
            post = {n: p + torch.randn_like(p) * .01 for n, p in initial.items()}
            post["fc3.bias"] = post["fc3.bias"] + (cid - 9.5) * .02
            posts.append(post)
        folder = root / "R20"
        folder.mkdir()
        torch.save(initial, folder / "global.pt")
        torch.save(template, folder / "c0_template.pt")
        files = {name: file_hash(folder / name) for name in ("global.pt", "c0_template.pt")}
        for cid, post in enumerate(posts):
            name = f"client_{cid}.pt"
            torch.save(post, folder / name)
            files[name] = file_hash(folder / name)
        config = dict(dataset="Cifar100", dataset_subdir="pat_20", model_family="Decom_CNN-5-512",
            num_clients=20, num_classes=100, local_epochs=5, batch_size=16, local_lr=.005,
            regularization=.001, is_regular=1)
        capacities = [dict(client_id=cid, **model_signature(template)) for cid in range(20)]
        coefficients = projection_coefficients(initial, posts[0], enumerate(posts))
        metadata = dict(schema=1, round=20, loop_round=19, training_seed=0, source_mode="synthetic_structure_fixture",
            client_ids=list(range(20)), aggregation_order=list(range(20)), config=config, capacities=capacities,
            split=split, files=files, projection_epsilon=1e-12, projection_coefficients=coefficients,
            split_active_before_first_local_training=True, synthetic=True)
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest = dict(schema=1, config=config, capacities=capacities, split=split,
            snapshots=[dict(round=20, folder="R20", metadata_sha256=file_hash(folder / "metadata.json"))])
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root, data_root


def ordinary_adaptation(template, full_params, batches, lr=.005, regularization=.001, seed=777):
    device = next(iter(full_params.values())).device
    with preserved_rng():
        model = copy.deepcopy(template).to(device)
        ratio = model.ratio_LR
        model.recover_larger_model()
        model.to(device=device, dtype=next(iter(full_params.values())).dtype)
        with torch.no_grad():
            for name, value in model.named_parameters():
                value.copy_(full_params[name])
        sync(device)
        start = time.perf_counter()
        model.decom_larger_model(ratio)
        # Baseline downlink copies CPU-allocated factors into the existing device model.
        model.to(device=device, dtype=next(iter(full_params.values())).dtype)
        sync(device)
        decomposition_seconds = time.perf_counter() - start
        with preserved_rng(seed):
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            sync(device)
            start = time.perf_counter()
            clipped = 0
            for x, y in batches:
                optimizer.zero_grad()
                ce = torch.nn.functional.cross_entropy(model(x), y)
                loss = ce + regularization * model.frobenius_decay()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                clipped += int(norm > 10.)
                optimizer.step()
            sync(device)
            train_seconds = time.perf_counter() - start
        return model, dict(ordinary_decomposition_seconds=decomposition_seconds,
            ordinary_train_seconds=train_seconds, ordinary_train_batches=len(batches), ordinary_clipped_steps=clipped)


def _benchmark(root, dataset_root, output, device="cuda:0", checkpoint_steps=0, fd_epsilon=1e-4):
    problem = MetaProblem(root, dataset_root, device, checkpoint_steps=checkpoint_steps)
    snapshot = problem.snapshots[0]
    seed = problem.virtual_seed + snapshot.metadata["round"] * 1009
    initial, template = snapshot.parameters(), snapshot.template()
    dtype = next(iter(initial.values())).dtype
    train = batch_plan(problem.train, 5, 16, seed, device, dtype)
    val = [(x.to(device=device, dtype=dtype), y.to(device)) for x, y in DataLoader(
        problem.validation, 16, generator=torch.Generator().manual_seed(seed + 1))]
    z = torch.zeros(len(snapshot.client_ids), dtype=torch.float64, device=device, requires_grad=True)
    # Warm both implementations before the reported comparison, not just the ordinary path.
    warm_z = torch.zeros_like(z, requires_grad=True)
    warm_full = differentiable_projection(initial, snapshot.parameters, warm_z.softmax(0), snapshot.coefficients, snapshot.client_ids)
    warm_ordinary, _ = ordinary_adaptation(template, {n: p.detach() for n, p in warm_full.items()}, train, seed=seed)
    warm_loss, _ = adapted_validation_loss(template, warm_full, train, val, seed=seed, checkpoint_steps=checkpoint_steps)
    warm_loss.backward()
    del warm_ordinary, warm_loss, warm_full, warm_z
    full = differentiable_projection(initial, snapshot.parameters, z.softmax(0), snapshot.coefficients, snapshot.client_ids)
    ordinary, report = ordinary_adaptation(template, {n: p.detach() for n, p in full.items()}, train, seed=seed)
    with torch.no_grad():
        ordinary.eval()
        ordinary_ce = sum(torch.nn.functional.cross_entropy(ordinary(x), y).item() * len(y) for x, y in val) / sum(len(y) for _, y in val)
        reference_parameters = {n: p.detach().cpu().clone() for n, p in ordinary.named_parameters()}
    del ordinary
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    loss, details, final_parameters = adapted_validation_loss(template, full, train, val, seed=seed,
                                                             checkpoint_steps=checkpoint_steps, return_parameters=True)
    sync(device)
    report["differentiable_forward_seconds"] = time.perf_counter() - start
    report["adapted_validation_ce"] = loss.detach().item()
    report["ordinary_adapted_validation_ce"] = ordinary_ce
    report["forward_ce_absolute_error"] = abs(loss.detach().item() - ordinary_ce)
    report["max_parameter_forward_error"] = max((p.detach().cpu() - reference_parameters[n]).abs().max().item()
                                               for n, p in final_parameters.items())
    start = time.perf_counter()
    loss.backward()
    sync(device)
    report.update(outer_backward_seconds=time.perf_counter() - start,
        hypergradient_norm=z.grad.norm().item(), hypergradient=z.grad.detach().cpu().tolist(),
        finite=bool(torch.isfinite(z.grad).all()), nonzero=bool(z.grad.abs().max() > 0),
        peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
        actual_train_samples=len(problem.train), actual_validation_samples=len(problem.validation),
        full_epochs=5, details=details, torch_version=torch.__version__, device=str(device),
        warmup_passes_each=1, gpu_name=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        synthetic=snapshot.metadata.get("synthetic", False))
    del loss, full, initial, template, final_parameters
    if not report["finite"] or not report["nonzero"]:
        report["hypergradient"] = [v if math.isfinite(v) else None for v in report["hypergradient"]]
        report["hypergradient_norm"] = report["hypergradient_norm"] if math.isfinite(report["hypergradient_norm"]) else None
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        raise RuntimeError("Full-epoch hypergradient is non-finite or exactly zero; inspect report, not a fixed-weight impossibility claim.")
    # Short unroll FD uses double precision to separate derivative errors from fp32 loss quantization.
    double_snapshot = Snapshot(snapshot.folder, device, torch.float64)
    initial, template = double_snapshot.parameters(), double_snapshot.template()
    short_train = [(x.double(), y) for x, y in train[:math.ceil(len(problem.train) / 16)]]
    double_val = [(x.double(), y) for x, y in val]
    direction = torch.randn(len(z), dtype=torch.float64, generator=torch.Generator().manual_seed(144)).to(device)
    direction /= direction.norm()
    def short_objective(logits):
        full = differentiable_projection(initial, double_snapshot.parameters, logits.softmax(0),
                                        snapshot.coefficients, snapshot.client_ids)
        return adapted_validation_loss(template, full, short_train, double_val, seed=seed,
                                       checkpoint_steps=checkpoint_steps)[0]
    short_z = torch.zeros_like(z, requires_grad=True)
    short_loss = short_objective(short_z)
    gradient = torch.autograd.grad(short_loss, short_z)[0]
    analytic = (gradient * direction).sum().item()
    def value(offset):
        logits = (short_z.detach() + offset * direction).requires_grad_(True)
        return short_objective(logits).detach().item()
    difference = (value(fd_epsilon) - value(-fd_epsilon)) / (2 * fd_epsilon)
    report["short_unroll_directional_finite_difference"] = dict(epochs=1, dtype="float64",
        epsilon=fd_epsilon, analytic=analytic, central_difference=difference,
        absolute_error=abs(analytic - difference), relative_error=abs(analytic - difference) / max(abs(analytic), abs(difference), 1e-12))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["finite"] or not report["nonzero"]:
        raise RuntimeError("Real-CNN full-epoch hypergradient is not finite/nonzero; inspect benchmark report.")
    return report


def benchmark(root, dataset_root, output, device="cuda:0", checkpoint_steps=0, fd_epsilon=1e-4):
    manifest = load_snapshot_manifest(root)
    metadata = json.loads((Path(root) / manifest["snapshots"][0]["folder"] / "metadata.json").read_text(encoding="utf-8"))
    try:
        with snapshot_backend(metadata.get("backend")):
            return _benchmark(root, dataset_root, output, device, checkpoint_steps, fd_epsilon)
    except (ValueError, RuntimeError) as error:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        failure = path.with_name(path.stem + "_failure.json")
        failure.write_text(json.dumps(dict(status="failed", error=str(error), approximation_used=False,
            snapshot_round=metadata["round"], checkpoint_steps=checkpoint_steps, device=device,
            peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == "cuda" else None), indent=2), encoding="utf-8")
        raise
