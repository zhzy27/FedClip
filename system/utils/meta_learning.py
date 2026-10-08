"""Validation-only Adam over fully unrolled C0 adaptation, one graph at a time."""

import gc
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from utils.meta_data import read_c0_training, validate_split
from utils.meta_snapshot import Snapshot, file_hash, load_snapshot_manifest
from utils.meta_virtual import adapted_validation_loss, differentiable_projection, sync, snapshot_backend


def batch_plan(data, epochs, batch_size, seed, device, dtype):
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
    # Freeze the batch order for all compared initializations/updates/finite differences.
    return [(x.to(device=device, dtype=dtype), y.to(device)) for _ in range(epochs) for x, y in loader]


def weight_metrics(weights):
    values = weights.detach().double().cpu().tolist()
    helpers = values[1:]
    mass = math.fsum(helpers)
    if mass <= 0 or any(not math.isfinite(w) or w < 0 for w in values):
        raise ValueError("Non-finite/underflowed meta weight distribution.")
    return dict(weights=values, target_weight=values[0], helper_total_weight=mass,
                effective_all_client_count=1. / math.fsum(w ** 2 for w in values),
                effective_helper_count=1. / math.fsum((w / mass) ** 2 for w in helpers))


class MetaProblem:
    def __init__(self, root, dataset_root, device="cpu", dtype=None, virtual_seed=777, checkpoint_steps=0):
        self.root, self.device, self.dtype = Path(root), device, dtype
        self.manifest = load_snapshot_manifest(root)
        self.snapshots = [Snapshot(self.root / row["folder"], device, dtype) for row in self.manifest["snapshots"]]
        record = self.manifest["split"]
        data = read_c0_training(dataset_root, record["dataset"], record["dataset_subdir"])
        validate_split(record, data)
        self.train = [data[i] for i in record["train_indices"]]
        self.validation = [data[i] for i in record["validation_indices"]]
        self.virtual_seed, self.checkpoint_steps = virtual_seed, checkpoint_steps
        config = self.manifest["config"]
        if config["local_epochs"] != 5 or config["batch_size"] != 16 or config["local_lr"] != .005 or config["regularization"] != .001 or config["is_regular"] != 1:
            raise ValueError("This diagnostic requires the declared 5-epoch/batch16/SGD.005/Frobenius.001 protocol.")
        self.config = config
        for snapshot in self.snapshots:
            if snapshot.metadata["config"] != config or snapshot.metadata["split"]["split_id"] != record["split_id"] or snapshot.metadata["capacities"] != self.manifest["capacities"]:
                raise ValueError("Incompatible snapshot collection.")
        self.client_ids = self.snapshots[0].client_ids

    def objective(self, snapshot, z):
        global_params = snapshot.parameters()
        dtype = next(iter(global_params.values())).dtype
        template = snapshot.template()
        seed = self.virtual_seed + snapshot.metadata["round"] * 1009
        train = batch_plan(self.train, 5, 16, seed, self.device, dtype)
        validation = [(x.to(device=self.device, dtype=dtype), y.to(self.device)) for x, y in DataLoader(
            self.validation, 16, shuffle=False, generator=torch.Generator().manual_seed(seed + 1))]
        full = differentiable_projection(global_params, snapshot.parameters, z.softmax(0),
                                         snapshot.coefficients, snapshot.client_ids,
                                         snapshot.metadata.get("aggregation_order"))
        return adapted_validation_loss(template, full, train, validation, .005, .001, seed,
                                       self.checkpoint_steps)

    def trajectory(self, snapshots, initialization, updates, outer_lr, bias_client=0, bias_logit=.25):
        if updates < 1 or not math.isfinite(outer_lr) or outer_lr <= 0:
            raise ValueError("Adam outer lr/updates must be positive.")
        z = torch.zeros(len(self.client_ids), dtype=torch.float64, device=self.device, requires_grad=True)
        if initialization == "biased":
            if not 0 <= bias_client < len(z) or not math.isfinite(bias_logit):
                raise ValueError("Invalid mild-bias initialization.")
            with torch.no_grad():
                z[bias_client] = bias_logit
        elif initialization != "uniform":
            raise ValueError("Initialization must be uniform or biased.")
        optimizer = torch.optim.Adam([z], lr=outer_lr)
        history, selected = [], None
        for iteration in range(updates + 1):
            optimizer.zero_grad(set_to_none=True)
            before = z.softmax(0).detach().clone()
            logits_before = z.detach().cpu().tolist()
            snapshot_losses, details, forward_seconds, backward_seconds, peak = {}, {}, 0., 0., 0
            try:
                for snapshot in snapshots:
                    gc.collect()
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.reset_peak_memory_stats(self.device)
                    sync(self.device)
                    start = time.perf_counter()
                    with snapshot_backend(snapshot.metadata.get("backend")):
                        loss, diagnostics = self.objective(snapshot, z)
                        sync(self.device)
                        forward_seconds += time.perf_counter() - start
                        key = f"R{snapshot.metadata['round']}"
                        snapshot_losses[key] = loss.detach().item()
                        details[key] = diagnostics
                        if iteration < updates:
                            start = time.perf_counter()
                            # One snapshot graph is freed by backward before constructing the next.
                            (loss / len(snapshots)).backward()
                            sync(self.device)
                            backward_seconds += time.perf_counter() - start
                        if torch.device(self.device).type == "cuda":
                            peak = max(peak, torch.cuda.max_memory_allocated(self.device))
                        del loss
                mean = math.fsum(snapshot_losses.values()) / len(snapshot_losses)
                gradient_norm = z.grad.norm().item() if z.grad is not None else None
                if not math.isfinite(mean) or z.grad is not None and not torch.isfinite(z.grad).all():
                    raise ValueError("Non-finite full-unroll hypergradient; no approximation will replace it.")
                metrics = weight_metrics(before)
                if selected is None or mean < selected["validation_ce"]:
                    selected = dict(iteration=iteration, validation_ce=mean, weights=metrics["weights"],
                                    initialization=initialization, per_snapshot_ce=dict(snapshot_losses))
                if iteration < updates:
                    optimizer.step()
                    if not torch.isfinite(z).all():
                        raise ValueError("Non-finite outer Adam logits.")
                row = dict(iteration=iteration, update_applied=int(iteration < updates),
                    initialization=initialization, mean_adapted_validation_ce=mean,
                    snapshot_adapted_validation_ce=snapshot_losses, z_gradient_norm=gradient_norm,
                    weight_change_norm=(z.softmax(0).detach() - before).norm().item(),
                    logits_before_update=logits_before,
                    weights_after_update=z.softmax(0).detach().cpu().tolist(),
                    forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                    peak_allocated_cuda_bytes=peak, nonfinite=False, virtual_details=details, **metrics)
                # No tensors/computation graphs retained in optimization logs.
                history.append(row)
                print(f"[MetaAdam] {initialization} step={iteration}/{updates} validation_ce={mean:.8g} "
                      f"gradient_norm={gradient_norm} C0={metrics['target_weight']:.6g}", flush=True)
            except (RuntimeError, ValueError) as error:
                history.append(dict(iteration=iteration, initialization=initialization,
                    nonfinite="non-finite" in str(error).lower(), failure=True,
                    error=str(error), snapshot_adapted_validation_ce=snapshot_losses,
                    failed_snapshot=f"R{snapshot.metadata['round']}", virtual_details=details,
                    weights=before.cpu().tolist(), target_weight=before[0].item(),
                    forward_seconds=forward_seconds, backward_seconds=backward_seconds,
                    peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(self.device)
                    if torch.device(self.device).type == "cuda" else None))
                return dict(status="failed", error=str(error), history=history, selected=None)
        return dict(status="complete", history=history, selected=selected)


def optimize(problem, output_dir, strategy="fixed", updates=50, outer_lr=.05,
             initializations=("uniform", "biased"), bias_client=0, bias_logit=.25):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    budget = dict(optimizer="Adam", lr=outer_lr, updates=updates, virtual_epochs=5,
        batch_size=16, virtual_lr=.005, momentum=0., regularization=.001, gradient_clip_norm=10.,
        checkpoint_steps=problem.checkpoint_steps, virtual_seed=problem.virtual_seed,
        initializations=list(initializations), bias_client=bias_client, bias_logit=bias_logit,
        model_dtype=str(problem.dtype) if problem.dtype else "snapshot_dtype",
        graph="complete_native_svd_full_unroll", final_candidate_evaluation=True)
    common = dict(schema=1, client_ids=problem.client_ids, config=problem.config,
        capacities=problem.manifest["capacities"], split=problem.manifest["split"],
        snapshot_sources=[dict(path=str(s.folder.resolve()), metadata_sha256=file_hash(s.folder / "metadata.json"),
                               round=s.metadata["round"], training_seed=s.metadata["training_seed"],
                               source_mode=s.metadata.get("source_mode", "projection"),
                               synthetic=s.metadata.get("synthetic", False))
                          for s in problem.snapshots], optimization_budget=budget)
    groups = {"fixed": problem.snapshots} if strategy == "fixed" else {
        f"R{s.metadata['round']}": [s] for s in problem.snapshots}
    artifacts = {}
    for label, snapshots in groups.items():
        successes = []
        for init in initializations:
            trace = problem.trajectory(snapshots, init, updates, outer_lr, bias_client, bias_logit)
            (output / f"{label}_{init}_trajectory.json").write_text(json.dumps({**common, **trace}, indent=2, allow_nan=False), encoding="utf-8")
            if trace["status"] == "complete":
                successes.append(trace["selected"])
        if not successes:
            (output / "failure.json").write_text(json.dumps(dict(group=label, reason="No complete finite optimization trajectory; inspect logs."), indent=2), encoding="utf-8")
            raise RuntimeError("Full-unroll optimization failed; no usable fixed artifact was exported.")
        selected = min(successes, key=lambda item: item["validation_ce"])
        artifact = dict(**common, kind="meta_projection_fixed" if strategy == "fixed" else "meta_projection_snapshot_diagnostic",
            weights=selected["weights"], selection=dict(criterion="minimum_adapted_validation_ce",
                test_used=False, selected_iteration=selected["iteration"], selected_initialization=selected["initialization"],
                validation_ce=selected["validation_ce"], per_snapshot_ce=selected["per_snapshot_ce"],
                initialization_comparison="same_budget_validation_only"))
        artifacts[label] = artifact
        name = "fixed_weights.json" if strategy == "fixed" else f"{label}_weights.json"
        (output / name).write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    if strategy == "per_snapshot":
        (output / "per_snapshot_weights.json").write_text(json.dumps(dict(schema=1, kind="per_snapshot_diagnostic_only", snapshots=artifacts), indent=2), encoding="utf-8")
    return artifacts
