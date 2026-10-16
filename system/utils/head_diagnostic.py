"""Offline, fixed-budget fc3 retraining on immutable C0 low-rank snapshots."""

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from utils.head_snapshot import (apply_backend, backend_settings, file_hash, read_image_shard,
                                 sample_fingerprint, tensor_fingerprint)
from utils.target_head import classifier_names, preserved_head_rng


PROTOCOL = dict(epochs=5, observations=[0, 1, 5], batch_size=16, learning_rate=.005,
                momentum=0., weight_decay=0., loss="CE_only", grad_clip_norm=10.,
                backbone="original_low_rank_eval_frozen_parameters_and_buffers_no_SVD",
                classifier="fc3.weight_and_bias_by_parameter_identity", feature_cache=False,
                feature_samples="all_original_C0_training_samples_in_original_order",
                selection="fixed_0_1_5_epochs_no_test_based_selection_no_federated_feedback")


def training_order(sample_count, shuffle_seed):
    if sample_count <= 0 or shuffle_seed < 0:
        raise ValueError("Nonempty C0 training data and nonnegative shuffle seed are required.")
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    return [torch.randperm(sample_count, generator=generator).tolist() for _ in range(5)]


def order_fingerprint(order):
    return hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest()


def batches(data, order, device):
    for start in range(0, len(order), 16):
        samples = [data[i] for i in order[start:start + 16]]
        yield torch.stack([x for x, _ in samples]).to(device), torch.stack([y for _, y in samples]).to(device)


@torch.no_grad()
def evaluate(model, data, device, features=False):
    states = [(module, module.training) for module in model.modules()]
    model.eval()
    feature_sum, feature_count = 0., 0
    def observe(module, inputs):
        nonlocal feature_sum, feature_count
        # This is the actual fc3 input, AFTER head.0's ReLU.
        feature_sum += float(inputs[0].double().norm(dim=1).sum())
        feature_count += inputs[0].shape[0]
    hook = model.fc3.register_forward_pre_hook(observe) if features else None
    try:
        ce, correct, count = 0., 0, 0
        for x, y in batches(data, list(range(len(data))), device):
            output = model(x)
            ce += float(F.cross_entropy(output, y, reduction="sum"))
            correct += int((output.argmax(1) == y).sum())
            count += len(y)
        if count == 0:
            raise ValueError("Head diagnostic evaluation requires nonempty data.")
        return dict(ce=ce / count, correct=correct, samples=count, accuracy=correct / count,
                    input_feature_mean_l2=feature_sum / feature_count if feature_count else 0.,
                    feature_sample_count=feature_count)
    finally:
        if hook is not None:
            hook.remove()
        for module, training in states:
            module.training = training


@torch.no_grad()
def fc2_weight_norm(model):
    # Reconstruct the weight algebraically for measurement only. Never replace
    # a module, create a full-rank layer or run SVD on the diagnostic model.
    layer = model.fc2
    weight = layer.reconstruct_full_weight() if hasattr(layer, "reconstruct_full_weight") else layer.weight
    return float(weight.double().norm())


def frozen_fingerprint(model, head_ids):
    parameters = {"parameter:" + n: p for n, p in model.named_parameters() if id(p) not in head_ids}
    parameters.update({"buffer:" + n: p for n, p in model.named_buffers()})
    return tensor_fingerprint(parameters)


def load_verified_data(metadata, dataset_root):
    root = Path(dataset_root) / metadata["dataset"] / metadata["dataset_subdir"]
    result = {}
    for split in ("train", "test"):
        path = root / split / "0.npz"
        expected = metadata["data"][split]
        if file_hash(path) != expected["file_sha256"]:
            raise ValueError(f"C0 {split} data file fingerprint mismatch: {path}")
        data = read_image_shard(path)
        if len(data) != expected["sample_count"] or sample_fingerprint(data) != expected["sample_sha256"]:
            raise ValueError(f"C0 {split} processed samples do not match the snapshot.")
        result[split] = data
    return result


def diagnose_snapshot(path, dataset_root, device="cpu", shuffle_seed=1729):
    path = Path(path).resolve()
    original_file_hash = file_hash(path)
    previous_backend = backend_settings()
    start = time.perf_counter()
    try:
        with preserved_head_rng():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("schema") != 1 or payload.get("stage") != "C0_post_local_before_aggregation":
                raise ValueError("Expected a C0 post-local head snapshot, not a meta/aggregate/feedback checkpoint.")
            metadata, model = payload["metadata"], payload["model"]
            if metadata["args"].get("target_local_schedule", "joint") != "joint":
                raise ValueError("Do not pair FedRep-style snapshots with the existing joint head diagnostic protocol.")
            if metadata["round"] != metadata["loop_round"] + 1 or metadata["client_id"] != 0:
                raise ValueError("Head snapshot round/client mismatch.")
            if tensor_fingerprint(model.state_dict()) != metadata["model_state_sha256"]:
                raise ValueError("Head snapshot model fingerprint mismatch.")
            initial = payload["initial_public_head"]
            if tensor_fingerprint(initial) != metadata["public_head_sha256"]:
                raise ValueError("Public classifier initialization fingerprint mismatch.")
            names = classifier_names(model)
            if set(initial) != set(names):
                raise ValueError("Public initialization must contain only fc3 weight/bias.")
            data = load_verified_data(metadata, dataset_root)
            order = training_order(len(data["train"]), shuffle_seed)
            order_sha = order_fingerprint(order)
            apply_backend(metadata["backend"])
            model = model.to(device)
            common = dict(mode=metadata["mode"], head_mode=metadata["head_mode"], round=metadata["round"],
                loop_round=metadata["loop_round"], seed=metadata["seed"], capacity=metadata["capacity"],
                snapshot=str(path), snapshot_sha256=original_file_hash,
                public_head_sha256=metadata["public_head_sha256"],
                train_data_sha256=metadata["data"]["train"]["sample_sha256"],
                test_data_sha256=metadata["data"]["test"]["sample_sha256"],
                train_file_sha256=metadata["data"]["train"]["file_sha256"],
                test_file_sha256=metadata["data"]["test"]["file_sha256"],
                shuffle_seed=shuffle_seed, batch_order_sha256=order_sha, device=str(device),
                metric_role="offline_head_diagnostic_not_original_algorithm_result")
            rows = []
            def record(variant, epoch, candidate):
                train = evaluate(candidate, data["train"], device, features=True)
                test = evaluate(candidate, data["test"], device)
                row = dict(**common, variant=variant, head_train_epoch=epoch,
                    train_ce=train["ce"], train_accuracy=train["accuracy"], train_correct=train["correct"],
                    train_samples=train["samples"], test_ce=test["ce"], test_accuracy=test["accuracy"],
                    test_correct=test["correct"], test_samples=test["samples"],
                    head_weight_norm=float(candidate.fc3.weight.detach().double().norm()),
                    head_bias_norm=float(candidate.fc3.bias.detach().double().norm()),
                    fc2_recovered_weight_frobenius=fc2_weight_norm(candidate),
                    head_input_mean_l2=train["input_feature_mean_l2"], feature_sample_count=train["feature_sample_count"],
                    head_train_batches=epoch * math.ceil(len(data["train"]) / 16),
                    head_train_sample_visits=epoch * len(data["train"]),
                    original_post_local_test_accuracy=metadata["native_test_accuracy"])
                if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
                    raise ValueError("Non-finite offline diagnostic metric.")
                rows.append(row)
                return row
            native = record("native", 0, model)
            if (native["test_correct"], native["test_samples"]) != (metadata["native_correct"], metadata["native_test_samples"]):
                raise ValueError(f"Native accuracy failed to reproduce the source snapshot: {native['test_accuracy']} vs {metadata['native_test_accuracy']}")
            # A fresh independent copy for EVERY snapshot, with no state inherited
            # from another mode/round/diagnostic. Retain the original low-rank path.
            fresh = copy.deepcopy(model)
            parameters = dict(fresh.named_parameters())
            with torch.no_grad():
                for name in names:
                    if parameters[name].shape != initial[name].shape:
                        raise ValueError(f"Public head shape mismatch: {name}")
                    parameters[name].copy_(initial[name])
            if tensor_fingerprint({n: parameters[n] for n in names}) != metadata["public_head_sha256"]:
                raise ValueError("Fresh classifier was not reset exactly to the public initialization.")
            head_parameters = list(fresh.fc3.parameters())
            head_ids = {id(p) for p in head_parameters}
            for p in fresh.parameters():
                p.requires_grad_(id(p) in head_ids)
                p.grad = None
            before = frozen_fingerprint(fresh, head_ids)
            initial_fresh_head = tensor_fingerprint({n: parameters[n] for n in names})
            optimizer = torch.optim.SGD(head_parameters, lr=.005, momentum=0., weight_decay=0.)
            record("fresh_head", 0, fresh)
            for epoch, indices in enumerate(order, 1):
                fresh.eval()
                fresh.fc3.train()  # Keep the aliased backbone ReLU in eval as well.
                for x, y in batches(data["train"], indices, device):
                    optimizer.zero_grad()
                    loss = F.cross_entropy(fresh(x), y)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite head training CE.")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head_parameters, 10., error_if_nonfinite=True)
                    optimizer.step()
                if frozen_fingerprint(fresh, head_ids) != before:
                    raise RuntimeError("Frozen backbone parameters or buffers changed during head retraining.")
                if epoch in (1, 5):
                    record("fresh_head", epoch, fresh)
            changed = tensor_fingerprint({n: parameters[n] for n in names}) != initial_fresh_head
            result = dict(snapshot=str(path), source_metadata=metadata, protocol=dict(**PROTOCOL, shuffle_seed=shuffle_seed),
                batch_order_sha256=order_sha, batch_indices=order, rows=rows,
                backbone_and_buffers_bitwise_unchanged=True, classifier_updated=changed,
                native_reproduced=True, elapsed_seconds=time.perf_counter() - start)
            return result
    finally:
        apply_backend(previous_backend)
        if file_hash(path) != original_file_hash:
            raise RuntimeError("Offline diagnosis modified its input snapshot.")


def compare_results(results):
    cohorts, initializations = {}, {}
    for result in results:
        meta = result["source_metadata"]
        # Four configurations at the same training seed must use one public head.
        source = (meta["seed"], meta["args"]["model_family"], meta["capacity"])
        previous = initializations.setdefault(source, meta["public_head_sha256"])
        if previous != meta["public_head_sha256"]:
            raise ValueError("Compared runs do not share the same original public classifier initialization.")
        key = (meta["mode"], meta["round"], meta["seed"], meta["data"]["train"]["sample_sha256"], meta["data"]["test"]["sample_sha256"], meta["capacity"])
        cohort = cohorts.setdefault(key, {})
        if meta["head_mode"] in cohort:
            raise ValueError("Duplicate snapshot for the same comparison round/mode/head; pass one matched run per configuration.")
        cohort[meta["head_mode"]] = result
    comparisons = []
    for key, cohort in sorted(cohorts.items()):
        if set(cohort) != {"shared", "local"}:
            continue
        rows = {kind: {(r["variant"], r["head_train_epoch"]): r for r in result["rows"]} for kind, result in cohort.items()}
        if cohort["shared"]["batch_order_sha256"] != cohort["local"]["batch_order_sha256"]:
            raise ValueError("Compared snapshots used different diagnostic batch orders.")
        gap = lambda variant, epoch: 100 * (rows["shared"][variant, epoch]["test_accuracy"] - rows["local"][variant, epoch]["test_accuracy"])
        native, one, five = gap("native", 0), gap("fresh_head", 1), gap("fresh_head", 5)
        comparisons.append(dict(mode=key[0], round=key[1], seed=key[2], capacity=key[5],
            native_shared_minus_local_pp=native, fresh1_shared_minus_local_pp=one, fresh5_shared_minus_local_pp=five,
            signed_gap_reduction1_pp=native - one, signed_gap_reduction5_pp=native - five,
            absolute_gap_reduction1_pp=abs(native) - abs(one), absolute_gap_reduction5_pp=abs(native) - abs(five)))
    return comparisons


def write_outputs(folder, results):
    folder = Path(folder)
    rows = [row for result in results for row in result["rows"]]
    comparisons = compare_results(results)
    for filename, values in (("head_diagnostic.csv", rows), ("head_comparison.csv", comparisons)):
        if values:
            with (folder / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(values[0]))
                writer.writeheader()
                writer.writerows(values)
    document = dict(schema=1, metric_role="offline_diagnostic_not_federated_final_best_last10_last20",
                    accuracy_units="fraction; comparison gaps in percentage points", protocol=PROTOCOL,
                    results=results, comparisons=comparisons)
    (folder / "head_diagnostic.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Frozen-backbone head diagnostic", "", "Gaps are shared minus local in percentage points; positive means local is lower.", "",
        "| Mode | Round | Native gap | Fresh 1 gap | Fresh 5 gap | Absolute gap reduction (5) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['mode']} | {row['round']} | {row['native_shared_minus_local_pp']:.3f} | {row['fresh1_shared_minus_local_pp']:.3f} | {row['fresh5_shared_minus_local_pp']:.3f} | {row['absolute_gap_reduction5_pp']:.3f} |")
    if not comparisons:
        lines += ["", "No matched shared/local pair was supplied; individual diagnostics are available."]
    lines += ["", "A substantially smaller gap supports classifier adaptation/training as a factor. A persistent gap supports representation quality as another possible factor. Partial reduction is consistent with both. Five fixed epochs do not prove optimal head convergence; no F-regularization cause is presumed.",
              "", "These are offline diagnostics on original C0 train/test data, never replacement algorithm results or feedback updates. No epoch/hyperparameter is chosen by test accuracy."]
    (folder / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparisons
