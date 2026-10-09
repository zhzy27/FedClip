"""Stage-local numerical audits, exact replay comparisons, and acceptance certificates.

Trajectory probes release each step's graph for comparing forward values only.
The separate hypergradient differentiates all five epochs without truncation.
No probe changes ordinary clients or snapshot files.
"""

import copy
from contextlib import contextmanager, nullcontext
import gc
import json
import math
from pathlib import Path
import time

import torch
from torch.func import functional_call
from torch.utils.data import DataLoader

from utils.meta_benchmark import ordinary_adaptation, ordinary_initialization
from utils.meta_learning import batch_plan
from utils.meta_snapshot import Snapshot, file_hash
from utils.meta_virtual import (_Objective, adapted_validation_loss, backend_state,
    clipped_functional_step, differentiable_decompose, functional_loss, preserved_rng,
    snapshot_backend, snapshot_candidate, sync)


DEFAULT_TOLERANCES = dict(parameter_atol=2e-6, parameter_rtol=2e-5,
    reconstruction_atol=5e-6, reconstruction_rtol=2e-5,
    output_atol=2e-5, output_rtol=2e-5, ce_atol=2e-6, ce_rtol=2e-6,
    gradient_atol=2e-7, gradient_rtol=2e-4, fd_atol=2e-6, fd_rtol=.05,
    noise_multiplier=4., maximum_noise_fraction=.25)
FD_STEPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def code_signature():
    root = Path(__file__).resolve().parents[1]
    files = ["meta_projection.py", "utils/meta_virtual.py", "utils/meta_learning.py", "utils/meta_diagnostics.py",
             "utils/meta_benchmark.py", "flcore/trainmodel/models.py"]
    return {name: file_hash(root / name) for name in files}


def configuration(problem):
    device = torch.device(problem.device)
    profiles = []
    for snapshot in problem.snapshots:
        with snapshot_backend(snapshot.metadata.get("backend"), override_for(problem.backend_policy)):
            profiles.append(dict(round=snapshot.metadata["round"], effective_backend=backend_state(),
                                 float32_matmul_precision=torch.get_float32_matmul_precision()))
    return dict(schema=2, client_ids=problem.client_ids, config=problem.config,
        capacities=problem.manifest["capacities"], split_id=problem.manifest["split"]["split_id"],
        dtype=str(problem.dtype or next(iter(problem.snapshots[0].parameters().values())).dtype),
        virtual_seed=problem.virtual_seed, checkpoint_steps=problem.checkpoint_steps,
        backend_policy=problem.backend_policy, torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, device_type=device.type,
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        cpu_threads=torch.get_num_threads(), cudnn_version=torch.backends.cudnn.version(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        backend_profiles=profiles, code_sha256=code_signature())


def source_signature(snapshot):
    return dict(round=snapshot.metadata["round"], loop_round=snapshot.metadata["loop_round"],
        training_seed=snapshot.metadata["training_seed"], metadata_sha256=file_hash(snapshot.folder / "metadata.json"),
        aggregation_order=snapshot.metadata.get("aggregation_order", snapshot.client_ids),
        backend=snapshot.metadata.get("backend"), synthetic=snapshot.metadata.get("synthetic", False))


def override_for(policy):
    if policy == "snapshot":
        return None
    if policy == "no_tf32":
        return dict(cudnn_allow_tf32=False, matmul_allow_tf32=False)
    if policy == "deterministic_only":
        return dict(cudnn_benchmark=False, cudnn_deterministic=True)
    if policy == "deterministic_no_tf32":
        return dict(cudnn_benchmark=False, cudnn_deterministic=True,
                    cudnn_allow_tf32=False, matmul_allow_tf32=False)
    raise ValueError("Unknown backend policy.")


def tensor_layout(value):
    return dict(dtype=str(value.dtype), device=str(value.device), shape=list(value.shape),
        stride=list(value.stride()), contiguous=value.is_contiguous(),
        storage_offset=value.storage_offset(), requires_grad=value.requires_grad)


def difference(left, right):
    if left.keys() != right.keys():
        raise ValueError("Compared parameter/gradient sets differ.")
    layers = {}
    for name in left:
        a, b = left[name].detach().double(), right[name].detach().double()
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Non-finite or incompatible tensor: {name}")
        error = (a - b).abs().max().item()
        magnitude = max(a.abs().max().item(), b.abs().max().item())
        layers[name] = dict(max_absolute_error=error, reference_magnitude=magnitude,
                            relative_error=error / max(magnitude, 1e-30))
    return dict(max_absolute_error=max(row["max_absolute_error"] for row in layers.values()), layers=layers)


def within(diff, absolute, relative):
    return all(row["max_absolute_error"] <= absolute + relative * row["reference_magnitude"]
               for row in diff["layers"].values())


def reconstructed(parameters, template):
    result, consumed = {}, set()
    for prefix, module in template.named_modules():
        if hasattr(module, "conv_u"):
            u, v = prefix + ".conv_u", prefix + ".conv_v"
            k = module.kernel_size
            matrix = parameters[u] @ parameters[v]
            result[prefix + ".weight"] = matrix.reshape(module.out_channels, k, module.in_channels, k).permute(0, 2, 1, 3)
        elif hasattr(module, "weight_u"):
            u, v = prefix + ".weight_u", prefix + ".weight_v"
            result[prefix + ".weight"] = parameters[u] @ parameters[v]
        else:
            continue
        consumed.update((u, v))
    result.update({name: p for name, p in parameters.items() if name not in consumed})
    return result


@contextmanager
def branch_trace(model):
    """Capture ReLU gates / max-pool winners without changing the forward graph."""
    selected, counts, handles = {}, {}, []
    def hook(name, module, inputs, output):
        index = counts.get(name, 0)
        counts[name] = index + 1
        x = inputs[0].detach()
        if isinstance(module, torch.nn.ReLU):
            value = x > 0
        else:
            value = torch.nn.functional.max_pool2d(x, module.kernel_size, module.stride,
                module.padding, module.dilation, module.ceil_mode, return_indices=True)[1]
        selected[f"{name}#{index}"] = value
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.MaxPool2d)):
            handles.append(module.register_forward_hook(
                lambda m, xs, output, name=name: hook(name, m, xs, output)))
    try:
        yield selected
    finally:
        for handle in handles:
            handle.remove()


def compare_stage(model, wrapper, parameters, buffers, validation, step, tolerances):
    normal = dict(model.named_parameters())
    previous = [(m, m.training) for m in model.modules()] + [(m, m.training) for m in wrapper.modules()]
    try:
        model.eval()
        wrapper.eval()
        with torch.no_grad():
            pd = difference(normal, parameters)
            rd = difference(reconstructed(normal, model), reconstructed(parameters, model))
            outputs_normal, outputs_functional = {}, {}
            normal_ce = functional_ce = 0.
            samples = 0
            for index, (x, y) in enumerate(validation):
                a = model(x)
                b = functional_call(wrapper.model, {**parameters, **buffers}, (x,), strict=True)
                outputs_normal[str(index)], outputs_functional[str(index)] = a, b
                normal_ce += torch.nn.functional.cross_entropy(a, y).item() * len(y)
                functional_ce += torch.nn.functional.cross_entropy(b, y).item() * len(y)
                samples += len(y)
            od = difference(outputs_normal, outputs_functional)
        normal_ce /= samples
        functional_ce /= samples
        ce_error = abs(normal_ce - functional_ce)
        # Signs/bases alone cannot fail acceptance; compare reconstruction and predictions.
        passed = within(rd, tolerances["reconstruction_atol"], tolerances["reconstruction_rtol"]) and within(
            od, tolerances["output_atol"], tolerances["output_rtol"]) and ce_error <= tolerances["ce_atol"] + tolerances["ce_rtol"] * max(abs(normal_ce), abs(functional_ce))
        return dict(step=step, status="passed" if passed else "failed", parameters=pd,
            reconstructed_weights=rd, outputs=od, ordinary_validation_ce=normal_ce,
            functional_validation_ce=functional_ce, ce_absolute_error=ce_error,
            layouts={name: dict(ordinary=tensor_layout(p), functional=tensor_layout(parameters[name])) for name, p in normal.items()})
    finally:
        for module, mode in previous:
            module.training = mode


def trajectory_probe(template, full, train, validation, seed, tolerances, return_parameters=False,
                     trace_branches=False):
    """One-step create_graph=True audits and continuous forward trajectories."""
    device = next(iter(full.values())).device
    model, initialization = ordinary_initialization(template, full, return_details=True)
    wrapper = _Objective(copy.deepcopy(template))
    functional, svd = differentiable_decompose(full, wrapper.model)
    functional = {n: p.detach().requires_grad_(True) for n, p in functional.items()}
    buffers = {n: b.detach().clone() for n, b in wrapper.model.named_buffers()}
    if buffers:
        raise ValueError("Trajectory diagnostic currently requires the audited buffer-free CNN.")
    optimizer = torch.optim.SGD(model.parameters(), lr=.005)
    stages, gradient_steps = [], []
    milestones = {0, 1, 8, len(train) // 5, len(train)}
    stages.append(compare_stage(model, wrapper, functional, buffers, validation, 0, tolerances))
    normal_seconds = functional_seconds = 0.
    with preserved_rng(seed):
        for step, (x, y) in enumerate(train, 1):
            model.train()
            wrapper.train()
            optimizer.zero_grad(set_to_none=True)
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            sync(device)
            start = time.perf_counter()
            with branch_trace(model) if trace_branches else nullcontext({}) as ordinary_branches:
                loss = torch.nn.functional.cross_entropy(model(x), y) + .001 * model.frobenius_decay()
            loss.backward()
            normal = dict(model.named_parameters())
            before = {n: p.grad.detach().clone() for n, p in normal.items()}
            same_point = None
            if step in milestones:
                point = {n: p.detach().clone().requires_grad_(True) for n, p in normal.items()}
                compared = {}
                # Distinguish trajectory drift from backward vs differentiable-backward
                # at the SAME parameters. Preserve the normal stochastic stream.
                with preserved_rng():
                    for create_graph in (False, True):
                        torch.set_rng_state(cpu_rng)
                        if cuda_rng:
                            torch.cuda.set_rng_state_all(cuda_rng)
                        probe_loss = functional_loss(wrapper, point, buffers, x, y, .001)
                        probe_grad = torch.autograd.grad(probe_loss, tuple(point.values()), create_graph=create_graph)
                        compared[str(create_graph)] = {n: g.detach().clone() for n, g in zip(point, probe_grad)}
                        del probe_loss, probe_grad
                same_point = dict(backward_vs_grad_no_graph=difference(before, compared["False"]),
                    backward_vs_grad_create_graph=difference(before, compared["True"]),
                    grad_no_graph_vs_create_graph=difference(compared["False"], compared["True"]))
                del point, compared
            normal_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            normal_norm_functional_reduction = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g, 2) for g in before.values()]), 2)
            normal_scale = torch.clamp(10. / (normal_norm + 1e-6), max=1.)
            after = {n: p.grad.detach().clone() for n, p in normal.items()}
            optimizer.step()
            sync(device)
            normal_seconds += time.perf_counter() - start
            after_cpu = torch.get_rng_state()
            after_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            torch.set_rng_state(cpu_rng)
            if cuda_rng:
                torch.cuda.set_rng_state_all(cuda_rng)
            start = time.perf_counter()
            with branch_trace(wrapper.model) if trace_branches else nullcontext({}) as functional_branches:
                loss = functional_loss(wrapper, functional, buffers, x, y, .001)
            gradients = torch.autograd.grad(loss, tuple(functional.values()), create_graph=True)
            updated, functional_norm = clipped_functional_step(functional, gradients, .005)
            functional_scale = torch.clamp(10. / (functional_norm + 1e-6), max=1.)
            raw_diff = difference(before, dict(zip(functional, gradients)))
            clipped_diff = difference(after, {n: g * functional_scale for n, g in zip(functional, gradients)})
            pd = difference(normal, updated)
            sync(device)
            functional_seconds += time.perf_counter() - start
            gradient_steps.append(dict(step=step, raw_gradients=raw_diff, clipped_gradients=clipped_diff,
                ordinary_norm=normal_norm.item(), functional_norm=functional_norm.item(),
                ordinary_gradients_functional_norm=normal_norm_functional_reduction.item(),
                ordinary_clip_scale=normal_scale.item(), functional_clip_scale=functional_scale.item(),
                updated_parameters=pd, same_point_gradients=same_point,
                branch_selector_changes={n: int((ordinary_branches[n] != functional_branches[n]).sum())
                    for n in ordinary_branches},
                gradient_layouts={n: dict(ordinary=tensor_layout(before[n]), functional=tensor_layout(g))
                    for n, g in zip(functional, gradients)} if step in milestones else None))
            functional = {n: p.detach().requires_grad_(True) for n, p in updated.items()}
            del loss, gradients, updated
            torch.set_rng_state(after_cpu)
            if after_cuda:
                torch.cuda.set_rng_state_all(after_cuda)
            if step in milestones:
                row = compare_stage(model, wrapper, functional, buffers, validation, step, tolerances)
                stages.append(row)
                print(f"[MetaTrajectory] step={step}/{len(train)} CE_error={row['ce_absolute_error']:.8g} status={row['status']}", flush=True)
    report = dict(stages=stages, gradient_steps=gradient_steps, svd=svd,
        first_failed_stage=next((s["step"] for s in stages if s["status"] != "passed"), None),
        first_gradient_difference_step=next((g["step"] for g in gradient_steps if not within(
            g["raw_gradients"], tolerances["gradient_atol"], tolerances["gradient_rtol"])), None),
        first_branch_difference_step=next((g["step"] for g in gradient_steps
            if any(g["branch_selector_changes"].values())), None),
        **initialization, ordinary_instrumented_sgd_seconds=normal_seconds,
        functional_instrumented_step_seconds=functional_seconds,
        ordinary_clipped_steps=sum(g["ordinary_clip_scale"] < 1. for g in gradient_steps),
        functional_clipped_steps=sum(g["functional_clip_scale"] < 1. for g in gradient_steps),
        actual_train_batches=len(train))
    return (report, {n: p.detach() for n, p in functional.items()}) if return_parameters else report


def fd_assessment(rows, noise, tolerances):
    """Require >=2 adjacent resolved h values rather than choosing the best h."""
    for row in rows:
        scale = max(abs(row["analytic"]), abs(row["central_difference"]))
        base = tolerances["fd_atol"] + tolerances["fd_rtol"] * scale
        uncertainty = tolerances["noise_multiplier"] * noise / row["h"]
        row.update(absolute_error=abs(row["analytic"] - row["central_difference"]),
            relative_error=abs(row["analytic"] - row["central_difference"]) / max(scale, 1e-30),
            repeat_noise_derivative_bound=uncertainty, allowed_error=base + uncertainty,
            resolved=uncertainty <= tolerances["maximum_noise_fraction"] * base)
        row["passed"] = row["resolved"] and row["absolute_error"] <= base + uncertainty
    intervals = []
    for left, right in zip(rows, rows[1:]):
        agreement = abs(left["central_difference"] - right["central_difference"])
        if left["passed"] and right["passed"] and agreement <= left["allowed_error"] + right["allowed_error"]:
            intervals.append([left["h"], right["h"]])
    return dict(status="passed" if intervals else "unconfirmed", convergence_intervals=intervals,
                reason=None if intervals else "No resolved adjacent-step convergence interval; nonsmoothness/noise/derivative error not distinguished.")


def objective_probe(snapshot, z, train, validation, seed, checkpoint_steps, return_parameters=False):
    full = snapshot_candidate(snapshot, z)
    if not train:
        # Decomposition-only derivative, for locating a failure before SGD.
        with preserved_rng(seed):
            wrapper = _Objective(snapshot.template()).eval()
            parameters, svd = differentiable_decompose(full, wrapper.model)
            buffers = {n: b.detach().clone() for n, b in wrapper.model.named_buffers()}
            total = sum(functional_loss(wrapper, parameters, buffers, x, y, 0.) * len(y)
                        for x, y in validation)
            loss = total / sum(len(y) for _, y in validation)
            details = dict(svd=svd, actual_train_batches=0, checkpoint_steps=checkpoint_steps)
            return (loss, details, parameters) if return_parameters else (loss, details)
    return adapted_validation_loss(snapshot.template(), full, train, validation, seed=seed,
        checkpoint_steps=checkpoint_steps, return_parameters=return_parameters)


def short_probe(snapshot, logits, train, validation, seed, checkpoint_steps, tolerances,
                h_values, directions=3, repeats=3):
    """Exact graph/replay comparison and FD of that same short functional objective."""
    sync(snapshot.device)
    started = time.perf_counter()
    results, reference = {}, None
    for size in sorted({0, 8, checkpoint_steps}):
        z = logits.detach().clone().requires_grad_(True)
        loss, details, parameters = objective_probe(snapshot, z, train, validation, seed, size, True)
        gradient = torch.autograd.grad(loss, z)[0]
        if not torch.isfinite(gradient).all():
            raise ValueError("Non-finite short-unroll hypergradient.")
        values = {n: p.detach().clone() for n, p in parameters.items()}
        row = dict(loss=loss.item(), gradient=gradient.detach().cpu().tolist(), gradient_norm=gradient.norm().item(),
                   svd=details["svd"])
        if reference is None:
            reference = (loss.item(), values, gradient.detach().clone())
            row["status"] = "passed"
        else:
            gd = difference({"z": reference[2]}, {"z": gradient})
            pd = difference(reference[1], values)
            template = snapshot.template()
            rd = difference(reconstructed(reference[1], template), reconstructed(values, template))
            error = abs(loss.item() - reference[0])
            passed = within(gd, tolerances["gradient_atol"], tolerances["gradient_rtol"]) and within(
                rd, tolerances["reconstruction_atol"], tolerances["reconstruction_rtol"]) and error <= tolerances["ce_atol"] + tolerances["ce_rtol"] * abs(reference[0])
            row.update(status="passed" if passed else "failed", gradient_difference=gd,
                       parameter_difference=pd, reconstructed_weight_difference=rd, ce_absolute_error=error)
        results[str(size)] = row
        del loss, parameters, gradient, values
        gc.collect()
    def value(point):
        # Custom recomputation graphs can contain Python/autograd cycles. No FD
        # evaluation may retain a previous perturbation graph on the GPU.
        gc.collect()
        evaluation_logits = point.detach().requires_grad_(True)
        result = objective_probe(snapshot, evaluation_logits, train, validation, seed, checkpoint_steps)[0]
        scalar = result.detach().item()
        # Consume, then discard the exact graph. In Torch 2.0 the custom segment
        # retains CNN saved tensors across loss-only calls even after Python GC.
        # This additional derivative calculation does not update logits/parameters
        # or change J+/-; it releases saved tensors just like an optimizer backward.
        torch.autograd.grad(result, evaluation_logits)
        del result, evaluation_logits
        gc.collect()
        return scalar
    repeated = [value(logits) for _ in range(repeats)]
    noise = max(repeated) - min(repeated)
    gradient = torch.tensor(results[str(checkpoint_steps)]["gradient"], dtype=logits.dtype, device=logits.device)
    generator = torch.Generator().manual_seed(144)
    scans = []
    for index in range(directions):
        v = torch.randn(len(logits), generator=generator, dtype=torch.float64).to(logits.device)
        v -= v.mean()
        v /= v.norm()
        analytic = torch.dot(gradient, v).item()
        rows = []
        for h in h_values:
            plus, minus = value(logits + h * v), value(logits - h * v)
            rows.append(dict(h=h, plus_loss=plus, minus_loss=minus,
                             analytic=analytic, central_difference=(plus - minus) / (2 * h)))
            print(f"[MetaFDValue] dtype={snapshot.dtype} direction={index} h={h:g} plus={plus:.12g} minus={minus:.12g}", flush=True)
        assessment = fd_assessment(rows, noise, tolerances)
        scans.append(dict(direction=index, vector=v.cpu().tolist(), rows=rows, **assessment))
        print(f"[MetaFD] dtype={snapshot.dtype} direction={index} status={assessment['status']} intervals={assessment['convergence_intervals']}", flush=True)
    status = "failed" if any(r["status"] == "failed" for r in results.values()) else (
        "passed" if all(s["status"] == "passed" for s in scans) else "unconfirmed")
    sync(snapshot.device)
    return dict(checkpoint_comparison=results, repeated_losses=repeated, forward_noise_range=noise,
        directions=scans, actual_train_batches=len(train), dtype=str(snapshot.dtype),
        status=status, elapsed_seconds=time.perf_counter() - started)


def diagnose(problem, output, rounds=None, initializations=("uniform", "biased"),
             bias_client=0, bias_logit=.25, fd_epochs=1, directions=3, repeats=3,
             h_values=FD_STEPS, tolerances=None, raise_on_failure=True, fd_stage_steps=(0, 1, 8),
             forward_only=False, trace_branches=False):
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    if fd_epochs not in range(1, 6) or directions < 2 or repeats < 2 or len(h_values) < 3 or any(
            not math.isfinite(h) or h <= 0 for h in h_values) or list(h_values) != sorted(set(h_values), reverse=True):
        raise ValueError("FD requires 1..5 epochs, >=2 directions/repeats and >=3 decreasing positive distinct steps.")
    if not initializations or any(i not in ("uniform", "biased") for i in initializations) or not 0 <= bias_client < len(problem.client_ids) or not math.isfinite(bias_logit):
        raise ValueError("Invalid diagnostic candidate initialization.")
    selected = [s for s in problem.snapshots if rounds is None or s.metadata["round"] in rounds]
    if not selected or rounds is not None and set(rounds) != {s.metadata["round"] for s in selected}:
        raise ValueError("Requested diagnostic rounds are missing from the manifest.")
    report = dict(schema=2, status="running", configuration=configuration(problem),
        tolerances=tolerances, diagnostic_budget=dict(initializations=list(initializations),
            bias_client=bias_client, bias_logit=bias_logit, fd_epochs=fd_epochs,
            directions=directions, repeats=repeats, h_values=list(h_values)), snapshots=[])
    if any(s < 0 for s in fd_stage_steps):
        raise ValueError("FD stage steps must be nonnegative.")
    report["diagnostic_budget"]["fd_stage_steps"] = list(fd_stage_steps)
    report["diagnostic_budget"]["forward_only"] = forward_only
    report["diagnostic_budget"]["trace_branches"] = trace_branches
    write_json(output, report)
    for snapshot in selected:
        row = dict(source=source_signature(snapshot), candidates=[], status="running")
        report["snapshots"].append(row)
        try:
            with preserved_rng(), snapshot_backend(snapshot.metadata.get("backend"), override_for(problem.backend_policy)):
                row["effective_backend"] = backend_state()
                row["float32_matmul_precision"] = torch.get_float32_matmul_precision()
                seed = problem.virtual_seed + snapshot.metadata["round"] * 1009
                dtype = next(iter(snapshot.parameters().values())).dtype
                train = batch_plan(problem.train, 5, 16, seed, problem.device, dtype)
                validation = [(x.to(device=problem.device, dtype=dtype), y.to(problem.device))
                    for x, y in DataLoader(problem.validation, 16, shuffle=False,
                        generator=torch.Generator().manual_seed(seed + 1))]
                for initialization in initializations:
                    print(f"[MetaDiagnose] R{snapshot.metadata['round']} {initialization}", flush=True)
                    logits = torch.zeros(len(problem.client_ids), device=problem.device, dtype=torch.float64)
                    if initialization == "biased":
                        logits[bias_client] = bias_logit
                    candidate = dict(initialization=initialization, logits=logits.cpu().tolist(),
                                     weights=logits.softmax(0).cpu().tolist())
                    candidate["active_phase"] = "staged_forward"
                    row["candidates"].append(candidate)
                    with preserved_rng(seed):
                        full = snapshot_candidate(snapshot, logits)
                        candidate["trajectory"], staged_parameters = trajectory_probe(
                            snapshot.template(), full, train, validation, seed, tolerances, True, trace_branches)
                        ordinary_reference, ordinary_timing = ordinary_adaptation(
                            snapshot.template(), full, train, seed=seed)
                    del full
                    write_json(output, report)
                    gc.collect()
                    if torch.device(problem.device).type == "cuda":
                        torch.cuda.reset_peak_memory_stats(problem.device)
                    z = logits.detach().requires_grad_(True)
                    candidate["active_phase"] = "full_five_epoch"
                    sync(problem.device)
                    start = time.perf_counter()
                    loss, details, parameters = objective_probe(snapshot, z, train, validation, seed, problem.checkpoint_steps, True)
                    sync(problem.device)
                    forward_seconds = time.perf_counter() - start
                    staged_ce = candidate["trajectory"]["stages"][-1]["functional_validation_ce"]
                    staged_error = abs(staged_ce - loss.item())
                    staged_pd = difference(staged_parameters, parameters)
                    staged_rd = difference(reconstructed(staged_parameters, snapshot.template()),
                                           reconstructed(parameters, snapshot.template()))
                    staged_match = staged_error <= tolerances["ce_atol"] + tolerances["ce_rtol"] * abs(staged_ce) and within(
                        staged_rd, tolerances["reconstruction_atol"], tolerances["reconstruction_rtol"])
                    ordinary_comparison = compare_stage(ordinary_reference, _Objective(snapshot.template()),
                        parameters, {}, validation, len(train), tolerances)
                    start = time.perf_counter()
                    gradient = torch.autograd.grad(loss, z)[0]
                    sync(problem.device)
                    finite = bool(torch.isfinite(gradient).all())
                    candidate["full_five_epoch"] = dict(validation_ce=loss.item(), details=details,
                        gradient=gradient.detach().cpu().tolist() if finite else None,
                        gradient_norm=gradient.norm().item() if finite else None,
                        finite=finite, nonzero=finite and gradient.norm().item() > 0,
                        forward_seconds=forward_seconds, backward_seconds=time.perf_counter() - start,
                        ordinary_timing=ordinary_timing, ordinary_comparison=ordinary_comparison,
                        staged_ce_difference=staged_error, staged_parameter_difference=staged_pd,
                        staged_reconstruction_difference=staged_rd, staged_match=staged_match,
                        peak_allocated_cuda_bytes=torch.cuda.max_memory_allocated(problem.device)
                            if torch.device(problem.device).type == "cuda" else None)
                    del loss, gradient, parameters, staged_parameters, ordinary_reference
                    gc.collect()
                    write_json(output, report)
                    if forward_only:
                        failed_forward = any(s["status"] == "failed" for s in candidate["trajectory"]["stages"]) or ordinary_comparison["status"] == "failed" or not finite or not candidate["full_five_epoch"]["nonzero"] or not staged_match
                        candidate["status"] = "failed" if failed_forward else "unconfirmed"
                        candidate["reason"] = "Forward trajectory/full hypergradient only; FD/recomputation acceptance not performed."
                        continue
                    short = train[:fd_epochs * math.ceil(len(problem.train) / 16)]
                    candidate["active_phase"] = "short_planned_dtype"
                    candidate["short_planned_dtype"] = short_probe(snapshot, logits, short, validation, seed,
                        problem.checkpoint_steps, tolerances, h_values, directions, repeats)
                    if dtype != torch.float64:
                        candidate["active_phase"] = "short_float64_diagnostic"
                        double_snapshot = Snapshot(snapshot.folder, problem.device, torch.float64)
                        candidate["short_float64_diagnostic"] = short_probe(double_snapshot, logits,
                            [(x.double(), y) for x, y in short], [(x.double(), y) for x, y in validation],
                            seed, problem.checkpoint_steps, tolerances, h_values, directions, repeats)
                    candidate["stage_derivative_probes"] = {}
                    # Always distinguish decomposition and early training from an
                    # epoch-long derivative failure. These use the planned dtype.
                    for steps in sorted(set(fd_stage_steps)):
                        if steps > len(train) or steps == len(short):
                            continue
                        candidate["active_phase"] = f"stage_derivative_{steps}"
                        candidate["stage_derivative_probes"][str(steps)] = short_probe(snapshot, logits,
                            train[:steps], validation, seed, problem.checkpoint_steps,
                            tolerances, h_values, directions, repeats)
                    failed = any(s["status"] != "passed" for s in candidate["trajectory"]["stages"]) or ordinary_comparison["status"] != "passed" or not finite or not candidate["full_five_epoch"]["nonzero"] or not staged_match
                    short_states = [candidate["short_planned_dtype"]["status"], candidate.get("short_float64_diagnostic", {}).get("status", "passed")]
                    short_states += [s["status"] for s in candidate["stage_derivative_probes"].values()]
                    candidate["status"] = "failed" if failed or "failed" in short_states else "passed" if all(s == "passed" for s in short_states) else "unconfirmed"
                    candidate["active_phase"] = "complete"
                    write_json(output, report)
            states = [c["status"] for c in row["candidates"]]
            row["status"] = "failed" if "failed" in states else "passed" if all(s == "passed" for s in states) else "unconfirmed"
        except (RuntimeError, ValueError) as error:
            row.update(status="failed", error=str(error), approximation_used=False)
        write_json(output, report)
    states = [r["status"] for r in report["snapshots"]]
    report["status"] = "failed" if "failed" in states else "passed" if all(s == "passed" for s in states) else "unconfirmed"
    report["acceptance_scope"] = "synthetic_structure_only" if any(r["source"]["synthetic"] for r in report["snapshots"]) else "snapshot_and_configuration_bound"
    if forward_only:
        report["acceptance_scope"] = "incomplete_exploratory"
    report["interpretation"] = "A pass validates sampled numerical paths, not every future Adam iterate or universally effective weights."
    write_json(output, report)
    if raise_on_failure and report["status"] != "passed":
        raise RuntimeError(f"Numerical acceptance {report['status']}; inspect {output}. No formal optimization authorized by this report.")
    return report


def validate_acceptance(problem, path, initializations, bias_client, bias_logit):
    if path is None:
        raise ValueError("Formal optimization requires --acceptance-report from diagnose; finite/nonzero alone is insufficient.")
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("schema") != 2 or report.get("status") != "passed" or report.get("acceptance_scope") != "snapshot_and_configuration_bound":
        raise ValueError("Acceptance is failed, unconfirmed, synthetic or obsolete; no formal weight export.")
    if report.get("configuration") != configuration(problem):
        raise ValueError("Acceptance configuration/code/environment differs from optimization.")
    budget = report["diagnostic_budget"]
    if not set(initializations).issubset(budget["initializations"]) or "biased" in initializations and (
        bias_client != budget["bias_client"] or bias_logit != budget["bias_logit"]):
        raise ValueError("Acceptance did not cover requested initialization(s).")
    for snapshot in problem.snapshots:
        matching = [r for r in report["snapshots"] if r["source"] == source_signature(snapshot)]
        if len(matching) != 1 or matching[0]["status"] != "passed":
            raise ValueError(f"No passing acceptance for R{snapshot.metadata['round']} and its immutable files.")
    return dict(status="passed", report_sha256=file_hash(path), configuration=report["configuration"],
        acceptance_scope=report["acceptance_scope"], rounds=[s.metadata["round"] for s in problem.snapshots])
