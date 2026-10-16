"""Real local head/backbone alternation; no change to the ordinary joint path."""

import copy
import time

import torch
from torch.utils.data import DataLoader

from flcore.clients.clientbase import save_item
from utils.target_head import classifier_names, preserved_head_rng
from utils.target_local_schedule import validate_local_schedule


def _sync(client):
    if str(client.device).startswith("cuda"):
        torch.cuda.synchronize(client.device)


@torch.no_grad()
def observe_endpoint(client, model, stage):
    """Infer on a private copy; never write an intermediate checkpoint.

    Copying protects parameters, buffers and mixed module modes even if an
    inference module has side effects. All loader/copy/inference RNG is restored.
    """
    start = time.perf_counter()
    with preserved_head_rng():
        candidate = copy.deepcopy(model)
        candidate.eval()
        dataset = client.load_train_data().dataset
        trainloader = DataLoader(dataset, batch_size=client.batch_size, shuffle=False, drop_last=False,
                                 generator=torch.Generator().manual_seed(0))
        result = dict(stage=stage, endpoint_loss_scope="CE_only_no_F_or_Prox")
        for label, loader in (("train", trainloader), ("test", client.load_test_data())):
            losses, correct, count = 0., 0, 0
            for x, y in loader:
                x, y = client._move_batch(x, y)
                output = candidate(x)
                losses += float(client.loss(output, y)) * len(y)
                correct += int((output.argmax(1) == y).sum())
                count += len(y)
            if not count:
                raise ValueError("FedRep C0 observations require nonempty original train/test data.")
            result.update({f"endpoint_{label}_ce": losses / count, f"endpoint_{label}_accuracy": correct / count,
                           f"endpoint_{label}_correct": correct, f"endpoint_{label}_samples": count})
        _sync(client)
    result["observation_seconds"] = time.perf_counter() - start
    return result


def run_phase(client, model, trainloader, parameters, stage, epochs):
    optimizer = torch.optim.SGD(parameters, lr=client.learning_rate, momentum=0., weight_decay=0.)
    _sync(client)
    start = time.perf_counter()
    ce_sum, f_sum, visits, batches = 0., 0., 0, 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = client._move_batch(x, y)
            optimizer.zero_grad()
            if stage == "head":
                ce = client.loss(model(x), y)
                weighted_f = ce.new_zeros(())
            else:
                # Frozen fc3 still propagates CE gradients to backbone features.
                ce, weighted_f = client._objective(model, x, y)
            total = ce + weighted_f
            if not torch.isfinite(total):
                raise ValueError(f"Non-finite FedRep {stage} loss.")
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
            optimizer.step()
            ce_sum += float(ce.detach()) * len(y)
            f_sum += float(weighted_f.detach()) * len(y)
            visits += len(y)
            batches += 1
    _sync(client)
    if visits == 0:
        raise ValueError("FedRep-style training requires nonempty full local epochs.")
    return dict(stage=stage, train_epochs=epochs, train_batches=batches, train_sample_visits=visits,
        train_mean_ce=ce_sum / visits, train_mean_weighted_frobenius=f_sum / visits,
        train_seconds=time.perf_counter() - start, train_loss_scope="CE_only" if stage == "head" else "CE_plus_weighted_F",
        learning_rate=client.learning_rate, momentum=0., weight_decay=0., grad_clip_norm=10.,
        optimizer_parameter_count=sum(p.numel() for p in parameters))


def train_fedrep(client, current_round):
    validate_local_schedule(client.args)
    start = time.perf_counter()
    model = client._load_model()
    original_flags = [(p, p.requires_grad) for p in model.parameters()]
    original_modes = [(module, module.training) for module in model.modules()]
    phases, endpoints = [], []
    try:
        names = classifier_names(model)  # Validate BEFORE temporarily freezing fc3.
        named = dict(model.named_parameters())
        head_ids = {id(named[n]) for n in names}
        head = [p for p, trainable in original_flags if trainable and id(p) in head_ids]
        backbone = [p for p, trainable in original_flags if trainable and id(p) not in head_ids]
        if len(head) != 2 or not backbone:
            raise ValueError("FedRep-style training requires two trainable fc3 parameters and a trainable backbone.")
        trainloader = client.load_train_data()
        if not isinstance(trainloader.sampler, torch.utils.data.RandomSampler) or trainloader.drop_last:
            raise ValueError("FedRep-style training requires shuffle=True and drop_last=False.")
        if client.id == 0:
            endpoints.append(observe_endpoint(client, model, "downloaded_before_head"))
        # Head stage: frozen backbone, including buffers and the aliased ReLU.
        for p, _ in original_flags:
            p.requires_grad_(id(p) in head_ids)
            p.grad = None
        model.eval()
        model.fc3.train()
        phases.append(run_phase(client, model, trainloader, head, "head", getattr(client.args, "fedrep_head_epochs", 5)))
        if client.id == 0:
            endpoints.append(observe_endpoint(client, model, "after_head"))
        # The SAME model continues immediately. No download/SVD/head reset here.
        for p, trainable in original_flags:
            p.requires_grad_(trainable and id(p) not in head_ids)
            p.grad = None
        model.train()
        model.fc3.eval()
        phases.append(run_phase(client, model, trainloader, backbone, "backbone", client.local_epochs))
        if client.id == 0:
            endpoints.append(observe_endpoint(client, model, "after_backbone_primary"))
    finally:
        for p, trainable in original_flags:
            p.requires_grad_(trainable)
            p.grad = None
        for module, training in original_modes:
            module.training = training
    save_item(model, client.role, "model", client.save_folder_name)
    elapsed = time.perf_counter() - start
    visits = sum(row["train_sample_visits"] for row in phases)
    train_seconds = sum(row["train_seconds"] for row in phases)
    client.train_time_cost["num_rounds"] += 1
    client.train_time_cost["total_cost"] += elapsed
    client.last_train_time_cost = elapsed
    client.last_normal_train_stats = dict(local_epochs=10, head_epochs=5, backbone_epochs=5, total_data_epochs=10,
        train_batches=sum(row["train_batches"] for row in phases), processed_train_samples=visits,
        ce_loss=sum(row["train_mean_ce"] * row["train_sample_visits"] for row in phases) / visits,
        regularization_loss=sum(row["train_mean_weighted_frobenius"] * row["train_sample_visits"] for row in phases) / visits,
        proximal_loss=0., train_seconds=train_seconds, loss_scope="mean_over_all_10_data_epochs")
    client.last_fedrep_train_stats = dict(round=current_round + 1, loop_round=current_round, client_id=client.id,
        target_local_schedule="fedrep", head_epochs=5, backbone_epochs=5, total_data_epochs=10,
        head_train_batches=phases[0]["train_batches"], backbone_train_batches=phases[1]["train_batches"],
        head_train_sample_visits=phases[0]["train_sample_visits"], backbone_train_sample_visits=phases[1]["train_sample_visits"],
        head_train_mean_ce=phases[0]["train_mean_ce"], backbone_train_mean_ce=phases[1]["train_mean_ce"],
        backbone_train_mean_weighted_frobenius=phases[1]["train_mean_weighted_frobenius"],
        head_train_seconds=phases[0]["train_seconds"], backbone_train_seconds=phases[1]["train_seconds"],
        optimizer_train_seconds=train_seconds, endpoint_observation_seconds=sum(row["observation_seconds"] for row in endpoints),
        total_local_call_seconds=elapsed, phases=phases, endpoints=endpoints)
    for row in phases:
        print(f"[FedRepStage] round={current_round + 1} client={client.id} " + " ".join(f"{k}={v}" for k, v in row.items()))
    for row in endpoints:
        print(f"[FedRepC0Endpoint] round={current_round + 1} " + " ".join(f"{k}={v}" for k, v in row.items()))
    print(f"[FedRepLocal] round={current_round + 1} client={client.id} head_epochs=5 backbone_epochs=5 total_data_epochs=10 total_seconds={elapsed:.3f}")
    return elapsed
