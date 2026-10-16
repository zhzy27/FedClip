"""Observe actual calls/batches without replaying loaders or consuming RNG."""
import copy
import csv
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from common.metrics import sample_weighted_accuracy, summarize_evaluations

from .metadata import jsonable, tensor_digest, write_json


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone(),
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def set_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def rng_digest():
    state = rng_state()
    return dict(python=hashlib.sha256(pickle.dumps(state["python"])).hexdigest(),
                numpy=hashlib.sha256(pickle.dumps(state["numpy"])).hexdigest(),
                torch=tensor_digest(state["torch"]), cuda=[tensor_digest(s) for s in state["cuda"]])


def model_state(models):
    return {f"model_{k}.{n}": p.detach().cpu().clone() for k, m in enumerate(models) for n, p in m.state_dict().items()}


def summary(evaluations):
    if not evaluations:
        return {}
    result = summarize_evaluations(evaluations, value_key="test_acc_percent", unit="percent",
                                   evaluation_object="selected_post_local_clients")
    return dict(primary="sample-weighted selected post-local accuracy, percent", final=result["final"],
                best=result["best"], best_loop_round=result["best_loop_round"],
                best_completed_train_count=result["best_completed_train_count"],
                last10_evaluations=result["last10"], last20_evaluations=result["last20"],
                last10_evaluation_count=result["last10_count"], last20_evaluation_count=result["last20_count"],
                evaluation_object=result["evaluation_object"], accuracy_unit=result["unit"],
                evaluation_count=result["point_count"], averaging_unit="evaluation points, NOT consecutive training rounds")


class Recorder:
    def __init__(self, output, trace_batches=False, event_sink=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.trace_batches = trace_batches
        self.event_sink = event_sink
        self.round = -1
        self.stage = None
        self.client = None
        self.rows, self.evaluations = [], []
        self.selection_losses, self.tests, self.train_metrics = {}, {}, {}
        self.latest_weights = None
        self.event_count = 0

    def event(self, name, payload=None, models=None):
        record = dict(event=self.event_count, loop_round=self.round, name=name,
                      payload=jsonable(payload), rng=rng_digest())
        states = model_state(models) if models is not None and self.event_sink else None
        if models is not None:
            record["model_digests"] = {f"model_{k}.{n}": tensor_digest(p) for k, m in enumerate(models) for n, p in m.state_dict().items()}
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if self.event_sink:
            self.event_sink(record, states)
        self.event_count += 1

    def loader_class(self, base_cls=DataLoader):
        recorder = self
        # External reference calls this Torch-derived class directly. The migrated
        # client injects it into the public factory at each actual construction.
        class ObservedLoader(base_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.pass_index = 0
                original = self.batch_sampler
                loader = self
                class ObservedSampler:
                    def __len__(self):
                        return len(original)

                    def __iter__(self):
                        pass_index = loader.pass_index
                        loader.pass_index += 1
                        batches = []
                        for batch in original:
                            # Observe the real sampler's output; do NOT iterate twice.
                            batches.append([int(i) for i in batch])
                            yield batch
                        digest = hashlib.sha256(json.dumps(batches).encode()).hexdigest()
                        payload = dict(client_id=recorder.client, stage=recorder.stage, loader_pass=pass_index,
                                       batches=batches if recorder.trace_batches else None, batch_order_sha256=digest,
                                       batch_count=len(batches), sample_visits=sum(map(len, batches)),
                                       batch_size=loader.batch_size, drop_last=loader.drop_last, num_workers=loader.num_workers)
                        if recorder.stage == "train":
                            payload.update(model_index=pass_index // recorder.server.local_epochs,
                                           local_epoch=pass_index % recorder.server.local_epochs)
                        elif recorder.stage == "deployment_selection":
                            payload["models_evaluated_per_batch_in_order"] = list(range(recorder.server.K))
                        recorder.event("loader", payload)
                object.__setattr__(self, "batch_sampler", ObservedSampler())
        return ObservedLoader

    def attach(self, server):
        self.server = server
        self.event("initialization", dict(parameter_count=sum(p.numel() for p in server.global_model.parameters()),
                                         schema={n: [list(p.shape), str(p.dtype)] for n, p in server.global_model.state_dict().items()}), server.global_model_set)
        # Acceptance-only forward probe, without a draw or mode change.
        # A normal run does not need an extra pre-training forward.
        if self.event_sink:
            x = torch.linspace(-1, 1, 2 * 3 * 32 * 32, device=server.device).reshape(2, 3, 32, 32)
            with torch.no_grad():
                self.event("initial_forward", server.global_model_set[0](x))
        original_select = server.select_clients
        def select():
            self.round += 1
            self.selection_losses, self.tests, self.train_metrics = {}, {}, {}
            result = original_select()
            self.event("select_clients", [c.id for c in result])
            return result
        server.select_clients = select
        self._wrap(server, "send_models", lambda result: self.event("download", [c.id for c in server.clients]))
        for c in server.clients:
            self._wrap_client(c, "train", "train", lambda result, c=c: self.event("client_train", dict(client_id=c.id,
                              train_model_losses=c.train_model_losses, lr=[o.param_groups[0]["lr"] for o in c.optimizer_set]), c.model_set))
            def selection(result, c=c):
                self.selection_losses[c.id] = result[1].tolist()
                self.event("deployment_selection", dict(client_id=c.id, selected_model=result[0], endpoint_ce=result[1]))
            self._wrap_client(c, "select_best_model", "deployment_selection", selection)
            def test(result, c=c):
                self.tests[c.id] = list(result)
                self.event("test", dict(client_id=c.id, correct=result[0], samples=result[1], auc=result[2]))
            self._wrap_client(c, "test_metrics", "test", test)
            def train_metrics(result, c=c):
                self.train_metrics[c.id] = list(result)
                self.event("endpoint_train_metrics", dict(client_id=c.id, loss_sum=result[0], samples=result[1]))
            self._wrap_client(c, "train_metrics", "endpoint_train_metrics", train_metrics)
        def evaluated(result):
            ids = [c.id for c in server.clients]
            accuracy = sample_weighted_accuracy([self.tests[cid][0] for cid in ids],
                                               [self.tests[cid][1] for cid in ids], client_ids=ids)
            # The external source's own evaluate() is an independent oracle for
            # the common export calculation. Do not mask an export-layer defect.
            if accuracy != server.rs_test_acc[-1]:
                raise AssertionError("Collected-count accuracy differs from algorithm's own evaluation")
            row = dict(loop_round=self.round, completed_train_count=self.round + 1,
                       evaluation_object="selected_post_local_clients",
                       test_acc_percent=100 * accuracy, train_loss=server.rs_train_loss[-1],
                       total_test_correct=sum(x[0] for x in self.tests.values()), total_test_samples=sum(x[1] for x in self.tests.values()),
                       clients=[dict(client_id=c.id, selected_model=c.best_model_idx, deployment_ce=self.selection_losses[c.id],
                                     correct=self.tests[c.id][0], test_samples=self.tests[c.id][1],
                                     test_acc_percent=100 * self.tests[c.id][0] / self.tests[c.id][1],
                                     endpoint_train_loss_sum=self.train_metrics[c.id][0], endpoint_train_samples=self.train_metrics[c.id][1]) for c in server.clients],
                       selected_model_counts=np.bincount([c.best_model_idx for c in server.clients], minlength=server.K).tolist())
            self.evaluations.append(row)
            self.event("evaluation", row)
        self._wrap(server, "evaluate", evaluated)
        self._wrap(server, "receive_models", lambda result: self.event("receive", dict(uploaded_ids=server.uploaded_ids, sample_weights=server.uploaded_weights)))
        def weights(result, all_loss_values):
            self.latest_weights = (np.array(result[0], copy=True), np.array(result[1], copy=True))
            self.event("dual_weights", dict(sample_weighted_losses=all_loss_values, alpha=result[0], w=result[1]))
        self._wrap(server, "_compute_alpha_weights", weights, include_args=True)
        self._wrap(server, "aggregate_parameters", self.aggregated)

    @staticmethod
    def _wrap(obj, name, after, include_args=False):
        original = getattr(obj, name)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            after(result, *args, **kwargs) if include_args else after(result)
            return result
        setattr(obj, name, wrapped)

    def _wrap_client(self, client, method, stage, after):
        original = getattr(client, method)
        def wrapped(*args, **kwargs):
            previous = self.stage, self.client
            self.stage, self.client = stage, client.id
            try:
                result = original(*args, **kwargs)
                after(result)
                return result
            finally:
                self.stage, self.client = previous
        setattr(client, method, wrapped)

    def aggregated(self, result):
        s = self.server
        alpha, w = self.latest_weights
        distances = {}
        with torch.no_grad():
            for k in range(s.K):
                for j in range(k):
                    squared = sum((p.double() - q.double()).square().sum().item() for p, q in zip(s.global_model_set[k].parameters(), s.global_model_set[j].parameters()))
                    distances[f"{j}-{k}"] = squared ** .5
        clients = []
        for idx, cid in enumerate(s.uploaded_ids):
            clients.append(dict(client_id=cid, train_samples=s.clients[cid].train_samples,
                                sample_weight=s.uploaded_weights[idx], train_model_losses=s.clients[cid].train_model_losses,
                                alpha=float(alpha[idx, 0]), w=w[idx], actual_weights=alpha[idx, 0] * w[idx]))
        row = jsonable(dict(loop_round=self.round, completed_train_count=self.round + 1, uploaded_ids=s.uploaded_ids,
                            train_order=[c.id for c in s.selected_clients], clients=clients,
                            server_received_mass=(alpha * w).sum(axis=0), server_pairwise_l2=distances,
                            evaluation=self.evaluations[-1] if self.evaluations and self.evaluations[-1]["loop_round"] == self.round else None))
        self.rows.append(row)
        with (self.output / "rounds.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.event("aggregate", row, s.global_model_set)

    def export(self):
        write_json(self.output / "summary.json", summary(self.evaluations))
        write_json(self.output / "evaluations.json", self.evaluations)
        with (self.output / "evaluations.csv").open("w", newline="", encoding="utf-8") as f:
            columns = ["loop_round", "completed_train_count", "test_acc_percent", "total_test_correct", "total_test_samples", "train_loss", "evaluation_object"]
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.evaluations)
        with (self.output / "clients.csv").open("w", newline="", encoding="utf-8") as f:
            columns = ["loop_round", "completed_train_count", "client_id", "train_samples", "sample_weight", "train_model_losses", "alpha", "w", "actual_weights", "deployment_ce", "selected_model", "correct", "test_samples", "test_acc_percent"]
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                ev = {c["client_id"]: c for c in row["evaluation"]["clients"]} if row["evaluation"] else {}
                for client in row["clients"]:
                    writer.writerow(dict(loop_round=row["loop_round"], completed_train_count=row["completed_train_count"], **client,
                                         **{k: v for k, v in ev.get(client["client_id"], {}).items() if k != "client_id"}))
