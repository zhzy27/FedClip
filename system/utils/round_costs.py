"""Exclusive round timings and algorithm payload sizes for sequential FL simulation.

All durations are synchronized wall times. File serialization/deserialization,
console writes, evaluation and diagnostics are suspended from the active timer.
MB denotes 1,000,000 bytes of tensor/array payload (no transport metadata).
"""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from contextvars import ContextVar
import sys
import time

import numpy as np
import torch


_ACTIVE = ContextVar("round_cost_recorder", default=None)
UPLOAD_ITEMS = {
    "FedCLIP": "model", "FD": "logits", "FedProto": "protos",
    "FedGH": "protos", "FedTGP": "protos", "FedKD": "compressed_param",
    "FML": "global_model", "FedMRL": "global_model", "PFedAFM": "global_model",
    "LG-FedAvg": "model", "FedGen": "model", "FedSPU": "updated_parameters",
}


def payload_bytes(value):
    """Count logical array payload, including dtype, without copying to CPU."""
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(payload_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(payload_bytes(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return 0
    raise TypeError(f"Unsupported upload object: {type(value).__name__}")


def upload_payload(algorithm, item_name, item):
    if UPLOAD_ITEMS[algorithm] != item_name or item is None:
        return None
    if algorithm in ("LG-FedAvg", "FedGen"):
        return list(item.head.parameters())
    if isinstance(item, torch.nn.Module):
        # These server implementations average named parameters, not buffers.
        return list(item.parameters())
    return item


@contextmanager
def exclude_measurement():
    recorder = _ACTIVE.get()
    if recorder is None:
        yield
    else:
        with recorder.scope(None):
            yield


def observe_loaded_upload(role, item_name, item):
    recorder = _ACTIVE.get()
    if recorder is not None and role.startswith("Client_"):
        recorder.observe_upload(int(role[len("Client_"):]), item_name, item)


class _OutputWithoutTiming:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        with exclude_measurement():
            return self.stream.write(value)

    def flush(self):
        with exclude_measurement():
            return self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class RoundCosts:
    def __init__(self, algorithm, device, clock=None, synchronize=None):
        if algorithm not in UPLOAD_ITEMS:
            raise ValueError(f"Round cost accounting is not audited for {algorithm}.")
        self.algorithm = algorithm
        self.device = device
        self.clock = clock or time.perf_counter
        self.synchronize = synchronize or self._synchronize
        self.current = None
        self.records = []
        self.state = None
        self.started = None
        self.local = {}
        self.events = {}
        self.uploads = {}

    def _synchronize(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _stop(self):
        if self.started is not None:
            self.synchronize()
            elapsed = self.clock() - self.started
            category, event, client_id = self.state
            if category == "local":
                self.local[client_id] = self.local.get(client_id, 0.0) + elapsed
            else:
                self.events[event] = self.events.get(event, 0.0) + elapsed
            self.started = None

    def _start(self):
        if self.current is not None and self.state is not None:
            self.synchronize()
            self.started = self.clock()

    @contextmanager
    def scope(self, category, event="server_other", client_id=None):
        state = None if category is None else (category, event, client_id)
        previous = self.state
        # Avoid charging nested pauses twice, or timing their bookkeeping.
        if previous == state:
            yield
            return
        self._stop()
        self.state = state
        self._start()
        try:
            yield
        finally:
            self._stop()
            self.state = previous
            self._start()

    @contextmanager
    def session(self):
        token = _ACTIVE.set(self)
        try:
            with redirect_stdout(_OutputWithoutTiming(sys.stdout)), \
                    redirect_stderr(_OutputWithoutTiming(sys.stderr)):
                with self.scope("server"):
                    yield
        finally:
            _ACTIVE.reset(token)

    def begin_round(self, round_idx, client_ids):
        if self.current is not None:
            raise RuntimeError("Finish the preceding round before starting another.")
        self.current = {"round": int(round_idx), "selected_client_ids": list(client_ids)}
        self.local, self.events, self.uploads = {}, {}, {}
        self._start()

    def observe_upload(self, client_id, item_name, item):
        if (self.current is None or self.state is None or self.state[0] != "server"
                or self.state[1] in ("send_parameters", "send_select_client_parameters")):
            return
        key = (client_id, item_name)
        if key in self.uploads:
            return
        with self.scope(None):
            payload = upload_payload(self.algorithm, item_name, item)
            if payload is not None:
                self.uploads[key] = payload_bytes(payload)

    def finish_round(self):
        if self.current is None:
            return None
        self._stop()
        record = self.current
        record.update({
            "local_train_sum_seconds": sum(self.local.values()),
            "local_train_max_seconds": max(self.local.values(), default=0.0),
            "server_processing_seconds": sum(self.events.values()),
            "upload_payload_bytes": sum(self.uploads.values()),
            "upload_payload_mb": sum(self.uploads.values()) / 1e6,
            "local_train_client_seconds": {str(k): v for k, v in sorted(self.local.items())},
            "server_processing_events": dict(sorted(self.events.items())),
            "upload_client_details": [
                {"client_id": cid, "item": item, "bytes": count, "mb": count / 1e6}
                for (cid, item), count in sorted(self.uploads.items())
            ],
        })
        self.records.append(record)
        self.current = None
        return record
