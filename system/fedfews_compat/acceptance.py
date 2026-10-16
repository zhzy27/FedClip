"""Controlled ORIGINAL-source vs migrated acceptance; never a long run."""
import argparse
import contextlib
import copy
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .config import CONFIGS, parse_args
from .data import audit_data
from .metadata import environment, migration_info, shared_contract, source_info, write_json
from .observe import rng_state, set_rng
from .reference import load_reference, reference_defaults, reference_config
from .runtime import build_server, migrated_backend, prepare_random
from .vendor.arguments import source_parser


def synthetic_data(root, clients=3):
    """Only synthetic acceptance fixtures, never writes an existing shard."""
    root = Path(root)
    if root.exists():
        raise FileExistsError(root)
    dataset = "Cifar100/noniid_dir_20_a0p5"
    directory = root / dataset
    rng = np.random.RandomState(2718)
    for split, base in [("train", 7), ("test", 3)]:
        (directory / split).mkdir(parents=True)
        for cid in range(clients):
            count = base + cid * 3
            x = rng.normal(size=(count, 3, 32, 32)).astype(np.float32)
            y = (np.arange(count) + cid * 7) % 100
            # Exercise both original image paths, without changing pixel values.
            data = dict(x=list(x) if cid % 3 == 1 else x, y=y)
            if cid % 2 == 0:
                with (directory / split / f"{cid}.pkl").open("wb") as stream:
                    pickle.dump(data, stream)
                # A deliberately stale npz proves pkl preference on BOTH sides.
                np.savez_compressed(directory / split / f"{cid}.npz", data=dict(x=np.zeros_like(x), y=np.full_like(y, 99)))
            else:
                np.savez_compressed(directory / split / f"{cid}.npz", data=data)
    write_json(directory / "config.json", dict(synthetic=True, alpha=None, num_clients=clients,
               description="Unequal counts; not real CIFAR nor a Dirichlet partition"))
    return dataset


def torch_load(path):
    # Explicit for Torch 2.6+ default weights_only=True; these are our own local traces.
    return torch.load(path, map_location="cpu", weights_only=False)


class Comparison:
    def __init__(self, directory, atol=0., rtol=0.):
        self.directory = Path(directory)
        self.directory.mkdir()
        self.atol, self.rtol = atol, rtol
        self.count = 0
        self.failures = []
        self.max_parameter_error = 0.
        self.max_parameter_location = None
        self.max_scalar_error = 0.
        self.compared_parameters = 0
        self.seen = set()
        self.first_parameter_mismatch = None

    def reference(self, record, states):
        torch.save(dict(record=record, states=states), self.directory / f"event_{record['event']:05d}.pt")

    def migrated(self, record, states):
        path = self.directory / f"event_{record['event']:05d}.pt"
        if not path.exists():
            self.failures.append(f"Unexpected migrated event {record['event']}: {record['name']}")
            return
        ref = torch_load(path)
        self.seen.add(record["name"])
        self.compare_value(ref["record"]["name"], record["name"], f"event {record['event']}.name")
        self.compare_value(ref["record"]["loop_round"], record["loop_round"], f"event {record['event']}.round")
        self.compare_value(ref["record"]["rng"], record["rng"], f"event {record['event']}.RNG")
        self.compare_value(ref["record"]["payload"], record["payload"], f"event {record['event']} {record['name']}.payload")
        if ref["states"] is not None:
            if states is None or ref["states"].keys() != states.keys():
                self.failures.append(f"event {record['event']} parameter keys differ")
            else:
                for name, expected in ref["states"].items():
                    actual = states[name]
                    self.compared_parameters += actual.numel()
                    if expected.shape != actual.shape or expected.dtype != actual.dtype:
                        self.failures.append(f"event {record['event']}.{name} shape/dtype")
                        continue
                    error = (expected.double() - actual.double()).abs()
                    maximum = error.max().item() if error.numel() else 0.
                    if maximum > self.max_parameter_error or self.max_parameter_location is None:
                        self.max_parameter_error = maximum
                        self.max_parameter_location = f"event {record['event']} {record['name']} {name}, flat index {error.flatten().argmax().item()}"
                    if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=self.atol, rtol=self.rtol):
                        self.failures.append(f"event {record['event']}.{name}: max_abs={maximum}")
                        if self.first_parameter_mismatch is None:
                            path = self.directory.parent / "first_parameter_mismatch.pt"
                            torch.save(dict(reference_record=ref["record"], actual_record=record,
                                            reference_states=ref["states"], actual_states=states,
                                            parameter=name, max_abs=maximum), path)
                            self.first_parameter_mismatch = str(path)
        self.count += 1

    def compare_value(self, ref, actual, where):
        if isinstance(ref, dict):
            if not isinstance(actual, dict) or ref.keys() != actual.keys():
                self.failures.append(where + " keys differ")
            else:
                for key in ref:
                    self.compare_value(ref[key], actual[key], where + "." + str(key))
        elif isinstance(ref, list):
            if not isinstance(actual, list) or len(ref) != len(actual):
                self.failures.append(where + " lengths differ")
            else:
                for i, (a, b) in enumerate(zip(ref, actual)):
                    self.compare_value(a, b, where + f"[{i}]")
        elif isinstance(ref, float):
            error = abs(ref - actual)
            self.max_scalar_error = max(self.max_scalar_error, error)
            if not np.isfinite(actual) or error > self.atol + self.rtol * abs(ref):
                self.failures.append(f"{where}: {ref} != {actual}")
        elif ref != actual:
            self.failures.append(f"{where}: {ref!r} != {actual!r}")

    def report(self):
        expected = len(list(self.directory.glob("event_*.pt")))
        if expected != self.count:
            self.failures.append(f"Event count: reference {expected}, migrated {self.count}")
        required = {"initialization", "initial_forward", "client_train", "deployment_selection", "test",
                    "endpoint_train_metrics", "receive", "dual_weights", "aggregate", "loader", "evaluation"}
        if required - self.seen:
            self.failures.append(f"Missing acceptance stages: {required - self.seen}")
        return dict(status="PASS" if not self.failures else "FAIL", event_count=self.count,
                    parameter_elements_compared=self.compared_parameters,
                    max_parameter_error=self.max_parameter_error, max_parameter_location=self.max_parameter_location,
                    max_scalar_error=self.max_scalar_error, atol=self.atol, rtol=self.rtol,
                    first_parameter_mismatch=self.first_parameter_mismatch,
                    rng_checks="Python, NumPy, Torch CPU and all initialized CUDA devices at every observed event",
                    failures=self.failures)


def _run_acceptance(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Independent original FedFew Git checkout")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", help="Omit to use synthetic unequal-count fixtures")
    parser.add_argument("--dataset", default="Cifar100/noniid_dir_20_a0p5")
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=4, help="Actual training count, must be 2..5")
    parser.add_argument("--eval-gap", type=int, default=2)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--expected-alpha", type=float)
    parser.add_argument("--expected-partition", choices=["dir", "pat"])
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--rtol", type=float, default=0.)
    parser.add_argument("--reference-repeat", action="store_true",
                        help="Diagnostic source-vs-source repeatability, NOT migration acceptance")
    parser.add_argument("--diagnostic-cudnn-deterministic", action="store_true",
                        help="Short diagnostic only: set cuDNN deterministic for BOTH endpoints; never changes formal defaults")
    opts = parser.parse_args(argv)
    if not 2 <= opts.rounds <= 5 or opts.atol < 0 or opts.rtol < 0 or not np.isfinite([opts.atol, opts.rtol]).all():
        parser.error("Require 2..5 actual rounds and finite nonnegative tolerances")
    output = Path(opts.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.device_id
    clients = opts.num_clients or (20 if opts.data_root else 3)
    root = Path(opts.data_root).resolve() if opts.data_root else output / "synthetic_data"
    dataset = opts.dataset if opts.data_root else synthetic_data(root, clients)
    args, config = parse_args(["--data-root", str(root), "--output-dir", str(output / "unused"),
                 "--dataset", dataset, "--num_clients", str(clients), "--batch_size", str(opts.batch_size or (50 if opts.data_root else 4)),
                 "--global_rounds", str(opts.rounds - 1), "--eval_gap", str(opts.eval_gap),
                 "--local_epochs", str(opts.local_epochs), "--device", opts.device, "--device_id", opts.device_id,
                 "--rng-mode", "controlled", "--control-seed", str(opts.control_seed), "--torch-threads", str(opts.torch_threads),
                 "--trace-batches", "--source-root", str(Path(opts.source_root).resolve())] +
                 (["--expected-alpha", str(opts.expected_alpha)] if opts.expected_alpha is not None else []) +
                 (["--expected-partition", opts.expected_partition] if opts.expected_partition else []))
    source = source_info(args.source_root)
    reference = load_reference(args.source_root)
    defaults = reference_defaults(args.source_root)
    migrated_defaults = vars(source_parser().parse_args([]))
    if defaults != migrated_defaults:
        write_json(output / "default_mismatch.json", dict(reference=defaults, migrated=migrated_defaults))
        raise RuntimeError("Source parser defaults differ; review migration before accepting")
    source_yaml, source_effective, source_command = reference_config(args.source_root, output)
    default_args, _ = parse_args(["--data-root", str(root), "--output-dir", str(output / "config_preview")])
    if source_yaml != config:
        raise RuntimeError("Upstream base + algorithm YAML merge differs")
    for key, value in source_effective.items():
        if key not in {"model_dir", "result_dir"} and getattr(default_args, key) != value:
            raise RuntimeError(f"Upstream effective YAML/default mismatch: {key}: {value!r} != {getattr(default_args, key)!r}")
    ref_data = audit_data(args, reference.reader)
    migrated_data = audit_data(args)
    if ref_data != migrated_data:
        raise AssertionError("Source and migrated decoded x/y contents/order or file selection differ")
    write_json(output / "data_manifest.json", migrated_data)
    (output / "data_config.json").write_bytes(Path(migrated_data["config_path"]).read_bytes())
    if opts.diagnostic_cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    prepare_random(args)
    if opts.device == "cuda":
        # Initialize once before sharing the exact initial CPU/CUDA states.
        torch.empty(0, device="cuda")
    initial = rng_state()
    torch.save(initial, output / "initial_rng.pt")
    comparison = Comparison(output / "reference_states", opts.atol, opts.rtol)
    manifests = []
    second = ("source_repeat", reference, comparison.migrated) if opts.reference_repeat else ("migrated", migrated_backend(), comparison.migrated)
    for name, backend, sink in [("source", reference, comparison.reference), second]:
        set_rng(initial)
        started = time.perf_counter()
        with (output / f"{name}.log").open("w", encoding="utf-8") as logfile, contextlib.redirect_stdout(logfile):
            server, recorder, cleanup = build_server(args, output / name, backend, sink)
            try:
                server.train()  # Original external source's train() vs migrated train().
                # Include persisted model/optimizer/scheduler and final RNG in parity.
                torch.save(rng_state(), output / f"{name}_final_rng.pt")
                manifests.append(dict(backend=name, seconds=time.perf_counter() - started,
                                      evaluations=[r["loop_round"] for r in recorder.evaluations],
                                      non_evaluation_rounds=[r["loop_round"] for r in recorder.rows if r["evaluation"] is None]))
            finally:
                cleanup()
        # Break observation closures before constructing the other backend.
        del server, recorder, cleanup, backend
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    comparison.compare_value(manifests[0]["evaluations"], manifests[1]["evaluations"], "evaluation schedule")
    for item in manifests:
        if not item["non_evaluation_rounds"]:
            comparison.failures.append("No non-evaluation branch covered; increase eval-gap/rounds")
    report = comparison.report()
    report.update(source=source, migration=migration_info(), shared_modules=shared_contract(args),
                  environment=environment(), effective_args=vars(args), runs=manifests,
                  acceptance_options=vars(opts),
                  command=[sys.executable] + (sys.argv if argv is None else ["system/verify_fedfews.py"] + list(argv)),
                  comparison_kind="external_source_repeatability" if opts.reference_repeat else "external_source_vs_migration",
                  acceptance_scope="diagnostic_cudnn_deterministic" if opts.diagnostic_cudnn_deterministic else "unchanged_backend_settings",
                  source_merged_yaml=source_yaml, source_effective_args=source_effective, source_launch_command=source_command,
                  dataset_kind="real shards" if opts.data_root else "synthetic unequal-count images, actual upstream CNN",
                  reference_adapter="External original files; imports isolated; original train/eval/aggregate executed; output hooks only",
                  reference_implementation=dict(reader=str(Path(args.source_root) / "PFLlib/system/utils/data_utils.py"),
                                                model=str(Path(args.source_root) / "PFLlib/system/flcore/trainmodel/models.py") + ":FedAvgCNN",
                                                loader="external original Client loader constructor, observed Torch DataLoader",
                                                common_reader_model_loader_used=False),
                  scope="Short-path behavioral acceptance; does not establish historical curve reproduction or accuracy gains")
    write_json(output / "acceptance_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


def main(argv=None):
    previous = torch.backends.cudnn.deterministic
    try:
        return _run_acceptance(argv)
    finally:
        # Diagnostic settings are process-local and must not leak to a caller.
        torch.backends.cudnn.deterministic = previous


if __name__ == "__main__":
    main()
