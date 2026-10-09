"""Run matched Client-0 low-rank CNN controls, each in a fresh Python process."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
from queue import Queue
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.projection_self_weight import validate_projection_self_weight, self_weight_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100, help="Inherited main.py -gr value.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--device-ids", nargs="+", help="GPU pool; at most one experiment per GPU at a time.")
    parser.add_argument("--model-family", default="Decom_CNN-5-512")
    parser.add_argument("--modes", nargs="+", choices=["avg", "target_only", "projection", "layer_mask", "layer_mask_budget", "layer_softmax", "layer_relu", "projection_local", "layer_projection_global", "layer_projection_local", "projection_same_label", "projection_cross_label", "projection_softmax", "projection_relu", "softmax_only", "apa", "apa_logit", "dwa_soft", "dwa_soft_projection", "dwa_adaptive_self", "dwa_adaptive_self_projection", "meta_projection_fixed"],
                        default=["avg", "target_only", "projection", "layer_mask"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--projection_self_weight", type=float, default=None)
    parser.add_argument("--meta_c0_split", default="")
    parser.add_argument("--meta_collect_snapshots", action="store_true")
    parser.add_argument("--meta_snapshot_rounds", default="20,50,80,100")
    parser.add_argument("--meta_snapshot_dir", default="")
    parser.add_argument("--meta_weight_file", default="")
    parser.add_argument("--dwa_distance_eps", type=float, default=1e-12)
    parser.add_argument("--apa_logit_lr", type=float, default=0.01)
    parser.add_argument("--apa_server_lr", type=float, default=0.01)
    parser.add_argument("--apa_momentum", type=float, default=0.9)
    parser.add_argument("--apa_self_weight", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    parser.add_argument("--parallel", action="store_true", help="Run selected modes concurrently in separate processes.")
    options = parser.parse_args()
    if options.projection_self_weight is not None:
        try:
            for mode in options.modes:
                validate_projection_self_weight(options.projection_self_weight, mode, options.meta_c0_split)
        except ValueError as error:
            parser.error(str(error))
    if options.rounds < 1:
        parser.error("--rounds must be at least 1")
    device_ids = options.device_ids or [options.device_id]
    if len(set(device_ids)) != len(device_ids) or any(
            not device.isdecimal() or str(int(device)) != device for device in device_ids):
        parser.error("GPU IDs must be distinct non-negative integers (one physical GPU per ID).")
    if len(set(options.modes)) != len(options.modes):
        parser.error("Each mode may appear only once per launch.")
    if len(set(options.seeds)) != len(options.seeds) or any(seed < 0 for seed in options.seeds):
        parser.error("Training seeds must be distinct non-negative integers.")
    if "meta_projection_fixed" in options.modes and not (options.meta_weight_file and options.meta_c0_split):
        parser.error("Frozen meta mode requires --meta_weight_file and --meta_c0_split.")
    system_dir = Path(__file__).resolve().parent
    data_dir = system_dir.parent / "dataset" / "Cifar100" / "pat_20"
    missing = [str(data_dir / split / f"{cid}.npz")
               for split in ("train", "test") for cid in range(20)
               if not (data_dir / split / f"{cid}.npz").is_file()]
    if missing and not options.dry_run:
        parser.error(
            f"Missing {len(missing)} CIFAR-100 pat_20 shards. From dataset/ run: "
            "python generate_Cifar100.py --niid 1 --partition pat --cpc 20"
        )
    common = [
        sys.executable, "-u", "main.py", "-algo", "FedTargetProj",
        "-t", "1", "--seed", "0", "--target_client_id", "0",
        "-data", "Cifar100", "-ncl", "100", "-nc", "20",
        "-niid", "1", "-pt", "pat", "-cpc", "20", "-jr", "1.0",
        "-m", options.model_family, "-lr", "0.005", "-lbs", "16",
        "-ls", "5", "-gr", str(options.rounds), "-eg", "1",
        "-is_regular", "1", "-regular_lamda", "1e-3",
        "-dev", options.device, "-did", options.device_id,
    ]
    root = system_dir / "target_proj_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    jobs = []
    available_devices = Queue()
    for device_id in device_ids:
        available_devices.put(device_id)

    def train_job(job):
        command, folder = job
        device_id = available_devices.get() if options.device == "cuda" else options.device_id
        try:
            command[command.index("-did") + 1] = device_id
            print(subprocess.list2cmdline(command), flush=True)
            (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"Training {folder.name} on {options.device}:{device_id}; log: {folder / 'train.log'}", flush=True)
            with (folder / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=system_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            if options.device == "cuda":
                available_devices.put(device_id)

    combinations = [(seed, mode) for seed in options.seeds for mode in options.modes]
    for index, (seed, mode) in enumerate(combinations):
        run_name = mode if options.projection_self_weight is None else f"{mode}_{self_weight_tag(options.projection_self_weight)}"
        folder = root / run_name if options.seeds == [0] else root / f"seed{seed}" / run_name
        command = common + [
            "--target_proj_mode", mode, "-exp_name", f"target0_seed{seed}_{run_name}",
            "-sfn", str(folder / "checkpoints"),
            "--h5_result_root", str(folder / "h5_results"),
            "--final-model-root", str(folder / "final_models"),
        ]
        command[command.index("--seed") + 1] = str(seed)
        if options.projection_self_weight is not None:
            command += ["--projection_self_weight", str(options.projection_self_weight)]
        if options.meta_c0_split:
            command += ["--meta_c0_split", str(Path(options.meta_c0_split).resolve())]
        if mode == "meta_projection_fixed":
            command += ["--meta_weight_file", str(Path(options.meta_weight_file).resolve())]
        if options.meta_collect_snapshots and mode == "projection":
            command += ["--meta_collect_snapshots", "--meta_snapshot_rounds", options.meta_snapshot_rounds]
            if options.meta_snapshot_dir:
                snapshot_dir = Path(options.meta_snapshot_dir).resolve()
                if options.seeds != [0]:
                    snapshot_dir /= f"seed{seed}"
                command += ["--meta_snapshot_dir", str(snapshot_dir)]
        if mode == "apa":
            command += ["--apa_server_lr", str(options.apa_server_lr),
                        "--apa_momentum", str(options.apa_momentum),
                        "--apa_self_weight", str(options.apa_self_weight)]
        if mode == "apa_logit":
            command += ["--apa_logit_lr", str(options.apa_logit_lr)]
        if mode in ("dwa_soft", "dwa_soft_projection", "dwa_adaptive_self", "dwa_adaptive_self_projection"):
            command += ["--dwa_distance_eps", str(options.dwa_distance_eps)]
        if options.dry_run:
            command[command.index("-did") + 1] = device_ids[index % len(device_ids)]
            print(subprocess.list2cmdline(command), flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=False)
        if options.parallel:
            jobs.append((command, folder))
        else:
            train_job((command, folder))
    if jobs:
        workers = min(len(jobs), len(device_ids)) if options.device == "cuda" else len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(train_job, jobs))
    if not options.dry_run:
        print(f"Completed matched controls: {root}")


if __name__ == "__main__":
    main()
