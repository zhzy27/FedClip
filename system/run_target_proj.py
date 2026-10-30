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
from utils.target_group_options import GROUP_MODES, validate_group_options, group_tag
from utils.training_logs import archive_logs
from utils.target_prox_options import validate_target_prox, prox_tag
from utils.feedback_options import FEEDBACK_MODES, DEFERRED_FEEDBACK_MODES, feedback_tag, validate_feedback_options
from utils.target_head import validate_target_head
from utils.head_diagnostic_options import validate_head_diagnostic
from utils.target_local_schedule import validate_local_schedule, schedule_tag
from utils.target_head_l2 import add_head_l2_options, validate_head_l2, head_l2_tag
from utils import target_z2
from utils import target_c0_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100, help="Inherited main.py -gr value.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--device-ids", nargs="+", help="GPU pool; at most one experiment per GPU at a time.")
    parser.add_argument("--model-family", default="Decom_CNN-5-512")
    parser.add_argument("--modes", nargs="+", choices=["avg", "target_only", "projection", "layer_mask", "layer_mask_budget", "layer_softmax", "layer_relu", "projection_local", "layer_projection_global", "layer_projection_local", "projection_same_label", "projection_cross_label", "projection_softmax", "projection_relu", "softmax_only", "apa", "apa_logit", "dwa_soft", "dwa_soft_projection", "dwa_adaptive_self", "dwa_adaptive_self_projection", "meta_projection_fixed", *GROUP_MODES, *FEEDBACK_MODES, *DEFERRED_FEEDBACK_MODES],
                        default=["avg", "target_only", "projection", "layer_mask"],
                        help="feedback_projection_self is deferred and rejected until a reference is defined.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument('--target_client_id', type=int, default=0)
    parser.add_argument("--projection_self_weight", type=float, default=None,
                        help="Fixed target mass for projection_softmax or softmax_only; omitted uses sample weight.")
    parser.add_argument("--feedback_anchor_weight", type=float, default=None,
                        help="feedback_avg_self: retained S_t fraction (default .125), not target upload mass.")
    parser.add_argument("--target_prox_mu", type=float, default=0.)
    parser.add_argument("--target_prox_scope", choices=["helpers", "all"], default="helpers")
    parser.add_argument("--target_head_mode", choices=["shared", "local"], default="shared")
    add_head_l2_options(parser)
    target_z2.add_options(parser)
    target_c0_replay.add_options(parser)
    parser.add_argument("--target_local_schedule", choices=["joint", "fedrep"], default="joint")
    parser.add_argument("--fedrep_head_epochs", type=int, default=5)
    parser.add_argument("--head_diag_collect", action="store_true")
    parser.add_argument("--head_diag_rounds", default="10,20,30")
    parser.add_argument("--head_diag_dir", default="")
    parser.add_argument("--target_group_ids", default=None)
    parser.add_argument("--target_group_weight", type=float, default=None,
                        help="Group mass for group_projection_softmax or group_softmax_only.")
    parser.add_argument("--training_log_root", default="training_logs",
                        help="Independent downloadable log-only root.")
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
    replay_requested = any(arg.startswith('--target_c0_replay_') for arg in sys.argv)
    z2_requested=target_z2.enabled(options) or any(arg in ('--target_z2_lambda','--target_z2_start_round','--target_z2_feature','--target_z2_diagnostics')
        or arg.startswith(('--target_z2_lambda=','--target_z2_start_round=','--target_z2_feature=','--target_z2_diagnostics=')) for arg in sys.argv)
    prox_requested = options.target_prox_mu > 0 or any(arg in ("--target_prox_mu", "--target_prox_scope")
        or arg.startswith(("--target_prox_mu=", "--target_prox_scope=")) for arg in sys.argv)
    try:
        for mode in options.modes:
            target_c0_replay.validate(argparse.Namespace(**vars(options),algorithm='FedTargetProj',target_proj_mode=mode))
            target_z2.validate(argparse.Namespace(**vars(options),algorithm='FedTargetProj',target_proj_mode=mode))
            validate_head_l2(argparse.Namespace(**vars(options), algorithm="FedTargetProj", target_proj_mode=mode))
            validate_local_schedule(argparse.Namespace(**vars(options), algorithm="FedTargetProj",
                num_classes=100, target_proj_mode=mode))
            validate_head_diagnostic(argparse.Namespace(**vars(options), algorithm="FedTargetProj",
                num_classes=100, target_proj_mode=mode))
            validate_target_head(argparse.Namespace(**vars(options), algorithm="FedTargetProj",
                num_classes=100, target_proj_mode=mode))
            validate_feedback_options(argparse.Namespace(**vars(options), algorithm="FedTargetProj",
                num_clients=20, local_epochs=5, eval_gap=1, target_proj_mode=mode))
            validate_target_prox(options.target_prox_mu, options.target_prox_scope, "FedTargetProj",
                mode, options.target_client_id, options.projection_self_weight, options.meta_c0_split, options.meta_collect_snapshots)
    except ValueError as error:
        parser.error(str(error))
    groups = {}
    try:
        for mode in options.modes:
            groups[mode] = validate_group_options(options.target_group_ids, options.target_group_weight,
                mode, 20, options.meta_c0_split, options.meta_collect_snapshots)
    except ValueError as error:
        parser.error(str(error))
    if options.projection_self_weight is not None:
        try:
            for mode in options.modes:
                validate_projection_self_weight(options.projection_self_weight, mode, options.meta_c0_split)
        except ValueError as error:
            parser.error(str(error))
    if options.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not 0 <= options.target_client_id < 20:
        parser.error('--target_client_id must be in [0,20) for this fixed 20-client launcher')
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
        "-t", "1", "--seed", "0", "--target_client_id", str(options.target_client_id),
        "-data", "Cifar100", "-ncl", "100", "-nc", "20",
        "-niid", "1", "-pt", "pat", "-cpc", "20", "-jr", "1.0",
        "-m", options.model_family, "-lr", "0.005", "-lbs", "16",
        "-ls", "5", "-gr", str(options.rounds), "-eg", "1",
        "-is_regular", "1", "-regular_lamda", "1e-3",
        "-dev", options.device, "-did", options.device_id,
    ]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = system_dir / "target_proj_runs" / stamp
    log_root = Path(options.training_log_root).expanduser()
    if not log_root.is_absolute():
        log_root = system_dir / log_root
    log_root /= stamp
    jobs = []
    available_devices = Queue()
    for device_id in device_ids:
        available_devices.put(device_id)

    def train_job(job):
        command, folder, log_folder = job
        device_id = available_devices.get() if options.device == "cuda" else options.device_id
        try:
            command[command.index("-did") + 1] = device_id
            print(subprocess.list2cmdline(command), flush=True)
            (log_folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"Training {folder.name} on {options.device}:{device_id}; log: {log_folder / 'train.log'}", flush=True)
            with (log_folder / "train.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=system_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
        finally:
            try:
                print(f"Download logs: {archive_logs(log_folder)}", flush=True)
            except OSError as error:
                print(f"Logs remain in {log_folder}; ZIP failed: {error}", flush=True)
            finally:
                if options.device == "cuda":
                    available_devices.put(device_id)

    combinations = [(seed, mode) for seed in options.seeds for mode in options.modes]
    for index, (seed, mode) in enumerate(combinations):
        run_name = mode if options.projection_self_weight is None else f"{mode}_{self_weight_tag(options.projection_self_weight)}"
        if mode in FEEDBACK_MODES:
            run_name = feedback_tag(mode, options.feedback_anchor_weight)
        if options.target_head_mode == "local":
            run_name += "_local_head"
        if options.target_head_l2_lambda > 0:
            run_name += "_" + head_l2_tag(options.target_head_l2_lambda)
        if target_z2.enabled(options):
            run_name += '_' + target_z2.tag(argparse.Namespace(**vars(options),target_proj_mode=mode))
        if target_c0_replay.enabled(options):
            run_name += '_' + target_c0_replay.tag(options)
        if options.target_local_schedule == "fedrep":
            run_name += "_" + schedule_tag(options)
        if options.head_diag_collect:
            run_name += "_head_diag"
        if prox_requested and (mode == "projection_softmax" or (mode == "feedback_avg" and options.target_prox_mu > 0)):
            run_name += "_" + prox_tag(options.target_prox_mu, options.target_prox_scope)
        if mode in GROUP_MODES:
            run_name += "_" + group_tag(groups[mode])
            if options.target_group_weight is not None:
                run_name += "_beta" + str(options.target_group_weight).replace(".", "p")
        folder = root / run_name if options.seeds == [0] else root / f"seed{seed}" / run_name
        log_folder = log_root / run_name if options.seeds == [0] else log_root / f"seed{seed}" / run_name
        base = list(common)
        name = f"target{options.target_client_id}_seed{seed}_{run_name}"
        if mode in GROUP_MODES:
            position = base.index("--target_client_id")
            del base[position:position + 2]
            name = f"seed{seed}_{run_name}"
        command = base + [
            "--target_proj_mode", mode, "-exp_name", name,
            "-sfn", str(folder / "checkpoints"),
            "--h5_result_root", str(folder / "h5_results"),
            "--final-model-root", str(folder / "final_models"),
            "--run_log_dir", str(log_folder),
        ]
        command[command.index("--seed") + 1] = str(seed)
        if options.target_head_mode == "local":
            command += ["--target_head_mode", "local"]
        if options.target_head_l2_lambda > 0:
            command += ["--target_head_l2_lambda", str(options.target_head_l2_lambda),
                        "--target_head_l2_diagnostics", str(options.target_head_l2_diagnostics)]
        if z2_requested:
            command += ['--target_z2_lambda',str(options.target_z2_lambda),'--target_z2_start_round',str(options.target_z2_start_round),
                        '--target_z2_feature',options.target_z2_feature,'--target_z2_diagnostics',str(options.target_z2_diagnostics)]
        if replay_requested:
            command += ['--target_c0_replay_mode',options.target_c0_replay_mode,
                        '--target_c0_replay_weight',str(options.target_c0_replay_weight),
                        '--target_c0_replay_diagnostics',str(options.target_c0_replay_diagnostics)]
        if options.target_local_schedule == "fedrep":
            command += ["--target_local_schedule", "fedrep", "--fedrep_head_epochs", str(options.fedrep_head_epochs)]
        if options.head_diag_collect:
            command += ["--head_diag_collect", "--head_diag_rounds", options.head_diag_rounds]
            if options.head_diag_dir:
                command += ["--head_diag_dir", str(Path(options.head_diag_dir).resolve())]
        if mode in GROUP_MODES:
            command += ["--target_group_ids", ",".join(map(str, groups[mode]))]
            if options.target_group_weight is not None:
                command += ["--target_group_weight", str(options.target_group_weight)]
        if options.projection_self_weight is not None:
            command += ["--projection_self_weight", str(options.projection_self_weight)]
        if options.feedback_anchor_weight is not None:
            command += ["--feedback_anchor_weight", str(options.feedback_anchor_weight)]
        if prox_requested and mode in ("projection_softmax", "feedback_avg"):
            command += ["--target_prox_mu", str(options.target_prox_mu),
                        "--target_prox_scope", options.target_prox_scope]
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
        log_folder.mkdir(parents=True, exist_ok=False)
        if options.parallel:
            jobs.append((command, folder, log_folder))
        else:
            train_job((command, folder, log_folder))
    if jobs:
        workers = min(len(jobs), len(device_ids)) if options.device == "cuda" else len(jobs)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(train_job, jobs))
    if not options.dry_run:
        print(f"Completed matched controls: {root}")


if __name__ == "__main__":
    main()
