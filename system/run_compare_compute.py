import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


METHODS = [
    {
        "algorithm": "FedGH",
        "model": "CNN-5-512",
        "extra": ["-slr", "0.005", "-se", "5"],
    },
    {
        "algorithm": "LG-FedAvg",
        "model": "CNN-5-512",
        "extra": [],
    },
    {
        "algorithm": "FedTGP",
        "model": "CNN-5-512",
        "extra": ["-lam", "10.0", "-se", "20", "-mart", "100", "-fd", "512"],
    },
    {
        "algorithm": "FedProto",
        "model": "CNN-5-512",
        "extra": ["-lam", "10"],
    },
    {
        "algorithm": "FML",
        "model": "CNN-5-512",
        "extra": ["-al", "0.5", "-bt", "0.5"],
    },
    {
        "algorithm": "FD",
        "model": "CNN-5-512",
        "extra": ["-lam", "1"],
    },
    {
        "algorithm": "FedKD",
        "model": "CNN-5-512",
        "extra": ["-mlr", "0.005", "-Ts", "0.95", "-Te", "0.98", "-fd", "512"],
    },
    {
        "algorithm": "FedGen",
        "model": "CNN-5-512",
        "extra": ["-fd", "512", "-nd", "32", "-glr", "0.05", "-hd", "512", "-se", "20"],
    },
    {
        "algorithm": "FedMRL",
        "model": "CNN-5-512",
        "extra": ["-fd", "512", "-sfd", "128"],
    },
    {
        "algorithm": "PFedAFM",
        "model": "CNN-5-512-AFM",
        "extra": ["-alpha_lr", "0.01"],
    },
    {
        "algorithm": "FedSPU",
        "model": "SPU_CNN1",
        "extra": [],
    },
    {
        "algorithm": "FedCLIP",
        "model": "Decom_CNN-5-512",
        "extra": [
            "-is_regular", "1",
            "-mse_lamda", "1",
            "-Cos_lamda", "0.0",
            "-regular_lamda", "1e-3",
            "-v_mse_lamda", "0",
            "-aggregate_tau", "1",
        ],
    },
]


CSV_FIELDS = [
    "run_index",
    "status",
    "returncode",
    "algorithm",
    "model_family",
    "dataset",
    "partition",
    "dir_alpha",
    "class_per_client",
    "num_classes",
    "num_clients",
    "join_ratio",
    "batch_size",
    "local_epochs",
    "global_rounds",
    "round",
    "num_selected_clients",
    "total_local_train_flops",
    "total_local_train_gflops",
    "total_forward_flops_per_epoch",
    "total_forward_gflops_per_epoch",
    "local_train_multiplier",
    "server_total_seconds",
    "server_events_json",
    "json_path",
    "log_path",
    "command",
]


def partition_args(args):
    if args.partition == "pat":
        return ["-pt", "pat", "-cpc", str(args.cpc)]
    if args.partition == "dir":
        return ["-pt", "dir", "-dir_alpha", str(args.dir_alpha)]
    if args.partition == "exdir":
        return ["-pt", "exdir", "-dir_alpha", str(args.dir_alpha)]
    raise ValueError(f"Unsupported partition: {args.partition}")


def build_command(method, args):
    cmd = [
        args.python,
        "main.py",
        "-t", str(args.times),
        "-lr", str(args.lr),
        "-jr", str(args.join_ratio),
        "-lbs", str(args.batch_size),
        "-gr", str(args.rounds),
        "-ls", str(args.local_epochs),
        "-nc", str(args.num_clients),
        "-ncl", str(args.num_classes),
        "-data", args.dataset,
        "-m", method["model"],
        "-did", str(args.device_id),
        "-algo", method["algorithm"],
        "-niid", str(args.niid),
    ]
    cmd.extend(partition_args(args))
    cmd.extend(method["extra"])
    cmd.extend([
        "--measure_local_flops", "1",
        "--local_flops_round", str(args.local_flops_round),
        "--local_flops_train_multiplier", str(args.local_flops_train_multiplier),
        "--local_flops_detail", str(args.local_flops_detail),
        "--measure_server_compute", "1",
        "--server_compute_detail", str(args.server_compute_detail),
    ])
    if args.extra_main_args:
        cmd.extend(args.extra_main_args)
    return cmd


def append_rows(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_paths = []
    json_pattern = re.compile(r"JSON文件已成功保存到:\s*(.+\.json)")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            match = json_pattern.search(line)
            if match:
                json_paths.append(match.group(1).strip())
        returncode = process.wait()
    return returncode, json_paths


def resolve_json_path(json_paths, cwd, algorithm, dataset, start_time):
    candidates = []
    for raw_path in json_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = cwd / path
        if path.exists() and path.parent.name == "json":
            candidates.append(path)

    json_dir = cwd / "json"
    if json_dir.exists():
        pattern = f"algo_{algorithm}-dataset{dataset}-*.json"
        for path in json_dir.glob(pattern):
            try:
                if path.stat().st_mtime >= start_time - 1:
                    candidates.append(path)
            except OSError:
                pass

    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def group_server_events(server_records):
    grouped = defaultdict(lambda: defaultdict(float))
    for event in server_records:
        round_idx = int(event.get("round", 0))
        event_name = str(event.get("event", "unknown"))
        grouped[round_idx][event_name] += float(event.get("seconds", 0.0))
    return grouped


def rows_from_json(payload, json_path, log_path, command, returncode, run_index, fallback):
    args_payload = payload.get("args", {})
    local_records = payload.get("local_flops_records", [])
    server_grouped = group_server_events(payload.get("server_compute_records", []))
    local_by_round = {int(record.get("round", 0)): record for record in local_records}
    rounds = sorted(set(local_by_round.keys()) | set(server_grouped.keys()))
    if not rounds:
        rounds = [None]

    rows = []
    for round_idx in rounds:
        local_record = local_by_round.get(round_idx, {})
        server_events = server_grouped.get(round_idx, {})
        server_total = sum(server_events.values())
        total_local = float(local_record.get("total_local_train_flops", 0.0))
        total_forward_epoch = float(local_record.get("total_forward_flops_per_epoch", 0.0))
        rows.append({
            "run_index": run_index,
            "status": "ok" if returncode == 0 else "process_failed_json_found",
            "returncode": returncode,
            "algorithm": args_payload.get("algorithm", fallback["algorithm"]),
            "model_family": args_payload.get("model_family", fallback["model_family"]),
            "dataset": args_payload.get("dataset", fallback["dataset"]),
            "partition": args_payload.get("partition", fallback["partition"]),
            "dir_alpha": args_payload.get("dir_alpha", fallback["dir_alpha"]),
            "class_per_client": args_payload.get("class_per_client", fallback["class_per_client"]),
            "num_classes": args_payload.get("num_classes", fallback["num_classes"]),
            "num_clients": args_payload.get("num_clients", fallback["num_clients"]),
            "join_ratio": args_payload.get("join_ratio", fallback["join_ratio"]),
            "batch_size": args_payload.get("batch_size", fallback["batch_size"]),
            "local_epochs": args_payload.get("local_epochs", fallback["local_epochs"]),
            "global_rounds": args_payload.get("global_rounds", fallback["global_rounds"]),
            "round": "" if round_idx is None else round_idx,
            "num_selected_clients": local_record.get("num_selected_clients", ""),
            "total_local_train_flops": total_local,
            "total_local_train_gflops": total_local / 1e9,
            "total_forward_flops_per_epoch": total_forward_epoch,
            "total_forward_gflops_per_epoch": total_forward_epoch / 1e9,
            "local_train_multiplier": local_record.get("train_multiplier", ""),
            "server_total_seconds": server_total,
            "server_events_json": json.dumps(server_events, ensure_ascii=False, sort_keys=True),
            "json_path": str(json_path),
            "log_path": str(log_path),
            "command": " ".join(command),
        })
    return rows


def failure_row(run_index, method, args, command, log_path, returncode, status):
    return {
        "run_index": run_index,
        "status": status,
        "returncode": returncode,
        "algorithm": method["algorithm"],
        "model_family": method["model"],
        "dataset": args.dataset,
        "partition": args.partition,
        "dir_alpha": args.dir_alpha,
        "class_per_client": args.cpc,
        "num_classes": args.num_classes,
        "num_clients": args.num_clients,
        "join_ratio": args.join_ratio,
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "global_rounds": args.rounds,
        "round": "",
        "num_selected_clients": "",
        "total_local_train_flops": "",
        "total_local_train_gflops": "",
        "total_forward_flops_per_epoch": "",
        "total_forward_gflops_per_epoch": "",
        "local_train_multiplier": args.local_flops_train_multiplier,
        "server_total_seconds": "",
        "server_events_json": "",
        "json_path": "",
        "log_path": str(log_path),
        "command": " ".join(command),
    }


def main():
    parser = argparse.ArgumentParser(description="Run comparison methods and collect local/server compute statistics.")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--device-id", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--partition", type=str, default="pat", choices=["pat", "dir", "exdir"])
    parser.add_argument("--cpc", type=int, default=2)
    parser.add_argument("--dir-alpha", type=float, default=0.3)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--niid", type=int, default=1)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--local-flops-round", type=int, default=-1,
                        help="Use -1 to collect every round; use 0 to collect only the first round.")
    parser.add_argument("--local-flops-train-multiplier", type=float, default=3.0)
    parser.add_argument("--local-flops-detail", type=int, default=0)
    parser.add_argument("--server-compute-detail", type=int, default=0)
    parser.add_argument("--output-csv", type=str, default="compute_results/comparison_compute.csv")
    parser.add_argument("--log-dir", type=str, default="compute_results/logs")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional algorithm names to run, e.g. --only FedCLIP FedKD")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_main_args", nargs=argparse.REMAINDER,
                        help="Extra args passed to main.py after '--', for example -- --eval_gap 5")
    args = parser.parse_args()

    if args.extra_main_args and args.extra_main_args[0] == "--":
        args.extra_main_args = args.extra_main_args[1:]

    cwd = Path(__file__).resolve().parent
    output_csv = cwd / args.output_csv
    log_dir = cwd / args.log_dir

    selected_methods = METHODS
    if args.only:
        allow = set(args.only)
        selected_methods = [method for method in METHODS if method["algorithm"] in allow]

    for run_index, method in enumerate(selected_methods[args.start_index:], start=args.start_index):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_algo = method["algorithm"].replace("/", "_")
        log_path = log_dir / f"{run_index:02d}_{safe_algo}_{args.dataset}_{args.partition}_{timestamp}.log"
        command = build_command(method, args)
        print("\n" + "=" * 100)
        print(f"[{run_index + 1}/{len(selected_methods)}] Running {method['algorithm']}")
        print(" ".join(command))
        print("=" * 100)

        if args.dry_run:
            continue

        start_time = time.time()
        returncode, json_paths = run_command(command, log_path, cwd)
        json_path = resolve_json_path(json_paths, cwd, method["algorithm"], args.dataset, start_time)
        fallback = {
            "algorithm": method["algorithm"],
            "model_family": method["model"],
            "dataset": args.dataset,
            "partition": args.partition,
            "dir_alpha": args.dir_alpha,
            "class_per_client": args.cpc,
            "num_classes": args.num_classes,
            "num_clients": args.num_clients,
            "join_ratio": args.join_ratio,
            "batch_size": args.batch_size,
            "local_epochs": args.local_epochs,
            "global_rounds": args.rounds,
        }

        if json_path is None:
            row = failure_row(run_index, method, args, command, log_path, returncode, "json_not_found")
            append_rows(output_csv, [row])
            print(f"⚠️ 未找到 {method['algorithm']} 的 JSON，已写入失败行: {output_csv}")
            if returncode != 0:
                print(f"⚠️ {method['algorithm']} 进程返回码: {returncode}")
            continue

        try:
            payload = load_json(json_path)
            rows = rows_from_json(payload, json_path, log_path, command, returncode, run_index, fallback)
        except Exception as exc:
            row = failure_row(run_index, method, args, command, log_path, returncode, f"json_load_failed:{repr(exc)}")
            append_rows(output_csv, [row])
            print(f"⚠️ 读取 JSON 失败: {json_path} | {exc}")
            continue

        append_rows(output_csv, rows)
        print(f"✅ {method['algorithm']} 统计已追加到: {output_csv}")
        if returncode != 0:
            print(f"⚠️ {method['algorithm']} 进程返回码: {returncode}，但已找到 JSON 并写入统计。")

    print("\n处理结束")
    print(f"CSV: {output_csv}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
