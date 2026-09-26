"""Run the three matched Client-0 controls, each in a fresh Python process."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100, help="Inherited main.py -gr value.")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without training.")
    options = parser.parse_args()
    if options.rounds < 1:
        parser.error("--rounds must be at least 1")
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
        "-m", "Decom_resnet18_5", "-lr", "0.005", "-lbs", "16",
        "-ls", "5", "-gr", str(options.rounds), "-eg", "1",
        "-is_regular", "1", "-mse_lamda", "1", "-regular_lamda", "1e-3",
        "--use_asymmetric_lr", "1", "--u_lr_ratio", "0.3", "--v_lr_ratio", "1.0",
        "-dev", options.device, "-did", options.device_id,
    ]
    root = system_dir / "target_proj_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for mode in ("avg", "target_only", "projection"):
        folder = root / mode
        command = common + [
            "--target_proj_mode", mode, "-exp_name", f"target0_seed0_{mode}",
            "-sfn", str(folder / "checkpoints"),
            "--h5_result_root", str(folder / "h5_results"),
            "--final-model-root", str(folder / "final_models"),
        ]
        print(subprocess.list2cmdline(command), flush=True)
        if options.dry_run:
            continue
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        print(f"Training {mode}; log: {folder / 'train.log'}", flush=True)
        with (folder / "train.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=system_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not options.dry_run:
        print(f"Completed matched controls: {root}")


if __name__ == "__main__":
    main()
