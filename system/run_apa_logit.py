"""APA-Logit launcher; run the five-round smoke check before the full experiment."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Default 5 gives six aggregations; explicitly use 100 after smoke checks.")
    parser.add_argument("--apa_logit_lr", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("run_target_proj.py")),
               "--modes", "apa_logit", "--rounds", str(options.rounds), "--device-id", options.device_id,
               "--apa_logit_lr", str(options.apa_logit_lr)]
    if options.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
