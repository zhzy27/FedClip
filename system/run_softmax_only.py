"""Client 0 / seed 0 / -gr 100 SoftmaxOnly control with the frozen shared launcher."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("run_target_proj.py")),
               "--modes", "softmax_only", "--rounds", "100", "--device-id", options.device_id]
    if options.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
