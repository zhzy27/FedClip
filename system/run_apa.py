"""Client 0 / seed 0 APA experiment; all public training settings use the shared launcher."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apa_server_lr", type=float, default=0.01)
    parser.add_argument("--apa_momentum", type=float, default=0.9)
    parser.add_argument("--apa_self_weight", type=float, default=0.5)
    options = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("run_target_proj.py")),
               "--modes", "apa", "--rounds", "100", "--device-id", options.device_id,
               "--apa_server_lr", str(options.apa_server_lr),
               "--apa_momentum", str(options.apa_momentum),
               "--apa_self_weight", str(options.apa_self_weight)]
    if options.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
