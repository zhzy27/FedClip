"""Check matched commands and concurrent process dispatch without real training."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "target_proj_launcher", Path(__file__).resolve().parents[1] / "system" / "run_target_proj.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TargetProjLauncherTests(unittest.TestCase):
    def test_defaults_still_select_original_four_modes(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run"]), redirect_stdout(output), \
                patch.object(launcher.subprocess, "run") as run:
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        for line, mode in zip(lines, ("avg", "target_only", "projection", "layer_mask")):
            self.assertIn(f"--target_proj_mode {mode} ", line)
        run.assert_not_called()

    def test_new_modes_dispatch_concurrently_with_matched_parameters(self):
        barrier = threading.Barrier(2)
        commands = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "system"
            system.mkdir()
            for split in ("train", "test"):
                shards = root / "dataset" / "Cifar100" / "pat_20" / split
                shards.mkdir(parents=True)
                for cid in range(20):
                    (shards / f"{cid}.npz").touch()

            def run(command, cwd, stdout, stderr, check):
                self.assertEqual(cwd, system.resolve())
                self.assertTrue(check)
                commands.append(command)
                barrier.wait(timeout=10)  # Sequential dispatch would fail this check.
                stdout.write("synthetic process completed\n")

            argv = ["run_target_proj.py", "--modes", "layer_softmax", "layer_relu", "--parallel", "--rounds", "29"]
            with patch.object(launcher, "__file__", str(system / "run_target_proj.py")), \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                    patch.object(launcher.subprocess, "run", side_effect=run):
                launcher.main()
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][:commands[0].index("--target_proj_mode")],
                             commands[1][:commands[1].index("--target_proj_mode")])
            for command in commands:
                for key, value in (("-gr", "29"), ("--seed", "0"), ("--target_client_id", "0"),
                                   ("-m", "Decom_CNN-5-512"), ("-lr", "0.005"), ("-ls", "5")):
                    self.assertEqual(command[command.index(key) + 1], value)
            records = list(system.glob("target_proj_runs/*/*/command.json"))
            self.assertEqual({path.parent.name for path in records}, {"layer_softmax", "layer_relu"})
            for path in records:
                self.assertIn(json.loads(path.read_text()), commands)
                self.assertEqual((path.parent / "train.log").read_text(), "synthetic process completed\n")


if __name__ == "__main__":
    unittest.main()
