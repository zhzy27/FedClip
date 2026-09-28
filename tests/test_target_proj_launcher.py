"""Check matched commands and concurrent process dispatch without real training."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "target_proj_launcher", Path(__file__).resolve().parents[1] / "system" / "run_target_proj.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class TargetProjLauncherTests(unittest.TestCase):
    def test_apa_logit_smoke_and_full_rounds_keep_old_apa_options_separate(self):
        for rounds in ("5", "100"):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes",
                    "apa", "apa_logit", "--rounds", rounds, "--apa_logit_lr", "0.02"]), redirect_stdout(output):
                launcher.main()
            apa, logit = output.getvalue().splitlines()
            self.assertNotIn("--apa_logit_lr", apa)
            self.assertIn("--apa_server_lr 0.01", apa)
            self.assertIn("--apa_momentum 0.9", apa)
            self.assertIn("--apa_self_weight 0.5", apa)
            self.assertIn("--apa_logit_lr 0.02", logit)
            for option in ("--apa_server_lr", "--apa_momentum", "--apa_self_weight"):
                self.assertNotIn(option, logit)
            for option in (f"-gr {rounds}", "-data Cifar100", "-pt pat", "-cpc 20", "-nc 20", "-jr 1.0",
                           "--target_client_id 0", "--seed 0", "-m Decom_CNN-5-512", "-ls 5", "-lbs 16",
                           "-lr 0.005", "-regular_lamda 1e-3"):
                self.assertIn(option + " ", logit)

    def test_apa_options_only_affect_apa_command_and_keep_public_configuration(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes",
                "softmax_only", "apa", "--apa_server_lr", "0.02", "--apa_momentum", "0.8",
                "--apa_self_weight", "0.25"]), redirect_stdout(output):
            launcher.main()
        old, apa = output.getvalue().splitlines()
        self.assertNotIn("--apa_", old)
        for option in ("--apa_server_lr 0.02", "--apa_momentum 0.8", "--apa_self_weight 0.25"):
            self.assertIn(option, apa)
        for option in ("-gr 100", "-data Cifar100", "-pt pat", "-cpc 20", "-nc 20", "-jr 1.0",
                       "--target_client_id 0", "--seed 0", "-m Decom_CNN-5-512", "-ls 5", "-lbs 16",
                       "-lr 0.005", "-regular_lamda 1e-3"):
            self.assertIn(option + " ", apa)
            self.assertIn(option + " ", old)

    def test_projection_weighting_choices_keep_all_frozen_training_parameters(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes",
                "projection_softmax", "projection_relu", "--parallel", "--device-ids", "0", "1"]), \
                redirect_stdout(output), patch.object(launcher.subprocess, "run") as run:
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        for line, mode in zip(lines, ("projection_softmax", "projection_relu")):
            self.assertIn(f"--target_proj_mode {mode} ", line)
            for option in ("-gr 100", "-data Cifar100", "-pt pat", "-cpc 20", "-nc 20", "-jr 1.0",
                           "--target_client_id 0", "--seed 0", "-m Decom_CNN-5-512", "-ls 5", "-lbs 16",
                           "-lr 0.005", "-regular_lamda 1e-3"):
                self.assertIn(option + " ", line)
        run.assert_not_called()

    def test_defaults_still_select_original_four_modes(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run"]), redirect_stdout(output), \
                patch.object(launcher.subprocess, "run") as run:
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        for line, mode in zip(lines, ("avg", "target_only", "projection", "layer_mask")):
            self.assertIn(f"--target_proj_mode {mode} ", line)
            self.assertIn("-gr 100 ", line)
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

            argv = ["run_target_proj.py", "--modes", "layer_softmax", "layer_relu", "--parallel", "--rounds", "100", "--device-ids", "2", "5"]
            with patch.object(launcher, "__file__", str(system / "run_target_proj.py")), \
                    patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()), \
                    patch.object(launcher.subprocess, "run", side_effect=run):
                launcher.main()
            self.assertEqual(len(commands), 2)
            self.assertEqual(commands[0][:commands[0].index("-did")],
                             commands[1][:commands[1].index("-did")])
            self.assertEqual({command[command.index("-did") + 1] for command in commands}, {"2", "5"})
            for command in commands:
                for key, value in (("-gr", "100"), ("--seed", "0"), ("--target_client_id", "0"),
                                   ("-m", "Decom_CNN-5-512"), ("-lr", "0.005"), ("-ls", "5")):
                    self.assertEqual(command[command.index(key) + 1], value)
            records = list(system.glob("target_proj_runs/*/*/command.json"))
            self.assertEqual({path.parent.name for path in records}, {"layer_softmax", "layer_relu"})
            for path in records:
                self.assertIn(json.loads(path.read_text()), commands)
                self.assertEqual((path.parent / "train.log").read_text(), "synthetic process completed\n")

    def check_gpu_queue(self, devices):
        modes = ["projection_local", "layer_projection_global", "layer_projection_local",
                 "projection_same_label", "projection_cross_label", "layer_softmax", "layer_relu"]
        active, seen, peak = set(), [], []
        lock = threading.Lock()
        first_wave = threading.Barrier(len(devices))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            system = root / "system"
            system.mkdir()
            for split in ("train", "test"):
                shards = root / "dataset" / "Cifar100" / "pat_20" / split
                shards.mkdir(parents=True)
                for cid in range(20):
                    (shards / f"{cid}.npz").touch()

            def run(command, **kwargs):
                device = command[command.index("-did") + 1]
                mode = command[command.index("--target_proj_mode") + 1]
                with lock:
                    self.assertNotIn(device, active)
                    active.add(device)
                    seen.append(mode)
                    wave = len(seen) <= len(devices)
                    peak.append(len(active))
                if wave:
                    first_wave.wait(timeout=10)
                time.sleep(.01)
                with lock:
                    active.remove(device)
                self.assertEqual(command[command.index("-gr") + 1], "100")

            gpu_args = ["--device-id", devices[0]] if len(devices) == 1 else ["--device-ids", *devices]
            with patch.object(launcher, "__file__", str(system / "run_target_proj.py")), \
                    patch.object(sys, "argv", ["run_target_proj.py", "--modes", *modes, "--parallel", *gpu_args]), \
                    redirect_stdout(io.StringIO()), patch.object(launcher.subprocess, "run", side_effect=run):
                launcher.main()
        self.assertCountEqual(seen, modes)
        self.assertEqual(max(peak), len(devices))
        self.assertFalse(active)

    def test_seven_modes_queue_on_two_gpus_without_overlap(self):
        self.check_gpu_queue(["0", "3"])

    def test_legacy_single_device_parallel_is_serial_per_gpu(self):
        self.check_gpu_queue(["4"])

    def test_duplicate_gpu_ids_or_modes_are_rejected(self):
        for options in (["--device-ids", "0", "0"], ["--device-ids", "0", "00"],
                        ["--modes", "projection", "projection"]):
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", *options]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main()


if __name__ == "__main__":
    unittest.main()
