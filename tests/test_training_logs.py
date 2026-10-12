"""Log-only bundles, console forwarding and independent process/run paths."""

from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.training_logs import archive_logs, server_log_folder, training_log_session
from collect_training_logs import collect_logs


class TrainingLogTests(unittest.TestCase):
    def args(self, root):
        return SimpleNamespace(algorithm="FedTargetProj", target_proj_mode="group_avg_only", seed=0,
                               training_log_root=str(root / "logs"), run_log_dir="")

    def test_capture_console_error_metrics_archive_and_stream_rng_restore(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            args = self.args(root)
            out, err = io.StringIO(), io.StringIO()
            before = random.getstate()
            with redirect_stdout(out), redirect_stderr(err):
                previous_out, previous_err = sys.stdout, sys.stderr
                with training_log_session(args) as folder:
                    print("ordinary epoch 5")
                    print("diagnostic warning", file=sys.stderr)
                    (folder / "metrics.csv").write_text("round,acc\n1,.5\n")
                    (folder / "model.pt").write_bytes(b"not in downloadable logs")
                self.assertIs(sys.stdout, previous_out)
                self.assertIs(sys.stderr, previous_err)
            self.assertEqual(random.getstate(), before)
            self.assertNotIn("FEDCLIP_TRAIN_LOG_DIR", os.environ)
            log = (folder / "train.log").read_text(encoding="utf-8")
            self.assertEqual(log.count("ordinary epoch 5"), 1)
            self.assertEqual(log.count("diagnostic warning"), 1)
            self.assertIn("ordinary epoch 5", out.getvalue())
            self.assertIn("diagnostic warning", err.getvalue())
            with zipfile.ZipFile(folder.with_name(folder.name + ".zip")) as archive:
                self.assertTrue(any(name.endswith("/metrics.csv") for name in archive.namelist()))
                self.assertFalse(any(name.endswith(".pt") for name in archive.namelist()))
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "complete")

    def test_exception_is_preserved_with_failed_status_and_archive(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(Path(directory))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "training failure"):
                with training_log_session(args) as folder:
                    raise RuntimeError("training failure")
            self.assertIn("training failure", (folder / "train.log").read_text())
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "failed")
            self.assertTrue(folder.with_name(folder.name + ".zip").is_file())

    def test_explicit_folder_precedes_environment_and_multiple_runs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            args.run_log_dir = str(root / "explicit")
            with patch.dict(os.environ, {"FEDCLIP_TRAIN_LOG_DIR": str(root / "outer")}), redirect_stdout(io.StringIO()):
                with training_log_session(args) as folder:
                    self.assertEqual(folder, (root / "explicit").resolve())
                    self.assertEqual(Path(os.environ["FEDCLIP_TRAIN_LOG_DIR"]), folder)
                self.assertEqual(os.environ["FEDCLIP_TRAIN_LOG_DIR"], str(root / "outer"))
            first = server_log_folder(args, "A", root / "models" / "A")
            second = server_log_folder(args, "B", root / "models" / "B")
            self.assertNotEqual(first, second)
            self.assertTrue(Path(first).is_relative_to((root / "explicit").resolve()))
            with self.assertRaises(ValueError):
                server_log_folder(args, "C", root / "explicit" / "models" / "C")

    def test_external_redirect_same_file_is_not_written_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.args(root)
            args.run_log_dir = str(root / "exact")
            Path(args.run_log_dir).mkdir()
            with (Path(args.run_log_dir) / "train.log").open("w", encoding="utf-8") as file:
                with redirect_stdout(file), redirect_stderr(file):
                    with training_log_session(args):
                        print("one copy")
            self.assertEqual((Path(args.run_log_dir) / "train.log").read_text().count("one copy"), 1)

    def test_archive_has_no_tensor_cache_or_previous_archive_and_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "download"
            (folder / "run_123").mkdir(parents=True)
            for name in ("metrics.json", "result.h5", "model.pt", "basis.pth", "old.zip"):
                (folder / "run_123" / name).write_bytes(b"content")
            first = archive_logs(folder)
            archive_logs(folder)
            with zipfile.ZipFile(first) as archive:
                self.assertEqual(set(archive.namelist()),
                    {"download/run_123/metrics.json", "download/run_123/result.h5"})

    def test_existing_logs_collected_without_moving_models_or_original_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old_models"
            source.mkdir()
            (source / "Client_0_model.pt").write_bytes(b"original model")
            (source / "metrics.csv").write_text("round,acc\n1,.5\n")
            log = root / "train.log"
            log.write_text("completed training")
            archive = collect_logs(source, root / "logs", [log])
            self.assertEqual((source / "Client_0_model.pt").read_bytes(), b"original model")
            self.assertTrue((source / "metrics.csv").exists())
            with zipfile.ZipFile(archive) as file:
                self.assertEqual(set(file.namelist()), {"logs/metrics.csv", "logs/train.log"})
            with self.assertRaises(ValueError):
                collect_logs(source, source / "recursive")


if __name__ == "__main__":
    unittest.main()
