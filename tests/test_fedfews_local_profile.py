"""User-reported FedFew configuration edits, without changing algorithm code."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from fedfews_compat.acceptance import synthetic_data
from fedfews_compat.config import CONFIGS, parse_args
from fedfews_compat.data import audit_data
from fedfews_compat.metadata import source_info
from fedfews_compat.observe import rng_digest
from fedfews_compat.reference import configuration_alignment, reference_config

REFERENCE = ROOT.parent / "FedFew_reference"
PARTITIONS = ["noniid_dir_20_a0p5", "noniid_pat_10"]


class LocalProfileTests(unittest.TestCase):
    def test_both_bundled_profiles_use_reported_case_and_device(self):
        for partition, count in zip(PARTITIONS, [20, 10]):
            args, config = parse_args(["--config", str(CONFIGS / partition / "algorithms/fedfews.yaml"),
                                      "--data-root", "unused", "--output-dir", "unused_out"])
            self.assertEqual((args.dataset, args.device_id, args.num_clients, args.test_batch_size),
                             (f"Cifar100/{partition}", "2", count, 256))
            self.assertEqual((args.num_server_models, args.smooth_mu, args.local_learning_rate), (3, .01, .005))


@unittest.skipUnless(REFERENCE.is_dir(), "Independent FedFew_reference checkout not available")
class ExternalConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        # Keep the pristine external checkout untouched. Config construction still
        # executes its independent runner and argparse, not migrated equivalents.
        files = ["scripts/run_pfllib.py", "PFLlib/system/main.py", "PFLlib/dataset/utils/dataset_utils.py"]
        files += [f"configs/cifar100/{part}/{suffix}" for part in PARTITIONS
                  for suffix in ["base.yaml", "algorithms/fedfews.yaml"]]
        for relative in files:
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REFERENCE / relative, target)

    def apply_reported_edits(self):
        for partition in PARTITIONS:
            path = self.source / f"configs/cifar100/{partition}/base.yaml"
            text = path.read_text(encoding="utf-8").replace("name: cifar100", "name: Cifar100").replace('id: "0"', 'id: "2"')
            path.write_text(text, encoding="utf-8")
        path = self.source / "PFLlib/dataset/utils/dataset_utils.py"
        path.write_text(path.read_text(encoding="utf-8").replace("alpha = 0.1", "alpha = 0.5"), encoding="utf-8")

    def alignment(self, partition=PARTITIONS[0], overrides=None):
        common = ["--config", str(CONFIGS / partition / "algorithms/fedfews.yaml"),
                  "--data-root", str(self.root / "data"), "--output-dir", str(self.root / "out")]
        default_args, migrated = parse_args(common)
        runtime = ["--dataset", f"Cifar100/{partition}", "--device", "cpu", "--device_id", "0",
                   "--num_clients", "3", "--batch_size", "4", "--global_rounds", "1", "--eval_gap", "2"] if overrides is None else overrides
        actual, _ = parse_args(common + runtime)
        with redirect_stdout(io.StringIO()):
            source_yaml, effective, command = reference_config(self.source, self.root / "refout", partition)
        return configuration_alignment(self.source, source_yaml, migrated, effective,
                                       vars(default_args), command, runtime, vars(actual))

    def test_reported_source_edits_align_exactly(self):
        self.apply_reported_edits()
        for partition in PARTITIONS:
            result = self.alignment(partition)
            self.assertEqual(result["status"], "PASS", result)
            self.assertEqual(result["yaml_differences"], [])
            self.assertEqual(result["effective_default_differences"], [])
            self.assertEqual(result["source_runtime_args"]["dataset"], f"Cifar100/{partition}")

    def test_pristine_source_requires_explicit_common_execution_overrides(self):
        result = self.alignment()
        self.assertEqual(result["status"], "PASS", result)
        self.assertEqual({d["field"] for d in result["yaml_differences"]}, {"dataset.name", "device.id"})
        self.assertEqual(result["source_runtime_args"]["device_id"], "0")
        self.assertIn("--dataset", result["source_runtime_command"])
        missing = self.alignment(overrides=[])
        self.assertEqual(missing["status"], "FAIL")
        self.assertTrue(any("Effective runtime dataset" in s for s in missing["failures"]))
        self.assertTrue(any("Effective runtime device_id" in s for s in missing["failures"]))

    def test_training_and_algorithm_mismatches_still_fail(self):
        self.apply_reported_edits()
        path = self.source / f"configs/cifar100/{PARTITIONS[0]}/base.yaml"
        text = path.read_text(encoding="utf-8")
        # A runtime short-run epoch override must not hide changed source defaults.
        path.write_text(text.replace("local_epochs: 1", "local_epochs: 2"), encoding="utf-8")
        result = self.alignment()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("training.local_epochs" in s for s in result["failures"]))
        path.write_text(text, encoding="utf-8")
        path = self.source / f"configs/cifar100/{PARTITIONS[0]}/algorithms/fedfews.yaml"
        path.write_text(path.read_text(encoding="utf-8").replace("num_server_models: 3", "num_server_models: 4"), encoding="utf-8")
        result = self.alignment()
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("algorithm.params.num_server_models" in s for s in result["failures"]))

    def test_metadata_reads_generator_constant_without_running_or_changing_rng(self):
        before = rng_digest()
        self.assertEqual(source_info(self.source)["source_data_generation"]["configured_alpha"], .1)
        self.apply_reported_edits()
        # Importing this file would fail; the audit must only parse its constant.
        path = self.source / "PFLlib/dataset/utils/dataset_utils.py"
        path.write_text(path.read_text(encoding="utf-8") + '\nraise RuntimeError("must never execute")\n', encoding="utf-8")
        info = source_info(self.source)
        self.assertEqual(info["source_data_generation"]["configured_alpha"], .5)
        self.assertFalse(info["source_data_generation"]["executed"])
        self.assertIn("PFLlib/dataset/utils/dataset_utils.py", info["actual_files"])
        self.assertEqual(rng_digest(), before)

    def test_generator_edit_does_not_override_existing_data_alpha(self):
        self.apply_reported_edits()
        dataset = synthetic_data(self.root / "data")
        path = self.root / "data" / dataset / "config.json"
        data = json.loads(path.read_text())
        data.update(alpha=.1, partition="dir")
        path.write_text(json.dumps(data), encoding="utf-8")
        args, _ = parse_args(["--data-root", str(self.root / "data"), "--output-dir", str(self.root / "out"),
                              "--num_clients", "3", "--expected-alpha", ".5", "--expected-partition", "dir"])
        with self.assertRaisesRegex(ValueError, "actual alpha=0.1"):
            audit_data(args)


if __name__ == "__main__":
    unittest.main()
