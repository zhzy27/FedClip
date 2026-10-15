"""Head collection policy, fixed offline protocol, forwarding and paired summaries."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from utils.head_diagnostic_options import diagnostic_rounds, ensure_head_diagnostic_names, validate_head_diagnostic
from utils.head_diagnostic import PROTOCOL, compare_results, evaluate, training_order, order_fingerprint
from test_meta_projection import assert_rng, rng_state
import run_target_proj as launcher


def args(**changes):
    values = dict(algorithm="FedTargetProj", target_proj_mode="avg", target_head_mode="shared",
        model_family="Decom_CNN-5-512", head_diag_collect=True, head_diag_rounds="10,20,30",
        global_rounds=30, target_client_id=0, num_classes=100, num_clients=20,
        exp_name="example", save_folder_name="checkpoints", h5_result_root="h5",
        final_model_root="final", training_log_root="logs", run_log_dir="")
    values.update(changes)
    return SimpleNamespace(**values)


class HeadDiagnosticTests(unittest.TestCase):
    def test_policy_default_noop_and_completed_round_validation(self):
        for mode in ("avg", "projection_softmax"):
            for head in ("shared", "local"):
                validate_head_diagnostic(args(target_proj_mode=mode, target_head_mode=head, projection_self_weight=.125))
        self.assertEqual(diagnostic_rounds(args(head_diag_rounds="30,10,20")), (10, 20, 30))
        self.assertEqual(diagnostic_rounds(args(head_diag_rounds="31")), (31,))
        for changes in (dict(head_diag_rounds="0"), dict(head_diag_rounds="32"), dict(head_diag_rounds="1,1"),
            dict(head_diag_rounds="x"), dict(head_diag_rounds=""), dict(target_proj_mode="feedback_avg"),
            dict(target_proj_mode="projection_softmax", projection_self_weight=.05), dict(algorithm="FedAvg"),
            dict(target_head_mode="bad"), dict(target_prox_mu=.1), dict(meta_c0_split="split.json"),
            dict(meta_collect_snapshots=True), dict(learning_rate_decay=True), dict(local_epochs=1),
            dict(batch_size=32), dict(local_learning_rate=.01), dict(is_regular=0), dict(regular_lamda=.01),
            dict(resume=True), dict(auto_break=True), dict(few_shot=1), dict(num_classes=20)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_head_diagnostic(args(**changes))
        off = args(head_diag_collect=False, target_proj_mode="dwa_soft")
        before = copy.deepcopy(vars(off))
        validate_head_diagnostic(off)
        ensure_head_diagnostic_names(off)
        self.assertEqual(vars(off), before)
        with self.assertRaises(ValueError):
            validate_head_diagnostic(args(head_diag_collect=False, head_diag_dir="snapshots"))

    def test_collection_names_idempotent_and_not_meta(self):
        value = args()
        ensure_head_diagnostic_names(value)
        previous = copy.deepcopy(vars(value))
        ensure_head_diagnostic_names(value)
        self.assertEqual(previous, vars(value))
        for key in ("exp_name", "save_folder_name", "training_log_root", "h5_result_root", "final_model_root"):
            self.assertIn("head_diag_avg_shared", getattr(value, key))
            self.assertNotIn("meta", getattr(value, key))
        external = args(training_log_dir="captured")
        with patch.dict("os.environ", {"FEDCLIP_TRAIN_LOG_DIR": "external_logs"}):
            ensure_head_diagnostic_names(external)
        self.assertIn("head_diag_avg_shared", external.training_log_dir)
        self.assertIn("head_diag_avg_shared", external.run_log_dir)

    def test_independent_exact_batch_orders_do_not_advance_normal_random_stream(self):
        before = rng_state()
        a = training_order(33, 1729)
        torch.rand(3)  # Normal process RNG does not select diagnostic batches.
        b = training_order(33, 1729)
        self.assertEqual(a, b)
        self.assertEqual(order_fingerprint(a), order_fingerprint(b))
        for indices in a:
            self.assertEqual(sorted(indices), list(range(33)))
        torch.set_rng_state(before[2])
        assert_rng(self, before)
        self.assertNotEqual(a, training_order(33, 1730))
        self.assertEqual(PROTOCOL["observations"], [0, 1, 5])
        self.assertEqual(PROTOCOL["loss"], "CE_only")

    def test_actual_fc3_input_features_include_preceding_relu(self):
        class FeatureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = torch.nn.BatchNorm1d(2, eps=0.)
                self.fc3 = torch.nn.Linear(2, 100)
                self.head = torch.nn.Sequential(torch.nn.ReLU(), self.fc3)
            def forward(self, x):
                return self.head(self.backbone(x))
        model = FeatureModel()
        buffers = {n: p.clone() for n, p in model.named_buffers()}
        data = [(torch.tensor([-5., 3.]), torch.tensor(0)), (torch.tensor([-2., -4.]), torch.tensor(1))]
        result = evaluate(model, data, "cpu", features=True)
        self.assertEqual(result["input_feature_mean_l2"], 1.5)
        self.assertEqual(result["feature_sample_count"], 2)
        self.assertTrue(model.training)
        for n, p in model.named_buffers():
            torch.testing.assert_close(p, buffers[n], rtol=0, atol=0)

    def test_comparison_gap_and_public_initialization_and_order_checks(self):
        def result(head, accuracies):
            return dict(source_metadata=dict(mode="avg", head_mode=head, round=10, seed=0, capacity=.9,
                args=dict(model_family="Decom_CNN-5-512"), public_head_sha256="same_init",
                data=dict(train=dict(sample_sha256="train"), test=dict(sample_sha256="test"))),
                batch_order_sha256="same_batches", rows=[dict(variant=v, head_train_epoch=e, test_accuracy=a)
                    for (v, e), a in zip((("native", 0), ("fresh_head", 1), ("fresh_head", 5)), accuracies)])
        shared, local = result("shared", [.5, .5, .55]), result("local", [.4, .45, .53])
        rows = compare_results([shared, local])
        self.assertAlmostEqual(rows[0]["native_shared_minus_local_pp"], 10.)
        self.assertAlmostEqual(rows[0]["fresh1_shared_minus_local_pp"], 5.)
        self.assertAlmostEqual(rows[0]["fresh5_shared_minus_local_pp"], 2.)
        self.assertAlmostEqual(rows[0]["absolute_gap_reduction5_pp"], 8.)
        wrong = copy.deepcopy(local)
        wrong["source_metadata"]["public_head_sha256"] = "different"
        with self.assertRaises(ValueError): compare_results([shared, wrong])
        wrong = copy.deepcopy(local)
        wrong["batch_order_sha256"] = "different"
        with self.assertRaises(ValueError): compare_results([shared, wrong])
        with self.assertRaises(ValueError): compare_results([shared, shared])
        self.assertEqual(compare_results([shared]), [])

    def test_launcher_forwards_independent_switch_and_defaults_leave_it_off(self):
        def run(extra):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "avg", "--rounds", "30", "--dry-run", *extra]), redirect_stdout(output):
                launcher.main()
            return shlex.split(output.getvalue().strip(), posix=False)
        self.assertNotIn("--head_diag_collect", run([]))
        command = run(["--head_diag_collect", "--head_diag_rounds", "10,20,30", "--head_diag_dir", "snapshots"])
        self.assertIn("--head_diag_collect", command)
        self.assertEqual(command[command.index("--head_diag_rounds") + 1], "10,20,30")
        self.assertTrue(Path(command[command.index("--head_diag_dir") + 1]).is_absolute())
        for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir"):
            self.assertIn("head_diag", command[command.index(key) + 1])
        self.assertNotIn("--meta_c0_split", command)

    def test_main_accepts_collection_without_meta_or_online_training(self):
        script = '''
import ast, contextlib, io, json, pathlib, sys
from unittest.mock import patch
path = pathlib.Path(sys.argv[1]); head = sys.argv[2]
sys.path.insert(0, str(path.parent))
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
node.body = ast.parse("capture_args(args)").body
ast.fix_missing_locations(tree)
captured = []
sys.argv = [str(path), "-algo", "FedTargetProj", "--target_proj_mode", "avg", "-m", "Decom_CNN-5-512", "-ncl", "100",
 "-nc", "20", "-gr", "30", "-ls", "5", "-lr", ".005", "-lbs", "16", "-regular_lamda", ".001", "-dev", "cpu",
 "--target_head_mode", head, "--head_diag_collect", "--head_diag_rounds", "10,20,30", "--head_diag_dir", "snapshots"]
with contextlib.redirect_stdout(io.StringIO()), patch("utils.training_logs.training_log_session", side_effect=lambda a: contextlib.nullcontext()):
 exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path), "capture_args": captured.append})
print(json.dumps(vars(captured[0])))
'''
        for head in ("shared", "local"):
            process = subprocess.run([sys.executable, "-c", script, str(ROOT / "system/main.py"), head], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            record = json.loads(process.stdout)
            self.assertTrue(record["head_diag_collect"])
            self.assertEqual(record["head_diag_rounds"], "10,20,30")
            self.assertEqual(record["head_diag_dir"], "snapshots")
            self.assertEqual(record["meta_c0_split"], "")
            self.assertFalse(record["meta_collect_snapshots"])
            self.assertFalse(record["learning_rate_decay"])

    def test_four_short_commands_are_aligned_and_run_in_two_separate_batches(self):
        for filename, mode in (("head_diagnostic_commands.sh", "avg"),
                               ("head_diagnostic_projection_commands.sh", "projection_softmax")):
            text = (ROOT / "system" / filename).read_text()
            commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
            self.assertEqual(len(commands), 2)
            roots, devices = [], []
            for command, head in zip(commands, ("shared", "local")):
                values = shlex.split(command)
                for key, expected in (("--target_proj_mode", mode), ("--target_head_mode", head),
                    ("-gr", "30"), ("-ls", "5"), ("-lbs", "16"), ("-lr", "0.005"), ("-eg", "1"),
                    ("-nc", "20"), ("-ncl", "100"), ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"),
                    ("-pt", "pat"), ("-cpc", "20"), ("--seed", "0"), ("--target_client_id", "0"),
                    ("-is_regular", "1"), ("-regular_lamda", "1e-3"), ("--head_diag_rounds", "10,20,30")):
                    self.assertEqual(values[values.index(key) + 1], expected)
                self.assertIn("--head_diag_collect", values)
                for key in ("--meta_c0_split", "--meta_collect_snapshots", "--target_prox_mu", "-ld"):
                    self.assertNotIn(key, values)
                if mode == "projection_softmax":
                    self.assertEqual(values[values.index("--projection_self_weight") + 1], "0.125")
                else:
                    self.assertNotIn("--projection_self_weight", values)
                paths = tuple(values[values.index(key) + 1] for key in ("-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir"))
                self.assertTrue(all("head_diag" in p for p in paths))
                roots.append(paths)
                devices.append(values[values.index("-did") + 1])
            self.assertTrue(all(a != b for a, b in zip(*roots)))
            self.assertEqual(len(set(devices)), 2)


if __name__ == "__main__":
    unittest.main()
