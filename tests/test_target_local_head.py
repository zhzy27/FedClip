"""Alias-aware scope, CLI and exact backbone-only aggregation checks."""

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
from utils.target_head import (backbone_parameters, classifier_names, ensure_head_names,
                               validate_target_head)
from utils.target_projection import aggregate_target_updates
from utils.projection_variants import aggregate_projection_weighting
from test_meta_projection import rng_state, assert_rng
import run_target_proj as launcher


def options(**updates):
    data = dict(target_head_mode="local", algorithm="FedTargetProj", target_proj_mode="avg",
        model_family="Decom_CNN-5-512", num_classes=100, target_client_id=0, target_prox_mu=0.,
        eval_gap=1, exp_name="trial", save_folder_name="temp", training_log_root="logs",
        final_model_root="final", h5_result_root="h5", run_log_dir="")
    data.update(updates)
    return SimpleNamespace(**data)


class AliasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 2)
        self.fc2 = torch.nn.Linear(2, 512)
        self.fc3 = torch.nn.Linear(512, 100)
        self.head = torch.nn.Sequential(torch.nn.ReLU(), self.fc3)


class LocalHeadTests(unittest.TestCase):
    def test_validation_is_narrow_and_shared_is_noop(self):
        validate_target_head(options())
        validate_target_head(options(target_proj_mode="projection_softmax", projection_self_weight=.125))
        for changes in (dict(algorithm="FedAvg"), dict(model_family="other"), dict(num_classes=20),
            dict(target_client_id=1), dict(target_proj_mode="projection"), dict(target_proj_mode="softmax_only"),
            dict(target_proj_mode="group_avg_all"), dict(target_proj_mode="feedback_avg"),
            dict(target_proj_mode="dwa_soft"), dict(target_proj_mode="meta_projection_fixed"),
            dict(target_proj_mode="projection_softmax"), dict(target_proj_mode="projection_softmax", projection_self_weight=.1),
            dict(target_prox_mu=.01), dict(target_prox_mu=float("nan")), dict(meta_c0_split="split.json"),
            dict(meta_collect_snapshots=True), dict(meta_weight_file="weights.json"), dict(target_group_ids=[0]),
            dict(feedback_anchor_weight=.125), dict(models_folder_name="pretrained"), dict(resume=True),
            dict(eval_gap=2), dict(auto_break=True), dict(few_shot=1), dict(train_slow_rate=.1), dict(target_head_mode="typo")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_target_head(options(**changes))
        # The default must not narrow any historical mode's policy.
        validate_target_head(options(target_head_mode="shared", algorithm="FedAvg", target_proj_mode="dwa_soft", target_prox_mu=.1))

    def test_alias_scope_retains_fc1_fc2_and_rejects_frozen_head(self):
        model = AliasModel()
        self.assertEqual(classifier_names(model), ("fc3.weight", "fc3.bias"))
        self.assertNotIn("head.1.weight", dict(model.named_parameters()))
        self.assertIn("head.1.weight", dict(model.named_parameters(remove_duplicate=False)))
        names = backbone_parameters(model)
        self.assertIn("fc1.weight", names)
        self.assertIn("fc2.weight", names)
        self.assertFalse(any(n.startswith(("fc3.", "head.")) for n in names))
        model.fc3.weight.requires_grad_(False)
        with self.assertRaises(ValueError):
            classifier_names(model)

    def test_names_idempotent_and_default_unchanged(self):
        args = options()
        ensure_head_names(args)
        before = copy.deepcopy(vars(args))
        ensure_head_names(args)
        self.assertEqual(vars(args), before)
        for key in ("exp_name", "save_folder_name", "training_log_root", "h5_result_root", "final_model_root"):
            self.assertIn("avg_local_head", getattr(args, key))
        shared = options(target_head_mode="shared")
        before = copy.deepcopy(vars(shared))
        ensure_head_names(shared)
        self.assertEqual(vars(shared), before)

    def test_head_only_perturbations_cannot_affect_scoring_projection_or_aggregation(self):
        torch.set_num_threads(2)
        model = AliasModel().double()
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        backbone_names = list(backbone_parameters(model))
        weights = [.05] * 20
        uploads = []
        for cid in range(20):
            post = {n: p.clone() for n, p in initial.items()}
            post["fc1.weight"].reshape(-1)[0] += 1. if cid == 0 else -2. if cid % 2 else .5
            uploads.append((cid, weights[cid], post))
        changed = copy.deepcopy(uploads)
        for cid, _, post in changed:
            post["fc3.weight"].fill_(1e6 * (cid + 1))
            post["fc3.bias"].fill_(-1e6 * cid)
        before = copy.deepcopy(uploads)
        def filtered(source):
            return [(cid, w, {n: p[n] for n in backbone_names}) for cid, w, p in source]
        global_params = {n: initial[n] for n in backbone_names}
        for mode in ("avg", "projection_softmax"):
            def aggregate(source):
                data = filtered(source)
                if mode == "avg":
                    result, metrics, clients = aggregate_target_updates(global_params, data[0][2], iter(data), 0, "avg")
                    return result, metrics, clients
                return aggregate_projection_weighting(global_params, data[0][2], lambda: iter(data), 0,
                                                      "projection_softmax", self_weight=.125)[:3]
            rng = rng_state()
            a, metrics, clients = aggregate(uploads)
            b, changed_metrics, changed_clients = aggregate(changed)
            assert_rng(self, rng)
            self.assertEqual(metrics, changed_metrics)
            self.assertEqual(clients, changed_clients)
            self.assertEqual(metrics["target_delta_norm"], 1.)
            for n in a:
                torch.testing.assert_close(a[n], b[n], rtol=0, atol=0)
            if mode == "avg":
                for n in a:
                    expected = torch.zeros_like(a[n])
                    for _, w, p in uploads:
                        expected.add_(p[n] * w)
                    torch.testing.assert_close(a[n], expected, rtol=0, atol=0)
            else:
                self.assertEqual(clients[0]["aggregation_weight"], .125)
                expected = torch.zeros_like(global_params["fc1.weight"])
                for row in clients:
                    cid, w = row["client_id"], row["aggregation_weight"]
                    delta = uploads[cid][2]["fc1.weight"] - initial["fc1.weight"]
                    coefficient = min(float(delta.reshape(-1)[0]), 0.) / (1. + 1e-12) if cid else 0.
                    delta = delta.clone()
                    delta.reshape(-1)[0] -= coefficient
                    expected.add_(w * delta)
                torch.testing.assert_close(a["fc1.weight"], initial["fc1.weight"] + expected, atol=1e-14, rtol=1e-14)
                self.assertTrue(any(c["conflict"] for c in clients))
        for original, current in zip(before, uploads):
            for n in original[2]:
                torch.testing.assert_close(original[2][n], current[2][n], rtol=0, atol=0)

    def test_zero_backbone_reference_despite_nonzero_head_delta(self):
        initial = {"fc1.weight": torch.zeros(2), "fc2.bias": torch.ones(2)}
        data = [(0, .05, copy.deepcopy(initial))] + [(i, .05, {n: p + 1 for n, p in initial.items()}) for i in range(1, 20)]
        _, metrics, rows, _, _ = aggregate_projection_weighting(initial, data[0][2], lambda: iter(data),
                                                              0, "projection_softmax", self_weight=.125)
        self.assertEqual(metrics["target_delta_norm"], 0.)
        self.assertTrue(all(row["cosine_before_projection"] == 0 for row in rows))
        self.assertTrue(all(row["projection_coefficient"] == 0 for row in rows))
        self.assertTrue(all(abs(row["aggregation_weight"] - .875 / 19) < 1e-15 for row in rows[1:]))

    def test_launcher_passes_mode_and_keeps_shared_commands_identical(self):
        def preview(extra):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "avg", "--dry-run", *extra]), redirect_stdout(output):
                launcher.main()
            return shlex.split(output.getvalue().strip(), posix=False)
        default = preview([])
        shared = preview(["--target_head_mode", "shared"])
        # Timestamped roots vary; the shared switch itself must not be forwarded.
        self.assertNotIn("--target_head_mode", default)
        self.assertNotIn("--target_head_mode", shared)
        local = preview(["--target_head_mode", "local"])
        self.assertEqual(local[local.index("--target_head_mode") + 1], "local")
        for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir"):
            self.assertIn("local_head", local[local.index(key) + 1])
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "feedback_avg", "--target_head_mode", "local", "--dry-run"]), \
                patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            launcher.main()

    def test_main_real_parser_accepts_two_configs_and_rejects_invalid_without_training(self):
        path = ROOT / "system/main.py"
        script = '''
import ast, contextlib, io, json, pathlib, sys
from unittest.mock import patch
path = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
sys.path.insert(0, str(path.parent))
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
node.body = ast.parse("capture_args(args)").body
ast.fix_missing_locations(tree)
captured = []
sys.argv = [str(path), "-algo", "FedTargetProj", "--target_proj_mode", mode, "-m", "Decom_CNN-5-512",
            "-ncl", "100", "-nc", "20", "-ls", "5", "-eg", "1", "-gr", "100", "-dev", "cpu", "--target_head_mode", "local"]
if mode == "projection_softmax": sys.argv += ["--projection_self_weight", ".125"]
with contextlib.redirect_stdout(io.StringIO()), patch("utils.training_logs.training_log_session", side_effect=lambda a: contextlib.nullcontext()):
    exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path), "capture_args": captured.append})
print(json.dumps(vars(captured[0])))
'''
        for mode in ("avg", "projection_softmax"):
            process = subprocess.run([sys.executable, "-c", script, str(path), mode], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            args = json.loads(process.stdout)
            self.assertEqual(args["target_head_mode"], "local")
            self.assertEqual(args["target_prox_mu"], 0.)
            for key in ("exp_name", "save_folder_name", "training_log_root", "h5_result_root", "final_model_root"):
                self.assertIn("local_head", args[key])
        process = subprocess.run([sys.executable, "-c", script, str(path), "feedback_avg"], capture_output=True, text=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("error:", process.stderr)

    def test_formal_commands_use_101_rounds_and_independent_local_head_roots(self):
        text = (ROOT / "system/target_local_head_commands.sh").read_text()
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 2)
        roots = []
        for command, mode in zip(commands, ("avg", "projection_softmax")):
            args = shlex.split(command)
            for key, value in (("--target_head_mode", "local"), ("--target_proj_mode", mode),
                ("-gr", "100"), ("-ls", "5"), ("-lr", "0.005"), ("-lbs", "16"), ("-nc", "20"),
                ("-ncl", "100"), ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"), ("-eg", "1"),
                ("-jr", "1.0"), ("-pt", "pat"), ("-cpc", "20"), ("--seed", "0"),
                ("--target_client_id", "0"), ("-is_regular", "1"), ("-regular_lamda", "1e-3")):
                self.assertEqual(args[args.index(key) + 1], value)
            paths = [args[args.index(key) + 1] for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir")]
            self.assertTrue(all("local_head" in p and mode in p for p in paths))
            roots.append(paths)
            for key in ("--meta_c0_split", "--meta_collect_snapshots", "--target_prox_mu", "--feedback_anchor_weight", "--target_group_weight"):
                self.assertNotIn(key, args)
            if mode == "projection_softmax":
                self.assertEqual(args[args.index("--projection_self_weight") + 1], "0.125")
            else:
                self.assertNotIn("--projection_self_weight", args)
        self.assertTrue(all(a != b for a, b in zip(*roots)))


if __name__ == "__main__":
    unittest.main()
