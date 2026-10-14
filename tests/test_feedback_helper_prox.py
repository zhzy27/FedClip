"""Narrow helper Prox feedback policy, exact reused gradients, real CLI forwarding and commands."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import shlex
import sys
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.feedback_options import validate_feedback_options, ensure_feedback_names
from utils.target_prox_options import validate_target_prox, effective_prox_mu, ensure_prox_names, prox_tag
from utils.target_prox import make_prox_anchor, proximal_penalty, trainable_parameters
from test_meta_projection import SmallMetaNet
from test_target_proj_launcher import launcher


def feedback_args(mode="feedback_avg", mu=.01, scope="helpers", **extra):
    values = dict(algorithm="FedTargetProj", target_proj_mode=mode, target_client_id=0, num_clients=20,
        local_epochs=5, eval_gap=1, target_prox_mu=mu, target_prox_scope=scope, projection_self_weight=None)
    values.update(extra)
    return SimpleNamespace(**values)


class FeedbackHelperProxTests(unittest.TestCase):
    def test_two_validators_allow_only_helper_avg_and_preserve_old_projection_rules(self):
        for mu in (0., .001, .01, .1):
            args = feedback_args(mu=mu, target_prox_experiment=mu > 0)
            validate_feedback_options(args)
            validate_target_prox(mu, "helpers", mode="feedback_avg", self_weight=None)
            self.assertEqual(effective_prox_mu(args, 0), 0.)
            self.assertEqual([effective_prox_mu(args, cid) for cid in range(1, 20)], [mu] * 19)
        for mu in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_feedback_options(feedback_args(mu=mu))
            with self.assertRaises(ValueError):
                validate_target_prox(mu, "helpers", mode="feedback_avg", self_weight=None)
        for args in (feedback_args(scope="all"), feedback_args(mu=0., scope="all"),
                     feedback_args(mode="feedback_avg_self"), feedback_args(mode="feedback_projection_self"),
                     feedback_args(algorithm="FedCLIP"), feedback_args(meta_c0_split="split.json")):
            with self.assertRaises(ValueError):
                validate_feedback_options(args)
        for mode, scope in (("feedback_avg", "all"), ("feedback_avg_self", "helpers"),
                            ("feedback_projection_self", "helpers"), ("softmax_only", "helpers")):
            with self.assertRaises(ValueError):
                validate_target_prox(.01, scope, mode=mode, self_weight=None)
        for scope in ("helpers", "all"):
            validate_target_prox(.01, scope, mode="projection_softmax", self_weight=.125)
        with self.assertRaises(ValueError):
            validate_target_prox(.01, "helpers", mode="projection_softmax", self_weight=None)
        validate_feedback_options(feedback_args(mode="feedback_avg_self", mu=0.))

    def test_reused_anchor_and_penalty_exact_gradients_exclude_buffers_and_frozen_head_bias(self):
        model = SmallMetaNet(.6).double()
        model.register_buffer("buffer_probe", torch.tensor([7.]))
        model.head.bias.requires_grad_(False)
        anchor = make_prox_anchor(model)
        named = trainable_parameters(model)
        self.assertIn("fc.weight_u", anchor); self.assertIn("fc.weight_v", anchor)
        self.assertIn("head.weight", anchor)
        self.assertNotIn("head.bias", anchor); self.assertNotIn("buffer_probe", anchor)
        saved = {n: p.clone() for n, p in anchor.items()}
        loss = proximal_penalty(named, anchor, .1)
        gradients = torch.autograd.grad(loss, tuple(named.values()))
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in gradients))
        with torch.no_grad():
            for p in named.values():
                p.add_(.2)
        loss = proximal_penalty(named, anchor, .1)
        gradients = torch.autograd.grad(loss, tuple(named.values()))
        for (name, parameter), gradient in zip(named.items(), gradients):
            torch.testing.assert_close(gradient, .1 * (parameter - anchor[name]), atol=1e-15, rtol=1e-15)
            torch.testing.assert_close(anchor[name], saved[name], atol=0, rtol=0)
            self.assertFalse(anchor[name].requires_grad)
            self.assertNotEqual(anchor[name].data_ptr(), parameter.data_ptr())

    def test_names_scope_mu_positive_and_zero_baseline_unchanged(self):
        for mu in (.001, .01, .1):
            args = feedback_args(mu=mu, exp_name="trial", save_folder_name="temp", final_model_root="final",
                                 h5_result_root="h5", training_log_root="logs", run_log_dir="")
            ensure_feedback_names(args); ensure_prox_names(args)
            previous = copy.deepcopy(vars(args))
            ensure_feedback_names(args); ensure_prox_names(args)
            self.assertEqual(vars(args), previous)
            for key in ("exp_name", "save_folder_name", "final_model_root", "h5_result_root", "training_log_root"):
                self.assertIn("feedback_avg", getattr(args, key))
                self.assertIn(prox_tag(mu, "helpers"), getattr(args, key))
        zero = feedback_args(mu=0., exp_name="feedback_avg_helpers_only_local5", save_folder_name="temp/feedback_avg_helpers_only_local5")
        previous = vars(zero).copy()
        ensure_prox_names(zero); ensure_feedback_names(zero)
        self.assertEqual(vars(zero), previous)

    def test_launcher_forwards_both_flags_and_preserves_zero_names(self):
        for mu in (0., .001, .01, .1):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "feedback_avg", "--dry-run",
                    "--rounds", "200", "--target_prox_mu", str(mu), "--target_prox_scope", "helpers"]), redirect_stdout(output):
                launcher.main()
            text = output.getvalue()
            self.assertIn("--target_prox_mu " + str(mu), text)
            self.assertIn("--target_prox_scope helpers", text)
            self.assertIn("-gr 200", text)
            if mu:
                self.assertIn(prox_tag(mu, "helpers"), text)
            else:
                self.assertNotIn("prox_helpers_mu", text)
        for flags in (["--modes", "feedback_avg", "--target_prox_scope", "all"],
                      ["--modes", "feedback_avg_self", "--target_prox_mu", ".01"],
                      ["--modes", "feedback_projection_self"],
                      ["--modes", "feedback_avg", "--target_prox_mu", "nan"]):
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", *flags]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main()

    def test_main_real_parser_validation_and_names_without_running_training(self):
        # Use a fresh process like the launcher. Replace only run(args) and file
        # logging; imports/parser/policy/experiment markers/names are production.
        path = Path(__file__).resolve().parents[1] / "system/main.py"
        script = '''
import ast, contextlib, io, json, pathlib, sys
from unittest.mock import patch
path = pathlib.Path(sys.argv[1]); mu = sys.argv[2]
sys.path.insert(0, str(path.parent))
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
assert len(nodes) == 1
nodes[0].body = ast.parse("capture_args(args)").body
ast.fix_missing_locations(tree)
captured = []
sys.argv = [str(path), "-algo", "FedTargetProj", "--target_proj_mode", "feedback_avg", "-nc", "20",
            "-ls", "5", "-eg", "1", "-gr", "200", "-dev", "cpu", "--target_prox_mu", mu,
            "--target_prox_scope", "helpers"]
with contextlib.redirect_stdout(io.StringIO()), patch("utils.training_logs.training_log_session", side_effect=lambda a: contextlib.nullcontext()):
    exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path), "capture_args": captured.append})
assert len(captured) == 1
print(json.dumps(vars(captured[0])))
'''
        for mu in (0., .001, .01, .1):
            process = subprocess.run([sys.executable, "-c", script, str(path), str(mu)], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            args = SimpleNamespace(**json.loads(process.stdout))
            self.assertEqual(args.target_prox_mu, mu)
            self.assertEqual(args.target_prox_scope, "helpers")
            self.assertEqual(args.target_prox_experiment, mu > 0)
            if mu:
                for key in ("exp_name", "save_folder_name", "training_log_root", "h5_result_root", "final_model_root"):
                    self.assertIn(prox_tag(mu, "helpers"), getattr(args, key))
        for mode, mu, scope in (("feedback_avg", .1, "all"), ("feedback_avg_self", .01, "helpers"),
                                ("feedback_projection_self", 0., "helpers"), ("feedback_avg", -1., "helpers")):
            process = subprocess.run([sys.executable, str(path), "-algo", "FedTargetProj", "--target_proj_mode", mode,
                "-nc", "20", "-ls", "5", "--target_prox_mu", str(mu), "--target_prox_scope", scope], capture_output=True, text=True)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("error:", process.stderr)

    def test_three_formal_commands_configuration_gpu_roots_and_no_confounded_options(self):
        source = (Path(__file__).resolve().parents[1] / "system/feedback_helper_prox_commands.sh").read_text()
        commands = re.findall(r'^\s+"([^"]+)"$', source, re.MULTILINE)
        self.assertEqual(len(commands), 3)
        roots, gpus = [], []
        for command, mu in zip(commands, (.001, .01, .1)):
            args = shlex.split(command)
            for key, value in (("--target_proj_mode", "feedback_avg"), ("--target_prox_scope", "helpers"),
                ("--seed", "0"), ("--target_client_id", "0"), ("-t", "1"), ("-gr", "200"), ("-ls", "5"),
                ("-eg", "1"), ("-lr", "0.005"), ("-lbs", "16"), ("-nc", "20"), ("-ncl", "100"),
                ("-jr", "1.0"), ("-m", "Decom_CNN-5-512"), ("-data", "Cifar100"), ("-pt", "pat"),
                ("-cpc", "20"), ("-is_regular", "1"), ("-regular_lamda", "1e-3")):
                self.assertEqual(args[args.index(key) + 1], value)
            self.assertEqual(float(args[args.index("--target_prox_mu") + 1]), mu)
            paths = [args[args.index(key) + 1] for key in ("-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir")]
            self.assertTrue(all("feedback_avg" in p and prox_tag(mu, "helpers") in p for p in paths))
            self.assertTrue(paths[-1].startswith("training_logs/"))
            for key in ("-mu", "--projection_self_weight", "--feedback_anchor_weight", "--meta_c0_split"):
                self.assertNotIn(key, args)
            roots.append(tuple(paths)); gpus.append(args[args.index("-did") + 1])
        self.assertEqual(len(set(roots)), 3)
        self.assertEqual(len(set(gpus)), 3)


if __name__ == "__main__":
    unittest.main()
