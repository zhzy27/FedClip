"""Exact proximal gradient/anchors, old zero path, helpers isolation and experiment naming."""

from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import re
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.clients.clientbase import load_item, save_item
from utils.target_prox import make_prox_anchor, proximal_penalty, trainable_parameters, observe_prox_local_train
from utils.target_prox_options import validate_target_prox, effective_prox_mu, ensure_prox_names, prox_tag
from test_meta_projection import SmallMetaNet, rng_state, assert_rng
from test_target_proj_launcher import launcher


def make_client(folder, cid, mu=0., scope="helpers"):
    client = object.__new__(clientTargetProj)
    client.id, client.role, client.device = cid, f"Client_{cid}", "cpu"
    client.save_folder_name = str(folder)
    client.learning_rate, client.local_epochs, client.train_slow = .005, 5, False
    client.loss = torch.nn.CrossEntropyLoss()
    client.args = SimpleNamespace(algorithm="FedTargetProj", is_regular=1, regular_lamda=.001, target_prox_mu=mu,
                                 target_prox_scope=scope, target_client_id=0, mu=100.)
    client.train_time_cost = dict(num_rounds=0, total_cost=0.)
    generator = torch.Generator().manual_seed(48)
    batches = [(torch.randn(4, 6, generator=generator, dtype=torch.float64), torch.tensor([0, 1, 2, 0])) for _ in range(2)]
    client.load_train_data = lambda: batches
    return client


class TargetProxMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_zero_start_gradient_mu_difference_head_and_no_buffers_or_frozen(self):
        model = SmallMetaNet(.6).double()
        model.register_buffer("not_a_parameter", torch.tensor([12.]))
        model.head.bias.requires_grad_(False)
        anchor = make_prox_anchor(model)
        named = trainable_parameters(model)
        self.assertIn("fc.weight_u", anchor)
        self.assertIn("fc.weight_v", anchor)
        self.assertIn("head.weight", anchor)
        self.assertNotIn("not_a_parameter", anchor)
        self.assertNotIn("head.bias", anchor)
        self.assertTrue(all(not p.requires_grad for p in anchor.values()))
        start = proximal_penalty(named, anchor, .3)
        self.assertEqual(start.item(), 0.)
        start.backward()
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for p in named.values()))
        with torch.no_grad():
            for p in named.values():
                p.add_(.2)
        for p in named.values():
            p.grad = None
        loss = proximal_penalty(named, anchor, .3)
        expected = .3 / 2 * sum(p.numel() for p in named.values()) * .2 ** 2
        self.assertAlmostEqual(loss.item(), expected)
        loss.backward()
        for name, p in named.items():
            torch.testing.assert_close(p.grad, .3 * (p - anchor[name]), atol=1e-15, rtol=1e-15)

    def test_anchor_is_immutable_and_strict_parameter_mapping(self):
        model = SmallMetaNet(.6).double()
        anchor = make_prox_anchor(model)
        old = {n: p.clone() for n, p in anchor.items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        proximal_penalty(trainable_parameters(model), anchor, .5).backward()
        # Independent update exercises immutability even at zero prox gradient.
        for p in model.parameters():
            p.grad.add_(1.)
        optimizer.step()
        for name in anchor:
            torch.testing.assert_close(anchor[name], old[name], atol=0, rtol=0)
            self.assertNotEqual(anchor[name].data_ptr(), dict(model.named_parameters())[name].data_ptr())
        wrong = dict(anchor)
        wrong.pop("head.bias")
        with self.assertRaises(ValueError):
            proximal_penalty(trainable_parameters(model), wrong, .1)

    def test_zero_and_helpers_C0_are_exact_old_path_including_rng_and_legacy_mu(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            with torch.random.fork_rng():
                torch.manual_seed(33)
                original = SmallMetaNet(.6, dropout=.2).double()
            values, states = [], []
            for cid, mu, scope in ((0, 0., "helpers"), (1, 0., "all"), (0, 1., "helpers")):
                save_item(copy.deepcopy(original), f"Client_{cid}", "model", str(folder))
                client = make_client(folder, cid, mu, scope)
                before = rng_state()
                with patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=AssertionError("old path")), \
                        patch("flcore.clients.clientTargetProj.proximal_penalty", side_effect=AssertionError("old path")):
                    client.train()
                values.append({n: p.detach().clone() for n, p in load_item(client.role, "model", str(folder)).named_parameters()})
                states.append(torch.get_rng_state().clone())
                # Restore streams so each old-path comparison starts identically.
                import random, numpy as np
                random.setstate(before[0]); np.random.set_state(before[1]); torch.set_rng_state(before[2])
                if before[3]:
                    torch.cuda.set_rng_state_all(before[3])
                self.assertEqual(client.last_normal_train_stats["proximal_loss"], 0.)
            for other in values[1:]:
                for name in values[0]:
                    torch.testing.assert_close(other[name], values[0][name], atol=0, rtol=0)
            for other in states[1:]:
                torch.testing.assert_close(other, states[0], atol=0, rtol=0)

    def test_anchor_rebuilt_each_round_from_actual_download_and_current_loss_differentiable(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            model = SmallMetaNet(.6).double()
            save_item(model, "Client_1", "model", str(folder))
            client = make_client(folder, 1, 1.)
            anchors = []
            native = make_prox_anchor
            def capture(model):
                anchor = native(model)
                anchors.append((anchor, {n: p.clone() for n, p in anchor.items()}))
                return anchor
            with patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=capture):
                client.train(0)
                self.assertGreater(client.last_normal_train_stats["proximal_loss"], 0)
                next_start = load_item("Client_1", "model", str(folder))
                with torch.no_grad():
                    for p in next_start.parameters():
                        p.add_(.05)
                save_item(next_start, "Client_1", "model", str(folder))
                client.train(1)
            self.assertEqual(len(anchors), 2)
            for actual, expected in anchors:
                for name in actual:
                    torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
            for name, p in next_start.named_parameters():
                torch.testing.assert_close(anchors[1][0][name], p, atol=0, rtol=0)
            self.assertFalse(torch.equal(anchors[0][0]["head.weight"], anchors[1][0]["head.weight"]))

    def test_zero_path_matches_manual_legacy_SGD_with_regularization_and_clipping(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            client = make_client(folder, 1, 0.)
            initial = SmallMetaNet(.6, smooth=False).double()
            # Exercise active total-gradient clipping in the original formula.
            batches = [(x * 1000., y) for x, y in client.load_train_data()]
            client.load_train_data = lambda: batches
            model = copy.deepcopy(initial).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=.005)
            for _ in range(5):
                for x, y in batches:
                    optimizer.zero_grad()
                    loss = client.loss(model(x), y) + .001 * model.frobenius_decay()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    optimizer.step()
            save_item(initial, "Client_1", "model", str(folder))
            client.train()
            result = load_item("Client_1", "model", str(folder))
            for name, value in model.named_parameters():
                torch.testing.assert_close(dict(result.named_parameters())[name], value, atol=0, rtol=0)

    def test_cpu_zero_observation_preserves_training_rng_and_evaluation_objective(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            original = SmallMetaNet(.6, dropout=.2).double()
            direct = make_client(folder, 0)
            save_item(original, "Client_0", "model", str(folder))
            before = rng_state()
            direct.train()
            expected_rng = torch.get_rng_state().clone()
            expected = copy.deepcopy(load_item("Client_0", "model", str(folder)).state_dict())
            import random, numpy as np
            random.setstate(before[0]); np.random.set_state(before[1]); torch.set_rng_state(before[2])
            if before[3]:
                torch.cuda.set_rng_state_all(before[3])
            save_item(original, "Client_0", "model", str(folder))
            observed = make_client(folder, 0)
            row = observe_prox_local_train(observed, 0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
            for n, p in load_item("Client_0", "model", str(folder)).state_dict().items():
                torch.testing.assert_close(p, expected[n], atol=0, rtol=0)
            self.assertEqual(row["proximal_loss"], 0)
            self.assertGreater(row["low_rank_start_distance"], 0)
            # No prox term is added to _objective or train_metrics.
            model = load_item("Client_0", "model", str(folder)).eval()
            x, y = observed.load_train_data()[0]
            legacy = observed._objective(model, x, y)
            observed.args.target_prox_mu = 10.
            prox_configured = observed._objective(model, x, y)
            for a, b in zip(legacy, prox_configured):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


class TargetProxConfigTests(unittest.TestCase):
    def test_formal_five_command_array_budget_mass_tags_and_gpu_independence(self):
        text = (Path(__file__).resolve().parents[1] / "system" / "target_prox_commands.sh").read_text()
        self.assertIn("declare -a COMMANDS=(", text)
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 5)
        gpus, directories = [], []
        for command, mu in zip(commands, (0., .001, .01, .1, 1.)):
            args = shlex.split(command)
            for key, value in (("--target_client_id", "0"), ("--seed", "0"), ("-gr", "200"), ("-ls", "5"),
                ("-nc", "20"), ("-ncl", "100"), ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"),
                ("-algo", "FedTargetProj"), ("-lr", "0.005"), ("-lbs", "16"), ("-jr", "1.0"),
                ("-pt", "pat"), ("-cpc", "20"), ("-regular_lamda", "1e-3"),
                ("--target_proj_mode", "projection_softmax"), ("--target_prox_scope", "helpers"),
                ("--projection_self_weight", "0.125")):
                self.assertEqual(args[args.index(key) + 1], value)
            self.assertEqual(float(args[args.index("--target_prox_mu") + 1]), mu)
            tag = prox_tag(mu, "helpers")
            for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--training_log_root"):
                self.assertIn(tag, args[args.index(key) + 1])
            self.assertNotIn("-mu", args)
            self.assertNotIn("--meta_c0_split", args)
            gpus.append(args[args.index("-did") + 1])
            directories.append(args[args.index("-sfn") + 1])
        self.assertEqual(len(set(gpus)), 5)
        self.assertEqual(len(set(directories)), 5)

    def test_last20_summary_counts_earliest_best_and_includes_current_round(self):
        from flcore.servers.serverTargetProj import FedTargetProj
        server = object.__new__(FedTargetProj)
        server.args = SimpleNamespace(projection_self_weight=.125)
        server.target_proj_mode = "projection_softmax"
        server.target_prox_experiment = True
        values = [.1] * 5 + [.9, .9] + [.2] * 18
        server.target_proj_history = [dict(round=i + 1, target_post_local_acc=value) for i, value in enumerate(values)]
        summary = server._local_accuracy_summary()
        self.assertEqual(summary["best_target_local_round"], 6)
        self.assertEqual(summary["last20_target_local_count"], 20)
        self.assertAlmostEqual(summary["last20_target_local_acc"], sum(values[-20:]) / 20)
        summary = server._local_accuracy_summary(dict(round=26, target_post_local_acc=.3))
        self.assertAlmostEqual(summary["last20_target_local_acc"], sum((values + [.3])[-20:]) / 20)

    def test_validation_nonzero_and_no_old_mu_coupling(self):
        for mu in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_target_prox(mu, "helpers")
        for mode in ("avg", "projection", "dwa_soft", "apa", "group_projection_softmax", "meta_projection_fixed"):
            validate_target_prox(0., "all", mode=mode)
            with self.assertRaises(ValueError):
                validate_target_prox(.01, "helpers", mode=mode)
        for kwargs in (dict(target_id=1), dict(self_weight=None), dict(self_weight=.2),
                       dict(meta_split="split"), dict(collect=True), dict(algorithm="FedCLIP")):
            with self.assertRaises(ValueError):
                validate_target_prox(.01, "helpers", **kwargs)
        validate_target_prox(.01, "all")
        self.assertEqual(effective_prox_mu(SimpleNamespace(mu=1.), 1), 0.)

    def test_names_contain_mu_scope_and_old_experiment_names_unchanged(self):
        for mu in (0., .001, .01, .1, 1.):
            args = SimpleNamespace(target_prox_mu=mu, target_prox_scope="helpers", target_prox_experiment=True,
                exp_name="self0p125", save_folder_name="temp", h5_result_root="h5_results",
                final_model_root="final_models", training_log_root="training_logs", run_log_dir="")
            ensure_prox_names(args)
            tag = prox_tag(mu, "helpers")
            for key in ("exp_name", "save_folder_name", "h5_result_root", "final_model_root", "training_log_root"):
                self.assertIn(tag, getattr(args, key))
            old = copy.deepcopy(args)
            ensure_prox_names(args)
            self.assertEqual(vars(args), vars(old))
        args = SimpleNamespace(exp_name="old", save_folder_name="temp")
        ensure_prox_names(args)
        self.assertEqual(vars(args), {"exp_name": "old", "save_folder_name": "temp"})

    def test_launcher_mu_zero_tag_and_positive_all_interface(self):
        for mu, scope in (("0", "helpers"), (".01", "helpers"), (".01", "all")):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "projection_softmax",
                    "--projection_self_weight", ".125", "--target_prox_mu", mu, "--target_prox_scope", scope,
                    "--rounds", "200", "--dry-run"]), redirect_stdout(output):
                launcher.main()
            text = output.getvalue()
            self.assertIn(prox_tag(float(mu), scope), text)
            self.assertIn("-gr 200", text)
            self.assertIn("--projection_self_weight 0.125", text)
            self.assertIn("--target_prox_mu", text)
            self.assertIn("--target_prox_scope " + scope, text)


if __name__ == "__main__":
    unittest.main()
