"""FedRep-style protocol, active gradient scopes, restoration and observation isolation."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.clients.clientbase import save_item
from utils.target_local_schedule import validate_local_schedule, ensure_schedule_names
from utils.head_diagnostic_options import validate_head_diagnostic
from utils.head_diagnostic import diagnose_snapshot
from utils.head_snapshot import tensor_fingerprint, file_hash
from utils import fedrep_training
from test_meta_projection import rng_state, assert_rng
import run_target_proj as launcher


def options(**updates):
    values = dict(algorithm="FedTargetProj", target_proj_mode="avg", target_head_mode="local",
        target_local_schedule="fedrep", fedrep_head_epochs=5, local_epochs=5, model_family="Decom_CNN-5-512",
        num_classes=100, target_client_id=0, num_clients=20, batch_size=16, local_learning_rate=.005,
        is_regular=1, regular_lamda=.001, target_prox_mu=0., exp_name="avg_local_head", save_folder_name="temp",
        h5_result_root="h5", final_model_root="final", training_log_root="logs", eval_gap=1)
    values.update(updates)
    return SimpleNamespace(**values)


class AlternatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.bn = torch.nn.BatchNorm1d(8)
        self.fc2 = torch.nn.Linear(8, 512)
        self.fc3 = torch.nn.Linear(512, 100)
        self.relu = torch.nn.ReLU()
        self.base = torch.nn.Sequential(self.fc1, self.bn, self.relu, self.fc2)
        self.head = torch.nn.Sequential(self.relu, self.fc3)
        self.frozen_probe = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    def forward(self, x):
        return self.head(self.base(x))
    def frobenius_decay(self):
        return self.fc1.weight.square().sum() + self.fc2.weight.square().sum()


def make_client(folder, model, cid=0):
    client = clientTargetProj.__new__(clientTargetProj)
    client.args = options()
    client.id, client.role, client.device = cid, f"Client_{cid}", "cpu"
    client.learning_rate, client.local_epochs, client.batch_size = .005, 5, 16
    client.save_folder_name, client.loss = str(folder), torch.nn.CrossEntropyLoss()
    client.train_time_cost = dict(num_rounds=0, total_cost=0.)
    generator = torch.Generator().manual_seed(81)
    data = list(zip(torch.randn(16, 4, generator=generator), torch.arange(16) % 3))
    client.load_train_data = lambda: DataLoader(data, batch_size=16, shuffle=True, drop_last=False)
    client.load_test_data = lambda: DataLoader(data[:4], batch_size=16, shuffle=False)
    client._load_model = lambda: model
    save_item(model, client.role, "model", str(folder))
    return client


class FedRepTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_narrow_policy_and_joint_noop(self):
        validate_local_schedule(options())
        for changes in (dict(algorithm="FedAvg"), dict(target_head_mode="shared"), dict(target_proj_mode="projection_softmax"),
            dict(target_proj_mode="feedback_avg"), dict(target_proj_mode="group_avg_all"), dict(target_proj_mode="dwa_soft"),
            dict(target_proj_mode="meta_projection_fixed"), dict(target_prox_mu=.1), dict(target_prox_mu=float("nan")),
            dict(meta_c0_split="split.json"), dict(meta_collect_snapshots=True), dict(head_diag_collect=True),
            dict(head_diag_dir="snapshots"), dict(learning_rate_decay=True), dict(local_epochs=1), dict(fedrep_head_epochs=0),
            dict(fedrep_head_epochs=6), dict(fedrep_head_epochs=5.0), dict(num_classes=20), dict(target_client_id=1), dict(num_clients=3),
            dict(batch_size=32), dict(local_learning_rate=.1), dict(is_regular=0), dict(regular_lamda=.01),
            dict(models_folder_name="old"), dict(resume=True), dict(few_shot=1), dict(train_slow_rate=.1),
            dict(target_group_ids=[0]), dict(projection_self_weight=.125), dict(target_local_schedule="bad")):
            with self.subTest(changes=changes), self.assertRaises(ValueError): validate_local_schedule(options(**changes))
        args = options(target_local_schedule="joint", target_proj_mode="projection_softmax")
        before = copy.deepcopy(vars(args))
        validate_local_schedule(args)
        ensure_schedule_names(args)
        self.assertEqual(vars(args), before)
        validate_local_schedule(SimpleNamespace())
        ensure_schedule_names(SimpleNamespace())

    def test_independent_names_and_legacy_diagnostic_guard(self):
        args = options()
        ensure_schedule_names(args)
        before = copy.deepcopy(vars(args))
        ensure_schedule_names(args)
        self.assertEqual(vars(args), before)
        for key in ("exp_name", "save_folder_name", "h5_result_root", "final_model_root", "training_log_root"):
            self.assertIn("fedrep_head5_backbone5", getattr(args, key))
        with self.assertRaises(ValueError): validate_head_diagnostic(options(head_diag_collect=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "R1.pt"
            torch.save(dict(schema=1, stage="C0_post_local_before_aggregation", model=None,
                            metadata=dict(args=dict(target_local_schedule="fedrep"))), path)
            with self.assertRaisesRegex(ValueError, "joint head diagnostic"):
                diagnose_snapshot(path, directory)

    def test_phase_freezing_ce_f_gradients_clip_scopes_and_state_restoration(self):
        model = AlternatingModel()
        model.bn.eval()  # A mixed original state must be restored, not flattened.
        flags = [p.requires_grad for p in model.parameters()]
        modes = [m.training for m in model.modules()]
        native_phase, native_clip = fedrep_training.run_phase, torch.nn.utils.clip_grad_norm_
        phases, clips = [], []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            client = make_client(directory, model)
            source_hash = file_hash(Path(directory) / "Client_0_model.pt")
            head_ids = {id(p) for p in model.fc3.parameters()}
            def phase(c, candidate, loader, parameters, stage, epochs):
                before = {n: p.clone() for n, p in candidate.named_parameters()}
                buffers = {n: p.clone() for n, p in candidate.named_buffers()}
                if stage == "head":
                    self.assertTrue(all(id(p) in head_ids for p in parameters))
                    self.assertFalse(candidate.bn.training)
                    self.assertTrue(candidate.fc3.training)
                else:
                    self.assertTrue(all(id(p) not in head_ids for p in parameters))
                    self.assertTrue(candidate.bn.training)
                    self.assertFalse(candidate.fc3.training)
                    x, y = next(iter(loader))
                    ce, f = c._objective(candidate, x, y)
                    # Both losses reach backbone despite frozen fc3. This probe
                    # is in the test only and does not replace the actual step.
                    for objective in (ce, f):
                        grad = torch.autograd.grad(objective, (candidate.fc1.weight, candidate.fc2.weight), retain_graph=True)
                        self.assertTrue(all(torch.isfinite(g).all() and g.norm() > 0 for g in grad))
                output = native_phase(c, candidate, loader, parameters, stage, epochs)
                for n, p in candidate.named_parameters():
                    must_stay = id(p) not in head_ids if stage == "head" else id(p) in head_ids
                    if must_stay or n == "frozen_probe":
                        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
                if stage == "head":
                    for n, p in candidate.named_buffers(): torch.testing.assert_close(p, buffers[n], rtol=0, atol=0)
                    self.assertTrue(any(not torch.equal(p, before[n]) for n, p in candidate.named_parameters() if id(p) in head_ids))
                    self.assertEqual(output["train_mean_weighted_frobenius"], 0.)
                else:
                    self.assertTrue(any(not torch.equal(p, before[n]) for n, p in candidate.named_parameters() if id(p) not in head_ids and n != "frozen_probe"))
                    self.assertGreater(output["train_mean_weighted_frobenius"], 0.)
                phases.append(stage)
                return output
            def clip(parameters, *a, **kw):
                parameters = list(parameters)
                clips.append({id(p) for p in parameters})
                return native_clip(parameters, *a, **kw)
            def observation(c, candidate, stage):
                state, rng = tensor_fingerprint(candidate.state_dict()), rng_state()
                original = [m.training for m in candidate.modules()]
                result = native_observe(c, candidate, stage)
                self.assertEqual(state, tensor_fingerprint(candidate.state_dict()))
                self.assertEqual(original, [m.training for m in candidate.modules()])
                self.assertEqual(file_hash(Path(directory) / "Client_0_model.pt"), source_hash)
                assert_rng(self, rng)
                return result
            native_observe = fedrep_training.observe_endpoint
            with patch.object(fedrep_training, "run_phase", side_effect=phase), \
                    patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=clip), \
                    patch.object(fedrep_training, "observe_endpoint", side_effect=observation):
                client.train(0)
            self.assertEqual(phases, ["head", "backbone"])
            self.assertEqual(clips[:5], [head_ids] * 5)
            self.assertTrue(all(head_ids.isdisjoint(ids) for ids in clips[5:]))
            self.assertEqual([p.requires_grad for p in model.parameters()], flags)
            self.assertEqual([m.training for m in model.modules()], modes)
            stats = client.last_fedrep_train_stats
            self.assertEqual((stats["head_epochs"], stats["backbone_epochs"], stats["total_data_epochs"]), (5, 5, 10))
            self.assertEqual((stats["head_train_sample_visits"], stats["backbone_train_sample_visits"]), (80, 80))
            self.assertEqual(client.last_normal_train_stats["local_epochs"], 10)
            self.assertEqual(client.train_time_cost["num_rounds"], 1)

    def test_intermediate_observations_do_not_change_final_training_or_rng(self):
        template = AlternatingModel()
        initial_rng = rng_state()
        outputs = []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for observe in (True, False):
                model = copy.deepcopy(template)
                client = make_client(Path(directory) / str(observe), model)
                import random, numpy as np
                random.setstate(initial_rng[0]); np.random.set_state(initial_rng[1]); torch.set_rng_state(initial_rng[2])
                if torch.cuda.is_available(): torch.cuda.set_rng_state_all(initial_rng[3])
                def omitted(c, m, stage): return dict(stage=stage, observation_seconds=0.)
                with patch.object(fedrep_training, "observe_endpoint", wraps=fedrep_training.observe_endpoint) if observe else \
                        patch.object(fedrep_training, "observe_endpoint", side_effect=omitted):
                    client.train(0)
                outputs.append((tensor_fingerprint(model.state_dict()), rng_state()))
            self.assertEqual(outputs[0][0], outputs[1][0])
            assert_rng(self, outputs[0][1])

    def test_exception_restores_flags_modes_and_does_not_overwrite_checkpoint(self):
        for failed_phase in ("head", "backbone"):
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                model = AlternatingModel()
                model.bn.eval()
                client = make_client(directory, model)
                flags, modes = [p.requires_grad for p in model.parameters()], [m.training for m in model.modules()]
                before = file_hash(Path(directory) / "Client_0_model.pt")
                native = fedrep_training.run_phase
                def failure(c, m, loader, params, stage, epochs):
                    if stage == failed_phase: raise RuntimeError("synthetic phase failure")
                    return native(c, m, loader, params, stage, epochs)
                with patch.object(fedrep_training, "run_phase", side_effect=failure), self.assertRaises(RuntimeError):
                    client.train(0)
                self.assertEqual([p.requires_grad for p in model.parameters()], flags)
                self.assertEqual([m.training for m in model.modules()], modes)
                self.assertEqual(file_hash(Path(directory) / "Client_0_model.pt"), before)
                self.assertEqual(client.train_time_cost["num_rounds"], 0)

    def test_launcher_and_main_parameter_forwarding(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "avg", "--target_head_mode", "local",
                    "--target_local_schedule", "fedrep", "--fedrep_head_epochs", "5", "--rounds", "30", "--dry-run"]), redirect_stdout(output):
            launcher.main()
        command = shlex.split(output.getvalue().strip(), posix=False)
        self.assertEqual(command[command.index("--target_local_schedule") + 1], "fedrep")
        self.assertEqual(command[command.index("--fedrep_head_epochs") + 1], "5")
        for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir"):
            self.assertIn("fedrep_head5_backbone5", command[command.index(key) + 1])
        script = '''
import ast, contextlib, io, json, pathlib, sys
from unittest.mock import patch
path = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(path.parent))
tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
node.body = ast.parse("capture_args(args)").body; ast.fix_missing_locations(tree)
captured=[]
sys.argv=[str(path),"-algo","FedTargetProj","--target_proj_mode","avg","--target_head_mode","local",
 "--target_local_schedule","fedrep","--fedrep_head_epochs","5","-ls","5","-gr","30","-nc","20",
 "-ncl","100","-m","Decom_CNN-5-512","-lr",".005","-lbs","16","-regular_lamda",".001","-dev","cpu"]
with contextlib.redirect_stdout(io.StringIO()), patch("utils.training_logs.training_log_session", side_effect=lambda a: contextlib.nullcontext()):
 exec(compile(tree,str(path),"exec"),{"__name__":"__main__","__file__":str(path),"capture_args":captured.append})
print(json.dumps(vars(captured[0])))
'''
        process = subprocess.run([sys.executable, "-c", script, str(ROOT / "system/main.py")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        values = json.loads(process.stdout)
        self.assertEqual(values["target_local_schedule"], "fedrep")
        self.assertEqual(values["local_epochs"], 5)
        self.assertEqual(values["fedrep_head_epochs"], 5)
        self.assertFalse(values["head_diag_collect"])
        self.assertFalse(values["learning_rate_decay"])
        for key in ("exp_name", "save_folder_name", "h5_result_root", "final_model_root", "training_log_root"):
            self.assertIn("fedrep_head5_backbone5", values[key])

    def test_formal_command_keeps_public_configuration_and_explicit_5_plus_5_budget(self):
        text = (ROOT / "system/fedrep_local_commands.sh").read_text()
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 1)
        values = shlex.split(commands[0])
        for key, expected in (("--target_proj_mode", "avg"), ("--target_head_mode", "local"),
            ("--target_local_schedule", "fedrep"), ("--fedrep_head_epochs", "5"), ("-ls", "5"), ("-gr", "30"),
            ("-lr", "0.005"), ("-lbs", "16"), ("-nc", "20"), ("-ncl", "100"), ("-jr", "1.0"),
            ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"), ("-pt", "pat"), ("-cpc", "20"),
            ("-is_regular", "1"), ("-regular_lamda", "1e-3"), ("--seed", "0"), ("--target_client_id", "0"), ("-eg", "1")):
            self.assertEqual(values[values.index(key) + 1], expected)
        for key in ("-exp_name", "-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir"):
            self.assertIn("avg_local_head_fedrep_head5_backbone5", values[values.index(key) + 1])
        for key in ("--head_diag_collect", "--meta_c0_split", "--target_prox_mu", "-ld", "--projection_self_weight"):
            self.assertNotIn(key, values)


if __name__ == "__main__":
    unittest.main()
