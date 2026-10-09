"""Numerical rejection, dtype preservation and continuous clipped-CNN trajectories."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from test_meta_projection import SmallMetaNet, tiny_snapshot_collection, rng_state, assert_rng
from utils.meta_benchmark import ordinary_initialization
from utils.meta_diagnostics import (DEFAULT_TOLERANCES, configuration, diagnose, fd_assessment,
    compare_stage, override_for, source_signature, trajectory_probe, validate_acceptance, write_json)
from utils.meta_learning import MetaProblem, optimize
from utils.meta_virtual import backend_state, differentiable_decompose, snapshot_backend, snapshot_candidate
from utils.meta_virtual import _Objective


class MetaDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_native_double_reference_does_not_round_through_default_float(self):
        torch.manual_seed(19)
        full = SmallMetaNet(1.).double()
        template = SmallMetaNet(.6).double()
        parameters = {n: p.detach().clone() for n, p in full.named_parameters()}
        expected, _ = differentiable_decompose(parameters, template)
        default = torch.get_default_dtype()
        before = rng_state()
        ordinary = ordinary_initialization(template, parameters)
        assert_rng(self, before)
        self.assertEqual(torch.get_default_dtype(), default)
        for name, p in ordinary.named_parameters():
            self.assertEqual(p.dtype, torch.float64)
            torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
        contaminated = copy.deepcopy(full)
        contaminated.decom_larger_model(.6)
        contaminated.double()
        self.assertGreater((contaminated.fc.weight_u - expected["fc.weight_u"]).abs().max().item(), 1e-10)

    def test_candidate_uses_saved_nontrivial_aggregation_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root)
            snapshot = problem.snapshots[0]
            order = list(reversed(snapshot.client_ids))
            snapshot.metadata["aggregation_order"] = order
            with patch("utils.meta_virtual.differentiable_projection", return_value={"sentinel": torch.tensor(0.)}) as aggregation:
                snapshot_candidate(snapshot, torch.zeros(20, requires_grad=True))
                self.assertEqual(aggregation.call_args.args[-1], order)

    def test_fd_requires_resolved_adjacent_interval_not_one_good_h(self):
        good = dict(analytic=1., central_difference=1.)
        bad = dict(analytic=1., central_difference=-5.)
        isolated = [dict(h=1e-2, **bad), dict(h=1e-3, **good), dict(h=1e-4, **bad)]
        self.assertEqual(fd_assessment(isolated, 0., DEFAULT_TOLERANCES)["status"], "unconfirmed")
        consecutive = [dict(h=1e-2, **bad), dict(h=1e-3, **good), dict(h=1e-4, **good)]
        self.assertEqual(fd_assessment(consecutive, 0., DEFAULT_TOLERANCES)["status"], "passed")

    def test_observed_wrong_sign_and_repeat_noise_cannot_pass(self):
        rows = [dict(h=h, analytic=-.0006453907, central_difference=5.3140623) for h in (1e-2, 1e-3, 1e-4)]
        self.assertEqual(fd_assessment(rows, 0., DEFAULT_TOLERANCES)["status"], "unconfirmed")
        noisy = [dict(h=h, analytic=.001, central_difference=.001) for h in (1e-2, 1e-3, 1e-4)]
        self.assertEqual(fd_assessment(noisy, .01, DEFAULT_TOLERANCES)["status"], "unconfirmed")
        self.assertTrue(all(not r["resolved"] for r in noisy))

    def test_override_applies_after_snapshot_and_restores_effective_backend(self):
        previous = backend_state()
        source = dict(cudnn_benchmark=True, cudnn_deterministic=False, cudnn_allow_tf32=True, matmul_allow_tf32=True)
        with snapshot_backend(source, override_for("deterministic_no_tf32")):
            self.assertEqual(backend_state(), dict(cudnn_benchmark=False, cudnn_deterministic=True,
                cudnn_allow_tf32=False, matmul_allow_tf32=False))
        self.assertEqual(backend_state(), previous)
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("medium")
            with snapshot_backend(source, override_for("no_tf32")):
                self.assertFalse(backend_state()["matmul_allow_tf32"])
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            torch.set_float32_matmul_precision(precision)

    def test_factor_sign_difference_alone_does_not_fail_forward_acceptance(self):
        model = SmallMetaNet(.6).double()
        parameters = {n: p.detach().clone() for n, p in model.named_parameters()}
        parameters["fc.weight_u"] *= -1
        parameters["fc.weight_v"] *= -1
        validation = [(torch.randn(4, 6, dtype=torch.float64), torch.tensor([0, 1, 2, 0]))]
        row = compare_stage(model, _Objective(copy.deepcopy(model)), parameters, {},
                            validation, 0, DEFAULT_TOLERANCES)
        self.assertGreater(row["parameters"]["max_absolute_error"], .1)
        self.assertEqual(row["status"], "passed")

    def test_failed_uncertain_synthetic_stale_and_wrong_config_block_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root, dtype=torch.float64, checkpoint_steps=8)
            report = dict(schema=2, status="passed", acceptance_scope="snapshot_and_configuration_bound",
                configuration=configuration(problem), diagnostic_budget=dict(initializations=["uniform"],
                bias_client=0, bias_logit=.25), snapshots=[dict(source=source_signature(s), status="passed") for s in problem.snapshots])
            path = Path(directory) / "report.json"
            write_json(path, report)
            self.assertEqual(validate_acceptance(problem, path, ["uniform"], 0, .25)["status"], "passed")
            for changed in (dict(status="failed"), dict(status="unconfirmed"), dict(acceptance_scope="synthetic_structure_only"),
                            dict(configuration={}), dict(snapshots=report["snapshots"][:1])):
                write_json(path, {**report, **changed})
                output = Path(directory) / "must_not_exist"
                with self.assertRaises(ValueError):
                    optimize(problem, output, updates=1, initializations=["uniform"], acceptance_report=path)
                self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                optimize(problem, Path(directory) / "missing", updates=1)

    def test_unaccepted_diagnostic_cannot_export_formal_filename_or_kind(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root, dtype=torch.float64)
            output = Path(directory) / "unaccepted"
            result = optimize(problem, output, updates=1, initializations=["uniform"], diagnostic_only=True)
            self.assertFalse((output / "fixed_weights.json").exists())
            self.assertEqual(result["fixed"]["kind"], "meta_projection_unaccepted_diagnostic")
            self.assertEqual(result["fixed"]["numerical_acceptance"]["status"], "not_accepted")

    def test_forward_only_is_incomplete_and_normal_reference_has_no_probe_overhead(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root, dtype=torch.float64, checkpoint_steps=8)
            path = Path(directory) / "forward.json"
            report = diagnose(problem, path, rounds=[20], initializations=["uniform"],
                              forward_only=True, raise_on_failure=False)
            self.assertEqual(report["status"], "unconfirmed")
            self.assertEqual(report["acceptance_scope"], "incomplete_exploratory")
            full = report["snapshots"][0]["candidates"][0]["full_five_epoch"]
            self.assertEqual(full["ordinary_comparison"]["status"], "passed")
            self.assertEqual(full["ordinary_timing"]["ordinary_train_batches"], 10)
            with self.assertRaises(ValueError):
                validate_acceptance(problem, path, ["uniform"], 0, .25)

    def test_stage_failure_is_written_and_raises_even_with_finite_nonzero_gradient(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root, dtype=torch.float64, checkpoint_steps=8)
            output = Path(directory) / "failure.json"
            from utils.meta_diagnostics import compare_stage
            def corrupt(*args, **kwargs):
                result = compare_stage(*args, **kwargs)
                result.update(status="failed", ce_absolute_error=.00315)
                return result
            with patch("utils.meta_diagnostics.compare_stage", side_effect=corrupt):
                with self.assertRaises(RuntimeError):
                    diagnose(problem, output, rounds=[20], initializations=["uniform"],
                             directions=2, repeats=2, h_values=[1e-2, 1e-3, 1e-4])
            report = json.loads(output.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["snapshots"][0]["candidates"][0]["full_five_epoch"]["nonzero"])

    def test_actual_cnn_sustained_clipping_stage_and_rng_isolation(self):
        from flcore.trainmodel.models import Hyper_CNN_512
        from utils.meta_virtual import preserved_rng
        with preserved_rng(79):
            full = Hyper_CNN_512(num_classes=100, ratio_LR=1.)
            template = Hyper_CNN_512(num_classes=100, ratio_LR=.9)
            parameters = {n: p.detach().clone() for n, p in full.named_parameters()}
            batches = [(torch.randn(2, 3, 32, 32) * 200., torch.tensor([0, 1])) for _ in range(10)]
        before = rng_state()
        old = copy.deepcopy(template.state_dict())
        with redirect_stdout(io.StringIO()):
            report = trajectory_probe(template, parameters, batches, batches[:1], 123, DEFAULT_TOLERANCES)
        assert_rng(self, before)
        self.assertEqual([s["step"] for s in report["stages"]], [0, 1, 2, 8, 10])
        self.assertGreater(report["ordinary_clipped_steps"], 0)
        self.assertTrue(all(s["status"] == "passed" for s in report["stages"]))
        for name, p in template.state_dict().items():
            torch.testing.assert_close(p, old[name], rtol=0, atol=0)
        self.assertIsNotNone(report["gradient_steps"][0]["same_point_gradients"])


if __name__ == "__main__":
    unittest.main()
