"""Full unroll, genuine SVD hypergradients, data provenance and offline selection."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.trainmodel.models import FactorizedLinear, Decom_LINEAR, Recover_LINEAR
from utils.meta_data import create_split, data_fingerprint, save_split, validate_split
from utils.meta_virtual import (adapted_validation_loss, differentiable_decompose, differentiable_projection,
    functional_train, preserved_rng, projection_coefficients)
from utils.meta_snapshot import file_hash, model_signature, load_fixed_weights
from utils.meta_learning import MetaProblem, optimize
from utils.target_projection import aggregate_target_updates


class SmallMetaNet(nn.Module):
    def __init__(self, ratio=.6, smooth=True, dropout=0.):
        super().__init__()
        self.ratio_LR, self.smooth = ratio, smooth
        self.fc = FactorizedLinear(6, 5, ratio) if ratio < 1 else nn.Linear(6, 5)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc(x)
        return self.head(self.dropout(torch.tanh(x) if self.smooth else x))

    def frobenius_decay(self):
        return self.fc.frobenius_loss() if hasattr(self.fc, "weight_u") else self.fc.weight.new_zeros(())

    def recover_larger_model(self):
        if hasattr(self.fc, "weight_u"):
            self.fc = Recover_LINEAR(self.fc)

    def decom_larger_model(self, ratio):
        if isinstance(self.fc, nn.Linear):
            self.fc = Decom_LINEAR(self.fc, ratio)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def assert_rng(test, before):
    test.assertEqual(random.getstate(), before[0])
    np.testing.assert_array_equal(np.random.get_state()[1], before[1][1])
    torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)
    for state, expected in zip(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], before[3]):
        torch.testing.assert_close(state, expected, rtol=0, atol=0)


def tiny_snapshot_collection(root):
    root = Path(root)
    root.mkdir()
    generator = torch.Generator().manual_seed(41)
    x, y = torch.randn(30, 6, generator=generator), torch.arange(30) % 3
    data_root = root / "dataset"
    shard = data_root / "Cifar100" / "pat_20" / "train"
    shard.mkdir(parents=True)
    np.savez(shard / "0.npz", data=dict(x=x.numpy(), y=y.numpy()))
    split = create_split(list(zip(x, y)))
    config = dict(dataset="Cifar100", dataset_subdir="pat_20", model_family="SmallMetaNet",
        num_clients=20, num_classes=3, local_epochs=5, batch_size=16, local_lr=.005, regularization=.001, is_regular=1)
    with preserved_rng(9):
        full, template = SmallMetaNet(1.), SmallMetaNet(.6)
    capacities = [dict(client_id=cid, **model_signature(template)) for cid in range(20)]
    rows = []
    for round_number in (20, 50):
        folder = root / f"R{round_number}"
        folder.mkdir()
        initial = {n: p.detach().clone() for n, p in full.named_parameters()}
        posts = [{n: p + torch.randn(p.shape, generator=generator) * .1 for n, p in initial.items()} for _ in range(20)]
        torch.save(initial, folder / "global.pt")
        torch.save(template, folder / "c0_template.pt")
        files = {name: file_hash(folder / name) for name in ("global.pt", "c0_template.pt")}
        for cid, post in enumerate(posts):
            name = f"client_{cid}.pt"
            torch.save(post, folder / name)
            files[name] = file_hash(folder / name)
        meta = dict(schema=1, round=round_number, loop_round=round_number - 1, training_seed=0,
            client_ids=list(range(20)), config=config, capacities=capacities, split=split, files=files,
            projection_epsilon=1e-12, projection_coefficients=projection_coefficients(initial, posts[0], enumerate(posts)),
            split_active_before_first_local_training=True)
        (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        rows.append(dict(round=round_number, folder=folder.name, metadata_sha256=file_hash(folder / "metadata.json")))
    (root / "manifest.json").write_text(json.dumps(dict(schema=1, config=config, capacities=capacities, split=split, snapshots=rows)), encoding="utf-8")
    return data_root, split, template, config


class MetaDataTests(unittest.TestCase):
    def test_stratified_split_reproducible_disjoint_independent_rng_and_immutable(self):
        data = [(torch.tensor([float(i), .1]), torch.tensor(i % 3)) for i in range(30)]
        before = rng_state()
        record = create_split(data)
        assert_rng(self, before)
        self.assertEqual(len(record["train_indices"]), 24)
        self.assertEqual(len(record["validation_indices"]), 6)
        with preserved_rng(999):
            self.assertEqual(create_split(data), record)
        validate_split(record, data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.json"
            save_split(path, record)
            save_split(path, record)
            with self.assertRaises(ValueError):
                save_split(path, create_split(data, split_seed=2))

    def test_fingerprint_and_index_tampering_rejected(self):
        data = [(torch.randn(3), torch.tensor(i % 2)) for i in range(20)]
        record = create_split(data)
        changed = copy.deepcopy(data)
        changed[0][0].add_(1.)
        with self.assertRaises(ValueError):
            validate_split(record, changed)
        record["validation_indices"][0] = record["train_indices"][0]
        with self.assertRaises(ValueError):
            validate_split(record, data)


class MetaVirtualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        with preserved_rng(71):
            self.full = SmallMetaNet(1.).double()
            self.template = SmallMetaNet(.6).double()
            self.initial = {n: p.detach().clone() for n, p in self.full.named_parameters()}
            self.posts = [{n: p + torch.randn_like(p) * .08 for n, p in self.initial.items()} for _ in range(4)]
            self.coefficients = projection_coefficients(self.initial, self.posts[0], enumerate(self.posts))
            self.train = [(torch.randn(4, 6, dtype=torch.float64), torch.randint(0, 3, (4,))) for _ in range(3)]
            self.val = [(torch.randn(5, 6, dtype=torch.float64), torch.randint(0, 3, (5,)))]

    def candidate(self, z):
        return differentiable_projection(self.initial, lambda cid: self.posts[cid], z.softmax(0), self.coefficients, list(range(4)))

    def objective(self, z, template=None, train=None, checkpoint_steps=0):
        return adapted_validation_loss(template or self.template, self.candidate(z), train or self.train,
                                       self.val, seed=119, checkpoint_steps=checkpoint_steps)[0]

    def test_differentiable_projection_forward_matches_original_kernel(self):
        z = torch.tensor([.1, -.2, .2, -.1], dtype=torch.float64, requires_grad=True)
        expected = aggregate_target_updates(self.initial, self.posts[0],
            [(cid, z.softmax(0)[cid].item(), self.posts[cid]) for cid in range(4)], 0, "projection")[0]
        actual = self.candidate(z)
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-14, atol=1e-14)
        self.assertTrue(all(p.requires_grad for p in actual.values()))

    def test_svd_factors_reconstruction_matches_baseline_rank_and_scaling(self):
        full = {n: p.float() for n, p in self.initial.items()}
        template = self.template.float()
        actual, _ = differentiable_decompose(full, template)
        baseline = copy.deepcopy(self.full).float()
        baseline.decom_larger_model(.6)
        for name, p in baseline.named_parameters():
            torch.testing.assert_close(actual[name], p, rtol=0, atol=0)
        reconstructed = actual["fc.weight_u"] @ actual["fc.weight_v"]
        torch.testing.assert_close(reconstructed, baseline.fc.reconstruct_full_weight(), rtol=0, atol=0)

    def test_functional_sgd_matches_ordinary_sgd_including_regularization_and_active_clip(self):
        for active_clip in (False, True):
            template = SmallMetaNet(.6, smooth=not active_clip).double()
            parameters = {n: p.detach().clone().requires_grad_(True) for n, p in template.named_parameters()}
            ordinary = copy.deepcopy(template)
            batches = [(x * (1000 if active_clip else 1), y) for x, y in self.train]
            optimizer = torch.optim.SGD(ordinary.parameters(), lr=.005)
            clipped = False
            for x, y in batches:
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(ordinary(x), y) + .001 * ordinary.frobenius_decay()
                loss.backward()
                clipped |= bool(torch.nn.utils.clip_grad_norm_(ordinary.parameters(), 10.) > 10.)
                optimizer.step()
            _, functional, _, details = functional_train(copy.deepcopy(template), parameters, batches)
            for name, p in ordinary.named_parameters():
                torch.testing.assert_close(functional[name], p, rtol=1e-12, atol=1e-12)
            if active_clip:
                self.assertTrue(clipped)
                self.assertGreater(details["clipped_step_count"], 0)

    def test_full_hypergradient_matches_central_finite_difference(self):
        z = torch.tensor([.1, -.1, .2, 0.], dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(self.objective(z), z)[0]
        epsilon = 1e-5
        numerical = []
        for index in range(len(z)):
            direction = torch.zeros_like(z)
            direction[index] = epsilon
            plus = self.objective((z.detach() + direction).requires_grad_(True)).item()
            minus = self.objective((z.detach() - direction).requires_grad_(True)).item()
            numerical.append((plus - minus) / (2 * epsilon))
        torch.testing.assert_close(gradient, torch.tensor(numerical, dtype=torch.float64), rtol=1e-5, atol=1e-8)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.norm(), 0.)

    def test_recompute_has_same_complete_gradient_including_dropout_rng(self):
        template = copy.deepcopy(self.template)
        template.dropout.p = .2
        gradients, values = [], []
        for checkpoint_steps in (0, 1):
            z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
            loss = self.objective(z, template=template, checkpoint_steps=checkpoint_steps)
            values.append(loss.detach())
            gradients.append(torch.autograd.grad(loss, z)[0])
        torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-11, atol=1e-12)

    def test_hypergradient_through_active_gradient_clipping_matches_finite_difference(self):
        template = copy.deepcopy(self.template)
        template.smooth = False
        train = [(x * 1000., y) for x, y in self.train]
        z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        loss = self.objective(z, template=template, train=train)
        gradient = torch.autograd.grad(loss, z)[0]
        direction = torch.tensor([.2, -.4, .1, .3], dtype=torch.float64)
        epsilon = 1e-5
        plus = self.objective((z.detach() + epsilon * direction).requires_grad_(True), template, train).item()
        minus = self.objective((z.detach() - epsilon * direction).requires_grad_(True), template, train).item()
        self.assertAlmostEqual((gradient * direction).sum().item(), (plus - minus) / (2 * epsilon), places=7)

    def test_virtual_forward_backward_preserves_normal_template_and_rng(self):
        old = copy.deepcopy(self.template.state_dict())
        before = rng_state()
        z = torch.zeros(4, dtype=torch.float64, requires_grad=True)
        self.objective(z).backward()
        assert_rng(self, before)
        for name, p in self.template.state_dict().items():
            torch.testing.assert_close(p, old[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in self.template.parameters()))


class MetaOptimizationTests(unittest.TestCase):
    def test_degenerate_native_svd_failure_is_logged_without_approximation_or_artifact(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "snapshots"
            data_root, _, _, _ = tiny_snapshot_collection(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for snapshot in manifest["snapshots"]:
                folder = root / snapshot["folder"]
                metadata = json.loads((folder / "metadata.json").read_text())
                for name in list(metadata["files"]):
                    if name == "c0_template.pt":
                        continue
                    params = torch.load(folder / name, weights_only=True)
                    params["fc.weight"].zero_()
                    torch.save(params, folder / name)
                    metadata["files"][name] = file_hash(folder / name)
                metadata["projection_coefficients"] = {str(cid): 0. for cid in range(20)}
                (folder / "metadata.json").write_text(json.dumps(metadata))
                snapshot["metadata_sha256"] = file_hash(folder / "metadata.json")
            (root / "manifest.json").write_text(json.dumps(manifest))
            problem = MetaProblem(root, data_root, dtype=torch.float64)
            output = Path(directory) / "failure"
            with self.assertRaises(RuntimeError):
                optimize(problem, output, updates=1, initializations=("uniform",))
            self.assertFalse((output / "fixed_weights.json").exists())
            log = json.loads((output / "fixed_uniform_trajectory.json").read_text())
            self.assertEqual(log["status"], "failed")
            self.assertTrue(log["history"][-1]["nonfinite"])
            self.assertIn("virtual_details", log["history"][-1])

    def test_fixed_and_per_snapshot_equal_budget_validation_artifacts_and_compatibility(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "snapshots"
            data_root, split, template, config = tiny_snapshot_collection(root)
            problem = MetaProblem(root, data_root, dtype=torch.float64, checkpoint_steps=1)
            before = rng_state()
            for strategy in ("fixed", "per_snapshot"):
                artifacts = optimize(problem, Path(directory) / strategy, strategy, updates=2, outer_lr=.05)
                self.assertEqual(len(artifacts), 1 if strategy == "fixed" else 2)
                for artifact in artifacts.values():
                    self.assertFalse(artifact["selection"]["test_used"])
                    self.assertEqual(artifact["optimization_budget"]["updates"], 2)
                    self.assertEqual(len(artifact["weights"]), 20)
                    self.assertAlmostEqual(sum(artifact["weights"]), 1.)
            assert_rng(self, before)
            args = SimpleNamespace(model_family="SmallMetaNet", num_classes=3)
            clients = [SimpleNamespace(id=cid, _load_model=lambda: copy.deepcopy(template), local_epochs=5,
                batch_size=16, learning_rate=.005, args=SimpleNamespace(is_regular=1, regular_lamda=.001)) for cid in range(20)]
            server = SimpleNamespace(args=args, clients=clients, dataset="Cifar100", num_clients=20, meta_split_record=split)
            path = Path(directory) / "fixed" / "fixed_weights.json"
            artifact, weights = load_fixed_weights(path, server)
            self.assertAlmostEqual(sum(weights.values()), 1.)
            wrong = copy.deepcopy(split)
            wrong["split_id"] = "different"
            server.meta_split_record = wrong
            with self.assertRaises(ValueError):
                load_fixed_weights(path, server)
            server.meta_split_record = split
            server.args.model_family = "WrongModel"
            with self.assertRaises(ValueError):
                load_fixed_weights(path, server)
            server.args.model_family = "SmallMetaNet"
            server.clients[2]._load_model = lambda: SmallMetaNet(.8)
            with self.assertRaises(ValueError):
                load_fixed_weights(path, server)
            server.clients[2]._load_model = lambda: copy.deepcopy(template)
            with self.assertRaises(ValueError):
                load_fixed_weights(Path(directory) / "per_snapshot" / "R20_weights.json", server)
            trajectory = json.loads((Path(directory) / "fixed" / "fixed_uniform_trajectory.json").read_text())
            self.assertEqual(len(trajectory["history"]), 3)
            self.assertEqual(set(trajectory["history"][0]["snapshot_adapted_validation_ce"]), {"R20", "R50"})
            self.assertGreater(trajectory["history"][0]["z_gradient_norm"], 0.)


if __name__ == "__main__":
    unittest.main()
