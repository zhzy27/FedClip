"""Data/loader/CNN/metric parity against the pre-refactor 189bacd sources."""
import ast
from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from common.data_io import (NPZ_ONLY, PKL_FIRST, TENSOR_LIST, NUMPY_STACK_LIST,
                            read_shard, select_client_shard, decode_image_data)
from common.data_loading import make_data_loader
from common.models import FedAvgCNN
from common.metrics import sample_weighted_accuracy, summarize_evaluations
from utils import data_utils as clip_data
from flcore.clients import clientbase as clip_clients
from flcore.trainmodel import models as clip_models
from fedfews_compat.vendor import clientbase as few_clients, data_utils as few_data, models as few_models
from fedfews_compat.observe import Recorder, rng_digest, rng_state, set_rng, summary
from fedfews_compat.data import audit_data
from fedfews_compat.acceptance import synthetic_data
from fedfews_compat.config import parse_args
from fedfews_compat.metadata import sha256, tensor_digest


def legacy_code(path):
    return subprocess.check_output(["git", "show", f"189bacd:{path}"], cwd=ROOT, text=True, encoding="utf-8")


def legacy_module(path):
    namespace = dict(__file__=str(ROOT / path), __name__="legacy_189_" + Path(path).stem)
    exec(compile(legacy_code(path), "189bacd:" + path, "exec"), namespace)
    return namespace


def legacy_class(path, name, supplied=None):
    node = next(n for n in ast.parse(legacy_code(path)).body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(torch=torch, nn=torch.nn, np=np, copy=copy, os=os, DataLoader=DataLoader)
    namespace.update(supplied or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), "189bacd:" + path, "exec"), namespace)
    return namespace[name], namespace


def assert_data(test, a, b):
    test.assertEqual(len(a), len(b))
    for (x, y), (u, v) in zip(a, b):
        if isinstance(x, tuple):
            for p, q in zip(x, u):
                test.assertTrue(torch.equal(p, q))
        else:
            test.assertTrue(torch.equal(x, u))
            test.assertEqual(x.dtype, u.dtype)
        test.assertTrue(torch.equal(y, v))
        test.assertEqual(y.dtype, v.dtype)


class SharedPrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_diagnostic_backend_setting_is_restored_on_failure(self):
        from fedfews_compat import acceptance
        previous = torch.backends.cudnn.deterministic
        def fail(argv):
            torch.backends.cudnn.deterministic = not previous
            raise RuntimeError("diagnostic failure")
        with patch.object(acceptance, "_run_acceptance", side_effect=fail), self.assertRaises(RuntimeError):
            acceptance.main([])
        self.assertEqual(torch.backends.cudnn.deterministic, previous)

    def test_file_policies_decoding_lists_order_files_unchanged_and_no_fallback(self):
        old_clip = legacy_module("system/utils/data_utils.py")
        old_few = legacy_module("system/fedfews_compat/vendor/data_utils.py")
        old_cwd = Path.cwd()
        args = SimpleNamespace(niid=1, partition="pat", class_per_client=20, dir_alpha=.1)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ), redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "system").mkdir()
            directory = root / "dataset/Cifar100/pat_20/train"
            directory.mkdir(parents=True)
            try:
                os.chdir(root / "system")
                os.environ["PFLLIB_DATA_DIR"] = str(root / "dataset")
                random = np.random.RandomState(52)
                for i, list_input in enumerate([False, True]):
                    x = random.randn(5 + i, 3, 32, 32)[:, :, :, ::-1].copy()
                    y = np.arange(len(x))[::-1] % 4
                    data = dict(x=list(x) if list_input else x, y=y)
                    npz = directory / f"{i}.npz"
                    pkl = directory / f"{i}.pkl"
                    np.savez_compressed(npz, data=data)
                    with pkl.open("wb") as f:
                        pickle.dump(dict(x=data["x"], y=(y + 2) % 4), f)
                    before = {p: sha256(p) for p in [npz, pkl]}
                    state = rng_digest()
                    for reader, legacy, params in [(clip_data, old_clip, ("Cifar100", i, args)),
                                                   (few_data, old_few, ("Cifar100/pat_20", i))]:
                        actual = reader.read_client_data(*params)
                        expected = legacy["read_client_data"](*params)
                        assert_data(self, actual, expected)
                        self.assertEqual(actual[0][0].dtype, torch.float32)
                        self.assertEqual(actual[0][1].dtype, torch.int64)
                    self.assertEqual(select_client_shard(directory, i, formats=NPZ_ONLY), npz)
                    self.assertEqual(select_client_shard(directory, i, formats=PKL_FIRST), pkl)
                    self.assertEqual(before, {p: sha256(p) for p in before})
                    self.assertEqual(state, rng_digest())
                    # Removing npz must NOT make FedClip silently read the pkl.
                    npz.unlink()
                    with self.assertRaises(FileNotFoundError):
                        clip_data.read_client_data("Cifar100", i, args)
                    assert_data(self, few_data.read_client_data("Cifar100/pat_20", i), old_few["read_client_data"]("Cifar100/pat_20", i))
                    # pkl gone too: neither policy searches other directories.
                    pkl.unlink()
                    with self.assertRaises(FileNotFoundError):
                        few_data.read_client_data("Cifar100/pat_20", i)
                    np.savez_compressed(npz, data=data)
                    assert_data(self, few_data.read_client_data("Cifar100/pat_20", i), old_few["read_client_data"]("Cifar100/pat_20", i))
                with self.assertRaises(ValueError):
                    select_client_shard(directory, 0, formats=(".auto",))
            finally:
                os.chdir(old_cwd)

    def test_legacy_text_shakespeare_few_shot_and_partition_mapping(self):
        old = legacy_module("system/utils/data_utils.py")
        text = dict(x=[([1, 2], 2), ([3, 0], 1)], y=[1, 0])
        shakespeare = dict(x=[[1, 3], [2, 4]], y=[3, 4])
        image = dict(x=np.arange(6 * 4).reshape(6, 1, 2, 2), y=[2, 1, 2, 0, 1, 2])
        for dataset, data in [("AGNews", text), ("Shakespeare", shakespeare), ("Cifar100", image)]:
            old["read_data"] = lambda *a, **kw: data
            with patch.object(clip_data, "read_data", return_value=data):
                for train in [True, False]:
                    for shots in [0, 1, 2]:
                        assert_data(self, clip_data.read_client_data(dataset, 0, None, train, shots),
                                    old["read_client_data"](dataset, 0, None, train, shots))
        for partition in ["dir", "pat", "exdir", "custom"]:
            for niid in [0, 1]:
                args = SimpleNamespace(niid=niid, partition=partition, dir_alpha=.3, class_per_client=20)
                self.assertEqual(clip_data.get_dataset_sub_dir(args), old["get_dataset_sub_dir"](args))
        nested = dict(x=[[[1., 2.]], [[3., 4.]]], y=[2, 1])
        assert_data(self, clip_data.process_image(nested), old["process_image"](nested))

    def test_audit_and_training_use_same_selected_file_and_public_decoder(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ):
            root = Path(temp)
            dataset = synthetic_data(root / "data")
            args, _ = parse_args(["--data-root", str(root / "data"), "--dataset", dataset, "--num_clients", "3", "--output-dir", str(root / "out")])
            before = rng_digest()
            audit = audit_data(args)
            for record in audit["files"]:
                train = record["split"] == "train"
                selected = few_data.client_shard_path(dataset, record["client_id"], train)
                self.assertEqual(str(selected.resolve()), record["path"])
                decoded = few_data.read_client_data(dataset, record["client_id"], train)
                self.assertEqual(record["decoded_x_sha256"], tensor_digest(torch.stack([p[0] for p in decoded])))
                self.assertEqual(record["decoded_y_sha256"], tensor_digest(torch.stack([p[1] for p in decoded])))
            self.assertEqual(before, rng_digest())

    def test_loaders_actual_indices_and_rng_match_legacy_in_all_phases_with_injection(self):
        data = [(torch.tensor([float(i)]), torch.tensor(i)) for i in range(13)]
        old_clip, old_clip_ns = legacy_class("system/flcore/clients/clientbase.py", "Client")
        old_few, old_few_ns = legacy_class("system/fedfews_compat/vendor/clientbase.py", "Client")
        for old, namespace, new, module, kind in [(old_clip, old_clip_ns, clip_clients.Client, clip_clients, "clip"),
                                                  (old_few, old_few_ns, few_clients.Client, few_clients, "few")]:
            namespace["read_client_data"] = lambda *a, **kw: data
            for phase, method, override in [("train", "load_train_data", None), ("deployment_selection", "load_train_data", None),
                                             ("test", "load_test_data", None), ("train_metrics", "load_train_data", 7)]:
                traces = []
                initial = rng_state()
                for cls in [old, new]:
                    set_rng(initial)
                    client = cls.__new__(cls)
                    client.id, client.dataset, client.args, client.few_shot = 0, "Synthetic", None, 0
                    client.batch_size, client.test_batch_size, client.use_cache = 4, 7, True
                    client._train_data_cache = client._test_data_cache = data
                    with patch.object(module, "read_client_data", return_value=data):
                        loader = getattr(client, method)(batch_size=override)
                    self.assertEqual(loader.num_workers, 0)
                    self.assertFalse(loader.drop_last)
                    self.assertIsNone(loader.generator)
                    self.assertEqual(rng_digest(), self._digest_of(initial))
                    # Two iterations checks reuse of a loader for sequential models/epochs.
                    traces.append(([[y.tolist() for _, y in loader] for _ in range(2)], rng_digest()))
                self.assertEqual(traces[0], traces[1], (kind, phase))
            # A module-level injected class must still be used by the factory.
            with tempfile.TemporaryDirectory() as temp:
                recorder = Recorder(temp, trace_batches=True)
                recorder.server = SimpleNamespace(local_epochs=1, K=3)
                recorder.client, recorder.stage = 0, "train"
                cls = new.__new__(new)
                cls.id, cls.dataset, cls.args, cls.few_shot, cls.batch_size = 0, "S", None, 0, 4
                cls.use_cache, cls._train_data_cache = True, data
                initial = rng_state()
                with patch.object(module, "read_client_data", return_value=data), patch.object(module, "DataLoader", recorder.loader_class()):
                    actual = [y.tolist() for _, y in cls.load_train_data()]
                actual_rng = rng_digest()
                set_rng(initial)
                expected = [y.tolist() for _, y in DataLoader(data, 4, shuffle=True, drop_last=False)]
                self.assertEqual(actual, expected)
                self.assertEqual(actual_rng, rng_digest())
                import json
                event = json.loads((Path(temp) / "events.jsonl").read_text())
                self.assertEqual(event["payload"]["batches"], actual)
            if kind == "few":
                for cached in [True, False]:
                    cls.use_cache = cached
                    cls.test_batch_size, cls._test_data_cache = 7, data
                    with patch.object(module, "read_client_data", return_value=data) as read:
                        cls.load_train_data()
                        cls.load_test_data()
                    self.assertEqual(read.call_count, 0 if cached else 2)

    def test_external_reference_reader_cnn_loader_remain_independent(self):
        from fedfews_compat.reference import load_reference
        from common import data_io, data_loading
        source = Path(os.environ.get("FEDFEW_SOURCE_ROOT", ROOT.parent / "FedFew_reference"))
        if not (source / "PFLlib/system/main.py").exists():
            self.skipTest("Use verify_fedfews.py with an external --source-root for this integration check")
        backend = load_reference(source)
        self.assertIsNot(backend.Model, FedAvgCNN)
        self.assertTrue(backend.Model.__module__.startswith("fedfew_reference_"))
        self.assertTrue(backend.reader.__module__.startswith("fedfew_reference_"))
        self.assertNotIn("make_data_loader", backend.clientbase_globals)
        self.assertNotIn("read_shard", backend.reader.__globals__)
        self.assertEqual(Path(backend.Model.forward.__code__.co_filename).resolve(),
                         (source / "PFLlib/system/flcore/trainmodel/models.py").resolve())
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ), redirect_stdout(io.StringIO()):
            root = Path(temp)
            dataset = synthetic_data(root / "data")
            os.environ["PFLLIB_DATA_DIR"] = str(root / "data")
            args, _ = parse_args(["--data-root", str(root / "data"), "--dataset", dataset, "--num_clients", "3",
                                  "--batch_size", "4", "--device", "cpu", "--output-dir", str(root / "out")])
            with patch.object(data_io, "read_shard", side_effect=AssertionError("common reader in reference")), \
                    patch.object(data_io, "decode_image_data", side_effect=AssertionError("common decoder in reference")), \
                    patch.object(data_loading, "make_data_loader", side_effect=AssertionError("common loader in reference")):
                decoded = backend.reader(dataset, 0)
                self.assertEqual(len(decoded), 7)
                args.model = backend.Model(3, 100, 1600)
                server = backend.Server(args, 0)
                server.send_models()
                client = server.clients[0]
                client.train()
                client.select_best_model()
                self.assertEqual(client.test_metrics()[1], 3)
                self.assertEqual(client.train_metrics()[1], 7)

    @staticmethod
    def _digest_of(state):
        current = rng_state()
        set_rng(state)
        result = rng_digest()
        set_rng(current)
        return result

    def test_cnn_constructor_rng_schema_forward_legacy_exports_and_checkpoints(self):
        old, _ = legacy_class("system/flcore/trainmodel/models.py", "FedAvgCNN")
        self.assertIs(clip_models.FedAvgCNN, FedAvgCNN)
        self.assertIs(few_models.FedAvgCNN, FedAvgCNN)
        for config, size in [((1, 10, 1024), 28), ((3, 100, 1600), 32)]:
            initial = rng_state()
            expected = old(*config)
            expected_rng = rng_digest()
            set_rng(initial)
            actual = FedAvgCNN(*config)
            self.assertEqual(expected_rng, rng_digest())
            self.assertEqual(list(expected.state_dict()), list(actual.state_dict()))
            for n, p in actual.state_dict().items():
                self.assertEqual(p.shape, expected.state_dict()[n].shape)
                self.assertTrue(torch.equal(p, expected.state_dict()[n]))
            x = torch.linspace(-1, 1, 2 * config[0] * size * size).reshape(2, config[0], size, size)
            self.assertTrue(torch.equal(expected(x), actual(x)))
        with tempfile.TemporaryDirectory() as temp:
            # Serialize an actual pre-refactor class at its ORIGINAL pickle path.
            old.__module__ = "flcore.trainmodel.models"
            with patch.object(clip_models, "FedAvgCNN", old):
                old_model = old(3, 100, 1600)
                torch.save(old_model, Path(temp) / "legacy.pt")
            restored = torch.load(Path(temp) / "legacy.pt", weights_only=False)
            self.assertIsInstance(restored, FedAvgCNN)
            self.assertTrue(torch.equal(old_model(x), restored(x)))
            torch.save(actual, Path(temp) / "common.pt")
            self.assertIsInstance(torch.load(Path(temp) / "common.pt", weights_only=False), FedAvgCNN)
        code = "import sys;sys.path.insert(0,'system');from fedfews_compat.vendor.models import FedAvgCNN;assert 'flcore.trainmodel.models' not in sys.modules;assert 'torchvision' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_pure_metrics_legacy_summary_units_subsets_invalid_inputs_and_no_rng(self):
        old = ast.parse(legacy_code("system/fedfews_compat/observe.py"))
        function = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "summary")
        namespace = dict(np=np)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "legacy_summary", "exec"), namespace)
        before = rng_digest()
        self.assertEqual(sample_weighted_accuracy([1, 9, 2], [2, 10, 8], client_ids=[4, 0, 7]), 12 / 20)
        self.assertEqual(sample_weighted_accuracy([1, 9, 2], [2, 10, 8], client_ids=[4, 0, 7], subset_ids=[7, 4]), 3 / 10)
        for length in [1, 3, 12, 25]:
            points = [dict(loop_round=i * 10, completed_train_count=i * 10 + 1, test_acc_percent=100 * ((i % 7) / 7)) for i in range(length)]
            expected, actual = namespace["summary"](points), summary(points)
            self.assertEqual(expected, {k: actual[k] for k in expected})
            self.assertEqual(actual["last10_evaluation_count"], min(length, 10))
        for correct, samples in [([], []), ([0], [0]), ([1], [float("nan")]), ([float("inf")], [2]), ([3], [2])]:
            with self.assertRaises(ValueError):
                sample_weighted_accuracy(correct, samples)
        for subset in [[], [4, 4], [99]]:
            with self.assertRaises(ValueError):
                sample_weighted_accuracy([1], [2], client_ids=[4], subset_ids=subset)
        with self.assertRaises(ValueError):
            summarize_evaluations([], value_key="acc", unit="fraction", evaluation_object="post_local")
        for value in [float("nan"), float("inf"), -1., 101.]:
            with self.assertRaises(ValueError):
                summary([dict(loop_round=0, completed_train_count=1, test_acc_percent=value)])
        self.assertEqual(before, rng_digest())


if __name__ == "__main__":
    unittest.main()
