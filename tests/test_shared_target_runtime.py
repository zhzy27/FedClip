"""Real file readers and heterogeneous low-rank FedTargetProj vs 189bacd."""
import ast
from contextlib import redirect_stdout
import copy
import io
import os
from pathlib import Path
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
from flcore.clients import clientbase as new_clientbase
from flcore.clients.clientbase import load_item
from flcore.servers.serverTargetProj import FedTargetProj
from fedfews_compat.observe import rng_digest
from fedfews_compat.metadata import tensor_digest
from test_shared_primitives import legacy_code, legacy_module


def legacy_module_with_bindings(path, bindings):
    tree = ast.parse(legacy_code(path))
    # Resolve only the specified legacy class imports to the pre-refactor bases.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            node.names = [item for item in node.names if (item.asname or item.name) not in bindings]
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.names]
    namespace = dict(__file__=str(ROOT / path), __name__="legacy_189_runtime", **bindings)
    exec(compile(tree, "189bacd:" + path, "exec"), namespace)
    return namespace


def model_digest(model):
    return {name: tensor_digest(p) for name, p in model.state_dict().items()}


class SharedTargetRuntimeTests(unittest.TestCase):
    def test_real_files_batches_uploads_aggregation_rng_equal_for_avg_and_projection(self):
        torch.set_num_threads(2)
        # Existing protected evaluation lazily queries CUDA RNG even for a CPU
        # model. Initialize that state before BOTH endpoints, so the first run
        # is not compared pre-initialization with an already-initialized second.
        if torch.cuda.is_available():
            torch.cuda.get_rng_state_all()
        old_reader = legacy_module("system/utils/data_utils.py")["read_client_data"]
        old_clientbase = legacy_module_with_bindings("system/flcore/clients/clientbase.py", dict(read_client_data=old_reader))
        old_client = legacy_module_with_bindings("system/flcore/clients/clientTargetProj.py", dict(Client=old_clientbase["Client"]))
        old_serverbase = legacy_module_with_bindings("system/flcore/servers/serverbase.py", dict(read_client_data=old_reader))
        old_server = legacy_module_with_bindings("system/flcore/servers/serverTargetProj.py", dict(Server=old_serverbase["Server"], clientTargetProj=old_client["clientTargetProj"]))
        initial_cwd = Path.cwd()
        histories = {}
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            (root / "system").mkdir()
            fixture_rng = np.random.RandomState(37)
            for split in ["train", "test"]:
                directory = root / "dataset/Cifar100/pat_20" / split
                directory.mkdir(parents=True)
                for cid in range(20):
                    count = (7 + cid % 3) if split == "train" else (3 + cid % 2)
                    data = dict(x=fixture_rng.randn(count, 3, 32, 32).astype(np.float32), y=(np.arange(count) + cid) % 100)
                    np.savez_compressed(directory / f"{cid}.npz", data=data)
                    # Ensure FedFew pkl priority cannot leak into the old route.
                    import pickle
                    with (directory / f"{cid}.pkl").open("wb") as f:
                        pickle.dump(dict(x=np.zeros_like(data["x"]), y=np.full(count, 99)), f)
            try:
                os.chdir(root / "system")
                for mode in ["avg", "projection_softmax"]:
                    for label, cls, base_ns in [("legacy", old_server["FedTargetProj"], old_clientbase),
                                                 ("shared", FedTargetProj, vars(new_clientbase))]:
                        directory = root / (mode[:1] + label[:1])
                        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
                        args = SimpleNamespace(device="cpu", dataset="Cifar100", num_classes=100, global_rounds=1,
                            local_epochs=5, batch_size=4, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                            random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                            goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                            client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                            model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat", class_per_client=20, dir_alpha=.1,
                            exp_name=mode, save_file_paths=[], models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                            global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                            is_regular=1, regular_lamda=.001, target_proj_mode=mode, target_prox_mu=0.,
                            projection_self_weight=.125 if mode == "projection_softmax" else None,
                            target_head_mode="shared", target_local_schedule="joint",
                            save_folder_name=str(directory / "c"), final_model_root=str(directory / "f"),
                            h5_result_root=str(directory / "h"), training_log_root=str(directory / "l"), training_log_dir=str(directory / "l"))
                        trace, current, rounds = [], {}, []
                        class TraceLoader(DataLoader):
                            def __init__(self, *a, **kw):
                                super().__init__(*a, **kw)
                                origin, context = self.batch_sampler, dict(current)
                                class Sampler:
                                    def __len__(self): return len(origin)
                                    def __iter__(self):
                                        indices = []
                                        for batch in origin:
                                            indices.append(list(map(int, batch)))
                                            yield batch
                                        trace.append((context, indices, rng_digest()))
                                object.__setattr__(self, "batch_sampler", Sampler())
                        original_loader = base_ns["DataLoader"]
                        base_ns["DataLoader"] = TraceLoader
                        try:
                            server = cls(args, 0)
                            for client in server.clients:
                                for method in ["load_train_data", "load_test_data"]:
                                    native = getattr(client, method)
                                    def wrapped(*a, native=native, client=client, method=method, **kw):
                                        previous = dict(current)
                                        current.update(client_id=client.id, phase=method)
                                        try: return native(*a, **kw)
                                        finally:
                                            current.clear()
                                            current.update(previous)
                                    setattr(client, method, wrapped)
                            native_aggregate = server.aggregate_parameters_avg
                            def aggregate():
                                uploads = [model_digest(load_item(c.role, "model", c.save_folder_name)) for c in server.clients]
                                native_aggregate()
                                rounds.append(dict(uploads=uploads, uploaded_ids=list(server.uploaded_ids),
                                    sample_weights=list(server.uploaded_weights), server=model_digest(load_item("Server", "model", server.save_folder_name)), rng=rng_digest()))
                            server.aggregate_parameters_avg = aggregate
                            server.train()
                            self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                            histories[(mode, label)] = dict(rounds=rounds, batches=trace, rng=rng_digest(),
                                                          summary=server._local_accuracy_summary())
                        finally:
                            base_ns["DataLoader"] = original_loader
                    old, new = histories[(mode, "legacy")], histories[(mode, "shared")]
                    self.assertEqual(old["rounds"], new["rounds"], mode + " uploads/servers/RNG")
                    self.assertEqual(len(old["batches"]), len(new["batches"]))
                    for index, (a, b) in enumerate(zip(old["batches"], new["batches"])):
                        self.assertEqual(a, b, f"{mode} batch trace {index}")
                    self.assertEqual(old["rng"], new["rng"], mode + " final RNG")
                    self.assertEqual(old["summary"], new["summary"], mode + " summary")
            finally:
                os.chdir(initial_cwd)


if __name__ == "__main__":
    unittest.main()
