import copy
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .data import audit_data
from .metadata import environment, migration_info, sha256, source_info, write_json
from .observe import Recorder, rng_state, summary
from .vendor import clientbase, data_utils
from .vendor.clientfedfews import clientFedFewS
from .vendor.models import FedAvgCNN
from .vendor.serverfedfews import FedFewS


def migrated_backend():
    return SimpleNamespace(Server=FedFewS, Client=clientFedFewS, Model=FedAvgCNN,
                           reader=data_utils.read_client_data, clientbase_globals=vars(clientbase))


def prepare_random(args):
    # Upstream main.py calls torch.manual_seed(0) once, before constructing models.
    # It does NOT fix Python/NumPy. Controlled mode deliberately fixes those too.
    if args.rng_mode == "controlled":
        random.seed(args.control_seed)
        np.random.seed(args.control_seed)
        torch.manual_seed(args.control_seed)
    torch.manual_seed(0)
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def build_server(args, output, backend=None, event_sink=None, run_index=0):
    backend = backend or migrated_backend()
    options = copy.deepcopy(args)
    options.model_dir = str(Path(output) / "models")
    options.result_dir = str(Path(output) / "metrics")
    options.save_folder_name = str(Path(output) / "items")
    options.model = backend.Model(in_features=3, num_classes=options.num_classes, dim=1600).to(options.device)
    recorder = Recorder(Path(output) / "metrics", args.trace_batches, event_sink)
    # Patch this isolated module namespace only, not torch or FedClip Client/Server.
    old_loader = backend.clientbase_globals["DataLoader"]
    backend.clientbase_globals["DataLoader"] = recorder.loader_class()
    try:
        server = backend.Server(options, run_index)
    except BaseException:
        backend.clientbase_globals["DataLoader"] = old_loader
        raise
    recorder.attach(server)
    original_save_results = server.save_results
    def save_results():
        original_save_results()
        recorder.export()
        h5path = Path(options.result_dir) / options.dataset / f"{options.algorithm}_{options.goal}_{run_index}.h5"
        with h5py.File(h5path, "a") as hf:
            hf.create_dataset("evaluation_loop_round", data=[r["loop_round"] for r in recorder.evaluations])
            hf.create_dataset("completed_train_count", data=[r["completed_train_count"] for r in recorder.evaluations])
            hf.create_dataset("server_received_mass", data=[r["server_received_mass"] for r in recorder.rows])
            hf.create_dataset("uploaded_ids", data=[r["uploaded_ids"] for r in recorder.rows])
            hf.create_dataset("deployment_ce", data=[[c["deployment_ce"] for c in r["clients"]] for r in recorder.evaluations])
            hf.create_dataset("deployed_model_idx", data=[[c["selected_model"] for c in r["clients"]] for r in recorder.evaluations])
            hf.create_dataset("test_correct", data=[[c["correct"] for c in r["clients"]] for r in recorder.evaluations])
            hf.create_dataset("test_samples", data=[[c["test_samples"] for c in r["clients"]] for r in recorder.evaluations])
            hf.create_dataset("selected_model_counts", data=[r["selected_model_counts"] for r in recorder.evaluations])
            hf.attrs["accuracy_unit_rs_test_acc"] = "fraction; CSV/JSON use percent"
            hf.attrs["tail_average_unit"] = "evaluation points"
    server.save_results = save_results
    def save_models():
        directory = Path(options.model_dir)
        directory.mkdir(parents=True, exist_ok=True)
        # Upstream inherited save_global_model saves the unused single template.
        # Save the ACTUAL K servers plus K local post-local models instead.
        torch.save(dict(server_models=[m.state_dict() for m in server.global_model_set],
                        clients=[dict(client_id=c.id, best_model_idx=c.best_model_idx,
                                      deployed_model_is_post_local=True,
                                      models=[m.state_dict() for m in c.model_set],
                                      optimizers=[o.state_dict() for o in c.optimizer_set],
                                      schedulers=[o.state_dict() for o in c.learning_rate_scheduler_set]) for c in server.clients],
                        schema={n: list(p.shape) for n, p in server.global_model.state_dict().items()},
                        model="upstream FedAvgCNN", completed_train_count=recorder.round + 1,
                        rng=rng_state()), directory / "FedFewS_final.pt")
    server.save_global_model = save_models
    def restore_loader():
        backend.clientbase_globals["DataLoader"] = old_loader
    return server, recorder, restore_loader


def run(args, config):
    import contextlib
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose an independent run directory: {output}")
    os.environ["PFLLIB_DATA_DIR"] = args.data_root
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU, as upstream main.py does")
        args.device = "cpu"
    prepare_random(args)
    data = audit_data(args)
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir()
    write_json(output / "data_manifest.json", data)
    # Preserve exact config bytes as well as decoded effective values.
    (output / "data_config.json").write_bytes(Path(data["config_path"]).read_bytes())
    write_json(output / "manifest.json", dict(effective_args=vars(args), merged_yaml=config,
               config_sha256=sha256(args.config), command=[sys.executable] + sys.argv,
               source=source_info(args.source_root), migration=migration_info(), environment=environment(),
               protocol=dict(optimizer="SGD lr=local_learning_rate momentum=0 weight_decay=0 (source ignores args.momentum)",
                             epochs_per_model=args.local_epochs, data_epochs_per_client=args.local_epochs * args.num_server_models,
                             dataloader="train AND test shuffle=True, drop_last=False, workers=0, global Torch RNG",
                             buffers="source parameter-only download and aggregation; original CNN has no buffers",
                             evaluation="post-local selection -> ALL tests -> ALL train_metrics -> receive -> aggregate",
                             rng="source evaluation RNG consumption is preserved; only diagnostics avoid new draws")))
    torch.save(rng_state(), output / "rng_before_initialization.pt")
    with (output / "logs/train.log").open("w", encoding="utf-8", buffering=1) as logfile:
        with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
            for index in range(args.prev, args.times):
                run_output = output / f"run_{index}"
                server, recorder, cleanup = build_server(args, run_output, run_index=index)
                try:
                    torch.save(rng_state(), run_output / "rng_after_initialization.pt")
                    # Execute the original inclusive loop, selection/eval/receive order.
                    server.train()
                    print("Migration summary (percent; tail means evaluation points):", summary(recorder.evaluations))
                finally:
                    cleanup()
                effective = dict(vars(server.args))
                effective["model"] = "CNN (upstream FedAvgCNN)"
                write_json(run_output / "manifest.json", dict(effective_args=effective, run_index=index,
                           environment=environment(),
                           summary=summary(recorder.evaluations), final_model="models/FedFewS_final.pt",
                           final_model_semantics="K aggregated servers; K post-local models/client; selected deployment index"))
    return output
