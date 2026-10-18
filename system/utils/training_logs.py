"""Downloadable log-only directories, console capture and per-process ZIP bundles."""

from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import traceback
import zipfile


def safe_name(value):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "run"


def initialize_log_folder(args):
    explicit = getattr(args, "run_log_dir", "") or os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
    if explicit:
        folder = Path(explicit).expanduser().resolve()
    else:
        mode = getattr(args, "target_proj_mode", "") if args.algorithm == "FedTargetProj" else args.algorithm
        if args.algorithm == "FedTargetProj":
            self_weight = getattr(args, "projection_self_weight", None)
            group_ids = getattr(args, "target_group_ids", None)
            if self_weight is not None:
                mode += "_self" + str(self_weight).replace(".", "p")
            if group_ids:
                mode += "_group" + "_".join(map(str, group_ids))
                beta = getattr(args, "target_group_weight", None)
                if beta is not None:
                    mode += "_beta" + str(beta).replace(".", "p")
        name = f"{datetime.now():%Y%m%d_%H%M%S}_{safe_name(mode)}_seed{getattr(args, 'seed', 0)}_{time.time_ns()}_{os.getpid()}"
        folder = Path(getattr(args, "training_log_root", "training_logs")).expanduser().resolve() / name
    if hasattr(args, "save_folder_name"):
        checkpoint_root = Path(args.save_folder_name).expanduser().resolve()
        if folder == checkpoint_root or folder in checkpoint_root.parents or checkpoint_root in folder.parents:
            raise ValueError("The log folder and model root must be separate trees.")
    folder.mkdir(parents=True, exist_ok=True)
    args.training_log_dir = str(folder)
    return folder


def server_log_folder(args, run_id, checkpoint_folder):
    """CLI runs separate logs; legacy programmatic fixtures retain their path."""
    process_folder = getattr(args, "training_log_dir", "")
    if process_folder:
        process_root = Path(process_folder).resolve()
        checkpoint = Path(checkpoint_folder).resolve()
        if process_root == checkpoint or process_root in checkpoint.parents or checkpoint in process_root.parents:
            raise ValueError("The process log folder and checkpoint folder must be separate trees.")
        folder = Path(process_folder) / f"run_{run_id}"
    elif hasattr(args, "training_log_root"):
        folder = Path(args.training_log_root) / safe_name(args.dataset) / safe_name(args.algorithm)
        mode = getattr(args, "target_proj_mode", "default")
        folder /= f"{safe_name(mode)}_seed{getattr(args, 'seed', 0)}"
        folder /= run_id
    else:
        return checkpoint_folder  # Backwards-compatible non-CLI API.
    folder = folder.expanduser().resolve()
    checkpoint = Path(checkpoint_folder).resolve()
    if folder == checkpoint or checkpoint in folder.parents:
        raise ValueError("Training logs must be outside the checkpoint directory.")
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder)


def mirror_result(path, log_dir):
    target = Path(log_dir) / Path(path).name
    if Path(path).resolve() != target.resolve():
        shutil.copy2(path, target)


def archive_logs(folder):
    """Archive only the logs tree, never tensor/model payloads."""
    folder = Path(folder)
    target = folder.with_name(folder.name + ".zip")
    temporary = target.with_name(target.name + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".log", ".csv", ".json", ".jsonl", ".h5", ".hdf5", ".txt", ".png", ".svg", ".pdf"):
                archive.write(path, arcname=str(Path(folder.name) / path.relative_to(folder)))
    temporary.replace(target)
    return target


def _same_output(stream, path):
    try:
        return os.path.samestat(os.fstat(stream.fileno()), os.stat(path))
    except (AttributeError, OSError, ValueError):
        return False


class _Tee:
    def __init__(self, console, file, lock):
        self.console, self.file, self.lock = console, file, lock

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.file.write(text)
        return len(text)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.file.flush()

    def __getattr__(self, key):
        return getattr(self.console, key)


@contextmanager
def training_log_session(args):
    folder = initialize_log_folder(args)
    log_path = folder / "train.log"
    previous_env = os.environ.get("FEDCLIP_TRAIN_LOG_DIR")
    os.environ["FEDCLIP_TRAIN_LOG_DIR"] = str(folder)
    stdout, stderr = sys.stdout, sys.stderr
    capture = None
    status = dict(status="running", log_dir=str(folder))
    try:
        # Launchers can already redirect stdout/stderr to this file. Avoid a
        # second writer or duplicate console lines when that is the case.
        stdout_redirected = _same_output(stdout, log_path)
        stderr_redirected = _same_output(stderr, log_path)
        if not (stdout_redirected and stderr_redirected):
            capture = log_path.open("a", encoding="utf-8", buffering=1)
            lock = threading.RLock()
            if not stdout_redirected:
                sys.stdout = _Tee(stdout, capture, lock)
            if not stderr_redirected:
                sys.stderr = _Tee(stderr, capture, lock)
        print(f"[TrainingLogs] folder={folder}", flush=True)
        yield folder
        status["status"] = "complete"
    except BaseException:
        status.update(status="failed", traceback=traceback.format_exc())
        if capture:
            capture.write(status["traceback"])
        raise
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = stdout, stderr
        if capture:
            capture.close()
        if previous_env is None:
            os.environ.pop("FEDCLIP_TRAIN_LOG_DIR", None)
        else:
            os.environ["FEDCLIP_TRAIN_LOG_DIR"] = previous_env
        (folder / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        try:
            archive = archive_logs(folder)
            print(f"[TrainingLogs] Download bundle: {archive}", flush=True)
        except OSError as error:
            print(f"[TrainingLogs] ZIP unavailable; logs remain in {folder}: {error}", file=stderr)
