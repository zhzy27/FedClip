"""Warmup schedule, bounded matrix evidence, and atomic fixed-group rendezvous."""
import json
import os
from pathlib import Path
import time

import numpy as np

from common.experiment import jsonable, rng_state, set_rng


def phase_for_loop(args, loop):
    return 'warmup' if loop < getattr(args, 'multi_warmup_rounds', 0) else 'grouped'


def grouping_protocol(args):
    warmup = getattr(args, 'multi_warmup_rounds', 0)
    basis = getattr(args, 'multi_group_basis', 'single_delta')
    requested_window = getattr(args, 'multi_history_window', None)
    window = (warmup if requested_window is None else requested_window) if basis == 'delta_history' else None
    formation = max(1, warmup)
    loops = list(range(warmup-window, warmup)) if basis == 'delta_history' else [formation-1]
    return dict(basis=basis, warmup_rounds=warmup, history_window=window,
        evidence_loop_rounds=loops, evidence_completed_train_counts=[r+1 for r in loops],
        formation_loop_round=formation-1, formation_completed_train_count=formation,
        grouped_start_loop_round=warmup)


def compatibility_default(args):
    return (getattr(args, 'multi_warmup_rounds', 0) == 0 and
            getattr(args, 'multi_group_basis', 'single_delta') == 'single_delta')


class GroupEvidence:
    """Retain only required N x N matrices/column matrices and fingerprints."""
    def __init__(self, args):
        self.protocol = grouping_protocol(args)
        self.records = []

    def collect(self, loop, servers, clients):
        if loop not in self.protocol['evidence_loop_rounds']:
            return
        if any(record['loop_round'] == loop for record in self.records):
            raise RuntimeError('Duplicate grouping evidence: no repeat local training or collection')
        from utils.multi_group_clustering import update_similarity, digest_json
        basis = 'full_weights' if self.protocol['basis'] == 'full_weights' else 'single_delta'
        matrix, diagnostics = update_similarity(servers, clients, basis=basis)
        self.records.append(dict(loop_round=loop, completed_train_count=loop+1, matrix=matrix.tolist(),
            vector_kind=basis, vector_sha256=diagnostics.get('upload_full_weights_sha256', diagnostics.get('first_upload_delta_sha256')),
            diagnostics=diagnostics, matrix_sha256=digest_json(matrix)))

    def finalize(self):
        from utils.multi_group_clustering import digest_json
        if [r['loop_round'] for r in self.records] != self.protocol['evidence_loop_rounds']:
            raise RuntimeError('Missing/out-of-order grouping evidence')
        # Preserve original single-delta arithmetic exactly (no extra averaging).
        matrix = (np.asarray(self.records[0]['matrix'], dtype=np.float64) if len(self.records) == 1 else
                  np.mean(np.asarray([r['matrix'] for r in self.records], dtype=np.float64), axis=0))
        evidence = dict(protocol=self.protocol, rounds=self.records, final_similarity_sha256=digest_json(matrix))
        evidence['evidence_sha256'] = digest_json(evidence)
        return matrix, evidence


def read_group_file(path, *, wait_seconds=0., poll_seconds=1., validate=None):
    """Only missing files are waited on. Invalid complete files fail immediately."""
    path = Path(path)
    state, started = rng_state(), time.monotonic()
    try:
        if wait_seconds:
            print(f'[GroupWait] path={path}; timeout={wait_seconds}s; no additional training', flush=True)
        next_notice = started
        while True:
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                elapsed = time.monotonic()-started
                if not wait_seconds:
                    raise FileNotFoundError(f'Grouping file does not exist: {path}; waiting is disabled')
                if elapsed >= wait_seconds:
                    raise TimeoutError(f'Grouping file timeout after {elapsed:.3f}s: {path}; completed local training was not repeated')
                if time.monotonic() >= next_notice:
                    print(f'[GroupWait] still waiting {elapsed:.1f}s/{wait_seconds}s: {path}', flush=True)
                    next_notice = time.monotonic()+15.
                time.sleep(min(poll_seconds, wait_seconds-elapsed, 15.))
                continue
            try:
                payload = json.loads(raw.decode('utf-8'))
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f'Grouping file is invalid JSON, not a published complete artifact: {path}') from error
            if validate:
                validate(payload)
            import hashlib
            return payload, hashlib.sha256(raw).hexdigest(), time.monotonic()-started
    finally:
        set_rng(state)


def atomic_publish_group(path, payload):
    """Publish an fsynced complete JSON atomically, without overwriting old runs.

    Temp and destination share a directory/filesystem. Atomic link creation
    provides both no-overwrite and no partially-visible JSON on POSIX/NTFS.
    """
    path = Path(path)
    state = rng_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp.{os.getpid()}.{time.monotonic_ns()}')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(jsonable(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f'Grouping publish path already exists; refusing to overwrite: {path}') from error
        print(f'[GroupPublish] complete fixed grouping published atomically: {path}', flush=True)
    finally:
        if temporary.exists():
            temporary.unlink()
        set_rng(state)
