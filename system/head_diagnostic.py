"""Run independent native/fresh-fc3 diagnostics after snapshot collection finishes."""

import argparse
import json
from pathlib import Path

import torch

from utils.head_diagnostic import diagnose_snapshot, write_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", nargs="+", required=True, help="R*.pt files or snapshot folders (recursive).")
    parser.add_argument("--dataset-root", default="../dataset")
    parser.add_argument("--output-dir", required=True, help="New independent diagnostic directory, never training metrics.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=1729)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    paths = set()
    for value in args.snapshots:
        path = Path(value).resolve()
        paths.update(path.rglob("R*.pt") if path.is_dir() else [path])
    paths = sorted(paths)
    if not paths or any(not p.is_file() for p in paths):
        parser.error("No complete head snapshots found.")
    output, data_root = Path(args.output_dir).resolve(), Path(args.dataset_root).resolve()
    for source in [data_root, *(p.parent for p in paths)]:
        if output == source or output in source.parents or source in output.parents:
            parser.error("Diagnostic outputs must be separate from input snapshots and datasets.")
    if args.shuffle_seed < 0 or args.device_id < 0 or args.threads < 1:
        parser.error("Use nonnegative seed/device ID and at least one CPU thread.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for local verification.")
    torch.set_num_threads(args.threads)
    device = f"cuda:{args.device_id}" if args.device == "cuda" else "cpu"
    if args.device == "cuda":
        torch.cuda.set_device(args.device_id)
    output.mkdir(parents=True, exist_ok=False)
    try:
        results = []
        for path in paths:
            result = diagnose_snapshot(path, data_root, device, args.shuffle_seed)
            results.append(result)
            print(f"[HeadDiagnostic] {path}: native reproduced; frozen backbone unchanged; classifier_updated={result['classifier_updated']}")
        comparisons = write_outputs(output, results)
        print(f"[HeadDiagnostic] {len(results)} snapshots, {len(comparisons)} matched comparisons: {output}")
        print((output / "comparison.md").read_text(encoding="utf-8"))
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(dict(status="failed", reason=str(error)), indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
