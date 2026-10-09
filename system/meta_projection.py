"""C0 full-unroll diagnostic: split, offline optimization, and measured CNN benchmark."""

import argparse
import json
from pathlib import Path
import tempfile

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    split = sub.add_parser("split")
    split.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    split.add_argument("--dataset", default="Cifar100")
    split.add_argument("--subdir", default="pat_20")
    split.add_argument("--fraction", type=float, default=.2)
    split.add_argument("--split-seed", type=int, default=1729)
    split.add_argument("--output", type=Path, required=True)
    opt = sub.add_parser("optimize")
    opt.add_argument("--snapshots", type=Path, required=True)
    opt.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    opt.add_argument("--output", type=Path, required=True)
    opt.add_argument("--strategy", choices=["fixed", "per_snapshot"], default="fixed")
    opt.add_argument("--updates", type=int, default=50)
    opt.add_argument("--outer-lr", type=float, default=.05)
    opt.add_argument("--initializations", nargs="+", choices=["uniform", "biased"], default=["uniform", "biased"])
    opt.add_argument("--bias-client", type=int, default=0)
    opt.add_argument("--bias-logit", type=float, default=.25)
    opt.add_argument("--virtual-seed", type=int, default=777)
    opt.add_argument("--checkpoint-steps", type=int, default=0)
    opt.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    opt.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    opt.add_argument("--cpu-threads", type=int, default=2)
    opt.add_argument("--backend-policy", choices=["snapshot", "no_tf32", "deterministic_only", "deterministic_no_tf32"], default="snapshot")
    opt.add_argument("--acceptance-report", type=Path)
    opt.add_argument("--diagnostic-only", action="store_true",
                     help="Unaccepted experiments only; cannot export usable fixed weights.")
    diag = sub.add_parser("diagnose", help="Staged SGD, exact recomputation and multi-step FD acceptance")
    diag.add_argument("--snapshots", type=Path, required=True)
    diag.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    diag.add_argument("--output", type=Path, required=True)
    diag.add_argument("--rounds", type=int, nargs="+", default=[20])
    diag.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    diag.add_argument("--checkpoint-steps", type=int, default=8)
    diag.add_argument("--virtual-seed", type=int, default=777)
    diag.add_argument("--backend-policy", choices=["snapshot", "no_tf32", "deterministic_only", "deterministic_no_tf32"], default="snapshot")
    diag.add_argument("--initializations", nargs="+", choices=["uniform", "biased"], default=["uniform", "biased"])
    diag.add_argument("--bias-client", type=int, default=0)
    diag.add_argument("--bias-logit", type=float, default=.25)
    diag.add_argument("--fd-epochs", type=int, default=1)
    diag.add_argument("--fd-directions", type=int, default=3)
    diag.add_argument("--fd-repeats", type=int, default=3)
    diag.add_argument("--fd-steps", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    diag.add_argument("--fd-stage-steps", type=int, nargs="*", default=[0, 1, 8],
                      help="Additional planned-dtype derivative probes before the epoch probe.")
    diag.add_argument("--forward-only", action="store_true",
                      help="Trajectory and full hypergradient only; always incomplete acceptance.")
    diag.add_argument("--trace-branches", action="store_true",
                      help="Extra ReLU/max-pool probes; record separately from uninstrumented acceptance.")
    diag.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    diag.add_argument("--cpu-threads", type=int, default=2)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--snapshots", type=Path)
    bench.add_argument("--dataset-root", type=Path, default=Path(__file__).resolve().parents[1] / "dataset")
    bench.add_argument("--synthetic", action="store_true")
    bench.add_argument("--synthetic-samples", type=int, default=40)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--checkpoint-steps", type=int, default=0)
    bench.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    bench.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.action == "split":
        from utils.meta_data import create_split, read_c0_training, save_split
        data = read_c0_training(args.dataset_root, args.dataset, args.subdir)
        record = create_split(data, args.fraction, args.split_seed, args.dataset, args.subdir)
        save_split(args.output, record)
        print(json.dumps(dict(split_id=record["split_id"], train=len(record["train_indices"]), validation=len(record["validation_indices"]), output=str(args.output.resolve())), indent=2))
    elif args.action in ("optimize", "diagnose"):
        from utils.meta_learning import MetaProblem, optimize
        torch.set_num_threads(args.cpu_threads)
        problem = MetaProblem(args.snapshots, args.dataset_root, args.device, getattr(torch, args.dtype),
                              args.virtual_seed, args.checkpoint_steps, args.backend_policy)
        if args.action == "diagnose":
            from utils.meta_diagnostics import diagnose
            report = diagnose(problem, args.output, args.rounds, args.initializations,
                args.bias_client, args.bias_logit, args.fd_epochs, args.fd_directions,
                args.fd_repeats, args.fd_steps, fd_stage_steps=args.fd_stage_steps,
                forward_only=args.forward_only, trace_branches=args.trace_branches)
            print(json.dumps(dict(status=report["status"], report=str(args.output)), indent=2))
        else:
            optimize(problem, args.output, args.strategy, args.updates, args.outer_lr,
                     args.initializations, args.bias_client, args.bias_logit,
                     args.acceptance_report, args.diagnostic_only)
    else:
        from utils.meta_benchmark import benchmark, synthetic_collection
        torch.set_num_threads(args.cpu_threads)
        if args.synthetic:
            with tempfile.TemporaryDirectory() as directory:
                root, data_root = synthetic_collection(Path(directory) / "snapshots", args.synthetic_samples)
                result = benchmark(root, data_root, args.output, args.device, args.checkpoint_steps)
        else:
            if args.snapshots is None:
                parser.error("Real-data benchmark requires --snapshots.")
            result = benchmark(args.snapshots, args.dataset_root, args.output, args.device, args.checkpoint_steps)
        print(json.dumps({k: v for k, v in result.items() if k not in ("details", "hypergradient")}, indent=2))


if __name__ == "__main__":
    main()
