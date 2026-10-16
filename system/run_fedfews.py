"""Independent FedFewS entry; never imports the FedTargetProj runtime."""
import json
import os
import sys


def main():
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    import importlib.util
    required = ["torch", "numpy", "scipy", "yaml", "sklearn", "h5py"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print(json.dumps(dict(python=sys.version, executable=sys.executable, missing_dependencies=missing,
                              environment_modified=False), indent=2))
        raise SystemExit(2)
    # Set visibility BEFORE importing Torch (upstream sets it after import).
    # Device identity does not change any arithmetic or random draws.
    for flag in ["-did", "--device_id"]:
        if flag in sys.argv:
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index(flag) + 1]
    from fedfews_compat.config import parse_args
    from fedfews_compat.metadata import environment, source_info, shared_contract
    args, config = parse_args()
    if args.preview:
        from pathlib import Path
        from fedfews_compat.data import audit_data
        from fedfews_compat.metadata import jsonable
        preview = dict(effective_args=vars(args), merged_yaml=config, environment=environment(),
                       source=source_info(args.source_root), shared_modules=shared_contract(args), training_started=False,
                       optimizer_effective=dict(momentum=0, weight_decay=0),
                       note="num_threads=1 is a source option, NOT torch.set_num_threads; record actual intra-op count",
                       inclusive_train_count=args.global_rounds + 1, data_epochs_per_client=args.local_epochs * args.num_server_models)
        if (Path(args.data_root) / args.dataset / "config.json").exists():
            preview["data"] = audit_data(args)
        else:
            preview["data"] = "UNAVAILABLE: cannot verify actual alpha/shards on this machine"
        print(json.dumps(jsonable(preview), indent=2, ensure_ascii=False))
        return
    from fedfews_compat.runtime import run
    run(args, config)


if __name__ == "__main__":
    main()
