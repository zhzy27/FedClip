"""Load real upstream files, not the migrated classes, for parity acceptance.

Only package imports are resolved to isolated namespaces. DLG is a rejecting
stub (disabled protocol); the original Server/Client and FedFewS bodies execute.
Unrelated main.py model/algorithm imports are not needed for the CNN protocol.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

from .unsupported import DLG


def load_reference(root):
    system = Path(root).resolve() / "PFLlib/system"
    def load(rel, supplied=None, only=None):
        path = system / rel
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if only:
            tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) or isinstance(n, ast.ClassDef) and n.name == only]
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and
                     (n.module or "").startswith(("flcore.", "utils.")))]
        ns = dict(__name__="fedfew_reference_" + path.stem, __file__=str(path), DLG=DLG)
        ns.update(supplied or {})
        exec(compile(tree, str(path), "exec"), ns)
        return ns
    reader = load("utils/data_utils.py")
    clientbase = load("flcore/clients/clientbase.py", {"read_client_data": reader["read_client_data"]})
    client = load("flcore/clients/clientfedfews.py", {"Client": clientbase["Client"]})
    serverbase = load("flcore/servers/serverbase.py", {"read_client_data": reader["read_client_data"]})
    server = load("flcore/servers/serverfedfews.py", {"Server": serverbase["Server"], "clientFedFewS": client["clientFedFewS"]})
    model = load("flcore/trainmodel/models.py", only="FedAvgCNN")
    return SimpleNamespace(Server=server["FedFewS"], Client=client["clientFedFewS"], Model=model["FedAvgCNN"],
                           reader=reader["read_client_data"], clientbase_globals=clientbase)


def reference_parser(root):
    path = Path(root) / "PFLlib/system/main.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    boolfn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "str2bool")
    main = next(n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and ast.unparse(n.test).startswith("__name__"))
    calls = [n for n in main.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "parser.add_argument"]
    ns = dict(argparse=argparse, parser=argparse.ArgumentParser())
    exec(compile(ast.Module(body=[boolfn] + calls, type_ignores=[]), str(path), "exec"), ns)
    return ns["parser"]


def reference_defaults(root):
    return vars(reference_parser(root).parse_args([]))


def reference_config(root, output, partition="noniid_dir_20_a0p5"):
    """Execute upstream YAML loading/command construction, with temp output paths."""
    import importlib.util
    if partition not in {"noniid_dir_20_a0p5", "noniid_pat_10"}:
        raise ValueError(f"No supported external reference config for {partition!r}")
    path = Path(root).resolve() / "scripts/run_pfllib.py"
    spec = importlib.util.spec_from_file_location("fedfew_reference_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config_with_base(Path(root).resolve() / f"configs/cifar100/{partition}/algorithms/fedfews.yaml")
    command = module.build_pfllib_command(config, model_dir=Path(output) / "source_config_models", result_dir=Path(output) / "source_config_results")
    return config, vars(reference_parser(root).parse_args(command[2:])), command


def _differences(reference, migrated, prefix=""):
    """Report every differing YAML leaf, including additions/removals."""
    result = []
    for key in sorted(reference.keys() | migrated.keys()):
        path = f"{prefix}.{key}" if prefix else key
        left, right = reference.get(key), migrated.get(key)
        if key in reference and key in migrated and isinstance(left, dict) and isinstance(right, dict):
            result.extend(_differences(left, right, path))
        elif key not in reference or key not in migrated or left != right:
            result.append(dict(field=path, reference=left, migrated=right,
                               reference_present=key in reference, migrated_present=key in migrated))
    return result


def configuration_alignment(root, source_yaml, migrated_yaml, source_defaults,
                            migrated_defaults, source_command, runtime_arguments, actual_args):
    """Strict protocol check with explicit dataset-case/GPU execution overrides.

    The user's local YAML spells Cifar100 and uses GPU 2. A pristine upstream
    checkout spells cifar100 and uses GPU 0. Both may be references ONLY when
    the actual short run explicitly supplies the same dataset and GPU to both.
    Training/algorithm YAML differences still fail, even if a short-run override
    would otherwise mask them. The external runner and parser stay independent.
    """
    yaml_differences = _differences(source_yaml, migrated_yaml)
    failures = []
    for difference in yaml_differences:
        field = difference["field"]
        left, right = difference["reference"], difference["migrated"]
        allowed = (field == "dataset.name" and isinstance(left, str) and isinstance(right, str)
                   and left.casefold() == right.casefold() == "cifar100") or field == "device.id"
        allowed = allowed and difference["reference_present"] and difference["migrated_present"]
        difference["disposition"] = "explicit runtime override" if allowed else "rejected protocol difference"
        if not allowed:
            failures.append(f"YAML {field}: {left!r} != {right!r}")
    effective_differences = []
    for key, value in source_defaults.items():
        if key in {"model_dir", "result_dir"}:
            continue  # Output hooks supply separate directories on both endpoints.
        actual = migrated_defaults.get(key)
        if actual != value:
            allowed = key == "device_id" or (key == "dataset" and isinstance(actual, str)
                       and isinstance(value, str) and actual.casefold() == value.casefold())
            effective_differences.append(dict(field=key, reference=value, migrated=actual,
                                              disposition="explicit runtime override" if allowed else "rejected protocol difference"))
            if not allowed:
                failures.append(f"Effective default {key}: {value!r} != {actual!r}")
    runtime_command = source_command + list(runtime_arguments)
    runtime = vars(reference_parser(root).parse_args(runtime_command[2:]))
    # Check that the EXTERNAL parser receives the exact configuration actually
    # supplied to both server constructors, rather than just assuming overrides work.
    for key, value in runtime.items():
        if key not in {"model_dir", "result_dir"} and actual_args.get(key) != value:
            failures.append(f"Effective runtime {key}: {value!r} != {actual_args.get(key)!r}")
    return dict(status="FAIL" if failures else "PASS", failures=failures,
                yaml_differences=yaml_differences, effective_default_differences=effective_differences,
                runtime_arguments=list(runtime_arguments), source_runtime_args=runtime,
                source_runtime_command=runtime_command,
                policy="Only dataset-name case and GPU id may differ, explicitly overridden equally; all protocol fields remain strict")
