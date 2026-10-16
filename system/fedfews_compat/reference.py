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


def reference_config(root, output):
    """Execute upstream YAML loading/command construction, with temp output paths."""
    import importlib.util
    path = Path(root).resolve() / "scripts/run_pfllib.py"
    spec = importlib.util.spec_from_file_location("fedfew_reference_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.load_config_with_base(Path(root).resolve() / "configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml")
    command = module.build_pfllib_command(config, model_dir=Path(output) / "source_config_models", result_dir=Path(output) / "source_config_results")
    return config, vars(reference_parser(root).parse_args(command[2:])), command
