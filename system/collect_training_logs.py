"""Copy an existing run's small reports into a downloadable model-free ZIP."""

import argparse
from pathlib import Path
import shutil

from utils.training_logs import archive_logs

REPORT_SUFFIXES = {".csv", ".json", ".log", ".h5", ".hdf5", ".txt"}


def collect_logs(source, output, extra=()):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError("Source must be an existing experiment directory.")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and destination must be separate directory trees.")
    extra = [Path(path).resolve() for path in extra]
    if any(not path.is_file() or path.suffix.lower() not in REPORT_SUFFIXES for path in extra):
        raise ValueError("Extra files must be logs/reports, not model tensors.")
    output.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in REPORT_SUFFIXES:
            destination = output / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for path in extra:
        destination = output / path.name
        if destination.exists():
            raise ValueError(f"Extra report name already exists: {path.name}")
        shutil.copy2(path, destination)
    return archive_logs(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path, nargs="*", default=[])
    options = parser.parse_args()
    print(collect_logs(options.source, options.output, options.extra))
