"""Locate original or organized experiment logs by a final-model run ID.

Run from system/:
    python find_log_by_model.py ./final_models/.../runs/<run_id>/Client_0_model.pt
    python find_log_by_model.py <model_path_1> <model_path_2> --log-root ./log
    python find_log_by_model.py <model_path> --log-root ./8-21-03.zip --include-zips

Model files need not exist locally. Only logs are read; no model is loaded.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import sys
import zipfile


RUN_PATH = re.compile(r"(?:^|/)runs/([0-9]+_([0-9]+)_[0-9]+)(?:/|$)")
RUN_ID = re.compile(r"[0-9]+_([0-9]+)_[0-9]+")
LOG_DATE = re.compile(r"(?<![0-9])([0-9]{8}_[0-9]{6})(?![0-9])")


@dataclass(frozen=True)
class Query:
    model_path: str
    run_id: str
    pid: str

    @classmethod
    def parse(cls, value):
        normalized = value.strip().strip("\"'").replace("\\", "/")
        match = RUN_PATH.search(normalized)
        if match:
            return cls(value, match[1], match[2])
        match = RUN_ID.fullmatch(normalized)
        if match:
            return cls(value, normalized, match[1])
        raise ValueError(
            f"Cannot extract run ID from {value!r}; use a model path containing "
            "runs/<timestamp_ns>_<pid>_<trial>/, or the complete run ID."
        )


@dataclass(frozen=True)
class Match:
    path: str
    line_number: int
    evidence: str
    timestamp: float
    kind: str


def log_timestamp(path, fallback):
    # Organized names and original run folder names retain experiment time.
    for value in reversed(LOG_DATE.findall(path.replace("\\", "/"))):
        try:
            return datetime.strptime(value, "%Y%m%d_%H%M%S").timestamp()
        except ValueError:
            continue
    return fallback


def scan_lines(lines, path, fallback_time, queries, allow_pid_fallback=False):
    exact = [None] * len(queries)
    pid_matches = [None] * len(queries)
    patterns = [re.compile(r"(?<![0-9])" + re.escape(q.run_id) + r"(?![0-9_])")
                for q in queries]
    pid_patterns = [re.compile(r"(?<![0-9])" + re.escape(q.pid) + r"(?![0-9])")
                    for q in queries]
    timestamp = log_timestamp(path, fallback_time)
    for line_number, raw in enumerate(lines, 1):
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        for index, pattern in enumerate(patterns):
            if exact[index] is None and pattern.search(line):
                exact[index] = Match(path, line_number, line.strip(), timestamp, "run_id")
            elif (allow_pid_fallback and pid_matches[index] is None
                  and pid_patterns[index].search(line)):
                pid_matches[index] = Match(path, line_number, line.strip(), timestamp, "pid_only")
    return exact, pid_matches


def iter_sources(roots, include_zips):
    seen = set()
    for root in roots:
        candidates = [root] if root.is_file() else (
            Path(folder) / name
            for folder, _, names in os.walk(root)
            for name in names
        )
        for path in candidates:
            if path.suffix.lower() != ".log" and not (
                include_zips and path.suffix.lower() == ".zip"
            ):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def find_logs(queries, roots, include_zips=False, allow_pid_fallback=False):
    exact = [[] for _ in queries]
    fallback = [[] for _ in queries]
    errors = []
    scanned = 0

    def collect(lines, name, timestamp):
        nonlocal scanned
        matches, pid_matches = scan_lines(
            lines, name, timestamp, queries, allow_pid_fallback
        )
        scanned += 1
        for index in range(len(queries)):
            if matches[index]:
                exact[index].append(matches[index])
            elif pid_matches[index]:
                fallback[index].append(pid_matches[index])

    for path in iter_sources(roots, include_zips):
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for entry in archive.infolist():
                        if entry.is_dir() or not entry.filename.lower().endswith(".log"):
                            continue
                        with archive.open(entry) as stream:
                            collect(stream, f"{path}::{entry.filename}",
                                    datetime(*entry.date_time).timestamp())
            else:
                with path.open("rb") as stream:
                    collect(stream, str(path), path.stat().st_mtime)
        except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
            errors.append(f"{path}: {exc}")

    results = []
    for matches, candidates in zip(exact, fallback):
        # PID may be reused. Only offer weak matches when no full run ID matched.
        results.append(sorted(matches or candidates,
                              key=lambda row: (row.timestamp, row.path), reverse=True))
    return results, scanned, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_paths", nargs="+", help="Model paths or complete run IDs.")
    parser.add_argument("--log-root", nargs="+", type=Path, default=[Path.cwd()],
                        help="Search directories or log files (default: current directory, recursively).")
    parser.add_argument("--include-zips", action="store_true",
                        help="Also search .log members inside ZIP files without extracting them.")
    parser.add_argument("--allow-pid-fallback", action="store_true",
                        help="If no run ID matches, show full-PID candidates for manual verification.")
    args = parser.parse_args(argv)
    try:
        queries = [Query.parse(value) for value in args.model_paths]
    except ValueError as exc:
        parser.error(str(exc))
    for root in args.log_root:
        if not root.exists():
            parser.error(f"Search root does not exist: {root}")
        if root.is_file() and root.suffix.lower() == ".zip" and not args.include_zips:
            parser.error("Searching a ZIP requires --include-zips.")

    results, scanned, errors = find_logs(
        queries, args.log_root, args.include_zips, args.allow_pid_fallback
    )
    print(f"Scanned {scanned} log files; results ordered newest first.")
    for query, matches in zip(queries, results):
        print(f"\nModel: {query.model_path}\nRun ID: {query.run_id} | PID: {query.pid}")
        if not matches:
            print("NOT FOUND: no matching log under the supplied search roots.")
        for index, match in enumerate(matches, 1):
            print(f"  [{index}] {match.path}")
            print(f"      Match: {match.kind} | line {match.line_number}")
            print(f"      Evidence: {match.evidence}")
        if matches and matches[0].kind == "pid_only":
            print("  WARNING: PID-only candidates are not confirmed matches; PIDs can be reused.")
    for error in errors:
        print(f"WARNING: could not fully read {error}", file=sys.stderr)
    return 2 if errors else (0 if all(results) else 1)


if __name__ == "__main__":
    raise SystemExit(main())
