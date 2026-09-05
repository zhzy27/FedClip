#!/usr/bin/env python3
"""Find experiment logs by the final reported best accuracy.

Examples:
    python find_logs_by_best_acc.py \
        --algorithm FedCLIP \
        --model Decom_CNN-5-512 \
        --best-acc 0.5521

    python find_logs_by_best_acc.py \
        --algorithm FedCLIP \
        --model Decom_CNN-5-512 \
        --best-acc 55.21092

The accuracy comparison is a prefix match after converting percentage input to
the decimal representation printed by the training code. Values greater than 1
are interpreted as percentages without reducing their decimal precision, so
``55.21092`` matches a log value beginning with ``0.5521092``. Multiple prefixes
may be supplied after ``--best-acc``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ARGUMENT_RE = re.compile(r"^\s*(algorithm|model_family)\s*=\s*(.*?)\s*$", re.IGNORECASE)
BEST_MARKER_RE = re.compile(r"^\s*Best accuracy\.?\s*$", re.IGNORECASE)
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?$")
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{8})_(\d{6})(?!\d)")


@dataclass(frozen=True)
class ParsedLog:
    path: Path
    algorithm: Optional[str]
    model: Optional[str]
    best_accuracy: Decimal
    best_accuracy_text: str
    run_time: datetime
    run_time_source: str
    modified_time: float


def _clean_line(line: str) -> str:
    return ANSI_ESCAPE_RE.sub("", line).strip()


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", ""}:
        return "0"
    return text


def _as_decimal_accuracy(raw: str) -> Optional[Decimal]:
    text = raw.strip()
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if is_percent or abs(value) > 1:
        value /= Decimal(100)
    return value


def normalize_accuracy_prefix(raw: str) -> str:
    """Normalize decimal or percentage input to a decimal accuracy prefix."""
    value = _as_decimal_accuracy(raw)
    if value is None:
        raise ValueError(f"Invalid accuracy prefix: {raw!r}")
    if value < 0 or value > 1:
        raise ValueError(f"Accuracy prefix must represent a value in [0, 1]: {raw!r}")

    # Preserve explicit decimal trailing zeros because they are meaningful in a
    # prefix search. Percentage inputs are shifted to the decimal scale first.
    stripped = raw.strip()
    if not stripped.endswith("%"):
        try:
            original = Decimal(stripped)
        except InvalidOperation:
            original = None
        if original is not None and abs(original) <= 1 and "e" not in stripped.lower():
            if stripped.startswith("."):
                return "0" + stripped
            if stripped.startswith("+."):
                return "0" + stripped[1:]
            if stripped.startswith("+0"):
                return stripped[1:]
            return stripped
    return _decimal_text(value)


def _extract_run_time(path: Path) -> Tuple[datetime, str, float]:
    stat = path.stat()
    for part in (path.name, *reversed(path.parts[:-1])):
        match = TIMESTAMP_RE.search(part)
        if not match:
            continue
        try:
            return (
                datetime.strptime("_".join(match.groups()), "%Y%m%d_%H%M%S"),
                "filename",
                stat.st_mtime,
            )
        except ValueError:
            continue
    return datetime.fromtimestamp(stat.st_mtime), "mtime", stat.st_mtime


def parse_log(path: Path) -> Optional[ParsedLog]:
    """Read one log and extract metadata plus its last Best accuracy value."""
    algorithm = None
    model = None
    best_accuracy = None
    best_accuracy_text = None
    waiting_for_best_value = False

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = _clean_line(raw_line)
                if not line:
                    continue

                argument_match = ARGUMENT_RE.match(line)
                if argument_match:
                    name, value = argument_match.groups()
                    if name.casefold() == "algorithm":
                        algorithm = value.strip()
                    else:
                        model = value.strip()

                if BEST_MARKER_RE.match(line):
                    waiting_for_best_value = True
                    continue

                if waiting_for_best_value:
                    waiting_for_best_value = False
                    if not NUMBER_RE.match(line):
                        continue
                    parsed = _as_decimal_accuracy(line)
                    if parsed is not None:
                        best_accuracy = parsed
                        best_accuracy_text = _decimal_text(parsed)
    except OSError:
        return None

    if best_accuracy is None or best_accuracy_text is None:
        return None

    run_time, run_time_source, modified_time = _extract_run_time(path)
    return ParsedLog(
        path=path.resolve(),
        algorithm=algorithm,
        model=model,
        best_accuracy=best_accuracy,
        best_accuracy_text=best_accuracy_text,
        run_time=run_time,
        run_time_source=run_time_source,
        modified_time=modified_time,
    )


def find_matching_logs(
    log_root: Path,
    algorithm: str,
    model: str,
    accuracy_prefixes: Sequence[str],
) -> Tuple[List[Tuple[ParsedLog, Tuple[str, ...]]], int, int]:
    """Return matches, scanned log count, and logs with a readable Best accuracy."""
    normalized_prefixes = tuple(normalize_accuracy_prefix(value) for value in accuracy_prefixes)
    matches: List[Tuple[ParsedLog, Tuple[str, ...]]] = []
    scanned = 0
    parsed_count = 0

    if not log_root.is_dir():
        return matches, scanned, parsed_count

    for path in log_root.rglob("*.log"):
        if not path.is_file():
            continue
        scanned += 1
        parsed = parse_log(path)
        if parsed is None:
            continue
        parsed_count += 1
        if parsed.algorithm is None or parsed.algorithm.casefold() != algorithm.casefold():
            continue
        if parsed.model is None or parsed.model.casefold() != model.casefold():
            continue

        matched_prefixes = tuple(
            prefix for prefix in normalized_prefixes if parsed.best_accuracy_text.startswith(prefix)
        )
        if matched_prefixes:
            matches.append((parsed, matched_prefixes))

    matches.sort(
        key=lambda item: (item[0].run_time, item[0].modified_time, str(item[0].path)),
        reverse=True,
    )
    return matches, scanned, parsed_count


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Find every training log whose final Best accuracy starts with the "
            "given value, newest experiment first."
        )
    )
    parser.add_argument("--algorithm", required=True, help="Exact algorithm name, e.g. FedCLIP.")
    parser.add_argument("--model", required=True, help="Exact model_family value.")
    parser.add_argument(
        "--best-acc",
        "--acc",
        dest="best_acc",
        nargs="+",
        required=True,
        help=(
            "One or more accuracy prefixes. Decimal and percentage forms are accepted, "
            "e.g. 0.5521092, 55.21092, or 55.21092%%. Values greater than 1 "
            "are treated as percentages without rounding."
        ),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=script_dir / "log",
        help="Log directory to search recursively (default: system/log).",
    )
    return parser


def _percentage_text(value: Decimal) -> str:
    return _decimal_text(value * Decimal(100)) + "%"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        normalized = [normalize_accuracy_prefix(value) for value in args.best_acc]
    except ValueError as exc:
        parser.error(str(exc))

    log_root = args.log_root.expanduser().resolve()
    matches, scanned, parsed_count = find_matching_logs(
        log_root=log_root,
        algorithm=args.algorithm,
        model=args.model,
        accuracy_prefixes=args.best_acc,
    )

    print(f"Log root: {log_root}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Model: {args.model}")
    print(f"Best accuracy prefix: {', '.join(normalized)}")
    print("-" * 72)

    for index, (record, matched_prefixes) in enumerate(matches, start=1):
        print(
            f"[{index}] {record.run_time:%Y-%m-%d %H:%M:%S} | "
            f"Best accuracy={record.best_accuracy_text} "
            f"({_percentage_text(record.best_accuracy)})"
        )
        if len(normalized) > 1:
            print(f"    Matched prefix: {', '.join(matched_prefixes)}")
        print(f"    {record.path}")

    print("-" * 72)
    print(
        f"Found {len(matches)} match(es); scanned {scanned} log file(s), "
        f"of which {parsed_count} contained a readable final Best accuracy."
    )
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
