import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


JSON_NAME_RE = re.compile(
    r"^algo_(?P<algorithm>.+?)-dataset(?P<dataset>.+?)-"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.json$"
)


def safe_path_component(value):
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "default"


def float_path_component(value):
    return str(value).replace(".", "p")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {to_jsonable(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def parse_json_name(path):
    match = JSON_NAME_RE.match(path.name)
    return match.groupdict() if match else {}


def parse_datetime(text):
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def h5_datetime_from_name(path):
    match = re.search(r"(\d{8}_\d{6})", path.name)
    return parse_datetime(match.group(1)) if match else None


def read_h5_acc(path):
    with h5py.File(path, "r") as hf:
        if "rs_test_acc" not in hf:
            return None
        return np.asarray(hf["rs_test_acc"], dtype=float)


def build_h5_index(source_dir):
    records = []
    bad_h5 = []
    for path in sorted(Path(source_dir).rglob("*.h5")):
        try:
            test_acc = read_h5_acc(path)
        except Exception as exc:
            bad_h5.append((path, str(exc)))
            continue
        if test_acc is None or len(test_acc) == 0:
            bad_h5.append((path, "missing or empty rs_test_acc"))
            continue
        records.append({
            "path": path,
            "test_acc": test_acc,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime),
            "name_time": h5_datetime_from_name(path),
            "used": False,
        })
    return records, bad_h5


def load_json_records(json_dir):
    records = []
    bad_json = []
    for path in sorted(Path(json_dir).rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            bad_json.append((path, str(exc)))
            continue

        if not isinstance(data, dict) or not data.get("test_acc"):
            continue

        name_meta = parse_json_name(path)
        args_dict = data.get("args", {}) if isinstance(data.get("args", {}), dict) else {}
        dataset = args_dict.get("dataset") or name_meta.get("dataset") or "unknown_dataset"
        algorithm = args_dict.get("algorithm") or name_meta.get("algorithm") or "unknown_algorithm"
        json_time = parse_datetime(name_meta.get("timestamp"))
        declared_h5_paths = args_dict.get("save_file_paths", [])
        if isinstance(declared_h5_paths, str):
            declared_h5_paths = [declared_h5_paths]
        if not isinstance(declared_h5_paths, list):
            declared_h5_paths = []
        records.append({
            "path": path,
            "data": data,
            "args": args_dict,
            "dataset": dataset,
            "algorithm": algorithm,
            "test_acc": np.asarray(data["test_acc"], dtype=float),
            "json_time": json_time,
            "declared_h5_paths": declared_h5_paths,
        })
    return records, bad_json


def partition_path_component(args_dict):
    partition = args_dict.get("partition", "unknown_partition")
    if partition == "dir":
        return f"dir_alpha{float_path_component(args_dict.get('dir_alpha', 'default'))}"
    if partition == "pat":
        return f"pat_cpc{args_dict.get('class_per_client', 'default')}"
    if partition == "exdir":
        return f"exdir_alpha{float_path_component(args_dict.get('dir_alpha', 'default'))}"
    return safe_path_component(partition)


def target_dir_from_json(target_root, json_record):
    args_dict = json_record["args"]
    dataset = args_dict.get("dataset") or json_record["dataset"]
    algorithm = args_dict.get("algorithm") or json_record["algorithm"]
    model_family = args_dict.get("model_family", "unknown_model")
    data_tag = f"ncl{args_dict.get('num_classes', 'unknown')}_niid{args_dict.get('niid', 'unknown')}"
    join_tag = (
        f"clients{args_dict.get('num_clients', 'unknown')}_"
        f"jr{float_path_component(args_dict.get('join_ratio', 'unknown'))}"
    )
    return Path(target_root) / safe_path_component(dataset) / safe_path_component(algorithm) / \
        safe_path_component(model_family) / partition_path_component(args_dict) / data_tag / join_tag


def curves_match(json_acc, h5_acc):
    return len(json_acc) == len(h5_acc) and np.allclose(
        json_acc,
        h5_acc,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )


def time_gap_seconds(json_record, h5_record):
    json_time = json_record.get("json_time")
    h5_time = h5_record.get("name_time") or h5_record.get("mtime")
    if json_time is None or h5_time is None:
        return float("inf")
    return abs((json_time - h5_time).total_seconds())


def h5_name_from_declared_path(value):
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    return Path(text).name


def find_h5_by_declared_path(json_record, h5_records, allow_reuse=False):
    declared_names = {
        h5_name_from_declared_path(path)
        for path in json_record.get("declared_h5_paths", [])
    }
    declared_names.discard("")
    if not declared_names:
        return None

    candidates = [
        h5_record
        for h5_record in h5_records
        if (allow_reuse or not h5_record["used"])
        and h5_record["path"].name in declared_names
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda item: (time_gap_seconds(json_record, item), str(item["path"])))
    return candidates[0]


def find_h5_for_json(json_record, h5_records, allow_reuse=False):
    declared_match = find_h5_by_declared_path(json_record, h5_records, allow_reuse=allow_reuse)
    if declared_match is not None:
        return declared_match

    candidates = [
        h5_record
        for h5_record in h5_records
        if (allow_reuse or not h5_record["used"])
        and curves_match(json_record["test_acc"], h5_record["test_acc"])
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (time_gap_seconds(json_record, item), str(item["path"])))
    return candidates[0]


def unique_target_path(target_dir, source_name):
    target = target_dir / source_name
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    idx = 1
    while True:
        candidate = target_dir / f"{stem}_dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def write_sidecar(target_h5_path, json_record, source_h5_path):
    sidecar_path = target_h5_path.with_suffix(target_h5_path.suffix + ".json")
    sidecar = {
        "source_h5": str(source_h5_path),
        "source_json": str(json_record["path"]),
        "dataset": json_record["dataset"],
        "algorithm": json_record["algorithm"],
        "args": json_record["args"],
    }
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(sidecar), f, ensure_ascii=False, indent=4)


def append_manifest(target_root, record):
    manifest_path = Path(target_root) / "organized_h5_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def organize(args):
    target_root = Path(args.target)
    h5_records, bad_h5 = build_h5_index(args.source)
    json_records, bad_json = load_json_records(args.json_dir)

    print(f"发现 h5 文件: {len(h5_records)}")
    print(f"发现可用 json 文件: {len(json_records)}")
    if bad_h5:
        print(f"跳过异常 h5: {len(bad_h5)}")
    if bad_json:
        print(f"跳过损坏 json: {len(bad_json)}")

    copied = 0
    skipped = 0
    unmatched_json = 0
    failed = 0

    for idx, json_record in enumerate(json_records, start=1):
        pct = idx * 100.0 / len(json_records) if json_records else 100.0
        print(f"\n[{idx}/{len(json_records)}] {pct:.2f}% 正在处理 JSON: {json_record['path'].name}")
        try:
            h5_record = find_h5_for_json(json_record, h5_records, allow_reuse=args.allow_reuse_h5)
            if h5_record is None:
                unmatched_json += 1
                print("未找到与该 JSON test_acc 完全匹配的 h5，跳过。")
                continue

            target_dir = target_dir_from_json(target_root, json_record)
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / h5_record["path"].name

            if target_path.exists() and target_path.stat().st_size == h5_record["path"].stat().st_size and not args.overwrite:
                print(f"已存在且大小一致，跳过: {target_path}")
                skipped += 1
                h5_record["used"] = True
                continue
            if target_path.exists() and not args.overwrite:
                target_path = unique_target_path(target_dir, h5_record["path"].name)

            if args.dry_run:
                print(f"DRY-RUN JSON: {json_record['path']}")
                print(f"DRY-RUN 源 h5: {h5_record['path']}")
                print(f"DRY-RUN 目标 h5: {target_path}")
            else:
                shutil.copy2(h5_record["path"], target_path)
                write_sidecar(target_path, json_record, h5_record["path"])
                print(f"已复制到: {target_path}")
                copied += 1

            h5_record["used"] = True
            append_manifest(target_root, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_json": str(json_record["path"]),
                "source_h5": str(h5_record["path"]),
                "target_h5": str(target_path),
                "dry_run": args.dry_run,
            })
        except Exception as exc:
            failed += 1
            print(f"处理失败: {exc}")

    unused_h5 = [record for record in h5_records if not record["used"]]
    print("\n处理结束")
    print(f"复制成功: {copied}")
    print(f"已存在跳过: {skipped}")
    print(f"未匹配 JSON: {unmatched_json}")
    print(f"未被 JSON 认领的 h5: {len(unused_h5)}")
    print(f"失败: {failed}")

    target_root.mkdir(parents=True, exist_ok=True)
    if unused_h5:
        unused_log = target_root / "unmatched_h5.log"
        with unused_log.open("w", encoding="utf-8") as f:
            for record in unused_h5:
                f.write(str(record["path"]) + "\n")
        print(f"未匹配 h5 列表已写入: {unused_log}")

    if bad_json or bad_h5:
        bad_log = target_root / "bad_inputs.log"
        with bad_log.open("w", encoding="utf-8") as f:
            for path, err in bad_json:
                f.write(f"BAD JSON: {path}\n{err}\n\n")
            for path, err in bad_h5:
                f.write(f"BAD H5: {path}\n{err}\n\n")
        print(f"异常输入详情已写入: {bad_log}")


def main():
    parser = argparse.ArgumentParser(description="Copy old flat H5 files into structured folders by iterating JSON logs first.")
    parser.add_argument("--source", type=str, default="./result", help="Old flat H5 result directory")
    parser.add_argument("--json-dir", type=str, default="./json", help="JSON log directory used as the source of experiment args")
    parser.add_argument("--target", type=str, default="./h5_results", help="Structured target root")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned copies")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite same target file if it already exists")
    parser.add_argument("--allow-reuse-h5", action="store_true", help="Allow multiple JSON files to match the same H5 curve")
    organize(parser.parse_args())


if __name__ == "__main__":
    main()
