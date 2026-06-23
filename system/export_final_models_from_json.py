import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime


def safe_path_component(value):
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "default"


def float_path_component(value):
    return str(value).replace(".", "p")


def as_dict(value):
    return value if isinstance(value, dict) else {}


def get_field(data, key, default=None):
    args = as_dict(data.get("args"))
    if key in args:
        return args[key]
    return data.get(key, default)


def partition_path_component(data):
    partition = get_field(data, "partition", "iid")
    if partition == "dir":
        alpha = get_field(data, "dir_alpha", "default")
        return f"dir_alpha{float_path_component(alpha)}"
    if partition == "pat":
        cpc = get_field(data, "class_per_client", "default")
        return f"pat_cpc{cpc}"
    if partition == "exdir":
        alpha = get_field(data, "dir_alpha", "default")
        return f"exdir_alpha{float_path_component(alpha)}"
    return safe_path_component(partition)


def resolve_path(path_text, base_dir):
    if not path_text:
        return ""
    if os.path.isabs(path_text):
        return os.path.normpath(path_text)
    return os.path.normpath(os.path.join(base_dir, path_text))


def source_model_dir(data, base_dir):
    path_text = (
        get_field(data, "save_folder_name_full")
        or get_field(data, "source_dir")
        or get_field(data, "save_folder_name")
    )
    return resolve_path(path_text, base_dir)


def final_model_dir(data, final_model_root, base_dir):
    dataset = get_field(data, "dataset", "unknown_dataset")
    algorithm = get_field(data, "algorithm", "unknown_algorithm")
    model_family = get_field(data, "model_family", get_field(data, "model", "unknown_model"))
    num_classes = get_field(data, "num_classes", "default")
    niid = get_field(data, "niid", "default")
    num_clients = get_field(data, "num_clients", "default")
    join_ratio = get_field(data, "join_ratio", "default")
    root = resolve_path(final_model_root, base_dir)
    data_tag = f"ncl{num_classes}_niid{niid}"
    join_tag = f"clients{num_clients}_jr{float_path_component(join_ratio)}"
    return os.path.join(
        root,
        safe_path_component(dataset),
        safe_path_component(algorithm),
        safe_path_component(model_family),
        partition_path_component(data),
        data_tag,
        join_tag,
    )


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def processed_index_path(work_dir):
    return os.path.join(work_dir, "processed_jsons.txt")


def load_processed_index(work_dir):
    index_path = processed_index_path(work_dir)
    if not os.path.exists(index_path):
        return set()
    with open(index_path, "r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def append_processed_index(work_dir, json_key):
    with open(processed_index_path(work_dir), "a", encoding="utf-8") as file:
        file.write(json_key + "\n")


def json_source_key(source_path, source_dir):
    return os.path.relpath(source_path, source_dir).replace("\\", "/")


def copy_jsons_to_work_dir(source_dir, work_dir):
    os.makedirs(work_dir, exist_ok=True)
    if not source_dir:
        return 0
    source_dir = os.path.normpath(source_dir)
    work_dir = os.path.normpath(work_dir)
    processed = load_processed_index(work_dir)
    copied = 0
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            source_path = os.path.join(root, filename)
            source_key = json_source_key(source_path, source_dir)
            if source_key in processed:
                continue
            rel_path = os.path.relpath(source_path, source_dir)
            target_path = os.path.join(work_dir, rel_path)
            if os.path.exists(target_path):
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied += 1
    return copied


def list_json_files(work_dir):
    json_files = []
    for root, _, filenames in os.walk(work_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_files.append(os.path.join(root, filename))
    return sorted(json_files)


def copy_model_files(source_dir, target_dir):
    copied_files = []
    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        raise RuntimeError(f"源目录和目标目录相同，跳过: {source_dir}")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"源模型目录不存在: {source_dir}")

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    for filename in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, filename)
        if not os.path.isfile(source_path) or not filename.endswith(".pt"):
            continue
        shutil.copy2(source_path, os.path.join(target_dir, filename))
        copied_files.append(filename)

    if not copied_files:
        raise RuntimeError(f"源目录存在，但没有找到任何 .pt 模型文件: {source_dir}")
    return copied_files


def count_model_files(source_dir):
    if not os.path.isdir(source_dir):
        return None
    return sum(
        1
        for filename in os.listdir(source_dir)
        if filename.endswith(".pt") and os.path.isfile(os.path.join(source_dir, filename))
    )


def write_manifest(target_dir, data, json_path, source_dir, copied_files):
    manifest = {
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "json_path": json_path,
        "source_dir": source_dir,
        "target_dir": target_dir,
        "exported_files": copied_files,
        "dataset": get_field(data, "dataset"),
        "algorithm": get_field(data, "algorithm"),
        "model_family": get_field(data, "model_family", get_field(data, "model")),
        "partition": get_field(data, "partition"),
        "dir_alpha": get_field(data, "dir_alpha"),
        "class_per_client": get_field(data, "class_per_client"),
        "num_classes": get_field(data, "num_classes"),
        "niid": get_field(data, "niid"),
        "num_clients": get_field(data, "num_clients"),
        "join_ratio": get_field(data, "join_ratio"),
        "raw_json": data,
    }
    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=4)


def remove_empty_parents(path, stop_dir):
    path = os.path.dirname(path)
    stop_dir = os.path.abspath(stop_dir)
    while os.path.abspath(path).startswith(stop_dir) and os.path.abspath(path) != stop_dir:
        try:
            os.rmdir(path)
        except OSError:
            break
        path = os.path.dirname(path)


def process_one_json(json_path, args):
    data = load_json(json_path)
    source_dir = source_model_dir(data, args.base_dir)
    target_dir = final_model_dir(data, args.final_model_root, args.base_dir)
    copied_files = copy_model_files(source_dir, target_dir)
    write_manifest(target_dir, data, json_path, source_dir, copied_files)
    append_processed_index(args.work_dir, os.path.relpath(json_path, args.work_dir).replace("\\", "/"))
    os.remove(json_path)
    remove_empty_parents(json_path, args.work_dir)
    return source_dir, target_dir, len(copied_files)


def append_failure_log(work_dir, json_path, exc):
    log_path = os.path.join(work_dir, "failed_jsons.log")
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"JSON: {json_path}\n")
        file.write(f"ERROR: {exc}\n\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export old temp model folders to final_models according to JSON logs."
    )
    parser.add_argument("--json-source-dir", type=str, default="",
                        help="可选：旧 JSON 日志目录。传入后会先复制到工作目录，再处理工作目录。")
    parser.add_argument("--work-dir", type=str, default="./json_model_export_work",
                        help="待处理 JSON 工作目录。处理成功一个 JSON 就删除一个，方便断点续跑。")
    parser.add_argument("--final-model-root", type=str, default="./final_models",
                        help="最终模型保存根目录，路径结构与 main.py 训练结束导出一致。")
    parser.add_argument("--base-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="解析 temp/... 和 final_models/... 这类相对路径时使用的基准目录，默认是 system 目录。")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要复制的源目录和目标目录，不真正复制，也不删除 JSON。")
    return parser.parse_args()


def main():
    args = parse_args()
    args.base_dir = os.path.abspath(args.base_dir)
    args.work_dir = resolve_path(args.work_dir, args.base_dir)
    os.makedirs(args.work_dir, exist_ok=True)

    json_source_dir = resolve_path(args.json_source_dir, args.base_dir) if args.json_source_dir else ""
    copied_json_count = copy_jsons_to_work_dir(json_source_dir, args.work_dir)
    if copied_json_count > 0:
        print(f"已复制 {copied_json_count} 个 JSON 到工作目录: {args.work_dir}")

    json_files = list_json_files(args.work_dir)
    total = len(json_files)
    if total == 0:
        print(f"工作目录中没有待处理 JSON: {args.work_dir}")
        return

    print(f"开始处理 JSON 工作目录: {args.work_dir}")
    print(f"待处理 JSON 数量: {total}")
    print(f"最终模型根目录: {resolve_path(args.final_model_root, args.base_dir)}")

    success_count = 0
    dry_run_count = 0
    failed_count = 0
    for index, json_path in enumerate(json_files, start=1):
        percent = index * 100.0 / total
        rel_json_path = os.path.relpath(json_path, args.work_dir)
        print(f"\n[{index}/{total}] {percent:.2f}% 正在处理: {rel_json_path}")
        try:
            data = load_json(json_path)
            source_dir = source_model_dir(data, args.base_dir)
            target_dir = final_model_dir(data, args.final_model_root, args.base_dir)
            if args.dry_run:
                dry_run_count += 1
                model_file_count = count_model_files(source_dir)
                print(f"DRY-RUN 源目录: {source_dir}")
                print(f"DRY-RUN 目标目录: {target_dir}")
                if model_file_count is None:
                    print("DRY-RUN 检查: 源目录不存在，正式运行时会失败。")
                else:
                    print(f"DRY-RUN 检查: 源目录存在，发现 {model_file_count} 个 .pt 文件。")
                continue

            source_dir, target_dir, file_count = process_one_json(json_path, args)
            success_count += 1
            print(f"完成: 复制 {file_count} 个 .pt 文件")
            print(f"源目录: {source_dir}")
            print(f"目标目录: {target_dir}")
        except Exception as exc:
            failed_count += 1
            append_failure_log(args.work_dir, json_path, exc)
            print(f"失败: {json_path}")
            print(f"原因: {exc}")

    print("\n处理结束")
    if args.dry_run:
        print(f"预演成功: {dry_run_count}")
    else:
        print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    if failed_count > 0:
        print(f"失败的 JSON 已保留在工作目录，修复问题后重新运行即可继续: {args.work_dir}")
        print(f"失败详情已写入: {os.path.join(args.work_dir, 'failed_jsons.log')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。已成功处理的 JSON 已删除，剩余 JSON 保留在工作目录。")
        sys.exit(130)
