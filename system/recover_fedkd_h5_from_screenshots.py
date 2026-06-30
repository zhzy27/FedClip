import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


# These paths were read from the FedKD screenshots referenced by the markdown
# files in C:\Users\Administrator\Desktop\实验结果\CNN实验结果.
SCREENSHOT_H5_PATHS = {
    "image-20260530194328656.png": "./result/Cifar10_FedKD_test_0_20260530_183336.h5",
    "image-20260531122638252.png": "./result/Cifar100_FedKD_test_0_20260531_031800.h5",
    "image-20260528103609004.png": "./result/TinyImagenet_FedKD_test_0_20260527_123108.h5",
    "image-20260526122518052.png": "./result/TinyImagenet_FedKD_test_0_20260526_084840.h5",
    "image-20260627124143497.png": "./result/Cifar10_FedKD_test_0_20260627_042552.h5",
    "image-20260628113138280.png": "./result/Cifar100_FedKD_test_0_20260627_191929.h5",
    "image-20260628114227287.png": "./result/TinyImagenet_FedKD_test_0_20260628_075854.h5",
    "image-20260524192422510.png": "./result/Cifar100_FedKD_test_0_20260524_191240.h5",
    "image-20260530120212675.png": "./result/TinyImagenet_FedKD_test_0_20260529_092810.h5",
    "image-20260613115532525.png": "./result/Cifar100_FedKD_test_0_20260613_024938.h5",
    "image-20260619192249201.png": "./result/TinyImagenet_FedKD_test_0_20260616_203411.h5",
    "image-20260615114721602.png": "./result/Cifar100_FedKD_test_0_20260615_015259.h5",
    "image-20260619192510895.png": "./result/TinyImagenet_FedKD_test_0_20260617_084004.h5",
    "image-20260615114131151.png": "./result/Cifar100_FedKD_test_0_20260615_101135.h5",
    "image-20260619193240536.png": "./result/TinyImagenet_FedKD_test_0_20260619_084900.h5",
    "image-20260607142453461.png": "./result/TinyImagenet_FedKD_test_0_20260607_020943.h5",
    "image-20260608121104436.png": "./result/TinyImagenet_FedKD_test_0_20260607_221630.h5",
    "image-20260629132513468.png": "./result/Cifar100_FedKD_test_0_20260629_023000.h5",
    "image-20260602120236279.png": "./result/Cifar10_FedKD_test_0_20260602_004048.h5",
    "image-20260604202542308.png": "./result/Cifar100_FedKD_test_0_20260604_191354.h5",
    "image-20260629133009705.png": "./result/TinyImagenet_FedKD_test_0_20260629_054250.h5",
    "image-20260603111054640.png": "./result/Cifar10_FedKD_test_0_20260603_015951.h5",
    "image-20260604203127256.png": "./result/Cifar100_FedKD_test_0_20260604_191238.h5",
    "image-20260603201746633.png": "./result/Cifar10_FedKD_test_0_20260603_181810.h5",
    "image-20260605105127267.png": "./result/Cifar100_FedKD_test_0_20260605_053224.h5",
    "image-20260606122416321.png": "./result/TinyImagenet_FedKD_test_0_20260606_032103.h5",
    "image-20260604112238855.png": "./result/Cifar10_FedKD_test_0_20260603_235018.h5",
    "image-20260612105832268.png": "./result/Cifar100_FedKD_test_0_20260611_163843.h5",
    "image-20260613115959356.png": "./result/Cifar100_FedKD_test_0_20260613_042419.h5",
}


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def safe_path_component(value):
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "default"


def float_path_component(value):
    text = str(value).strip()
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p")


def infer_dataset(md_name):
    name = md_name.lower().replace(" ", "")
    if "cifar100" in name:
        return "Cifar100", "100"
    if "cifar10" in name:
        return "Cifar10", "10"
    if "tiny" in name:
        return "TinyImagenet", "200"
    raise ValueError(f"无法从文件名推断数据集: {md_name}")


def infer_partition_tag(md_name):
    name = md_name.lower().replace(" ", "")
    pat_match = re.search(r"pat_(\d+)", name)
    if pat_match:
        return f"pat_cpc{pat_match.group(1)}"

    dir_match = re.search(r"dir_?(\d+(?:\.\d+)?)", name)
    if dir_match:
        return f"dir_alpha{float_path_component(dir_match.group(1))}"

    raise ValueError(f"无法从文件名推断异构划分: {md_name}")


def infer_join_ratio(md_name):
    name = md_name.lower().replace(" ", "")
    jr_match = re.search(r"jr(\d+(?:\.\d+)?)", name)
    return jr_match.group(1) if jr_match else "1.0"


def infer_model_family(dataset):
    return "CNN-5-512-tiny" if dataset == "TinyImagenet" else "CNN-5-512"


def infer_target_dir(target_root, md_path):
    dataset, num_classes = infer_dataset(md_path.name)
    model_family = infer_model_family(dataset)
    partition_tag = infer_partition_tag(md_path.name)
    join_ratio = infer_join_ratio(md_path.name)
    return (
        Path(target_root)
        / safe_path_component(dataset)
        / "FedKD"
        / safe_path_component(model_family)
        / partition_tag
        / f"ncl{num_classes}_niid1"
        / f"clients20_jr{float_path_component(join_ratio)}"
    )


def resolve_source_path(system_root, relative_h5_path):
    raw = relative_h5_path.replace("\\", "/")
    if raw.startswith("./"):
        raw = raw[2:]
    path = Path(raw)
    if path.is_absolute():
        return path
    return Path(system_root) / path


def find_fedkd_screenshots(md_path):
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if re.search(r"FedKD", line, re.IGNORECASE):
            for image_path in IMAGE_RE.findall(line):
                records.append({
                    "md": md_path,
                    "line": line_no,
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                })
    return records


def write_sidecar(target_h5, record, source_h5, target_dir):
    sidecar = {
        "recovered_by": "recover_fedkd_h5_from_screenshots.py",
        "recovered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "FedKD",
        "source_markdown": str(record["md"]),
        "markdown_line": record["line"],
        "source_screenshot": record["image_path"],
        "screenshot_name": record["image_name"],
        "source_h5": str(source_h5),
        "target_dir": str(target_dir),
    }
    with target_h5.with_suffix(target_h5.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=4)


def append_manifest(target_root, item):
    manifest = Path(target_root) / "fedkd_screenshot_recovery_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def recover(args):
    md_root = Path(args.md_root)
    system_root = Path(args.system_root)
    target_root = Path(args.target)

    md_files = sorted(md_root.glob("*.md"))
    records = []
    for md_path in md_files:
        records.extend(find_fedkd_screenshots(md_path))

    print(f"发现 md 文件: {len(md_files)}")
    print(f"发现 FedKD 截图引用: {len(records)}")

    copied = 0
    skipped = 0
    missing_mapping = 0
    missing_h5 = 0
    failed = 0

    for idx, record in enumerate(records, start=1):
        pct = idx * 100.0 / len(records) if records else 100.0
        image_name = record["image_name"]
        print(f"\n[{idx}/{len(records)}] {pct:.2f}% {record['md'].name} -> {image_name}")

        relative_h5 = SCREENSHOT_H5_PATHS.get(image_name)
        if not relative_h5:
            missing_mapping += 1
            print("未登记该 FedKD 截图中的 h5 路径，跳过。")
            continue

        try:
            source_h5 = resolve_source_path(system_root, relative_h5)
            target_dir = infer_target_dir(target_root, record["md"])
            target_h5 = target_dir / source_h5.name

            print(f"源 h5: {source_h5}")
            print(f"目标目录: {target_dir}")

            if not source_h5.exists():
                missing_h5 += 1
                print("源 h5 不存在，跳过。")
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            if target_h5.exists() and not args.overwrite:
                skipped += 1
                print(f"目标已存在，跳过: {target_h5}")
                continue

            if args.dry_run:
                print(f"DRY-RUN 将复制到: {target_h5}")
            else:
                shutil.copy2(source_h5, target_h5)
                write_sidecar(target_h5, record, source_h5, target_dir)
                append_manifest(target_root, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source_markdown": str(record["md"]),
                    "source_screenshot": record["image_path"],
                    "source_h5": str(source_h5),
                    "target_h5": str(target_h5),
                })
                copied += 1
                print(f"已复制到: {target_h5}")
        except Exception as exc:
            failed += 1
            print(f"处理失败: {exc}")

    print("\n处理结束")
    print(f"复制成功: {copied}")
    print(f"已存在跳过: {skipped}")
    print(f"未登记截图: {missing_mapping}")
    print(f"源 h5 缺失: {missing_h5}")
    print(f"失败: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Recover FedKD h5 files from the FedKD screenshot rows in experiment markdown files."
    )
    parser.add_argument(
        "--md-root",
        type=str,
        default=r"C:\Users\Administrator\Desktop\实验结果\CNN实验结果",
        help="Directory containing the markdown experiment summaries.",
    )
    parser.add_argument(
        "--system-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="FedClip/system directory. Screenshot paths like ./result/*.h5 are resolved under this directory.",
    )
    parser.add_argument("--target", type=str, default="./h5_results", help="Structured h5 target root.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned copies.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target h5 files.")
    recover(parser.parse_args())


if __name__ == "__main__":
    main()
