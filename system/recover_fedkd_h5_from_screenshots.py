import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


# The FedKD JSON logs for these runs were broken, so the old JSON-driven h5
# organizer could not place them correctly. I manually checked the FedKD
# screenshots in C:\Users\Administrator\Desktop\实验结果\CNN实验结果 and wrote the
# source h5 path plus target structured folder here directly.
FEDKD_H5_JOBS = [
    {
        "source": "./result/Cifar10_FedKD_test_0_20260530_183336.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/dir_alpha0p3/ncl10_niid1/clients20_jr1p0",
        "note": "dir_0.3_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260531_031800.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p3/ncl100_niid1/clients20_jr1p0",
        "note": "dir_0.3_cifar100",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260527_123108.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p3/ncl200_niid1/clients20_jr1p0",
        "note": "dir_0.3_tinyimagenet",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260526_084840.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p5/ncl200_niid1/clients20_jr1p0",
        "note": "dir_0.5_tinyimagenet",
    },
    {
        "source": "./result/Cifar10_FedKD_test_0_20260627_042552.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/dir_alpha0p8/ncl10_niid1/clients20_jr1p0",
        "note": "dir_0.8_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260627_191929.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p8/ncl100_niid1/clients20_jr1p0",
        "note": "dir_0.8_cifar100",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260628_075854.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p8/ncl200_niid1/clients20_jr1p0",
        "note": "dir_0.8_tinyimagenet",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260524_191240.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha1p0/ncl100_niid1/clients20_jr1p0",
        "note": "dir_1_cifar100",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260529_092810.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha1p0/ncl200_niid1/clients20_jr1p0",
        "note": "dir_1_tinyimagenet",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260613_024938.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p5/ncl100_niid1/clients20_jr0p2",
        "note": "jr0.2_gr500_cifar100_dir0.5",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260616_203411.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p5/ncl200_niid1/clients20_jr0p2",
        "note": "jr0.2_gr500_tinyImageNet_dir0.5",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260615_015259.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p5/ncl100_niid1/clients20_jr0p4",
        "note": "jr0.4_gr500_cifar100_dir0.5",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260617_084004.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p5/ncl200_niid1/clients20_jr0p4",
        "note": "jr0.4_gr500_tinyImageNet_dir0.5",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260615_101135.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p5/ncl100_niid1/clients20_jr0p8",
        "note": "jr0.8_gr500_cifar100_dir0.5",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260619_084900.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/dir_alpha0p5/ncl200_niid1/clients20_jr0p8",
        "note": "jr0.8_gr500_tinyImageNet_dir0.5",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260607_020943.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/pat_cpc120/ncl200_niid1/clients20_jr1p0",
        "note": "pat_120_tiny",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260607_221630.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/pat_cpc160/ncl200_niid1/clients20_jr1p0",
        "note": "pat_160_tiny",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260629_023000.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/pat_cpc20/ncl100_niid1/clients20_jr1p0",
        "note": "pat_20_cifar100",
    },
    {
        "source": "./result/Cifar10_FedKD_test_0_20260602_004048.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/pat_cpc2/ncl10_niid1/clients20_jr1p0",
        "note": "pat_2_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260604_191354.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/pat_cpc40/ncl100_niid1/clients20_jr1p0",
        "note": "pat_40_cifar100",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260629_054250.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/pat_cpc40/ncl200_niid1/clients20_jr1p0",
        "note": "pat_40_tiny",
    },
    {
        "source": "./result/Cifar10_FedKD_test_0_20260603_015951.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/pat_cpc4/ncl10_niid1/clients20_jr1p0",
        "note": "pat_4_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260604_191238.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/pat_cpc60/ncl100_niid1/clients20_jr1p0",
        "note": "pat_60_cifar100",
    },
    {
        "source": "./result/Cifar10_FedKD_test_0_20260603_181810.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/pat_cpc6/ncl10_niid1/clients20_jr1p0",
        "note": "pat_6_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260605_053224.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/pat_cpc80/ncl100_niid1/clients20_jr1p0",
        "note": "pat_80_cifar100",
    },
    {
        "source": "./result/TinyImagenet_FedKD_test_0_20260606_032103.h5",
        "target_dir": "TinyImagenet/FedKD/CNN-5-512-tiny/pat_cpc80/ncl200_niid1/clients20_jr1p0",
        "note": "pat_80_tiny",
    },
    {
        "source": "./result/Cifar10_FedKD_test_0_20260603_235018.h5",
        "target_dir": "Cifar10/FedKD/CNN-5-512/pat_cpc8/ncl10_niid1/clients20_jr1p0",
        "note": "pat_8_cifar10",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260611_163843.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p5/ncl100_niid1/clients20_jr0p4",
        "note": "z_drop_jr0.4_gr100_cifar100_dir0.5",
    },
    {
        "source": "./result/Cifar100_FedKD_test_0_20260613_042419.h5",
        "target_dir": "Cifar100/FedKD/CNN-5-512/dir_alpha0p5/ncl100_niid1/clients20_jr0p4",
        "note": "z_drop_jr0.4_gr250_cifar100_dir0.5",
    },
]


def resolve_under(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    text = value.replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    return Path(root) / text


def write_sidecar(target_h5, job, source_h5):
    sidecar = {
        "recovered_by": "recover_fedkd_h5_from_screenshots.py",
        "recovered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "FedKD",
        "note": job["note"],
        "source_h5": str(source_h5),
        "target_h5": str(target_h5),
        "target_dir": job["target_dir"],
    }
    with target_h5.with_suffix(target_h5.suffix + ".json").open("w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False, indent=4)


def append_manifest(target_root, record):
    manifest_path = Path(target_root) / "fedkd_h5_recovery_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_jobs():
    seen_sources = set()
    duplicate_sources = []
    for job in FEDKD_H5_JOBS:
        source = job["source"]
        if source in seen_sources:
            duplicate_sources.append(source)
        seen_sources.add(source)
    if duplicate_sources:
        raise ValueError("硬编码表里存在重复 source: " + ", ".join(duplicate_sources))


def recover(args):
    validate_jobs()
    system_root = Path(args.system_root)
    target_root = Path(args.target)

    print(f"硬编码 FedKD h5 搬运任务数: {len(FEDKD_H5_JOBS)}")
    print(f"system_root: {system_root}")
    print(f"target: {target_root}")

    copied = 0
    skipped = 0
    missing = 0
    failed = 0

    for idx, job in enumerate(FEDKD_H5_JOBS, start=1):
        pct = idx * 100.0 / len(FEDKD_H5_JOBS)
        source_h5 = resolve_under(system_root, job["source"])
        target_dir = target_root / job["target_dir"]
        target_h5 = target_dir / source_h5.name

        print(f"\n[{idx}/{len(FEDKD_H5_JOBS)}] {pct:.2f}% {job['note']}")
        print(f"源 h5: {source_h5}")
        print(f"目标 h5: {target_h5}")

        try:
            if not source_h5.exists():
                missing += 1
                print("源 h5 不存在，跳过。")
                continue

            target_dir.mkdir(parents=True, exist_ok=True)
            if target_h5.exists() and not args.overwrite:
                skipped += 1
                print("目标已存在，跳过。")
                continue

            if args.dry_run:
                print("DRY-RUN 将复制。")
                continue

            shutil.copy2(source_h5, target_h5)
            write_sidecar(target_h5, job, source_h5)
            append_manifest(target_root, {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_h5": str(source_h5),
                "target_h5": str(target_h5),
                "note": job["note"],
            })
            copied += 1
            print("已复制。")
        except Exception as exc:
            failed += 1
            print(f"处理失败: {exc}")

    print("\n处理结束")
    print(f"复制成功: {copied}")
    print(f"已存在跳过: {skipped}")
    print(f"源 h5 缺失: {missing}")
    print(f"失败: {failed}")


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Recover broken-json FedKD h5 files with a hard-coded checked mapping.")
    parser.add_argument("--system-root", type=str, default=str(script_dir), help="FedClip/system directory.")
    parser.add_argument("--target", type=str, default=str(script_dir / "h5_results"), help="Structured h5 result root.")
    parser.add_argument("--dry-run", action="store_true", help="Only print planned copies.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target h5 files.")
    recover(parser.parse_args())


if __name__ == "__main__":
    main()
