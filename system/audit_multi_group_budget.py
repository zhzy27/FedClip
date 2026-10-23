"""Standalone, RNG-preserving parameter audit; does not run federated training."""
import argparse
from pathlib import Path

from common.experiment import write_json
from utils.multi_group_budget import DEFAULT_CONFIG, build_audit, markdown_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--capacity-manifest', help='Optional actual download capacity_manifest.json from this run')
    parser.add_argument('--group-file', help='Optional new-batch fixed group file')
    parser.add_argument('--previous-group-file', help='Optional previous batch membership, comparison only')
    parser.add_argument('--output', required=True, help='Independent report directory (never a checkpoint)')
    args = parser.parse_args()
    report = build_audit(args.config,args.capacity_manifest,args.group_file,args.previous_group_file)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'parameter_budget.json',report)
    (output/'parameter_budget.md').write_text(markdown_report(report),encoding='utf-8')
    print(f"PASS: {len(report['clients'])} clients, K={report['configuration']['num_server_models']}; {output.resolve()}")
    print('Parameter budget only; communication unchanged, peak memory not measured.')


if __name__ == '__main__': main()
