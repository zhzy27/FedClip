"""Read collected R1-R10 diagnostics; never load a model or execute training."""
import argparse
import csv
import json
from pathlib import Path

from compare_svd_five_round import compare

CASES=('S120','S200','H','H_first_band')


def comparisons(runs):
    roots={};manifests={}
    for case,path in runs.items():
        path=Path(path);files=[path/'five_cohort_metrics.json'] if (path/'five_cohort_metrics.json').is_file() else list(path.rglob('five_cohort_metrics.json'))
        if len(files)!=1:raise ValueError(f'{case}: require one exact metrics directory, found {len(files)}')
        roots[case]=files[0].parent;manifests[case]=json.loads((files[0].parent/'manifest.json').read_text(encoding='utf-8'))
    rates=manifests['H']['svd_capacity_protocol']['client_rank_rates'];tiers=sorted(set(rates))
    if len(tiers)<2:raise ValueError('H must contain at least two distinct actual capacities')
    expected=dict(S120=[tiers[0]]*len(rates),S200=[tiers[1]]*len(rates),H=rates,H_first_band=rates)
    for case,m in manifests.items():
        if (m['svd_capacity_protocol']['client_rank_rates']!=expected[case] or m['aggregation_mode']!='group_avg_only'
            or m['effective']['multi_svd_spectral_mode']!=('first_band' if case=='H_first_band' else 'none')):
            raise ValueError(f'{case}: wrong capacity, aggregation or spectral mode')
    return compare(roots,contrasts=[('H_first_band','H'),('S200','S120')],counterfactual_branch='H_first_band')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in CASES:p.add_argument('--'+name,required=True)
    p.add_argument('--output-dir',required=True);args=p.parse_args()
    rows=comparisons({name:getattr(args,name) for name in CASES});out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparisons.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'comparisons.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# First-band and uniform capacity comparisons','Accuracy differences: percentage points; CE differences: nats/sample.',
        '| contrast | R | pre (pp) | post (pp) | change during local5 (pp) |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['evaluation_scope']!='all' or row['split']!='test':continue
        fmt=lambda v:'N/A' if v is None else f'{100*v:.6f}'
        lines.append(f"| {row['contrast']} | {row['completed_train_count']} | {fmt(row['pre_accuracy_difference'])} | {fmt(row['post_accuracy_difference'])} | {fmt(row['same_round_local_gain_difference'])} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
