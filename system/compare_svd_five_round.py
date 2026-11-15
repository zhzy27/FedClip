"""Read-only S/L/H/H+ roundwise contrasts. No training or model selection."""
import argparse
import csv
import json
from pathlib import Path


def compare(runs):
    indexed={}
    for name,root in runs.items():
        root=Path(root)
        # Accept an exact metrics directory or a unique report below a run root.
        paths=[root/'five_cohort_metrics.json'] if (root/'five_cohort_metrics.json').is_file() else list(root.rglob('five_cohort_metrics.json'))
        if len(paths)!=1:raise ValueError(f'{name}: need exactly one five_cohort_metrics.json, got {len(paths)}')
        rows=json.loads(paths[0].read_text(encoding='utf-8'))
        manifest=json.loads((paths[0].parent/'manifest.json').read_text(encoding='utf-8'))
        protocol=manifest['five_round_diagnostics']
        common_keys=('seed','num_clients','num_server_models','num_classes','model_family','local_epochs','batch_size',
            'local_learning_rate','global_rounds','eval_gap','local_grad_clip','test_batch_size','is_regular',
            'regular_lamda','multi_svd_init','multi_svd_init_rank_rate','multi_proto_mode','multi_proto_lambda')
        common={k:manifest['effective'][k] for k in common_keys}
        identity=(protocol['shared_initialization_sha256'],protocol['data_fingerprint'],protocol['fixed_cohort_ratios'],common)
        if indexed and identity!=next(iter(indexed.values()))['identity']:raise ValueError('Data/initialization/fixed cohorts differ')
        lookup={(r['completed_train_count'],r['evaluation_scope'],r['split'],r['stage']):r for r in rows}
        if len(lookup)!=len(rows):raise ValueError('Duplicate stage metrics')
        indexed[name]=dict(rows=lookup,identity=identity)
    result=[]
    for high,low in [('L','S'),('L','H'),('Hplus','H')]:
        a=indexed[high]['rows'];b=indexed[low]['rows']
        keys={key for key in a if key[3]=='post_local'}
        if keys!={key for key in b if key[3]=='post_local'}:raise ValueError('Different observed rounds/cohorts')
        for r,scope,split,_ in sorted(keys):
            post=(r,scope,split,'post_local');pre=(r,scope,split,'download')
            if a[post]['member_ids']!=b[post]['member_ids']:raise ValueError('Fixed members differ')
            dpre=a[pre]['accuracy']-b[pre]['accuracy'];dpost=a[post]['accuracy']-b[post]['accuracy']
            result.append(dict(contrast=high+'_minus_'+low,completed_train_count=r,loop_round=r-1,
                evaluation_scope=scope,split=split,accuracy_unit='fraction',
                pre_accuracy_difference=dpre,post_accuracy_difference=dpost,
                same_round_local_gain_difference=dpost-dpre,
                pre_CE_difference=a[pre]['CE']-b[pre]['CE'],post_CE_difference=a[post]['CE']-b[post]['CE']))
    # Within H+, compare SAME current mother, not H's diverged server.
    a=indexed['Hplus']['rows']
    for key in sorted(k for k in a if k[3]=='same_mother_raw_counterfactual'):
        r,scope,split,_=key;corrected=a[r,scope,split,'download'];raw=a[key]
        result.append(dict(contrast='Hplus_same_mother_corrected_minus_raw',completed_train_count=r,loop_round=r-1,
            evaluation_scope=scope,split=split,accuracy_unit='fraction',
            pre_accuracy_difference=corrected['accuracy']-raw['accuracy'],post_accuracy_difference=None,
            same_round_local_gain_difference=None,pre_CE_difference=corrected['CE']-raw['CE'],post_CE_difference=None))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('S','L','H','Hplus'):p.add_argument('--'+key,required=True)
    p.add_argument('--output-dir',required=True);args=p.parse_args()
    rows=compare({name:getattr(args,name) for name in ('S','L','H','Hplus')})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'comparisons.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
    with (out/'comparisons.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# S/L/H/H+ fixed-round comparisons','Accuracy differences: percentage points; pure CE differences: nats/sample.',
        '| contrast | R | pre Δ(pp) | post Δ(pp) | local gain Δ(pp) |','| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['evaluation_scope']!='all' or row['split']!='test':continue
        fmt=lambda v:'N/A' if v is None else f'{100*v:.6f}'
        lines.append(f"| {row['contrast']} | {row['completed_train_count']} | {fmt(row['pre_accuracy_difference'])} | {fmt(row['post_accuracy_difference'])} | {fmt(row['same_round_local_gain_difference'])} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
