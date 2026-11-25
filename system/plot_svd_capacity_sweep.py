"""Plot already collected tables only; intentionally no Torch import."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();root=Path(a.output_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((root/'manifest.json').read_text());cohorts=json.loads((root/'cohort_results.json').read_text())
    fig,ax=plt.subplots(figsize=(10,6))
    if (root/'client_results.json').is_file():
        rows=json.loads((root/'client_results.json').read_text())
        for cid in manifest['client_ids']:
            chosen=sorted((r for r in rows if r['variant']=='resvd' and r['client_id']==cid),key=lambda r:r['parameter_count'])
            ax.plot([r['parameter_count']/1e6 for r in chosen],[r['drop_pp'] for r in chosen],alpha=.45,marker='.',label=f'C{cid}')
    for source in dict.fromkeys(r['source'] for r in cohorts):
        chosen=sorted((r for r in cohorts if r['source']==source),key=lambda r:r['parameter_count'])
        ax.plot([r['parameter_count']/1e6 for r in chosen],[r['drop_pp'] for r in chosen],marker='o',linewidth=2,label=source+' sample-weighted')
        native=next(r for r in chosen if r['rank_rate']==.9)
        ax.scatter([native['parameter_count']/1e6],[0],marker='D',s=65,label=source+' native .90 reference',zorder=5)
    limits=[manifest['requested']['max_drop_pp']] if 'requested' in manifest else [m['requested']['max_drop_pp'] for m in manifest['sources']]
    for limit in sorted(set(limits)):ax.axhline(limit,color='red',linestyle='--',label=f'Reference: {limit:g} pp')
    ax.axhline(0,color='gray',linewidth=.8);ax.set(xlabel='Actual whole-model parameters (millions)',ylabel='Drop vs OWN native .90 (percentage points)',
        title='Offline whole-network SVD capacity sweep; measured points only')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(root/'parameters_vs_drop.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
