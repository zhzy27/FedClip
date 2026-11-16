"""Render collected matrix-compensation curves; deliberately no Torch import."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(a.output_root);rows=json.loads((root/'cohort_endpoints.json').read_text())
    scopes=list(dict.fromkeys(r['scope'] for r in rows));plots=[]
    for scope in scopes:
        fig,axes=plt.subplots(2,4,figsize=(18,8))
        for ri,round_id in enumerate((6,7)):
            for branch in ('H','Hplus','matrix'):
                for si,split in enumerate(('train','test')):
                    selected=sorted((r for r in rows if r['round']==round_id and r['branch']==branch and r['phase']=='local' and r['scope']==scope and r['split']==split),key=lambda r:r['epoch'])
                    for ci,(field,scale) in enumerate((('accuracy',100),('ce',1))):
                        ax=axes[ri,2*si+ci];ax.plot([r['epoch'] for r in selected],[scale*r[field] for r in selected],marker='o',label=branch)
                        ax.set(xlabel='Local epoch',ylabel='Accuracy (%)' if field=='accuracy' else 'Pure CE (nats/sample)',title=f'R{round_id} {split} {scope}')
                        ax.grid(alpha=.2);ax.legend()
        fig.suptitle('Paired offline R6/R7: original Avg, all_tail, subgroup matrix compensation')
        fig.tight_layout();name='curves_'+scope+'.png';fig.savefig(root/name,dpi=120);plt.close(fig);plots.append(name)
    (root/'curves.json').write_text(json.dumps(plots,indent=2),encoding='utf-8')


if __name__=='__main__':main()
