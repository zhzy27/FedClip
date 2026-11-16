"""Read collected replay tables only; no Torch/model/data-loader imports."""
import argparse
import json
from pathlib import Path


def plot_curves(root,metrics,training,epochs,branches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths=[]
    for scope in ('all','fixed_large','target'):
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
        for branch in branches:
            for col,split in enumerate(('train','test')):
                rows=sorted([r for r in metrics if r['branch']==branch and r['phase']=='R6_local' and r['split']==split and r['evaluation_scope']==scope],key=lambda r:r['epoch'])
                axes[0,col].plot([r['epoch'] for r in rows],[100*r['accuracy'] for r in rows],marker='o',label=branch)
                axes[1,col].plot([r['epoch'] for r in rows],[r['ce'] for r in rows],marker='o',label=branch)
        for col,split in enumerate(('train','test')):
            axes[0,col].set_title(split+' accuracy');axes[0,col].set_ylabel('Accuracy (%)')
            axes[1,col].set_title(split+' pure CE');axes[1,col].set_ylabel('CE (nats/sample)')
        for ax in axes.flat:ax.set_xlabel('R6 local epoch');ax.set_xticks(range(epochs+1));ax.grid(alpha=.25)
        axes[0,0].legend(fontsize=8);fig.suptitle('Offline R6: '+scope+' (fixed budget, no test selection)')
        path=root/f'R6_curves_{scope}.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path.name)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for branch in branches:
        rows=sorted([r for r in training if r['branch']==branch and r['evaluation_scope']=='fixed_large'],key=lambda r:r['epoch'])
        axes[0].plot([r['epoch'] for r in rows],[r['train_mean_weighted_F'] for r in rows],marker='o',label=branch)
        axes[1].plot([r['epoch'] for r in rows],[r['clip_trigger_fraction'] for r in rows],marker='o',label=branch)
    axes[0].set_ylabel('Training weighted F');axes[1].set_ylabel('Clip trigger fraction / SGD step')
    for ax in axes:ax.set_xlabel('R6 local epoch');ax.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.suptitle('Fixed largest-capacity clients: training-process statistics')
    path=root/'R6_F_clip_fixed_large.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path.name)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    arms=[b for b in branches if b!='L_raw']
    for col,scope in enumerate(('all','fixed_large')):
        rows=[next(r for r in metrics if r['branch']==b and r['phase']=='R7_pre_ordinary' and r['split']=='test' and r['evaluation_scope']==scope) for b in arms]
        axes[0,col].bar(range(4),[100*r['accuracy'] for r in rows]);axes[0,col].set_title(scope+' test accuracy');axes[0,col].set_ylabel('Accuracy (%)')
        axes[1,col].bar(range(4),[r['ce'] for r in rows]);axes[1,col].set_title(scope+' test pure CE');axes[1,col].set_ylabel('CE (nats/sample)')
    for ax in axes.flat:ax.set_xticks(range(4),arms,rotation=20);ax.grid(axis='y',alpha=.25)
    fig.suptitle('R7 ordinary SVD download only (no second repair, no SGD)')
    path=root/'R7_ordinary_download.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path.name)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir');p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--check',action='store_true',help='Render a tiny in-memory figure; no model/data imports or filesystem output')
    cli=p.parse_args()
    if cli.check:
        import io
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots();ax.plot([0,1],[0,1]);fig.savefig(io.BytesIO(),format='png');plt.close(fig)
        print('Replay renderer preflight passed',flush=True);return
    if not cli.output_dir:p.error('--output-dir is required unless --check is used')
    root=Path(cli.output_dir).resolve()
    read=lambda name:json.loads((root/(name+'.json')).read_text(encoding='utf-8'))
    paths=plot_curves(root,read('cohort_endpoints'),read('cohort_epoch_training'),cli.epochs,read('manifest')['branches'])
    (root/'curves.json').write_text(json.dumps(paths,indent=2),encoding='utf-8')
    print('Rendered',len(paths),'replay figures',flush=True)


if __name__=='__main__':main()
