"""Five paired offline R6 branches from H/H+/L post-Avg R5 servers."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for label in ('h','hplus','l'):
        group=p.add_mutually_exclusive_group(required=True)
        group.add_argument('--'+label+'-run',help='Run root with exactly one final_models/**/Server_model_0.pt, or exact checkpoint directory')
        group.add_argument('--'+label+'-server-file',help='Explicit full server with adjacent same-run manifest.json')
    p.add_argument('-did','--device-id',type=int,default=0)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100')
    p.add_argument('--data-subdir',default='pat_20');p.add_argument('--source-round',type=int,choices=[5],default=5)
    p.add_argument('--target-client-id',type=int,default=0)
    p.add_argument('--focus-client-ids',default='0,5,10,15',help='Same highest-capacity clients in all five branches')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--temp-root',help='Optional scratch parent for transient recovered uploads; default a cleaned subdirectory of output')
    p.add_argument('--output-dir',required=True,help='New independent directory; tables and curves only')
    return p


def main():
    cli=parser().parse_args()
    if cli.device_id<0:raise ValueError('GPU ID must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES']=str(cli.device_id) if cli.device=='cuda' else '-1'
    from utils.svd_r6_replay import run
    run(cli)


if __name__=='__main__':main()
