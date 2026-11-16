"""Three paired R6/R7 local rounds: H, existing all_tail, subgroup matrix compensation."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-run',help='H run root with one final_models/**/Server_model_0.pt, or exact checkpoint directory')
    group.add_argument('--server-file',help='H post-Avg R5 full server, adjacent manifest required')
    p.add_argument('--clients-dir',help='Same-round ordinary Client_i_model_0.pt directory; default server directory')
    p.add_argument('-did','--device-id',type=int,default=0);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100');p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--source-atol',type=float,default=1e-5);p.add_argument('--source-rtol',type=float,default=1e-5)
    p.add_argument('--matrix-rank-rtol',type=float,default=None,help='Default max(matrix dimensions)*source machine epsilon')
    p.add_argument('--matrix-rank-atol',type=float,default=0.)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--target-client-id',type=int,default=0)
    p.add_argument('--temp-root');p.add_argument('--output-root',required=True,help='New root; separate H/Hplus/matrix tables and figures')
    return p


def main():
    cli=parser().parse_args()
    if cli.device_id<0:raise ValueError('GPU ID must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES']=str(cli.device_id) if cli.device=='cuda' else '-1'
    from utils.matrix_comp_replay import run
    run(cli)


if __name__=='__main__':main()
