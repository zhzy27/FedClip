"""Offline fc1 rank-capacity experiment from a B-fc1 full server checkpoint."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run',help='B_fc1 run root: exactly one final_models/**/Server_model_0.pt')
    source.add_argument('--server-file',help='Complete server checkpoint with adjacent manifest.json')
    p.add_argument('--source-round',type=int,default=101,help='Completed source aggregations; offline training is source+1')
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100')
    p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--client-ids',default='0,10,18')
    p.add_argument('--fc1-ranks',default='120,200,720')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('-did','--device-id',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--diagnostics',type=int,choices=[0,1],default=1,
        help='Disable optional CE/slot/spectrum/gradient probes; keep endpoint and rollback evaluations')
    p.add_argument('--output-dir',required=True,help='Independent new directory; tables only, no trained models')
    return p


def main():
    args=parser().parse_args()
    if args.device_id<0:raise ValueError('device-id must be nonnegative')
    # Only the requested physical GPU is visible before any Torch import.
    if args.device=='cuda':os.environ['CUDA_VISIBLE_DEVICES']=str(args.device_id)
    from utils.fc1_capacity_diagnostic import run
    run(args)


if __name__=='__main__':main()
