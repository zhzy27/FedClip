"""Offline whole-CNN rank sweep of ordinary post-local prediction checkpoints."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-run',help='Exact run root or final prediction-checkpoint directory')
    group.add_argument('--checkpoint',action='append',help='Repeat CLIENT_ID=PATH; bare PATH allowed for one selected client')
    group.add_argument('--compare-results',nargs='+',help='Read completed sweep directories only, no checkpoint/data evaluation')
    p.add_argument('--source-name',default='source');p.add_argument('--manifest',help='Explicit same-run manifest when not adjacent to checkpoint')
    p.add_argument('--completed-round','--source-round',dest='completed_round',type=int,default=100)
    p.add_argument('--client-ids',default='0,5,10,15');p.add_argument('--model-id',type=int,default=0)
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100');p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--rank-rates',default='.90,.60,.45,.35,.25,.15')
    p.add_argument('-did','--device-id',type=int,default=0);p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-drop-pp',type=float,default=.5,help='Reporting reference in percentage points, not a training/model-selection rule')
    p.add_argument('--roundtrip-atol',type=float,default=1e-4);p.add_argument('--roundtrip-rtol',type=float,default=1e-4)
    p.add_argument('--output-dir',required=True)
    return p


def main():
    a=parser().parse_args()
    if a.device_id<0:raise ValueError('device-id must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.device_id) if a.device=='cuda' else '-1'
    from utils.svd_capacity_sweep import run,compare_results
    (compare_results if a.compare_results else run)(a)


if __name__=='__main__':main()
