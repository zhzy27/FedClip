"""Four offline controls: synthetic full Avg, oracle spectrum repair, native sizes."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',default='../dataset');p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--dataset',default='Cifar100');p.add_argument('--num-clients',type=int,default=20)
    p.add_argument('--class-per-client',type=int,default=20);p.add_argument('--client-id',type=int,default=0)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--capacity-ratios',default='0.15,0.25,0.35,0.90',help='Unique rates in (0,1); native E3=min, mother/E1/E2/E4=max')
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--test-batch-size',type=int,default=16);p.add_argument('--lr',type=float,default=.005)
    p.add_argument('--f-lambda',type=float,default=.001);p.add_argument('--grad-clip',type=float,default=10.)
    p.add_argument('--atol',type=float,default=1e-5);p.add_argument('--rtol',type=float,default=1e-5)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('-did','--device-id',type=int,default=0)
    p.add_argument('--diagnostics',type=int,choices=[0,1],default=1,help='Detailed spectra/norm tables; mandatory construction checks remain ON')
    p.add_argument('--output-dir',required=True,help='New independent directory; tables only, no trained models')
    return p


def main():
    args=parser().parse_args()
    if args.device_id<0:raise ValueError('device-id must be nonnegative')
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.device_id) if args.device=='cuda' else '-1'
    from utils.whole_network_spectrum_diagnostic import run
    run(args)


if __name__=='__main__':main()
