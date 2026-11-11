"""Offline paired fc1 tail-factor experiment. Never enters a federated server."""
import argparse
import os


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run',help='Explicit FA02 run root; require exactly one final_models/**/Server_model_0.pt')
    source.add_argument('--server-file',help='Explicit complete server checkpoint; packet and manifest must be adjacent')
    p.add_argument('--source-round',type=int,default=101)
    p.add_argument('--data-root',default='../dataset');p.add_argument('--dataset',default='Cifar100')
    p.add_argument('--data-subdir',default='pat_20')
    p.add_argument('--client-ids',help='Comma-separated IDs; overrides capacity-rate selection')
    p.add_argument('--client-rates',default='0.35,0.90')
    p.add_argument('--epochs',type=int);p.add_argument('--batch-size',type=int);p.add_argument('--test-batch-size',type=int)
    p.add_argument('--lr',type=float);p.add_argument('--f-lambda',type=float);p.add_argument('--proto-lambda',type=float)
    p.add_argument('--grad-clip',type=float);p.add_argument('--seed',type=int)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('-did','--device-id',type=int,default=0)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--init-atol',type=float,default=1e-6);p.add_argument('--init-rtol',type=float,default=1e-5)
    p.add_argument('--diagnostics',type=int,choices=[0,1],default=1,help='0 disables CE/gradient/spectral probes, not paired training or endpoint tests')
    p.add_argument('--historical-client-dir',help='Optional R101 post-local checkpoints: read-only historical evaluation ONLY')
    p.add_argument('--save-models',action='store_true',help='Opt-in final offline students; default saves diagnostics only')
    p.add_argument('--output-dir',required=True,help='New independent directory; existing directory refused')
    return p


def main():
    args=parser().parse_args()
    if args.device_id<0:raise ValueError('device-id must be nonnegative')
    # Do this BEFORE importing Torch. Legacy local recovery's CUDA-state helper
    # can then see only the one requested physical device, never other GPUs.
    if args.device=='cuda':os.environ['CUDA_VISIBLE_DEVICES']=str(args.device_id)
    from utils.fc1_tail_diagnostic import run
    run(args)


if __name__=='__main__':main()
