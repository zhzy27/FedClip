#!/usr/bin/env bash
# From system/. DEFINE ONLY: one simulation Avg, then four independent local5 runs.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u whole_network_spectrum_diagnostic.py -did 0 --device cuda --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --num-clients 20 --class-per-client 20 --client-id 0 --seed 0 --capacity-ratios 0.15,0.25,0.35,0.90 --epochs 5 --batch-size 16 --test-batch-size 16 --lr 0.005 --f-lambda 0.001 --grad-clip 10 --threads 4 --atol 1e-5 --rtol 1e-5 --diagnostics 1 --output-dir whole_network_spectrum_runs/pat20_C0_seed0_local5_20261114_v1"
)
