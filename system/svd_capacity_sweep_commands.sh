#!/usr/bin/env bash
# From server system/. DEFINE ONLY: offline inference, no SGD.
# Source roots verified against S94 (11-17.tar.gz) L/H actual training logs.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u svd_capacity_sweep.py -did 0 --device cuda --source-name L --source-run svd_first_band_runs/pat20_seed0_init090_R100_20261117_v1/L --completed-round 100 --client-ids 0,5,10,15 --model-id 0 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --rank-rates 0.90,0.60,0.45,0.35,0.25,0.15 --batch-size 16 --threads 4 --max-drop-pp 0.5 --output-dir svd_capacity_sweep_runs/pat20_R100_20261125_v1/L"
  "/data/zh/miniconda3/envs/htfllib/bin/python -u svd_capacity_sweep.py -did 1 --device cuda --source-name H --source-run svd_first_band_runs/pat20_seed0_init090_R100_20261117_v1/H --completed-round 100 --client-ids 0,5,10,15 --model-id 0 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --rank-rates 0.90,0.60,0.45,0.35,0.25,0.15 --batch-size 16 --threads 4 --max-drop-pp 0.5 --output-dir svd_capacity_sweep_runs/pat20_R100_20261125_v1/H"
)
