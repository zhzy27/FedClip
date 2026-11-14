#!/usr/bin/env bash
# From system/. Definition only: six offline local tasks, no federated training.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u fc1_capacity_diagnostic.py -did 0 --device cuda --comparison spectrum --source-run svd_spectral_runs/pat20_seed0_init090_gr100_20261113_v1/B_fc1 --source-round 101 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --client-ids 0,10,18 --fc1-ranks 200 --threads 4 --diagnostics 1 --output-dir fc1_capacity_diagnostic_runs/B_fc1_S101_offline_R102_raw_vs_corrected_r200_20261114_v1"
)
