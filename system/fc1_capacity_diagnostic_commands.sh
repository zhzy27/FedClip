#!/usr/bin/env bash
# From server system/. Definition only: offline R102, never starts FL training.
# -did is the physical GPU; default exports ONLY diagnostic tables.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u fc1_capacity_diagnostic.py -did 0 --device cuda --source-run svd_spectral_runs/pat20_seed0_init090_gr100_20261113_v1/B_fc1 --source-round 101 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --client-ids 0,10,18 --fc1-ranks 120,200,720 --threads 4 --diagnostics 1 --output-dir fc1_capacity_diagnostic_runs/B_fc1_S101_offline_R102_r120_200_720_20261114_v1"
)
