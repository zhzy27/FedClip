#!/usr/bin/env bash
# Run from server system/. Definition only; no automatic execution.
# One coordinated process keeps H / Hplus / matrix paired; branch tables have separate directories.
# Requires same-run H R5 Server_model_0.pt, adjacent manifest and 20 ordinary Client_i_model_0.pt.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u matrix_comp_replay.py -did 0 --device cuda --source-run svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/H --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --target-client-id 0 --threads 4 --output-root matrix_comp_replay_runs/pat20_S5_R6_R7_seed0_20261116_v1"
)
