#!/usr/bin/env bash
# From server system/. DEFINE ONLY. Inputs are existing post-Avg R5 final models.
# Exactly five R6 branches x local5; only four heterogeneous Avg operations; R7 inference only.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u svd_r6_replay.py -did 0 --device cuda --h-run svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/H --hplus-run svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/Hplus --l-run svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/L --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --source-round 5 --target-client-id 0 --focus-client-ids 0,5,10,15 --threads 4 --output-dir svd_r6_replay_runs/pat20_S5_R6_R7_seed0_20261116_v1"
)
