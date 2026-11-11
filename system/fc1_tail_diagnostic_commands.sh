#!/usr/bin/env bash
# From server system/. DEFINE ONLY: one offline experiment, no federated training.
# Default cohorts: source assignment's .35 IDs 2/7/12/17 and .90 IDs 0/5/10/15.
# Override --client-ids if desired; -did is the physical GPU exposed to this process.
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u fc1_tail_diagnostic.py -did 0 --device cuda --source-run rank_partition_runs/pat20_seed0_gr100_20261110_v1/FA02_full_avg_class_sample_proto_control --source-round 101 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --client-rates 0.35,0.90 --epochs 5 --batch-size 16 --test-batch-size 16 --lr 0.005 --f-lambda 0.001 --proto-lambda 1.0 --grad-clip 10 --seed 0 --threads 4 --diagnostics 1 --output-dir fc1_tail_diagnostic_runs/FA02_R101_offline_R102_floor120_20261111_v1"
)
