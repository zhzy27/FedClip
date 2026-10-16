#!/usr/bin/env bash
# Definitions only; run/supply to the usual scheduler FROM THE system DIRECTORY.
# This is the ordinary-CNN native baseline, never the low-rank FedavgCNN mapping.
FEW_NATIVE_PYTHON=${FEW_NATIVE_PYTHON:-python}
FEW_NATIVE_DATA_ROOT=${FEW_NATIVE_DATA_ROOT:-/data/zh/AllRepo/FedFew/PFLlib/dataset}
FEW_NATIVE_RUN_TAG=${FEW_NATIVE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
FEW_NATIVE_RUN_ROOT=${FEW_NATIVE_RUN_ROOT:-/data/zh/AllRepo/FedClip/fedfews_native_runs/$FEW_NATIVE_RUN_TAG}
FEW_NATIVE_GPU_DIR=${FEW_NATIVE_GPU_DIR:-2}
FEW_NATIVE_GPU_PAT=${FEW_NATIVE_GPU_PAT:-3}

# Original RNG protocol; 0..2000 inclusive = 2001 train/aggregate steps.
# Per-client data traversals per round: 3 models x 1 epoch = 3.
# Actual Dirichlet alpha is .1, DESPITE the historical a0p5 folder name.
declare -a COMMANDS=(
"$FEW_NATIVE_PYTHON main.py -algo FedFewS -data Cifar100/noniid_dir_20_a0p5 -m FedAvgCNN-plain -did $FEW_NATIVE_GPU_DIR -dev cuda -ncl 100 -nc 20 -nsm 3 -smu 0.01 -lr 0.005 -lbs 50 -ls 1 -gr 2000 -jr 1.0 -eg 10 -pv 0 -t 1 -go ours -is_regular 0 -fs 0 -cdr 0 -tsr 0 -ssr 0 -pt dir -dir_alpha 0.1 --fedfews_rng original --data-root '$FEW_NATIVE_DATA_ROOT' --data-formats pkl_first --data-expected-alpha 0.1 --data-expected-partition dir -sfn '$FEW_NATIVE_RUN_ROOT/dir01/checkpoints' --run_log_dir '$FEW_NATIVE_RUN_ROOT/dir01/logs' --h5_result_root '$FEW_NATIVE_RUN_ROOT/dir01/h5' --final_model_root '$FEW_NATIVE_RUN_ROOT/dir01/final_models' -exp_name FedFewS_plain_dir01_K3_original"
"$FEW_NATIVE_PYTHON main.py -algo FedFewS -data Cifar100/noniid_pat_10 -m FedAvgCNN-plain -did $FEW_NATIVE_GPU_PAT -dev cuda -ncl 100 -nc 10 -nsm 3 -smu 0.01 -lr 0.005 -lbs 50 -ls 1 -gr 2000 -jr 1.0 -eg 10 -pv 0 -t 1 -go ours -is_regular 0 -fs 0 -cdr 0 -tsr 0 -ssr 0 -pt pat -cpc 10 --fedfews_rng original --data-root '$FEW_NATIVE_DATA_ROOT' --data-formats pkl_first --data-expected-partition pat -sfn '$FEW_NATIVE_RUN_ROOT/pat10/checkpoints' --run_log_dir '$FEW_NATIVE_RUN_ROOT/pat10/logs' --h5_result_root '$FEW_NATIVE_RUN_ROOT/pat10/h5' --final_model_root '$FEW_NATIVE_RUN_ROOT/pat10/final_models' -exp_name FedFewS_plain_pat10_K3_original"
)

declare -a PREVIEW_COMMANDS=(
"${COMMANDS[0]} --fedfews_preview"
"${COMMANDS[1]} --fedfews_preview"
)

# CPU exact parity: frozen c73ec5b compatibility vs actual main.py, no external
# FedFew checkout/YAML needed. Real shards are read in place, never regenerated.
declare -a SYNTHETIC_ACCEPTANCE_COMMANDS=(
"$FEW_NATIVE_PYTHON verify_fedfews_native.py --output-dir '$FEW_NATIVE_RUN_ROOT/accept_synthetic_cpu' --num-clients 20 --rounds 3 --eval-gap 10 --batch-size 4 --local-epochs 1 --device cpu --device-id $FEW_NATIVE_GPU_DIR --threads 2 --atol 0 --rtol 0"
)
declare -a REAL_CPU_ACCEPTANCE_COMMANDS=(
"$FEW_NATIVE_PYTHON verify_fedfews_native.py --data-root '$FEW_NATIVE_DATA_ROOT' --dataset Cifar100/noniid_dir_20_a0p5 --expected-alpha 0.1 --expected-partition dir --num-clients 20 --rounds 3 --eval-gap 10 --local-epochs 1 --batch-size 50 --device cpu --device-id $FEW_NATIVE_GPU_DIR --threads 2 --atol 0 --rtol 0 --output-dir '$FEW_NATIVE_RUN_ROOT/accept_dir01_cpu'"
"$FEW_NATIVE_PYTHON verify_fedfews_native.py --data-root '$FEW_NATIVE_DATA_ROOT' --dataset Cifar100/noniid_pat_10 --expected-partition pat --num-clients 10 --rounds 3 --eval-gap 10 --local-epochs 1 --batch-size 50 --device cpu --device-id $FEW_NATIVE_GPU_PAT --threads 2 --atol 0 --rtol 0 --output-dir '$FEW_NATIVE_RUN_ROOT/accept_pat10_cpu'"
)

# Same explicit CUDA tolerance as previous acceptance; never relax to mask a fail.
declare -a REAL_CUDA_ACCEPTANCE_COMMANDS=(
"$FEW_NATIVE_PYTHON verify_fedfews_native.py --data-root '$FEW_NATIVE_DATA_ROOT' --dataset Cifar100/noniid_dir_20_a0p5 --expected-alpha 0.1 --expected-partition dir --num-clients 20 --rounds 3 --eval-gap 10 --local-epochs 1 --batch-size 50 --device cuda --device-id $FEW_NATIVE_GPU_DIR --threads 2 --atol 1e-6 --rtol 1e-5 --output-dir '$FEW_NATIVE_RUN_ROOT/accept_dir01_cuda'"
"$FEW_NATIVE_PYTHON verify_fedfews_native.py --data-root '$FEW_NATIVE_DATA_ROOT' --dataset Cifar100/noniid_pat_10 --expected-partition pat --num-clients 10 --rounds 3 --eval-gap 10 --local-epochs 1 --batch-size 50 --device cuda --device-id $FEW_NATIVE_GPU_PAT --threads 2 --atol 1e-6 --rtol 1e-5 --output-dir '$FEW_NATIVE_RUN_ROOT/accept_pat10_cuda'"
)
