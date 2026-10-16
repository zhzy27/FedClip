#!/usr/bin/env bash
# Command definitions only. Sourcing this file DOES NOT launch training.
# Run from the FedClip repository root. Replace paths / GPU as needed.
FEW_PYTHON=${FEW_PYTHON:-python}
FEW_SOURCE_ROOT=${FEW_SOURCE_ROOT:-/data/zh/AllRepo/FedFew_reference}
FEW_DATA_ROOT=${FEW_DATA_ROOT:-/data/zh/AllRepo/FedFew/PFLlib/dataset}
FEW_OUTPUT_ROOT=${FEW_OUTPUT_ROOT:-fedfews_runs}

# Review first; validates original config.json alpha/partition and hashes all shards.
declare -a PREVIEW_COMMANDS=(
"$FEW_PYTHON system/run_fedfews.py --config system/fedfews_compat/configs/noniid_dir_20_a0p5/algorithms/fedfews.yaml --data-root '$FEW_DATA_ROOT' --dataset Cifar100/noniid_dir_20_a0p5 --expected-alpha 0.5 --expected-partition dir --source-root '$FEW_SOURCE_ROOT' --output-dir '$FEW_OUTPUT_ROOT/dir05_original' -dev cuda -did 0 --preview"
)

# External original source vs migration. Four actual rounds; evaluation at 0,2,3.
# Synthetic fixtures are written ONLY inside this independent acceptance output.
declare -a SYNTHETIC_ACCEPTANCE_COMMANDS=(
"$FEW_PYTHON system/verify_fedfews.py --source-root '$FEW_SOURCE_ROOT' --output-dir '$FEW_OUTPUT_ROOT/accept_synthetic_cpu' --num-clients 20 --rounds 4 --eval-gap 2 --local-epochs 1 --batch-size 4 --device cpu --torch-threads 2 --control-seed 0 --atol 0 --rtol 0"
)

# Existing real shards, three actual rounds; eval at 0 and forced-final 2, skip 1.
# CPU exact comparison is preferred. GPU tolerances must be explicit and logged.
declare -a REAL_ACCEPTANCE_COMMANDS=(
"$FEW_PYTHON system/verify_fedfews.py --source-root '$FEW_SOURCE_ROOT' --data-root '$FEW_DATA_ROOT' --dataset Cifar100/noniid_dir_20_a0p5 --expected-alpha 0.5 --expected-partition dir --num-clients 20 --rounds 3 --eval-gap 10 --local-epochs 1 --batch-size 50 --device cuda --device-id 0 --torch-threads 2 --control-seed 0 --atol 1e-6 --rtol 1e-5 --output-dir '$FEW_OUTPUT_ROOT/accept_dir05_cuda'"
)

# Formal original-RNG protocol: 0..2000 inclusive = 2001 training/aggregation steps.
# Python/NumPy are intentionally NOT newly seeded. Source client Torch resets remain.
# No --torch-threads override: like upstream, -nth 1 is sequential CLIENT execution,
# not torch.set_num_threads(1). Actual runtime thread counts are in the manifest.
declare -a COMMANDS=(
"$FEW_PYTHON system/run_fedfews.py --config system/fedfews_compat/configs/noniid_dir_20_a0p5/algorithms/fedfews.yaml --data-root '$FEW_DATA_ROOT' -data Cifar100/noniid_dir_20_a0p5 --expected-alpha 0.5 --expected-partition dir --source-root '$FEW_SOURCE_ROOT' --rng-mode original --output-dir '$FEW_OUTPUT_ROOT/dir05_original' -algo FedFewS -m CNN -ncl 100 -nc 20 -nsm 3 -smu 0.01 -rep False -gr 2000 -ls 1 -lbs 50 -lr 0.005 -jr 1.0 -eg 10 -nth 1 -mo 0 -fs 0 -cdr 0 -tsr 0 -ssr 0 -tth 10000 -pv 0 -t 1 -go ours -uc True -dev cuda -did 0"
)

# Optional, only AFTER the Dirichlet first-stage real-data acceptance.
# No alpha is inferred for pat10. Actual config alpha and label statistics are saved.
declare -a PAT10_COMMANDS=(
"$FEW_PYTHON system/run_fedfews.py --config system/fedfews_compat/configs/noniid_pat_10/algorithms/fedfews.yaml --data-root '$FEW_DATA_ROOT' -data Cifar100/noniid_pat_10 --expected-partition pat --source-root '$FEW_SOURCE_ROOT' --rng-mode original --output-dir '$FEW_OUTPUT_ROOT/pat10_original' -algo FedFewS -m CNN -ncl 100 -nc 10 -nsm 3 -smu 0.01 -rep False -gr 2000 -ls 1 -lbs 50 -lr 0.005 -jr 1.0 -eg 10 -nth 1 -mo 0 -fs 0 -cdr 0 -tsr 0 -ssr 0 -tth 10000 -pv 0 -t 1 -go ours -uc True -dev cuda -did 1"
)
