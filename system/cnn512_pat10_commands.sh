#!/usr/bin/env bash
# Definitions only. Run from FedClip/system; reads EXISTING 20-client pat_10.
PAT10_PYTHON=${PAT10_PYTHON:-python}
PAT10_DATA_ROOT=${PAT10_DATA_ROOT:-$(cd ../dataset && pwd)}
PAT10_RUN_TAG=${PAT10_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
PAT10_RUN_ROOT=${PAT10_RUN_ROOT:-../fedfews_native_runs/pat10/$PAT10_RUN_TAG}
PAT10_GPU_AVG=${PAT10_GPU_AVG:-0}
PAT10_GPU_FEW=${PAT10_GPU_FEW:-1}

# Both: same complete CNN-512, NPZ pat_10, 20 clients, seed0, SGD .005,
# batch16, each model local5, clip10, F/Prox off, shared/joint, every-round
# all-client post-local evaluation. gr100 = 101 actual train/aggregate rounds.
# Budget/client/round: Avg=5 data epochs; FedFewS=3*5=15 data epochs.
declare -a COMMANDS=(
"$PAT10_PYTHON main.py -algo FedTargetProj -data Cifar100 -m CNN-512 -did $PAT10_GPU_AVG -dev cuda --cnn512_control --control_eval_scope all --target_proj_mode avg --target_client_id 0 --seed 0 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 10 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 --data-root '$PAT10_DATA_ROOT' --data-subdir pat_10 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_pat10_avg_all20_seed0 -sfn '$PAT10_RUN_ROOT/avg/checkpoints' --run_log_dir '$PAT10_RUN_ROOT/avg/logs' --h5_result_root '$PAT10_RUN_ROOT/avg/h5' --final_model_root '$PAT10_RUN_ROOT/avg/final_models'"
"$PAT10_PYTHON main.py -algo FedFewS -data Cifar100 -m CNN-512 -did $PAT10_GPU_FEW -dev cuda --cnn512_control --control_eval_scope all --target_client_id 0 --seed 0 -nsm 3 -smu 0.01 --fedfews_rng controlled --fedfews_control_seed 0 --fedfews_torch_threads 4 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 10 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 --data-root '$PAT10_DATA_ROOT' --data-subdir pat_10 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_pat10_fedfews_all20_K3_seed0 -sfn '$PAT10_RUN_ROOT/fedfews/checkpoints' --run_log_dir '$PAT10_RUN_ROOT/fedfews/logs' --h5_result_root '$PAT10_RUN_ROOT/fedfews/h5' --final_model_root '$PAT10_RUN_ROOT/fedfews/final_models'"
)

declare -a PREVIEW_COMMANDS=(
"${COMMANDS[0]} --control_preview"
"${COMMANDS[1]} --control_preview"
)
