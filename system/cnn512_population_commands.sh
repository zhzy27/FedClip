#!/usr/bin/env bash
# Three population comparisons ONLY; definitions do not start training.
# Run from FedClip/system. The older cnn512_control_commands.sh stays unchanged.
POP_PYTHON=${POP_PYTHON:-python}
POP_DATA_ROOT=${POP_DATA_ROOT:-$(cd ../dataset && pwd)}
POP_RUN_TAG=${POP_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
POP_RUN_ROOT=${POP_RUN_ROOT:-../fedfews_native_runs/cnn512_population/$POP_RUN_TAG}
POP_GPU_AVG=${POP_GPU_AVG:-0}
POP_GPU_GROUP=${POP_GPU_GROUP:-1}
POP_GPU_FEW=${POP_GPU_FEW:-2}

# Avg/FedFewS: all 20 test-sample-weighted post-local clients.
# GroupProjection: seven test-sample-weighted post-local clients, beta=.50.
# These populations are DIFFERENT; group is a stage-wise proxy, not the full method.
# All: gr100=101 train/aggregate steps. Budget/client: 5 vs 5 vs K3*5=15 epochs.
declare -a COMMANDS=(
"$POP_PYTHON main.py -algo FedTargetProj -data Cifar100 -m CNN-512 -did $POP_GPU_AVG -dev cuda --cnn512_control --control_eval_scope all --target_proj_mode avg --target_client_id 0 --seed 0 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 --data-root '$POP_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_population_avg_all20_seed0 -sfn '$POP_RUN_ROOT/avg_all20/checkpoints' --run_log_dir '$POP_RUN_ROOT/avg_all20/logs' --h5_result_root '$POP_RUN_ROOT/avg_all20/h5' --final_model_root '$POP_RUN_ROOT/avg_all20/final_models'"
"$POP_PYTHON main.py -algo FedTargetProj -data Cifar100 -m CNN-512 -did $POP_GPU_GROUP -dev cuda --cnn512_control --control_eval_scope group --target_proj_mode group_projection_softmax --target_group_ids 0,1,2,3,4,6,7 --target_group_weight 0.50 --seed 0 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 --data-root '$POP_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_population_group7_beta0p50_seed0 -sfn '$POP_RUN_ROOT/group7_beta0p50/checkpoints' --run_log_dir '$POP_RUN_ROOT/group7_beta0p50/logs' --h5_result_root '$POP_RUN_ROOT/group7_beta0p50/h5' --final_model_root '$POP_RUN_ROOT/group7_beta0p50/final_models'"
"$POP_PYTHON main.py -algo FedFewS -data Cifar100 -m CNN-512 -did $POP_GPU_FEW -dev cuda --cnn512_control --control_eval_scope all --seed 0 -nsm 3 -smu 0.01 --fedfews_rng controlled --fedfews_control_seed 0 --fedfews_torch_threads 4 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 --data-root '$POP_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_population_fedfews_all20_K3_seed0 -sfn '$POP_RUN_ROOT/fedfews_all20/checkpoints' --run_log_dir '$POP_RUN_ROOT/fedfews_all20/logs' --h5_result_root '$POP_RUN_ROOT/fedfews_all20/h5' --final_model_root '$POP_RUN_ROOT/fedfews_all20/final_models'"
)
