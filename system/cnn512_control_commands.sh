#!/usr/bin/env bash
# Definitions only. Run from FedClip/system with your existing scheduler.
CONTROL_PYTHON=${CONTROL_PYTHON:-python}
CONTROL_DATA_ROOT=${CONTROL_DATA_ROOT:-$(cd ../dataset && pwd)}
CONTROL_RUN_TAG=${CONTROL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
CONTROL_RUN_ROOT=${CONTROL_RUN_ROOT:-../fedfews_native_runs/cnn512_control/$CONTROL_RUN_TAG}
CONTROL_GPU_AVG=${CONTROL_GPU_AVG:-0}
CONTROL_GPU_PROJ=${CONTROL_GPU_PROJ:-1}
CONTROL_GPU_FEW=${CONTROL_GPU_FEW:-2}

# Common NPZ pat_20, full shared CNN-512, CE only, SGD .005, norm cap10.
# -gr100 means 101 train/aggregate steps. No command below is executed here.
# Data epochs/client/round: Avg/Projection=5, FedFewS=3 models*5=15.
declare -a COMMANDS=(
"$CONTROL_PYTHON main.py -algo FedTargetProj -data Cifar100 -m CNN-512 -did $CONTROL_GPU_AVG -dev cuda --cnn512_control --target_proj_mode avg --target_client_id 0 --seed 0 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 --data-root '$CONTROL_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_control_avg_seed0 -sfn '$CONTROL_RUN_ROOT/avg/checkpoints' --run_log_dir '$CONTROL_RUN_ROOT/avg/logs' --h5_result_root '$CONTROL_RUN_ROOT/avg/h5' --final_model_root '$CONTROL_RUN_ROOT/avg/final_models'"
"$CONTROL_PYTHON main.py -algo FedTargetProj -data Cifar100 -m CNN-512 -did $CONTROL_GPU_PROJ -dev cuda --cnn512_control --target_proj_mode projection_softmax --target_client_id 0 --projection_self_weight 0.125 --seed 0 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 --data-root '$CONTROL_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_control_projection_self0p125_seed0 -sfn '$CONTROL_RUN_ROOT/projection/checkpoints' --run_log_dir '$CONTROL_RUN_ROOT/projection/logs' --h5_result_root '$CONTROL_RUN_ROOT/projection/h5' --final_model_root '$CONTROL_RUN_ROOT/projection/final_models'"
"$CONTROL_PYTHON main.py -algo FedFewS -data Cifar100 -m CNN-512 -did $CONTROL_GPU_FEW -dev cuda --cnn512_control --seed 0 -nsm 3 -smu 0.01 --fedfews_rng controlled --fedfews_control_seed 0 --fedfews_torch_threads 4 -t 1 -pv 0 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -eg 1 -is_regular 0 -niid 1 -pt pat -cpc 20 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 --data-root '$CONTROL_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat -exp_name cnn512_control_fedfews_K3_seed0 -sfn '$CONTROL_RUN_ROOT/fedfews/checkpoints' --run_log_dir '$CONTROL_RUN_ROOT/fedfews/logs' --h5_result_root '$CONTROL_RUN_ROOT/fedfews/h5' --final_model_root '$CONTROL_RUN_ROOT/fedfews/final_models'"
)
