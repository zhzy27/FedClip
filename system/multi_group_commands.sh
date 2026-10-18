#!/usr/bin/env bash
# Definitions only. Run from FedClip/system. Nothing is launched by sourcing.
MULTI_PYTHON=${MULTI_PYTHON:-python}
MULTI_DATA_ROOT=${MULTI_DATA_ROOT:-$(cd ../dataset && pwd)}
MULTI_RUN_TAG=${MULTI_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
MULTI_RUN_ROOT=${MULTI_RUN_ROOT:-../fedfews_native_runs/multi_group/$MULTI_RUN_TAG}
MULTI_GPU_PROJ=${MULTI_GPU_PROJ:-0}
MULTI_GPU_AVG=${MULTI_GPU_AVG:-1}
# For strict membership reuse, source this file again with MULTI_FIXED_GROUPS set
# to the first run's logs/run_<run_id>/fixed_groups.json. First-round training is
# identical in both modes; without a file verify equal group_fingerprint before
# treating automatic groups as the same comparison (especially across GPUs).
MULTI_FIXED_GROUPS=${MULTI_FIXED_GROUPS:-}
MULTI_REUSE_ARG=""
if [[ -n "$MULTI_FIXED_GROUPS" ]]; then
    MULTI_REUSE_ARG="--multi_group_file '$MULTI_FIXED_GROUPS'"
fi

declare -a COMMANDS=(
"$MULTI_PYTHON main.py -algo FedMultiGroupProj -data Cifar100 -m CNN-512 -did $MULTI_GPU_PROJ -dev cuda -nc 20 -ncl 100 -nsm 3 --seed 0 -t 1 -pv 0 -jr 1.0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 0 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 -niid 1 -pt pat -cpc 20 --data-root '$MULTI_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat --multi_group_mode group_projection_softmax --multi_group_weight 0.50 --group_softmax_tau 0.20 --multi_cluster_seed 0 --multi_cluster_restarts 8 --multi_cluster_iterations 50 $MULTI_REUSE_ARG -exp_name multigroup_cnn512_pat20_K3_projection_beta050_seed0 -sfn '$MULTI_RUN_ROOT/projection/checkpoints' --run_log_dir '$MULTI_RUN_ROOT/projection/logs' --h5_result_root '$MULTI_RUN_ROOT/projection/h5' --final_model_root '$MULTI_RUN_ROOT/projection/final_models'"
"$MULTI_PYTHON main.py -algo FedMultiGroupProj -data Cifar100 -m CNN-512 -did $MULTI_GPU_AVG -dev cuda -nc 20 -ncl 100 -nsm 3 --seed 0 -t 1 -pv 0 -jr 1.0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 0 --target_prox_mu 0 --target_head_mode shared --target_local_schedule joint --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 -niid 1 -pt pat -cpc 20 --data-root '$MULTI_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat --multi_group_mode group_avg_only --multi_group_weight 0.50 --group_softmax_tau 0.20 --multi_cluster_seed 0 --multi_cluster_restarts 8 --multi_cluster_iterations 50 $MULTI_REUSE_ARG -exp_name multigroup_cnn512_pat20_K3_avg_only_seed0 -sfn '$MULTI_RUN_ROOT/avg_only/checkpoints' --run_log_dir '$MULTI_RUN_ROOT/avg_only/logs' --h5_result_root '$MULTI_RUN_ROOT/avg_only/h5' --final_model_root '$MULTI_RUN_ROOT/avg_only/final_models'"
)

declare -a PREVIEW_COMMANDS=(
"${COMMANDS[0]} --multi_preview"
"${COMMANDS[1]} --multi_preview"
)
