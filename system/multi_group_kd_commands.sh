#!/usr/bin/env bash
# From system/: defines commands only; never starts jobs.
# full publishes its unchanged first-round grouping; not_true waits/reuses it.
MULTI_KD_RUN_TAG="${MULTI_KD_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
MULTI_KD_DATA_ROOT="${MULTI_KD_DATA_ROOT:-../dataset}"
MULTI_KD_RUN_ROOT="${MULTI_KD_RUN_ROOT:-multi_kd_runs/$MULTI_KD_RUN_TAG}"
MULTI_KD_GROUP_FILE="${MULTI_KD_GROUP_FILE:-$MULTI_KD_RUN_ROOT/shared/K3_original_capacity_R1_full_weights.json}"
MULTI_KD_GROUP_WAIT="${MULTI_KD_GROUP_WAIT:-7200}"

declare -a COMMANDS=(
"python -u main.py -algo FedMultiGroupProj -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 1 -nc 20 -ncl 100 -nsm 3 --seed 0 -t 1 -pv 0 -jr 1.0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -regular_lamda 1e-3 --target_prox_mu 0 --target_head_l2_lambda 0 --target_head_mode shared --target_local_schedule joint --control_eval_scope all --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 -niid 1 -pt pat -cpc 20 --data-root '$MULTI_KD_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat --multi_svd --multi_svd_ratios 0.9,0.37,0.35,0.25,0.15 --multi_svd_init low_rank_recovered --multi_svd_init_rank_rate 0.15 --multi_svd_diagnostics 1 --multi_warmup_rounds 0 --multi_group_basis full_weights --multi_group_sizes 7,7,6 --multi_group_mode group_projection_softmax --multi_group_weight 0.50 --group_softmax_tau 0.20 --multi_cluster_seed 0 --multi_cluster_restarts 8 --multi_cluster_iterations 50 --multi_kd_mode full --multi_kd_lambda 0.1 --multi_kd_temperature 2 --multi_kd_teacher_batch_size 16 --multi_kd_observe_gap 1 --multi_kd_probe_samples 32 --multi_group_publish '$MULTI_KD_GROUP_FILE' -exp_name multi_svd_K3_projection_full_seed0_gr100 -sfn '$MULTI_KD_RUN_ROOT/K3_projection_full/checkpoints' --run_log_dir '$MULTI_KD_RUN_ROOT/K3_projection_full/logs' --h5_result_root '$MULTI_KD_RUN_ROOT/K3_projection_full/h5' --final_model_root '$MULTI_KD_RUN_ROOT/K3_projection_full/final_models'"
"python -u main.py -algo FedMultiGroupProj -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 2 -nc 20 -ncl 100 -nsm 3 --seed 0 -t 1 -pv 0 -jr 1.0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -regular_lamda 1e-3 --target_prox_mu 0 --target_head_l2_lambda 0 --target_head_mode shared --target_local_schedule joint --control_eval_scope all --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 -niid 1 -pt pat -cpc 20 --data-root '$MULTI_KD_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat --multi_svd --multi_svd_ratios 0.9,0.37,0.35,0.25,0.15 --multi_svd_init low_rank_recovered --multi_svd_init_rank_rate 0.15 --multi_svd_diagnostics 1 --multi_warmup_rounds 0 --multi_group_basis full_weights --multi_group_sizes 7,7,6 --multi_group_mode group_projection_softmax --multi_group_weight 0.50 --group_softmax_tau 0.20 --multi_cluster_seed 0 --multi_cluster_restarts 8 --multi_cluster_iterations 50 --multi_kd_mode not_true --multi_kd_lambda 0.1 --multi_kd_temperature 2 --multi_kd_teacher_batch_size 16 --multi_kd_observe_gap 1 --multi_kd_probe_samples 32 --multi_group_file '$MULTI_KD_GROUP_FILE' --multi_group_wait_seconds $MULTI_KD_GROUP_WAIT --multi_group_poll_seconds 1 -exp_name multi_svd_K3_projection_not_true_seed0_gr100 -sfn '$MULTI_KD_RUN_ROOT/K3_projection_not_true/checkpoints' --run_log_dir '$MULTI_KD_RUN_ROOT/K3_projection_not_true/logs' --h5_result_root '$MULTI_KD_RUN_ROOT/K3_projection_not_true/h5' --final_model_root '$MULTI_KD_RUN_ROOT/K3_projection_not_true/final_models'"
)

# Optional only: not added to COMMANDS and not automatically scheduled.
# Reuse an existing baseline only after checking its SAME first 101 rounds/protocol.
declare -a OPTIONAL_BASELINE_COMMANDS=(
"python -u main.py -algo FedMultiGroupProj -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 0 -nc 20 -ncl 100 -nsm 3 --seed 0 -t 1 -pv 0 -jr 1.0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -regular_lamda 1e-3 --target_prox_mu 0 --target_head_l2_lambda 0 --target_head_mode shared --target_local_schedule joint --control_eval_scope all --local_grad_clip 10 --control_test_batch_size 16 -clip_cpu_threads 4 -niid 1 -pt pat -cpc 20 --data-root '$MULTI_KD_DATA_ROOT' --data-subdir pat_20 --data-formats npz_only --data-expected-partition pat --multi_svd --multi_svd_ratios 0.9,0.37,0.35,0.25,0.15 --multi_svd_init low_rank_recovered --multi_svd_init_rank_rate 0.15 --multi_svd_diagnostics 1 --multi_warmup_rounds 0 --multi_group_basis full_weights --multi_group_sizes 7,7,6 --multi_group_mode group_projection_softmax --multi_group_weight 0.50 --group_softmax_tau 0.20 --multi_cluster_seed 0 --multi_cluster_restarts 8 --multi_cluster_iterations 50 --multi_kd_mode none --multi_kd_lambda 0 --multi_kd_temperature 2 --multi_kd_teacher_batch_size 16 --multi_kd_observe_gap 1 --multi_kd_probe_samples 0 --multi_group_file '$MULTI_KD_GROUP_FILE' --multi_group_wait_seconds $MULTI_KD_GROUP_WAIT --multi_group_poll_seconds 1 -exp_name multi_svd_K3_projection_none_seed0_gr100 -sfn '$MULTI_KD_RUN_ROOT/K3_projection_none/checkpoints' --run_log_dir '$MULTI_KD_RUN_ROOT/K3_projection_none/logs' --h5_result_root '$MULTI_KD_RUN_ROOT/K3_projection_none/h5' --final_model_root '$MULTI_KD_RUN_ROOT/K3_projection_none/final_models'"
)

# To use an already validated ORIGINAL-capacity recovered-init grouping:
# export MULTI_KD_GROUP_FILE=/absolute/path/to/original/fixed_groups.json
# export MULTI_KD_REUSE_GROUP_FILE=1
# then source this file. All three commands now READ that same group artifact;
# first-round training is still unchanged and KD remains off there.
if [[ "${MULTI_KD_REUSE_GROUP_FILE:-0}" == "1" ]]; then
    MULTI_KD_OLD_GROUP_ARG="--multi_group_publish '$MULTI_KD_GROUP_FILE'"
    MULTI_KD_NEW_GROUP_ARG="--multi_group_file '$MULTI_KD_GROUP_FILE' --multi_group_wait_seconds $MULTI_KD_GROUP_WAIT --multi_group_poll_seconds 1"
    COMMANDS[0]="${COMMANDS[0]/"$MULTI_KD_OLD_GROUP_ARG"/"$MULTI_KD_NEW_GROUP_ARG"}"
fi
