# Source from system/ to define COMMANDS. Defining this array starts no training.
# Corrected helper-first flow: each client local5 ONCE per round, 201 rounds.
# Self retention is S_t anchor .125, not a current-round C0 upload weight.
# Feedback Projection is DEFERRED until its reference is defined. No warmup.
# GPU 0/1 are replaceable examples.
declare -a COMMANDS=(
    "python -u main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 200 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 0 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 -eg 1 --target_client_id 0 --seed 0 --target_proj_mode feedback_avg --target_prox_mu 0 -exp_name target0_seed0_feedback_avg_helpers_only_local5 -sfn temp/feedback_avg_helpers_only_local5 --h5_result_root h5_results/feedback_avg_helpers_only_local5 --final-model-root final_models/feedback_avg_helpers_only_local5 --run_log_dir training_logs/feedback_avg_helpers_only_local5_seed0_gr200"
    "python -u main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 200 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 1 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 -eg 1 --target_client_id 0 --seed 0 --target_proj_mode feedback_avg_self --feedback_anchor_weight 0.125 --target_prox_mu 0 -exp_name target0_seed0_feedback_avg_self_anchor0p125_local5 -sfn temp/feedback_avg_self_anchor0p125_local5 --h5_result_root h5_results/feedback_avg_self_anchor0p125_local5 --final-model-root final_models/feedback_avg_self_anchor0p125_local5 --run_log_dir training_logs/feedback_avg_self_anchor0p125_local5_seed0_gr200"
)
