# From system/. Source to define COMMANDS only; no automatic training.
# Each of 20 clients: head CE5 -> backbone CE+F5, total 10 training data epochs.
# -ls 5 is BACKBONE epochs here; -gr 30 is 31 inclusive federated rounds.
# Reuse Avg shared/local joint5 in 10-15-03.zip; do not automatically rerun controls.
declare -a COMMANDS=(
    "python -u main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 30 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 0 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 -eg 1 --target_client_id 0 --seed 0 --target_proj_mode avg --target_head_mode local --target_local_schedule fedrep --fedrep_head_epochs 5 -exp_name target0_seed0_avg_local_head_fedrep_head5_backbone5_gr30 -sfn temp/avg_local_head_fedrep_head5_backbone5 --h5_result_root h5_results/avg_local_head_fedrep_head5_backbone5 --final-model-root final_models/avg_local_head_fedrep_head5_backbone5 --run_log_dir training_logs/avg_local_head_fedrep_head5_backbone5_seed0_gr30"
)
