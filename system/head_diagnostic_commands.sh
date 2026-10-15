# From system/. Define the first-batch COMMANDS only; this starts no training.
# -gr 30 is 31 ordinary local/aggregation rounds. Snapshots R10/R20/R30 are
# post-local C0 endpoints BEFORE aggregation, not loop_round 10/20/30.
declare -a COMMANDS=(
    "python -u main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 30 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 0 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 -eg 1 --target_client_id 0 --seed 0 --target_proj_mode avg --target_head_mode shared --head_diag_collect --head_diag_rounds 10,20,30 --head_diag_dir head_diag_snapshots/avg_seed0_gr30 -exp_name head_diag_avg_shared_seed0_gr30 -sfn temp/head_diag_avg_shared --h5_result_root h5_results/head_diag_avg_shared --final-model-root final_models/head_diag_avg_shared --run_log_dir training_logs/head_diag_avg_shared_seed0_gr30"
    "python -u main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 30 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 1 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 -eg 1 --target_client_id 0 --seed 0 --target_proj_mode avg --target_head_mode local --head_diag_collect --head_diag_rounds 10,20,30 --head_diag_dir head_diag_snapshots/avg_seed0_gr30 -exp_name head_diag_avg_local_head_seed0_gr30 -sfn temp/head_diag_avg_local_head --h5_result_root h5_results/head_diag_avg_local_head --final-model-root final_models/head_diag_avg_local_head --run_log_dir training_logs/head_diag_avg_local_head_seed0_gr30"
)
# After BOTH Avg runs complete, run separately:
# python head_diagnostic.py --snapshots head_diag_snapshots/avg_seed0_gr30 --dataset-root ../dataset --output-dir head_diag_results/avg_seed0_gr30 --device cuda --device-id 0 --shuffle-seed 1729
