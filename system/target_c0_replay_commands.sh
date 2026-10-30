#!/usr/bin/env bash
# Run from system/. Defines one command only; does not start training.
# Original SGD step count, but each helper performs two forwards and extra target TRAIN visits.
declare -a COMMANDS=(
  "python -u main.py -algo FedTargetProj -data Cifar100 -m Decom_CNN-5-512 -dev cuda -did 0 -t 1 -pv 0 -nc 20 -ncl 100 -jr 1.0 --target_client_id 0 --seed 0 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -niid 1 -pt pat -cpc 20 -is_regular 1 -regular_lamda 1e-3 --target_head_mode shared --target_local_schedule joint --target_prox_mu 0 --target_head_l2_lambda 0 --target_z2_lambda 0 --local_grad_clip 10 --target_proj_mode projection_softmax --projection_self_weight 0.125 --projection_softmax_tau 0.20 --target_c0_replay_mode batch_pair --target_c0_replay_weight 0.5 --target_c0_replay_diagnostics 1 -exp_name target0_seed0_projection_softmax_self0p125_replay_pair0p5_gr100 -sfn target_c0_replay_runs/pat20_seed0_pair0p5_gr100/checkpoints --run_log_dir target_c0_replay_runs/pat20_seed0_pair0p5_gr100/logs --h5_result_root target_c0_replay_runs/pat20_seed0_pair0p5_gr100/h5 --final-model-root target_c0_replay_runs/pat20_seed0_pair0p5_gr100/final_models"
)
