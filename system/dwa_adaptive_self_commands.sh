# Array entries for system/run_now.sh; defining this array does not start training.
# Run from system/, and choose the GPU IDs before adding these entries to a runner.
declare -a COMMANDS=(
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -did 0 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 --target_client_id 0 --seed 0 --target_proj_mode dwa_adaptive_self --dwa_distance_eps 1e-12 -exp_name target0_seed0_dwa_adaptive_self"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -did 1 -algo FedTargetProj -is_regular 1 -regular_lamda 1e-3 -niid 1 -pt pat -cpc 20 --target_client_id 0 --seed 0 --target_proj_mode dwa_adaptive_self_projection --dwa_distance_eps 1e-12 -exp_name target0_seed0_dwa_adaptive_self_projection"
)
