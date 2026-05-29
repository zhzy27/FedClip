#!/bin/bash

# 1. 设定定时等待时间：8小时 = 28800 秒
WAIT_TIME=28800
echo "🕒 脚本已启动，当前时间: $(date)。将在 8 小时后开始并行执行任务..."
sleep $WAIT_TIME

# 2. 获取启动时的时间戳
START_TIME=$(date +"%Y%m%d_%H%M%S")
echo "🚀 8小时等待结束，开始执行任务。启动时间: $START_TIME"

# 3. 定义你的 12 个训练命令
declare -a COMMANDS=(
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 4 -algo FedGH -slr 0.005 -se 5 -niid 1 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 4 -algo LG-FedAvg -niid 1 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 4 -algo FedTGP -lam 10.0 -se 20 -mart 100 -fd 512 -niid 1 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 5 -algo FedProto -lam 10 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 5 -algo FML -al 0.5 -bt 0.5 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 5 -algo FD -lam 1 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -did 7 -algo FedKD -mlr 0.005 -Ts 0.95 -Te 0.98 -fd 512 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -fd 512 -did 6 -algo FedGen -nd 32 -glr 0.05 -hd 512 -se 20 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-tiny -fd 512 -did 6 -algo FedMRL -sfd 128 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m CNN-5-512-AFM-tiny -did 6 -algo PFedAFM -alpha_lr 0.01 -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m SPU_CNN1-tiny -did 7 -algo FedSPU -pt dir -dir_alpha 0.3"
    "python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 200 -data TinyImagenet -m Decom_CNN-5-512 -did 7 -algo FedCLIP -is_regular 1 -mse_lamda 1 -Cos_lamda 0.0 -regular_lamda 1e-3 -niid 1 -pt dir -dir_alpha 0.3 -v_mse_lamda 0 -aggregate_tau 1"
)

# 4. 循环解析参数，创建文件夹并执行
for CMD in "${COMMANDS[@]}"; do
    # 使用 awk 自动提取命令中的核心参数
    ALGO=$(echo "$CMD" | awk -F'-algo ' '{print $2}' | awk '{print $1}')
    DATA=$(echo "$CMD" | awk -F'-data ' '{print $2}' | awk '{print $1}')
    ALPHA=$(echo "$CMD" | awk -F'-dir_alpha ' '{print $2}' | awk '{print $1}')

    # 兜底默认值（防止极个别格式问题导致解析失败）
    ALGO=${ALGO:-UnknownAlgo}
    DATA=${DATA:-UnknownData}
    ALPHA=${ALPHA:-UnknownAlpha}

    # 构造输出文件夹路径：时间_算法_数据集_异构度
    FOLDER_NAME="${START_TIME}_${ALGO}_${DATA}_dir${ALPHA}"
    mkdir -p "$FOLDER_NAME"

    # 5. 后台并行执行（&），并将标准输出与错误合并重定向（2>&1）
    nohup $CMD > "${FOLDER_NAME}/train.log" 2>&1 &

    echo "✅ 已启动: $ALGO | 日志: $FOLDER_NAME/train.log | PID: $!"
done

echo "------------------------------------------------------------"
echo "🎉 12 个任务均已在后台并行启动！"
echo "主调度脚本将挂起，等待所有子任务完成..."
wait
echo "✅ 所有任务执行完毕！"
