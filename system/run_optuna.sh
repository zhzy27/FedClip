#!/bin/bash

echo "🚀 正在单线程初始化数据库，防止并发建表冲突..."
# 先用单进程把数据库和表结构建好！
python -c "import optuna; optuna.create_study(study_name='fedclip_tuning_db', storage='sqlite:///fedclip_tuning_db.db', load_if_exists=True)"

echo "✅ 数据库就绪，准备启动分布式阵列 (6卡 x 2进程) ..."

mkdir -p tune_logs
GPUS=(2 3 4 5 6 7)

for gpu_id in "${GPUS[@]}"
do
    echo ">> 正在启动显卡 $gpu_id 上的 Worker 1..."
    nohup python tune.py --gpu $gpu_id > tune_logs/optuna_gpu${gpu_id}_w1.log 2>&1 &
    
    # 错峰 10 秒，防止同时加载模型瞬间爆显存和挤爆硬盘IO
    sleep 10 
    
    echo ">> 正在启动显卡 $gpu_id 上的 Worker 2..."
    nohup python tune.py --gpu $gpu_id > tune_logs/optuna_gpu${gpu_id}_w2.log 2>&1 &
    sleep 10
done

echo "✅ 12 个打工人已平稳上线！"