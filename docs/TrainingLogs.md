# 独立训练日志目录

新训练默认将日志与模型分开，日志根目录为 system/training_logs/。
不用改现有训练参数或COMMANDS数组；模型checkpoint仍按-sfn放在原temp路径，
最终模型仍在final_models，旧实验目录不自动搬迁、删除或重写。

直接运行main.py时，默认结构为：

~~~text
training_logs/<时间_模式_seed_唯一编号>/
    train.log
    status.json
    run_<run_id>/
        run_paths.json
        target_proj_*.csv/json 或 target_group_*.csv/json
        layer/APA/DWA等模式的指标与诊断文件
        完整H5结果副本
        实验配置与总结JSON
training_logs/<时间_模式_seed_唯一编号>.zip
~~~

每次实验结束自动生成旁边的ZIP，下载这一份即可，不包含模型张量、
SVD缓存、APA basis或DWA guidance。运行过程中也可以直接查看train.log、
下载日志目录；ZIP是结束时的快照，异常信息在status.json中记录。
压缩失败会保留原日志目录，训练结果不会因压缩失败丢失。

使用run_target_proj.py时：

~~~text
training_logs/<启动时间>/<模式及群体或自身权重标签>/
    command.json
    train.log
    status.json
    run_<run_id>/指标、H5、总结
training_logs/<启动时间>/<模式及标签>.zip
target_proj_runs/<启动时间>/<模式及标签>/checkpoints/和final_models/
~~~

模式和多seed分别建目录。日志文件不会再写进模型目录，也不再复制进
final_models；模型manifest用training_log_dir指向对应日志。
H5原位置保留以兼容已有分析脚本，完整H5也会复制到日志目录，
单目标版本会在添加完target_projection数据后复制，不是未补全的早期副本。

从system/手动查看命令，不启动训练：

~~~bash
python run_target_proj.py --modes group_avg_all group_avg_only group_projection_softmax \
  --target_group_ids 0,1,2,3,4,6,7 --training_log_root training_logs --dry-run
~~~

可配置：

- --training_log_root /path/to/training_logs：日志根目录，默认training_logs。
- --run_log_dir /path/to/one_experiment_logs：指定这次进程的准确目录。
- 已有run_now.sh设置FEDCLIP_TRAIN_LOG_DIR时，指标会跟随该现有train.log目录；
  --run_log_dir优先于该环境变量。无显式目录或环境变量时使用新默认根目录。

main.py自动保存stdout/stderr到train.log，同时继续输出到终端。
启动器或shell已经重定向到同一个train.log时避免双写。
一个进程若使用-t多次实验，会在同一进程日志目录下按run_id隔离指标。
模型目录和日志目录应是两棵独立目录树，不允许把日志目录设成模型目录或其父子目录。

日志ZIP只收集日志、CSV、JSON、H5、文本和图，不收集.pt/.pth/.ckpt、
数据库或numpy张量。这次仅改文件保存/归档，训练、聚合、随机种子和评价规则未改。

## 已经完成的旧实验

旧文件不自动移动。可以单独复制和打包其中的日志，不带模型、不改原实验：

~~~bash
python collect_training_logs.py \
  --source "temp/group7/group_projection_softmax/Cifar100/FedTargetProj/<run_id>" \
  --output training_logs/old_group_projection_softmax \
  --extra /path/to/old/train.log /path/to/old/result.h5
~~~

--extra可省略，用于补充原来保存在别处的终端日志/H5；output须为新目录。
下载training_logs/old_group_projection_softmax.zip即可。
