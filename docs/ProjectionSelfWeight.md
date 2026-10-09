# ProjectionSoftmax 固定自身质量消融

新增可选 `--projection_self_weight s`，默认 None。
不传参数时按原 sample-count weight 设置 target mass，旧数值与原日志字段保持一致。
显式传入时只对 `projection_softmax` 生效：C0 固定 s，helpers 总质量 1-s，s 必须有限且在 [0,1]。
其他 mode/algorithm 显式传入即报错；本批完整训练集消融也不允许与 `--meta_c0_split` 组合。
这与 APA 的 `--apa_self_weight` 无关。

每轮仍以投影前普通 global-delta 的 cosine，按原稳定 Softmax/T=.2 计算 helper 相对概率 q。
不乘 sample-count scores，不改变投影参考、coefficients、epsilon、参数范围或求和顺序。
实际 alpha 为 `alpha_C0=s`、`alpha_helper=(1-s)*q`；q 每轮重新计算，不是跨轮固定。
原 `aggregate_target_updates(..., 'projection')` 保持未修改，只有输入聚合质量发生变化。

共享服务器模型、各容量 SVD 下发、普通 5-epoch SGD、Frobenius、梯度裁剪、head 和 buffers 处理均未改变。
主指标仍为本轮聚合前的普通 C0 `target_post_local_acc`；聚合后准确率保留。
分析 `local_t → aggregate_t → local_(t+1)`，本行权重影响下一轮初始化，不能把同一行差值当作 local FT 收益。

## 配置和结果文件

用户提供的完整训练集、seed0 历史基线为 final 45.20%、last10 44.03%；本地没有重新跑正式实验验证。

| 自身质量 | helpers 质量 | 顺序/用途 |
|---:|---:|---|
| .10 | .90 | 优先 |
| .20 | .80 | 优先 |
| .05 | .95 | 显式兼容入口；数据/配置一致时可复用旧结果 |
| .50 | .50 | 补充观察 |

`system/projection_self_weight_commands.sh` 提供完整 `declare -a COMMANDS=(...)` 数组，
顺序为 .10/.20/.05/.50，GPU ID 为 0/1/2/3 示例，按机器调整；只声明数组，不执行训练。
可把条目加入现有 `system/run_now.sh`，该 runner 未修改。
每条 main 命令有独立 `-sfn temp/projection_softmax_self0pXX` 与 `-exp_name target0_seed0_projection_softmax_self0pXX`。
H5 与 final-model 根目录也按该质量标签分开。
显式消融的 main 保留提供的名称，并确保包含质量标签；其他入口的命名行为不变。

统一 launcher 的单卡优先命令（在 system/，用户自行启动）：

```bash
python run_target_proj.py --modes projection_softmax --projection_self_weight 0.10 --rounds 100 --device-id 0
python run_target_proj.py --modes projection_softmax --projection_self_weight 0.20 --rounds 100 --device-id 0
python run_target_proj.py --modes projection_softmax --projection_self_weight 0.05 --rounds 100 --device-id 0
python run_target_proj.py --modes projection_softmax --projection_self_weight 0.50 --rounds 100 --device-id 0
```

末尾加 `--dry-run` 只显示命令。默认旧 modes 不变，传入新参数必须明确选择 `--modes projection_softmax`。
公共配置来自同一 launcher：Cifar100/pat_20、20 clients/full participation、C0、seed0、
Decom_CNN-5-512、SGD .005、batch16、local5、正则 .001、-gr100（实际 101 次正常训练/聚合）。
各调用使用独立时间戳和 `projection_softmax_self0pXX` 目录。

仍保存公共 target_proj CSV/JSON/H5 和 projection_softmax metrics/clients/matrices 文件。
新配置在 metrics/JSON/H5 中记录 `projection_self_weight`、实际 `target_weight`、
实际 `helper_total_weight`、`effective_all_client_count` 与原 `effective_helper_count`、helper min/max。
逐客户端仍记录实际 aggregation weight、sample weight、原 cosine、raw Softmax score、conflict、coeff、删除/投影范数。
原 `.05` 未传参数时不增加此批权重诊断字段。

`s=1` 时 helpers 实际权重全零，helper effective count/min/max 为零，全部有效数量为 1。
仍计算原余弦和投影几何诊断，因此未按质量加权的 removed_update_ratio 可能非零；
它不是这轮实际 helper 贡献。`s=0` 时 C0 实际贡献为零，仍作为普通 Projection 的参考方向。

显式消融每轮和最终 summary 增加 `last10_target_local_acc` / `last10_target_local_count`，
与 final、best、1-based 最早最佳 round 一同保存/打印。短 smoke 少于十次时记录已有次数。
正式 inclusive run 最后十次对应显示 R92–R101。
训练日志见 launcher 打印的 train.log；逐轮文件路径见 `metrics_dir`，最终导出复制同一份完整指标。

## 验证

新增 13 项测试：8 项数学/训练循环/数组检查、2 项 launcher、3 项实际异构 CNN 两轮 smoke。
全仓库 26 个模块独立运行，共 270 项通过。覆盖显式 .05 逐位一致、.10/.20/.50 质量约束、
helper 相对比例/原 score/投影 coefficients 不变、解析结果、0/1 和非法参数、full-data 协议、
共享下发、普通训练、准确率时序、last10、CSV/JSON/H5/模型导出。
与 `9772343` 做 48 组随机 20-client float32/64 对照：旧 projection_softmax/projection_relu/softmax_only
无参返回和显式 .05 的参数、原诊断字段逐位一致。
原客户端、本地优化器和基础 Projection 核未修改；旧模式测试继续通过。
CLI/语法/diff/dry-run 检查通过，原 staged 绘图修改保留。
本地没有 Cifar100/pat_20，只执行合成数据 smoke；未推送，未启动正式长实验或报告新准确率收益。
