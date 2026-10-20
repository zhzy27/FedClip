# Avg 本地分类头专项 L2：31 轮消融

本次只增加显式的 `lambda_h / 2 * sum(fc3.weight**2)`。全部客户端沿用同步 Avg、持久化本地分类头及 joint 训练；`fc3.bias` 不加此项，`fc1/fc2` 仍为共享骨干。分类头以参数身份检查别名，只计一次。原 Frobenius 正则、全部可训练参数的总梯度裁剪及 SGD（weight_decay=0）保持不变。

## 参数与边界

- `--target_head_l2_lambda`：默认 `0.0`，有限且非负。
- 非零仅支持 `FedTargetProj / avg / local / joint / Decom_CNN-5-512`，且 `target_prox_mu=0`。原有本地头入口的 C0、100 类、全参与、完整数据及每轮评价限制继续生效。
- `--target_head_l2_diagnostics 1` 为默认专项观察；设 `0` 仅关闭额外 endpoint 评价和专项导出，**不关闭 L2 损失**，用于核验观察隔离。
- 不传或显式传零时，不计算头 L2、头范数或额外 endpoint，也不建立专项结果文件；仍直接执行原 `(CE + F).backward()`。`_objective()` 继续仅返回 CE 和加权 F，原 `train_metrics()` 损失含义不变。
- 第一次分类头从公共服务器初始化，随后保留各自训练后的头。无重新初始化、额外本地 epoch、FedRep、反馈、数据留出或 Prox。

## 启动

完整三条 `declare -a COMMANDS` 在 [target_head_l2_commands.sh](../system/target_head_l2_commands.sh)，从服务器 `system/` 使用：

```bash
source target_head_l2_commands.sh
printf '%s\n' "${COMMANDS[@]}"   # 仅查看，交给既有调度器执行
```

三条分别为 `1e-4 / 1e-3 / 1e-2`，GPU 0/1/2 可替换。均明确指定：Cifar100/pat_20、20 人/100 类、原 Decom_CNN-5-512、C0/seed0、lr .005、batch16、joint5、F .001、Prox0、`-gr 30` 实际31轮、每轮评价。数据仍由原 reader 读取已有分片，不生成或改写真实数据。

启动器同样支持新参数，例如仅预览：

```bash
python run_target_proj.py --modes avg --target_head_mode local --target_local_schedule joint --target_head_l2_lambda 1e-3 --rounds 30 --device-id 0 --dry-run
```

三组复用 `10-15-03.zip` 中的 Avg local joint5 无专项 L2、Avg shared joint5 作为历史对照；本次没有重跑它们，也没有根据测试结果选择系数。

## 输出与字段

系数标签分别为 `head_l2_0p0001 / head_l2_0p001 / head_l2_0p01`。实验名和 checkpoint、日志、H5、最终模型根目录均包含 `avg_local_head` 与对应标签。真实运行 ID 继续隔离同配置的多次运行。

- 日志：`training_logs/avg_local_head_head_l2_<标签>_seed0_gr30/`；控制台在 `train.log`，逐次运行的指标在 `run_<run_id>/`。下载同目录生成的日志 ZIP 即可，无 `.pt` 模型。
- checkpoint：`temp/avg_local_head_head_l2_<标签>_gr30/.../<run_id>/`。
- H5：`h5_results/avg_local_head_head_l2_<标签>_gr30/.../*.h5`；日志目录还有镜像。
- 最终模型：`final_models/avg_local_head_head_l2_<标签>_gr30/.../`，仍导出骨干专用服务器文件、各客户端普通 post-local 模型、独立的 C0 聚合骨干＋本地头评价模型及 manifest。

专项文件：

| 文件 | 内容 |
| --- | --- |
| `target_head_l2_clients.csv/json` | 每轮每客户端系数、训练期间损失均值、结束头 weight 范数、epoch/batch/样本访问量、训练耗时 |
| `target_head_l2_target.csv/json` | 每轮 C0 普通 post-local 同一模型的 train/test CE、correct、samples、accuracy、头 weight 范数 |
| H5 `target_projection/target_head_l2/{clients,target}` | 上述数值列；`policy_json` 记录系数、参数范围、客户端 IDs、单位及语义 |
| 最终 `manifest.json` 中 `target_head_l2` | 作用范围、系数、最后一轮训练统计、C0 endpoint 序列及专项文件名 |

`train_mean_ce / train_mean_weighted_frobenius / train_mean_weighted_head_l2 / train_mean_total_loss` 均按**实际样本访问次数**加权（包括不足一个 batch 的尾批）。F 和头 L2 已包含各自系数，不能再次乘系数。训练过程均值与 `endpoint_*_ce` 不同；后者仅为训练结束模型的原始 CE，不加任何正则。

`head_weight_norm` 只测 `fc3.weight`。endpoint 在私有模型副本上、`eval` 和 `no_grad` 下执行；包括加载、loader 迭代在内的额外观察恢复 Python/NumPy/Torch CPU/CUDA 随机状态，不写 checkpoint、不恢复或分解模型。模型、副本 buffers 和模块模式均不影响正常训练。

主结果仍读原 `target_proj_metrics.json` 的 C0 普通 post-local 序列：final、同轮序列 best（最早轮次）、last10、last20。专项 target JSON 的 `primary_summary` 在结束时与之相同；JSON/H5 准确率沿用 fraction（显示百分比时乘100），完成轮次为 `round=loop_round+1`。专项诊断准确率不替代主结果。

## 本地验收

详见 [target_head_l2_acceptance.md](target_head_l2_acceptance.md)。测试只在临时目录生成合成数据，结束后清理，不对真实 CIFAR-100 准确率作结论。

从仓库根目录执行（使用已有环境，不安装依赖）：

```bash
python -m unittest discover -s tests -p 'test_target_head_l2*.py' -v
python -m unittest discover -s tests -p test_target_local_head.py -v
python -m unittest discover -s tests -p test_target_prox.py -v
python -m unittest discover -s tests -p test_projection_weighting.py -v
```
