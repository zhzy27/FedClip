# 单目标训练数据共享：双 batch 诊断

仅给旧同步 `FedTargetProj / projection_softmax / Decom_CNN-5-512` 增加一个可关闭的训练分支。没有改变初始化、容量分配、SVD、上传恢复、global-delta、投影、投影前余弦或 softmax 计算。不是蒸馏，也不复制目标客户端模型。

## 参数及范围

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--target_c0_replay_mode` | `none` | `none` / `batch_pair` |
| `--target_c0_replay_weight` | `.5` | 有限 `[0,1]`，两路平均 CE 中目标训练数据的质量 |
| `--target_c0_replay_diagnostics` | `1` | `0/1`，关闭逐客户端明细打印及序列化；训练统计和累计预算仍计算 |

数据源为实际 `--target_client_id`，并非固定 client 0。`main.py` 和 `run_target_proj.py` 均支持参数；启动器也支持 `--target_client_id`（原 20-client 启动器的其他公共配置不变）。

非零有效开关只支持共享分类头、joint、全员参与、普通完整训练集，拒绝非零 Prox / head L2 / Z2、FedRep、反馈、DWA、群组、meta、完整 CNN control、断点/预训练、slow/drop/few-shot/early-stop 等未验收组合。`none` 或 `batch_pair + weight=0` 不施加这些新组合限制，也不建立源 loader，不额外读取目标分片、不前向。

## 每步语义

helper 原有 loader 决定 epoch 和优化步数。每步分别用自己的当前低秩模型计算自身 batch 和目标训练 batch 的平均 CE：

`loss = (1-alpha)*CE_own + alpha*CE_target + original_weighted_F`

末 batch 大小不同也保持配置的两路系数。原 `_objective()` 只在自身 batch 调用一次，F 只计算一次；目标 CE 直接调用原交叉熵。两次前向、一次联合反向、一次原梯度裁剪、一次原无动量 SGD 更新。目标客户端仍走原一批 CE＋F 路径，有效 replay 权重与额外 batch/访问次数均为 0。

各 helper、各轮新建自己的目标训练数据流。种子为 `SHA256(target-replay-v1:seed:loop_round:helper_id:target_id)` 的前 8 字节小端整数模 `2**63`。独立 Torch Generator 同时用于 DataLoader base seed 与 shuffle sampler；耗尽只重建迭代器，继续该私有随机流。没有全局共享的目标迭代器。源数据调用普通客户端 reader 的 `is_train=True`，使用原预处理，workers=0、drop_last=False、不增广。额外分片读取与 loader 构造保护 Python/NumPy/Torch CPU/CUDA 随机状态。

## 预算及损失统计

- 保持原 SGD 步数，增加 helper 的数据访问与计算；每 helper 每步多一个目标训练 batch 和一次前向，两路都参与一次联合反向。
- `train_samples` 始终是原分片样本量。样本权重和 ProjectionSoftmax 聚合质量不会因额外访问翻倍。目标自身质量由原 `projection_self_weight` 决定。
- `own_sample_visits` 与 `c0_sample_visits` 分开累积实际 batch 大小，`own_batches`、`c0_batches`、`optimizer_steps` 分开记录。源数据流跨本地 epoch 连续读取，耗尽重启；不能把目标访问量直接假定为自身访问量。
- `train_mean_ce_own` / `train_mean_ce_c0` 各按该来源的实际样本访问数加权。`train_mean_weighted_frobenius` 和 `train_mean_total_loss` 是实际 SGD 目标的逐步算术平均。末 batch 不等大时，不能用两列样本加权 CE 均值重新拼出逐步平均总损失。
- 过程均值不是训练结束后的 endpoint CE。原 `train_metrics`、测试和旧 normal-train CE 仍使用自身数据定义，不混入 replay 数据。
- `actual_budget` 汇总真实训练调用、两路 batch/访问量、SGD 步数、普通训练耗时，不预设每轮步数或访问数。额外数据共享不能宣称与旧实验同计算预算。

## 日志与模型

复制 [启动数组](../system/target_c0_replay_commands.sh)，在服务器 `system/` 中交给原调度器。该文件只定义数组，不启动训练。完整配置为 seed0、pat20、20 客户端、原五档 SVD 容量、local5、lr=.005、batch16、F=.001、clip10、gr100（101 次）、self=.125、tau=.20、alpha=.5。

所有根目录位于独立 `system/target_c0_replay_runs/pat20_seed0_pair0p5_gr100/`：

- `logs/` 下为实际日志目录（程序追加模式、目标 ID、replay 系数、运行 ID；显式零 Prox 观察还保留其原有命名）。保存 `target_c0_replay_clients.csv/json`、持续 JSONL、`target_c0_replay_primary.csv/json`、`target_c0_replay_summary.json`，以及旧聚合权重和投影诊断。
- `checkpoints/` 为普通 post-local 低秩模型，不混进可下载的日志 ZIP。
- `h5/` 的 `target_projection/target_c0_replay` 保存配置、实际预算、客户端/主指标序列和 summary。该新实验的根 `rs_test_acc` 为普通目标 post-local；旧 evaluate 曲线另存 `legacy_server_diagnostic_test_acc`，不改旧模式根字段。
- `final_models/` 保存实际服务器模型、客户端模型及 manifest，记录源数据路径、文件 SHA256、容量、初始化指纹、优化配置、预算与 C0 主指标汇总。

主指标为目标客户端普通 local5 后准确率；final、同轮序列的最早 best、last10、last20，内部用 fraction。聚合后按容量下载的准确率另外命名为 `target_aggregated_download_acc`，不替代主指标。未增加额外测试遍历。

## 验收入口

仓库根目录运行：

```powershell
D:/miniconda/envs/htfllib/python.exe -m unittest discover -s tests -p test_target_c0_replay.py -v
D:/miniconda/envs/htfllib/python.exe -m unittest discover -s tests -p test_target_c0_replay_runtime.py -v
```

独立真实入口合成 smoke（从 `system/`；输出目录必须尚不存在，Windows 建议短路径）：

```bash
python -u verify_target_c0_replay.py --output /tmp/fedclip_replay_acceptance_20261030 --jobs 2
```

只在该新目录创建合成 NPZ，不触碰现有数据。使用真实 `main.py`、原宽度异构 CNN、20 客户端、两轮、每次 local5；比较诊断开/关、原聚合核、上传 checkpoint、RNG、各格式导出和 21 个模型重载。默认验证后删除该目录内自己的 checkpoint/最终模型，保留报告；`--keep-models` 可保留模型。CPU 合成验收不代表真实 CIFAR-100 准确率改善。
