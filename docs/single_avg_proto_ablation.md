# 单模型 Avg：原型 MSE、F 与裁剪机制消融

这批协议继续聚合完整模型参数，同时可选上传、合并和下发类别原型。它借鉴 FedProto 的最后 epoch 原型收集和客户端等权合并，不声称完整复现 FedProto。原联邦入口、K3/群组原型及其他基线默认行为不变；新入口不支持 Z2。

## 入口与八组

在 `FedMultiGroupProj` 中显式添加 `--multi_avg_proto_ablation`。只支持 K1、`group_avg_only`、全员参与、共享分类头、joint、同一 Hyper_CNN_512 架构。所有人固定使用模型 0；不计算分组相似度或聚类，不读取 K3 分组文件，也不使用 Projection、softmax 或目标客户端特殊权重。

| 编号 | 每轮训练模型 | F | 梯度裁剪 | 原型 MSE |
|---|---|---|---|---|
| 01 | 原五档低秩 | .001 | 10 | 关闭 |
| 02 | 原五档低秩 | .001 | 10 | 1 |
| 03 | 原五档低秩 | 关闭 | 10 | 关闭 |
| 04 | 原五档低秩 | 关闭 | 10 | 1 |
| 05 | 完整 Hyper_CNN_512 | 关闭 | 10 | 关闭 |
| 06 | 完整 Hyper_CNN_512 | 关闭 | 10 | 1 |
| 07 | 原五档低秩 | .001 | 关闭（显式 0） | 关闭 |
| 08 | 原五档低秩 | .001 | 关闭（显式 0） | 1 |

完整数组：[single_avg_proto_commands.sh](../system/single_avg_proto_commands.sh)。从服务器 `system/` 目录 source 到原 `run_now.sh` 调度器。该文件只定义变量和八条命令，不执行训练。

GPU 分别可用 `AVG_PROTO_GPU_01` 至 `AVG_PROTO_GPU_08` 覆盖，默认 0–7。输出根 `AVG_PROTO_RUN_ROOT` 默认 `avg_proto_ablation_runs/pat20_seed0_20261102_v1`，各实验有独立编号目录，其下 `checkpoints / logs / h5 / final_models`。正式数组公共参数完整展开：已有 Cifar100/pat_20 NPZ、N20、K1、全参与、seed0、lr=.005、batch16、local5、gr100（101 次）、eg1、无增广；不生成或改写数据。

## 初始化与每轮模型训练分开配置

八组都指定：

```bash
--multi_svd_init low_rank_recovered --multi_svd_init_rank_rate 0.15
```

在原有 CPU seed fork 中构造 rank .15 的 Hyper_CNN_512，调用原恢复函数，规范母模型 `ratio_LR=1`，得到同一 S0。初始化恢复后不重新随机初始化完整层。

- 低秩组另传 `--multi_svd` 和五档 `.90,.37,.35,.25,.15`，按 client_id % 5 固定分配。每轮使用原 SVD、原低秩参数化、本地训练及完整恢复上传。截断／恢复距离保留在 SVD 诊断中。
- 完整组不传 `--multi_svd`，直接复制同一完整 Hyper 母模型参数，训练和上传完整参数。除初始化的一次恢复外，每轮均不执行 SVD 或恢复。没有换成另一种 CNN，没有实现完整模型 F。
- 两种路径均为 RGB32、卷积通道 16→32、800→2000→512→100，共享 fc3。记录实际服务器初始指纹、架构、客户端实际参数量、容量及数据文件/解码指纹。低秩下载截断后的参数与完整模型不要求逐元素相同。

## 显式全局原型协议

开启原型使用：

```bash
--multi_proto_mode global_last_epoch --multi_proto_loss mse
--multi_proto_target_type raw_mean --multi_proto_feature base_output
--multi_proto_lambda 1 --multi_proto_start_round 2
```

关闭使用 `--multi_proto_mode none --multi_proto_lambda 0`。`global_last_epoch + lambda=0` 也严格不采集、不编码、不下发原型。旧 `group_priority` 模式仍采用旧 SGD 前采集及群内优先、按样本数合并的时序，不受此协议影响。

每轮：下载模型和上一轮库 → 普通 local5 → 上传参数和最后 epoch 类别统计 → 全员按原训练样本数 Avg → 类别原型等权合并 → 下一轮使用。

只在最后一个 epoch 的现有训练前向中提取原始 `base_output`，在后续 ReLU 前 clone 并立即 detach 到 float64 类别 sum / int64 count。每样本仅贡献这最后一遍，不收集全 5 epoch，不额外前向或训练，不使用测试数据。特征位置可配置，采集和损失必须一致。

先计算客户端内类别均值 `sum_i,c / count_i,c`，再对拥有该类的客户端均值等权平均。不是把所有 sum/count 直接合并。原型保持原始符号和自然尺度，不归一化、不定长、不 EMA。有贡献者但均值为零仍有效；没有贡献者才无效。

原型在下一轮整个 local5 内冻结。包类型为 `global_last_epoch_local/library`，区别于旧群组库；接收校验特征位置、来源轮次和等权均值表示。第 R1 轮无目标但轮末生成库，R2 只能使用 R1 库，不能提前使用当前轮统计。

## 损失、缺类与裁剪

`CE + is_regular*regular_lamda*原F + multi_proto_lambda*MSE`

F 完全使用原 `frobenius_decay()`，仅涉及 conv2/fc1/fc2 的恢复权重。`is_regular=0` 时不调用它，有效损失和梯度为 0；SGD momentum=0、weight_decay=0。

MSE 与官方缺类约定一致：有效类别用冻结全局原型作为目标；无原型位置的目标只取其自身特征的 detach 副本，使这些位置误差和梯度为零。最终使用 `F.mse_loss(..., reduction='mean')`，分母为整个 batch×特征维度。缺类位置没有被虚构为零原型。目标不参与求导。全无效 batch 为可微零项；没有 1/2、单位化或额外维度缩放。

每 batch 一次原模型前向、一次联合反向、一次 SGD。首轮、关闭和零系数保持原 CE＋F 的反向图，不添加零辅助分支。`local_grad_clip=0` 直接禁用裁剪；诊断只读取梯度计算总 L2 范数，不调用会修改梯度的裁剪函数。

## 数据顺序与观察

仅在本次显式协议中，为每客户端/轮建立私有 Torch Generator：`SHA256(single-avg-proto-v1:seed:loop:client_id)` 的前 8 字节小端整数模 2**63。同时控制 DataLoader base seed 和 sampler；train shuffle=True、drop_last=False、workers=0。八组同客户端/轮批次配对，诊断和原型统计不消耗训练 RNG。

完整模型跳过 SVD 构造器，故其全局 Torch RNG 演化可与低秩路径不同，不为追求相同 RNG 而伪造分解调用或改变旧轨迹。该差异不影响本批私有采样器的批次配对。

`multi_proto_diagnostics=0/1` 控制特征尺度和梯度观察，不改变 SGD。无原型组默认也有相同诊断。特征范数为训练过程样本访问加权均值；梯度范数、裁剪系数和触发比例为逐 SGD 步统计；分类头 weight 范数是训练结束参数范数。它们不是额外 endpoint 测试，不计算各损失的独立梯度。

## 输出与主指标

主指标为每轮全客户端普通 post-local 模型的 `sum(correct)/sum(test_samples)`，使用原 fc3 分类，不是最近原型分类。输出 Final、同轮加权曲线最早 Best、Last10、Last20（评价点）及逐客户端计数。

- 原 `primary_metrics.csv/json`、`primary_summary.json`、`client_metrics.csv/json`、`local_training` 保留。
- `avg_proto_training.csv/json`：CE、有效 F、raw/weighted MSE、总损失、特征和 head 范数、裁剪前范数/系数/触发率、真实步数/访问/时间/参数量、使用原型来源轮次、最后 epoch 采集计数。
- `avg_proto_classes.csv/json`：每类来源轮次、贡献者、类别样本数、范数、有效/零均值状态及库指纹。
- `avg_proto_rounds.csv/json`：采集样本数、0 次额外采集前向、实际编码统计包及下发库字节数、合并时间。通信为逻辑包成本，不宣称实测网络传输。
- H5 的对应 `*_json`、协议属性、主曲线与 JSON/CSV 一致；`rounds.jsonl` 持续记录预算和训练数据。
- 原型只在 checkpoints 和 final_models 中保存最新 `Global_last_epoch_prototypes.packet`；不保存逐样本特征、不放入日志 ZIP。
- 模型加载复用 `utils.multi_group_svd.load_checkpoint()`，完整组的 ratio=1，低秩组保存真实容量的普通预测模型。最终服务器模型为 Avg 后参数，不能代替普通 post-local 主指标模型。

## 验收命令

仓库根运行单元测试：

```bash
python -m unittest discover -s tests -p test_single_avg_proto.py -v
```

从 system/ 运行合成真实入口验收（目录须全新，Windows 建议用短路径）：

```bash
python -u verify_single_avg_proto.py --output /tmp/fedclip_avg_proto_acceptance_20261102 --jobs 2
```

该工具只在新建的专属目录创建合成数据，运行八组两轮 N20 原宽度 CNN，以及零系数和诊断关闭对照。每组完成后默认校验重载并清理自己的模型二进制，只保留轻量日志/凭证。`--keep-models` 可保留。工具内部合成工作区的 `--resume-owned-output` 仅续跑验收用例，不是正式联邦断点续训功能。

本地合成验收不能用于判断真实准确率收益。完整模型容量和五档低秩不同，差异不能全部归因于 SVD；这八组也不是对每种机制做完全部公平调优。

官方依据：[类别原型等权合并](https://github.com/yuetan031/FedProto/blob/main/lib/utils.py#L202)、[最后 epoch 训练特征及缺类损失](https://github.com/yuetan031/FedProto/blob/main/lib/update.py#L122)。本项目仍用原 CE、原 SGD 和参数 Avg，没有照搬官方的数据生成、增广、优化器或推理协议。
