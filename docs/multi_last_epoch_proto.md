# K 列最后 epoch 原型协议与五组消融

这是在固定 K、多模型、异构 SVD 框架中的显式新协议。默认不启用，旧 K1 八组、旧 K3 的训练前原型扫描、KD、初始化和聚合核保持原路径。

## 五组完整命令

[multi_last_epoch_commands.sh](../system/multi_last_epoch_commands.sh) 包含五条完整 `declare -a COMMANDS`，从服务器 `system/` 目录交给原调度器。该文件只定义数组，不启动训练；`-did` 在命令开头，GPU 直接写为 0、1、2、3、4，没有 GPU 环境变量。

| 编号 | 模型聚合 | 原型贡献者 |
|---|---|---|
| 01 | group_avg_only | 不启用 |
| 02 | group_projection_softmax | 不启用 |
| 03 | group_avg_only | 群内有类则群内，缺类才组外 |
| 04 | group_projection_softmax | 群内有类则群内，缺类才组外 |
| 05 | group_projection_softmax | 全部客户端中拥有该类者 |

04 与 05 仅贡献者范围不同。Avg-only 的实际群内/群外模型质量为 1/0；Projection 仍为 .50/.50、tau=.20，逐列调用原 `aggregate_group_updates`。

公共命令为：完整已有 Cifar100/pat_20 NPZ；20 客户端，全参与；K3，各模型 local5（每客户端每轮共15次数据遍历）；原五档 .90/.37/.35/.25/.15、id%5；low_rank_recovered 母模型初始化 rank=.15；F=.001、SGD .005/batch16/momentum0/weight_decay0；显式 clip=0；gr100 实际101轮、eg1；共享分类头、joint、无增广。

输出根为 `multi_last_epoch_runs/pat20_seed0_gr100_20261102_v1/`，五组编号目录分别含 `checkpoints / logs / h5 / final_models`，不会覆盖旧实验。旧 K1 第07/08组不自动重跑。

## 显式参数

```bash
--multi_last_epoch_protocol
--multi_proto_mode none/last_epoch
--multi_proto_scope group_priority/global
--multi_proto_loss mse
--multi_proto_feature base_output
--multi_proto_target_type raw_mean
--multi_proto_lambda 1
--multi_proto_start_round 2
```

无原型命令使用 mode=none、lambda=0，目标类型保持未生效的默认 fixed_norm。`last_epoch + lambda=0` 也不构造统计、不编码、不下发库。该显式协议仍启用配对 loader 和训练诊断，保证无原型对照与原型组预算/采样一致。

新协议只开放 FedMultiGroupProj 的 SVD、完整数据、全参与、shared/joint、无预热、固定文件成员、原始 base_output MSE；不能混用 KD/Prox/Z2/head L2/FedRep/反馈。不将 K 写死为3，N/K/容量/学习率/epoch/路径等仍读取实际参数。正式数组固定 K3 和7/7/6。

## 复用 K1 训练与 MSE

`single_avg_proto.py` 的最后 epoch sum/count、等权合并和全 batch 缺类 MSE 是唯一共享实现。训练函数接受模型列索引，K1 默认仍为0且保留原包内容；K列协议通过薄的调度层调用每列相同的 SGD 代码，不复制一套近似损失。

每轮：全体下载/分解 K 列 → 绑定上一轮 K 套库 → 每客户端依次训练全部 K 模型 → 只采各模型最后 epoch 的现有训练前向 → 上传恢复参数与类别统计 → 原核逐列模型聚合 → 逐列类别合并 → 下一轮使用。

R1 没有 MSE，但原型开启时仍收集最后 epoch，轮末产生供 R2 使用的库。接收库必须满足 `source_completed_train_count == loop_round`。库中每列有独立维度切片和 model_id 检查，禁止错列、旧轮次、错 scope/特征位置/分组指纹的库。

统计是 ReLU 前的有符号512维特征；原 forward hook 内 clone 防后续原地 ReLU，采集立即 detach。只累积 float64 类别 sum / int64 count，不保留样本图，不额外前向，不扫描测试集。

先在客户端内求类别均值，再按所选客户端等权平均。群组优先按每列自己的群组与真实训练计数选择贡献者；全局则每列独立使用全客户端贡献者。不会把 A/B/C 特征混到一套库中。样本数只用于本地均值和诊断，不直接合并为服务器样本加权原型。有贡献者的零均值有效，没贡献者才无效；不归一化、不定长、不 EMA。

损失为 CE＋原F＋lambda*MSE。MSE 复用 K1：缺类位置使用学生特征的 detach 副本作为目标，只产生零误差/零梯度，不伪造零原型，最终按整个 batch×实际维度均值归约。全部客户端、全部模型列使用其对应库，包含该列组外 helper。每 batch 一次原模型前向、联合反向和 SGD。clip=0 时只观察梯度范数，绝不调用修改梯度的裁剪函数。

## 配对采样

第0列完全复用 K1 的 `loader_seed(seed,client,loop)`。其他列使用相同稳定 SHA256 key 加 `:column:k`。各客户端/轮/列独立 Torch Generator 同时控制 loader base seed 和 shuffle，workers0、drop_last=False。五组的同列批次严格配对。该行为仅显式新协议启用，不修改旧 K3 全局随机流规则；统计、打包和诊断不推进训练 RNG。

## 固定分组与受控复用

五条命令只读同一个既有文件：

```text
/data/zh/AllRepo/FedClip/system/multi_proto_runs/pat20_seed0_gr30_20261027_v1/shared/K3_full_weights.json
```

用户提供的来源核对名单为 G0=`0,2,3,8,9,10,11`，G1=`4,5,6,12,13,14,15`，G2=`1,7,16,17,18,19`。这只是文档中的核对信息；算法始终读取文件，不把该名单写入训练或用于错误时回退。

使用显式 `--multi_group_reuse_policy members_only`。它首先通过旧校验器完整验证原文件自身 context/hash、evidence、成员规范化、分区、映射及 group_fingerprint；随后继续严格对比数据/解码指纹、客户端ID全集、N/K、人数、模型族、初始权重指纹、原始容量协议及初始化协议。

受控模式只允许记录并接受 `training.grad_clip`、`sampling_protocol`、`grouping_protocol` 的差异；seed/lr/batch/local_epochs 或其他上下文差异仍拒绝。原上下文、当前上下文、所有差异、来源文件 SHA256 与 source_path 保存在 `group_reuse_verification.json` 及 manifest。输入 JSON 不被修改，不把它伪装成本次新协议重新生成的文件。

默认 `strict` 仍使用原完整匹配校验，未被放宽；非新协议不能启用 members_only。缺文件立即报错，不等待、不自动聚类，也不开始 SGD。新协议从训练前便使用已固定成员；source 的原始形成时点作为来源保留，当前时点明确记录为固定文件复用。

本地没有服务器上的原文件，不能宣称该真实文件已通过兼容性验证。若其实际母模型/容量或数据与本次不符，会明确失败，不能仅凭成员名单相同绕过。文中不硬编码原名单到算法。

## 日志与模型

- 保留全员所属模型 post-local 测试样本量加权主指标、Final/同轮平均最早Best/Last10/Last20、逐成员/群组/容量、训练前评价与原投影诊断。用原分类头推理，不选CE/测试最优模型、不融合预测。
- `last_epoch_training.csv/json`：每客户端每列 CE、加权F、raw/weighted MSE、总损失、特征/head范数、只读梯度范数、clip系数/触发、真实 epoch/batch/访问/时间、所用库来源轮次及哈希、所属模型、分组来源、统计编码字节/耗时。
- `last_epoch_classes.csv/json`：每轮每列每类 scope、group/outside/global/invalid、实际贡献者/人数/类别样本数、范数/零均值/有效性、来源及首次可用轮次、库与分组指纹。
- `last_epoch_rounds.csv/json`：每轮最后 epoch 样本计数、0额外采集前向、实际编码上传/下发字节，以及解码、合并、编码、分发耗时。字节为本进程逻辑包，不声称物理网络测量。
- 同步写 H5 对应表和协议属性；持续 `rounds.jsonl` 仍含训练、主指标和真实预算。
- 普通低秩 checkpoint 包含 model_id、assigned_model_id、来源分组 SHA、新原型协议、训练实际使用的库来源轮次/指纹以及本轮统计来源轮次；manifest 区分最新使用库与最新生成库，避免将轮末库误记成本轮训练目标。最终导出全部 K 服务器与 N×K 普通客户端模型。最新原型库另存 `Last_epoch_columns_prototypes.packet`，不参与预测，也不放日志 ZIP。

## 短程验收

仓库根目录：

```bash
python -m unittest discover -s tests -p test_multi_last_epoch_proto.py -v
```

从 system/ 运行真实入口合成验收（目录须不存在）：

```bash
python -u verify_multi_last_epoch_proto.py --output /tmp/fedclip_last_epoch_acceptance_20261102 --jobs 1
```

该工具只创建专属合成数据和一份有完整校验字段的合成来源分组，运行五组 N20/K3/local5 两轮及零系数、诊断关闭对照；逐列对照旧聚合核、独立合并公式、分组文件只读、各格式导出和63个模型重载。成功后清理自己的临时 checkpoint/模型，只保留轻量结果。Windows 建议短输出路径、串行运行；本机并行运行曾耗尽 CPU 内存提交量。

本批是统一新协议后的重测，同时改变旧K3的原型时序、客户端等权规则、尺度及裁剪，并采用新的配对采样。不能将结果变化只解释为时序作用，也不能用合成 smoke 判断真实准确率收益。
