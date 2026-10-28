# Cosine、原始特征 MSE 与 Z2：多群组 SVD 辅助损失

在 `d213477` 上扩展原型模块。默认仍为 `cosine + classifier_input`；旧分组、原型合并、全模型 global-delta 聚合、预测 checkpoint 及评价规则保持不变。本次不启动真实训练，不改变环境。

## 参数与计算

| 参数 | 默认 | 含义 |
|---|---|---|
| `--multi_proto_loss` | `cosine` | `cosine / mse / z2` |
| `--multi_proto_feature` | `classifier_input` | 实际 fc3 输入（ReLU 后）或 `base_output`（最后一个 ReLU 前） |
| `--multi_proto_target_norm` | `1` | MSE 固定目标范数 r，有限正数 |
| `--multi_proto_mode` | `none` | `group_priority` 启用辅助训练协议；Z2 仍完全不使用原型 |
| `--multi_proto_lambda` | `0` | 有限非负系数 |
| `--multi_proto_start_round` | `2` | 人类轮次，第二次本地训练前才生效 |
| `--multi_proto_batch_size` | `16` | 仅原型采集批次大小 |
| `--multi_proto_eps` | `1e-12` | 原型单位化／Cosine 数值约定 |
| `--multi_proto_diagnostics` | `1` | `0` 关闭方向表、训练特征尺度与裁剪统计；不改变目标或训练 |

仍只开放全参与 `FedMultiGroupProj --multi_svd`，无预热、首轮 `full_weights` 分组，共享头、joint、无 KD/Prox/FedRep/头 L2。非默认目标范数在非 MSE 模式报错；Z2 显式使用非默认采集 batch/epsilon 也报错，避免参数被静默忽略。关闭或零系数时不执行辅助特征 hook、辅助项、采集或通信。

一次原模型 forward 通过局部 hook 提取特征。`base_output` 在 head 的原地 ReLU 执行前克隆输出，因此保留负值及正确梯度。hook 不替换模块、参数或 forward，退出时移除；state_dict 和预测结果不变。采集与训练使用同一位置，接收时检查位置一致。

- Cosine：有效类别样本上的 `mean(1-cos(z,p))`，沿用旧 epsilon。
- MSE：有效样本的 `MSELoss(z, r*p)`，默认 reduction=mean，同时除以有效样本数及实际维度；**z 不单位化**，无隐藏 1/2 或额外维度系数，目标 detach。
- Z2：全部样本的 `mean(z.square())`，不读取标签或有效类别掩码来计算辅助项。标签仅用于原 CE。

总目标均为 `CE + 原加权 F + lambda*raw_auxiliary_loss`。三列均训练，辅助项只给骨干特征梯度，分类头继续由 CE 训练。有效样本为空返回可微零；特征、目标或辅助损失非有限会失败。

Cosine/MSE 完全沿用原统计：单位化单样本特征，合并真实类别 sums/counts；群内存在就只用群内，缺类才组外补齐；每轮重建，所有采集完成后才开始任意 SGD。MSE 的单位方向和固定 r=1 不代表历史 CLIP 锚点具有相同尺度，也不宣称异构坐标严格对齐。

Z2 不调用采集、合并、packet 编解码或广播，不生成 `Prototype_library.packet`。来源、原型指纹等字段为 null／不适用，实际原型通信量为 0。`prototype_rounds` 只写明不适用及零成本，原型来源／采集表为空。首轮与其他模式一样完全关闭辅助项。

## 启动数组

从服务器 `system/` 目录将 [multi_group_proto_loss_commands.sh](../system/multi_group_proto_loss_commands.sh) 中的变量定义和完整 `declare -a COMMANDS` 一起复制到现有调度配置。文件只定义数组，本身不启动训练。

四条命令：无辅助、Cosine λ=1、MSE λ=1/r=1、Z2 λ=1；**全部 base_output**。全部 `-gr 100`，实际 101 次训练／聚合。原五档容量、K3、7/7/6、rank .15 恢复初始化、Projection β=.5/τ=.2、local5、F=.001、clip10 等公共设置一致；GPU 示例 0/1/2/3 可替换。

默认根目录 `multi_proto_loss_runs/$PROTO_LOSS_RUN_TAG/`。每个实验独立 checkpoints/logs/h5/final_models。无辅助首轮发布共享分组，其余运行完成同一首轮训练后有限等待该文件；不跳过训练或重置随机流。

复用已有**兼容**分组时，在读取命令文件前定义：

```bash
export PROTO_LOSS_GROUP_FILE=/absolute/path/to/original_capacity_recovered_init_groups.json
export PROTO_LOSS_REUSE_GROUP_FILE=1
source ./multi_group_proto_loss_commands.sh
# 将 COMMANDS 交给已有调度器；本文件不调用调度器。
```

不提供该覆盖时，使用本批独立目录的新共享分组文件。协议仍检查模型、容量、数据、初始化与训练兼容性。四组不会独立重分组。

同为 λ=1 不代表三种损失具有相同尺度、梯度强度或已完成公平调优。旧 31 轮 ReLU 后 Cosine 仅作背景，不能替代本次 ReLU 前 Cosine 对照。原 31 轮命令文件保留不变。

## 输出和机制诊断

沿用 `prototype_training.csv/json`、`class_prototypes.jsonl`、H5 `prototype_training_json` 和 manifest。新增字段：

- `aux_requested_loss / aux_effective_loss / aux_feature_position`，请求／实际系数仍为 `proto_requested_lambda / proto_effective_lambda`。
- `aux_target_norm / aux_effective_target_norm`：仅 MSE 有意义。
- `raw_auxiliary_loss / weighted_auxiliary_loss / total_loss`；CE 和 `regularization_loss` 仍为原定义。
- `auxiliary_valid_samples / auxiliary_valid_ratio / auxiliary_valid_unit`：Cosine/MSE 为具有有效类别原型的 SGD 样本访问；Z2 为全部 SGD 样本访问。
- `prototype_participation / prototype_status`；Z2 的旧 `raw_prototype_loss`、原型有效数和指纹为 null，以免把无锚点损失称为原型损失。
- `training_mean_feature_l2`、`training_mean_feature_squared_per_dimension`：在原训练前向直接累积，按实际样本访问量加权，**不是 endpoint 全数据评价**。
- `preclip_gradient_norm_mean`：原 `clip_grad_norm_` 返回的裁剪前范数，按 SGD 步数算术平均；`gradient_clipped_steps / gradient_clip_trigger_ratio` 使用原实现的 `cap/(norm+1e-6)<1`，不额外 backward 或再次裁剪。

诊断关闭、辅助未生效或裁剪关闭时相应观测为 null/0，不伪造数据。关闭辅助的对照完全沿用旧路径，因此不额外采集特征尺度。损失过程均值是 batch 目标按实际 SGD 样本访问加权，不是 endpoint CE，也不是把所有有效样本重新拼起来求全局均值。

继续使用原型来源／采集／上传／逐容量方向表及 `prototype_rounds`。原型前向样本量、SGD 访问量、packet 字节分别记录，不把采集计入训练 epoch。日志 ZIP 不保存逐样本特征或模型副本。辅助项不参与推理，最终普通低秩 checkpoint 加载方式不变。

主指标仍为固定归属模型普通 post-local 的全员 `sum(correct)/sum(test_samples)`，保留同轮曲线 Final、最早 Best、Last10/Last20，微调前、群组和容量结果不改。

## 本地短程验收

```powershell
& D:\miniconda\envs\htfllib\python.exe -m unittest discover -s tests -p test_multi_group_proto_losses.py -v
& D:\miniconda\envs\htfllib\python.exe system/verify_multi_group_proto_losses.py --output system/temp/multi_proto_losses_acceptance_new --jobs 2
```

Linux 服务器可在 `system/` 执行 `python verify_multi_group_proto_losses.py --output temp/multi_proto_losses_acceptance_new --jobs 2`。工具自行在新目录创建**合成**小数据，只运行两轮，不读取或改写真实分片；固定 Git `d213477` 作为行为参照。默认保存重载检查后清理本工具创建的模型文件，保留轻量日志和验收收据。

验收结果见 [独立报告](multi_group_proto_loss_acceptance.md)。本地没有真实训练分片，本次不判断真实涨点，也不据两轮合成实验判断长期效果。
