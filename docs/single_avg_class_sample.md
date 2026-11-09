# K1 Avg：类别样本数加权原型与采集时序对照

这是 `FedMultiGroupProj --multi_avg_proto_ablation` 的独立扩展：K=1、全员构成唯一群组、参数按原样本量Avg，不聚类、不读取K3分组、不引入目标客户端偏置。继续聚合模型参数，因此不是完整FedProto算法。

历史AM02及约55%的客户端等权结果、旧八组命令文件均保留。本次01是**需要重新运行**的类别样本数加权基线，不能覆盖或重新命名历史结果。

## 参数和范围

新增 `--multi_proto_aggregation client_equal/class_sample`，默认`client_equal`。新增 `--multi_proto_mode global_pre_train`，旧`global_last_epoch`继续有效。两条新命令都显式设为`class_sample`。

| 配置 | 采集 | 使用 |
| --- | --- | --- |
| 01 `global_last_epoch + class_sample` | 最后一个本地epoch已有训练前向，编码器随SGD变化 | 下一次联邦训练 |
| 02 `global_pre_train + class_sample` | 本轮实际下载并SVD后的低秩模型，完整自身TRAIN集eval/no_grad一次 | 本次联邦训练 |

本次新增协议只开放K1异构SVD客户端、shared/joint、全参与Avg。`class_sample`不向旧K3或其他算法开放；训练前协议需显式使用`class_sample`。非零新协议不支持完整模型路径，避免启用未验收的实现；旧完整模型的客户端等权消融保留。none或系数0不采集、不编码、不下发库、不施加MSE。

损失为原CE＋原加权F＋`multi_proto_lambda * MSE`。使用原`single_avg_proto.global_mse`：无效类别对应误差和梯度为零，分母仍是**整个batch×实际特征维度**，而非有效样本数。每batch一次原forward、一次联合backward、一次原裁剪和一次SGD。骨干与分类头均训练；预测仍用原fc3。

默认客户端等权继续调用原函数及原packet格式，默认训练路径不增加采集。新协议使用独立v2统计／库格式及更严格的阶段校验，不改旧packet指纹。

## 聚合公式

复用`LastEpochStats`及已有float64类别sum、int64计数。对类别c：`P_c = sum_i(s_i,c) / sum_i(n_i,c)`。使用真实类别计数，不使用客户端总样本数，不再除以客户端人数，不归一化，不定长，无EMA。零计数无效；有计数的零均值有效。拒绝重复贡献者、负数／错误dtype／非有限统计、空类非零sum、维度错配、错轮次、缺包及未完成的全员统计。

两种采集均使用`Features(..., base_output)`，及时clone防止分类头原地ReLU改写。01仅最后epoch立即detach后累积，不保留逐样本图；02使用相同原始特征位置，不用完整服务器模型替代异构低秩编码器。

## 时序与来源

01：全员下载→上一轮库→本地5epoch（最后epoch收集）→模型Avg→本轮类别sum/count合并→下一轮库。首轮MSE为0，但收集最后epoch统计。

02：首轮普通CE＋F，无任何原型采集。第二轮起，**全员全部下载完成→全员独立TRAIN扫描→收齐统计→合并→给全员绑定同一库→任何客户端SGD**。库只用于本次训练，下轮重建，不能回用上轮。复用真实下载模型，不增加SVD或优化步骤。

采集使用独立Sequential DataLoader（无增强、不shuffle、不drop_last、workers=0）。快照恢复参数、buffers、模块状态、Python/NumPy/Torch CPU与实际模型所在CUDA设备的RNG，不查询所有可见GPU。训练继续使用旧单模型协议的seed/client/round私有generator，两组同轮同客户端批次配对。

v2包、记录和checkpoint明确包含：

- `collection_loop_round`和`collection_phase`（`last_local_epoch/pre_local_download`）；
- `intended_use_loop_round/intended_use_completed_train_count`；
- `source_server_completed_aggregations`、服务器及逐客户端实际下载模型指纹；
- 最后epoch协议另记各客户端post-local指纹，并标记`encoder_evolves_within_collection=true`，不把这些特征说成固定最终编码器的输出；
- `aggregation`、特征位置、初始化与数据指纹、实际库sha256。

例如loop=1：02在此前已完成1次服务器聚合、尚未完成本次第2次训练时采集，用于loop=1；01此时使用loop=0最后epoch采集的库。两者某个source计数虽可能同为1，阶段／采集loop／预期使用loop不同，不能互换。

## 配置与命令

[`system/single_avg_class_sample_commands.sh`](../system/single_avg_class_sample_commands.sh)只定义两条完整COMMANDS，不启动任务，GPU为0/1且`-did`靠前。公共配置与原AM02一致：CIFAR-100已有pat20、N20/K1、seed0、原五档SVD容量、恢复初始化rank=.15、每模型5epoch、lr=.005、batch16、F=.001、clip10、raw_mean/base_output MSE=1、第二次训练起使用、gr100实际101次。01和02只有时序及独立输出位置不同，均为类别样本数加权。

输出根目录：`system/single_avg_class_sample_runs/pat20_seed0_gr100_20261109_v1/`。

命令使用服务器已有的`/data/zh/miniconda3/envs/htfllib/bin/python`，避免`run_now`的login shell回落到系统Python；不安装或修改环境。

- `01_last_epoch_next_round/{logs,checkpoints,h5,final_models}`
- `02_pre_train_this_round/{logs,checkpoints,h5,final_models}`

不重跑真实纯Avg、不生成或改写真实数据、不改旧分组或绘图、不更改环境。

## 记录与成本

沿用`primary_metrics/primary_summary/client_metrics`与全20人post-local正确数／测试样本量加权主指标；best取同轮总体曲线最大值及最早轮次，last10/20为评价点。微调前指标仍在真实下载后、SGD前只读观察。

`avg_proto_training.csv/json`记录CE、有效F、raw/weighted MSE、总损失、样本访问、步数、梯度和裁剪。新字段分别记录训练前额外采集样本／batch／耗时，以及最后epoch统计batch／耗时、编码耗时、逐客户端逻辑上传／下发字节。

`avg_proto_classes.csv/json`记录贡献者ID、各自类别计数、类别总数、实际`n_i,c / sum(n_i,c)`、有效性、零均值、范数及来源。`avg_proto_rounds.csv/json`记录统计上传、原型下发和合并成本；`avg_proto_protocol.jsonl`逐轮持久化。训练、额外推理和统计／编码成本分开，不能把02的额外推理称为SGD epoch，也不能声称两组计算成本相同。

CSV／JSON／H5和最终manifest使用同一组记录。保存两个可追溯的库：

- `Global_class_sample_latest_generated.packet`
- `Global_class_sample_last_used.packet`

01在第101次训练后生成的库计划给第102次使用，实际没有使用；最后实际使用的是第100次训练生成的库。02最后生成与最后使用均为第101次训练前的同一库。manifest分别写出metadata及sha256，不用“最新库”冒充训练目标。预测checkpoint不依赖这些辅助库。

## 验收

纯数值测试覆盖不等类计数、同编码器分布式与合并数据均值、缺类／零均值／非法包、原MSE梯度与来源拒绝；冻结`63b91d5`的默认客户端等权路径，比较低秩模型、恢复上传、Avg、原型packet、实际batch及RNG。

```powershell
$env:PYTHONPATH='tests'
python -m unittest -v test_single_avg_class_sample test_single_avg_class_sample_runtime test_single_avg_proto test_multi_last_epoch_proto test_multi_shared_proto
python system/verify_single_avg_class_sample.py --output E:/single_avg_class_sample_unique
```

真实main合成smoke使用N20/K1、原宽度CNN和五档容量、两次训练、每次5epoch，含两种时序、诊断关闭、none／零系数。检查屏障、公式、训练预算、导出与重载。所有合成结果只证明实现，不代表真实准确率改善，也不宣称异构特征等价于统一编码器。

结构化验收结果见[`single_avg_class_sample_acceptance.json`](single_avg_class_sample_acceptance.json)。
