# K3共用一套自然尺度全局原型

在 `8080557` 上新增显式 `--multi_proto_bank_mode shared`，用于与已有开裁剪第05组的 `per_model` 全局原型对照。原有五组命令不改写，默认仍为 `per_model`。本次只改变是否跨模型列共享原型；模型聚合、固定成员、SVD、CE、原F正则、MSE、采样和评价入口继续复用。

## 使用范围

新开关只用于 `FedMultiGroupProj --multi_svd --multi_last_epoch_protocol`。开启辅助项时要求：

```text
--multi_proto_mode last_epoch
--multi_proto_scope global
--multi_proto_bank_mode shared
--multi_proto_loss mse
--multi_proto_target_type raw_mean
--multi_proto_feature base_output
--multi_proto_lambda 1.0
--multi_proto_start_round 2
```

`none` 或系数0仍走原无原型SGD：无统计、原型编码、上传和下发。关闭时允许 `shared` 配置配合原默认scope，但不会生成任何库。非last-epoch协议拒绝非默认bank mode；有原型的shared模式拒绝group_priority。

## 两级等权合并

每个客户端每列只收最后一个epoch原训练前向中的 `base_output`，在最后一次原地ReLU前clone；立即detach，累积float64类别sum和int64 count。每batch仍一次前向、一次联合反向、一次SGD，不增加扫描或微调。

服务器先核验完整N×K包、客户端/列/轮次/特征位置及shape/dtype/有限性，并要求同一客户端各列的完整类别count逐元素相同。对每个类别：

1. 每客户端、每列先以真实类别样本数计算局部均值。
2. 同一客户端内按模型列编号升序等权平均K个均值。
3. 对拥有该类的客户端按ID升序等权平均，**不按样本数或群组人数加权**。

无拥有者才无效；有样本但均值抵消为0仍有效。不单位化，不乘固定范数，不用EMA。K/C/D均从实际配置或模型读取。跨模型坐标直接算术平均就是本次实验变量，不声明这些坐标天然严格对齐。

继续使用原MSE：缺失原型位置误差为0，以整个batch×特征维度为分母；目标detach。所有客户端的所有模型列都使用同一套P和有效性掩码，包括模型所属组外的helper。

## 时序与一套库的含义

首轮只训练CE＋F，同时收集最后epoch统计。轮末生成共享库；第二轮开始，每轮先下发上一轮库，整个本轮冻结使用，轮末再生成下一轮库。没有旧训练前扫描路径，没有额外训练。

共享包kind为 `last_epoch_shared_library`，`prototypes` 的真实shape是 `[C,D]`，`valid` 是 `[C]`。不会堆叠成K份。每个客户端只clone一次P与mask，K列训练均引用这个对象；客户端和模型列身份保留在训练行与预测checkpoint中。

上传仍是N×K份原始sum/count包。下发按 **N×一套共享包字节数** 计，不乘K。实际包中另有每类贡献/样本计数，按真实编码长度统计；不是只用P元素数估算。这里统计的是原型逻辑传输量，不包含模型通信，也不是物理网络测量。初始化、完整母模型下发和恢复完整上传方式均未改。

## 正式命令与对照

从服务器 `system/` 使用 [multi_last_epoch_shared_commands.sh](../system/multi_last_epoch_shared_commands.sh)，文件只定义一条完整 `declare -a COMMANDS`，不自动训练。

配置为pat20、20人、K3、7/7/6、seed0，原五档秩与low_rank_recovered/.15初始化；每列local5、batch16、lr.005、F.001、**clip10**、group_projection_softmax/.50/.50/tau.20、gr100实际101轮。每客户端每轮15次数据遍历，推理只使用固定归属的一个普通post-local模型。

只读既有分组：

```text
/data/zh/AllRepo/FedClip/system/multi_proto_runs/pat20_seed0_gr30_20261027_v1/shared/K3_full_weights.json
```

继续使用 `members_only`：检查真实数据、初始参数、N/K、组大小、模型和容量，记录原文件SHA与协议差异，不修改原JSON、不重分组。对照为用户已完成的**clip10第05组**；旧命令文件和其他实验保持原样。

新输出根：`system/multi_shared_proto_runs/pat20_seed0_gr100_clip10_20261105_v1/01_global_shared_bank/`，分别包含logs、checkpoints、h5、final_models，运行ID和shared_bank名称继续隔离产物。

## 日志、重载及来源轮次

复用 `last_epoch_training/classes/rounds.csv/json`、H5对应JSON数据集、协议属性和最终manifest。

- 配置记录 `bank_mode`、K、库数量和两级等权公式。训练行的 `actual_prototype_bank_count` 表示实际收到的库数：首轮0，第二轮起1。轮末类/库记录描述本轮刚生成的库，因此首轮轮末已有1套。
- 每类记录实际贡献客户端ID/数量、`client_model_contributor_count`、`deduplicated_class_samples` 和 `cross_column_class_sample_visits`。后二者分别是Σn与K×Σn；绝不以跨列访问量伪装为新增训练样本。
- 记录原型L2范数、零均值有效标记、来源轮次、首次可用轮次、库SHA，及本轮训练所用库的SHA/来源轮次。
- 记录真实统计上传/库下发字节、交付次数、客户端编码时间、服务器解码/合并/编码及分发成本。SGD样本访问与最后epoch统计访问分开；额外原型采集前向为0。

最终分别保存：

- `Last_epoch_shared_prototypes.packet`：最后生成的库。101轮实验中source=101，供潜在下一轮使用。
- `Last_epoch_shared_training_prototypes.packet`：最后一轮实际训练使用的库。101轮实验中source=100，used_training_round=101。

读取示例（先确认可信文件来源）：

```python
from pathlib import Path
from utils.multi_group_proto import decode_packet
values, metadata = decode_packet(Path(path).read_bytes(), 'last_epoch_shared_library')
P, valid = values['prototypes'], values['valid']  # float64 [C,D], bool [C]
```

P不参与推理。普通低秩客户端模型和三个完整服务器模型继续按现有 `utils.multi_group_svd.load_checkpoint` 重载。manifest中的latest-generated和latest-training-used文件、SHA及轮次分开，不把最新生成库误称为最后训练使用库。

主指标仍是固定归属post-local模型的全20人测试样本量加权准确率；Final、同轮总体曲线的最早Best、Last10/Last20及逐客户端/组/容量结果沿用原实现。

## 短程验收

从 `system/`：

```bash
python -m unittest discover -s ../tests -p 'test_multi_shared_proto.py' -v
python verify_multi_shared_proto.py --output /tmp/fedclip_shared_proto_acceptance
```

输出目录必须未存在。Windows建议短路径，例如 `D:/sh_acceptance`。工具只在指定输出根建立合成分片、独立分组fixture，串行跑诊断开启/关闭的20客户端、K3、两轮原宽度异构CNN；不会读取、生成或修改真实分片。每case600步，验证单前向/反向/裁剪/更新、独立两级等权公式、第二轮全部60个训练任务同库、真实字节数、CSV/JSON/H5/manifest及63个预测模型和两份原型文件重载。默认核验后删除模型二进制，保留报告/日志；`--keep-models`可保留。

另以冻结 `8080557` 验证默认每列库的训练参数、上传、原型包、batch及RNG；覆盖非均衡样本、K1/2/4、2/1不等组人数、零/缺失类别、缺列、错轮次、计数错配、零系数和诊断隔离。合成验收不能证明真实准确率改善。具体结果见 [验收报告](multi_shared_proto_acceptance.json)。

2026-11-05 本地149项检查通过，1项旧CUDA专项因主机分配/cuSolver资源问题未验收；其余补充回归在独立CPU测试进程中完成，未改变依赖或生产后端设置。新模式诊断开/关两组均完成20客户端、K3、两轮、clip10实际入口合成检查，独立两级均值核对误差0，参数/上传/batch/RNG及原型指纹隔离通过。126个普通预测模型、最新生成与实际训练使用的共享库均重载核对后清理。

该小分片fixture第二轮统计上传为24,641,670字节，生成一套库414,060字节，实际下发为20×414,060＝8,281,200字节，交付20次而非60次；类别样本去重总数50、跨列统计访问150。数值仅说明此fixture的包编码和计数，不代表真实数据准确率、网络时延或正式训练开销。
