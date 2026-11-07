# 按客户端×模型列管理群内共享头与helper头

基于 `dc30a19`，新增独立开关；旧 `shared + full` 不绑定新角色、不复制头、不改变原上传或聚合核。旧单目标 `target_head_mode local` 不改写。本次不推送、不修改环境/真实分片/分组JSON，不启动正式训练。

## 参数与三组新对照

| head mode | projection scope | 下载及跨轮状态 | 分类头聚合 |
| --- | --- | --- | --- |
| shared（默认） | full（默认） | 旧流程 | 旧全模型Projection |
| shared | backbone | 每列全员下载服务器头 | 使用该列骨干评分产生的原权重，全员平均头，不投影头 |
| group_only_reset | backbone | 每列全员下载服务器头 | 仅群内，按群内训练样本数归一化，质量1 |
| group_only_local | backbone | 群内下载；helper首次用公共初始化，此后保留同客户端同列上一轮post-local头 | 仅群内，同上 |

非默认配置首版只开放：SVD `FedMultiGroupProj`、`Decom_CNN-5-512`、已知固定分组、无预热、`group_projection_softmax`、`multi_last_epoch_protocol`、per_model/group_priority、共享分类头设置、joint训练；拒绝KD/Prox/Z2/head L2/replay/K1专项协议。原型可关闭；开启时仍由原last-epoch协议校验MSE等配置。默认shared/full不增加这些限制。

分类器按参数对象确认且打印规范名 `fc3.weight/bias`；`head.1`与fc3的别名只计一次，fc1/fc2保留在骨干。K从参数读取，不把每人一项群内、其余helper角色硬编码为3列。

## 执行与参数范围

固定分组在服务器初始化时读取，随后每轮**下载前**绑定client×column角色。SVD沿用原实现，不创建额外随机分类头。local helper在重建模型前保存旧头，分解完成后复制到新fc3，再创建绑定新Parameter的无动量SGD。群内头每轮覆盖为本列服务器头。两个group_only模式首轮使用同一头并产生相同完整训练轨迹，第二轮起reset与local的头状态分化。

backbone分支把原 `aggregate_group_updates` 的输入范围限定为恢复后的骨干named_parameters；参考、点积、范数、余弦、softmax、冲突和投影均不包含fc3。核内epsilon、顺序、归一化、温度与公式均不改。骨干仍接受所有客户端贡献，正式命令质量.50/.50。

分类头独立加权：shared使用同一列骨干权重；group_only使用 `n_i / sum_group(n)`，总质量1。group_only helper的**实际上传字典没有fc3条目**，不塞零头/服务器头占位。完整恢复仅用于本地生成可上传视图，普通低秩预测checkpoint保留实际头。聚合完成前不修改服务器任一列。

SVD截断、本地训练恢复距离、global-delta诊断在新分支明确限定骨干；私有头与服务器头的差异不计入这些范数。服务器模型之间的几何诊断仍看完整实际模型，单独标注范围。

训练预算、CE/F/MSE、最后epoch统计、下一轮原型冻结使用、数据顺序及原整体梯度裁剪不变。helper头不上传不会删掉helper原型统计。每客户端每轮仍K×local_epochs次数据遍历，正式K3每列5epoch共15次；不冻结头、不增加头预热/微调。

## 记录与保存

沿用 `local_training`、`last_epoch_training`、`client_metrics`、聚合权重的CSV/JSON和H5 JSON数据集；新增 `head_aggregation.csv/json` 与H5 `head_aggregation_json`，逐列逐客户端记录头/骨干权重、角色及头权重和。

训练记录及客户端checkpoint中的 `multi_head_policy` 包括：client/model/assigned_model、group/helper角色、模式、范围、头是否实际覆盖/恢复、是否上传、私有头来源完成训练次数、下载后/训练后SHA256和范数、骨干/头上传参数数及字节。`private_head_source_completed_train_count=0`为公共初始头，1为R1真实post-local头。endpoint行明确 `all_client_shared/group_shared/reset_helper/private_helper`；主指标只取固定所属列。

manifest、capacity_manifest、H5属性、最终服务器checkpoint明确实际头策略及骨干范围；源分组的不可变容量/初始化上下文保留原值供members_only校验，运行时容量描述另外标记新的分类头策略，避免混淆来源和本次执行协议。

当前仍完整母模型下发后再恢复helper本地头。日志中的full/head download bytes据此计数，**不声称减少头下行通信**；上传视图计数与完整恢复缓存/预测checkpoint大小分开。字节为参数逻辑载荷，不是物理网络测量或峰值内存。日志ZIP不放模型副本。

模型加载继续使用：

```python
from utils.multi_group_svd import load_checkpoint
model, metadata = load_checkpoint(path, device='cpu')
model.eval()
```

`Client_i_model_k.pt`是该客户端该列实际低秩预测模型，包含实际共享/重置/私有fc3；`Server_model_k.pt`包含实际聚合骨干和群内/全员头。推理不依赖原型库或私有头缓存。

## 三条正式命令

从服务器 `system/` 读取 [multi_group_head_commands.sh](../system/multi_group_head_commands.sh)。文件仅定义完整 `declare -a COMMANDS`，三条命令GPU0/1/2且`-did`靠前；不会自动执行。

公共配置：dir0.5真实分片、N20/K3、7/7/6、seed0、gr100实际101次、本地每列5epoch、batch16/lr.005、原五档秩、low_rank_recovered/.15初始化、F.001、clip10、beta.50/tau.20、last_epoch/per_model/group_priority/raw_mean MSE/base_output/λ1/R2开始。输入为 `../dataset/Cifar100/dir_0.5`，NPZ-only；真实config必须匹配dir/alpha.5/N20，错误会失败，不自动找其他目录。

三组只读同一既有分组：

```text
/data/zh/AllRepo/FedClip/system/dir05_proto_runs/dir0p5_seed0_gr100_20261105_v1/shared/K3_full_weights.json
```

使用members_only受控复用，核验数据/初始化/容量/N/K/组大小，保存来源SHA及协议差异，不重分组、不发布新文件。已有shared/full 04_AB由用户复用；新批独立根为 `system/multi_head_runs/dir0p5_seed0_gr100_20261107_v1/`，三个子目录各有logs/checkpoints/h5/final_models。最终实际路径还带原框架运行名/运行ID，见启动日志。

## 验收与复验

数值/训练回归：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& D:/miniconda/envs/htfllib/python.exe -m unittest discover -s tests -p test_multi_group_head.py -v
```

包含非均衡样本、明确反向helper、旧核逐元素对照、头质量1、改helper头不改骨干、占位头拒绝、别名/优化器绑定、两轮reset/local、原型包/批次/RNG一致，以及默认shared/full相对冻结`dc30a19`训练与聚合的严格回归。

服务器可复验标准结构、20人×K3×两轮、三种新配置、模型重载与导出的**合成验收**：

```bash
python verify_multi_group_head.py --output temp/multi_group_head_acceptance_20261107
```

此工具默认完整卷积宽度16、512维特征、100类、五档SVD，local5、F.001、clip10和原型协议均保留。只在自己新建目录内生成合成夹具，拒绝覆盖目录；不会读取/改写真实训练分片或旧分组。每组完成后核验63份模型重载，随后清理自己创建的大模型文件，保留小型日志和报告。需要保留模型加`--keep-models`。

本机内存限制及实际运行状态见 [验收报告](multi_group_head_acceptance.json)。`--width 2`是明确标记的缩小卷积夹具，不等同于正式结构验收。可选`--upload-backing mmap`仅让测试中的恢复上传张量使用无损文件映射存储，逐元素核验值、dtype和stride，不改模型、优化器、生产代码或数值计算；`--svd-diagnostics 0`只关额外距离缓存观察。实际采用的宽度、诊断与存储条件均在报告中列出；文件映射验收不等同于默认纯内存上传缓存的资源验收。

本机标准宽度重试命令（只生成合成夹具）：

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& D:/miniconda/envs/htfllib/python.exe system/verify_multi_group_head.py --output E:/mh_1107h --width 16 --svd-diagnostics 1 --upload-backing mmap
```

上述目录必须尚不存在。复验请使用新的独立目录；不覆盖已完成报告。本机较早的内存不足尝试及外置SSD不支持硬链接的脚本故障均保留记录；硬链接现在会回退到正常文件复制。失败产物清理被自动安全检查拒绝，拒绝信息只有 `blocked by policy`；详见报告中的明确目录，不宣称已清理。

主指标仍是固定归属模型的普通post-local全员测试样本量加权准确率，Final/最早Best/Last10/Last20保持原定义；额外列评价有头类型标签。合成检查不代表真实准确率提升。新增三组能分别检查参数范围、排除组外头和helper跨轮私有头的影响，不能把三项同时变化只解释为私有头收益。
