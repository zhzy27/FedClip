# 自然尺度类别原型 MSE

基于 `43eb806` 扩展群组原型。新增 `--multi_proto_target_type fixed_norm/raw_mean`，默认 `fixed_norm`。原实验命令保持不变，原训练、分组、恢复上传、群组聚合核及主指标不变。

## 含义与范围

`raw_mean` 仅允许 `multi_proto_mode=group_priority`、`multi_proto_loss=mse`，仍要求当前全参与 SVD、多群组、无预热、首轮 full_weights 分组、共享头、joint、无 KD/Prox/FedRep/头 L2。两种特征位置均可用，首批命令固定为 `base_output`：最后一次 ReLU 前的实际 base 输出。

默认 fixed_norm 保留两处单位化和旧近零处理。raw_mean 同时取消：

1. 客户端不单位化单样本特征，直接用 float64 累加各类别原始特征，上传 int64 真实标签计数。
2. 服务器按群内优先来源合并 sum/count 后直接保留类别均值，不再单位化、不乘 r、不按容量或客户端范数加权。

群内存在某类样本时只使用群内；否则使用组外；全员无样本才无效。有样本但特征正负抵消、均值为零或近零时仍有效，并保留群内来源，不另选组外。

每个学生用自己的模型列库计算 `F.mse_loss(raw_features[valid], raw_class_mean[y[valid]].detach(), reduction='mean')`。学生与目标均不单位化、不加隐藏 1/2；整个 batch 无有效类别时返回可微零。单次原 forward 同时提供 logits 与特征，辅助梯度经骨干回传；分类头仍由原 CE 训练。原 CE、加权 F、SGD 和裁剪保持不变。

**这不只是改变目标范数。** 从单位特征均值改为原始特征均值后，大范数样本的贡献增大，类别原型方向也可能变化。本实验检验完整的自然尺度原型方案。

`multi_proto_target_norm` 在 raw_mean 下不参与计算，新命令不传；非默认值报错。完整请求配置里该旧参数仍可能显示 argparse 默认值 1，但语义明确的 `target_norm / aux_target_norm / aux_effective_target_norm / actual_fixed_target_norm` 均为 null，不能将请求默认值视为实际定长目标。

模式关闭或 λ=0 时跳过辅助采集、特征 hook 与通信，走原 CE＋F；raw_mean 的合法零系数组合仍是 group_priority＋mse。首轮不采集、不约束，分组后从配置的第2轮开始：全体下载／SVD完成 → 全体统计 → 合并与下发 → 任意 SGD。轮内目标固定，下一轮重建，不继承缺类目标。

## 类型与兼容性

| 类型 | statistics_representation | prototype_representation |
|---|---|---|
| fixed_norm | unit_feature_class_sum | unit_class_direction |
| raw_mean | raw_feature_class_sum | raw_class_mean |

原 fixed_norm 的 canonical packet 不插入新默认键，保持旧内容和指纹；缺失类型标记仅解释为 legacy fixed_norm。raw_mean 的统计包与库必须包含上述完整类型标记，服务端接收、合并、客户端接收、训练 Session 和保存最新库均检查一致性。不能把旧单位方向库当作 raw_mean 使用，也不能在保存时把 raw_mean 标成旧库。

预测 checkpoint 的参数名称、forward 和加载方式不变。最终库仍为 `Prototype_library.packet`，写入 checkpoint／最终模型目录供追溯；预测不需要该文件。类型信息同时写入模型 metadata 与 manifest。非有限特征、统计、目标、转换到学生 dtype 后的目标或损失均失败，不跳过。

## 日志

沿用 CSV/JSON、H5、持续 `class_prototypes.jsonl`、manifest 和日志 ZIP。无逐样本特征或模型副本进入日志 ZIP。

- `prototype_target_type / prototype_effective_target_type`：请求与当前生效的目标类型。
- `statistics_representation / prototype_representation`：统计与库的实际含义。
- `prototype_sources` 的 `prototype_l2_norm`：库中该类原型真实 L2 范数；raw_mean 可大于1，也可为0。
- raw_mean 的 `raw_class_mean_norm`、`mean_is_exact_zero / mean_is_near_zero`；有样本且为0时 `valid=true`、`invalid_reason=null`。无样本为 `no_training_samples`。
- raw_mean 的 `pre_normalization_norm=null`，因为此时没有最终单位化；`norm_unit` 明确为原始类别均值 L2，**不是单位特征的方向集中度**。
- raw_mean 逐客户端／容量的 `local_mean_norm` 同样是原始类别均值尺度，带独立单位说明，不能用“越接近1越紧致”解释。
- 训练过程原始／加权 MSE、CE、加权 F、总损失、有效样本比例与原特征尺度／裁剪诊断保留。
- 新增 `training_actual_target_l2_mean`：实际 detach 目标在有效 SGD 样本访问上的加权平均范数；`training_target_observed_valid_sample_visits` 记录分母。不是训练结束后的全数据评价；关闭诊断时为 null/0。

采集前向、合并时间、统计上传和原型下发 packet 字节仍单独报告，不混入 SGD epoch 或样本访问量。原型记录的来源轮次、下载模型及分组指纹保持可追溯。

## 两条新命令

从服务器 `system/` 使用 [multi_group_proto_raw_mean_commands.sh](../system/multi_group_proto_raw_mean_commands.sh)。完整数组含 λ=.1／1 两条命令，GPU 示例0／1可替换；原五档容量、恢复初始化rank=.15、K3、7/7/6、β=.5、τ=.2、pat20、seed0、lr=.005、batch16、local5、F=.001、clip10、gr100（实际101次）均保持一致。

该文件只定义变量与 `COMMANDS`，不执行训练。只读取上一批八组共享的分组文件，不发布新文件，不独立重新分组。分组不含本辅助损失配置，因为首轮完全相同；原有数据、初始化、容量、训练及分组协议兼容性仍强制检查。

本机无法推断服务器上上一批分组 JSON 的实际路径，使用显式必填变量：

```bash
# 如果同一调度配置已定义上一批的 PROTO_LOSS_GROUP_FILE，会自动沿用它。
# 否则将下列值设为上一批八组实际使用的现有文件，不是新输出路径。
export PROTO_RAW_GROUP_FILE=/absolute/path/to/the_previous_eight_runs_fixed_groups.json
source ./multi_group_proto_raw_mean_commands.sh
# 交给原 run_now.sh 调度器；source 本身不会启动训练。
```

若两者都未提供，脚本立即报错，不猜测文件、自动搜寻或创建新分组。默认等待0，保持现有缺失文件报错规则；可显式设置 `PROTO_RAW_GROUP_WAIT` 采用已有有限等待机制。

输出根为 `multi_proto_raw_mean_runs/$PROTO_RAW_RUN_TAG/`，两个独立子目录 `raw_mean_lam0p1 / raw_mean_lam1`，各含 checkpoints、logs、h5、final_models。旧 `multi_group_proto_commands.sh` 和 `multi_group_proto_loss_commands.sh` 保留不变。

## 验收工具

```powershell
# 仓库根目录；使用现有环境
& D:\miniconda\envs\htfllib\python.exe -m unittest discover -s tests -p test_multi_group_proto_raw_mean.py -v
& D:\miniconda\envs\htfllib\python.exe system/verify_multi_group_proto_raw_mean.py --output system/temp/multi_proto_raw_mean_acceptance_NEW --jobs 2
```

服务器在 `system/` 可执行 `python verify_multi_group_proto_raw_mean.py --output temp/multi_proto_raw_mean_acceptance_NEW --jobs 2`。工具仅在新建独立目录生成微型合成数据，不读取／重划真实分片。固定 Git `43eb806` 作为原算法参照，实际调用 main 进行 N20/K3 两轮低秩训练，独立复算原始均值并验证模型重载。验收后默认清理自己创建的模型二进制，保留日志和收据。

见 [验收报告](multi_group_proto_raw_mean_acceptance.md)。合成验收不代表真实准确率收益；本次不启动正式实验、不推送、不修改环境或绘图。
