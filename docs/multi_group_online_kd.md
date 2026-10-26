# 多群组SVD在线跨组蒸馏

这是从初始化开始的新联邦训练实验，不读取R201检查点。原离线入口 `offline_distillation.py` 保持独立。默认 `--multi_kd_mode none`，旧同构、SVD、多群组、单目标及FedFew入口的算法路径不变。

## 时序与教师来源

每轮全部客户端下载全部K个服务器模型，按各自原容量完成原SVD下发。服务器按已验证的固定成员映射给出真实所属模型ID，不使用client_id取模。

首轮所有模型仍只做CE＋原F、各local_epochs；恢复完整上传、计算完整权重余弦、固定分组、执行首轮原群组聚合。即使提供现成分组文件，首轮也不启用KD。第二轮（loop=1，完成轮次=2）起：

1. 在该客户端任一模型SGD开始前，核对全部低秩下载状态与训练数据指纹。
2. 在受随机状态保护的作用域内，复制外部K−1个**本轮下载后低秩模型**，eval/no_grad前向原训练图，按原样本索引缓存原始logits；临时副本冻结，原模型不冻结。
3. 按旧model_id顺序训练全部K个模型，各自原SGD、local_epochs、shuffle、F正则及裁剪。只给所属模型加KD；学生原forward结果复用于CE/KD，不增加一次学生前向。
4. 每列按原方法恢复完整上传、调用原 `aggregate_group_updates`。固定成员部署各自普通post-local低秩模型。

缓存只驻留该客户端本轮训练期间的CPU内存，完成后释放，不跨轮或跨客户端复用。缓存指纹绑定客户端、下载轮次、归属、数据、教师状态、标签、原始logits及样本索引。下载状态被提前更新、缓存来源过期或batch标签/索引错配时明确失败。

原B/C等模型在生成缓存后继续正常训练；“教师不变”指缓存生成期间的冻结副本及本轮固定目标，不代表停止这些模型自己的本地SGD任务。临时副本和缓存有额外内存/计算成本；维护的本地训练模型数量仍为K，不能据此宣称峰值内存或通信预算不变。

## 两种损失

共同目标：CE＋原加权F＋λKD·T²·KL(q∥p学生)。原F直接复用 `clientTargetProj._objective`；不改变作用层、权重或SGD设置。KL按类别求和、样本平均，λ和T²各乘一次。

- `full`：每个外部教师先做softmax(logits/T)，再均匀平均概率。没有平均logits。
- `not_true`：对每张图，先分别删除学生及每个教师的真实类别logit，对各自剩余C−1类分别做温度softmax，再平均教师概率。没有先平均完整概率再删除，也没有引入完整DKD的TCKD、warmup、门控或其他机制。

教师目标完全detach，没有KL+DP选择、A₀回退、EMA、正确性门控或测试信息。`not_true`项对被删除的真实类别logit的**直接**梯度为零；CE、共享参数以及其他logit变化仍可能影响真实类别预测，不能解释为完全不影响真实类别知识。

非真类公式参考 [mdistiller/DKD.py的NCKD部分](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py)。实现按本批用户公式直接删除logit，不导入其训练框架/其他损失；是跨群组适配，不声称论文完整复现。

## 参数与边界

|参数|默认|含义|
|---|---:|---|
|`--multi_kd_mode`|none|none/full/not_true|
|`--multi_kd_lambda`|.1|有限非负系数|
|`--multi_kd_temperature`|2|有限正温度；检查加权T²有限|
|`--multi_kd_teacher_batch_size`|16|教师及探针前向batch|
|`--multi_kd_observe_gap`|1|按完成轮次的探针观察间隔，末轮包括在内|
|`--multi_kd_probe_samples`|0|每客户端固定原训练探针数量；0关闭|

首版要求FedMultiGroupProj、multi_svd、共享头、joint、零Prox/头L2、无预热、full_weights首轮分组。沿用现有全参与、数据/模型、固定K和分组合法性约束。K、N、容量、数据目录、训练预算、聚合质量/温度等继续使用已有参数。正式首批只配置K3原五档容量的Projection。

- K1：有效KD=0，不生成缓存/教师前向。
- K2：一个外部教师。
- K≥3：均匀平均K−1个外部教师。
- none、λ=0、首轮：训练走原CE＋F分支，不引入额外教师/探针前向。开启full/not_true但系数0会记录明确的未启用原因。

新代码不修改现有联邦backend，不套用离线脚本的确定性设置，也不重设训练随机流。教师/探针使用私有副本与已有Python、NumPy、Torch CPU/CUDA状态保护。indexed训练loader保留原sampler、batch、shuffle、drop_last=False和workers=0；不重排模型、不额外遍历sampler，不强制不同列使用相同shuffle。

## 分组配对与启动命令

完整定义见 [multi_group_kd_commands.sh](../system/multi_group_kd_commands.sh)。从服务器 `system/`：

```bash
export MULTI_KD_RUN_TAG=original_capacity_K3_KD_T2_lam0p1_seed0
export MULTI_KD_DATA_ROOT=../dataset
source multi_group_kd_commands.sh
declare -p COMMANDS
```

脚本只定义数组，不启动任务。默认COMMANDS只有两条：full/GPU1、not_true/GPU2。完整公共参数显式展开：pat_20、N20/K3、7/7/6、原秩.9/.37/.35/.25/.15、low_rank_recovered初始化秩.15、seed0、SGD .005、batch16、每模型5epoch、F .001、clip10、gr100实际101轮、逐轮评价、群内/外.50/.50、聚合温度.20、KD温度2/系数.1；探针32张/客户端、每轮观察。

把文件内容放进原调度脚本的配置区，或在原 `declare -a COMMANDS` 位置改为 `source multi_group_kd_commands.sh`，避免随后被旧数组覆盖。重复运行使用新tag；同一shell中重新配置时也应清空或重新指定派生的 `MULTI_KD_RUN_ROOT/MULTI_KD_GROUP_FILE`。

full在首轮原子发布本批分组，not_true完成同样首轮训练后有限等待并读取该文件；不重复训练。KD配置独立记录，不进入首轮分组兼容指纹。

若要复用**已确认协议兼容**的原Projection分组文件：

```bash
export MULTI_KD_GROUP_FILE=/absolute/path/to/original/fixed_groups.json
export MULTI_KD_REUSE_GROUP_FILE=1
source multi_group_kd_commands.sh
```

此时两条命令及可选基线均读取同一文件，仍从初始化训练首轮且不启用KD。文件的容量、初始化、数据、N/K、训练协议仍由原校验核对；不兼容不能绕过。未配置复用时默认发布到本批独立新路径，拒绝覆盖旧文件。

`OPTIONAL_BASELINE_COMMANDS`单独放一条none/GPU0，不在默认调度数组。若无法核对旧对照，交由用户决定是否运行。本机没有旧正式Projection全量日志，未自动认定旧结果可复用。使用旧201轮结果时只能截取其**前101轮**，核对数据/初始化/容量/训练与backend/分组；Final取第101次、Best在这101次同轮全员均值曲线上取、Last10/20也只取该窗口，不能拿201轮Best比较。

可选基线同样读取共享分组文件：可在full发布后运行，也可并行有限等待；若只运行基线，需先提供已有兼容分组文件，不能等待一个永远不会启动的发布者。

## 输出与诊断

模式、λ和T自动加入实验名，脚本另设独立checkpoint/log/H5/final目录，根目录默认 `system/multi_kd_runs/$MULTI_KD_RUN_TAG/`。日志ZIP沿用现有独立日志设施，不混入预测checkpoint。

每个运行的metrics/log目录增加：

- `online_kd.jsonl`：每客户端本轮训练完成立即追加，非仅结束导出；包含所有列训练记录与探针记录。
- `online_kd_training.csv/json`：请求/有效模式与系数、未启用原因、归属及教师ID、下载轮次、低秩教师/完整母模型/缓存/数据指纹、按样本访问量平均CE/加权F/rawKL/加权KD/总损失、步数/访问量与教师缓存成本。
- `online_kd_probes.csv/json`：固定探针ID及指纹，训练前/后逐模型对的JS（T=1，nats）、预测一致率、全模型一致率、观察耗时和前向样本数。探针用独立局部Generator，仅诊断，不门控或调参。

H5包含上述两表和`online_kd_protocol_json`。运行与最终manifest、完整服务器及低秩客户端模型的元数据记录online_kd协议。教师缓存只在内存，不额外保存每轮模型/logits历史。

主指标继续为全20人固定所属模型普通post-local的sum(correct)/sum(test_samples)，保留下载后/训练前、群组、容量档、逐成员以及Final/最早Best/Last10/20。没有新增本地训练阶段。每客户端每轮仍K×local_epochs：首批3×5=15次数据遍历；教师/探针前向与SGD成本分开，不计成额外SGD步。

## 验收

```bash
# 仓库根目录
python -m unittest discover -s tests -p test_multi_group_kd.py -v
python system/verify_multi_group_kd.py --output system/temp/multi_kd_acceptance_NEW
```

验收工具调用真实main入口，隔离生成小型合成分片，20客户端、原宽度SVD CNN、K3、每模型5epoch、两轮，验证原e1e7f61训练参照、零KD、full、not_true及探针关闭；核对原聚合核、模型重载、CSV/JSON/H5/JSONL及随机状态。参照需要Git对象e1e7f61（未将参照算法切换到新KD实现）。默认验证后仅清理自己独立目录内的合成模型二进制，保留可读报告；`--keep-models`可保留。

已完成的合成目录可加 `--audit-only` 重读并核对各格式、参数/RNG记录及缓存来源，不再次训练。

实际验收结果见 [multi_group_online_kd_acceptance.md](multi_group_online_kd_acceptance.md)。本地没有真实pat_20数据；合成通过仅验证实现，不用于声称真实准确率改善。没有推送、改环境或启动正式101轮训练。
