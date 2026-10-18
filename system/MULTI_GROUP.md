# 固定K模型、固定多群组入口

独立算法 `-algo FedMultiGroupProj`，不改变 FedFewS 或 FedTargetProj。
首版完整同构CNN-512，CIFAR-100/100类/RGB32，包含共享分类器。
不做SVD/容量分配/F/Prox/私有头/反馈/模型CE部署选择/STCH双层权重。
仅此新算法的CLI缺省设置为CNN-512/Cifar100/100类/20人、local5、batch16、lr.005、gr100、eg1、pat/cpc20；
显式命令行值仍可覆盖，数据root/subdir必须明确提供。其他算法的默认值不变。

每轮下载K模型→每客户端按模型列顺序各训练local_epochs→上传N×K份普通post-local结果
→分别聚合K列。第一轮结束时分组，此后固定；不增加预热或第三阶段训练。
部署模型仅由固定成员映射 `g(i)` 决定。优化器持有原Parameter对象，下载不重建优化器。

## 启动与参数

从 `system/` 目录：`source multi_group_commands.sh` **只定义**两个完整COMMANDS数组项，
不执行训练。`PREVIEW_COMMANDS`追加`--multi_preview`，只审计已有分片/打印生效设置。
GPU默认0/1，可用`MULTI_GPU_PROJ`/`MULTI_GPU_AVG`替换。

公共设置：pat_20、20人、K3、seed0、lr.005、batch16、每模型5 epoch、gr100、eg1、
clip10、测试batch16、群内.50/群外.50、tau.2。实际101次训练/聚合。
每客户端每轮三个不同模型各遍历5次数据，共15次；全员Avg旧入口只有5次。

| 参数 | 缺省/含义 |
|---|---|
| `-nsm/--num_server_models` | 3，1≤K≤N，运行内固定 |
| `--multi_group_mode` | `group_projection_softmax` 或 `group_avg_only` |
| `--multi_group_sizes` | 省略按N/K均衡，如20/3→7,7,6；显式K个正整数之和必须为N |
| `--multi_group_file` | 读取已有fixed_groups.json；核对数据/初始化/N/K/容量/训练协议及映射指纹 |
| `--multi_group_weight` | .5，有限[0,1]，只在Projection模式控制质量；avg_only实际群内1/群外0 |
| `--group_softmax_tau` | .2，有限正数，计算与日志同一实参；不使用其他算法temperature |
| `--multi_cluster_seed` | 0，聚类独立随机流，不消耗训练RNG |
| `--multi_cluster_restarts` | 8，正整数 |
| `--multi_cluster_iterations` | 50，正整数上限，记录是否收敛 |
| `--multi_no_pre_eval` | 仅关闭下载后观察；不改变本地训练、上传或随机状态 |
| `--multi_preview` | 数据审计与配置预览，不构造模型、不训练 |

N/seed/lr/batch/local_epochs/gr/eg/clip/test batch使用现有公共参数。
clip省略为10、显式0关闭；gr≥0实际执行gr+1次。评价loop%eg=0或末轮。
last10/20始终是最后10/20次**评价**，summary同时列实际loop和完成训练次数。
仅支持全参与、fresh run、共享头、joint CE；暂不支持resume/drop/slow/衰减等组合，明确报错。
公共数据审计与前一版同构入口相同：精确root+dataset+subdir、格式策略、真实config、
pat类别信息/Dirichlet alpha、文件与解码指纹，不生成/改写真实数据。

## 分组算法与复用

首轮第k列的更新为`W_post[i,k]-S_download[k]`。
逐列计算N×N更新余弦，最后取K列算术平均。
逐参数分块减法在原dtype完成，float64 Gram归约，临时内存限制在N×chunk_size。
不比较完整权重，不把不同模型基座之差混进更新。

使用平衡核球面聚类：独立NumPy Generator初始化中心，按群体平均核向量计算球面评分，
Hungarian分配到容量展开后的槽位，再更新中心；多次重启按目标值选择。
目标值差≤1e-12时按规范成员列表字典序决定；相同分数的分配使用固定排序输入的SciPy确定性实现。
保存SciPy/NumPy版本、种子、重启及迭代/收敛情况。零更新的余弦为0，零中心评分为0；
全零/全相似/并列时仍确定性满足容量。编号按人数降序、再按成员字典序，成员内部升序。
这是明确记录的首版启发式分组，不声称全局最优。

`fixed_groups.json`包含全部成员、client_to_model、原始相似矩阵/逐列矩阵、
算法参数、首轮更新指纹、数据和初始化指纹。只使用训练更新，不读取测试计数、准确率或标签重叠。
数据标签统计用于数据审计，绝不传入聚类函数。

推荐先得到第一种模式的分组文件，再设置
`MULTI_FIXED_GROUPS=/absolute/path/logs/run_<id>/fixed_groups.json`，重新source命令文件运行对照。
使用文件时保留原映射，不重新聚类；另外记录本次首轮上传/相似度与来源是否逐位一致。
跨GPU的浮点差异只作报告，不偷偷换组。若两个模式都自动分组，必须核对
`group_fingerprint`与数据/初始化指纹一致，不能只看人数都是7/7/6。

## 聚合、评价与产物

每个模型k直接调用原`aggregate_group_updates`，只读取第k列普通上传。
群内按样本数构造global-delta参考，群外逐个投影、按投影前余弦softmax评分，
epsilon/零向量/归一化完全沿用旧核。所有K个结果构造后才写回服务器。
avg_only同样训练全部N×K个模型，聚合时群外质量为0，不做模型选择或减少任务。
K=1时Projection若配置正群外质量明确拒绝，beta=1或avg_only可运行。

每评价点保存全部客户端×全部模型的下载后训练前及普通训练后correct/samples。
首轮先保存计数，训练更新确定分组后按g(i)提取对应模型。
主指标为全员对应普通post-local模型的sum(correct)/sum(samples)，不按测试选模型。
每组同样使用测试样本量加权，同时记录宏均值/worst/std。
同轮微调增益为同一次下载的post-pre；没有额外训练。
观察恢复模型全部state_dict、模块train/eval标记及Python/NumPy/Torch CPU/CUDA随机状态。
真实训练流正常保留。当前CNN无buffers；下发与聚合仅针对named_parameters。

日志位于独立 `--run_log_dir/run_<id>/`，过程日志ZIP不包含模型：

- `fixed_groups.json`、`group_similarity.json`、（复用时）`group_reuse_verification.json`
- `primary_metrics.csv/json`、`primary_summary.json`、`member_summaries.json`
- `client_metrics.csv/json`（每客户端/模型、pre/post、归属与是否对应主指标）
- `local_training.csv/json`（每个真实任务CE/样本访问/步数/耗时）
- `aggregation_metrics.csv/json`、`aggregation_weights.csv/json`（群质量/参考/冲突/投影/矩阵）
- `server_geometry.csv/json`（每轮聚合后模型间全参数距离/余弦）
- `round_budget.csv/json`、持续写入的`rounds.jsonl`、`data_manifest.json`、`manifest.json`
- 结构化H5同时镜像到日志目录；与JSON/CSV使用同一计数与summary。

模型分别保存并导出到独立最终目录：`Server_model_k.pt`（聚合后）、
`Client_i_model_k.pt`（最后普通post-local）、`MultiGroup_rng.pt`、`fixed_groups.json`。
主指标对应`Client_i_model_g(i).pt`，不是最终服务器模型。
每文件包含实际CNN构造参数、schema、列ID/客户端ID/归属及state_dict；客户端另含对应优化器状态。
加载示例：

```python
from types import SimpleNamespace
import torch
from common.models import create_full_model
p = torch.load('Client_0_model_0.pt', map_location='cpu')
m = create_full_model(SimpleNamespace(model_family=p['model_family'], num_classes=p['model_kwargs']['out_dim']))
m.load_state_dict(p['state_dict'], strict=True)
m.eval()
```

列ID须读取fixed_groups.json，不能默认C0总属于模型0。
已有Avg/Few/单组命令保留，不混入这些新参数。

## 短程验收命令

```bash
# 合成数据只写入独立验收目录；不会读取/改写正式dataset。
python verify_multi_group.py --output ../fedfews_native_runs/multi_group_acceptance
python -m unittest discover -s ../tests -p test_multi_group.py -v
```

真实数据完整实验由用户自行启动；合成检查不说明准确率改善。
