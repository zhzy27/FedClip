# 固定多模型、自动分组与 SVD 容量异构

新增 `FedMultiGroupProj --multi_svd` 可选路径。原同构 `CNN-512`、原预热/历史分组入口以及其他算法默认路径保留。本批正式配置为 K=3、无预热、首轮全部正常本地训练完成后以恢复完整权重的余弦均值自动分组，人数7/7/6，此后固定成员。另提供同一引擎的 K=1 全20人 Avg 对照。

## 入口与参数

| 参数 | 说明 |
| --- | --- |
| `--multi_svd` | 默认关闭；开启时要求 `-algo FedMultiGroupProj -m Decom_CNN-5-512` |
| `--multi_svd_ratios` | 默认 `.9,.37,.35,.25,.15`，有限、不同、位于 `(0,1]` 的秩比例 |
| `--multi_svd_assignment` | 可选N个容量档索引；默认 `client_id % len(ratios)`，同一客户端跨模型/跨轮固定 |
| `--multi_svd_diagnostics 1/0` | 默认1，记录截断/训练/global距离；0关闭额外起点评估，仍生成必需的恢复上传，不改变训练目标 |
| `-is_regular / -regular_lamda` | 新异构路径支持原 F 正则；首批显式 `1 / 1e-3`。旧同构仍是原CE目标 |

复用 `-nsm`、`--multi_group_sizes`、`--multi_group_mode`、`--multi_group_weight`、`--group_softmax_tau`、聚类seed/迭代、数据/训练/评价/裁剪参数。每次运行固定K和成员，不动态改变。SVD首版明确要求 `--multi_warmup_rounds 0 --multi_group_basis full_weights`，拒绝预热、其他分组依据、私有头、FedRep、Prox、分类头专项正则、meta等未支持组合。原同构路径的合法配置不受影响。

## 模型与每轮操作

母模型为原 `flcore.trainmodel.models.Hyper_CNN_512`，`ratio_LR=1.0`、通道16/32、FC 800→2000→512→100。公共seed生成一次完整初始化，再深拷贝为K份。它与旧低秩模型恢复后的参数名称/形状一致，包含共享 `fc3`，不重复计入 `base/head` 别名。

每客户端对每列独立复制该列母模型，调用原 `decom_larger_model(rate)`，设置实际 `ratio_LR` 并移至训练设备。原分解函数会创建新的CPU模块/Parameter，因此**分解和移设备之后重新建立无动量、无weight-decay的SGD**。原分解方法不自动更新 `ratio_LR`，显式更新该标记保证原 F 与恢复方法按实际低秩执行。

训练调用原 `clientTargetProj._objective()` 的 CE＋F，全部原可训练参数统一裁剪。没有Prox、私有分类头、额外微调、FedFew双层权重或CE部署选择。保存普通 post-local 低秩checkpoint，再在保护RNG的模型副本上调用原 `recover_larger_model()`，缓存去重完整参数的CPU张量用于上传。

恢复不重新SVD、不训练，不改变普通低秩模型/checkpoint。重复聚合读取使用同一份恢复上传缓存，逐客户端转到服务器设备；不把全部恢复模型同时放在GPU。起点恢复张量仅保留到该列训练结束，之后释放，不保存逐轮完整上传历史。

每列按原 `aggregate_group_updates` 独立计算，global-delta严格为 `recovered_post - full_server_download`，**包含SVD截断变化**。Projection仍为群内样本量参考、群外逐个投影、投影前余弦softmax、原epsilon和温度；Avg-only实际质量为1/0，全部客户端仍训练全部K列。

实际默认容量（原宽度模型，按去重参数计数）：

| rate | conv2 rank | fc1 rank | fc2 rank | 可训练参数 | 恢复完整参数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| .90 | 72 | 720 | 461 | 3,246,372 | 2,691,860 |
| .37 | 30 | 296 | 189 | 1,365,828 | 2,691,860 |
| .35 | 28 | 280 | 179 | 1,295,428 | 2,691,860 |
| .25 | 20 | 200 | 128 | 941,396 | 2,691,860 |
| .15 | 12 | 120 | 77 | 587,364 | 2,691,860 |

20人默认每档4人。rate不是参数保留比例，.90档甚至比完整参数量大；保留原容量规则，不为压缩率另改秩。

## 分组与配对文件

首轮20×3次训练结束后，每列在恢复完整参数上计算两两余弦，包含分类头，三份矩阵算术平均后调用原平衡聚类。没有减服务器参数，也没有比较U/V、测试成绩或标签名单。分组完成后立即执行首轮聚合，无额外预热/补训。

分组context新增完整的容量列表、client→档位映射、原F配置、恢复参数口径，另保留模型、初始化、数据、训练协议和分组依据指纹。旧同构文件、不同容量/数据/训练协议的文件明确拒绝。

三条正式命令中的K3 Projection原子发布本批分组；K3 Avg-only在自己完成首轮同样训练后读取同一文件（默认最多等待7200秒）。等待不重复训练、不推进随机流。输出记录当前首轮证据与源文件是否逐位一致，不静默重新分组。每次使用新的 `SVD_RUN_TAG` / 分组路径；已有文件不覆盖。

## 三条正式 COMMANDS

[multi_group_svd_commands.sh](../system/multi_group_svd_commands.sh) 中是同一个完整 `declare -a COMMANDS`，顺序为：

1. K=1、唯一组20人、SVD异构全员Avg；每人每轮5次数据遍历。
2. K=3、group_avg_only、7/7/6；每人每轮15次遍历。
3. K=3、group_projection_softmax、7/7/6、群内/外.50/.50、tau .2；每人每轮15次遍历。

均为现有 Cifar100/pat_20、20人/100类、seed0、全参与、lr .005、batch16、每模型local5、F .001、clip10、测试batch16、`-gr 100 -eg 1`，实际101次训练/聚合。GPU示例0/1/2可替换。不会生成或重新划分数据。

从服务器 `system/` 目录：

```bash
source multi_group_svd_commands.sh
printf '%s\n' "${COMMANDS[@]}"           # 仅打印
eval "${COMMANDS[2]} --multi_preview"   # 只审计已有数据与参数，不训练、不发布分组
```

在原 `run_now.sh` 中用 `source multi_group_svd_commands.sh` 替换其原COMMANDS定义，保留原后台调度循环。此文件本身不启动任务。也可事先设置 `SVD_DATA_ROOT`、`SVD_RUN_ROOT`、`SVD_GROUP_FILE`、`SVD_GROUP_WAIT`。GPU编号在三条命令的 `-did` 修改。

默认输出为 `system/multi_svd_runs/<本次tag>/`：三个实验各有独立 `checkpoints/`、`logs/`（及logs.zip）、`h5/`、`final_models/`，本批共享分组文件位于 `shared/`。日志ZIP没有模型张量。

## 指标、诊断与模型加载

主预测仅为 `Client_i_model_g(i).pt` 的普通post-local低秩模型。主曲线为同一轮全部客户端所选模型的 `sum(correct)/sum(test_samples)`；best是这条曲线的最大值及最早轮次。last10/last20按最后评价点，末轮强制评价，不先取各客户端历史best再平均。

原有 primary/client/group、微调前后、worst/std、聚合权重/投影及服务器几何诊断保留。新增：

- `capacity_manifest.json`：配置和实际各层秩、可训练/恢复参数量（首轮下载后即记录，最终manifest也保存）。
- `capacity_metrics.csv/json`、`member_summaries.json` 的capacities：各容量档按固定归属汇总的样本量加权pre/post及final/best/tail。
- `svd_diagnostics.csv/json`、`local_training`和逐轮`rounds.jsonl`：每客户端每列的 `svd_truncation_norm`、`local_training_recovered_norm`、`global_delta_norm`及平方值，另有CE/加权F均值、样本访问量、优化步数及耗时。关闭距离观察时不填伪零，manifest明确标记未观察。
- H5保存同一组表、主summary、实际模型类、容量协议、数据及初始化指纹；最终manifest给出真实模型配置和加载说明。

`Server_model_k.pt` 是聚合后的完整母模型；`Client_i_model_k.pt` 是普通post-local低秩预测checkpoint，记录实际capacity和结构。没有把恢复上传缓存导出为客户端预测模型。另保存K模型的优化器状态、固定分组、训练随机状态；不宣称支持断点续训。

```python
from utils.multi_group_svd import load_checkpoint
model, metadata = load_checkpoint('Client_0_model_0.pt', device='cpu')
# metadata['assigned_model_id']给出客户端固定使用的列；加载相应文件。
# model已为eval模式；函数会恢复加载操作消耗的随机状态。
```

额外评价仍保护参数、buffers、模块模式及Python/NumPy/Torch CPU/CUDA RNG；训练和实际SVD下载的变化正常保留。本实验只有普通训练，无第三阶段。

## 验收命令

从仓库根目录使用已有环境：

```bash
python -m unittest discover -s tests -p test_multi_group_svd.py -v
python system/verify_multi_group_svd.py --output system/temp/svd_acceptance_new
python system/verify_multi_group_warmup.py --output system/temp/svd_homogeneous_new --compat-only
python -m unittest discover -s tests -p test_multi_group.py -v
python -m unittest discover -s tests -p test_multi_group_warmup.py -v
python -m unittest discover -s tests -p test_target_projection.py -v
```

验收工具只创建自己的合成数据目录。包含两轮K3两模式、观察关闭精确对照、K1三轮稀疏评价；保存并重载模型检查后默认清理合成模型文件，保留轻量报告。`--keep-models`可保留模型，`--pair-only`只运行两个K3配对用例。详见本批 [验收记录](multi_group_svd_acceptance.md)。合成检查不能说明真实数据准确率收益。
