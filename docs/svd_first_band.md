# K1 全量 Avg：第一新增秩段修复与 R1–R10 容量对照

新增`--multi_svd_spectral_mode first_band`；默认仍是`none`。已有`none`、`single_point`、`next_tier`、`all_tail`的公式和默认行为保留。服务器始终对恢复完整上传按原训练样本数做`full_avg`。

## 第一段的定义

每层按实际分配容量得到整数秩，取最小及次小不同值`k,b`。当前原五档对应：

| 层 | k | b | 改动范围（1基） |
| --- | ---: | ---: | --- |
| conv2 | 12 | 20 | 13–20 |
| fc1 | 120 | 200 | 121–200 |
| fc2 | 77 | 128 | 78–128 |

R1完整走原始下发；从R2起，复用原下载SVD的完整谱及左右向量，使用已有`all_tail`的样本质量和前向单调约束计算第一段：`q_j=sum(n_i for r_i>=j)/sum(n_i)`，`corrected_j=min(raw_j/q_j, corrected_(j-1))`。实现复用原纯谱核，只保留它的第一段；后续秩的计算不会反向影响第一段。**前k项和b之后全部逐元素保留原谱**，不重新排序、不拟合、不使用1.5倍上限。零保持零。各容量截取同一条完整修复谱的前缀，最低档实际因子与同母模型普通SVD完全一致。

原五档、等训练样本量时第一段贡献质量为16/20，理论倍率1.25；不等样本量按实际分片计数，不把访问次数或参数量当权重。单调约束可能使实际倍率低于理论倍率。每层必须有两个不同的实际整数秩；不支持的配置明确失败。

`max_gain`参数在first_band中不生效，manifest记录`max_gain=null`、`max_gain_parameter_effective=false`。每个训练任务仍为原来的5 epoch、一次batch前向/反向/裁剪/SGD，不增加原型、反馈或额外微调。R2以后H与H_first_band各自的服务器可以走出不同轨迹，同母模型原谱即时对照使用该分支**当轮**的SVD，不误用H的母模型。

## 本地 epoch 观察

新增默认关闭的参数：

```text
--multi_svd_local_epoch_diagnostics 0|1
--multi_svd_local_epoch_client_ids 3,8,13,18,0,5,10,15
```

要求已有`--multi_svd_five_diagnostics 1`，仅支持当前K1、全参与、共享头、joint、无原型的SVD full_avg路径；客户端ID检查非空、重复和越界。开启但未指定ID时记录所有客户端。关闭时不能传ID让其被静默忽略。

只额外评价epoch1至`local_epochs-1`。epoch0、最终epoch直接复用已有下载后和post-local计数，不重复遍历。独立、不打乱、不drop_last的评价loader在`eval/no_grad`下只计算纯CE、正确数和样本数，恢复参数、buffers、各模块模式及实际设备/Python/NumPy/Torch随机状态，保留当前梯度及优化器引用。测试数据只用于报告。

训练过程`train_seconds`扣除新增的中间epoch评价耗时，`local_epoch_diagnostic_seconds`及`wall_seconds_including_local_epoch_diagnostics`另列；轮次总耗时仍包含实际观察成本。额外前向样本量不记入SGD访问量或训练epoch。

本批观察8人：原`.25`固定成员`3,8,13,18`，原`.90`固定成员`0,5,10,15`。这些固定成员在S120/S200中仍按原分配分组，不因为实际容量统一而改变成员。`S120`/`S200`按fc1秩命名，但conv2、fc2也同时使用相应整网容量；二者收益不能仅归因于fc1。

## 日志位置与含义

四组独立根目录：`system/svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/{S120,S200,H,H_first_band}/`。各有`checkpoints/`、`logs/`、`h5/`、`final_models/`，具体带运行名的路径以manifest为准。命令数组只定义，不自动执行；GPU为0/1/2/3，`-did`靠前。

- `primary_metrics`/`primary_summary`：全20人普通post-local测试样本量加权主指标；Final、同轮总体曲线最早Best、Last10/20。
- `five_client_endpoints.csv/json`：每轮全20人的epoch0/最终epoch train/test准确率、纯CE、正确数、样本数；first_band另保留同母模型原谱下载对照。
- `five_cohort_metrics.csv/json`：全体、C0和原五档固定成员的加权汇总。
- `five_epoch_client_endpoints.csv/json`：选定客户端每轮完整epoch0→5曲线，`local_epoch`和`observation_origin`明确区分复用端点与新增观察。
- `five_epoch_cohort_metrics.csv/json`：`selected_clients`、C0（若选中）及`selected_fixed_tier_k`汇总。后者只包含显式观察的成员，成员ID始终随记录保存，不能把未观察的客户端当成已测量。
- `five_diagnostic_costs.csv/json`：中间epoch只读前向样本数及耗时；`local_training`记录原SGD步数、样本访问和扣除观察后的耗时。
- `svd_first_band_ranks.csv/json`：每个实改秩的层、k/b、贡献者ID/人数、训练样本总质量q、原谱、修复谱、理论/实际倍率和单调约束。R1无逐秩修复，`svd_spectral_downloads`说明原路径跳过；JSONL持续记录。
- `five_layer_spectra`、`five_C0_fixed_slots`及`five_paired_gains`保留原来的谱、槽位变化和同母模型即时对照。

上述新增表同步写入已有H5（同名`*_json`数据集）及manifest。主指标没有混入中间epoch观察。低秩预测模型和完整服务器模型沿用原保存、加载方式。

## 启动与逐轮比较

完整四条数组在`system/svd_first_band_commands.sh`，从服务器`system/`使用。公共配置：Cifar100/pat_20、20客户端/K1全参与、seed0、`.90`恢复母模型初始化、SGD.005/batch16/local5/F.001/clip10、无原型、逐轮评价；`-gr 9`确实为R1–R10。S120全部`.15`，S200全部`.25`，H及H_first_band使用原五档及id%5分配。

四组完成后从`system/`运行只读比较器：

```bash
python compare_svd_first_band.py --S120 svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/S120 --S200 svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/S200 --H svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/H --H_first_band svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/H_first_band --output-dir svd_first_band_runs/pat20_seed0_init090_R10_20261117_v1/comparisons
```

生成CSV/JSON和简表：`H_first_band−H`与`S200−S120`的逐轮微调前差、微调后差和`post差−pre差`，并保留纯CE差。内部准确率fraction，简表差值为百分点。同母模型原谱对照单独标识，不与跨分支服务器轨迹比较混淆。若同一根目录有多个运行报告，需指定唯一的具体metrics目录，避免混用结果。

## 验收范围

`verify_svd_first_band.py`使用独立合成分片，通过真实main.py运行20客户端两轮四组、关闭epoch观察对照、冻结b49c54e旧训练器/客户端/服务器对照和旧all_tail回归。检查参数、上传、Avg、批次及RNG，逐秩范围、完整曲线、预算、CSV/JSON/H5及21模型重载；不读取或改写真实pat_20。结果见`docs/svd_first_band_acceptance.json`。合成验收证明实现，不代表真实测试收益。正式实验由用户启动。
