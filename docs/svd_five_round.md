# K1 full Avg: S / L / H / H+ 五轮机制对照

仅显式开启的新路径。`none/single_point/next_tier` 的原规则、旧命令和聚合核不变；没有原型、蒸馏或额外SGD。四组均用 seed0 生成的同一个 `.90` 原生低秩初始化，原恢复函数得到完整母模型。`-gr 4 -eg 1` 是 R1–R5，不是每轮只训练一个epoch：每客户端每轮仍训练5 epoch。

| 组别 | 实际下载容量 | 谱修复 |
| --- | --- | --- |
| S | 全部 .15 | none |
| L | 全部 .90 | none |
| H | .90/.37/.35/.25/.15，id % 5 | none |
| H+ | 与H相同 | all_tail，从R2开始 |

## all_tail

`--multi_svd_spectral_mode all_tail` 只支持当前 K1、`multi_avg_proto_ablation + multi_svd`、全参与、共享头/joint、`group_avg_only + full_avg`、无其他辅助训练模块。边界来自实际分配容量的各层整数秩；质量来自审计后的原客户端训练样本数，与服务器Avg一致。

对最低秩之后、最大客户端秩以内的每个一基秩j，`q_j=sum(n_i for rank_i>=j)/sum(n_i)`，候选为 `sigma_j/q_j`；顺序取 `min(candidate, corrected_previous)`。最低秩前缀、原零值、最大秩以外的部分不变。只在原下载SVD的S上操作，沿用原U/V并用平方根平衡构造因子。服务器矩阵不改、不截断；聚合仍是原样本量完整矩阵Avg。

本配置实际秩：conv2=12/20/28/30/72；fc1=120/200/280/296/720；fc2=77/128/179/189/461。按实测训练样本量算q；只有等样本量时，fc1四段理论倍率为1.25、5/3、2.5、5。非递增约束可能降低实际倍率。**不使用旧1.5上限**；日志将实际max_gain标为null，旧参数请求值仍在配置中，policy明确该参数不生效。R1完全不修复。

## 独立诊断开关

`--multi_svd_five_diagnostics 1` 开启全轮观察，默认0。`--multi_svd_fixed_cohort_ratios .90,.37,.35,.25,.15` 定义固定比较群：`client_id % 5`，因此S/L也按与H相同的客户端集合汇总，不按S/L实际只有一档来分组。开启本协议要求每轮评价。

| 文件（CSV/JSON；同名H5 JSON数据集） | 内容 |
| --- | --- |
| five_client_endpoints | 每客户端download/post_local的train/test纯CE、correct/samples；H+另有同母模型原谱即时对照 |
| five_cohort_metrics | all、C0、固定五组，均按各自数据样本量加权 |
| five_paired_gains | 同轮post−pre；H+同母模型修复−原谱的即时差值 |
| five_layer_spectra | 三层全逐秩值、所有固定容量边界的相邻比、每段sigma平方能量 |
| five_C0_fixed_slots | C0原始下载因子槽位的UV块变化、初始/结束块范数、绝对/相对变化 |
| five_diagnostic_costs | 独立前向样本访问和耗时，绝不计入SGD遍历量 |
| svd_all_tail_ranks（H+） | 每轮/客户端/层/秩的贡献质量、理论倍率、实际倍率、顺序约束标记；避免把整个尾部塞入一个CSV字段 |

谱每行包含一个stage/client/layer的完整`singular_values`数组，下标j−1对应一基秩j；CSV以JSON数组保存，避免为每个秩重复元信息。stage为server_original_download（下载原SVD谱）、client_actual_download（所有客户端恢复矩阵的实测谱）、C0_post_local、server_post_aggregation。后三类在只读float64 CPU副本上重新计算谱；**这些观察SVD与训练下载每层一次SVD分别计成本**。

C0槽位变化按本次下载固定槽位计算，矩阵乘积用float64观察值；与训练后重新SVD的谱分开命名。各块训练后未必正交，不能把块范数平方相加当作谱总能量。

同轮实际恢复矩阵哈希完全相同时，下载谱观察可以复用已算结果，记录`diagnostic_cache_hit`和矩阵指纹；仅缓存当前轮的谱向量。该复用不参与模型下发或训练，实际新增观察SVD次数按缓存命中情况计数。

H+的R2–R5，每客户端由当前这次SVD的原U/S/V直接构造原谱因子，复制为只评价的raw twin，另测train/test。它不进入优化器、不用于SGD，不恢复或重新分解一次完整模型。实际修复模型训练不变。评价保护参数、buffers、各模块模式和实际使用设备的随机流；异常也恢复。两套预测的当前母模型完全相同，区别仅为下载修复；不会用H已经分歧的服务器代替这个即时对照。

## 启动与比较

从服务器`system/`使用 [svd_five_round_commands.sh](../system/svd_five_round_commands.sh) 的完整四条数组；只定义不执行，GPU为0/1/2/3，输出批次为`svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/{S,L,H,Hplus}/`。旧数据只读，不生成分片。四组配置都是local5、lr.005、batch16、F.001、clip10、原型关闭。

完成四组后，从system运行：

```bash
python compare_svd_five_round.py --S svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/S --L svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/L --H svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/H --Hplus svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/Hplus --output-dir svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/comparison
```

每组目录应恰有一份报告；多次运行时指定确切metrics目录，拒绝猜测。比较工具核对数据、初始权重和固定成员协议后输出逐轮`L−S`、`L−H`、`Hplus−H`，同时给出微调前差值、微调后差值和`(post−pre)_Hplus−(post−pre)_H`，另保留同母模型修复−原谱的即时差值。准确率内部fraction，summary表为百分点；测试只用于报告。无真实数据时的合成smoke不能证明性能收益。

`verify_svd_five_round.py --output E:/...` 仅创建自有合成工作区，执行真实main入口，核对五轮、原训练SVD/步数、H/H+首轮一致、诊断开关与冻结默认路径、权重指标和模型重载，删除自己工作区的验收模型副本以控制磁盘占用。

验收：37项专项及相关回归通过；四组均完成20客户端、原完整宽度CNN的合成R1–R5；另做诊断关闭和冻结`b150128`的两轮回归。上传、聚合、批次和RNG逐元素/状态精确一致，H/H+首轮精确一致。四组初始化及数据指纹、逐成员加权指标、CSV/JSON/H5、模型重载和266行比较输出核对通过。详细记录见 [svd_five_round_acceptance.json](svd_five_round_acceptance.json)。没有运行真实pat20实验，不根据这些合成准确率判断修复效果。
