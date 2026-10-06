# FedTargetProj 实验与投影消融

当前主性能指标为 Client 0 的 `final_target_local_acc` 与 `best_target_local_acc`。
全客户端平均准确率和 `target_client_test_acc` 仅用于诊断。
本批长程实验固定 `--rounds 100` / `-gr 100`，沿用历史循环实际执行 101 次本地训练/聚合，
不为了对齐自然语言的轮数改成 99，也不使用此前短程实验的 29。
以下两轮合成数据测试仅为代码验证，不是缩短后的正式实验。
APA-Logit 新实验先执行 `--rounds 5` 的 smoke（实际六次训练/聚合），检查通过后才显式启动 100 轮。

分支 `target_proj` 基于 `simple_v` 的 `a46b5e4`。原有 `serverCLIP.py`、
`clientCLIP.py`、`serverbase.py`、模型和本地优化器均未修改。

## 实现与对照

`clientTargetProj` 直接继承通用 `Client`，`FedTargetProj` 直接继承通用 `Server`。
本地训练、低秩下发和服务器训练循环独立实现，不继承或导入 `FedCLIP/clientCLIP`。
仅复用仓库低秩模型、数据接口、基础指标/存储与样本量权重。

本地目标为 `CrossEntropy(model(x), y) + regular_lamda * model.frobenius_decay()`。
U、V、分类头和其他可训练参数使用同一个 SGD 学习率；保留原梯度范数上限 10。
没有 CLIP 编码器、文本特征、对齐模块/损失、非对称学习率或 U 特殊梯度缩放。
本算法默认启用低秩正则，`-is_regular 0` 可显式关闭用于消融；非对称学习率和
U 特殊缩放开关若被误开启会报错。日志 `ce_loss` 和 `regularization_loss` 分别记录
分类损失与已乘系数的低秩正则；训练集 loss 指标记录两者之和。

十九种模式全部客户端仍然参与同样的普通本地训练，服务器写回参数的规则不同：

- `avg`：原始样本量加权平均；保留基线逐参数加权求和的浮点计算顺序。
- `target_only`：服务器参数直接取目标客户端恢复后的参数，目标权重为 1。
- `projection`：以目标 delta 为参考，仅对其他客户端中全模型内积小于 0 的
  delta 应用 `delta_j - dot(delta_j, delta_k)/(norm(delta_k)^2 + 1e-12)*delta_k`，
  再使用原样本量权重聚合，不重新归一化。
- `layer_mask`：新增模式，使用纯本地变化量逐层判方向，以目标 post-local 模型为 anchor，
  加上通过 hard mask 的辅助客户端 local updates，详见下一节。
- `layer_mask_budget`：在相同 hard mask 之后，对每层全部 helper 的加权和施加固定 beta=1 的
  target-local norm budget；不缩放 target anchor，不改变旧四个模式。
- `layer_softmax`：只在正方向 helper 内按样本权重与 `exp(c/0.2)` 重新分配该层原有 helper 总权重。
- `layer_relu`：只在正方向 helper 内按样本权重与正 cosine 重新分配该层原有 helper 总权重。
- `projection_local`：pure-local full-model projection，以原 Avg post model 加 correction。
- `layer_projection_global`：global delta 的逐层 projection，只删除负向平行分量。
- `layer_projection_local`：pure-local delta 的逐层 projection，以原 Avg post model 加 correction。
- `projection_same_label`：原 full-model global projection，只有 helpers 1–3 进入聚合。
- `projection_cross_label`：原 full-model global projection，只有 helpers 4–19 进入聚合。
- `projection_softmax`：原 full-model/global projection，以投影前 cosine 的 Softmax 重分配全部 helper mass。
- `projection_relu`：原 full-model/global projection，以投影前正 cosine 重分配全部 helper mass；全非正时回退原样本权重。
- `softmax_only`：沿用 `projection_softmax` 的 cosine 和 helper 权重，直接累积原始 global delta，完全不做 Projection。
- `apa`：使用上一轮 full-W basis 与本轮 target post-model 的 residual 学习一组服务器聚合权重，再聚合本轮上传。
- `apa_logit`：固定 target=.05、helpers 总质量=.95，只对 helper Softmax logits 做 centered APA 梯度下降。
- `dwa_soft`：C0 普通 post-local 副本额外训练一个 epoch，以 guidance 的倒数平方距离分配 helper 质量。
- `dwa_soft_projection`：使用完全相同的 guidance 距离权重，再应用原 full-model global-delta Projection。

这里的参数空间是 **simple_v Avg 实际聚合的恢复后完整模型** 的 `named_parameters()`，
包含分类头。**旧 projection** 先恢复低秩 U/V，再减本轮服务器参数；不是只投影 U/V，也不是减去
低秩下发后的客户端参数。因此 delta 也包含基线低秩截断带来的变化。
BatchNorm 等 buffer 不进入聚合或内积，遵循原有低秩聚合约定。

内积逐 tensor 以 float64 归约；仅使用一个全模型投影系数。利用线性关系
`G_proj = G_avg - sum(p_j * coefficient_j) * delta_k` 流式累积，
不用把 20 份恢复模型同时放入显存。输入客户端参数不会被修改。
零目标更新不投影；零向量的 cosine 日志约定为 0。

第一版要求全参与、无掉线、固定目标，不支持 resume。没有新增投影强度、阈值等参数。

## 新增 layer_mask

旧 `projection` 的数学实现保持不变。新模式每轮在低秩下发完成后、任何本地训练开始前，
逐个加载客户端实际收到的 checkpoint，将其恢复到 full-W 并保存纯参数快照 `W_i_pre`。
快照位于工作目录的 `layer_mask_pre/Client_i.pt`，每轮覆盖，不加入最终模型导出。
恢复过程保护 Python、NumPy、CPU/CUDA RNG，避免额外观测改变训练随机序列。
快照带有服务器当前轮次检查，缺失或过期不会退回用 global model 代替。

本地训练后恢复 `W_i_post`，只使用 `D_i = W_i_post - W_i_pre` 判断方向。
因此 SVD 下发截断差 `W_i_pre - W_global` 不参与新模式的冲突判断。
所有判断只读已有模型参数，不使用目标客户端训练/测试数据，不执行额外训练。
目标准确率仍在聚合完成之后独立评估，不参与 mask 或聚合决策。

按恢复后 `named_parameters()` 的模块前缀分组，同层 weight/bias 共用一个 cosine 和 mask。
`Decom_CNN-5-512` 实际识别为 `conv1, conv2, fc1, fc2, fc3`，其中 `fc3` 是分类头；
共享模块别名由 `named_parameters()` 去重，不重复计算。程序启动时打印分组及参数名。

每层执行：

```text
c_i,l = dot(D_i,l, D_0,l) / (norm(D_i,l) * norm(D_0,l) + 1e-12)
M_i,l = 1 if c_i,l > 0 else 0
W_new,l = W_0_post,l + sum(i != 0, p_i * M_i,l * D_i,l)
p_i = n_i / sum(all clients' n_i)
```

目标层自身 `cosine=1, mask=1`，anchor 系数固定为 1，不乘目标样本权重。
辅助层通过 mask 后保留完整的 local update，未通过则贡献严格为零；不重新归一化剩余权重。
若任一层范数小于 `1e-12`，辅助 cosine 定为 0 并记录 `zero_norm=True`，该层被屏蔽。
目标即使为零范数仍保留 anchor 和自身 mask=1，同时标记 zero_norm。
不引入 softmax、temperature、可学习权重或新的可调阈值。

每轮终端完整打印 20×L cosine/mask/norm/zero_norm 矩阵，逐客户端输出
`old_global_delta_cos`、`full_local_delta_cos` 和 local-update norm，
逐层输出负方向数量、冲突率、被 mask 数量和 masked norm ratio。
**冲突统计只计负 cosine；mask 还会删除零 cosine，因此两者可能不同。**

`masked_update_ratio` 为 `sum(i,l, p_i*(1-M_i,l)*norm(D_i,l)) /
(sum(i,l, p_i*norm(D_i,l))+eps)`，分母包含目标客户端，分子中目标贡献为零。
每层 masked norm ratio 使用相同定义，只在该层求和。

新增文件均按轮写入工作目录，并复制到最终模型目录：

- `layer_mask_metrics.csv`：整体冲突、mask 比例、masked_update_ratio、目标准确率。
- `layer_mask_clients.csv`：每个客户端两种 full-model cosine、local/old update norm、零范数标记。
- `layer_mask_cosines.csv`：每个客户端×层的 cosine、mask、norm、zero_norm 和加权删除范数。
- `layer_mask_matrices.json`：`history` 列表保存每轮完整矩阵、逐层统计、客户端 ID 及层名。
  行按 client ID 排序，列按 `layer_names`；包含 `layer_groups`，可以直接对应实际参数。

新模式同样写通用 `target_proj_metrics.*` 和 H5 `target_projection` 组，
其中诊断字段为新 mask 统计；旧三种模式的 projection 诊断定义保持不变。

## 新增 layer_mask_budget：只限制联合 helper

复用 `layer_mask` 的 W_pre 快照、纯本地 delta、逻辑分组、cosine 和 hard mask，
本地训练、通信量及样本权重均不变。不增加 EMA、softmax、temperature、prefix mask 或可学习权重。
服务器先按原样本权重汇总全部通过 mask 的辅助更新，再对整层的联合向量做一次预算限制：

```text
H_l = sum(i != 0, p_i * M_i,l * D_i,l)
T_l = norm(D_0,l)
G_l = norm(H_l)
beta = 1.0, eps = 1e-12
s_l = 1                         if G_l <= T_l
s_l = T_l / (G_l + eps)          if G_l > T_l
W_new,l = W_0_post,l + s_l * H_l
```

每层 weight/bias 共用一个 scale。预算不逐客户端裁剪，不重新归一化权重，不放大 helper；
Client 0 的完整 `W_0_post` 始终系数为 1。helper 的原方向保留，范数限制在 target 本层
local-update 范数以内（浮点舍入误差除外）。`G=2*T` 时 scale 约为 0.5，epsilon 会带来微小差异。

退化情况：helper 为零时 scale=1；target norm < eps 且 helper 非零时 scale=0。
若两个范数同时很小但 helper 非零，优先遵守 target 零预算规则；其他 G < eps 情况取 scale=1。
实际 hard mask 已经会屏蔽零范数 target 层的所有辅助更新，预算函数仍单独处理这些边界。

联合 helper 独立累积，不能从最终权重减 anchor 反推，以免大权重吞掉小更新。
未触发预算的层保留旧 `layer_mask` 的浮点累加顺序，确保与原结果逐位一致。
只有触发预算的层才重新计算 `W_0_post + clipped_helper`。

原有全部 LayerMask 终端诊断照常打印，每轮另打印 `[LayerBudget]` 表：
`target_local_norm`、`raw_helper_norm`、`helper_target_ratio`、`budget_scale`、
`clipped_helper_norm`、`budget_active`。整体打印生效层数、最大/平均 helper-target ratio，
以及整个模型的 raw/clipped helper 范数。ratio 使用 `G/(T+eps)`。

该模式使用独立文件名，不写入或覆盖旧 `layer_mask_*` 结果：

- `layer_mask_budget_metrics.csv`：目标准确率、原 mask 指标及整体预算统计。
- `layer_mask_budget_layers.csv`：每轮每层的六项预算指标及目标准确率。
- `layer_mask_budget_matrices.json`：保留所有原矩阵，新增 `budget_beta`、`budget_layers`、
  `budget_summary`；每个 `history` 条目包含本轮目标准确率。
- `layer_mask_budget_clients.csv`、`layer_mask_budget_cosines.csv`：保留逐客户端/逐层原始诊断。

这些文件同步复制到最终模型目录。通用 `target_proj_metrics.*` 和 H5 仍记录本轮结果，
H5 `target_projection` 属性额外记录 `budget_beta=1.0`，没有 beta 调参开关。

在 `system/` 单独启动新模式：

```bash
python run_target_proj.py --modes layer_mask_budget --rounds 100
```

或并排运行两个 mask 实验：

```bash
python run_target_proj.py --modes layer_mask layer_mask_budget --rounds 100
```

直接调用 `main.py` 时使用 `--target_proj_mode layer_mask_budget`，其余参数与 layer_mask 相同。
启动器默认仍运行原四个模式；只有显式选择 budget 才运行新实验。

## 新增 layer_softmax / layer_relu：保留总权重的正方向连续加权

两个模式沿用原 `layer_mask` 的快照、纯本地 delta、五层分组、cosine 和 hard rejection。
Client 0 的 `W_0_post` 完整保留，anchor 系数始终为 1；不进入 helper 归一化。
它们不接入 helper budget、EMA、prefix mask、时间平滑或可学习权重，不增加训练或通信。
旧五种模式的聚合核与本地训练均保持不变。

每层只在 `positive = {i != 0: c_i,l > 0}` 内计算，`p_i` 仍以全部客户端样本量为分母：

```text
P_l = sum(i in positive, p_i)
layer_softmax: score_i = p_i * exp((c_i,l - max_positive_cos) / 0.2)
layer_relu:    score_i = p_i * (c_i,l / max_positive_cos)
alpha_i,l = P_l * score_i / sum(positive scores)
W_new,l = W_0_post,l + sum(i != 0, alpha_i,l * D_i,l)
```

非正 cosine 的 helper 权重严格为零。空正方向集合或 `P_l=0` 时，全部 helper 权重为零，
直接返回 target anchor。Softmax 温度固定 0.2，不增加调参开关。
ReLU 除以最大正 cosine 只为数值稳定，归一化后等价于 `p_i * max(c_i,l, 0)`。
按用户确认，优先严格保留 `sum(alpha)=P_l`（浮点舍入误差除外），不在正分母额外加 epsilon；
否则微小正 cosine 下总权重会收缩。原 cosine 计算的 epsilon/零范数规则保持不变。

当其他 helper 仍有固定正分数时，ReLU 对某个趋近零的正 cosine 分配的权重也趋近零。
若只剩一个正方向 helper，它必须保留全部 `P_l`；因此不能同时保证该特殊情况下权重也趋近零。
这里保留的是 sample-weight mass，不是 helper 向量范数；联合辅助更新仍可能大于 target 更新。

服务器复用原 mask 诊断后，再逐个读取已有 checkpoint 完成连续加权，不同时保留全部恢复模型。
额外读取过程保护 Python、NumPy、CPU/CUDA RNG，不改变后续训练随机流。
目标准确率仍只在聚合后评估，不参与权重选择。

原始 cosine/mask/norm/zero-norm 矩阵、两种 full-model cosine、冲突统计和
`masked_update_ratio` 全部保留；后者仍只度量 hard rejection 删除的比例。
每轮额外打印完整 `[LayerSoftmax]` 或 `[LayerReLU]` aggregation weight matrix，
Client 0 行标为 `anchor`。每层打印正方向人数、原/最终总权重、质量误差、正 cosine 最小/最大/均值、
最大权重、最小正权重和 `effective_helper_count = 1 / sum((alpha/P_l)^2)`；空集合取 0。

两个模式分别使用独立前缀 `layer_softmax` / `layer_relu`，不覆盖旧模式文件：

- `*_metrics.csv`：目标准确率、原 mask 统计及平均有效 helper 数、最大总权重误差。
- `*_clients.csv`、`*_cosines.csv`：原逐客户端/逐层诊断。
- `*_weights.csv`：每轮客户端×层最终 helper 权重、角色及 anchor 系数；target 的 helper 权重为 0、anchor 系数为 1。
- `*_layers.csv`：每轮逐层正方向权重统计。
- `*_matrices.json`：保留原矩阵、client IDs、layer names、目标准确率，增加 `final_weight_matrix`、
  `original_weight_mass`、`effective_helper_count`、`weight_summary` 和归一化规则。
  最终权重矩阵的 target 行为字符串 `anchor`；另存 target 行为 0 的纯数值 `helper_weight_matrix`。

上述文件复制到最终模型目录；H5 同时记录模式、归一化规则及 softmax 的固定温度。

在 `system/` 用两个 GPU 运行两组完整实验，和既有长程结果统一使用 `-gr 100`：

```bash
python run_target_proj.py --modes layer_softmax layer_relu --parallel --device-ids 0 1 --rounds 100
```

其余参数与当前 LayerMask 完全一致。各组使用独立进程、checkpoint、日志和结果目录。
不加 `--parallel` 则顺序运行；默认模式仍为原四组，不自动加入新模式。
`--rounds` 沿用原循环上限语义，因此 100 对应 101 次聚合；可加 `--dry-run` 查看完整命令。

## 五个独立 Projection 变体

新增 kernel 位于 `system/utils/projection_variants.py`；旧 `target_projection.py`、
`layer_mask.py`、`layer_mask_budget.py`、`layer_weighting.py` 不改动。
同层 weight/bias 合并内积，仍使用恢复后 full-W 参数，不包含 buffers。

| 模式 | 参考更新 | 投影粒度 | 聚合基准 |
|---|---|---|---|
| 原 `projection` | `W_post - W_global` | 全模型 | 原实现，不修改浮点运算顺序 |
| `projection_local` | `W_post - W_pre` | 全模型 | `sum(p_i * W_i_post)` |
| `layer_projection_global` | `W_post - W_global` | conv1/conv2/fc1/fc2/fc3 | `sum(p_i * W_i_post)` |
| `layer_projection_local` | `W_post - W_pre` | conv1/conv2/fc1/fc2/fc3 | `sum(p_i * W_i_post)` |

对非目标客户端，令 `u` 为表中对应更新。每个 full-model / layer group 计算：

```text
dot_i,l = <u_i,l, u_0,l>
c_i,l = dot_i,l / (norm(u_0,l)^2 + 1e-12)  if dot_i,l < 0, else 0
W_new,l = W_avg,l - sum(i != 0, p_i * c_i,l) * u_0,l
```

目标不投影，非负 dot 完整保留；负 dot 只删除反向平行分量，正交分量保留，绝不整层置零。
零 target 对应 dot=0，不产生 NaN。三个新变体无冲突时直接保留 Avg 累加结果，逐位相同。
pure-local 模式不会使用 `W_global + sum(p_i * D_i)`，从而保留异构低秩下发产生的 pre-local 结构差异。
单一 group 的 global variant 与原 projection 数学一致，可能因 Avg 加参数与 global 加 delta 的
浮点运算顺序存在舍入差异；旧 projection 本身的参数和诊断要求逐位不变。
没有 hard mask、continuous weighting、budget、EMA 或新超参数。

`PRE_LOCAL_MODES` 只含原四个逐层 mask/weight 模式及两个 pure-local projection 模式；
`LAYER_GROUP_MODES` 只含原四个逐层模式和两个逐层 projection 模式。
`layer_projection_global` 和两个 source 模式不读取或生成 pre-local 快照。

Source ablation 使用固定元信息 `same={1,2,3}`、`cross={4,...,19}`，不读取私有数据决定组别。
启动时要求 `dataset=Cifar100, partition=pat, class_per_client=20, target_client_id=0, num_clients=20`，
任一不匹配直接报错。所有客户端仍正常下载、训练、上传，排除客户端仅在聚合时权重为零。

```text
P_H = 1 - p_0
q_0 = p_0
q_i = P_H * p_i / sum(j in selected, p_j)    if i in selected
q_i = 0                                    for excluded helpers
```

然后对选中上传使用**原 full-model/global-delta projection kernel**，以 `q_i` 替代 `p_i`。
这两个模式没有 target anchor=1，也没有 pure-local 或逐层投影；目标权重始终为原 `p_0`。
选中组样本质量为零而 helper 总质量非零时报错，不静默改公式。
它们回答“固定总 helper mass，某来源是否足以提供帮助”，不是 Shapley 或因果贡献分解。

每个新模式独立保存 `<mode>_metrics.csv`、`<mode>_clients.csv`、`<mode>_matrices.json`。
逐层投影另保存 `<mode>_layers.csv`（客户端×层）和 `<mode>_layer_summary.csv`（每层统计）。
full-local CSV 包含 local dot/cosine、冲突、coefficient、update/removed norm 与比例；
逐层 CSV 包含对应 layer 字段；JSON 保存完整 cosine/conflict/coefficient/norm 矩阵和逐层统计，
终端完整输出所有客户端×层的 cosine/conflict 表。
source CSV 保存 original/effective weight、selected 标记、cosine、conflict 和实际 removed norm；
JSON/终端保存 helper IDs、数量、target weight、helper 总质量及排除数量。
排除客户端的 conflict/cosine 是原始观测，removed norm 为零；source 总冲突统计只计选中 helpers。

新 projection 的 full-model `removed_update_ratio` 为 helper 删除范数之和 / helper 更新范数之和；
逐层 `overall_removed_update_ratio` 为 `sum(p_i * removed_norm_i,l) / (sum(p_i * update_norm_i,l) + eps)`，
分子分母均不含 target。`mean_removed_ratio` 是逐层 helper 比例的算术平均，
`weighted_removed_norm` 是该层 `sum(p_i * removed_norm_i,l)`。冲突率分母不含 target。
这与 LayerMask 的统计口径有区别，比较时需使用同一分母或原始逐层数据。

## Projection 后的 full-model similarity weighting

`projection_softmax` / `projection_relu` 位于 `utils/projection_variants.py`，通过独立
`aggregate_projection_weighting()` 复用**未修改的** `aggregate_target_updates(..., "projection")`。
它们使用 `Delta_i = W_i_post - W_global` 和投影前的原始 full-model cosine，
不使用 projected cosine，不使用 pure-local snapshot 或 logical layer groups。
weight、bias、分类头共享整个模型的一个冲突判定与 projection coefficient。

原 projection 规则保持原样（epsilon=1e-12），目标自身不投影，负方向 helper 仅删除反向平行分量：

```text
k_i = dot(Delta_i, Delta_0) / (norm(Delta_0)^2 + 1e-12)    if i != 0 and dot < 0
k_i = 0                                                  otherwise
projected_i = Delta_i - k_i * Delta_0
alpha_0 = p_0
P_H = 1 - p_0
W_new = W_global + alpha_0 * Delta_0 + sum(i != 0, alpha_i * projected_i)
```

`projection_similarity_weights()` 只改变 helper 内部权重，不将目标放进归一化：

```text
projection_softmax:
    tau = 0.2
    c_max = max(helper raw cosine)
    s_i = exp((c_i - c_max) / tau)
    alpha_i = P_H * s_i / sum(helper s_j)

projection_relu:
    s_i = max(c_i, 0)
    if any helper s_i > 0:
        alpha_i = P_H * s_i / sum(helper s_j)
        relu_fallback_used = 0
    else:
        alpha_i = p_i
        relu_fallback_used = 1
```

正常加权时**不乘 sample weight**；不平衡样本下，同 cosine 的 helpers 仍获得同样权重。
只有 ReLU 全非正 fallback 才恢复原始（可能不均匀的）sample weights，逐位复现原 Projection。
ReLU 对正分数先除最大值以避免微小 cosine 的归一化下溢，不在分母加 epsilon 缩小 helper mass。
因此始终 `alpha_0=p_0`，`sum(helper alpha)=1-p_0`（原权重及浮点舍入精度内）。
Softmax 对 [-1,1] 内的负 cosine 仍给出正权重（helper 总质量非零时），保留投影后的正交信息；
ReLU 在有正方向 helper 时将非正 cosine helper 的整个 aggregation weight 置零。
单一正 helper 获得全部 helper mass；有效 helper 数可从约 19 降至约 1。

实现先用旧 Projection 核得到原始 dot/norm/cosine 所需标量，再按新权重调用同一个旧核。
额外遍历只逐个重新读取服务器已有上传，不增加客户端训练、下载、上传或服务器数据优化；
该遍历保护 Python/NumPy/PyTorch CPU/CUDA RNG，不影响之后的训练随机流。
没有 layer-wise、helper source group、budget、EMA、prefix mask 或其他调参机制。

两种模式分别保存以下独立前缀文件，并复制到最终模型目录：

- `projection_softmax_metrics.csv` / `projection_relu_metrics.csv`：原 projection 诊断、主准确率和权重 summary。
- `projection_softmax_clients.csv` / `projection_relu_clients.csv`：每轮每客户端完整诊断。
- `projection_softmax_matrices.json` / `projection_relu_matrices.json`：每轮客户端数据、权重 summary 和主指标最终汇总。

客户端字段包括 `client_id,is_target,sample_weight,cosine_before_projection,dot_before_projection,conflict,`
`projection_coefficient,removed_component_norm,projected_update_norm,raw_similarity_score,aggregation_weight`。
目标的 similarity score 记 0，表示不进入 helper 归一化；目标 aggregation weight 为 `p_0`，不是 anchor=1。
Softmax 的 raw score 是稳定移位后的指数，并记录 `temperature=0.2,softmax_shift_max`；
ReLU 记录 `relu_score,relu_fallback_used`，fallback 在 metrics 和每个客户端行中均明确可见。
`projected_update_norm` 是投影后、乘 aggregation weight 前的完整更新范数，诊断使用 double precision。
每轮 summary 包含 target/helper mass、最小/最大 helper weight、mass error、
`effective_helper_count = 1/sum((alpha_i/P_H)^2)` 及原冲突/删除指标。
空 helper 或 helper mass=0 时有效 helper 数记 0。

`conflict_client_count/ratio` 和 `removed_update_ratio` 沿用原 Projection 的几何诊断口径，
不会因 ReLU 最终权重为零而抹去该客户端原有冲突；它们不表示 ReLU 额外整客户端删除的比例。
`avg_target_cos/proj_target_cos` 则使用实际新 aggregation weights。
通用 CSV/JSON/H5 中的 Client 0 post-local final/best 指标继续完整保存，H5 记录 weighting scope 和固定温度。

在 `system/` 下启动两组**完整**实验，公共参数保持冻结；下面命令仅供用户自行执行：

```bash
python main.py -algo FedTargetProj --target_proj_mode projection_softmax --target_client_id 0 --seed 0 -t 1 -data Cifar100 -ncl 100 -nc 20 -niid 1 -pt pat -cpc 20 -jr 1.0 -m Decom_CNN-5-512 -ls 5 -lbs 16 -lr 0.005 -is_regular 1 -regular_lamda 1e-3 -gr 100 -eg 1 -dev cuda -did 0 -exp_name target0_seed0_projection_softmax -sfn target_proj_runs/projection_softmax/checkpoints --h5_result_root target_proj_runs/projection_softmax/h5_results --final-model-root target_proj_runs/projection_softmax/final_models

python main.py -algo FedTargetProj --target_proj_mode projection_relu --target_client_id 0 --seed 0 -t 1 -data Cifar100 -ncl 100 -nc 20 -niid 1 -pt pat -cpc 20 -jr 1.0 -m Decom_CNN-5-512 -ls 5 -lbs 16 -lr 0.005 -is_regular 1 -regular_lamda 1e-3 -gr 100 -eg 1 -dev cuda -did 1 -exp_name target0_seed0_projection_relu -sfn target_proj_runs/projection_relu/checkpoints --h5_result_root target_proj_runs/projection_relu/h5_results --final-model-root target_proj_runs/projection_relu/final_models
```

也可使用现有 GPU 排队 launcher（默认 modes 不变）：

```bash
python run_target_proj.py --modes projection_softmax projection_relu --rounds 100 --parallel --device-ids 0 1
```

## SoftmaxOnly：删除 Projection 的严格消融

`softmax_only` 与 `projection_softmax` 使用同样的 full-model global delta：
`Delta_i = W_i_post - W_global`。完整参数包含各层 weight/bias/分类头，恢复及低秩下发流程不变。
不需要 pre-local snapshot，也不建立 logical layer groups。

在 `aggregate_projection_weighting()` 入口独立分支调用 `aggregate_softmax_only()`。
它复用 `_delta/_dot/_cosine` 与 `projection_similarity_weights(..., "projection_softmax")`，
后者完全未修改，因此同一人工输入的 cosine、稳定 Softmax scores 和权重与 projected control 逐位相同。
当前非零向量 cosine 为 `dot/(norm_i*norm_0)`，零向量记 0；为严格对齐现有实现，
没有按需求展示公式额外加入 denominator epsilon，否则微小更新时会改变权重。

```text
c_i = cosine(Delta_i, Delta_0)
alpha_0 = p_0
alpha_i = (1-p_0) * exp((c_i-max_helper_cos)/0.2) / sum(helper exp scores)
W_new = W_global + sum(all clients, alpha_i * Delta_i)
```

负 cosine helper 仍有正权重（helper 总质量非零时），其反向平行分量和正交分量都保留。
不执行 Projection、clipping、ReLU、mask、负向筛选或 helper 删除。
实现甚至不调用旧 `aggregate_target_updates(..., "avg")`，因为该函数的 Avg 分支仍计算假设性投影诊断。
两次服务器读取只观察/累积原始更新，额外恢复过程保护 RNG，客户端训练与通信完全不变。
结果数学上等价于 `sum(alpha_i * W_i_post)`；使用 global+delta 的浮点运算顺序与 `projection_softmax` 对齐。
因此同 cosine、等样本权重时与 Avg 数学一致，但非零 global 下可能存在不同求和顺序的舍入差异。
权重质量在浮点精度内保持 `alpha_0=p_0` 和 `sum(helper alpha)=1-p_0`，不添加差额分配等额外机制。

独立输出 `softmax_only_metrics.csv`、`softmax_only_clients.csv`、`softmax_only_matrices.json`，
继续使用现有 projection variant CSV/JSON 格式，并导出通用 CSV/JSON/H5 的 Client 0 post-local final/best。
保留 `cosine_before_projection` 等原列名便于比较；`projection_scope=none, projection_enabled=0`。
`projection_coefficient`、`removed_component_norm`、`removed_update_ratio` 恒为 0，
`projected_update_norm` 兼容列记录原始更新范数，`proj_*` 兼容指标与 `avg_*` 相同。
conflict 只表示原始 dot<0，不触发任何处理。
summary 增加 `mean_helper_weight`、`std_helper_weight`（所有 helpers、population std/ddof=0），
并保留 min/max weight、helper mass、effective helper count、temperature、shift max 等诊断。
现有 projected control 没有 weight entropy，本次不引入该额外指标。

干净的 2×2 对照为：

| | Sample-size weights | Softmax similarity weights |
|---|---|---|
| 无 Projection | `avg` | `softmax_only` |
| 有 Projection | `projection` | `projection_softmax` |

新增独立启动脚本 `system/run_softmax_only.py`，调用原 launcher，仅选择新模式并固定 `--rounds 100`。
Client 0/seed 0、数据切分、20 clients/full participation、低秩模型/正则/学习率/local epochs/batch 等公共设置
全部来自同一个 launcher，不复制或修改训练配置。结果目录按 mode 隔离，默认旧 modes 不变。

在 `system/` 执行（本次只验证 dry-run，不自动启动正式实验）：

```bash
python run_softmax_only.py --device-id 0
# 查看实际命令但不训练：
python run_softmax_only.py --device-id 0 --dry-run
# 等价的共享 launcher 命令：
python run_target_proj.py --modes softmax_only --rounds 100 --device-id 0
```

## APA：跨轮学习服务器聚合权重

`system/utils/apa_aggregation.py` 独立实现 APA-style surrogate；旧聚合核、客户端训练和统一评价时序保持不变。
只维护目标客户端的一组权重，不实现原始 FedAPA 的所有个性化服务器模型。
不加入 cosine、Projection、mask、Softmax、ReLU、Top-K 或额外客户端上传。

循环编号 `loop_round=0` 时，权重直接初始化为原 sample-count weights（等量 20 clients 为各 .05），
velocity 为零；按旧 Avg 的运算顺序聚合本轮 post-model，结果逐位一致。
这一轮不进行 APA update，也不执行 self-weight 覆写；只缓存第一次真实聚合使用的 full-W basis。
`apa_weight_update_enabled=0`，梯度及其范数在 CSV 留空、JSON 为 null、H5 为 NaN，
此时记录的 proxy loss 仅是可观测 residual，不代表存在合法的初始 basis 梯度。

从 `loop_round=1` 起，设当前服务器模型为上一轮权重与 basis 的混合，计算：

```text
R = W_server_full - W_target_post_full
J = 0.5 * ||R||²
g_j = <B_previous_j, R>
v_new = 0.9 * v_old + g
A_raw = A_old - 0.01 * v_new
A_clipped = clip(A_raw, 0, 1)
A_clipped[target] = apa_self_weight   # 默认 0.5，归一化前设置
A_new = A_clipped / sum(A_clipped)
W_new = sum_j A_new[j] * W_current_post[j]
B_next[j] = W_current_post[j]
```

梯度必须先完整读取上一轮 basis，随后才读取并缓存本轮上传用于新的聚合。
残差中的 server 是**分解前的 full-W 服务器模型**，与已缓存 basis 的加权混合相对应；
不把低秩分解后再恢复的 C0 pre-model 冒充为这个线性混合，不对 SVD 求导。
这是参数空间代理目标，服务器不读取 C0 数据或测试准确率来优化权重。

`--apa_server_lr .01`、`--apa_momentum .9`、`--apa_self_weight .5` 均显式暴露，正式比较使用这些默认值。
self-weight 在 clip 后、normalize 前覆写，因此**最终 C0 权重不保证为 .5**。
总和为零时回退 uniform；默认 self-weight=.5 时总和不会为零，显式设为零的消融仍有保护。
权重保持有限、非负且在浮点精度内和为 1。遇到 NaN/Inf 输入或 optimizer 溢出时报错，
不把无效结果提交为新的权重状态。零 residual 对应零梯度，但不会清空已有 momentum，
后处理也继续执行；因此不能把零梯度解释为最终权重必定不变。

服务器在 `checkpoints/apa_basis/slot_0/` 和 `slot_1/` 轮换缓存两套 recovered full-W 参数，
带 client ID/loop round 标记；每次只读取一个客户端，避免把全部 basis 常驻 GPU。
两槽占用最多两轮完整 basis 的磁盘空间，不增加客户端训练或通信。
权重与 velocity 在服务器内存跨轮保存；沿用当前协议要求 fresh run，不支持 resume。

独立日志及最终导出文件：

- `apa_metrics.csv`：主/诊断准确率、proxy loss、residual/gradient/weight-update norm、优化器参数、
  target/helper weight 汇总、helper min/max/mean/std、effective helper count、update-enabled 和 basis 轮号。
- `apa_weights.csv`：每轮每客户端的 sample weight、更新前权重、SGD 原始权重、最终权重、gradient、velocity。
- `apa_history.json`：上述完整记录、评价 summary 和 proxy/basis/self-weight 的明确语义。
- 公共 CSV/JSON 和 H5 继续保存；H5 `target_projection` 另含 `[round, client]` 的
  `apa_weight`、`apa_grad`、`apa_velocity`。helper 总权重为零时 effective helper count 记为零。

日志 `round` 从 1 开始，所以显示 `Round 1` 对应 `loop_round=0`，没有权重更新。
`apa_weight_update_norm` 测量最终后处理权重相对旧权重的变化。
proxy loss 的下降不保证 post-local accuracy 上升；分析继续使用
`local_t → aggregate_t → local_(t+1)`，不把同一行两种准确率解释为 local FT gain。

在 `system/` 执行：

```bash
python run_apa.py --device-id 0
python run_apa.py --device-id 0 --dry-run
# 等价的共享 launcher：
python run_target_proj.py --modes apa --rounds 100 --device-id 0 \
  --apa_server_lr 0.01 --apa_momentum 0.9 --apa_self_weight 0.5
```

独立脚本只转发到共享 launcher，公共训练参数不复制。
正式配置仍为 CIFAR-100/pat_20、20 clients、全参与、C0、Decom_CNN-5-512、5 local epochs、
batch 16、client lr .005、regularization .001、seed 0、`-gr 100`（101 次训练/聚合）。
APA 是否超过用户提供的 ProjectionSoftmax final 45.20% / peak 45.60% 基线，需完成正式实验后判断。

## APA-Logit：固定质量与中心化 helper logit 梯度

基于 `b682f19` 新增独立模块 `system/utils/apa_logit_aggregation.py`。
原 `apa_aggregation.py`、APA 服务器方法和旧模式的计算路径保持不变，原 APA 继续作为对照。
用户报告原 APA 的首次有效 raw gradient 约 599–607，直接 weight SGD 后 helpers 被 clip 到零；
本模式消除共同偏置并改变权重参数化，proxy objective 仍然为原 full-W residual 的平方范数。

```text
alpha_target = 0.05
z_initial = zeros(19, float64)
q = softmax(z)                         # 只用于 logits 的 simplex 参数化
alpha_helpers = 0.95 * q
r = W_server_full - W_target_post_full
J = 0.5 * ||r||²
raw_j = <B_previous_j, r>              # 上一轮真正构造当前服务器的 basis
mean_raw = sum_helpers(q_j * raw_j)
centered_j = raw_j - mean_raw
grad_z_j = 0.95 * q_j * centered_j
z_next = z - 0.01 * grad_z             # 无 momentum
W_next = 0.05 * W_target_post + sum_helpers(0.95 * softmax(z_next)_j * W_j_post)
```

`loop_round=0` 不更新 logits、无有效梯度；以零 logits 聚合当前 uploads，并建立首套 basis。
20 个等样本量客户端时，首轮每个权重均为 .05；使用字面值 .05 避免 `.95*(1/19)`
在 float64 下的一 ULP 表示差异，保留 Avg 原运算顺序，首轮参数逐位一致。
后续先完整读取上一轮 basis 求梯度，再使用新 logits 聚合本轮上传并缓存为下一套 basis。
与原 APA 一样，server residual 使用分解前的 full-W 服务器模型，不对低秩 SVD 求导。

只提供新参数 `--apa_logit_lr`，默认 .01。固定 `apa_logit_momentum=0`；
不使用原 APA 的 server lr、momentum 或 self-weight 参数。
没有 direct weight SGD、weight clip、额外权重归一化、fallback、logit recentering、
gradient clipping、similarity weighting、Projection 或 helper pruning。
logits、q、raw/centered/logit gradients 与内积使用 float64，聚合时按模型 dtype/device 乘权重。
Softmax 使用减最大值的稳定实现，检查 logits/q/weights 有限且 q>0、两组质量分别为 .05/.95。
有限精度下若极端 logit 差导致 Softmax 下溢为零，会明确报错，不偷偷加下限、删除 helper 或改 lr。
当前实现的通用数学测试可以使用更少 helpers，此时仍固定两组质量；首轮等价 sample-weight Avg
的正式比较前提是本任务指定的 20 个等样本量客户端。

独立缓存位于 `apa_logit_basis/slot_0` 与 `slot_1`，逐客户端保存/读取，最多两套 full-W basis；
不会覆盖 `apa_basis`。沿用 fresh-run 协议，不新增 resume。

独立保存 `apa_logit_metrics.csv`、`apa_logit_weights.csv`、`apa_logit_history.json`，
保留公共 CSV/JSON/H5 与 final-model 导出。每轮含所有主/诊断准确率、proxy loss、residual norm、
raw/centered/logit gradient norm、logit update norm、lr、update-enabled、basis loop round、
target/helper weight 汇总、min/max/mean/std、effective helper count、max-abs-logit、logit std。
三个 gradient norm **仅统计 helpers**；`effective_helper_count=1/sum(q²)`，初始为 19。
客户端行同时记录 `aggregation_weight` 与 `apa_weight`；target 固定为 .05，其 logit/q/梯度留空。
helper 的 `apa_logit`/`apa_q`/`apa_weight` 为**更新后、实际用于本轮聚合**的值，
额外保存 `apa_logit_before`/`apa_q_before`，对应 raw/centered/logit gradients 的计算时刻。
因此检查 `grad_z=.95*q*centered` 时必须使用 `apa_q_before`，不能混用更新后的 q。
首轮所有 gradient 在 CSV 留空、JSON 为 null、H5 为 NaN，不伪造初始梯度。
H5 另保存这些逐客户端字段的 `[round, client]` 矩阵，JSON/H5 元数据说明前后时序。

在 `system/` 先运行 smoke（显示 Round 1–6，对应 loop 0–5）：

```bash
python run_apa_logit.py --device-id 0
# 等价命令，lr 仍然为 .01：
python run_target_proj.py --modes apa_logit --rounds 5 --device-id 0 --apa_logit_lr 0.01
# 只预览、不训练：
python run_apa_logit.py --device-id 0 --dry-run
```

确认 target=.05、helpers=.95、初始 N_eff=19，首轮 Avg 对齐；随后检查 raw/centered/logit gradient
尺度、helper weights 没有骤降为零、max_abs_logit 合理。centered gradient 的实际大小由 basis 差异决定，
不能承诺所有真实轮次一定比 raw 小；Softmax 也不保证长程不会集中，需观察 N_eff。
**只有六次 smoke 检查通过后**，再显式启动正式实验：

```bash
python run_apa_logit.py --device-id 0 --rounds 100
```

独立脚本只转发共享 launcher；默认只跑 smoke，不自动继续完整实验。
共享 launcher 原四模式和 100 轮默认值不变。公共训练设置仍为 CIFAR-100/pat_20、20 clients、
全参与、C0、Decom_CNN-5-512、local epochs 5、batch 16、client lr .005、正则 .001、seed 0。
主要比较 `target_post_local_acc`，不要把 proxy loss 下降或聚合后准确率提高视为 personalized 收益。

## C0 FedDWA 改造：guidance 距离评分

`dwa_soft` / `dwa_soft_projection` 参考 [FedDWA（IJCAI 2023）](https://www.ijcai.org/proceedings/2023/0444.pdf)
的前瞻 guidance 与模型距离思路，按本次 C0 实验定义实现。
这是 C0 改造版，不是原论文完整复现：只生成 C0 guidance，仍下发同一个服务器 full-W 模型，
不为全部客户端维护个性化服务器模型，不做论文中的 Top-K。

每轮时序为：

```text
所有客户端正常 5 epochs → 普通 C0 post-local test
→ C0 低秩 post-local 副本训练 1 完整 epoch → recovery → guidance 参数上传
→ 投影前普通上传与 guidance 计算距离 → 分配权重 → 聚合
→ C0 正常低秩下载聚合模型 test
```

`clientTargetProj.build_dwa_guidance()` 只在新模式中调用，原 `_objective`、`train`、SGD、
下载和评价方法保持原行为。guidance 使用当前普通 checkpoint 的深拷贝，在低秩副本上
执行完整原 train loader 的一个 epoch（保留原 batch/drop-last 规则），使用相同 lr、CE+正则和梯度范数上限 10。
不调用正常 `train()` 来计数，不写普通 checkpoint，不用 test loader，不计算 guidance accuracy。
副本训练结束后才恢复 full-W。整个分支（加载、训练、恢复）保存/恢复 Python、NumPy、
Torch CPU 和所有 CUDA RNG，异常路径也恢复。普通模型的参数、buffers、train-time 计数保持原值。

服务器为每个本轮成功完成普通训练的上传记录 loop round；guidance 同时携带 client ID、
guidance loop round 与 source post-local round。只允许三个来源都与当前轮一致，聚合成功后
guidance 标记失效，不能重复使用或使用上一轮 guidance。`dwa_guidance.pt` 每轮覆盖为最新评分上传，
不缓存跨轮 basis，不维护可训练权重、logits 或 EMA。

`system/utils/dwa_aggregation.py` 是独立聚合模块；两模式共用 `squared_parameter_distance()`
和 `guidance_distance_weights()`。距离覆盖与原 Avg 相同的 recovered `named_parameters()`，包含 head，
排除 buffers；不比较低秩 U/V 因子或客户端 alignment 参数。

```text
s_j = sum_named_parameters ||W_guide - W_j_post||_F²
r_j = 1 / (s_j + dwa_distance_eps)
q_j = r_j / sum_helpers(r)
alpha_C0 = 0.05
alpha_helpers = 0.95 * q
```

默认新参数 `--dwa_distance_eps 1e-12`。`s_j` 已经是平方距离，不再平方。
逐参数先转换 float64，再相减、平方、归约；只逐个读取客户端，不拼接模型或常驻全部 uploads。
倒数归一化先等比例缩放以避免倒数溢出/总和溢出，不更换评分规则，无 Softmax、temperature、
ReLU、筛选、同标签先验或 fallback；遇到不可表示的下溢/非法距离明确报错。
C0 不参加 helper 归一化；sample-count weights 只保留用于上传检查与日志，不与距离评分相乘。
20 个客户端所有 helper 距离相等时，各最终权重均为 .05（浮点精度内）。

`dwa_soft` 直接逐参数累积 `sum_i alpha_i * W_i_post`，不调用 Projection 核。
`dwa_soft_projection` 先计算同一组**投影前**距离和权重，再调用未修改的
`aggregate_target_updates(..., mode='projection')`。
其参考方向仍为普通 C0 的 `W0_post - W_global_before_aggregation`，epsilon=原 1e-12，
只处理 helper 的负向平行分量，保留正交分量；不是 guidance delta 或低秩 pre/post local delta。
为聚合而第二次恢复上传时保留随机状态，不发生新的客户端通信。

两个模式的主对照分别为 `softmax_only` / `projection_softmax`。评分变化伴随额外的
C0 guidance 计算与上传，不能把这部分成本隐去。当前实现上传额外 full-W 参数字典：

- `guidance_extra_train_seconds`：C0 副本的一个 epoch，包括 loader 和训练步，CUDA 同步后计时。
- `guidance_recovery_seconds`：恢复 full-W 与 CPU 参数打包时间。
- `guidance_extra_upload_bytes` / `guidance_extra_upload_parameters`：额外 full-W 参数张量负载。
- `ordinary_target_low_rank_parameter_bytes`：普通 C0 低秩参数大小，便于比较增量。
- `guidance_serialized_record_bytes`：本地 guidance 记录的实际文件大小，含序列化/元数据开销。

这是仓库的本地 checkpoint 通信模拟；张量字节数不包含网络协议开销，记录大小不等于真实网络流量。
guidance 是评分上传，不作为第 21 个客户端或普通 C0 上传参与聚合；guided buffers 不上传或聚合。

各 mode 保存独立文件并导出到 final-model 目录：

- `<mode>_metrics.csv`：轮次/seed、普通 C0 主/聚合后准确率、guidance 训练规则和来源轮次、
  epsilon、guide-target 平方距离、固定质量、helper min/max/mean/std、N_eff、C1–C3/C4–C19 质量及上述成本。
- `<mode>_weights.csv`：每轮每客户端的普通 sample weight、helper 平方距离、q、实际 aggregation weight，
  Projection 版本额外包含 conflict、投影前后 dot、delta norm 与 removed-component norm。
- `<mode>_history.json`：完整记录、字段语义及 final/last10/best summary。
- 公共 CSV/JSON/H5 继续保存，H5 增加 `[round, client]` 的距离、q、实际权重；
  target 的 helper 距离/q 记空值/NaN，guide-target 距离使用单独字段。

`last10_target_local_acc` 是最后十次**普通 post-local** 的平均值，短 smoke 使用已有次数并记录
`last10_target_local_count`；best 轮次为 1-based、并列取最早。正式 `-gr 100` 对应 101 个普通 post-local 值。
所有分析继续使用 `local_t → aggregate_t → local_(t+1)`；不拿 guidance 6-epoch accuracy 替代主指标。
同/跨标签质量仅按此协议的 C1–C3/C4–C19 ID 分组统计，不参与选权。

在 `system/` 先运行短 smoke（`-gr 2` 实际三次普通训练/聚合，每轮正常 5 epochs）：

```bash
python run_target_proj.py --modes dwa_soft dwa_soft_projection --rounds 2 --device-id 0 --dwa_distance_eps 1e-12
```

正式配置保持 CIFAR-100/pat_20、20 clients/full participation、C0、原 Decom_CNN-5-512 容量分配、
local epochs 5、batch 16、SGD .005、正则 .001、seed 0、100 参数轮（101 次普通训练/聚合）：

```bash
# 如果先跑一个，优先 DWA 无投影版本：
python run_target_proj.py --modes dwa_soft --rounds 100 --device-id 0 --dwa_distance_eps 1e-12
python run_target_proj.py --modes dwa_soft_projection --rounds 100 --device-id 0 --dwa_distance_eps 1e-12
# 仅查看完整命令，不启动训练：末尾加 --dry-run
```

日志位于 `system/target_proj_runs/<timestamp>/<mode>/train.log` 和该目录的 `checkpoints/`，
H5 在 `h5_results/`，导出文件在 `final_models/`。默认旧四模式不变，不自动开启新实验。

## 统一主评价：Client 0 post-local accuracy

所有十九个模式每次完成全部普通本地训练后、`receive_ids()` 和聚合之前，调用
`clientTargetProj.test_post_local()`，只读取 Client 0 当前 checkpoint。
不下载服务器模型、不恢复 full-W、不写 checkpoint、不创建或更新优化器。
使用现有 `shuffle=False` 的 test loader；推理使用 no-grad/eval，结束后恢复每个子模块的 train/eval 状态。
Python/NumPy/PyTorch CPU 和 CUDA RNG 在整个加载、读取及推理过程前后恢复，异常路径也恢复。
准确率不传入任何聚合核，不参与权重或 helper 选择。

每轮记录 `target_post_local_acc`，不受 `eval_gap` 影响，`-gr 100` 对应 101 个有效值，包括最终一次训练。
`final_target_local_acc` 取最后值，`best_target_local_acc` 取最大值；
`best_target_local_round` 使用 1-based 完成本地训练次数，最大值并列时取最早一轮。

- `target_proj_metrics.csv`：逐轮准确率及截至本轮的 final/best/round；最终行即最终汇总。
- `target_proj_metrics.json`：完整 `history` 和最终 `summary`，`primary_accuracy_metric=target_post_local_acc`。
- H5 `target_projection`：逐轮同名 datasets；最终三个指标也写为同名 attributes。
- 各模式独立 metrics/matrices 文件也保存主指标，导出模型目录包含这些文件的副本。

训练结束明确打印 Final/Best Client 0 post-local accuracy 和 best round。
`main.py` 对 FedTargetProj 跳过原基于 `rs_test_acc` 的 generic best-accuracy 汇总，
避免在最后输出中把全客户端平均值当成项目主性能；其他算法的汇总保持不变。
旧 `target_client_test_acc`、旧 JSON/H5 `accuracy_scope` 保留原定义；新增独立
`target_post_local_accuracy_scope`，避免把两个时机混淆。全客户端平均准确率继续保留为诊断。

## 本批完整实验与 GPU 调度

冻结配置：CIFAR-100/pat_20、20 clients、全参与、Client 0、Decom_CNN-5-512、local epochs=5、
batch=16、SGD lr=.005、regularization=.001、seed=0、`-gr 100`；梯度裁剪及 rank ratios 沿用原实现。
本地训练函数、模型配置和旧聚合核均不修改，诊断不会影响训练随机流。

在服务器 `system/` 执行（由用户自行同步、启动；GPU ID 按实际空闲卡调整）：

```bash
python run_target_proj.py \
  --modes projection_local layer_projection_global layer_projection_local \
          projection_same_label projection_cross_label layer_softmax layer_relu \
  --rounds 100 --parallel --device-ids 0 1 2 3 4 5 6
```

`--device-ids` 只控制调度，每张 GPU 同时最多运行一个 mode，任务超过卡数时等待空闲卡。
单卡旧参数 `--device-id 0` 保持兼容；单卡配合 `--parallel` 也不会把多个实验同时塞入该卡。
默认仍为原四个 modes，不隐式启动新增实验。每个实验使用独立进程和时间戳/mode 目录，
保存实际设备分配后的 `command.json`、`train.log`、checkpoints/H5/final_models。
`--dry-run` 仅展示轮转设备的预览命令，实际排队由可用 GPU 决定，不改变训练配置。

## 数据与运行

使用具备仓库基础依赖（PyTorch、torchvision 等）的 Python 环境，无需安装或下载 CLIP。
`main.py` 按所选算法延迟加载 server，避免其他算法引入 CLIP 依赖。
数据只需生成一次，四个实验复用同一组切分：

```bash
cd dataset
python generate_Cifar100.py --niid 1 --partition pat --cpc 20
cd ../system
python run_target_proj.py --rounds 100 --device-id 0
```

本批保持 `--rounds 100`；可用 `--dry-run` 仅显示完整命令。
启动器按 `avg → target_only → projection → layer_mask` 顺序运行，各使用独立进程和输出目录。
本轮默认模型改为 `Decom_CNN-5-512`，四组统一使用同一异构 CNN；保留 5 local epochs、
batch 16、lr 0.005、regularization 1e-3、Client 0、seed 0、20 clients、pat_20、全参与。
只运行新模式可用 `python run_target_proj.py --modes layer_mask --rounds 100`；
只重跑旧三组可用 `--modes avg target_only projection`。
如需复现之前的 ResNet 配置，再显式加 `--model-family Decom_resnet18_5`。

单独运行某个模式（在 `system/`）：

```bash
python main.py -algo FedTargetProj --target_proj_mode layer_mask --target_client_id 0 --seed 0 -t 1 -data Cifar100 -ncl 100 -nc 20 -niid 1 -pt pat -cpc 20 -jr 1.0 -m Decom_CNN-5-512 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -regular_lamda 1e-3 -did 0
```

`simple_v` 的循环是 `range(global_rounds + 1)`：`-gr 100` 实际执行 101 次本地训练和聚合。
本实验保留这个行为，并固定使用 `-gr 100` / 启动器 `--rounds 100` 与既有长程实验比较。
新增日志的 `round` 从 1 开始，表示已完成的聚合次数，`loop_round` 对应原循环编号。

模型初始化继续使用基线构造器的固定 seed 0；`--seed` 固定 Python、NumPy、PyTorch
训练随机流，构造前后各设置一次。四个独立进程使用相同 seed 和配置；没有改动基线
CUDA 确定性设置，因此不承诺不同 GPU/软件版本间逐位复现。

## 指标及解释

每次聚合写入工作目录（启动时打印 `metrics_dir`）：

- `target_proj_metrics.csv` / `target_proj_metrics.json`：目标准确率及轮级诊断。
- `target_proj_clients.csv`：逐客户端权重、是否冲突、投影前后内积、更新范数和删除范数。
- 原 H5 文件新增 `target_projection` 组；最终模型导出目录同时复制三份诊断文件。

`target_client_test_acc` 测量 **本轮聚合模型按正常低秩下发后、尚未进行下一轮本地训练时**
在目标客户端测试集的准确率（0–1）。使用分类预测、低秩下发流程和本地 buffers；
测试后恢复本地 checkpoint 及 RNG，避免干扰后续训练。按 `eval_gap` 记录并额外记录最终轮。
未评估轮在 CSV 留空、JSON 为 null、H5 为 NaN。原 `rs_test_acc` 仍是继承的本地模型指标，
不能把它误当作这里的目标准确率。

旧三个模式每轮都计算同一套 **假设执行 projection 时** 的诊断：
`conflict_client_ratio`、`removed_update_ratio`、`avg_target_cos`、`proj_target_cos`。
因此 avg / target_only 的 removed_update_ratio 也可能非零，它表示本轮投影会删除的比例，
不是这两个模式实际执行了投影。删除比例为其他客户端删除范数之和除以原范数之和，
不按样本量加权。附加 target/avg/proj 范数和冲突数量帮助解释零更新等退化情况。
`dot_after_projection` 用投影公式解析计算；含 epsilon 和浮点误差时不强制等于零。

比较相同完整实验配置下的 Client 0 final/best post-local accuracy，结合冲突率、删除比例和 cosine。
`target_client_test_acc` 仅为 post-aggregation 诊断。算法保证删除负向平行分量；
不保证 cosine 必然提高，也不保证准确率必然提升。

## 本地验证

```bash
python -m unittest discover -s tests -p "test_target_projection.py" -v
python -m unittest discover -s tests -p "test_layer_mask.py" -v
python -m unittest discover -s tests -p "test_layer_mask_budget.py" -v
python -m unittest discover -s tests -p "test_layer_weighting.py" -v
python -m unittest discover -s tests -p "test_layer_mask_cnn_runtime.py" -v
python -m unittest discover -s tests -p "test_target_proj_launcher.py" -v
python -m unittest discover -s tests -p "test_projection_variants.py" -v
python -m unittest discover -s tests -p "test_projection_weighting.py" -v
python -m unittest discover -s tests -p "test_softmax_only.py" -v
python -m unittest discover -s tests -p "test_apa_aggregation.py" -v
python -m unittest discover -s tests -p "test_apa_logit_aggregation.py" -v
python -m unittest discover -s tests -p "test_dwa_aggregation.py" -v
python -m unittest discover -s tests -p "test_target_post_local.py" -v
python system/run_target_proj.py --dry-run
```

测试涵盖三模式数学结果、原 Avg 数值对齐、整模型投影、零/微小目标更新、样本权重、
不修改上传和 buffer、目标评估恢复 checkpoint/RNG、真实 CE+正则反向更新、统一学习率，
以及独立训练循环中三种模式的合成模型冒烟测试。
真实 CIFAR-100 收敛表现需完成上述四组实验后判断。

当前专用测试 14 项通过；已验证在没有 `clip` 的 `htfllib` 环境下独立导入 server/client。
聚合诊断、存储隔离和客户端初始化回归共 19 项通过。另用实际低秩 ResNet（缩小通道数）
和合成图像测试两个异构客户端（rank ratio 0.5/0.25）：三个模式各训练两轮，
验证相同初始化、真实本地反向传播、低秩下发、聚合、评估、H5 与模型导出，全程未加载 CLIP。
本地尚无 `Cifar100/pat_20` 切分，因此未执行真实 CIFAR-100 训练。

LayerMask 验证：新增 8 项数值/快照/日志测试、1 项实际 CNN 集成测试通过，旧 14 项测试继续通过。
实际 CNN 测试使用 rank ratio 0.9/0.15 的两个客户端及合成 32×32 图像，运行两轮，
验证恢复后层名/形状对应、W_pre 与实际下发模型一致、本地训练、聚合、评估及 H5/JSON/模型导出。

Helper Budget 验证：8 项新增预算数值/保存测试通过，旧 projection 14 项、旧 layer_mask 8 项
继续通过；实际异构 CNN 的 layer_mask 和 layer_mask_budget 两项两轮集成测试通过，共 32 项。
另与修改前的 layer_mask 做 10 组随机 20-client 对照（float32/float64），参数与诊断逐位一致。
尚未运行真实 CIFAR-100 的 budget 收敛实验。

连续加权验证：11 项新数值/日志/训练循环测试覆盖正方向权重、固定温度、ReLU 近零极限、
质量守恒、空集合与微小 cosine、完整 anchor、无 budget、RNG 保护和文件隔离。
旧三个聚合核及客户端文件未改动，旧 30 项单元测试继续通过。
实际异构低秩 CNN 集成测试另外覆盖 softmax / ReLU 两轮训练、快照、评估和文件导出；
启动器测试检查原四模式默认值及两组新模式的并发调度和配置一致性。
本地缺少 `Cifar100/pat_20` 切分，尚未进行两组真实 CIFAR-100 的 30 轮实验，不能据此比较收敛收益。

本次 Projection variants 验证共 **76 项全部通过**：原 projection 14、LayerMask 8、Budget 8、
Softmax/ReLU 11，新 variants 19、post-local 评价 4、launcher 5、真实异构 CNN 集成 7。
新 variants 覆盖无冲突逐位 Avg、异构 pre-local 结构保留、单层修正与正交分量保留、零 target、
单 group 等价、source 配置保护/质量守恒/排除参数无影响、五模式训练与独立文件导出。
对旧提交 `a2ab191` 的 projection 做 10 组随机 20-client float32/float64 对照，
参数与原诊断逐位一致。post-local 测试覆盖 101 个有效值、最终/最佳/并列轮次、CSV/JSON/H5、
失败恢复与下次训练结果逐位不变。实际 CNN 测试还以 checkpoint SHA-256 验证评估不写文件。
launcher 测试验证 7 组任务排队到 2 张卡、单卡旧参数和冻结的 `-gr 100`。
按用户后续指示，只完成本地开发和验证，不推送远程，不在服务器启动训练。

Projection similarity weighting 验证：本次新增 **18 项测试**（15 项数学/完整客户端流程、
2 项真实异构 CNN 集成、1 项 launcher 冻结参数检查），相关测试总计 **94 项全部通过**。
覆盖纯 similarity 而非 sample×similarity、Softmax 负 cosine 正权重、ReLU 非正删除和不均匀样本 fallback、
target/helper mass、固定温度、微小与零更新、project-then-weight 独立解析结果、RNG 与输入不变、
无 pre-local/group、Client 0 主指标、独立文件与 H5 导出。旧十二个模式的测试继续通过；
旧 `projection` 再次与 `a2ab191` 在 10 组随机模型上验证参数/诊断逐位一致。
本次没有修改本地训练、旧聚合核、公共超参数，也没有运行正式 CIFAR-100 收敛实验。

SoftmaxOnly 验证：新增 **11 项测试**（10 项数学/回归/启动脚本测试、1 项真实异构 CNN 集成），
相关测试共 **105 项全部通过**。覆盖权重质量、负方向保留、禁止调用 Projection 核、
与 ProjectionSoftmax 的 cosine/weights 逐位一致及结果只差 projection correction、
零/微小更新、输入与 RNG 不变、mean/std 日志、主准确率/H5/CSV/JSON/模型导出和冻结的完整启动配置。
与 `c12bfc5` 的 `projection_softmax/projection_relu` 做 12 组随机 float32/float64 对照，
参数及诊断全部逐位一致；旧 `projection` 与 `a2ab191` 的原回归仍通过。
正式服务器实验未启动，本地只执行合成测试和启动脚本 dry-run。

APA 验证：新增 **17 项测试**（15 项梯度/优化器/因果 basis/日志/启动脚本测试、
1 项真实异构 CNN 两轮集成、1 项共享 launcher 参数隔离），TargetProj 相关 **122 项全部通过**。
覆盖手工梯度与 autograd 一致、beneficial/harmful 方向、首轮逐位 Avg、跨轮 momentum、
clip→self→normalize、uniform fallback、零 residual、NaN/Inf、RNG 不变、旧 basis 因果性及失效检查，
以及低秩下发、本地反向传播、post-local 评价、独立 CSV/JSON/H5 和最终模型导出。
旧聚合核和 `clientTargetProj.py` 与本次修改前逐字不变，CLI/dry-run/语法检查通过。
仓库全部 20 个测试模块分别在独立进程运行，共 **197 项通过**。
其中旧 `test_tsne_accuracy.py` 在本机有 OpenMP 库冲突，使用已激活的 Conda 环境，
仅对该测试进程设置 `MKL_THREADING_LAYER=SEQUENTIAL` 后 19 项通过；未改动仓库或正式训练环境配置。
本次没有运行正式 CIFAR-100 收敛实验，APA 的 final/peak accuracy 待测。

APA-Logit 验证：新增 **15 项测试**（13 项数学/服务器/启动脚本、1 项异构 CNN 集成、
1 项共享 launcher 隔离），TargetProj 相关共 **137 项通过**；仓库 21 个测试模块独立运行，
共 **212 项通过**。原 APA 15 项测试全部通过，旧聚合模块及客户端文件与 `b682f19` 相同。
新测试覆盖手工梯度/autograd/两种 centered 公式一致、1e8 公共偏移抵消、零 logits、
float32/64 首轮逐位 Avg、beneficial/harmful helper、600/6000 共同梯度不塌缩、无 momentum、
严格上一轮 basis、零 residual、RNG/输入不变、非法/下溢 fail-fast 和日志隔离。
20 个合成客户端实际执行六次本地训练/聚合，验证两组固定质量和 19 维 logits；
另用真实低秩 CNN 的三个 rank (.9/.15/.5) 验证两轮训练和 CSV/JSON/H5/模型导出。
这两项均为合成数据代码验证；本地缺少 `Cifar100/pat_20`，尚未运行用户要求的真实数据六次 smoke，
更未运行 100 轮完整实验，不据此报告 APA-Logit 的准确率收益或真实梯度尺度。

C0 DWA 验证：新增 **16 项测试**（13 项数学/guidance/服务器测试、2 项异构 CNN smoke、
1 项启动器冻结配置检查），TargetProj 相关 **153 项通过**；仓库 22 个模块独立运行，
共 **228 项通过**。两模式使用 20 个合成客户端执行三次普通 5-epoch 训练/聚合及 C0 独立 1-epoch guidance，
另以真实低秩 CNN 的 rank .9/.15/.5 各运行两次普通 5-epoch 训练，验证参数恢复、
checkpoint SHA-256 不变、guidance RNG 隔离、普通准确率时序和 CSV/JSON/H5/final-model 导出。
覆盖等距离均匀、零距离/零 target、float64 距离、两版本相同权重、原 Projection 数值、过期轮次与布局检查，
guidance 完整 epoch 与手工 CE+正则 SGD 一致、失败恢复、buffers 保护、下一次普通训练逐位不变及 last10/best 统计。
原所有客户端方法逐一通过 AST 对照；旧聚合模块保持未改动，旧模式测试继续通过。
CLI、语法、diff 检查和正式 100 参数轮 dry-run 通过；旧绘图测试沿用本机临时 MKL 环境设置。
本地仍无 `Cifar100/pat_20`，本批只执行合成 smoke，未自动启动完整实验或报告真实 DWA 准确率。
