# `new_agg_v1` 分支讨论与实现交接文档

> 用途：把本分支的设计背景、公式、实现状态、诊断方法和实验入口迁移到新的对话。
> 新对话继续工作前，应先阅读本文，并重新核对当前分支和提交号。

## 1. 分支身份与目标

- 分支：`new_agg_v1`
- 本文整理时的提交：`c64ac3d`（`修正范数问题`）
- 远端：`origin/new_agg_v1`
- 核心目标：研究一套不依赖原 FedCLIP 客户端相似度加权的新聚合框架，重点比较客户端真实更新的 Avg、公共子空间投影、方向一致性和符号个性化聚合。
- 当前新聚合路径只针对 `Decom_CNN-5-512` 完整实现。
- ResNet 遇到非 `avg` 模式时会明确打印提示并回退到 FedAvg，不能把当前结果解释为 ResNet 新聚合结果。

本分支不是主分支方法的简单参数调整，而是一条独立的聚合研究线。主分支原来的低秩 V 相似度、层融合、自保留等逻辑不要和本文中的新模式混为一谈。

### 1.1 讨论与提交演进

这条研究线不是一次完成的，主要演进如下：

1. `a0f70a6 初版`
   - CNN 移除秩 Dropout 和 U/V 非对称学习率；
   - 大幅清理旧聚合代码；
   - 实现未中心化公共子空间投影、全局公共更新和客户端残差。
2. `18aa73d 小bug`
   - 修复初版模型恢复、设备或接口细节。
3. `19aac8c 去点个性化残差的EMA`
   - 增加 current-round residual 路径；
   - 可关闭历史 EMA，仅用当轮残差作为临时个性化补偿。
4. `bb89a3f 增加新的avg对照`
   - 新增独立 `DeltaAvg`；
   - 强调 delta 必须是“上传满秩有效权重减客户端本轮实际训练起点”。
5. `e0f164b 新的聚合策略`
   - 新增方向一致性投影和符号一致性个性化投影；
   - 从“统一公共方向”转向“每个目标客户端按奇异方向选择同号来源客户端”。
6. `15af259 增加自身参与度决定融入强度`
   - 在符号同组系数外乘目标客户端方向强度 `g_{i,k}`；
   - 避免客户端几乎不含某方向时仍完整接收该方向。
7. `6adac2b 增加大量日志打印`
   - 增加 K 覆盖率、符号放大、g 缩放、正负抵消、K5/K10/KR 和第 6 至 10 方向增量等诊断；
   - 完整结果写入两个 CSV，控制台只抽样打印。
8. `c64ac3d 修正范数问题`
   - 新增模式 A：保留组内归一化并恢复整体范数；
   - 新增模式 B：取消组内归一化后再恢复整体范数；
   - 原有模式保持不变，便于严格对照。

这段演进对应三个连续追问：公共方向能否减少冲突，符号条件聚合能否提供个性化，以及符号分组和 `g` 是否又造成了不合理的幅值放大/缩小。

## 2. 最初的大改范围

本分支最初按“最小可跑、减少干扰变量”的原则重写 CNN 路径：

1. 删除旧 `serverCLIP.py` 中大量不再使用的旧聚合实现，只保留 Avg 和新投影路线需要的模型恢复、分解、保存等基础设施。
2. CNN 的低秩层关闭 ordered/rank dropout：
   - `Hyper_CNN_512` 构造和重新分解后都会执行 `_disable_rank_dropout()`；
   - `clientCLIP.train()` 在 CNN 路径再次关闭相关开关，防止旧模型对象残留状态。
3. CNN 取消 U/V 非对称学习率，使用：

   ```python
   torch.optim.SGD(model.parameters(), lr=self.learning_rate)
   ```

4. ResNet 不在这次清理范围内：仍可使用原来的多阶段 CLIP 对齐、U/V 非对称学习率和秩 Dropout 配置。
5. CNN 的 CLIP 文本锚点 MSE、分类交叉熵、可选 Frobenius 正则和梯度裁剪仍然保留。

因此，本分支 CNN 实验和主分支默认 FedCLIP 并非只差聚合器；它还关闭了 CNN 的秩 Dropout 和非对称学习率。做论文归因时必须说明这一点，或者额外设计严格对照。

## 3. 模型生命周期与真实 Delta

### 3.1 客户端本轮训练起点

客户端接收服务器模型后，服务器模型按该客户端 `ratio_LR` 分解，再覆盖客户端低秩模型。客户端同时将这个“实际下发后的低秩起点”保存到：

```text
<save_folder_name>/low_rank_start/Server_model_<client_id>.pt
```

对于依赖严格 delta 的模式，服务器在 `send_parameters()` 完成后保存该客户端实际训练起点的满秩有效权重：

```text
S_i^t
```

### 3.2 上传后恢复

客户端训练完成后上传的是低秩模型。服务器先恢复为满秩有效权重：

```math
\widehat W_i^{t+1}=\widehat U_i^{t+1}\widehat V_i^{t+1}.
```

真实客户端更新必须计算为：

```math
d_i=\operatorname{vec}(\widehat W_i^{t+1}-S_i^t).
```

不能减服务器当前模型，也不能用其他客户端的起点。

### 3.3 起点缺失策略

- `delta_avg` 和三个符号个性化模式要求本轮内存中的 `client_start_full_weights` 存在；缺失时直接报错，防止 delta 错位。
- 较早实现的公共投影/方向一致性辅助路径会优先读取 `low_rank_start`；缺失时打印 warning，再回退到服务器保存的个性化模型，最后才回退到通用服务器模型。
- 如果实验涉及断点恢复，应重点检查起点缓存是否与当前轮次对应。

## 4. Warm-up 规则

所有非 Avg 模式共享：

```text
--projection_warmup_ratio
```

实现为：

```python
warmup_rounds = round(global_rounds * projection_warmup_ratio)
is_warmup = current_round <= warmup_rounds
```

例如总轮数 100、比例 0.2 时，第 0 至 20 轮使用普通 Avg，第 21 轮开始进入指定的新聚合模式。

## 5. 当前聚合模式总览

入口参数：

```text
--aggregation_mode avg
--aggregation_mode delta_avg
--aggregation_mode projection
--aggregation_mode consensus_projection
--aggregation_mode sign_personalized_projection
--aggregation_mode sign_projection_norm_restore
--aggregation_mode sign_projection_no_group_renorm
```

如果不传 `--aggregation_mode`，当前兼容逻辑由 `--use_common_residual_projection` 决定；其默认值为 1，因此默认进入 `projection`，不是 Avg。正式实验建议总是显式传模式名。

### 5.1 `avg`

1. 将客户端低秩模型恢复到满秩有效权重。
2. 对整个模型按客户端样本量权重直接平均。
3. 所有客户端保存相同的聚合模型。

这是满秩有效模型 FedAvg 对照，不计算客户端训练 delta。

### 5.2 `delta_avg`

对可低秩恢复的 weight 层：

```math
\bar d=\sum_i\alpha_i d_i,
\qquad
G^{t+1}=G^t+\operatorname{reshape}(\bar d).
```

其中：

```math
\alpha_i=\frac{n_i}{\sum_jn_j}.
```

不计算 Gram、SVD、投影或 residual。非低秩参数按上传后的完整参数普通加权平均。最终所有客户端收到相同的更新后全局模型。

它是判断“从各自下发起点计算更新”本身是否有用的关键对照。

### 5.3 `projection`

先对每层客户端原始更新归一化并按样本量加权：

```math
\widetilde d_i=\sqrt{\alpha_i}\frac{d_i}{\lVert d_i\rVert_2+\epsilon},
\qquad
D=[\widetilde d_1,\ldots,\widetilde d_m].
```

通过客户端维度 Gram 矩阵：

```math
K=D^\top D
```

恢复左奇异方向。使用 `projection_energy` 选择累计能量达到阈值的最小 K，并受 `projection_k_max` 限制。

原始更新分为公共投影和正交残差：

```math
d_{g,i}=P d_i,
\qquad
d_{p,i}=d_i-d_{g,i}.
```

服务器公共更新：

```math
\bar d_g=\sum_i\alpha_i d_{g,i}.
```

残差有三种行为：

- `--projection_use_residual 0`：不保留残差；
- `--projection_use_residual 1 --projection_residual_ema 0`：仅使用当轮残差，系数为 `personal_residual_beta`；
- `--projection_use_residual 1 --projection_residual_ema 1`：使用历史 EMA，参数为 `personal_residual_mu`、`personal_residual_gamma`，并可通过 `personal_residual_clip` 裁剪。

### 5.4 `consensus_projection`

沿用相同的 delta、SVD、能量筛选和 K 上限。对候选方向 `u_k` 计算：

```math
a_{i,k}=\left\langle\frac{d_i}{\lVert d_i\rVert+\epsilon},u_k\right\rangle,
```

```math
c_k=
\frac{\left|\sum_i\alpha_i a_{i,k}\right|}
{\sum_i\alpha_i|a_{i,k}|+\epsilon}.
```

再对普通平均更新进行一致性衰减：

```math
d_g=\sum_{k=1}^{K}c_k\langle\bar d,u_k\rangle u_k.
```

不保留 residual，不生成按客户端不同的投影更新。非低秩参数仍普通平均。

### 5.5 `sign_personalized_projection`

这是后续讨论最多的个性化模式。

SVD 仍基于样本量加权的归一化更新矩阵。设右奇异向量中客户端 i 在方向 k 的系数为 `v_{i,k}`，同号掩码为：

```math
m_{ij,k}=\mathbf 1(v_{i,k}v_{j,k}>0).
```

来源客户端原始更新在左奇异方向上的系数：

```math
a_{j,k}=\langle d_j,u_k\rangle.
```

同号组按样本量重新归一化：

```math
b_{i,k}=
\frac{\sum_j\alpha_jm_{ij,k}a_{j,k}}
{\sum_j\alpha_jm_{ij,k}+\epsilon}.
```

后来增加了目标客户端自身对方向的参与强度：

```math
g_{i,k}=\left|\left\langle
\frac{d_i}{\lVert d_i\rVert+\epsilon},u_k
\right\rangle\right|,
\qquad g_{i,k}\in[0,1].
```

最终个性化更新：

```math
q_i=\sum_{k=1}^{K}g_{i,k}b_{i,k}u_k.
```

服务器全局模型仍使用 DeltaAvg 更新：

```math
G^{t+1}=G^t+\bar d.
```

但目标客户端可投影层的最终权重不是 `G+q_i`，而是：

```math
W_i^{t+1}=S_i^t+q_i.
```

这样避免把自身更新重复加到上传后模型。非投影参数继承本轮普通加权平均后的服务器参数。

### 5.6 `sign_projection_norm_restore`（模式 A）

保持 `sign_personalized_projection` 的同号组重新归一化和方向不变，只恢复整体范数：

```math
\gamma_i^{raw}=\frac{\lVert d_{avg}\rVert}
{\lVert q_i\rVert+\epsilon},
```

```math
\gamma_i=\min(\gamma_i^{raw},\gamma_{max}),
\qquad
d_i^{new}=\gamma_iq_i.
```

参数：

```text
--projection_norm_scale_max 2.0
```

目的：只判断原符号个性化模式下降中有多少来自 `g` 导致的整体更新范数过小，不改变方向和组内归一化方式。

### 5.7 `sign_projection_no_group_renorm`（模式 B，优先实验）

取消同号组内部再次除以同号组样本权重质量：

```math
\widetilde b_{i,k}=\sum_j\alpha_jm_{ij,k}a_{j,k}.
```

构造：

```math
q_i=\sum_{k=1}^{K}g_{i,k}\widetilde b_{i,k}u_k,
```

然后使用与模式 A 相同的 DeltaAvg 范数恢复和 `gamma_max` 限制。

目的：避免同号客户端较少时，组内重新归一化把本来在 Avg 中应被抵消的方向人为放大；再通过整体范数恢复，避免取消归一化后整体更新过小。

## 6. 非低秩参数处理

当前投影只作用于从低秩 `conv_u/conv_v` 或 `weight_u/weight_v` 可恢复出的满秩 weight 层。

- bias、BN、norm 等非投影参数不进入 Gram/SVD；
- 它们沿用上传模型的样本量加权平均；
- 符号个性化模式只覆盖个性化模型中的可投影 weight 层，其余参数使用服务器本轮普通平均值。

## 7. 诊断日志设计

### 7.1 打印时机

符号投影系列在：

1. warm-up 结束后的第一轮；
2. 此后每 10 轮；

执行完整诊断。

控制台只详细打印：

- `conv2.weight`
- `fc2.weight`
- 前 3 个本轮上传客户端

所有可投影层和所有上传客户端的完整指标写 CSV。

### 7.2 CSV 位置

CSV 存在本次实验的 `save_folder_name` 下，通常也就是对应的 `temp/<dataset>/FedCLIP/<timestamp>/`：

```text
projection_client_diagnostics.csv
projection_direction_diagnostics.csv
```

### 7.3 核心诊断问题

日志专门区分三类性能损失：

1. K 截断丢失了多少客户端自身方向；
2. 符号分组相对 DeltaAvg 引入了多大偏差或放大；
3. 乘 `g_{i,k}` 后又缩小了多少，以及范数恢复是否补回。

客户端级重点字段：

- `coverage_K5/K10`、`residual_K5/K10`
- 最大 g 所在方向及其是否被当前 K 选中
- 自身 delta、DeltaAvg、self projection、sign、g-sign 的范数和余弦
- 第 6 至 10 个方向增量与自身/Avg/K5 的余弦
- `norm_before_g`
- `norm_after_g_before_restore`
- `gamma_raw`、`gamma_used`
- `norm_after_restore`
- 恢复后与自身 delta、DeltaAvg 的余弦

方向级重点字段：

- `sigma`、单方向能量、累计能量
- `g`、`a_self`、`a_avg`
- `b_sign`、`g_times_b`
- 同号客户端数量和样本权重质量
- 正负客户端数量、权重质量、正负加权和
- `cancellation_ratio`
- `sign_amplification=|b|/(|a_avg|+eps)`
- `final_ratio=|g*b|/(|a_avg|+eps)`
- `weight_sum_before_g`、`weight_sum_after_g`
- 两种组系数 `group_coeff_with_renorm` 和 `group_coeff_without_renorm`
- 范数恢复后的方向系数

### 7.4 自动数值检查

当前实现会检查：

- 所有 delta、SVD 方向、系数和最终更新无 NaN/Inf；
- `g` 位于 `[0,1]`；
- `m_{ij,k}=m_{ji,k}`；
- 显著非零方向满足 `m_{ii,k}=1`；
- 直接内积计算 g 与 SVD 公式计算 g 的最大误差；
- Gram/SVD 重构误差是否合理。

## 8. 主要参数

```text
--aggregation_mode
--projection_warmup_ratio        默认 0.2
--projection_energy              默认 0.8
--projection_k_max               默认 5
--projection_norm_scale_max      默认 2.0，仅两种范数恢复模式使用
--projection_use_residual        默认 1，仅原 projection 使用
--projection_residual_ema        默认 0
--personal_residual_beta         默认 0.1
--personal_residual_mu           默认 0.9
--personal_residual_gamma        默认 0.5
--personal_residual_clip         默认 0.0
```

旧兼容参数：

```text
--use_common_residual_projection
```

正式对照建议不要依赖旧兼容参数，直接显式设置 `--aggregation_mode`。

## 9. 推荐实验矩阵

保持模型、随机种子、参与率、学习率、本地轮数、CLIP 损失和正则完全一致，只改变聚合模式：

1. `avg`
2. `delta_avg`
3. `projection --projection_use_residual 0`
4. `projection --projection_use_residual 1 --projection_residual_ema 0`
5. `consensus_projection`
6. `sign_personalized_projection`
7. `sign_projection_norm_restore`
8. `sign_projection_no_group_renorm`

当前讨论建议符号系列优先使用：

```text
--projection_energy 0.95
--projection_k_max 20
--projection_warmup_ratio 0.2
--projection_use_residual 0
```

注意：`projection_use_residual` 对符号系列没有实际聚合作用，但命令中保留为 0 有助于明确实验意图。

## 10. 完整启动命令

### 10.1 模式 A：保留组内重新归一化，只恢复范数

```bash
python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -did 1 -algo FedCLIP -is_regular 1 -mse_lamda 1 -Cos_lamda 0.0 -regular_lamda 1e-3 -niid 1 -pt dir -dir_alpha 0.5 -aggregate_tau 1 --aggregation_mode sign_projection_norm_restore --projection_norm_scale_max 2.0 --projection_energy 0.95 --projection_k_max 20 --projection_warmup_ratio 0.2 --projection_use_residual 0
```

### 10.2 模式 B：取消组内重新归一化，并恢复范数

```bash
python main.py -t 1 -lr 0.005 -jr 1.0 -lbs 16 -gr 100 -ls 5 -nc 20 -ncl 100 -data Cifar100 -m Decom_CNN-5-512 -did 2 -algo FedCLIP -is_regular 1 -mse_lamda 1 -Cos_lamda 0.0 -regular_lamda 1e-3 -niid 1 -pt dir -dir_alpha 0.5 -aggregate_tau 1 --aggregation_mode sign_projection_no_group_renorm --projection_norm_scale_max 2.0 --projection_energy 0.95 --projection_k_max 20 --projection_warmup_ratio 0.2 --projection_use_residual 0
```

### 10.3 其他模式只需替换

```text
--aggregation_mode avg
--aggregation_mode delta_avg
--aggregation_mode projection
--aggregation_mode consensus_projection
--aggregation_mode sign_personalized_projection
```

## 11. 修改文件与关键职责

### `system/main.py`

- 注册聚合模式和投影相关命令行参数；
- CNN 模型构造不再接收秩 Dropout 调度参数；
- ResNet 的旧秩 Dropout 参数仍保留。

### `system/flcore/clients/clientCLIP.py`

- CNN 关闭 rank dropout；
- CNN 使用统一 SGD；
- ResNet 训练逻辑仍保留旧策略；
- `set_parameters()` 负责接收满秩个性化/全局模型、按客户端秩率分解，并保存本轮低秩起点。

### `system/flcore/trainmodel/models.py`

- `Hyper_CNN_512` 构造及重新分解后关闭低秩层的 rank dropout；
- 原始低秩层类仍保留采样能力，供其他模型/分支使用，但本分支 CNN FedCLIP 不启用。

### `system/flcore/servers/serverCLIP.py`

- 聚合模式路由和 warm-up；
- 满秩恢复、客户端真实起点快照和 delta 构造；
- Avg、DeltaAvg、公共投影、方向一致性、符号个性化及两个范数恢复变体；
- 个性化模型保存；
- 完整投影诊断与 CSV。

## 12. 已完成的验证

在实现最后两个模式时完成过以下检查：

1. `python -m py_compile system/flcore/servers/serverCLIP.py system/main.py` 通过。
2. `git diff --check` 通过，仅出现 Windows CRLF 提示。
3. 使用合成 delta 回归验证原 `sign_personalized_projection`：
   - 原模式输出与改造前最大差异为 0；
   - 原模式的 DeltaAvg 全局更新差异为 0。
4. 模式 A 的范数恢复公式与独立计算最大差异为 0，且 `gamma_used<=2.0`。
5. 模式 B 输出无 NaN/Inf；选择的方向系数与“不做组内重新归一化”的公式一致。
6. 两个新模式不改变服务器使用的 DeltaAvg 全局更新。
7. 诊断 CSV 中恢复后方向系数满足 `g_times_b * gamma_used`，仅有 CSV 浮点序列化误差。

这些是代码级和合成数据验证，不等于完整联邦训练结果。两个新模式仍需在服务器上完成同种子、多次运行的性能与稳定性比较。

## 13. 已知限制与风险

1. 新模式只完整支持 CNN；ResNet 会回退 Avg。
2. CNN 同时关闭了秩 Dropout 和非对称学习率，若与主分支方法对比，变量不只一个。
3. `avg` 是恢复后完整模型平均，`delta_avg` 是相对各自训练起点的更新平均，二者语义不同。
4. 公共投影/consensus 的旧辅助路径允许 low-rank 起点缺失后 warning + fallback；严格实验应检查日志中没有该 warning。
5. 符号分组使用严格乘积 `>0`，接近零只用于日志标记，没有额外阈值。
6. SVD 实际通过客户端维度 Gram 矩阵实现；K 最大不会超过参与客户端数和数值正秩。
7. 模式 A/B 的范数恢复只恢复到该层 DeltaAvg 范数，并受 `gamma_max` 限制；它不保证与客户端自身 delta 同范数。
8. 个性化符号模式中，服务器全局模型和下发给参与客户端的可投影层采用不同构造：服务器是 DeltaAvg，客户端是 `S_i+个性化投影 delta`。
9. 未参与客户端继续保留从当前服务器全局模型复制得到的模型，不会获得本轮特定的符号个性化更新。
10. 详细诊断会增加服务器时间和磁盘写入，性能计时实验应关闭或单独剔除诊断轮。

## 14. 新对话建议先做的事

新对话开始时建议依次执行：

```bash
git branch --show-current
git log -1 --oneline
git status --short
python -m py_compile system/flcore/servers/serverCLIP.py system/main.py
```

然后先回答以下问题再继续设计：

1. 比较目标是相对 `avg`、`delta_avg`，还是主分支旧 FedCLIP？
2. CNN 关闭 Dropout 和非对称学习率是否作为固定实验前提？
3. 评价重点是最终精度、收敛速度、不同容量客户端收益，还是更新方向诊断？
4. 新改动是否必须保持现有七种模式完全不变并新增独立模式？
5. 是否需要把 CNN 新聚合迁移到 ResNet？若需要，必须先明确残差块/层映射和低秩可投影层边界。

## 15. 一句话总结

`new_agg_v1` 已经从“公共子空间 + 残差”逐步扩展为一套可对照的 CNN 更新聚合实验框架：以客户端真实训练起点构造满秩 delta，用 Gram/SVD 提取候选方向，再分别研究 DeltaAvg、方向一致性、符号同组个性化、目标方向参与强度 `g`、组内重新归一化以及整体范数恢复对性能的影响。
