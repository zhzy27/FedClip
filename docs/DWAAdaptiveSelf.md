# DWA 自适应 C0 质量消融

新增独立 `dwa_adaptive_self` 和 `dwa_adaptive_self_projection`，只改变权重分配。
代码在 `system/utils/dwa_adaptive_aggregation.py`；原 `dwa_aggregation.py`、
`clientTargetProj.py`、global-delta Projection 核均未修改。

## 实验和 guidance

所有客户端每轮仍从同一服务器 full-W 模型按原容量分解，再正常训练 5 epochs。
不延续各自上一轮普通 post-local 模型，不维护多个个性化服务器模型。
主指标仍是普通 C0 的 5-epoch `target_post_local_acc`。
随后复用现有 C0 guidance：低秩 post-local 模型副本额外训练一个完整 epoch，再恢复 full-W。
仍保护普通 checkpoint、buffers、训练计数和 Python/NumPy/Torch CPU/CUDA RNG，异常路径也恢复。
guidance 不读取 test loader，不使用测试准确率选权；C1–C19 没有额外训练。
仍进行每轮独立 guidance 与上传来源检查。

公共配置：CIFAR-100/pat_20、20 clients、全参与、C0、Decom_CNN-5-512、原容量分配、
5 normal epochs、batch 16、SGD .005、正则 .001、seed 0、`-gr 100`。
循环 `range(global_rounds + 1)` 保持不变，100 参数轮实际为 101 次普通训练/聚合。
guidance 额外计算和 full-W 参数上传继续单独记录。

## 归一化全部客户端

```text
s_i = sum_named_parameters ||W_C0_guide - W_i_post||_F², i=0..19
r_i = 1 / (s_i + dwa_distance_eps)
alpha_i = r_i / sum_all_clients(r)
```

epsilon 默认 `1e-12`，沿用 `--dwa_distance_eps`。距离包括 recovered full-W 分类器 head，
不比较低秩因子，不包括 buffers。逐参数 float64 距离归约，不拼接完整模型。
使用与旧 DWA 相同的稳定倒数距离缩放计算，没有第二次平方、Softmax、temperature 或其他评分。
所有客户端都参与，不加 Top-K、EMA、权重学习率、目标权重上下限或额外的首轮 Avg 强制规则。
全部距离相同时权重均为 .05；C0 可以高于或低于 .05，甚至接近 1。

`dwa_adaptive_self` 直接累积 `sum_i alpha_i * W_i_post`。
Projection 版先计算完全相同的投影前距离与权重，再复用原
`aggregate_target_updates(..., 'projection')`，参考普通 C0 的 global delta：

```text
delta_i = W_i_post - W_global_before_aggregation
W_next = W_global + alpha_0 * delta_0 + sum_helpers(alpha_j * projected_delta_j)
```

只删除冲突 helper 的负向平行分量，epsilon/零向量处理完全来自原核，
不使用 guidance delta，不做逐层 Projection。两版在同输入下权重相同。
新旧 DWA helper 之间的相对权重比例相同，只改变 target/helper 总质量。
不保存聚合权重优化状态；历史权重仅用于日志和统计。

## 接近 1 的稳定诊断

```text
helper_mass = fsum(alpha_j for j != 0)        # 不能写成 1-alpha_0
helper_q_j = alpha_j / helper_mass
effective_helper_count = 1 / fsum(helper_q_j²)
effective_all_client_count = 1 / fsum(alpha_i²)
```

C0 权重可能在 float64 中显示为 1，但 helpers 仍有可表示的极小正质量。
直接累加实际 helper 权重使概率、有效数量和同/跨标签份额仍然有限。
遇到超出 float64 表示范围的倒数权重下溢时明确报错，不静默删除客户端或切换评分。
模型聚合仍使用原模型 dtype；极小贡献可能低于模型自身的浮点精度。

## 日志与统计字段

沿用公共 `target_proj_metrics.csv/json`、clients CSV、H5、final-model 导出和 final/last10/best 评价。
新增模式各自保存 `<mode>_metrics.csv`、`<mode>_weights.csv`、`<mode>_history.json`。
不覆盖旧两组 DWA 文件。

新增/明确的每轮字段：

- `target_squared_distance` / `guidance_target_post_squared_distance`：guidance 与普通 C0 的平方距离。
- `target_weight`：实际 C0 权重；`helper_total_weight`：实际 helpers 权重之和。
- `effective_helper_count`：helper 内部归一化后计算；`effective_all_client_count`：全部客户端有效数量。
- `same_label_helper_weight` / `cross_label_helper_weight`：C1–C3/C4–C19 的绝对质量。
- `same_label_helper_share` / `cross_label_helper_share`：上述质量除以实际 helper 总质量。
- `target_weight_phase`：本轮 early/mid/late 标签。

每客户端行都记录 `guidance_squared_distance`、`aggregation_weight`、`all_client_q=alpha_i`。
`helper_q` / 沿用字段 `dwa_q` 表示 **helper 内部概率**，C0 为 null；它不是全客户端 alpha。
H5 保存全部距离/权重、helper `dwa_q` 和 `all_client_q` 矩阵，metadata 写明概率范围。
Projection 版继续保存 conflict、投影前后 dot、delta norm、删除幅度诊断。
所有 guidance 来源、损失规则、lr、训练时间、额外信息量字段保持原定义。

全程及分阶段统计保存于 JSON summary、公共 summary、H5 attrs 和逐轮 metrics：

- `target_weight_count/mean/min/max`：当前已完成轮次的累计统计。
- `early_target_weight_count/mean/min/max`，以及 `mid_...`、`late_...`。
- `target_weight_gt_0_5_count`、`target_weight_gt_0_8_count`、`target_weight_gt_0_95_count`：严格大于阈值的次数。

阶段固定锚定计划的 inclusive run：`phase = loop_round * 3 // (global_rounds + 1)`。
正式 101 次聚合的 early 为 loop 0–33，mid 为 34–67，late 为 68–100，
分别对应显示 Round 1–34、35–68、69–101；不会随历史长度增长重新划分已完成的轮次。
尚无样本的阶段 count=0、min/mean/max=null（H5 为 NaN）。短 smoke 按自身计划轮数同样分段。
阈值仅用于观察，绝不限制权重。

## 对照与启动

用户提供的旧实验主指标（百分比，未在本地重新跑正式实验验证）：

| 新模式 | 固定权重对照 | 对照 final | 对照 last10 |
|---|---|---:|---:|
| `dwa_adaptive_self` | `dwa_soft` | 43.07 | 42.61 |
| `dwa_adaptive_self_projection` | `dwa_soft_projection` | 43.73 | 43.55 |

主要比较普通 post-local 的 final 和最后 10 次平均，best/轮次为辅助。
如 C0 权重过高导致退化，应报告统计和真实性能，不在此批加入限幅或固定权重扫描。
继续按 `local_t → aggregate_t → local_(t+1)` 解读聚合与后续收益。

`system/dwa_adaptive_self_commands.sh` 提供与 `system/run_now.sh` 相同的 `declare -a COMMANDS=(...)`
结构和两条完整 main.py 参数，GPU 示例为 0/1，按机器实际情况调整。
文件只定义数组，不启动程序；既有 run_now.sh 的任务未修改。
可将选中的条目加入既有数组，由用户手动启动。其 train.log 使用 run_now.sh 创建的任务目录；
checkpoint/metrics 路径由启动日志的 `metrics_dir` 指明，默认位于 `temp/Cifar100/FedTargetProj/<run_id>/`。

在 `system/` 使用统一 launcher 的等价命令：

```bash
# 短轮 smoke：每客户端普通 5 epochs，实际三次普通训练/聚合
python run_target_proj.py --modes dwa_adaptive_self dwa_adaptive_self_projection --rounds 2 --device-id 0
# 正式命令（此批不会自动执行）
python run_target_proj.py --modes dwa_adaptive_self --rounds 100 --device-id 0 --dwa_distance_eps 1e-12
python run_target_proj.py --modes dwa_adaptive_self_projection --rounds 100 --device-id 1 --dwa_distance_eps 1e-12
# 仅展示命令：加 --dry-run
```

launcher 日志位于 `system/target_proj_runs/<timestamp>/<mode>/train.log`；
逐轮文件在其 `checkpoints/Cifar100/FedTargetProj/<run_id>/`，H5 在 `h5_results/`，导出在 `final_models/`。
默认旧四模式和其参数不变。

## 本地验证

本批新增 15 项测试：12 项数学/服务器/COMMANDS、2 项实际低秩 CNN smoke、1 项统一 launcher 配置检查。
TargetProj 相关 168 项通过；全仓库 23 个测试模块独立运行，共 243 项通过。
覆盖均匀距离、C0 更近/更远、首轮非 Avg、helper 相对比例守恒、两版相同权重、原 Projection 核结果，
以及零距离、极端距离、C0 在 float64 中显示 1 而 helper 质量仍非零、两种有效数量和阶段边界/严格阈值计数。
原 guidance 隔离、失败恢复、下一次普通训练逐位不变和 checkpoint/buffers/RNG 测试继续通过。
20 个合成客户端每个模式实际运行三次普通 5-epoch 本地训练与 C0 1-epoch guidance，
验证正常共享模型下载、主指标评价在 guidance 之前、来源轮次、独立日志和导出。
另以真实异构低秩 CNN 的 rank .9/.15/.5 各运行两次普通 5-epoch 训练，
检查 checkpoint SHA-256、RNG、完整参数距离和 CSV/JSON/H5/final-model 导出。

旧 DWA 模块/客户端文件与 `6867b4a` 未改动，服务器 guidance 和旧 DWA 日志方法通过 AST 对照。
两种旧 DWA 与 `6867b4a` 做 24 组随机 20-client float32/64 比较，参数和全部诊断逐位一致。
语法、CLI、diff、100 参数轮 dry-run 和 COMMANDS 内容检查通过。
旧绘图测试使用本机 Conda 环境，临时设置 `MKL_THREADING_LAYER=SEQUENTIAL`，不改正式训练配置。
本地没有 `Cifar100/pat_20` 切分，只有合成 smoke；未推送、未自动启动正式长实验，
不将合成准确率视为真实 final/last10 性能或声称已超过旧控制。
