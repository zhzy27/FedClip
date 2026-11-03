# 单目标当前硬保护＋有限历史软约束

基于 `0f8c844`，新增 `projection_history_softmax`。普通客户端训练不变，仍为所有客户端各训练一个低秩模型、每轮各5 epoch。只改变 helper 的服务器投影；不学习权重，不对 SGD/SVD 求梯度，无额外客户端训练或通信。历史是含原 F 正则和 SVD 截断影响的 **global-delta 更新参考**，不称为纯目标知识。

## 入口与参数

`main.py` 和 `run_target_proj.py --modes projection_history_softmax` 均支持：

| 参数 | 默认 | 含义 |
|---|---:|---|
| `--projection_history_window` | 10 | 正整数；只保留最近 L 个已成功聚合轮次 |
| `--projection_history_decay` | .8 | 有限 `[0,1]`；1为窗口内等权，0只给上一轮非零权重 |
| `--projection_history_strength` | 1 | 有限非负；0直接调用旧核，不创建、读取或保存历史缓存 |
| `--projection_history_diagnostics` | 1 | 0关闭额外逐helper双参考诊断；求解、权重、约束检查不变 |
| `--projection_self_weight` | 新模式省略时 .125 | 旧模式省略时仍使用原样本量份额 |
| `--projection_softmax_tau` | .20 | 原始当前更新余弦的稳定Softmax；不使用历史评分 |

首版只开放单目标 `FedTargetProj + Decom_CNN-5-512 + shared + joint`、完整数据/epoch及全员参与。目标ID读取实际配置。拒绝非零Prox/Z2/head-L2、replay、meta、原型、在线KD、私有头、FedRep、同构control、预训练恢复或断点续训组合。其他模式不能传入非默认历史控制参数。既有模式没有改为新核。

## 求解及兼容边界

对原始helper全模型更新 d，用单位当前方向 u、单位过去参考 q，最小化

`||v-d||²/2 + strength * min(0,q·v)²/2`，满足 `u·v >= 0`。

记 `a=u·d, b=q·d, c=u·q`。先计算不受硬约束的软惩罚解
`v=d-strength/(1+strength)*min(0,b)*q`。若其当前内积非负，即为最优解；否则当前约束取等号，在正交分量 `t=q-c*u` 上求

`v=d-a*u - min(0,t·d)/(1/strength+||t||²)*t`。

这是强凸目标的活动集解析解，非两个方向的连续投影。普通情况仅归约两个方向的内积。接近同/反向时显式用float64正交分量，避免 `1-c*c` 或 `b-c*a` 的消减；不构造参数维度平方矩阵，不添加QP依赖。所有服务器计算在no_grad下，逐参数处理，不拼接全模型。

当前/历史参考范数 `<=sqrt(旧EPS)=1e-6` 时明确退回旧核。首轮没有过去参考，直接使用旧核；严格同向时历史项冗余，也保留旧EPS算术。新分支对实际模型dtype的helper结果检查当前约束，允许的单位方向负残差为 `64*finfo(dtype).eps*max(raw_helper_norm,1e-12)`，超出即失败。近零退化时保留旧EPS约定，不宣称旧核的残差是精确零。

旧核本身没有修改。新模式先复用旧核得到原始评分、**完全相同的权重**及旧投影对照，再在受RNG保护的额外上传读取中对原始helper更新求解。没有把旧投影结果当作新问题的输入。额外读取是服务器侧恢复，不要求客户端再次上传。目标本轮更新从不投影。

## 历史生命周期

缓存是最近L个过去目标global delta的独立、detach CPU副本。按最新至最远，以 `decay**j` 归一化平均；窗口1就是上一轮参考。当前轮读取缓存时不包含本轮。服务器参数保存和受保护评价成功后才提交本轮更新；失败不提前提交。缓存文件原子替换失败时恢复内存中的旧缓存。

`projection_history_cache.pt` 单独保存轮次、参数名/shape/dtype schema和更新张量。预测不依赖缓存，当前没有恢复训练入口。缓存字节数按实际tensor计数，最多约为L份full-W更新；不是额外通信量，也不是峰值内存估计。

## 三条正式命令

从服务器 `system/` 使用 [projection_history_commands.sh](../system/projection_history_commands.sh)，提供一个完整 `declare -a COMMANDS` 数组：旧Project、强度1、强度10，GPU示例0/1/2。可以复制数组交给原调度器，文件本身只定义命令，不启动。

公共设置为 CIFAR100/pat20、20人、seed0、原五档低秩、原单目标初始化，local5/batch16/lr.005、F.001、**clip0**、shared/joint、self.125/tau.20、gr100实际101轮。不是K3恢复初始化，也未对齐历史约55%实验的完整随机轨迹。强度1/10仅为预先设定的探索值。

输出根为 `system/target_history_runs/pat20_seed0_gr100_20261103_v1/`，01/02/03各自独立。模型、H5和日志使用各自子目录，内部还追加初始化不变的模式/历史参数标签与运行ID，避免覆盖。

## 读取结果与字段

主指标统一读 `target_proj_metrics.json` 的 `summary`、CSV的 `target_post_local_acc`，或H5的 `target_projection/target_post_local_acc`。Final、同轮曲线最早Best、Last10/Last20单位均为fraction。`target_client_test_acc` 是聚合后正常下载且尚未本地训练的附加评价。

新模式另有：

- `projection_history_metrics.csv`：参考原始范数、实际历史长度、两套来源轮次编号、参考余弦、退化原因、有限性、缓存前后字节数及成本。
- `projection_history_clients.csv`：原样本份额、原始余弦、实际Softmax权重，原始冲突，以及投影前后双参考内积/余弦、硬约束/软惩罚活动标志、对原始更新/旧Project的修正范数、约束残差/容差。
- `projection_history.json`：配置、数据文件SHA256、初始服务器指纹、完整参数schema、逐轮/逐helper记录及主指标汇总。
- H5 `target_projection/projection_history`：同一policy、summary及逐轮JSON数据。新模式root `rs_test_acc` 明确为普通目标post-local；旧root系列单独保存为 `legacy_server_diagnostic_test_acc`。旧模式H5语义不改。
- 最终 `manifest.json`：真实目标ID、配置、主汇总、模型路径与缓存非推理依赖说明。`Client_<target>_model.pt` 为普通post-local低秩模型；`Server_model.pt` 为聚合服务器。

`legacy_projection_coefficient`、`legacy_removed_component_norm` 和原 `removed_update_ratio` 均为旧current-only对照。双参考修正用独立字段描述，不伪装成原单方向删除比例。近共线活动边界额外记录正交历史系数。无过去参考的历史内积/余弦为null；明确退化或关闭诊断的未计算值也为null，不伪造为0。

`legacy_kernel_seconds` 是旧核成本；`history_solve_seconds` 是小系统求解计时；`history_extra_compute_seconds` 包含参考构造、额外恢复读取、应用/检查和诊断；`history_cache_maintenance_seconds` 包含CPU拷贝及缓存写盘。未把这些时间算成额外SGD预算。普通训练、SVD和上传恢复规则未变。

## 验收方式

单元/冻结回归：

```bash
python -m unittest discover -s ../tests -p 'test_projection_history*.py' -v
```

独立真实入口合成验收，从 `system/` 运行（输出必须是新目录；Windows建议短路径）：

```bash
python verify_projection_history.py --output /tmp/fedclip_history_acceptance
```

仅在指定目录生成少量合成分片。串行运行旧核、强度0、1、诊断关闭、10五个case，每个20客户端/两轮/原宽度异构CNN、local5。核对输入上传不变、全参数含head、相同权重、每batch一次前向、40次本地任务/200次SGD、CSV/JSON/H5/manifest、21个预测模型重载和缓存schema。重载核验后默认清理checkpoint及最终模型二进制，保留验收报告/日志；`--keep-models`才保留。

测试还将强度0与冻结 `0f8c844` 对比上传、聚合、实际batch和Python/NumPy/Torch RNG，使用非零目标ID3；独立SciPy约束优化对照220组解析问题，并覆盖近反向极小夹角/极大强度、窗口淘汰、缓存无当前泄漏、失败提交及输入保护。详见 [验收报告](projection_history_acceptance.json)。

2026-11-03 本地共177项检查通过，五个真实入口合成case全部通过。强度0、首轮及诊断开关的参数/batch/RNG严格一致；模型核对最大参数误差为0。强度1和10的第二轮当前方向最大负残差分别约 `1.81e-9` 和 `3.77e-10`，均在所声明的float32容差内。诊断关闭时未保存的残差为null，不解释为0。已重载核验105个预测模型，并清理合成验收模型二进制。

本地没有真实CIFAR100分片，合成验收只说明实现一致性及可运行性，不说明真实准确率收益。未启动正式长实验、未推送、未改变环境或真实数据；绘图文件未修改。
