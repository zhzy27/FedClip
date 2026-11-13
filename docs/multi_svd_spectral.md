# D01 全量 Avg：下发奇异值尾部修正

本模块是用户指定的 CNN 下发机制实验，不改变服务器的样本量完整矩阵 Avg，不引入服务器谱截断。实现基于 `444ad38`；D01 配置以本次需求为准：K1、原五档容量、**low_rank_recovered 初始化 .90**、F=.001、clip10、无原型。仓库本地没有历史 D01 的完整运行日志，因此没有宣称已核验历史结果或准确率。

## 参数与限制

| 参数 | 缺省 | 含义 |
| --- | --- | --- |
| `--multi_svd_spectral_mode` | `none` | `none / single_point / next_tier` |
| `--multi_svd_spectral_layers` | `conv2,fc1,fc2` | 非空、无重复的层名子集 |
| `--multi_svd_spectral_max_gain` | `1.5` | 有限且 ≥1，原奇异值的最大放大倍数 |

开启时限于显式 `--multi_avg_proto_ablation --multi_svd`、K1、全参与、`group_avg_only + full_avg`、共享头、joint、无原型。与旧 `multi_fc1_tail_scale_strength` 非零值及 KD、Prox、Z2、头 L2、replay、课程等组合明确报错。容量必须有至少两个不同的实际整数秩。默认 none 不创建变换回调；其他路径保持原行为。

## 同一次 SVD 与公共谱

`clientMultiGroupSVD.set_parameters` → `Hyper_CNN_512.decom_larger_model` → 原 `Decom_COV / Decom_LINEAR`。每层仍只有原来的一个 `torch.linalg.svd`，只在构造 `sqrt(S[:rank])` 前应用可选变换。左右奇异向量保持原值，两个因子使用平衡平方根。原 `fc1_svd_observer` 仍收到未修改的 P/S/Qh，偏置、conv1、fc3 不变。

每层从**实际分配**容量及原整数秩函数得到最小两个不同秩 k、b。卷积保持原恢复矩阵布局 `[out_channels*kernel, in_channels*kernel]`。原宽度和五档容量的运行时核验为：

| 层 | k | b | 拟合窗口长度 |
| --- | --- | --- | --- |
| conv2 | 12 | 20 | 8 |
| fc1 | 120 | 200 | 20 |
| fc2 | 77 | 128 | 20 |

拟合窗口为最低档最后 min(window,k) 个位置，在有限正值上使用 float64 OLS 拟合 log(sigma) 对**从1开始的秩号**，至少需要两个有效点，斜率限制非正，以实测 sigma_k 为锚点。两种曲线均在完整谱上计算，之后才按客户端秩截断；完整原谱和修正谱有独立 SHA256，用于核对跨容量共同前缀。

- single_point：用 k+1 的预测／实测比，限制在 [1,max_gain]，统一乘到全部尾部。该位置为0则跳过并记录原因。
- next_tier：k+1..b 各自取 max(original,min(prediction,max_gain*original))，b 之后沿用 b 的实际倍率。sigma_b=0 时，该位置和后续零尾部保持0。
- 前 k 个奇异值不变，最低秩客户端直接走原分解。没有抬升0、重新排序、累计最小值修复或额外截断。
- 检查非递增、有限、前缀、零值及增益。统一倍率的除乘可能产生 float64 单 ulp 边界越界时，只将该**统一倍率**向下移动最多4个 ulp；不改单独秩。原 dtype 的实际增益检查容差为 8×机器 epsilon，记录舍入调整次数。

## 生效时点和日志编号

首次下发（loop=0，即将进行 R1 本地训练）不修正。首次生效为 loop=1：已经聚合1次，即将进行 R2 本地训练。每次后续下发重新计算，服务器参数本身不被修改。

`svd_spectral_downloads.csv/json/jsonl` 使用两套明确字段：

- `completed_aggregations_before_download`：本次下发前已完成的聚合次数。
- `upcoming_completed_train_count`：接下来第几次本地训练。

采样按后者记录 R1..R10、R20、R50 和末轮；gr100 的末轮为 R101，即聚合100次后的下发。正式命令同步指定相同的 `multi_rank_spectrum_rounds`。

下载表保留所有客户端×三层的收据，包括跳过原因；有效层记录拟合斜率、k/b、边界逐秩原值／修正值、k+1／中点／b 实际倍率、区间能量、保留／完整尾部触顶数、校正时间、原始与修正完整谱指纹。未选层只在采样轮用原 S 作只读观察，返回原 S，不更改因子。

同一母模型上同时复算两条标量曲线，不做第二次 SVD。`counterfactual_both_entire_interval_at_cap` 与 `counterfactual_same_full_corrected_spectrum` 同时为真时，明确记为“两规则实际修正相同”。这不代表两次完整训练在轨迹分开后的不同母模型上仍相同。

## 实际矩阵谱与准确率

扩展原 `rank_spectrum.csv/json/jsonl`：复用它已经计算的 `svdvals`，增加各层最低档末窗口到下一档之后4个秩的逐秩谱、k/b、实际秩档、整档谱能量。三个观察点仍是 C0 下载后、C0 普通本地训练后、服务器聚合后；没有为扩展字段额外计算 SVD。

下载修正谱与恢复矩阵的实际谱分开命名。原 SVD 距离诊断中的下载模型相对服务器距离，在本实验中同时包含截断和谱修正，不能解释为纯截断误差。谱诊断成本和下载校正成本分别记录。

主指标仍是全20客户端普通 post-local correct 总和／test_samples 总和，保留微调前、逐客户端和五档容量结果。Final、同轮总体曲线最早 Best、Last10、Last20 保持原定义。CSV/JSON/H5、最终 manifest 和模型 checkpoint 记录实际谱策略；预测不依赖谱修正模块。

## 命令与目录

从服务器 `system/` 目录使用 [svd_spectral_commands.sh](../system/svd_spectral_commands.sh)。文件仅定义数组，不执行训练：

- `COMMANDS`：A/B × all/conv2/fc1/fc2，共8条，全部 gain1.5、gr100（101轮）、初始化 .90、原容量、无原型、旧 fc1 尺度0。
- `OPTIONAL_CONTROL_COMMANDS`：1条 D01 none 诊断对照，补齐历史日志缺少的逐秩谱。不要求重跑已匹配的 D01 准确率对照。
- 独立根目录：`system/svd_spectral_runs/pat20_seed0_init090_gr100_20261113_v1/`，各组下分 checkpoints、logs、h5、final_models。GPU ID 在每条命令前部直接写入，可自行修改。

实现来源是本次用户给出的拟合／校正规则，SVD、恢复、训练与 Avg 复用仓库现有实现。没有据此宣称复现其他算法或改善真实准确率。

## 本地验收

`tests/test_multi_svd_spectral.py` 覆盖手算、两规则相同／不同、零尾部、拟合退化、精度／单调性／上限、实际整数秩、一次 SVD、原 observer、首轮／最低档／未选层精确一致及8条命令。

真实入口合成验收工具：

```powershell
D:/miniconda/envs/htfllib/python.exe system/verify_multi_svd_spectral.py --output E:/svd_spectral_acceptance_new
```

它仅在指定的新目录生成合成小样本；使用原 CNN 宽度、100类、20客户端、五档容量、每模型5 epoch、两轮。串行运行 single_point、next_tier、显式 none、冻结 `444ad38` 客户端／服务器参照及关闭谱观察的 single_point。独立 Python 闭式 OLS 复算因子，核对批次、RNG、首次下发／训练、普通 Avg、加权主指标、各格式和21个模型重载。默认模型函数另由冻结源码回归检查。

本机 NumPy 的 `polyfit/lstsq` 参照曾触发底层 DLL 异常；没有修改环境，而将**独立参照工具**改为 Python 标量闭式 OLS。生产实现原本使用 Torch float64 OLS，不依赖该调用；容差未放宽。最终数值与回归结果见 [验收报告](multi_svd_spectral_acceptance.json)。验收为 CPU 合成结果，不代表真实数据准确率已验证。
