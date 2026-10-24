# SVD母模型初始化方式验证

本批仅增加初始化选项，基于 `target_proj / a8ceabf`。训练随机数协议、五档容量、下载/恢复、CE＋原F、梯度裁剪、训练预算、原群组聚合核及post-local评价保持不变。不增加预算压缩或K3正式配置。

本地 `simple_v`（`a46b5e4d6a12b1430ac2f9703b9da25d899cdc37`）的 `system/main.py` 在 `Decom_CNN-5-512` 分支将 `args.global_model` 设为 `Hyper_CNN_512(... ratio_LR=0.15 ...)`，客户端五档为 `.9/.37/.35/.25/.15`。这里只对齐低秩母模型初始化方式，不宣称完整复现旧版随机轨迹。

## 参数和实际操作

| 参数 | 缺省/合法范围 | 作用 |
| --- | --- | --- |
| `--multi_svd_init` | `dense` / `low_rank_recovered`，默认dense | 只影响初始服务器母模型 |
| `--multi_svd_init_rank_rate` | 默认 `.15`，有限且在 `(0,1]` | 新模式随机低秩初始化的秩比例，不从客户端最小容量推断 |

仅在 `FedMultiGroupProj --multi_svd` 开放新模式。dense的初始秩参数不生效；为避免静默忽略，dense下显式使用非默认秩值报错。未启用multi_svd时拒绝非默认初始化选项。`rank_rate=1`是允许的完整秩边界，协议仍与dense区分。

`dense`继续执行原始完整模型构造。新模式在**原有CPU seed-fork作用域**中：

1. 使用当前训练seed创建指定初始秩的原 `Hyper_CNN_512`。
2. 直接调用其原 `recover_larger_model()`，不另写恢复算法。
3. 将母模型的 `ratio_LR` 设为 `1.0`，然后按原流程复制为K份。

恢复函数内部创建完整层并用恢复权重覆盖；恢复之后没有再次随机初始化任何完整层或分类头。客户端每次下载仍按自己的五档容量执行原SVD，并设置本地实际 `ratio_LR`。服务器状态为完整模型、母模型的初始化来源和客户端实际容量是三个不同概念。

Python/NumPy/Torch CPU/CUDA种子规则未改；客户端构造后的原 `seed_control`、真实下载消耗的随机流和只读恢复/评价保护机制均未改。新路径额外构造/恢复发生在已有初始化作用域内。

## 唯一正式 COMMANDS

[multi_group_svd_init_commands.sh](../system/multi_group_svd_init_commands.sh) 提供同一个 `declare -a COMMANDS=(...)` 中的**一条**完整命令：GPU0示例，K1、全20人样本量Avg、原五档、新低秩恢复初始化 `.15`。从服务器 `system/` 目录供原 `run_now.sh` 使用：

```bash
source multi_group_svd_init_commands.sh
printf '%s\n' "${COMMANDS[@]}"            # 只打印，不启动
eval "${COMMANDS[0]} --multi_preview"    # 只核验现有数据与配置
```

将该source放到原调度器的COMMANDS定义处即可。本文件自己不启动任务。变量都有完整缺省定义：

```bash
SVD_INIT_RUN_TAG="${SVD_INIT_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SVD_INIT_DATA_ROOT="${SVD_INIT_DATA_ROOT:-../dataset}"
SVD_INIT_RUN_ROOT="${SVD_INIT_RUN_ROOT:-multi_svd_init_runs/$SVD_INIT_RUN_TAG}"
```

公共配置为已有 Cifar100/pat_20、20客户端/100类、seed0、全参与、每人每轮一个模型5epoch、SGD .005、batch16、无momentum/weight decay、clip10、F .001、共享头joint、Prox/头L2均0，`gr100 / eg1`实际101轮。K1唯一群组为全20人，group_avg_only实际质量1/0，不启用Projection、softmax评分或额外微调，不复用任何旧分组文件。

输出根目录为 `system/multi_svd_init_runs/<tag>/K1_avg_low_rank_recovered_r0p15/`，其下 `checkpoints/`、`logs/run_*/`及logs.zip、`h5/`、`final_models/`互相隔离，不覆盖旧K1或其他实验。原高容量/预算容量/K3命令文件未修改。

## 元数据及兼容性

- 配置与requested/effective manifest保存两个参数；启动日志打印初始化协议及实际初始参数SHA256。
- `manifest.json`、`capacity_manifest.json`、H5的 `svd_initialization_json`、服务器及普通客户端checkpoint保存实际初始化来源。
- 服务器 `model_kwargs.ratio_LR=1.0`；`svd_initialization.effective_low_rank_rate=.15`表示母模型来源；客户端capacity中的rank_rate是其真实训练档位，不混为一个字段。
- 新初始化分组context加入完整 `svd_initialization`，包括方式与初始秩。改变方式或初始秩后，旧文件拒绝；即使rank=1导致参数哈希与dense相同，协议仍拒绝混用。
- 既有dense分组context继续原样；缺少新初始化字段的旧文件明确按历史dense解释，避免使已有文件/分组指纹失效。dense配置和manifest仍显式记录方式，配置秩为非生效的 `.15`。

模型加载继续使用 `utils.multi_group_svd.load_checkpoint`。服务器文件是完整聚合模型；主指标对应所属模型的普通post-local低秩checkpoint。没有更改评价时点或模型选择规则。

## 结果比较与解释

主指标仍是同一轮全20人所属模型的 `sum(correct)/sum(test_samples)`；K1时每人只用自己的第0列低秩模型。沿用final、同轮平均曲线的best及最早轮次、last10/last20。正式比较已有dense K1与本次新K1时，从各自 `logs/run_*/primary_metrics.json` 查看R1、R1–R10曲线及其均值，再读取 `primary_summary.json` 的final/best/last10/last20。完成轮次为 `loop_round+1`。

可在两组运行完成后只读提取（将路径替换为实际run目录；此命令不训练）：

```bash
python - /path/to/old_K1/logs/run_x /path/to/new_K1/logs/run_y <<'PY'
import json, pathlib, sys
for folder in map(pathlib.Path, sys.argv[1:]):
    rows = json.loads((folder/'primary_metrics.json').read_text())
    summary = json.loads((folder/'primary_summary.json').read_text())
    early = [r for r in rows if 1 <= r['completed_train_count'] <= 10]
    print(folder)
    print('R1-R10 (percent):', [(r['completed_train_count'],100*r['post_local_accuracy']) for r in early])
    print('early mean / actual count:', 100*sum(r['post_local_accuracy'] for r in early)/len(early), len(early))
    print({k:100*summary[k] for k in ('final','best','last10','last20')}, 'earliest best:',summary['best_completed_train_count'])
PY
```

先核对数据指纹、训练预算和主指标人群一致；历史约48.7%只作为研究参考，不是合成验收标准。尚未完成正式运行，不能认定旧Base已恢复。若回升，先表述为“初始化差异是重要原因”，不宣称旧版完整轨迹已复现；若没有回升，下一步再核对批次顺序等流程，本批不同时修改这些因素。

## 短程复验

从仓库根目录：

```bash
python -m unittest discover -s tests -p 'test_multi_group*.py' -v
python -m unittest discover -s tests -p test_target_projection.py -v
python system/verify_multi_group_svd.py --initialization-only \
  --output system/temp/svd_init_acceptance_new
```

仅运行K1新初始化两轮，以及原dense三轮/稀疏评价回归，不运行K3或预算新实验。原宽度模型、20客户端、每模型5epoch，合成数据仅在独立目录创建。若有此前原dense K1的 `avg_all/verification.json`，可加 `--dense-reference <该文件>`，严格比较参数/上传/批次/分组/最终随机状态及主summary。工具保存重载所有模型后自动清理自己的合成模型，保留轻量报告；可用 `--keep-models` 保留。

新初始化与同seed的直接低秩构造＋原恢复参照要求逐元素一致；首次再分解恢复只允许float32 SVD归约误差，检查每参数相对Frobenius误差≤3e-6且最大绝对误差≤1e-5。此界限只用于该恢复检查，没有放宽旧聚合/随机轨迹一致性标准。
