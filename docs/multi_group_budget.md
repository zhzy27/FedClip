# K3客户端总参数预算对齐实验

本批只更换容量配置，使用已有 `FedMultiGroupProj --multi_svd`。不增加聚合算法，不改训练、恢复、评价或通信路径，也不替换默认 `.9,.37,.35,.25,.15`。旧同构、旧高容量和K1命令文件保持不变。

## 配置与预算

[独立配置](../system/experiments/multi_group_budget_v1.json) 保存五档新秩、原 `CNN_1_512` 至 `CNN_5_512` 的预算及预期计数；[独立参数报告](multi_group_parameter_budget.md) 给出实际实例化计数、层秩和20人逐一核验。基线是旧五档普通CNN，包含分类头和bias，不是K1低秩模型或统一CNN-512。

新秩为 `.299,.1615,.114,.0891,.0518`，客户端按id%5，每档4人，每人三模型同档且跨轮固定。审计逐个读取实际Parameter对象并去重，包含U/V、未分解层、bias、fc3，不重复计入base/head别名。改变K、宽度、类别数或输入大小后须重新核算；没有自动求秩功能。

本批只提供两个正式配置：

- `group_avg_only`：全部20人每轮各训练三个模型，每列只用本组，实际组内/外质量1/0。
- `group_projection_softmax`：相同任务量，原群组参考、逐个helper投影、投影前余弦softmax，质量.50/.50，tau=.2。

均为K3、无预热、首轮20×3次正常训练后恢复完整权重，三列完整权重余弦矩阵取均值，再自动形成7/7/6。分组只做一次，立即用于首轮聚合。没有预设成员，没有使用测试成绩或标签名单。每模型5 epoch，因此每人每轮15次数据遍历；`gr100`仍为101轮。

原CE＋F(.001)、SGD(.005，无动量和weight decay)、batch16、clip10、共享头joint、Prox0和头L2=0保持不变。保留全部SVD截断距离、本地训练恢复距离和global-delta距离。global-delta仍包含截断变化。

## 启动数组与独立目录

从服务器 `system/` 目录使用 [multi_group_budget_commands.sh](../system/multi_group_budget_commands.sh)。文件有完整变量定义和同一个 `declare -a COMMANDS`，顺序为Avg-only/GPU0、Projection/GPU1，没有K1新长实验。

```bash
# 每批使用一个新tag；以下仅定义和预览，不启动训练。
export BUDGET_RUN_TAG="budget3_$(date +%Y%m%d_%H%M%S)"
export BUDGET_DATA_ROOT="../dataset"
export BUDGET_RUN_ROOT="multi_svd_budget_runs/$BUDGET_RUN_TAG"
export BUDGET_GROUP_FILE="$BUDGET_RUN_ROOT/shared/budget_K3_first_full_weights.json"
export BUDGET_GROUP_WAIT=7200
source multi_group_budget_commands.sh
printf '%s\n' "${COMMANDS[@]}"
eval "${COMMANDS[1]} --multi_preview"  # 只核验已有pat_20数据及配置，不训练或发布分组
```

在原 `run_now.sh` 中source该文件替换本批数组定义，保留调度循环。两任务可同时交给调度器；Avg-only完成自己的首轮训练后等待Projection原子发布，最多7200秒，不重复训练、不消耗训练随机流。超时或不兼容文件明确失败。第一次务必使用新路径，不覆盖旧分组。现有context会核验新容量列表、映射、数据、初始化、F及训练/分组协议，旧高容量或同构分组不能混用。

默认目录为 `system/multi_svd_budget_runs/<tag>/`：

| 路径 | 内容 |
| --- | --- |
| `shared/budget_K3_first_full_weights.json` | 本批Projection首轮产生、两个模式共用的固定分组、矩阵及证据 |
| `budget3_group_avg_only/` | 仅组内Avg实验 |
| `budget3_group_projection_softmax/` | 群组Projection实验 |
| 各实验的 `checkpoints/` | 普通post-local低秩checkpoint |
| 各实验的 `logs/run_*/` | CSV/JSON、capacity_manifest、指标、距离/权重诊断；日志ZIP不包含模型 |
| 各实验的 `h5/` | 同一主指标及表的H5 |
| 各实验的 `final_models/` | 三个完整服务器模型、客户端各列低秩模型、manifest和分组映射 |

具体run子目录由现有日志设施生成，不改旧日志协议。主预测仍由固定归属g(i)选择普通低秩模型；服务器模型和恢复上传不冒充主指标checkpoint。重载沿用 `utils.multi_group_svd.load_checkpoint`。

## 独立只读审计

训练无需新增开关。审计不会执行客户端训练、SVD或模型forward，也不读取真实数据；构造模型时保存并恢复Python/NumPy/Torch CPU/CUDA随机流。真实运行后可交叉核验其 `capacity_manifest.json` 中的实际下载层秩/参数量。

```bash
# 在system/，不训练即可核算这批模型预算。
python audit_multi_group_budget.py \
  --config experiments/multi_group_budget_v1.json \
  --output "$BUDGET_RUN_ROOT/parameter_budget"

# 运行产生capacity_manifest后，核验真实下载模型与独立计数一致。
# 自动寻找该实验当前唯一的run目录；多run时明确失败，避免选错。
mapfile -t BUDGET_CAPACITY_FILES < <(find "$BUDGET_RUN_ROOT/budget3_group_projection_softmax/logs" -type f -name capacity_manifest.json)
if [ "${#BUDGET_CAPACITY_FILES[@]}" -ne 1 ]; then
  echo "需要且只能有一个capacity_manifest，请核对本次独立目录" >&2
  return 1 2>/dev/null || exit 1
fi
python audit_multi_group_budget.py \
  --config experiments/multi_group_budget_v1.json \
  --capacity-manifest "${BUDGET_CAPACITY_FILES[0]}" \
  --group-file "$BUDGET_GROUP_FILE" \
  --output "$BUDGET_RUN_ROOT/parameter_budget_actual"
```

可额外传 `--previous-group-file /实际旧批次/fixed_groups.json`，只在审计中保存新旧成员、各自容量构成和文件指纹，不把旧组注入训练。新旧容量会重新分组，比较准确率时须保留两批实际名单，不能把所有差异都归因于参数量。

报告输出 `parameter_budget.json/md`。每客户端列出对应基线、各个实际模型计数、三模型总数/比例、单次所属模型推理参数数及PASS；与真实manifest不符或超预算时失败退出。原K1五档实际单模型计数也列出，不能称其与新K3严格等参数。

## 参数预算、通信与内存分别记录

完整母模型每个仍为2,691,860参数，三个服务器共8,075,580。当前每客户端每轮逻辑下载及恢复上传也各对应三个完整模型；恢复上传缓存保持完整参数范围。这些是去重参数元素数，不是实测网络字节，也没有包含buffers/序列化开销。

客户端三个低秩模型总参数、预测时一个模型参数、服务器/恢复缓存参数和15次数据遍历量是不同指标。没有改变上下行传输方式，**本批仅对齐客户端模型总参数预算，不宣称通信预算对齐，也不把参数量当成峰值内存**。

## 评价及复验

主指标仍为固定归属模型的全员 `sum(correct)/sum(test_samples)`。保留pre/post、三个组、五档容量、逐客户端计数，best为同轮全员平均曲线的最大值与最早轮次；final/last10/last20与CSV/JSON/H5/manifest一致。没有测试选模型或各人历史best平均。

```bash
# 从仓库根目录；只在独立目录创建合成分片。
python -m unittest discover -s tests -p 'test_multi_group*.py' -v
python -m unittest discover -s tests -p test_target_projection.py -v
python system/verify_multi_group_svd.py \
  --budget-config system/experiments/multi_group_budget_v1.json \
  --output system/temp/svd_budget_acceptance_new
```

最后一条包含原宽度20人/K3/两轮两个模式，以及只关预算审计的精确轨迹对照；逐列复算原聚合核、验证首轮相同上传/分组、实际层秩、非有限值、指标导出和全部模型重载。验证后默认清理自己生成的模型张量，保留轻量报告；`--keep-models`可保留。合成验收不代表压缩后真实准确率已经达标；未自动启动正式训练。
