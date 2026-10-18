# FedMultiGroupProj：预热与分组依据

基于2fc0e09。固定K模型、固定成员归属、每客户端每轮训练全部K模型的机制不变。
旧缺省仍是`multi_warmup_rounds=0, multi_group_basis=single_delta`，首轮更新余弦后直接群组聚合。
群组聚合继续直接调用原`aggregate_group_updates`；没有新增投影/权重公式。

## 参数及阶段

| 参数 | 缺省及约束 |
|---|---|
| `--multi_warmup_rounds W` | 0；非负整数，包含在`gr+1`总预算中 |
| `--multi_group_basis` | `single_delta`、`full_weights`、`delta_history`；缺省single_delta |
| `--multi_history_window H` | 只适用于delta_history；省略时H=W，显式要求1≤H≤W且W>0；其他basis显式传入会报错 |
| `--multi_group_publish PATH` | 分组形成时原子发布；不会覆盖已有文件，与读取参数互斥 |
| `--multi_group_file PATH` | 复用同配置固定分组；仍完整执行本次预热/分组轮训练 |
| `--multi_group_wait_seconds` | 0；显式有限非负超时，正值必须配置读取路径；0时缺失文件立即报错 |
| `--multi_group_poll_seconds` | 1；有限正数，非默认值要求启用等待 |

总预算至少为W+1轮：`global_rounds >= W`。所有轮次均为真实N×K模型训练，
每模型local_epochs，默认每客户端每轮3×5=15次数据遍历。
N/K/seed、容量、聚类参数及训练参数仍可配置，未写死为本批设置。

- W=0：完成第1次普通训练后分组，当轮直接群组聚合。
- W>0：前W轮每列全员样本量Avg。第W轮本地训练后先采集证据，**仍完成本轮普通Avg**，
  随后形成/读取/发布固定分组。从第W+1轮开始群组训练。
- 阶段切换保留预热后的K个服务器模型，不重新初始化，不重设训练随机流。

日志用`loop_round`（0起）和`completed_train_count`（1起）双标记。
分组文件中`formation_completed_train_count=max(1,W)`，`grouped_start_loop_round=W`。

| 本批配置 | 命令参数 | gr100的实际组成 |
|---|---|---|
| A | `--multi_warmup_rounds 0 --multi_group_basis full_weights` | 101轮群组训练，证据取首轮post-local |
| B | `--multi_warmup_rounds 10 --multi_group_basis full_weights` | 10轮全员Avg+91轮群组，证据取第10轮post-local |
| C | `--multi_warmup_rounds 10 --multi_group_basis delta_history --multi_history_window 10` | 10轮全员Avg+91轮群组，证据取前10轮 |

single_delta在W>0时也有明确含义：只取最后一轮预热的普通更新余弦；不隐式改成历史平均。

## 分组证据

- `full_weights`：完整普通post-local named_parameters，含分类器；逐列做客户端两两权重余弦，
  再对K列算术平均。**不减服务器参数**。
- `single_delta`：逐列对`W_post[i,k,t]-S_download[k,t]`做余弦，再对K列平均。
- `delta_history`：先按同一轮/同一列计算上述更新余弦，再对K列及预热末尾H轮平均。
  不先求跨轮更新向量之和，不混用不同轮次的服务器起点。

历史采集发生在聚合前，独立于eval_gap；只保存实际窗口的N×N矩阵、逐列矩阵和向量/矩阵指纹。
不保存历史完整模型。H<W时仅采集预热末尾H轮。最终平均输入原来的确定性平衡聚类。
聚类不读取测试计数、准确率或标签重叠。

分组JSON新增`grouping_protocol`和`evidence`：basis/W/H、真实采用loop/完成训练次数、
分组形成时点、每轮完整权重/更新指纹、逐列/逐轮相似度、最终矩阵及证据SHA256。
新协议使用schema2；旧0预热single_delta的schema1文件保持可读，旧数据/初始化/训练协议指纹保持兼容。
不同basis/W/H或数据/初始化/N/K/容量/训练协议的文件明确拒绝。

## 六条统一调度命令

从`system/`目录source `multi_group_warmup_commands.sh`。
它只定义6项`declare -a COMMANDS`和6项`PREVIEW_COMMANDS`，不自动训练。
顺序为A Projection、A Avg-only、B Projection、B Avg-only、C Projection、C Avg-only。
六条完整字符串保留`-algo/-data/-m/-did`，可交给现有并行调度器。

公共配置：CNN-512/Cifar100/pat_20/20人/K3/7,7,6/seed0/.005/batch16/local5/gr100/eg1/clip10/test batch16，
Projection beta=.50、tau=.2；Avg-only的实际质量为1/0。全部包含分类头；无F/Prox/私有头/反馈。
GPU示例0–5，可分别修改`WARM_GPU_A_PROJ/A_AVG/B_PROJ/B_AVG/C_PROJ/C_AVG`。
`WARM_DATA_ROOT`默认`../dataset`的绝对路径，不下载或生成数据。

同一配置的两条命令共享预先确定的`group_files/A.json`（或B/C）：

1. Projection在形成分组时写完整UTF8 JSON到同目录临时文件，flush/fsync后原子链接到目标路径，
   目标文件在写完前不可见；已有目标拒绝覆盖，失败时不会留下半个正式JSON。
2. Avg-only完成自己的必要本地训练和预热Avg后，才在分组边界读取该文件；
   尚未生成时等待，默认上限7200秒，15秒一次清晰进度；`WARM_WAIT_SECONDS`可配置。
3. 等待只做文件检查/休眠，不重训、不抽样，不推进Python/NumPy/Torch CPU/CUDA流。
4. 超时、坏JSON、协议错配立即明确失败，不永久等待，不静默重新聚类。

请使用新`WARM_RUN_TAG`/输出根目录。生产者拒绝覆盖已有分组文件；不要将同一路径当成可覆盖槽位。
同组对照强制通过文件共享成员。跨GPU证据的逐位差异只记录在
`group_reuse_verification.json`，不因差异改变映射；source/current的证据和最终相似矩阵均可核对。
读取文件不会省略预热。加载现成文件时，实际采用的聚类参数来自文件并写入manifest的effective_clustering。

## 指标与输出

所有评价点都保存N×K模型的微调前/后correct、test_samples，`stage`为pre/post，另用`phase`区分训练阶段。

- warmup：成员归属`None`、membership_status=undetermined；按每一列分别输出全员加权准确率。
  不产生grouped主指标，不填0，不用后来群组映射回填warmup。
- grouped：按固定g(i)提取普通post-local计数，主指标为全员sum(correct)/sum(test_samples)。
  首轮A可以使用已经保存的同轮pre/post计数。逐组和逐成员指标、同轮增益保留。
- 主summary只用grouped评价点计算Final/最早Best/last10/last20，列实际loop和完成训练次数。
  last10/20是评价次数，不是隐含训练轮数；末轮强制评价，至少存在一个grouped评价点。

新增/扩展文件仍位于独立日志目录，与模型分离：

- `warmup_metrics.csv/json`、`warmup_summary.json`：每列独立预热观察，不择优选部署模型。
- `primary_metrics.csv/json`、`primary_summary.json`：仅grouped。
- `client_metrics.csv/json`、`local_training.csv/json`、`round_budget.csv/json`、`rounds.jsonl`：明确phase、实际预算。
- `group_history.json`、`group_similarity.json`、`fixed_groups.json`：矩阵、证据、形成时点、映射。
- 原聚合/权重/冲突/模型间距诊断保留；warmup聚合标记为group_avg_all/all clients。
- CSV/JSON/H5/最终manifest记录相同阶段计数；模型文件带grouping_protocol，加载方式沿用MULTI_GROUP.md。

额外评价仍恢复参数、buffers、模块状态和随机流；没有第三阶段训练或额外C0处理。

## 合成验收

```bash
# 从system目录：仅创建独立合成fixtures，不触碰真实dataset。
python verify_multi_group_warmup.py --output ../fedfews_native_runs/multi_warmup_acceptance
python -m unittest discover -s ../tests -p test_multi_group_warmup.py -v
```

验收包含冻结2fc0e09默认路径的20人/K3两轮参数/批次/RNG逐位对照、A完整权重独立公式、
B/C真正10轮预热、逐轮逐列C平均、小窗口/稀疏评价、生产者/消费者实际等待、
超时/错配、观察开关、真实CNN全模型导出重载。
为避免临时磁盘膨胀，工具的大多数用例只保留指纹/日志，不落盘原始模型；B_projection保留完整模型重载检查。
这只是验收工具内部补丁，正式训练没有关闭保存功能的参数。

独立float64矩阵公式与分块GEMM归约顺序不同，约2e-14舍入误差实测，验收阈值明确为1e-12；
聚合参数、冻结默认路径、批次及RNG对照仍是零容差。合成验收不能证明真实性能提升。
