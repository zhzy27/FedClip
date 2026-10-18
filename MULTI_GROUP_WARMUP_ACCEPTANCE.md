# FedMultiGroupProj 预热与分组依据验收

基于`2fc0e09`。新增可配置W/basis/H及分组文件原子发布/有限等待。
K模型、本地SGD、固定归属、原群组ProjectionSoftmax/Avg-only聚合核均保留。
未修改环境，未生成/改写真实数据，未推送或启动正式长实验，绘图文件未改动。

## 阶段和预算

- 缺省仍为W0+single_delta，首轮训练后按原更新余弦分组，当轮群组聚合。
- A：W0+full_weights，首轮完整普通post-local参数余弦分组，当轮群组聚合。
- B：W10+full_weights，第10次本地训练后采集完整权重；该轮仍完成全员Avg，随后形成分组，第11次才群组聚合。
- C：W10+delta_history/H10，逐轮同列更新余弦→K列平均→10轮平均；只保留矩阵和指纹，不存历史模型。
- 预热包含在总预算。gr100=101次，A为101次群组，B/C为10次全员Avg+91次群组。
- 所有阶段每客户端每轮都训练K个不同模型各local_epochs；缺省3×5=15次数据遍历，无额外预热/补训/重置。

## 独立短程与旧默认严格回归

已有htfllib：Python3.11.14、Torch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3；CPU2线程。
合成分片仅在忽略目录`fedfews_native_runs/`内，不触碰正式dataset。

`multi_warmup_default0`：Git archive固定导出2fc0e09作为参照，独立进程运行新旧真实main入口。
20客户端/K3/seed0/cluster_seed0/restarts8/max_iterations50、lr.005/batch16/local5/clip10/test batch16、两轮。
初始化、本地/服务器参数SHA、实际DataLoader批次、每个事件Python/NumPy/Torch CPU/CUDA RNG、最终RNG、主指标完全相同。
分组指纹也相同，没有使用浮点容差掩盖默认路径变化。
另有cluster_seed9短程逐位回归通过。

## A/B/C真实main入口

短程采用4客户端/K2/seed7/local1、不等训练/测试样本量；只缩短正式训练预算与数据规模。

| 配置 | 真实完成次数 | 分组形成 | grouped评价loop | 每模式实际训练任务 | 结果 |
|---|---|---|---|---|---|
| A两模式 | 3 | 完成第1次 | 0、2（eg2，末轮强制） | 24 | PASS |
| B两模式 | 12 | 完成第10次Avg后 | 10、11 | 96 | PASS |
| C两模式 | 12 | 完成第10次Avg后 | 10、11 | 96 | PASS |

验证内容：

1. 所有本地下载基座与对应服务器列的参数SHA相同，无模型串用。
2. 预热每列与独立按原接收顺序累积`sum(sample_weight*W_post)`逐元素一致，最大参数差0。
3. 聚合记录loop9（第10次）明确为`group_avg_all`；loop10（第11次）才是指定群组模式。
4. 群组阶段逐列与原群组核一致，最大参数差0；分组及采样没有改变预算。
5. 完整权重余弦独立按所有参数点积/范数归约，包含分类器、不减服务器；与分块GEMM计算一致。
6. C的每轮每列更新独立相对当轮下载参数计算，平均矩阵一致。B/C预热上传、服务器和RNG事件完全相同，证明分组证据观察未改变训练。
7. 三个配置的Avg-only均直接读取各自Projection原子发布的同一文件，完整执行同样预热；当前/来源证据与矩阵逐位相同。
8. B_projection实际导出并strict重新加载全部10个服务器/客户端模型文件，固定输入预测逐元素一致；客户端优化器状态可加载。
9. 主指标仅包含grouped；warmup按列单独统计、成员归属为None，原始N×K pre/post计数保留。
10. 所有主指标、预算与CSV/JSON/H5/manifest一致；独立复算不等测试样本量加权、最早Best、评价点last10/20。

独立float64逐参数归约与分块GEMM累加顺序不同；完整CNN相似度最大实测误差为
`1.9206858326015208e-14`。初次采用过严4e-15诊断阈值未通过，明确记录后采用1e-12独立公式阈值，
仍远小于float32精度；未改算法、分组矩阵或训练设置来迎合验收。
聚合参数及冻结回归始终要求零误差/SHA与批次/RNG完全相同。

## 历史窗口、评价隔离与同步

- tail：N5/K3/seed11、W4/H2、六轮、eg3；只采集loop2/3（完成第3/4次）的历史，grouped只在末轮loop5评价。
- tail_no_pre：相同文件归属，关闭pre观察并改eg1。全部轮次上传、服务器、实际训练批次及最终RNG完全相同，公共评价点post准确率一致。
- 真实IPC：先启动Avg-only消费者。它完成两轮普通Avg后进入等待，再启动Projection生产者。
  等待约15.89秒，读取完整原子文件后继续；两边仍各24个训练任务，无重训；等待前后参数/批次/RNG与生产者一致。
- 单元测试覆盖无等待缺失报错、明确超时、坏JSON、已有文件禁止覆盖、W/basis/H/形成时点/矩阵指纹/N/K错配失败，独立RNG保护。
- 最终元数据复验：另一次W2/H2真实入口检查H5与manifest的协议、阶段计数、当前/来源证据回执一致。
- C预览：计划101=10+91，每客户端15次数据遍历；即使目标分组文件尚未生成，预览不等待、不构造模型、不训练。

## 测试与产物

共71项相关单元/回归通过：

- 新warmup8项，旧multi-group9项。
- CNN/可配置控制15项、旧群组核12项。
- 原生Few7项、population6项、公共reader/model/loader/metrics8项、日志6项。

机器报告（忽略目录）：

- `fedfews_native_runs/multi_warmup_acceptance_v2/acceptance_report.json`
- 每case的`verification.json`（事件、批次、RNG、矩阵独立公式误差、指标、日志位置）
- `fedfews_native_runs/multi_warmup_acceptance_v2/ipc_acceptance.json`
- `fedfews_native_runs/multi_warmup_acceptance_v2/final_metadata/verification.json`
- `fedfews_native_runs/multi_warmup_default0/acceptance_report.json`
- `fedfews_native_runs/multi_warmup_acceptance_v2/preview_C.json`

为控制磁盘占用，大多数验收用例只在工具内部跳过原始模型文件落盘，保留全部计算/指纹/报告；
仅B_projection落盘小规模全模型并测试重载，导出副本同卷硬链接。正式入口始终保存全部K模型/客户端模型，未增加关闭保存的训练参数。
不清理或覆盖既有模型、真实数据及绘图文件。

六条完整COMMANDS：`system/multi_group_warmup_commands.sh`；协议与日志说明：`system/MULTI_GROUP_WARMUP.md`。
从system目录运行复验工具：

```bash
python verify_multi_group_warmup.py --output ../fedfews_native_runs/multi_warmup_recheck
python verify_multi_group_warmup.py --output ../fedfews_native_runs/multi_default_recheck --compat-only
```

本地没有真实pat20分片，本次仅CPU合成验收，不声称真实准确率改善或CUDA逐位一致性已验收。
