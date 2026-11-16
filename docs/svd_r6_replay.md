# 从 S5 回放五支 R6，并测 R7 普通下发

独立入口：`system/svd_r6_replay.py`。现有联邦训练、聚合、模型、初始化和默认参数均未修改。只读 H/H+/L 的 R5 **完整聚合服务器**及其同目录 manifest，不读取旧客户端模型，不需要原型库。

| 分支 | S5来源 | R6下载 | 容量 | R6后处理 |
| --- | --- | --- | --- | --- |
| H_raw | H | 普通SVD | 原五档 | 一次原 full Avg，然后普通SVD测R7前 |
| H_repair | H | all_tail，三层 | 原五档 | 同上 |
| Hplus_raw | H+ | 普通SVD | 原五档 | 同上 |
| Hplus_repair | H+ | all_tail，三层 | 原五档 | 同上 |
| L_raw | L | 普通SVD | 全员.90 | 只做R6，不再聚合或R7 |

每支每客户端仅训练一个模型，固定5 epoch；本次合计100次本地训练任务。H与H+各自同一客户端的raw/repair共享**同次原下载SVD**的左右奇异向量。修复谱调用已有`all_tail_spectrum`，按实际整数秩和真实训练样本数计算贡献质量，无1.5上限，保留最低秩前缀、bias、conv1和fc3。原谱直接用同次U/S/V构造平衡因子，不重新SVD、不按本地训练后的奇异值排序。

## 来源与隔离

- 每个输入必须有相邻的`manifest.json`；检查角色`post_aggregation_full_server`、R5完成轮次、K1/N20/full_avg、原Hyper_CNN_512完整结构、100类、pat数据、共享头/joint，以及无其他训练目标。
- H的实际容量为`.90/.37/.35/.25/.15`按id%5；H+相同且来源谱策略为三层all_tail；L全部.90且谱策略none。核对checkpoint/manifest容量及初始化协议、源seed重新生成的.90恢复S0指纹、共同SGD与后端设置。
- 读取实际指定数据根目录；按原公共NPZ reader审计40份分片、config、文件哈希与解码张量指纹，逐项对照三个来源。H+修复质量也与实测训练样本数核对。来源格式未提供最终权重内容哈希的独立外部声明；本次记录真实输入文件/状态哈希并在结束复核，不能凭空声称更强的来源绑定。
- 原文件、数据和模型对象始终只读，结束复核其哈希；临时普通训练checkpoint与四支恢复上传在独立临时目录，运行结束或异常清理，不导出学生/服务器模型。

## 随机流、训练与预算

本地直接复用`single_avg_proto.train`和`clientTargetProj._objective`：CE＋原F `.001`，SGD lr `.005`、momentum0、weight_decay0、batch16、全模型梯度裁剪10。新SGD在实际低秩模型构造完成后创建，核对参数引用。不会从R5继续使用旧优化器。

批次仍用现有`EpochLoader`和`loader_seed(seed,client_id,loop=5)`，即原R6的私有采样协议。每客户端从稳定用途标识、源seed/轮次/客户端派生一份离线配对CPU/Python/NumPy及实际GPU随机状态，五支均恢复该状态。逐支比较真实索引序列、整轮及每epoch指纹和前后RNG指纹。**这是离线配对实验，不声称恢复了未提供的原进程R5 RNG。**

聚合采用同一份独立NumPy选择流：从源seed执行6次原全员`choice`，记录得到的R6接收顺序；四支按相同顺序调用原`aggregate_group_updates/group_avg_only`。权重只用原训练分片样本数，不用测试样本数、访问量或参数量。聚合后不做服务器截断。R7四支均按原五档**普通SVD**下发，只推理，无修复、无SGD。

每epoch只有原SGD前向/反向/更新；额外端点评价独立计成本，保护参数、buffers、模块状态和随机流。训练F/CE/总目标按访问样本加权，另保留明确命名的按SGD步均值。裁剪前范数来自原裁剪函数的返回值，只裁剪一次，触发比例/系数按SGD步汇总。训练过程F不混入端点纯CE。采集不保留计算图，观察器与optimizer临时方法异常时恢复，清除回调环引用避免累积模型。

## 输出

| 文件（CSV和JSON） | 定义 |
| --- | --- |
| endpoints | 五支逐客户端epoch0–5的训练/测试correct、samples、accuracy、纯CE；四支R7普通下发前指标 |
| cohort_endpoints | 全20人、目标C0、固定`0,5,10,15`四人；分别按对应train/test样本数加权 |
| epoch_training / cohort_epoch_training | 训练平均CE、有效F、总目标、原裁剪前范数及触发比例/系数；汇总F按训练访问量、裁剪按SGD步 |
| batch_fingerprints | 每客户端五支实际批次及每epoch指纹、配对RNG前后指纹；不默认保存逐样本明细 |
| paired_downloads | 三层真实秩、原谱/修复谱指纹、贡献质量规则及限制诊断 |
| aggregations | 四次实际full Avg权重、顺序、聚合状态指纹；R7训练步数0 |
| comparisons | 同母模型repair−raw、固定四人Hplus_repair−L_raw，以及R7普通下发repair−raw |
| fitting_trends | 相邻固定epoch的训练/测试CE及准确率变化；“训练CE下降且测试CE上升”只作事后描述 |
| local_training / costs | 实际步数、访问量、参数量、训练及只读评价/下载/聚合成本 |

准确率表内fraction、图中percent；CE为nats/sample。原型始终为0；辅助项、测试结果不参与训练或选模型。曲线为三张R6 all/fixed_large/target的训练与测试CE/准确率、固定四人的F/裁剪、R7普通下发结果，默认只保存这些图与诊断表。绘图在独立轻量进程中只读已采集JSON，不导入Torch，不影响训练随机流。

R6对照同时给出epoch0差值与相对于epoch0的差值变化。固定四人在五支均为.90，避免把客户端/容量变化混入`Hplus_repair−L_raw`。R7排除的是**R7再次修谱的直接影响**；它保留了R6起点修复与R6 SGD共同形成的结果，不能单靠R7将这两部分完全拆开。没有按测试集选择epoch或分支，不额外引入零SGD实验。

## 从服务器system目录启动

完整一条命令见 [svd_r6_replay_commands.sh](../system/svd_r6_replay_commands.sh)，只定义数组，不自动执行。它读取上一批固定输出目录的H/H+/L，不重新运行R1–R5。若目录中有多次最终模型，拒绝猜测最新；改用`--h-server-file`、`--hplus-server-file`、`--l-server-file`提供对应精确路径及相邻manifest。

图可单独从已完成表格重新生成，不重训：

```bash
python plot_svd_r6_replay.py --output-dir svd_r6_replay_runs/pat20_S5_R6_R7_seed0_20261116_v1 --epochs 5
```

验证工具：`verify_svd_r6_replay.py --output <新建合成工作目录>`。它明确使用人工R5角色的有限权重测试文件，而非伪称真实联邦R5历史；不读取/修改真实pat20产物。合成验收只证明流程，不能判断真实效果。

验收已完成：28项专项/相关回归、4项来源错误用例，以及20客户端五支完整宽度CNN的真实入口合成回放。实际500次SGD、420次原下载SVD（R6为180、R7为240）、4次Avg、R7训练0步；输出1360条端点、500条epoch训练记录、96条群体对照及5张曲线。实际批次与训练前后RNG配对、固定四人三层整数秩、CSV/JSON、加权F/裁剪汇总与输入只读检查均通过。细节见 [svd_r6_replay_acceptance.json](svd_r6_replay_acceptance.json)。

入口会在训练前用独立进程预检绘图。Windows本机验收使用既有conda环境的正常进程DLL路径，没有安装或改动包；图可单独重画，不需要重训。未运行真实服务器R5回放，不解释合成准确率。
