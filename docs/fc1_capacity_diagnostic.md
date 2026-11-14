# fc1 本地容量：B-fc1 S101 → 离线 R102

独立入口：`system/fc1_capacity_diagnostic.py`。仅复制输入模型进行离线本地训练，未修改任何联邦入口、SVD、损失函数或聚合代码。默认不保存模型，只保存诊断表、汇总和来源清单；不反馈服务器。

## 来源校验与使用

输入为 B-fc1 的最终完整 `Server_model_0.pt` 和相邻 `manifest.json`，**不读取 FA02 原型包或 R101 客户端模型**。`--source-run` 只接受该目录直接给出的唯一服务器，或 `final_models/**/Server_model_0.pt` 的唯一结果；多个候选时失败，可改用 `--server-file` 精确指定。

默认来源完成101次聚合，结果明确标为“基于 S101 的离线 R102 本地训练”。`--source-round` 可配置，合成验收使用 S2→离线R3，不冒充真实 R102。

校验 K1、20客户端全参与、`group_avg_only + full_avg`、共享头/joint、原五档容量和 id%5 分配、`.90 low_rank_recovered` 初始化、`next_tier`、仅 fc1、gain1.5、无原型/其他辅助损失，并核对源 SGD/local5/lr.005/batch16/F.001/clip10。检查服务器角色、轮次、完整层类型、别名一致性、模型结构、容量、初始化和谱策略在 checkpoint/manifest 间一致。按源 seed 重建初始化核对哈希。

实际读取原 NPZ，不生成或改写数据。对全部20客户端训练/测试分片核对 config SHA256、文件 SHA256、解码 x/y 指纹和样本顺序，与源 manifest 完全一致。不存在、Dirichlet/错误分片或内容不同均失败，不靠目录名猜测划分。源格式没有最终服务器文件哈希声明，因此不能额外证明一个被人为替换但元数据完全相同的服务器；工具记录实际输入文件与状态哈希供追溯。结束时复核所有输入哈希。

从服务器 `system/` 目录执行以下完整命令，GPU直接改 `-did`；默认输出目录必须不存在：

```bash
/data/zh/miniconda3/envs/htfllib/bin/python -u fc1_capacity_diagnostic.py -did 0 --device cuda --source-run svd_spectral_runs/pat20_seed0_init090_gr100_20261113_v1/B_fc1 --source-round 101 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --client-ids 0,10,18 --fc1-ranks 120,200,720 --threads 4 --diagnostics 1 --output-dir fc1_capacity_diagnostic_runs/B_fc1_S101_offline_R102_r120_200_720_20261114_v1
```

同一命令的 `declare -a COMMANDS` 见 [命令文件](../system/fc1_capacity_diagnostic_commands.sh)，只定义、不执行。支持 `--server-file` 替代 `--source-run`、可配置客户端/整数秩子集、数据根目录、线程、GPU和输出目录。秩限制在源实际最低/最高 fc1 秩120..720，禁止重复；未请求的比较对或720回退不会伪造结果。CPU可用 `--device cpu`。

## 一份公共谱，三个 fc1

只对完整 S101.fc1 做**一次**原 `torch.linalg.svd(full_matrices=False)`。用原 `correct_spectrum` 按实际 k=120、b=200、窗口101..120计算 next_tier/gain1.5：121..200逐秩修正，之后沿用200的实际倍率。

在同一组 P/Qh 和修正谱上平衡构造最大请求秩的 U/V，再切出120、200、720的共同前缀，保证三档公共槽位逐元素相同。保留服务器 fc1.bias。每个客户端的 conv2/fc2 只做各自原容量的一次正常 SVD；conv1、fc3、bias和 buffers 保持来源。C0/C10 的其他层为 .90（conv2=72、fc2=461），C18 为 .25（conv2=20、fc2=128）。

`model.ratio_LR` 保留客户端原值，fc1实际整数秩单独记录。替换 fc1 后调用原 `_rebuild_base()`，再创建无动量、无weight decay SGD；原训练器核验优化器参数身份。构造与评价保护 RNG，三分支从同一训练 RNG 起点恢复，采用原 `single-avg-proto-v1` seed/client/loop=101 的私有训练 loader，记录稳定样本索引哈希并强制核对三档相同。没有重新排序、额外抽样或补训。离线训练 RNG 从来源seed独立建立，不接续整个联邦进程的 `MultiGroup_rng.pt`，不声称重放完整联邦 R102 的随机轨迹。

训练直接调用未修改的 `utils.single_avg_proto.train`：CE＋原 F.001，完整参数裁剪10，5 epoch，每 batch 一次正常前向、一次联合 backward、一次 SGD。原训练器的中间 checkpoint 只写在新输出目录内的私有临时目录，退出删除；不作为结果导出。无原型采集、原型库或通信，恢复上传仅为复用原训练器，未执行服务器聚合。

## 输出与指标口径

所有表提供 CSV/JSON；manifest 保存输入/数据/初始化指纹、请求参数、有效源配置、公共修正谱收据、实际秩、任务/步数/样本访问量及只读检查。

| 文件 | 内容 |
| --- | --- |
| `endpoints` | 各客户端×秩，epoch0/1/5 的原训练集与测试集 correct/samples、准确率、**纯 CE**；准确率为 fraction |
| `fixed_slot_updates` | 下载时及第1步、epoch1/5，1–120、121–200、201–720 固定槽位块的初始范数、绝对乘积变化、相对变化 |
| `first_batch_CE_signal` | 用首个真实训练前向读取 `G=dy.T@fc1_input`，各段下载 P/Q 方向投影和 CE 的 U/V 梯度范数；不含F |
| `first_step_gradients` | 实际 CE＋F 各段 U/V 裁剪前后梯度、全模型总范数、实际裁剪系数 |
| `local_training` | 原训练过程平均 CE、加权F、总目标、全程裁剪比例、梯度范数、参数量、实际 batch/epoch/访问量、批次及训练 RNG 指纹 |
| `batch_order` | 稳定样本索引的每batch哈希，不保存图像或逐样本特征 |
| `rank_comparisons` | 每客户端200−120、720−200的 epoch0/5测试差及差距变化；CE与准确率分别记录 |
| `tail_rollback` | 720秩训练后只恢复201–720到**该分支自身epoch0因子**的测试计数、CE、准确率及相对训练后模型的差 |
| `capacity_summary` | 仅对本次被选中的客户端按原容量/实际fc1秩分组，累加 correct/samples；不是全20人联邦主指标 |
| `summary.md` | 逐客户端初始/末期秩差和尾部回退对照表 |

固定槽位统计使用实际参数的私有 float64 CPU 副本直接计算 `U_segment @ V_segment`，不重排因子；相对变化除以该块下载时 Frobenius 范数，初始块为零时记null并注明原因。未保留的槽位标明 `present=false`，不误解为“该段已参与训练但没学动”。

同一表另外记录**重新 SVD 后实际矩阵的谱能量**，独立命名。训练后槽位未必正交，不将块范数平方相加冒充谱能量。重 SVD 是额外只读诊断，默认每分支在epoch0/第1步/epoch1/epoch5执行，不属于下发或训练更新；与公共下载fc1的一次SVD分开。

CE探针读取已有训练图，用 `autograd.grad(..., fc1_output, retain_graph=True)`，不重复模型前向、不写 `.grad`；真实训练仍只反向/裁剪/更新一次。梯度hook只观察，异常时移除并恢复损失及optimizer.step。所有endpoint评价保护参数、buffers、模式及Python/NumPy/CPU/实际CUDA设备的 RNG；临时评价不保存或覆盖普通 checkpoint。`-did`在导入Torch前限制可见设备，避免旧恢复辅助函数触碰其他物理GPU。

回退推理不用SVD、不清零、不替换其他层：只复制720分支模型，将两侧201–720因子恢复自身下载值，强制核验1–200、fc1.bias、其他参数和buffers来自训练后快照。它不同于重新训练200秩模型，也不同于最佳rank200截断。

端点评价的CE与训练过程平均CE分开。训练墙钟含在线观察，并单列诊断时间；评分数据没有进入优化器。测试不参与选秩、选epoch或选模型。

## 验收及解释边界

[验收报告](fc1_capacity_diagnostic_acceptance.json)：专项数值测试、观察开关隔离、原生客户端参照、真实入口合成 smoke 与旧入口回归。

```powershell
D:/miniconda/envs/htfllib/python.exe system/verify_fc1_capacity_diagnostic.py --output E:/fc1_capacity_unique
```

验收先用真实联邦main生成20客户端、原宽度模型、两轮合成 B-fc1 来源，再用新CLI执行9组 local5（另跑关闭诊断的9组配对）。这属于 **S2→离线R3**。C0/C10的720秩、C18的200秩还与原 `clientMultiGroupSVD.set_parameters/train` 对照，CPU参数、损失、批次、RNG精确一致。默认没有模型输出；CSV/JSON、缺失/错轮次/错误数据指纹及只读源文件核验均检查。

本机没有真实 B-fc1 S101 和匹配pat20分片，未运行用户真实9组，也未验证真实准确率。判读应结合 epoch0 差异、epoch5 新增差异、裁剪后的实际块更新及条件化回退：720≈200且高秩块小、回退无影响，支持本地高尾部作用弱；块更新明显而回退更好，支持其在该训练后模型中拖累测试。该离线实验不证明跨轮保留能力。

## 同秩原谱／修正谱：`--comparison spectrum`

`--comparison` 默认 `rank`，保留上面的120/200/720容量诊断及旧命令。新增模式要求显式 `--fc1-ranks 200`，其他秩配置报错，不悄悄覆盖用户参数。默认客户端仍是0、10、18，每客户端 raw/corrected 各独立 local5，共6次本地训练。

来源仍严格要求 **B-fc1 S101**；raw 不是 D01 来源，也不是原随机初始化。两组均从同一个 B-fc1 完整检查点出发：对其 fc1 只做一次 SVD，raw 用该检查点的原始奇异值，corrected 对该谱再执行一次原 next_tier/gain1.5 下发规则。使用同一 P/Qh、相同200秩矩阵尺寸和平衡平方根，前120槽位逐元素相同；bias、其他层、分类头、buffers和参数量相同。两者121–200因子、下载矩阵、epoch0 logits/CE/准确率允许不同。若拟合跳过或实际增益为1，也允许数值相同，公共谱收据会注明实际修正。

C0/C10的conv2/fc2维持.90，C18维持.25，`ratio_LR`不变；两组都使用原CE＋F.001、clip10、SGD .005/local5/batch16和各自原数据。每客户端恢复同一训练RNG，核对批次指纹及训练结束RNG。所有新行为只在离线工具中生效，联邦入口与聚合未改。

分支表 `endpoints`、`fixed_slot_updates`、`first_batch_CE_signal`、`first_step_gradients`、`local_training`、`batch_order` 都增加 `spectrum_variant=raw|corrected`。固定槽位只观察1–120、121–200，更新量使用**各自分支下载时**的U/V计算；保留epoch0、第1步、epoch1和epoch5的绝对/相对变化。原梯度和端点评价口径不变。

新增 `spectrum_comparisons.csv/json`：先按 `(client_id,spectrum_variant,epoch,split)` 验证完整、唯一的配对，再输出每客户端的宽表：

- epoch0/1/5各组的train/test correct、samples、纯CE和准确率。
- 每观察点的corrected−raw准确率/CE差。
- epoch5差−epoch0差，区分下发时优势与本地训练后的差距变化。
- raw、corrected各自的epoch5−epoch1训练CE和测试CE变化。

`capacity_summary` 同样按谱分支分开，不能因为秩都是200而混在一起。`summary.md`展示测试准确率差及两组训练/测试CE变化；这里没有验证集，也不根据测试选择epoch。spectrum模式**不执行、不保存**201–720回退表，不生成旧rank比较表，不导出模型。

manifest 的 `source.source_config`保留检查点本身的B-fc1来源，`spectrum_comparison.offline_download_variants`单独说明此次raw/corrected干预；公共原谱和修正谱各有哈希。预测结果差异不能解释成来源检查点的差异，也不能把epoch0下发优势全部算成本地学习收益。

从服务器 `system/` 直接执行：

```bash
/data/zh/miniconda3/envs/htfllib/bin/python -u fc1_capacity_diagnostic.py -did 0 --device cuda --comparison spectrum --source-run svd_spectral_runs/pat20_seed0_init090_gr100_20261113_v1/B_fc1 --source-round 101 --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --client-ids 0,10,18 --fc1-ranks 200 --threads 4 --diagnostics 1 --output-dir fc1_capacity_diagnostic_runs/B_fc1_S101_offline_R102_raw_vs_corrected_r200_20261114_v1
```

同一条命令的数组放在 [独立命令文件](../system/fc1_spectrum_diagnostic_commands.sh)，不会自动执行，也不修改旧rank命令。新输出目录必须不存在。

[本次验收报告](fc1_spectrum_diagnostic_acceptance.json)记录合成CPU检查和冻结6697331的rank回归。可独立生成合成B-fc1来源：

```powershell
D:/miniconda/envs/htfllib/python.exe system/verify_fc1_spectrum_diagnostic.py --output E:/fc1_spectrum_unique
```

也可传 `--source-workspace`，复用前一验收工具已经生成并验收的**合成**B-fc1目录，避免重复生成来源模型。此工具不读取或改写真实分片，不代表用户真实S101/R102已经运行。
