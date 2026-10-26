# 在线跨组KD验收，2026-10-26

基线提交：`e1e7f61`。本次是从初始化开始的FedMultiGroupProj多模型SVD实验，原离线蒸馏入口未修改。未推送、未修改环境、未启动真实101轮实验或改写真实数据。绘图文件保持原值。

## 环境与有效设置

使用已有htfllib环境：Python3.11.14、PyTorch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3、h5py3.15.1、torchvision0.15.2+cu117。验收为CPU、2线程；未做CUDA验收。可选transformers/fused-window旧警告不影响本次CNN执行，未安装依赖来消除警告。

实际main子进程backend：deterministic=False、cuDNN deterministic=False、benchmark=False、matmul TF32=False、cuDNN TF32=True，与旧联邦入口一致；未套用离线脚本的确定性配置。

模型为原宽度Hyper_CNN_512、100类、五档SVD秩.9/.37/.35/.25/.15、id%5容量分配、低秩恢复母模型初始化秩.15。使用原CE＋F .001、SGD .005/momentum0/weight_decay0、clip10、batch16；每模型5epoch，无预热、首轮full_weights自动分组，group_projection_softmax质量.50/.50、聚合温度.2，KD温度2、系数.1。

合成分片隔离在验收目录内，20客户端，每人训练2或3样本、测试1至4样本，专门覆盖不等样本量。没有使用或生成原pat_20分片。

## 单元及旧路径回归

新增9项在线KD检查：

- full逐教师概率平均、KL(q∥p)方向、按类别求和/样本平均，λ及T²仅一次。
- not_true独立删除每个教师/学生的真类logit，再归一化平均；独立公式校验；删除项直接logit梯度精确为0，教师无梯度。
- 原始低秩下载状态和训练数据指纹核验；缓存前模型已更新时拒绝；缓存固定及过期轮次拒绝。
- 非0所属列、K2单教师、K3多教师；其他列保持原训练，原模型未永久冻结。
- K1无教师缓存/额外前向，实际KD=0。
- 关闭、零系数、首轮的参数、恢复上传与RNG精确等于固定e1e7f61原训练方法参照。
- 探针开关不改变训练结果和RNG，固定样本跨pre/post一致。
- 带非平凡BatchNorm buffers的教师副本检查：原buffers/模式/梯度标记及backend/RNG不变。
- 参数非法组合拒绝、KD不进入分组兼容指纹、正式数组预算与共享分组检查。

另有87项既有回归通过，覆盖：

```text
test_multi_group
test_multi_group_warmup
test_multi_group_svd
test_multi_group_svd_init
test_target_projection
test_projection_variants
test_group_projection
test_offline_distillation
```

没有放宽CPU参数或随机状态的精确相等要求。

## 真实main入口：20客户端、K3、两轮

执行：

```bash
python system/verify_multi_group_kd.py \
  --output system/temp/multi_kd_acceptance_20261026

# 纯读取复核，不再次训练：
python system/verify_multi_group_kd.py \
  --output system/temp/multi_kd_acceptance_20261026 --audit-only
```

五组均通过：

|case|训练方法|两轮KD启用任务|训练记录|探针记录|
|---|---|---:|---:|---:|
|legacy|固定e1e7f61原train参照|0|120|0|
|zero_full|full，λ=0|0|120|0|
|full|完整输出KD|20|120|40|
|not_true|非真类KD|20|120|40|
|full_no_probe|full，关闭探针|20|120|0|

每case实际120次模型本地训练、600个SGD步、1500次训练样本访问；每客户端每轮3×5=15次数据遍历。缓存和探针没有增加SGD步或额外本地阶段。两种KD都在第二轮且仅20个真实所属模型任务启用，覆盖所属ID0/1/2；首轮60个任务全部无KD。

共同检查结果：

- 零KD与旧版参照的所有本地模型、完整上传、服务器参数、批次及Python/NumPy/Torch CPU/CUDA状态摘要精确一致。
- 全部case初始化、首轮完整轨迹与分组严格一致；所有case的完整训练batch顺序及最终RNG一致。
- full/not_true及关闭探针case在第二轮的下载教师状态、原始logits缓存来源/索引指纹精确一致。
- full与full_no_probe的完整本地/上传/聚合参数事件、batch、RNG及主指标序列精确一致。
- 每列独立调用原群组聚合核的最大参数误差为0；独立float64全权重余弦与分组输入矩阵在固定1e-12容差内一致。
- 按不等测试样本量独立复算全员、逐群组及容量档指标，最早Best/评价轮次/Last10/20口径检查通过。
- CSV数值、JSON、H5表内容、逐客户端持久化JSONL、协议与最终manifest互相一致；最终服务器/客户端低秩模型重载预测一致。

本批合成分组为：`[0,3,5,10,11,13,19] / [1,2,6,8,12,15,18] / [4,7,9,14,16,17]`，仅记录验收产物，不写入正式命令或算法。正式数据仍从首轮自动产生其自己的分组。

## 成本与边界

本次小型CPU合成case中，full的SGD总耗时约141s；20个客户端第二轮教师前向合计约0.15s、缓存准备（含状态/数据核验、复制等）约4.54s、40条pre/post探针观察约10.26s。not_true对应前向约0.16s、缓存约4.54s、探针约10.34s。微型数据下复制/指纹开销占比明显，不能据此推算真实数据或GPU耗时；各项成本已独立记录，探针频率/数量可配置。

源码参照不是“新实现与自身比较”：测试从Git对象e1e7f61读取原SVD client.train，下载、恢复、聚合、数据读取/评价等沿用本次未改变的原实现。模型/数据不来自真实训练，不能据此声称准确率提升。

完整可读报告在 `system/temp/multi_kd_acceptance_20261026/acceptance_report.json`，各case包含日志、JSONL、CSV/JSON/H5及参数/RNG事件。已在验证重载后清理该目录内合成模型二进制，剩余记录约22MiB，无遗留.pt文件。

本机缺少原pat_20数据及旧正式201轮的完整日志，尚未确认其前101轮是否可直接作为对照。提供的正式数组只含full/not_true两条，另附可选none基线；均由用户决定启动。真实准确率、同化程度与正式运行成本需要服务器结果判断，不用合成准确率作结论。

命令和参数、文件路径说明：[multi_group_online_kd.md](multi_group_online_kd.md)；完整数组：[multi_group_kd_commands.sh](../system/multi_group_kd_commands.sh)。
