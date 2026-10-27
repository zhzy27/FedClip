# 类别原型约束验收（2026-10-27）

基线为target_proj提交`5d6103a`。仅增加独立prototype开关，不修改模型forward/state_dict、原聚合核、普通评价口径；默认关闭。未推送、未修改环境/真实分片/绘图，未启动31轮正式训练。

## 环境及参照

已有htfllib：Python3.11.14、PyTorch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3、h5py3.15.1。CPU、每子进程2线程；没有CUDA验收。原联邦backend保持原值（deterministic=False、cuDNN deterministic/benchmark=False、TF32按原设置），未借用离线脚本的确定性设置。

参照从Git对象5d6103a读取原SVD客户端train方法；数据、下载、恢复、分组、聚合及评价使用本次没有改变的原计算路径，不是新原型实现与自身比较。当前Git克隆需含该对象才能运行参照工具。

## 单元及回归

11项原型检查通过：

- 不等类别样本量正确合并sum/count，结果不同于客户端方向等权平均。
- 群内有类时修改组外同类统计不影响原型；群内缺类确实用组外；不同列独立。
- 无样本、群内零向量、非有限统计与全无效batch处理；群内零向量不触发组外替代。有限极大特征的全无效batch也返回有限零项。
- hook取到实际fc3输入，等于ReLU(base(x))，不同于包含负值的base输出；原forward、state_dict及hook移除正常。
- 原型停止梯度，骨干获得非零有限梯度，原型项不对fc3参数产生直接梯度。
- 真实逻辑包往返、字节长度、schema/kind及截断检查。
- 全体采集屏障、每客户端全K模型、原数据/下载状态与RNG保持不变。
- 独立带BatchNorm buffers模型验证采集不改变真实参数、buffers、模式、requires_grad及随机流。
- 方向诊断开关不改变包内容、本地训练、完整上传及随机状态；过期库拒绝。
- 关闭/零λ/首轮与固定旧train参照参数及完整上传精确一致，无额外特征前向。
- 非法参数/不支持组合、首轮分组兼容性及四条31轮命令检查。

96项既有回归通过：

```text
test_multi_group / test_multi_group_warmup
test_multi_group_svd / test_multi_group_svd_init
test_multi_group_kd
test_target_projection / test_projection_variants / test_group_projection
test_offline_distillation
```

CPU张量及随机状态要求精确相等；未放宽原容差来掩盖回归。

## 真实main入口合成验收

```bash
# 仓库根目录
python system/verify_multi_group_proto.py \
  --output system/temp/multi_proto_acceptance_20261027 --jobs 2

# 完成后只读取复核，不重训：
python system/verify_multi_group_proto.py \
  --output system/temp/multi_proto_acceptance_20261027 --audit-only
```

原宽度Hyper_CNN_512、20客户端/K3/7-7-6、原五档容量、low_rank_recovered初始化秩.15、每模型5epoch、两轮。各客户端合成训练2或3样本、测试1至4样本；原pat_20没有读取或重划。每case正常120次模型本地训练、600个SGD步、1500次训练样本访问。

|case|实际原型训练任务|采集前向样本数|结果|
|---|---:|---:|---|
|legacy：旧train参照|0|0|通过|
|zero：group_priority，λ0|0|0|通过|
|projection：原Projection＋λ.1|60|150|通过|
|avg_only：原群内Avg＋λ.1|60|150|通过|
|no_diagnostics：Projection＋λ.1，关闭方向表|60|150|通过|

关键结果：

- 零λ的全部本地模型、完整上传、服务器聚合、batch顺序、主指标及Python/NumPy/Torch CPU/CUDA状态摘要与旧参照精确一致。
- 所有case初始化、首轮上传及分组一致；原型配置没有进入分组兼容指纹。两种聚合在首轮以后的服务器参数自然不同，不要求其原型库相同。
- 用真实server.prepare入口检查：每轮全部客户端已下载、均无上传，采集/生成/分发结束后模型及RNG与进入前完全一致，随后才开始任何SGD。首轮没有原型或采集。
- 正λ第二轮是全部20×3=60个模型任务生效，不只所属列；每人仍3×5数据epoch。额外采集150个样本前向、90个前向batch，未计为SGD步。
- Projection与关闭方向表case的完整本地/上传/聚合参数事件、batch/RNG、主指标序列精确一致。
- 每列按同一恢复上传快照独立调用原群组核，最大参数误差0；不等测试样本量的全员、群组和容量档指标、最早Best及尾部点数检查通过。
- CSV/JSON/H5/持久化JSONL相互核对，模型重载预测一致；当前原型包的形状、来源轮次、指纹及实际字节核验通过。合成模型清理后audit-only读取原验证回执，不声称重新加载已删除文件。

本批合成数据全局只有50个实际出现的训练类，第二轮三套库共150个有效/150个无效类别原型。group/outside/invalid三种来源均实际出现；不会把这批合成成员名单或标签统计写入正式算法。

## 字节与成本实测

Projection case第二轮：

- 采集150个模型-样本前向，90个batch，采集墙钟约7.02s。
- 合并约0.52s、分发约0.03s。
- 实际统计包上行总计24,613,446 bytes。
- K套原型的一个下发包1,231,199 bytes，20接收者合计24,623,980 bytes。
- 原SGD总访问量1500、步数600保持不变，SGD部分耗时约160.6s。

统计使用稠密float64总和/int64计数，库为float64原型/bool有效掩码；因此微型样本数下包大小仍按K×C×D增长。上述是实测canonical in-process字节，含header，不是网络协议链路测量，也不混入原完整模型通信。并行CPU合成计时不能外推真实数据/GPU成本。

## 交付边界

原始可读报告在 `system/temp/multi_proto_acceptance_20261027/acceptance_report.json`，每case有相应JSONL/CSV/JSON/H5及参数/RNG事件。验证重载后，工具仅清理自己独立目录里的合成模型与类库二进制，保留约22.5MiB记录；日志ZIP检查不含.pt/.pth/.packet或逐样本数组文件。

实现与合成验收完成，不代表真实准确率改善。本机没有原pat_20训练数据，也没有运行任何31轮真实实验；先按交付的无原型/.01/.1/1四组短程协议观察损失尺度、容量和准确率，再决定后续预算，不能据此自动延长。

参数、实现和输出说明：[multi_group_class_prototypes.md](multi_group_class_prototypes.md)；完整四条数组：[multi_group_proto_commands.sh](../system/multi_group_proto_commands.sh)。
