# 离线蒸馏验收（2026-10-26）

实现基线：`target_proj`，父提交 `5f19d464f3babca394eb0198234dabf7d9ea52d7`。本次新增独立离线入口；联邦main、训练、聚合路径未改。唯一公共接口扩展是既有head诊断评价函数的可选batch_size/返回预测，缺省值及旧字段保持不变。

## 环境与范围

Windows，本地既有 `D:\miniconda\envs\htfllib\python.exe`：Python3.11.14、PyTorch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3、torchvision0.15.2+cu117。使用CPU、2线程；训练子进程deterministic/cudnn deterministic均为True。未安装/升级依赖，没有完成CUDA训练验收。现有可选transformers/fused-window警告仍存在，没有影响本次CNN执行。

原包：`C0_three_models_R201_20261025_214112_203491.zip`，39,694,724 bytes；SHA256：

```text
0c32555211b0e412260de9720ff1d1b6b4a1747f3df8fdc8f3debcc0113800b8
```

确认其中是C0、完成201次训练、loop200、归属模型0的三个普通低秩预测模型，rank .9、3,246,372个去重参数，原分类器完整保留。包内有750张原测试图、原config和分组文件，**没有原训练集**；本机也没有对应pat_20训练分片。因此下面真实检查只建立未训练基线，不能声称已完成真实5epoch蒸馏或复现训练集655/1595的选择计数。

## 真实原检查点，直接低秩前向

使用 `--native-test-only --test-from-bundle --client-id 0 --assigned-model-id 0 --completed-round 201 --device cpu --threads 2`，未执行恢复/SVD/训练：

|冻结模型|correct/samples|test accuracy %|test CE|
|---|---:|---:|---:|
|A₀，model0|343/750|45.733333|3.194584757487|
|B₀，model1|312/750|41.600000|3.240997726440|
|C₀，model2|320/750|42.666667|3.283768353780|

两次独立读取评价一致。输入包保持只读；实际模型元数据、别名、秩与分组核验通过。源包不包含显式语义class_order或逐分片训练哈希，未将这些缺失证据标成已核验。正式服务器训练会核对原分组内的全数据指纹，或核对用户提供的原data_manifest与实际C0分片。

本地可读原始输出：`system/temp/offline_distill_native_final_20261026/`。

## 单元与旧路径回归

新增10项测试全部通过（约3.98s）：

- 真实类别概率选择、A/B/C确定性并列优先级，不按最大预测概率选择。
- 概率平均与logits平均区别；KL方向、按类别求和/样本平均、T及仅一次T²；选A₀时仍有蒸馏损失。
- λKD=0的两蒸馏模式与继续训练逐元素一致；与独立普通SGD＋原_objective＋裁剪参照逐元素一致；batch与Python/NumPy/Torch CPU/CUDA状态摘要一致。
- 教师无梯度、参数/buffers/模式不变；原checkpoint文件哈希不变；学生fc3正常更新。
- 缓存/直接前向一致，缓存命中和内容损坏检测、稳定样本索引、随机流不前进。
- 实际CNN结构运行三个模式，禁止调用recover/decom（用抛异常替身验收），epoch0/每epoch指标、最终低秩模型加载、跨运行固定epoch对照及错配拒绝。
- checkpoint客户端/轮次/容量/类顺序/别名/归属错配拒绝。
- 原数据全指纹及实际读取分片指纹核验；缺数据明确失败，不使用合成回退。
- 额外评价在私有副本上执行，带可变buffer模型也不改变原参数/buffers/模式/RNG。
- 三条正式命令解析、预算、独立输出/GPU、非法数值配置检查。

另有60项既有回归全部通过（约21.26s）：

```text
test_head_diagnostic
test_head_diagnostic_initialization
test_target_projection
test_projection_variants
test_group_projection
test_multi_group_svd_init
```

CPU参数及随机流回归要求精确相等。log→softmax的float64数学单元仅允许2个machine-epsilon的舍入误差；没有放宽训练参数回归容差。

## 实际命令行、原宽度低秩CNN合成验收

执行：

```bash
python system/verify_offline_distillation.py \
  --output-dir system/temp/offline_distill_smoke_final_20261026 --epochs 2
```

该工具通过子进程运行真实 `offline_distillation.py` 入口两次，分别λKD=1和0，每次三种模式、2epoch。客户端2、归属模型1，以覆盖非C0/非model0；模型是原Hyper_CNN_512、width16、100类、rank .9、3,246,372参数，7张合成训练图、5张合成测试图。每模式实际访问14个训练样本、更新4步。

|项目|结果|
|---|---|
|三组epoch0状态、batch、训练RNG摘要|完全一致|
|零KD时kd_mean/kl_dp相对continue参数最大误差|0 / 0|
|缓存与直接前向最大误差|0；固定容差atol1e-6、rtol1e-5|
|教师状态与输入文件|完全不变、无教师梯度|
|最终学生重载后的预测|与保存时逐样本一致|
|固定终点对照、CSV/JSON/模型元数据|通过；配置错配拒绝|

正KD时两种蒸馏学生与继续训练的最大参数差分别约0.00421690、0.00369948，仅说明蒸馏项实际生效，**不代表真实准确率收益**。

这次CPU微型数据上的每模式2epoch训练时间约1.58–1.65s，教师缓存与直接校验前向约0.05s；包含三组及准备/评价/保存的正KD子进程内部墙钟约8.27s。此计时不能推算完整2250样本/GPU的正式耗时。成本分别写入teacher_cache、每epoch metrics和summary。

本地可读报告：`system/temp/offline_distill_smoke_final_20261026/acceptance.json`，及两个case的所有CSV/JSON/log。验收后已删除自己生成的临时合成checkpoint、学生模型和logits二进制，仅保留可读报告。未删除原模型包、原数据或其他实验。

## 交付判断

离线入口、数学/隔离/导出及CPU合成短程验收完成，已提供固定5epoch的三条正式命令，但未执行正式实验。真实训练效果与655/1595训练选择计数需要在服务器原pat_20数据上运行后判断；不能凭合成结果宣称知识迁移有效。

输出、参数、缓存、固定终点比较及命令使用方式见 [offline_distillation.md](offline_distillation.md)；正式数组为 [offline_distillation_commands.sh](../system/offline_distillation_commands.sh)。未推送，未修改绘图文件或真实数据。
