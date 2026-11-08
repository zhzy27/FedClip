# 单目标 helper 软课程

本模块是自定义软课程消融，不是 FedBSS 复现。基于 `34a3331`，仅在普通 `FedTargetProj + projection_softmax + Decom_CNN-5-512 + shared + joint` 全参与路径显式开启。目标客户端读取 `target_client_id`，始终走原普通训练；初始化、SVD、F、上传恢复、投影及聚合核均未修改。

## 参数

| 参数 | 默认 | 范围与含义 |
| --- | --- | --- |
| `--target_curriculum_mode` | `none` | `none / loss_rank / random_rank` |
| `--target_curriculum_strength` | `.5` | 有限，`0 <= strength < 1` |
| `--target_curriculum_warmup_rounds` | `10` | 非负整数；前10次原训练，第11次（loop=10）起生效 |
| `--target_curriculum_diagnostics` | `1` | `0/1`；关闭额外梯度统计与逐epoch打印，不改变训练 |

none、强度0、预热及目标客户端直接调用原训练，无评分loader、额外前向或排名计算。非零课程拒绝 Prox、head L2、Z2、replay、meta、KD、原型、私有头、FedRep、反馈／历史投影、部分参与、慢客户端及学习率衰减等未验收组合。`run_target_proj.py` 支持并完整转发四参数；也可直接使用 main。

## 数值与随机流

每个活跃helper每轮SGD前，用实际下载的低秩模型对自己的完整TRAIN集评分一次。eval/no_grad，保护参数、buffers、模块模式、Python/NumPy/Torch CPU及实际模型所在CUDA设备；不额外初始化其他GPU。稳定样本索引跟随真实训练batch，重用原训练sampler和collate，不额外遍历sampler。

每个真类内按CE升序取平均并列排名，`r=(rank-.5)/n_class`。random_rank使用SHA256导出的独立每类Torch Generator，仅置换同类排名；仍执行相同评分，保持同快照类内权重集合一致。

第e个本地epoch：`beta_e=strength*(E-e)/(E-1)`，E=1时为0；`w=1+beta_e*(1-2*r)`。目标是 `mean_B(w*CE_per_sample) + original_weighted_F`，分母为batch样本数，不是权重和。F计算一次；一个模型前向、一次backward、一次原裁剪和一次SGD更新。最后epoch直接使用原平均CE的反向路径。

课程不修改 `train_samples`，不把逐样本权重上传为聚合权重。原始余弦、self=.125、tau=.20和Projection不变。

## 预算和产物

三组正式配置改为 **`gr100`，实际101次普通训练／聚合**；5个本地epoch，预热仍为10次。每个helper从第11次起每轮增加一次完整自身TRAIN评分遍历；SGD步数和5次训练遍历不变。C0没有评分遍历。评分成本不能称为原训练成本相同。

完整命令在 [`system/target_curriculum_commands.sh`](../system/target_curriculum_commands.sh)，从服务器system目录供现有run_now使用。三个GPU为0/1/2，命令使用服务器已有htfllib解释器，不安装依赖。独立根目录为 `system/target_curriculum_runs/pat20_seed0_gr100_20261108_v1/{01_none,02_loss_rank,03_random_rank}`，各有logs/checkpoints/h5/final_models。已有目录层级可能追加self/prox标签及run ID，以实际日志为准。

`target_curriculum_clients.csv/json`：活跃helper按真实epoch记录；C0、none、零强度及预热保留旧路径的一条client训练汇总，**不伪造不可用的逐epoch旧损失或梯度统计**。CE/F/总损失提供样本访问加权均值；活跃epoch另存F和总目标的SGD步均值。评分访问、批次和耗时只在epoch1记一次；分位数与下载指纹是整次评分的描述。诊断关闭保留基本损失/预算，额外梯度范数为空。

`target_curriculum_primary.csv/json`、H5与最终manifest明确主指标为目标客户端普通post-local准确率（fraction）。聚合后实际下载的准确率单独列出，不能把两种时序的差值称为本轮微调收益。final、同轮曲线最早best、last10/20沿用原统计。显式课程协议下H5根`rs_test_acc`为主指标，旧全员诊断曲线保存在`legacy_server_diagnostic_test_acc`。普通未传新参数的旧运行不改字段语义。

## 本地验收

冻结参照为提交 `34a3331` 的原客户端和服务器代码。独立单元检查排名、梯度、F一次、数据索引、异常恢复、诊断隔离及launcher；runtime比较关闭、零强度、预热的低秩预测checkpoint、恢复上传、聚合、批次及RNG。

```powershell
$env:PYTHONPATH='tests'
python -m unittest -v test_target_curriculum test_target_curriculum_runtime
python system/verify_target_curriculum.py --output E:/target_curriculum_acceptance_unique
```

真实main合成验收使用20客户端、原宽度异构低秩CNN、三个训练轮次、每次5epoch；仅为覆盖边界将验收warmup设为1。分别测试none/loss_rank/random_rank及loss诊断关闭；正式命令warmup仍为10。无真实数据性能结论，合成验收不能证明C0涨点。最终计数与结果见同目录验收JSON。
