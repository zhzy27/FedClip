# 双 batch 目标训练数据共享验收

参照：改动前 `target_proj` 提交 `1abe63f`。本地没有真实 `dataset/Cifar100/pat_20/train/0.npz`；本报告只说明数学、行为及合成真实入口验收，不证明真实准确率收益。

## 已通过

1. `tests/test_target_c0_replay.py`：7 项。
   - 关/零权重不增加目标数据读取和前向；目标客户端即使开关启用仍沿用原训练。
   - 两路梯度符合加权 CE＋一次 F；两路均可反传，每对 batch 一次 SGD。包括不等末 batch。
   - 独立源为实际 target ID=2 的**训练**数据；非 C0 目标、不同 helper/轮次 seed、私有数据流重启与可复现。
   - 自身 5 样本、目标 3 样本、batch2、local2 的用例：自身 10 次访问、目标 9 次访问、6 次 SGD、两次源迭代器重启；`train_samples` 仍为 5。
   - 明细开关不改变参数和 RNG，私有目标采样不改变原自身采样顺序；参数范围与组合校验、启动器转发和完整命令检查。
2. `tests/test_target_c0_replay_runtime.py`：冻结 `1abe63f` 客户端/服务器与新版本 omitted / explicit none / batch_pair weight0，原五档异构 CNN、两轮。
   - 普通低秩 checkpoint、完整恢复上传、服务器聚合结果、上传 ID 顺序、实际训练批次和 Python/NumPy/Torch CPU/CUDA RNG 摘要均严格一致，最大参数误差 0。
3. `verify_target_c0_replay.py`：真实 `main.py`，20 客户端、原宽度 16 的 `Decom_CNN-5-512`、原五档容量、每次 local5、两轮；开启 replay、比较逐客户端诊断开/关。
   - 两个用例分别完成 40 次普通本地训练、200 次 SGD，200 个自身 batch、190 个额外目标 batch，共 390 次训练前向。
   - 实测自身样本访问 590，额外目标样本访问 380；原 `train_samples` 和样本权重保持原分片计数。
   - C0 每步一次前向，helper 每步两次前向；没有重复调用客户端训练。
   - 与原 `aggregate_projection_weighting` 在相同恢复上传上的参数最大误差 0；同时核对原 softmax-only 与 projection-softmax 的逐客户端权重一致。
   - 恢复、聚合和聚合后评价不改普通上传 checkpoint 文件哈希。
   - 诊断开/关：实际自身 batch、每次训练结束参数、每轮服务器参数、RNG 和主指标 summary 完全一致。
   - CSV、JSON、JSONL、H5、manifest 的训练明细/主指标一致；final、最早 best、last10、last20 由普通目标 post-local 序列复算。
   - 21 个最终模型逐一重载，对应参数及固定输入预测相同。日志 ZIP 不含模型或数据分片。

原始轻量凭证：`system/temp/target_c0_replay_verified_20261030/acceptance_report.json` 和 `diagnostics_on/off.json`。可重新运行独立验收脚本。较大的合成模型/checkpoint 已在验收成功后清理，未删除原实验文件。

另有 **150 项旧路径回归全部通过**：Projection、Softmax、post-local 评价、启动器、Prox、local head、head L2、FedRep、多群组原型/Z2、KD、SVD，以及刚完成的单目标 Z2 单元测试。未修改上述聚合核或多群组辅助损失模块。命令数组通过 `bash -n`，改动通过 `git diff --check`。

## 环境与边界

使用现有 `D:/miniconda/envs/htfllib/python.exe`，Python 3.11.14、Torch 2.0.1+cu117、NumPy 1.26.4、SciPy 1.16.3、h5py 3.15.1；上述 smoke 在 CPU 上完成，没有修改环境。CUDA 的真实数据效果及成本仍由服务器实验验证。

聚合核、原 `_objective()` 返回值、SVD/容量和初始化代码均未改。共享的目标训练数据仅作为额外监督；源训练数据访问量不计入聚合样本量。不能把该实验与原方法称为相同计算或数据可用性预算。

正式数组只定义一组 gr100/local5、self=.125、tau=.20、replay=.5 的配置，不自动执行。服务器应按 [说明](target_c0_replay.md) 收集真实运行预算和 C0 普通 post-local final/best/last10/last20；本次未启动正式长实验。
