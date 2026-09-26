# FedTargetProj 第一阶段实验

分支 `target_proj` 基于 `simple_v` 的 `a46b5e4`。原有 `serverCLIP.py`、
`clientCLIP.py`、`serverbase.py`、模型和本地优化器均未修改。

## 实现与对照

`clientTargetProj` 继承 `clientCLIP` 的训练和下发逻辑；`FedTargetProj` 继承
`FedCLIP` 的训练循环、样本量权重和原有 accuracy/loss 记录，只覆盖聚合并添加观测。
三种模式全部客户端仍然参与训练，只有服务器写回参数的规则不同：

- `avg`：原始样本量加权平均；保留基线逐参数加权求和的浮点计算顺序。
- `target_only`：服务器参数直接取目标客户端恢复后的参数，目标权重为 1。
- `projection`：以目标 delta 为参考，仅对其他客户端中全模型内积小于 0 的
  delta 应用 `delta_j - dot(delta_j, delta_k)/(norm(delta_k)^2 + 1e-12)*delta_k`，
  再使用原样本量权重聚合，不重新归一化。

这里的参数空间是 **simple_v Avg 实际聚合的恢复后完整模型** 的 `named_parameters()`，
包含分类头。先恢复低秩 U/V，再减本轮服务器参数；不是只投影 U/V，也不是减去
低秩下发后的客户端参数。因此 delta 也包含基线低秩截断带来的变化。
BatchNorm 等 buffer 和客户端独立的 CLIP aligner 不进入聚合或内积，遵循原实现。

内积逐 tensor 以 float64 归约；仅使用一个全模型投影系数。利用线性关系
`G_proj = G_avg - sum(p_j * coefficient_j) * delta_k` 流式累积，
不用把 20 份恢复模型同时放入显存。输入客户端参数不会被修改。
零目标更新不投影；零向量的 cosine 日志约定为 0。

第一版要求全参与、无掉线、固定目标，不支持 resume。没有新增投影强度、阈值等参数。

## 数据与运行

使用原仓库可运行 FedCLIP 的 Python 环境（包括 PyTorch、torchvision 和 OpenAI CLIP）。
数据只需生成一次，三个实验复用同一组切分：

```bash
cd dataset
python generate_Cifar100.py --niid 1 --partition pat --cpc 20
cd ../system
python run_target_proj.py --rounds 100 --device-id 0
```

可先用 `--rounds 50` 看趋势，或 `--dry-run` 仅显示完整命令。
启动器按 `avg → target_only → projection` 顺序运行，各使用独立 Python 进程和输出目录。
固定 Client 0、seed 0、20 clients、pat_20、full participation，复用
`simple_v/system/run_now.sh` 的 ResNet 配置：`Decom_resnet18_5`、5 local epochs、
batch 16、lr 0.005、regularization 1e-3、CLIP MSE 1、U/V LR 比例 0.3/1.0。

单独运行某个模式（在 `system/`）：

```bash
python main.py -algo FedTargetProj --target_proj_mode projection --target_client_id 0 --seed 0 -t 1 -data Cifar100 -ncl 100 -nc 20 -niid 1 -pt pat -cpc 20 -jr 1.0 -m Decom_resnet18_5 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -mse_lamda 1 -regular_lamda 1e-3 --use_asymmetric_lr 1 --u_lr_ratio 0.3 --v_lr_ratio 1.0 -did 0
```

`simple_v` 的循环是 `range(global_rounds + 1)`：`-gr 100` 实际执行 101 次本地训练和聚合。
本实验保留这个行为；如要恰好 100 次聚合，用 `-gr 99` / 启动器 `--rounds 99`。
新增日志的 `round` 从 1 开始，表示已完成的聚合次数，`loop_round` 对应原循环编号。

模型初始化继续使用基线构造器的固定 seed 0；`--seed` 固定 Python、NumPy、PyTorch
训练随机流，构造前后各设置一次。三个独立进程使用相同 seed 和配置；没有改动基线
CUDA 确定性设置，因此不承诺不同 GPU/软件版本间逐位复现。

## 指标及解释

每次聚合写入工作目录（启动时打印 `metrics_dir`）：

- `target_proj_metrics.csv` / `target_proj_metrics.json`：目标准确率及轮级诊断。
- `target_proj_clients.csv`：逐客户端权重、是否冲突、投影前后内积、更新范数和删除范数。
- 原 H5 文件新增 `target_projection` 组；最终模型导出目录同时复制三份诊断文件。

`target_client_test_acc` 测量 **本轮聚合模型按正常低秩下发后、尚未进行下一轮本地训练时**
在目标客户端测试集的准确率（0–1）。使用原客户端测试函数、原下发流程和本地 buffers；
测试后恢复本地 checkpoint 及 RNG，避免干扰后续训练。按 `eval_gap` 记录并额外记录最终轮。
未评估轮在 CSV 留空、JSON 为 null、H5 为 NaN。原 `rs_test_acc` 仍是继承的本地模型指标，
不能把它误当作这里的目标准确率。

所有模式每轮都计算同一套 **假设执行 projection 时** 的诊断：
`conflict_client_ratio`、`removed_update_ratio`、`avg_target_cos`、`proj_target_cos`。
因此 avg / target_only 的 removed_update_ratio 也可能非零，它表示本轮投影会删除的比例，
不是这两个模式实际执行了投影。删除比例为其他客户端删除范数之和除以原范数之和，
不按样本量加权。附加 target/avg/proj 范数和冲突数量帮助解释零更新等退化情况。
`dot_after_projection` 用投影公式解析计算；含 epsilon 和浮点误差时不强制等于零。

比较相同完成轮数下三个模式的目标准确率，重点看 projection 是否同时超过 avg 和
target_only，并结合冲突率、删除比例和两个 cosine。算法保证删除负向平行分量；
不保证 cosine 必然提高，也不保证准确率必然提升。

## 本地验证

```bash
python -m unittest discover -s tests -p "test_target_projection.py" -v
python system/run_target_proj.py --dry-run
```

测试涵盖三模式数学结果、原 Avg 数值对齐、整模型投影、零/微小目标更新、样本权重、
不修改上传和 buffer、目标评估恢复 checkpoint/RNG，以及继承训练循环的合成模型冒烟测试。
合成测试不需要下载 CLIP；真实 CIFAR-100 收敛表现需完成上述三组实验后判断。

本次实现验证：`htfllib` 环境下新增测试 12 项、聚合诊断回归 12 项、存储隔离回归 4 项、
客户端初始化回归 3 项全部通过；语法检查和启动器 dry-run 通过。
该环境当前缺少 `clip`，本地也尚无 `Cifar100/pat_20` 切分，因此未执行真实数据训练。
