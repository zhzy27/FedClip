# FedTargetProj 第一阶段实验

分支 `target_proj` 基于 `simple_v` 的 `a46b5e4`。原有 `serverCLIP.py`、
`clientCLIP.py`、`serverbase.py`、模型和本地优化器均未修改。

## 实现与对照

`clientTargetProj` 直接继承通用 `Client`，`FedTargetProj` 直接继承通用 `Server`。
本地训练、低秩下发和服务器训练循环独立实现，不继承或导入 `FedCLIP/clientCLIP`。
仅复用仓库低秩模型、数据接口、基础指标/存储与样本量权重。

本地目标为 `CrossEntropy(model(x), y) + regular_lamda * model.frobenius_decay()`。
U、V、分类头和其他可训练参数使用同一个 SGD 学习率；保留原梯度范数上限 10。
没有 CLIP 编码器、文本特征、对齐模块/损失、非对称学习率或 U 特殊梯度缩放。
本算法默认启用低秩正则，`-is_regular 0` 可显式关闭用于消融；非对称学习率和
U 特殊缩放开关若被误开启会报错。日志 `ce_loss` 和 `regularization_loss` 分别记录
分类损失与已乘系数的低秩正则；训练集 loss 指标记录两者之和。

五种模式全部客户端仍然参与同样的本地训练，服务器写回参数的规则不同：

- `avg`：原始样本量加权平均；保留基线逐参数加权求和的浮点计算顺序。
- `target_only`：服务器参数直接取目标客户端恢复后的参数，目标权重为 1。
- `projection`：以目标 delta 为参考，仅对其他客户端中全模型内积小于 0 的
  delta 应用 `delta_j - dot(delta_j, delta_k)/(norm(delta_k)^2 + 1e-12)*delta_k`，
  再使用原样本量权重聚合，不重新归一化。
- `layer_mask`：新增模式，使用纯本地变化量逐层判方向，以目标 post-local 模型为 anchor，
  加上通过 hard mask 的辅助客户端 local updates，详见下一节。
- `layer_mask_budget`：在相同 hard mask 之后，对每层全部 helper 的加权和施加固定 beta=1 的
  target-local norm budget；不缩放 target anchor，不改变旧四个模式。

这里的参数空间是 **simple_v Avg 实际聚合的恢复后完整模型** 的 `named_parameters()`，
包含分类头。**旧 projection** 先恢复低秩 U/V，再减本轮服务器参数；不是只投影 U/V，也不是减去
低秩下发后的客户端参数。因此 delta 也包含基线低秩截断带来的变化。
BatchNorm 等 buffer 不进入聚合或内积，遵循原有低秩聚合约定。

内积逐 tensor 以 float64 归约；仅使用一个全模型投影系数。利用线性关系
`G_proj = G_avg - sum(p_j * coefficient_j) * delta_k` 流式累积，
不用把 20 份恢复模型同时放入显存。输入客户端参数不会被修改。
零目标更新不投影；零向量的 cosine 日志约定为 0。

第一版要求全参与、无掉线、固定目标，不支持 resume。没有新增投影强度、阈值等参数。

## 新增 layer_mask

旧 `projection` 的数学实现保持不变。新模式每轮在低秩下发完成后、任何本地训练开始前，
逐个加载客户端实际收到的 checkpoint，将其恢复到 full-W 并保存纯参数快照 `W_i_pre`。
快照位于工作目录的 `layer_mask_pre/Client_i.pt`，每轮覆盖，不加入最终模型导出。
恢复过程保护 Python、NumPy、CPU/CUDA RNG，避免额外观测改变训练随机序列。
快照带有服务器当前轮次检查，缺失或过期不会退回用 global model 代替。

本地训练后恢复 `W_i_post`，只使用 `D_i = W_i_post - W_i_pre` 判断方向。
因此 SVD 下发截断差 `W_i_pre - W_global` 不参与新模式的冲突判断。
所有判断只读已有模型参数，不使用目标客户端训练/测试数据，不执行额外训练。
目标准确率仍在聚合完成之后独立评估，不参与 mask 或聚合决策。

按恢复后 `named_parameters()` 的模块前缀分组，同层 weight/bias 共用一个 cosine 和 mask。
`Decom_CNN-5-512` 实际识别为 `conv1, conv2, fc1, fc2, fc3`，其中 `fc3` 是分类头；
共享模块别名由 `named_parameters()` 去重，不重复计算。程序启动时打印分组及参数名。

每层执行：

```text
c_i,l = dot(D_i,l, D_0,l) / (norm(D_i,l) * norm(D_0,l) + 1e-12)
M_i,l = 1 if c_i,l > 0 else 0
W_new,l = W_0_post,l + sum(i != 0, p_i * M_i,l * D_i,l)
p_i = n_i / sum(all clients' n_i)
```

目标层自身 `cosine=1, mask=1`，anchor 系数固定为 1，不乘目标样本权重。
辅助层通过 mask 后保留完整的 local update，未通过则贡献严格为零；不重新归一化剩余权重。
若任一层范数小于 `1e-12`，辅助 cosine 定为 0 并记录 `zero_norm=True`，该层被屏蔽。
目标即使为零范数仍保留 anchor 和自身 mask=1，同时标记 zero_norm。
不引入 softmax、temperature、可学习权重或新的可调阈值。

每轮终端完整打印 20×L cosine/mask/norm/zero_norm 矩阵，逐客户端输出
`old_global_delta_cos`、`full_local_delta_cos` 和 local-update norm，
逐层输出负方向数量、冲突率、被 mask 数量和 masked norm ratio。
**冲突统计只计负 cosine；mask 还会删除零 cosine，因此两者可能不同。**

`masked_update_ratio` 为 `sum(i,l, p_i*(1-M_i,l)*norm(D_i,l)) /
(sum(i,l, p_i*norm(D_i,l))+eps)`，分母包含目标客户端，分子中目标贡献为零。
每层 masked norm ratio 使用相同定义，只在该层求和。

新增文件均按轮写入工作目录，并复制到最终模型目录：

- `layer_mask_metrics.csv`：整体冲突、mask 比例、masked_update_ratio、目标准确率。
- `layer_mask_clients.csv`：每个客户端两种 full-model cosine、local/old update norm、零范数标记。
- `layer_mask_cosines.csv`：每个客户端×层的 cosine、mask、norm、zero_norm 和加权删除范数。
- `layer_mask_matrices.json`：`history` 列表保存每轮完整矩阵、逐层统计、客户端 ID 及层名。
  行按 client ID 排序，列按 `layer_names`；包含 `layer_groups`，可以直接对应实际参数。

新模式同样写通用 `target_proj_metrics.*` 和 H5 `target_projection` 组，
其中诊断字段为新 mask 统计；旧三种模式的 projection 诊断定义保持不变。

## 新增 layer_mask_budget：只限制联合 helper

复用 `layer_mask` 的 W_pre 快照、纯本地 delta、逻辑分组、cosine 和 hard mask，
本地训练、通信量及样本权重均不变。不增加 EMA、softmax、temperature、prefix mask 或可学习权重。
服务器先按原样本权重汇总全部通过 mask 的辅助更新，再对整层的联合向量做一次预算限制：

```text
H_l = sum(i != 0, p_i * M_i,l * D_i,l)
T_l = norm(D_0,l)
G_l = norm(H_l)
beta = 1.0, eps = 1e-12
s_l = 1                         if G_l <= T_l
s_l = T_l / (G_l + eps)          if G_l > T_l
W_new,l = W_0_post,l + s_l * H_l
```

每层 weight/bias 共用一个 scale。预算不逐客户端裁剪，不重新归一化权重，不放大 helper；
Client 0 的完整 `W_0_post` 始终系数为 1。helper 的原方向保留，范数限制在 target 本层
local-update 范数以内（浮点舍入误差除外）。`G=2*T` 时 scale 约为 0.5，epsilon 会带来微小差异。

退化情况：helper 为零时 scale=1；target norm < eps 且 helper 非零时 scale=0。
若两个范数同时很小但 helper 非零，优先遵守 target 零预算规则；其他 G < eps 情况取 scale=1。
实际 hard mask 已经会屏蔽零范数 target 层的所有辅助更新，预算函数仍单独处理这些边界。

联合 helper 独立累积，不能从最终权重减 anchor 反推，以免大权重吞掉小更新。
未触发预算的层保留旧 `layer_mask` 的浮点累加顺序，确保与原结果逐位一致。
只有触发预算的层才重新计算 `W_0_post + clipped_helper`。

原有全部 LayerMask 终端诊断照常打印，每轮另打印 `[LayerBudget]` 表：
`target_local_norm`、`raw_helper_norm`、`helper_target_ratio`、`budget_scale`、
`clipped_helper_norm`、`budget_active`。整体打印生效层数、最大/平均 helper-target ratio，
以及整个模型的 raw/clipped helper 范数。ratio 使用 `G/(T+eps)`。

该模式使用独立文件名，不写入或覆盖旧 `layer_mask_*` 结果：

- `layer_mask_budget_metrics.csv`：目标准确率、原 mask 指标及整体预算统计。
- `layer_mask_budget_layers.csv`：每轮每层的六项预算指标及目标准确率。
- `layer_mask_budget_matrices.json`：保留所有原矩阵，新增 `budget_beta`、`budget_layers`、
  `budget_summary`；每个 `history` 条目包含本轮目标准确率。
- `layer_mask_budget_clients.csv`、`layer_mask_budget_cosines.csv`：保留逐客户端/逐层原始诊断。

这些文件同步复制到最终模型目录。通用 `target_proj_metrics.*` 和 H5 仍记录本轮结果，
H5 `target_projection` 属性额外记录 `budget_beta=1.0`，没有 beta 调参开关。

在 `system/` 单独启动新模式：

```bash
python run_target_proj.py --modes layer_mask_budget --rounds 50
```

或并排运行两个 mask 实验：

```bash
python run_target_proj.py --modes layer_mask layer_mask_budget --rounds 50
```

直接调用 `main.py` 时使用 `--target_proj_mode layer_mask_budget`，其余参数与 layer_mask 相同。
启动器默认仍运行原四个模式；只有显式选择 budget 才运行新实验。

## 数据与运行

使用具备仓库基础依赖（PyTorch、torchvision 等）的 Python 环境，无需安装或下载 CLIP。
`main.py` 按所选算法延迟加载 server，避免其他算法引入 CLIP 依赖。
数据只需生成一次，四个实验复用同一组切分：

```bash
cd dataset
python generate_Cifar100.py --niid 1 --partition pat --cpc 20
cd ../system
python run_target_proj.py --rounds 100 --device-id 0
```

可先用 `--rounds 50` 看趋势，或 `--dry-run` 仅显示完整命令。
启动器按 `avg → target_only → projection → layer_mask` 顺序运行，各使用独立进程和输出目录。
本轮默认模型改为 `Decom_CNN-5-512`，四组统一使用同一异构 CNN；保留 5 local epochs、
batch 16、lr 0.005、regularization 1e-3、Client 0、seed 0、20 clients、pat_20、全参与。
只运行新模式可用 `python run_target_proj.py --modes layer_mask --rounds 50`；
只重跑旧三组可用 `--modes avg target_only projection`。
如需复现之前的 ResNet 配置，再显式加 `--model-family Decom_resnet18_5`。

单独运行某个模式（在 `system/`）：

```bash
python main.py -algo FedTargetProj --target_proj_mode layer_mask --target_client_id 0 --seed 0 -t 1 -data Cifar100 -ncl 100 -nc 20 -niid 1 -pt pat -cpc 20 -jr 1.0 -m Decom_CNN-5-512 -lr 0.005 -lbs 16 -ls 5 -gr 100 -eg 1 -is_regular 1 -regular_lamda 1e-3 -did 0
```

`simple_v` 的循环是 `range(global_rounds + 1)`：`-gr 100` 实际执行 101 次本地训练和聚合。
本实验保留这个行为；如要恰好 100 次聚合，用 `-gr 99` / 启动器 `--rounds 99`。
新增日志的 `round` 从 1 开始，表示已完成的聚合次数，`loop_round` 对应原循环编号。

模型初始化继续使用基线构造器的固定 seed 0；`--seed` 固定 Python、NumPy、PyTorch
训练随机流，构造前后各设置一次。四个独立进程使用相同 seed 和配置；没有改动基线
CUDA 确定性设置，因此不承诺不同 GPU/软件版本间逐位复现。

## 指标及解释

每次聚合写入工作目录（启动时打印 `metrics_dir`）：

- `target_proj_metrics.csv` / `target_proj_metrics.json`：目标准确率及轮级诊断。
- `target_proj_clients.csv`：逐客户端权重、是否冲突、投影前后内积、更新范数和删除范数。
- 原 H5 文件新增 `target_projection` 组；最终模型导出目录同时复制三份诊断文件。

`target_client_test_acc` 测量 **本轮聚合模型按正常低秩下发后、尚未进行下一轮本地训练时**
在目标客户端测试集的准确率（0–1）。使用分类预测、低秩下发流程和本地 buffers；
测试后恢复本地 checkpoint 及 RNG，避免干扰后续训练。按 `eval_gap` 记录并额外记录最终轮。
未评估轮在 CSV 留空、JSON 为 null、H5 为 NaN。原 `rs_test_acc` 仍是继承的本地模型指标，
不能把它误当作这里的目标准确率。

旧三个模式每轮都计算同一套 **假设执行 projection 时** 的诊断：
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
python -m unittest discover -s tests -p "test_layer_mask.py" -v
python -m unittest discover -s tests -p "test_layer_mask_budget.py" -v
python -m unittest discover -s tests -p "test_layer_mask_cnn_runtime.py" -v
python system/run_target_proj.py --dry-run
```

测试涵盖三模式数学结果、原 Avg 数值对齐、整模型投影、零/微小目标更新、样本权重、
不修改上传和 buffer、目标评估恢复 checkpoint/RNG、真实 CE+正则反向更新、统一学习率，
以及独立训练循环中三种模式的合成模型冒烟测试。
真实 CIFAR-100 收敛表现需完成上述四组实验后判断。

当前专用测试 14 项通过；已验证在没有 `clip` 的 `htfllib` 环境下独立导入 server/client。
聚合诊断、存储隔离和客户端初始化回归共 19 项通过。另用实际低秩 ResNet（缩小通道数）
和合成图像测试两个异构客户端（rank ratio 0.5/0.25）：三个模式各训练两轮，
验证相同初始化、真实本地反向传播、低秩下发、聚合、评估、H5 与模型导出，全程未加载 CLIP。
本地尚无 `Cifar100/pat_20` 切分，因此未执行真实 CIFAR-100 训练。

LayerMask 验证：新增 8 项数值/快照/日志测试、1 项实际 CNN 集成测试通过，旧 14 项测试继续通过。
实际 CNN 测试使用 rank ratio 0.9/0.15 的两个客户端及合成 32×32 图像，运行两轮，
验证恢复后层名/形状对应、W_pre 与实际下发模型一致、本地训练、聚合、评估及 H5/JSON/模型导出。

Helper Budget 验证：8 项新增预算数值/保存测试通过，旧 projection 14 项、旧 layer_mask 8 项
继续通过；实际异构 CNN 的 layer_mask 和 layer_mask_budget 两项两轮集成测试通过，共 32 项。
另与修改前的 layer_mask 做 10 组随机 20-client 对照（float32/float64），参数与诊断逐位一致。
尚未运行真实 CIFAR-100 的 budget 收敛实验。
