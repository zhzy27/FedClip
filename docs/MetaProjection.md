# C0 适应后验证 CE 的完整展开诊断

本批提供四个入口：带未见 C0 holdout 的 Projection 快照采集、共享固定 logits 离线优化、
逐快照独立 logits 离线优化，以及冻结 alpha 的 `meta_projection_fixed` 联邦运行。
这是可以读取 C0 训练/验证数据的离线诊断，不声称满足没有额外信息反馈的真实联邦部署协议。
默认没有 EMA、Top-K、自身质量限制、APA 距离代理、DWA guidance、FOMAML、停止 SVD 梯度或 straight-through。

## 数据与基线先后顺序

`meta_projection.py split` 仅读取 C0 原训练 shard，按类别分层留出默认 20%，划分 seed=1729 独立于 training seed。
保存 train/validation 样本索引、逐类数量、预处理后的张量数据 SHA-256 和整个 manifest 的 split_id。
同一路径已有不同划分时拒绝覆盖；不修改原始 npz 文件。
类别至少有两个样本，确保每类 train/validation 非空。训练 seed 0/1/2 使用同一份索引文件。

`--meta_c0_split` 在所有正常训练之前绑定 C0 的训练子集。
正常 `clientTargetProj.train()`、损失、SGD、梯度裁剪和低秩下载未改变；新增 loader 在显式启用协议时只给 C0 返回训练子集。
普通 sample-count 聚合使用实际 train_samples，因此 C0 留出验证后，Projection/ProjectionSoftmax 的样本量质量也相应变化。
所有新比较必须启用同一划分；以前 full-training 的准确率只能作历史参考。
新协议不允许未知验证暴露的预训练本地 checkpoint，沿用 fresh-run、不 resume 的实验约束。

为满足原测试集仅用于最终评价：启用此协议后训练期间不做测试性能评价或测试早停。
保留每轮普通 C0 的低秩 checkpoint，整个联邦运行结束后统一评价这些已保存的 5-epoch endpoint，
回填 final/last10/best/最早最佳轮次。最后的聚合模型另作一次下载后测试，全客户端均值也仅在最终报告阶段测量。
基础框架初始化仍读取 test shard 的样本数量；不会把测试标签、损失或准确率传给离线权重学习。
离线优化只读取 `train/0.npz` 中的固定 train/validation 两个子集；不需要 test shard。

## 快照

仅在 `projection` 上显式开启 `--meta_collect_snapshots`。
默认 `--meta_snapshot_rounds 20,50,80,100` 是显示编号 R20/R50/R80/R100，分别为 loop 19/49/79/99，
并非最终 R101。收集位置在本轮所有正常 local train 完成、receive IDs 后，聚合之前。
短 smoke 需显式指定范围内编号，例如 `--rounds 1 --meta_snapshot_rounds 1,2`。

逐客户端恢复与保存，只保存请求的轮次，采集过程恢复 Python、NumPy、Torch CPU/CUDA RNG，
不会覆盖普通 checkpoint 或改变参数/训练轨迹。每个 `Rxx/` 包含：

- `global.pt`：聚合前 recovered full-W 参数。
- `client_0.pt` … `client_19.pt`：本轮普通 post-local recovered full-W 参数，含 head，不含 buffers。
- `c0_template.pt`：普通 C0 的低秩结构、参数布局、buffers、模块状态。
- `capture_rng.pt`：采集时随机状态，便于审计/复现。
- `metadata.json`：轮次、seed、IDs、原聚合读取顺序、各客户端容量/布局、训练配置、数据划分、backend、版本及文件哈希。
- 原全模型 global-delta Projection 的系数在服务器当前设备上计算后保存，快照优化将其视为常量。

根 `manifest.json` 汇总所有快照与 metadata 指纹。重复同一 R 文件夹或混合不同配置/划分会报错。
快照存的是不可微常量；既不反传 helpers 的训练，也不展开整个联邦历史。

## 完整可微 C0 适应

二十个 float64 logits 的 `softmax(z)` 同时学习 C0 和 helpers 质量，默认 z=0。
候选 full-W 模型为 `Wg + alpha0*delta0 + sum_j alpha_j*(delta_j - coeff_j*delta0)`，
其中普通 C0 不投影，helper coeff 来自原快照。聚合包含分类头。

`utils/meta_virtual.py` 使用原空间卷积展平和原 rank 规则：
卷积 `[out,in,K,K] → [out*K,in*K]`，`rank=max(1,round(rank_rate*max_rank))`，
两侧因子均按 `sqrt(S)` 分配，保持 `U[:, :r] @ diag(sqrt(S))` 和 `diag(sqrt(S)) @ Vh[:r]`。
直接调用 native `torch.linalg.svd`，参数保持计算图，不用普通 `.data`/copy/no_grad/新建 Parameter 切断连接。

使用 `torch.func.functional_call` 执行原模型 forward 与其当前虚拟参数的 Frobenius 正则。
每步通过 `autograd.grad(..., create_graph=True)`，再作函数式 SGD，lr=.005、momentum=0、
正则 .001、范数上限 10；裁剪公式含原 `1e-6`，保留 norm/scale 的参数依赖。
完整消费训练子集的 5 epochs，batch=16、shuffle=True、drop_last=False。
适应后 eval-mode 验证 **CE** 按样本数平均，不加 Frobenius。
模板模型/原 checkpoint/buffers/计数不改动，所有随机状态在虚拟操作前后恢复。

训练批次计划使用独立 virtual seed=777，并按显示 round 派生，在相同快照所有初始化、外层步骤和两种策略中一致。
这固定了比较中的随机批次噪声，不假装知道未来联邦运行的实际批次顺序。
captured RNG 另行保存，默认虚拟训练使用显式 virtual seed。backend 与 source snapshot 对齐，操作后恢复。

默认 `--dtype float32` 对齐普通模型；float64 仅作为显式数值诊断选项，日志标明精度。
当前完整路径针对本任务的无 BatchNorm `Hyper_CNN_512` 审计。
mutable BatchNorm running-stat 的完整导数尚未支持，会明确拒绝，而不是悄悄把这些状态当常量。

## 精确重算和数值失败

默认 `--checkpoint-steps 0` 保留完整普通图；服务器命令示例使用 `8` 降低状态显存。
分段重算仍保留原输入，不在段间 detach。反向重算该段全部 SGD 并计算完整 VJP，包含内部训练 Hessian，
再传回前一段和 SVD。每段重放随机状态并保护模块 train/eval 状态。
PyTorch 2.0.1 的现成 non-reentrant checkpoint 在嵌套 autograd.grad 时失败，本实现使用独立精确段算子。
已与普通展开比较损失/完整梯度，包含 Dropout；不是截断或一阶近似。
非空 buffers 的分段重放目前会拒绝；本任务 CNN 无此限制。

native SVD 在重复/近零奇异值处存在反向不稳定性，参见
[PyTorch SVD 文档](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.svd.html)。
日志保存各分解层 rank、矩阵形状、最小/最大/最小保留 singular value、相对 singular gap 和截断边界 gap。
不会加 jitter、detach SVD 或自动改精度/目标；非有限 hypergradient/资源错误会保存失败位置与信息，
未完成预算的轨迹不会导出可正式使用的固定产物。
人为零 singular value 测试会失败并记录，说明该数值路径的问题，不是固定权重不存在的证明。

## 两种离线优化

`--strategy fixed` 的一个 z 共用所有快照，以各快照适应后验证 CE 的平均值优化。
逐快照构图、回传 `loss / snapshot_count`、释放图；全部快照完成后才作一次 Adam 更新。
`--strategy per_snapshot` 为每个快照初始化独立 z，分别优化；结果仅作灵活诊断参考，不是已证明的全局最优上界。

正式诊断初始预算是 outer Adam lr=.05、50 次更新，两策略相同预算和初始化规则。
默认同时跑 `uniform` 和 `biased`；轻度偏置默认 C0 logit +.25，可用 `--bias-client/--bias-logit` 配置。
每个初始化记录更新前实际权重、更新后权重、weight-change norm、各快照/平均验证 CE、
z-gradient norm、C0/helper 质量、全部/条件 helper 有效数量、时间、CUDA peak allocated bytes、非有限/失败信息。
weights 与验证 CE 对应更新前候选；最后另评估更新后的第 50 个候选，`update_applied=0`，无虚构梯度。

按最小适应后验证 CE（shared 时是平均值）选择 iteration，并在相同预算的初始化之间选择。
从不以测试结果选择初始化、更新次数、lr 或 alpha。
输出 `<group>_<initialization>_trajectory.json`、`fixed_weights.json`，或各 `Rxx_weights.json` 加 `per_snapshot_weights.json`。
产物包含 IDs、alpha、配置、容量、数据索引/指纹、快照来源和 SHA、初始化/优化预算及选择规则。
逐快照 JSON 的 kind 不允许进入冻结正式模式。

## 冻结正式验证

`meta_projection_fixed` 在构造时检查 IDs、模型/训练配置、容量/参数布局、split_id 和 validation-only 选择声明，
然后只加载一次 alpha。每次从共享服务器按容量下发、普通训练 5 epochs、重新计算原 Projection 后用该 alpha 聚合。
不调用 guidance、不进行虚拟训练或 hypergradient，不用验证/测试表现更新权重。
检查不限定 training seed，允许用相同 alpha 与划分运行 seed 0/1/2。

两类结论分开：快照实验只检验共同 alpha 在观察到的状态上是否有效；
冻结运行检验其改变后续客户端轨迹后是否仍有效。
少量更新失败不能否定固定权重；完整 50 次预算失败也不是所有固定权重无效的证明。
不能把复制合成状态的 smoke 当作跨训练阶段的效果实验。

## 服务器完整命令

在 `system/` 使用 `meta_work/`。先完成划分，再采集基线（不会由本次开发自动启动）：

```bash
python meta_projection.py split --dataset-root ../dataset --fraction 0.2 --split-seed 1729 --output meta_work/c0_split.json
python run_target_proj.py --modes projection --rounds 100 --device-id 0 --seeds 0 --meta_c0_split meta_work/c0_split.json --meta_collect_snapshots --meta_snapshot_rounds 20,50,80,100 --meta_snapshot_dir meta_work/projection_seed0_snapshots
```

先测实际 C0 数据完整 5 epochs 开销，再决定是否执行 50 次诊断预算：

```bash
python meta_projection.py benchmark --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset --device cuda:0 --checkpoint-steps 8 --output meta_work/real_data_benchmark.json
python meta_projection.py optimize --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset --strategy fixed --outer-lr 0.05 --updates 50 --initializations uniform biased --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 --checkpoint-steps 8 --device cuda:0 --output meta_work/fixed
python meta_projection.py optimize --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset --strategy per_snapshot --outer-lr 0.05 --updates 50 --initializations uniform biased --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 --checkpoint-steps 8 --device cuda:0 --output meta_work/per_snapshot
```

较短计时优化可将 `--updates 50` 改成 `1` 并改输出目录；虚拟适应仍为完整 5 epochs。
输出目录需为新目录，避免覆盖既有轨迹。
评价冻结产物、相同划分下的 ProjectionSoftmax（Projection 对照已经由采集运行获得）：

```bash
python run_target_proj.py --modes meta_projection_fixed --rounds 100 --device-id 0 --seeds 0 --meta_c0_split meta_work/c0_split.json --meta_weight_file meta_work/fixed/fixed_weights.json
python run_target_proj.py --modes projection_softmax --rounds 100 --device-id 0 --seeds 0 --meta_c0_split meta_work/c0_split.json
# seed 0 完成后，再进行多 seed；不改变划分/alpha：
python run_target_proj.py --modes meta_projection_fixed --rounds 100 --device-id 0 --seeds 0 1 2 --meta_c0_split meta_work/c0_split.json --meta_weight_file meta_work/fixed/fixed_weights.json
```

相同命令还封装为 `system/meta_projection_commands.sh` 中的具名函数。
`source` 只定义函数，无自动执行；按 `prepare_meta_split → collect_meta_baseline → time_meta_real_data → fit_meta_fixed/fit_meta_per_snapshot → run_meta_fixed_seed0/control` 调用。
GPU ID/设备通过 META_GPU_ID/META_DEVICE 设置。

联邦日志仍在 `target_proj_runs/<timestamp>/<mode>/`（多 seed 为 `seedN/<mode>/`），metadata/metrics 具体目录启动时打印。
归档 `meta_post_local/R*/Client_0_model.pt` 用于最后评价，快照单独位于指定目录。
4 个快照约 80 个 full-W 模型文件，加上 101 个 C0 低秩 endpoint；磁盘和采集时间是额外诊断成本。
最终导出复制 split manifest、公共指标和固定 alpha JSON。

## 本地已测结果与边界

本地没有真实 Cifar100/pat_20，未运行真实 101 次联邦采集/正式冻结对照。
已完成实际 Hyper_CNN_512/C0 rank=.9/100 类的合成完整 5-epoch 超梯度：
40 个原样本分成 train32/val8，batch16，共 10 个 SGD 步，两种实现各预热一次。
设备 RTX 3070 Ti Laptop GPU，PyTorch 2.0.1+cu117：

| 测量 | 完整普通图 | 精确分段（2 步） |
|---|---:|---:|
| 普通分解秒 | .142 | .139 |
| 普通 SGD 秒，不含分解 | .066 | .063 |
| 可微前向秒，含分解/训练/验证 | .201 | .189 |
| outer backward 秒 | .126 | .189 |
| peak allocated VRAM | 839 MiB | 513 MiB |
| 普通/函数式 SGD 最大参数误差 | 1.49e-8 | 2.24e-8 |
| z-gradient norm | .007866 | .007866 |

短 1-epoch 展开的随机方向 float64 中心有限差分（epsilon=1e-4）相对误差约 1.1e-7。
计时记录在 `meta_projection_smoke_benchmark.json` / `meta_projection_smoke_checkpoint_benchmark.json`。
这些是小合成训练集的一次预热后测量，不按固定倍数外推到实际 2000 个 C0 训练样本的 625 步展开。

`meta_projection_smoke_optimization/` 记录实际 CNN 两种 CLI 策略、两种初始化、1 次 Adam 更新的烟测。
四个标签 R20/R50/R80/R100 是同一合成状态的副本，仅更换虚拟批次种子，验证接口和逐图处理；
不是实际不同训练阶段，也不能用于真实冻结实验（指纹与容量不匹配）。
小网络另验证 SGD/正则/触发裁剪、完整 hypergradient/中心有限差分、Dropout 重算、RNG/模板不变、
固定/逐快照预算、产物兼容检查、零 singular value 的显式失败；
真实异构 CNN 联邦 smoke 验证 holdout-before-snapshot、采集 RNG/checkpoint 保护、结束后测试和冻结 alpha 的共享下发。

本批新增 14 项测试（11 项数据/数学/优化/失败检查、1 项真实 CNN 联邦集成、2 项 launcher），
全仓库 25 个测试模块独立运行，共 257 项通过。旧全部模式测试继续通过，旧聚合核未改动。
CLI/语法/diff/dry-run 检查通过；原 staged 绘图改动保留，不纳入本批提交。
旧绘图测试沿用本机临时 MKL 环境设置。当前没有真实数据或正式 50-update/101-round 效果结论。
