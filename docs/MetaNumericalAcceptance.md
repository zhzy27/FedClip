# 真实快照的数值验收

当前不能进入 50 次正式优化。服务器报告的普通/函数式 CE 差 .00315054
和有限差分反号尚未在本地真实数据上定位或修复；本机没有 R20/R50/R80/R100
或真实 CIFAR-100/pat_20。原快照与基线可以直接复用，不需要重新采集。
不能把下面的小网络或合成 CNN 检查当作真实数据验收。

## 已确认的入口问题

1. benchmark 原先没有传快照的 aggregation_order，优化入口有。
   现在统一通过 snapshot_candidate 构造目标，包含原顺序。
   两种前向对照本来就共享候选，因此这不能单独解释服务器的 SGD 偏差。
2. 普通分解新建低秩参数采用全局默认 dtype：默认 float32 下，double SVD
   因子先复制到 float32，再 model.double()，会在尚未训练时丢失精度。
   小网络测试精确复现这一路径。修复只在诊断参考初始化期间临时匹配 default dtype，
   finally 恢复默认值；正常客户端分解函数没有改动。float32 路径的因子前向不变。
   该问题不能解释仅经过函数式路径的 double 有限差分反号。
3. 原 benchmark 只以 finite/nonzero 判断成功。现在还运行完整数值验收，
   failed 或 unconfirmed 均非零退出；旧版成功 JSON 不能作为新优化许可。

尚未确认 TF32、SVD 截断边界、长展开非光滑性、梯度布局或重算是服务器主问题。
不因有限差分反号直接断言解析梯度实现错误。独立诊断分别检查这些路径。

本地持续裁剪合成 GPU 对照已经观察到一次前向分歧：分解后完全一致；
第1步参数差 7.45e-9，第6步原始梯度差放大到约 .104，第8步验证 CE 差 .00736。
同一点 backward 与 create_graph 梯度差仅约 1e-6；两套参数/梯度布局一致。
该例10步均触发裁剪。仅关闭 TF32 的对照仍失败；仅启用 cuDNN deterministic
以及 combined 对照的 CE 前向误差为0。这些设置是分开记录的显式诊断选项；
未据此更改默认优化 backend，也没有认定同一个原因解释服务器真实数据。
后续启用额外 branch hooks 时偏差幅度改变，未发现分支选择改变；
非光滑分支放大仍是待核实假设，不能宣称已经证实。

重复 FD 还暴露本机 Torch2.0 的资源滞留：实际 CNN double 的 checkpoint8
纯前向评价 RSS 每次增加约520MiB，普通图不增长。有限差分现在消费并丢弃
同一完整图的反向结果，测得 checkpoint8 的 RSS 约935MiB并保持稳定。
计算的是同一 J，既不更新权重也不修改 SGD；额外反向计入诊断耗时。
这是多次扫描的资源处理，不是服务器解析/数值导数反号的已确认根因。

## 诊断与判定

diagnose 使用和 optimize 相同的候选构造、快照顺序、批次计划与随机种子。
默认候选包括均匀 z 与 C0 logit +.25，两者都要验收。
训练仍是 batch16、SGD .005、CE + Frobenius .001、clip10、完整 5 epochs。
阶段为分解后、1 步、8 步、1 epoch、5 epochs；1800 样本时是 0/1/8/113/565。
普通训练使用 backward/clip_grad_norm_/SGD.step，函数式对照使用
autograd.grad(create_graph=True) 和保留裁剪参数依赖的更新。

每步记录原始梯度差、两种总 norm/scale、裁剪后梯度差及更新后参数差。
指定阶段额外在普通训练的同一点对比 backward、autograd.grad(False) 和
autograd.grad(True)，区分单步算子差异与此前轨迹积累。
阶段记录参数/重构 full-W/验证集输出/CE 的差异及参数和梯度的布局。
SVD 记录谱间隔、相对间隔、截断附近奇异值、矩阵 dtype/device/stride。
因子符号或基底不同不单独导致失败，重构与输出是前向验收依据。

为释放资源，逐步的前向诊断不保留跨步图；这是明确标记的诊断工具。
独立的 full_five_epoch 仍通过 native SVD、全部 SGD 更新和裁剪计算完整 z 梯度，
并核对其最终参数和 CE 与逐步函数式轨迹一致。
优化路径没有增加 detach、截断、一阶近似或更换 SVD 梯度。

短展开比较 checkpoint-steps=0、8 和配置值的 CE、所有参数、完整 z 梯度。
0 表示不重算。默认对 1 epoch 扫描有限差分，另检查训练前、1 步及8 步；
后者使用计划的 dtype。float32 配置还额外运行同设备的 float64 epoch 检查，
但仅 double 检查通过不能给 float32 配置发通过证书。
每个检查先同点评价 3 次，再检查 3 个固定去除 softmax 零方向的随机方向。
默认 h 为 1e-2/1e-3/1e-4/1e-5/1e-6/1e-7，保存 J+、J- 和每个结果，
不只挑误差最小的 h。每个方向必须存在至少两个相邻、误差合格且噪声可分辨的 h。
没有收敛区间记为 unconfirmed；重算不一致或前向阈值失败记为 failed。

仅排查前向可以显式 --forward-only，仍会计算完整5epoch超梯度，
但报告 scope 为 incomplete_exploratory，永远不能用于正式优化。
可选 --trace-branches 记录 ReLU 门与 max-pool winner 差异；
额外 hooks 可能影响非确定 backend 的执行，默认关闭，不能据此替代无 hooks 的验收。

默认阈值均写入 JSON。CE 为 2e-6 + 2e-6*|CE|；
重构为 5e-6 + 2e-5*对应层最大幅度；输出为 2e-5 + 2e-5*对应批次最大幅度；
重算 z 梯度为 2e-7 + 2e-4*梯度最大幅度；
方向导数为 2e-6 + .05*max(|解析|,|有限差分|)，外加
4*重复前向噪声范围/h。噪声项不得超过无噪声阈值的四分之一，否则该 h 不可分辨。
确定性 float32 舍入即使重复噪声为零也可能使小 h 不可靠；没有相邻收敛区间仍不通过。
这些数值阈值不是跨任意新目标的普适保证，不允许用 finite/nonzero 取代误差判定。

## 服务器命令

从仓库 system/ 执行。下面命令只用已有快照及 C0 train shard 的原 holdout，
不会读取 test shard 优化权重，不会修改正常 checkpoint、快照或原数据。
CUDA_VISIBLE_DEVICES=0 只固定可见 GPU，不当作本次数值问题的修复。

~~~bash
export CUDA_VISIBLE_DEVICES=0
python meta_projection.py diagnose \
  --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset \
  --rounds 20 --dtype float32 --checkpoint-steps 8 --virtual-seed 777 \
  --initializations uniform biased --bias-client 0 --bias-logit 0.25 \
  --backend-policy snapshot --fd-epochs 1 --fd-directions 3 --fd-repeats 3 \
  --fd-stage-steps 0 1 8 --fd-steps 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 \
  --device cuda:0 --cpu-threads 2 --output meta_work/acceptance_R20.json
~~~

若需隔离 backend：用下面单独的报告。override 在快照 backend 恢复之后应用，
保存实际生效的四个 backend flag；全局 deterministic_algorithms、cuDNN/Torch/CUDA 版本也记录。
该选项启用 cuDNN deterministic、关闭 benchmark 与 TF32，没有声称所有算子必然确定。
不能把此报告用于仍采用 snapshot policy 的优化。

另支持 no_tf32 与 deterministic_only 两种 policy，用于分别隔离因素；
将下面命令的 policy 和输出名同时替换即可。最终正式配置仍需全部快照验收。

~~~bash
python meta_projection.py diagnose \
  --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset \
  --rounds 20 --dtype float32 --checkpoint-steps 8 --virtual-seed 777 \
  --initializations uniform biased --bias-client 0 --bias-logit 0.25 \
  --backend-policy deterministic_no_tf32 --fd-epochs 1 --fd-directions 3 --fd-repeats 3 \
  --fd-stage-steps 0 1 8 --fd-steps 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 \
  --device cuda:0 --cpu-threads 2 --output meta_work/diagnostic_R20_no_tf32.json
~~~

进一步 double 对照可将 dtype 改成 float64，同时换输出文件名；
不能因此宣称 float32 已修复。若最终决定用另一套 dtype/backend 做正式优化，
必须对该配置的全部快照重新验收，并在优化命令中显式使用相同配置。
完整 5 epoch 的 FD 可以使用 --fd-epochs 5，不能由短展开失败静默降低真实训练目标。

R20 完成定位并修复后，复验四个快照：

~~~bash
python meta_projection.py diagnose \
  --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset \
  --rounds 20 50 80 100 --dtype float32 --checkpoint-steps 8 --virtual-seed 777 \
  --initializations uniform biased --bias-client 0 --bias-logit 0.25 \
  --backend-policy snapshot --fd-epochs 1 --fd-directions 3 --fd-repeats 3 \
  --fd-stage-steps 0 1 8 --fd-steps 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 \
  --device cuda:0 --cpu-threads 2 --output meta_work/acceptance_all.json
~~~

只有 acceptance_all.json 为 passed 且配置匹配，才可手动运行以下命令：

~~~bash
python meta_projection.py optimize \
  --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset \
  --strategy fixed --outer-lr 0.05 --updates 50 --initializations uniform biased \
  --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 \
  --checkpoint-steps 8 --backend-policy snapshot --device cuda:0 --cpu-threads 2 \
  --acceptance-report meta_work/acceptance_all.json --output meta_work/fixed_accepted
python meta_projection.py optimize \
  --snapshots meta_work/projection_seed0_snapshots --dataset-root ../dataset \
  --strategy per_snapshot --outer-lr 0.05 --updates 50 --initializations uniform biased \
  --bias-client 0 --bias-logit 0.25 --virtual-seed 777 --dtype float32 \
  --checkpoint-steps 8 --backend-policy snapshot --device cuda:0 --cpu-threads 2 \
  --acceptance-report meta_work/acceptance_all.json --output meta_work/per_snapshot_accepted
~~~

报告绑定所有快照 metadata/file SHA、数据划分、IDs/容量、代码 SHA、
模型 dtype、virtual seed、checkpoint 配置、有效 backend、Torch/CUDA/cuDNN、
GPU 型号、CPU threads、初始化。只测 R20 的报告不能授权四快照优化。
改代码、换 dtype/backend/环境后需重新验收，先前报告不会被静默复用。
验收针对被测初始候选的数值路径，不保证每个后续 Adam 迭代都数值稳定或权重效果理想。

未验收时若只需开发烟测，可以显式 --diagnostic-only；只写
unaccepted_*_weights.json，kind 为 meta_projection_unaccepted_diagnostic，
冻结联邦模式拒绝这类产物，不生成 fixed_weights.json。
旧固定权重产物不被自动删除；本次没有修改冻结正式聚合路径或 Projection 自身占比消融。

## 查看报告

snapshots[].candidates[].trajectory.first_failed_stage 是最早被测阶段的前向失败；
first_gradient_difference_step 为每步原始梯度首次超阈值，不是自动判定根因。
结合 gradient_steps[].same_point_gradients 判断是否同一点的 backward/create_graph 已不同。
full_five_epoch 保存完整前向、反向时间、峰值显存、批次数、超梯度及与诊断轨迹的差异。
ordinary_timing 是没有逐步梯度探针的普通5epoch SGD计时，ordinary_comparison
是这个普通结果与完整函数式结果的直接对照；仪器化轨迹耗时另作命名，避免混用。
short_planned_dtype 与 short_float64_diagnostic 保存重算比较、噪声和全部原始 FD 损失。
stage_derivative_probes 保存训练前及早期步骤的解析/数值检查。
若某个操作异常，报告保留此前证据、错误位置所在快照及 approximation_used=false，
并以 failed 非零退出，不退回近似或零阶优化。

本地测试：test_meta_projection.py、test_meta_diagnostics.py，另外复验 meta runtime
及 ProjectionSelfWeight 相关旧测试。实际 CNN 的 CPU 持续裁剪对照与 GPU 合成五 epoch
仅用于结构/机制检查。真实 1800/450、565 步的结论等待以上服务器报告。

本次共93项相关测试通过（含11项新增诊断测试）。合成 train32/val8、
实际 Hyper_CNN_512、C0 rank=.9 的完整5epoch为10步，不能外推为565步通过：

|本地检查|结果|
|---|---|
|CPU double 完整5epoch超梯度 norm|.00786094，有限非零|
|CPU double checkpoint0/8 epoch梯度最大差|0|
|CPU double 0/1/8步及1epoch方向检查|2方向、h=1e-2..1e-7均存在相邻收敛区间|
|GPU float32、输入放大200、deterministic_only|10/10步裁剪；普通/完整函数式CE与重构差均0|
|上述GPU完整可微前向/反向|.480/.227秒，peak allocated约769MiB|

CPU报告为 synthetic_structure_only；GPU仅前向报告为 incomplete_exploratory，
两者都不能授权真实数据正式优化。普通输入的float32 FD未找到可信相邻收敛区间，
仍为 unconfirmed。上述GPU设置是显式测量用配置，没有静默改默认 snapshot policy。
原始本地报告位于 system/meta_work/numerical_acceptance_*，合成快照未纳入Git。
