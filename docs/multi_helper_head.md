# helper冻结本轮下载群组头

在 `6fe2a52` 分类头角色机制上增加 `--multi_helper_head_update train|frozen`。默认train不冻结、不增加监控前向/反向，原初始化、SVD、训练顺序、采样、原型包及聚合公式保留。

## 唯一新增正式组

```text
--multi_head_mode group_only_reset
--multi_projection_scope backbone
--multi_helper_head_update frozen
```

仅frozen有组合限制，必须是已有SVD `FedMultiGroupProj`、固定分组、无预热、last_epoch/per_model/group_priority、`group_projection_softmax`。拒绝shared/full、group_only_local、KD、Prox、FedRep、反馈、Z2等未支持混用，保持原参数校验。

客户端所属列：实际下载服务器骨干与头，头/骨干一起训练。其他helper列：下载同一列最新服务器头，在原SVD和头复制完成后、创建本轮SGD前，只将别名感知识别的 `fc3.weight/bias` 设为不求梯度，清除残留头梯度。fc1/fc2仍属骨干。优化器只接收当前模型实际可训练Parameter。

没有在forward周围加no_grad，不detach特征，CE仍通过固定线性头与原ReLU传回骨干。每batch保留一次前向、一次CE＋F＋上一轮MSE联合反向、一次原裁剪及一次SGD。每客户端K3各local5，仍15次数据遍历，没有头预热、补训或新损失。

## 临时状态与跨轮语义

冻结状态只作用于当前客户端当前列。真实训练成功后保留参数、buffers与随机流变化，只恢复临时requires_grad标记；异常时也恢复标记、模块模式并清除冻结头梯度。异常的部分训练禁止直接重试，必须重新下载；正常重复训练也受原fresh-download检查限制。

下一轮重新从当前服务器各列复制并SVD，再根据角色设置requires_grad、创建SGD。helper不继承旧本地头，也不把初始随机头冻结跨轮。服务器母模型及其他客户端不被设置冻结。

普通预测checkpoint仍含完整低秩骨干和实际fc3；元数据记录训练期间冻结策略、来源轮次与优化器参数范围。加载工厂恢复普通可训练参数，推理结果不依赖requires_grad；下一轮训练必须重新按角色下载/冻结，不直接续用已训练模型的优化器。此入口不开放断点续训。

## 损失、原型及聚合

继续沿用CE＋原F（.001）＋上一轮原型MSE（1）。原型per_model、group_priority、raw_mean/base_output，最后一个epoch现有前向detach统计，轮末聚合，下一轮固定；首轮只CE＋F但照常生成下一轮库。helper头不上传不影响其各列原型统计上传。

骨干逐列调用原群组Projection核，参考/点积/范数/余弦/评分/投影均不含fc3，质量.50/.50，tau.20。群内头单独按训练样本量归一化平均，总质量1；helper上传字典没有头条目。仍完整母模型下载，不宣称节省下行通信。

## 记录与成本解释

沿用local_training、last_epoch_training、head_aggregation、client_metrics及对应CSV/JSON/H5/manifest。frozen新增记录：

- `helper_head_update/effective_head_update`：请求策略、当前列实际train/frozen；角色、client/model/assigned ID继续保留。
- `head_source_completed_train_count`：下载头来自已完成多少次训练的服务器，首轮0，第二轮1；下载/训练后指纹和范数沿用原字段。
- `head_optimizer_parameter_count`：所属列2、helper列0；`head_gradient_produced`表示任意一步是否存在头grad张量，冻结helper必须false。
- `head_update_norm/head_update_squared_norm/head_unchanged_bitwise`：同次下载到普通本地训练结束的头变化，冻结helper严格逐元素不变。
- `backbone_pre_clip_grad_norm_mean_per_step/total_pre_clip_grad_norm_mean_per_step`：每个SGD步骤裁剪前范数的算术均值；总范数直接复用原clip返回值，骨干范数在该次裁剪前只读计算，不额外backward。
- `helper_clip_coefficient_mean_per_step/helper_clip_trigger_fraction`：同一原裁剪操作的系数与触发比例；helper冻结时实际可训练集合就是骨干，所属列总范数仍包括头。
- `trainable_parameter_count`仅表示本列更新参数数；`total_local_parameter_count`及capacity中的`stored_parameter_count`仍包含冻结头。冻结不会减小预测模型存储容量。

原CE/F/MSE/total、样本访问、SGD步数、耗时、骨干恢复训练距离、global-delta与SVD截断距离不改含义。参数空间仍分别标记为低秩训练参数或恢复骨干，不把头差异记成SVD误差。附加范数观察由原`multi_proto_diagnostics`控制，关闭不改变参数/RNG；冻结一致性检查始终执行。

已有正在运行的train对照是旧版日志，只有原总梯度范数等字段，未记录独立骨干范数。不能从旧总范数倒推出骨干范数；本次不因此自动重跑对照。测试准确率只用于固定成员post-local评价，不参与模型选择、冻结决策或权重计算。

## 命令与本次服务器启动

[multi_helper_head_commands.sh](../system/multi_helper_head_commands.sh) 提供一条完整COMMANDS，GPU6，`-did`靠前，明确使用服务器已有 `/data/zh/miniconda3/envs/htfllib/bin/python`。原因是服务器run_now中的`bash -lc`默认会找到系统Python；没有安装或修改环境。

真实数据固定为 `/data/zh/AllRepo/FedClip/dataset/Cifar100/dir_0.5`，NPZ-only，核验alpha.5/N20，源文件及解码指纹保留。分组只读：

```text
/data/zh/AllRepo/FedClip/system/dir05_proto_runs/dir0p5_seed0_gr100_20261105_v1/shared/K3_full_weights.json
```

使用members_only复用，不重新聚类、不改分组JSON。其余公共参数为K3、7/7/6、seed0、原五档.90/.37/.35/.25/.15、low_rank_recovered/.15初始化、lr.005/batch16/local5、F.001、clip10、gr100实际101轮、每轮评价。主指标仍为全20人固定所属模型普通post-local测试样本量加权准确率，Final/同轮最早Best/Last10/Last20及训练前指标不变。

本次用户一次性授权代理同步并启动。为保留GPU0/1/2上已有三组实验的代码与脚本，部署在隔离worktree：

```text
/data/zh/AllRepo/FedClip_helper_head_frozen_20261107
```

调度沿用服务器当前`run_now.sh`的副本，只替换其活动COMMANDS数组为新一条命令；原调度代码与主仓库run_now均不改。完整下发参数的逻辑字节统计继续如实记录。新输出统一为：

```text
/data/zh/AllRepo/FedClip/system/multi_helper_head_runs/dir0p5_seed0_gr100_20261107_v1/01_helper_head_frozen/
  checkpoints/  logs/  h5/  final_models/
```

实际启动回执（PID、代码提交、run_now来源SHA、日志路径、执行环境）另保存在该批输出目录的launch_receipt.json，不与模型预测状态混用。已有匹配的group_only_reset/backbone/train长实验复用，不再次启动。

## 验收

[验收报告](multi_helper_head_acceptance.json) 记录本地与服务器的实际结果。服务器短测命令示例（只生成新目录内合成夹具）：

```bash
PYTHONPATH=tests:system CUDA_VISIBLE_DEVICES=-1 /data/zh/miniconda3/envs/htfllib/bin/python -m unittest -v test_multi_helper_head test_multi_group_head test_single_avg_proto test_multi_last_epoch_proto test_multi_shared_proto test_multi_group_svd test_multi_group_proto test_group_projection test_target_head_l2_runtime
CUDA_VISIBLE_DEVICES=-1 /data/zh/miniconda3/envs/htfllib/bin/python system/verify_multi_group_head.py --output temp/multi_helper_head_check --width 16 --svd-diagnostics 1 --helper-head-update frozen
```

包括冻结6fe2a52默认train严格回归、只开CE的骨干非零梯度、跨轮新群组头、别名及优化器绑定、原型首轮/下一轮时序、诊断/评价隔离、异常恢复与模型重载；完整入口使用20人K3、标准卷积宽度16、512维/100类、五档异构SVD、两轮local5、F.001/clip10和原生内存上传。合成检查只证明实现，不声称真实涨点。
