# raw_mean 自然尺度原型验收

参照提交：`43eb806`。新增目标类型默认 `fixed_norm`，本次仅实现、合成短程验收及命令交付。未修改环境、真实分片、旧命令或绘图，未推送、未启动正式训练。

## 单位及旧模式回归

现有环境：Python 3.11.14、PyTorch 2.0.1+cu117、NumPy 1.26.4、SciPy 1.16.3、h5py 3.15.1；使用 `D:\miniconda\envs\htfllib\python.exe`。本次 CPU 验收，不声称 CUDA 数值验收。

- **12 项新测试 PASS**：`system/temp/multi_proto_raw_unit_20261029_final.log`。
- **120 项旧测试 PASS**：原型相关24项见 `multi_proto_raw_regression_focus_20261029.log`；其他多群组、SVD、初始化、KD、Projection核及离线蒸馏96项见 `multi_proto_raw_regression_20261029.log`。
- 新两条命令通过 Bash 语法检查与实际 main 参数解析／协议校验，实际预算均为101次训练／聚合。

新增检查包括：

1. 范数相差明显的样本 `[100,0]` 一次与 `[0,2]` 三次，得到原始均值 `[25,1.5]`，不是等客户端均值，也不是单位特征均值。不同类别范数可为5、25以上等，不做定长或再次单位化；模型列独立。
2. 不等类别样本数、群内优先、组外补齐。改变组外同类数据不改变已有群内来源原型。
3. 群内 `[3,-4]` 与 `[-3,4]` 抵消成零：仍有效、来源群内、无无效原因。近零仍有效；全员无样本才无效。
4. 原始 MSE 损失和梯度与直接 `F.mse_loss(..., reduction='mean')` 逐元素一致。目标无梯度；全无效batch可微零；骨干获得有限非零梯度，分类器无直接辅助梯度。
5. 采集结果与相同实际批次的独立 `model.base` 前向 sum/count 逐元素一致。模型参数、buffers、模式和 Python/NumPy/Torch RNG 不变。
6. 统计包接收、合并、Session、保存库均拒绝原型类型错配；采集／训练特征位置错配拒绝。raw_mean 非 MSE、关闭模式或非默认固定范数等非法配置拒绝。
7. 零系数／首轮调用采集或辅助特征接口即失败的测试仍通过；没有额外forward或SGD。
8. 对固定 `43eb806` 原原型模块，当前默认 fixed_norm packet **逐字节一致**，训练及 RNG 严格一致。
9. raw_mean diagnostics 0/1 的库 packet、训练模型及 RNG 一致。
10. 非有限特征、统计、目标、损失均失败，不静默跳过。

## 真实入口两轮合成验收

入口 `system/verify_multi_group_proto_raw_mean.py`，输出 `system/temp/multi_proto_raw_mean_acceptance_20261029/`；最终状态以 `acceptance_report.json` 为准。

**六个案例及最终跨案例验收全部 PASS，工具退出码0。** 默认fixed_norm、零系数、首轮及诊断开关的严格对照全部通过；独立原始均值复算和逐列旧聚合核最大参数误差均为0，所有预测checkpoint重载一致。

真实 `main.py`、20客户端、K3、原宽度16的异构低秩CNN、五档原容量、rank .15恢复初始化、full_weights首轮分组、7/7/6、两轮、每模型5epoch。每个客户端训练2/3张、测试1/2/3/4张合成图像，覆盖不等样本量评价。每例120个模型训练任务、600个SGD step、1500次训练样本访问。

六个案例：固定43eb806无辅助参照、raw_mean零系数、固定43eb806 fixed_norm MSE参照、当前默认fixed_norm MSE、raw_mean MSE、raw_mean关闭诊断。所有案例复用参照首轮发布的同一合成分组，不代表服务器正式分组成员。

已完成的默认 fixed_norm 完整入口对照与43eb806严格一致，包括实际原型packet指纹及字节数。raw_mean正系数案例的独立全库复算误差为0；其有效类别范数约150.5012–185.6523，确认未单位化或定长到1，仅为这组微型合成数据的机制检查。

raw_mean第二轮额外采集150个样本前向／90批，统计上传24,616,726字节、单份库1,231,363字节、20人下发24,627,260字节；采集约6.68秒、合并约.48秒。均为本机合成实测，不代替真实数据运行成本；SGD访问量仍单独记录。

要求以下检查全部通过才报告PASS：

- 默认fixed_norm参数、上传、逐列聚合、batch、RNG、主指标及实际packet与43eb806严格回归。
- raw_mean零系数与原无辅助流程严格一致；全部首轮上传／分组一致。
- 两个raw_mean案例在全体下载后、任意SGD前，独立用真实base前向、手工sum/count复算全部20×3来源及每类目标，要求误差0；观察前后模型和RNG不变。
- 每列对同一恢复上传调用旧群组核，参数最大误差0，不放宽旧容差。
- 全20人／群组／容量加权指标由correct/samples独立复算，Final／同轮Best最早轮次／Last10/20正确。
- CSV／JSON／H5／持续JSONL与manifest一致。raw_mean固定目标范数字段null、目标类型一致、目标样本访问量和特征尺度诊断明确。
- raw_mean库可追溯并经类型检查，预测不依赖库；三份服务器及60份普通低秩客户端checkpoint重载预测一致。
- 诊断开关的模型、上传、聚合、batch、RNG严格一致。

验收工具仅在完成保存重载检查后清理自己创建的模型二进制，保留轻量数据、日志、ZIP及验收收据。`--audit-only` 复核既有收据，不声称重载已清理的模型。

本批临时 `.pt` 和 `.packet` 文件已清理，不删除用户原模型、真实数据或旧实验结果。

## 边界与交付

本地没有真实 `dataset/Cifar100/pat_20/train/0.npz`。合成验收不能用于判断真实涨点。

[实现与字段说明](multi_group_proto_raw_mean.md) · [两条完整命令](../system/multi_group_proto_raw_mean_commands.sh)。命令只读取上一批现有分组；服务器实际分组路径由 `PROTO_RAW_GROUP_FILE` 或旧 `PROTO_LOSS_GROUP_FILE` 明确提供，不自动创建或搜寻。

旧绘图 SHA256 保持 `90ff3a5c9ff8615ce28953d89598a6295ad6b513ba235293309e6875d429f95c`。

自然尺度方案同时改变样本尺度对均值方向的贡献，不将其描述为仅改变目标范数。
