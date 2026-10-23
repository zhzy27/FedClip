# K3参数预算批次验收

日期2026-10-23，起点 `target_proj / 1e641ab`。本批没有修改生产训练、分解/恢复、聚合或评价代码；新增独立配置/命令、只读审计及验收覆盖。没有修改环境、真实数据或绘图，没有推送或启动正式101轮训练。

## 预算及范围

实际实例化原五档普通CNN和20×3个低秩模型，层秩/参数量与用户表格逐项一致。20个客户端全部满足三模型总数不超过对应旧普通CNN预算；同时与实际SVD下载后的 `capacity_manifest.json` 逐客户端、逐模型核验一致。[预算报告](multi_group_parameter_budget.md) 和 [机器可读计数](multi_group_parameter_budget.json) 保存详细计数、新旧合成分组及容量构成。

普通低秩推理模型、三个模型总参数、完整服务器/恢复上传缓存和训练访问量已分别命名。原K1单模型实际参数另列，没有声称与新K3严格等参数。通信方式未改变，不称通信预算对齐；参数量不作为峰值内存。

## 已通过验收

| 检查 | 结果 |
| --- | --- |
| 原宽度低容量实际main入口 | N20、K3、五档新秩、每模型5epoch、2轮；Projection和Avg-only均通过。额外执行同配置Projection关闭预算审计的精确对照。 |
| 训练任务与预算 | 每个用例120次独立本地模型训练、600个SGD步骤、1500次合成样本访问；每人每轮3×5次数据遍历。没有额外预热、漏列或复制训练结果。 |
| 首轮与配对 | 全部20×3训练及恢复上传完成才按完整权重分组，立即首轮聚合。两模式首轮模型/上传指纹、随机状态和分组文件严格一致。 |
| 分组公式 | 每列恢复完整权重float64逐参数Gram独立复算，再对三列余弦取均值。仅对不同归约顺序使用1e-12绝对容差；没有改变训练、投影或聚合容差。 |
| 聚合核 | 每轮每列重新调用原 `aggregate_group_updates`，所有服务器参数逐元素一致，最大参数误差0。 |
| 审计隔离 | 开启/关闭预算观察：每次本地模型、恢复上传、服务器聚合、实际batch、分组和最终Python/NumPy/Torch CPU/CUDA随机状态严格一致。只读调用也验证模型参数、buffers、模式和RNG不变。 |
| 数值与模型 | 低秩参数和恢复上传全部有限；各客户端容量及层秩正确。全部3个服务器和20×3普通低秩checkpoint保存重载，参数与固定输入预测严格一致。 |
| 评价/导出 | 不等训练/测试样本数；按固定归属独立复算全20人、三个群组、五档容量的correct/samples。CSV/JSON/H5/manifest主序列与summary一致，best取同轮曲线最早最高点。 |
| 原高容量严格回归 | 另跑原默认秩、N20/K3、两轮，与 `1e641ab` 交付时已保存的报告比较：初始模型、每次本地上传与服务器参数指纹、batch、分组、主summary和最终RNG全部相同。 |
| 旧路径 | 原多群组9项、SVD9项、预热8项、旧单目标Projection14项通过；生产同构/高容量及单目标文件未改。 |
| 新审计测试 | 6项通过：实际计数、20人预算、别名和跨模型独立性、只读状态、实际manifest错配、K/形状/预算变化、旧容量分组拒绝、独立两命令及默认容量不变。 |
| 启动入口 | 两条完整正式命令经真实main预览解析（数据仅替换为隔离合成目录、设备为CPU，未训练）；`bash -n`、Python编译、diff空白检查通过。 |

合计46项不同单元/集成测试，加上述三个实际main合成用例及原高容量严格回归。使用已有 Python3.11.14、Torch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3；实际原宽度main验收使用CPU、2线程。旧环境原有transformers/fused-window可选依赖提示未改；未将CPU合成检查描述为服务器真实数据性能验收。

## 证据与复验

[轻量验收报告](multi_group_budget_acceptance.json) 保存用例、轮次、数据/初始化指纹及判定。本机更详细的只读记录为：

- `system/temp/svd_budget_acceptance_20261023/acceptance_report.json` 及各用例 `verification.json`：实际batch、逐次参数/上传/RNG指纹、配对分组、模型重载及主指标。
- 其中 `projection/parameter_budget.json`、`avg_only/parameter_budget.json`：真实SVD容量交叉核验；`budget_observations.json` 为每轮20客户端读取记录。
- `system/temp/svd_budget_legacy_regression_20261023/acceptance_report.json`：默认高容量严格回归。
- `system/temp/svd_budget_report_20261023/parameter_budget.json/md`：新旧容量两份合成分组名单及各自容量构成。名单仅作合成证据，正式运行仍重新自动分组。

所有合成模型张量在保存重载验证后已由验收工具清理，保留轻量日志、分片和报告；正式模型未被触碰。复验命令与正式两条COMMANDS见 [配置及结果目录说明](multi_group_budget.md)。

绘图 `system/plot_hyperparameter_sensitivity.py` SHA256保持 `90FF3A5C9FF8615CE28953D89598A6295AD6B513BA235293309E6875D429F95C`，不纳入本次提交。

本次仅说明实现、参数预算和合成流程通过。**不能据此判定压缩后的真实准确率已经达标**；真实数据性能由用户启动正式实验后判断，新旧分组变化需与参数量变化一并记录。
