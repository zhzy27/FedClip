# SVD 多模型分组验收记录

日期：2026-10-23。实现起点：`target_proj / 1826784`。仅执行隔离合成验收，没有修改环境、真实分片或绘图，没有推送或启动正式101轮实验。

## 本地环境与边界

使用已有 `D:/miniconda/envs/htfllib/python.exe`：Python 3.11.14、PyTorch 2.0.1+cu117、NumPy 1.26.4、SciPy 1.16.3。环境原有可选transformers/fused-window提示未改动。完整入口数值验收在CPU上运行；另有CUDA小宽度Hyper_CNN测试，验证新分解模块移设备、优化器绑定、CE/F训练和CPU恢复上传。不能将合成准确率当成真实数据收益，也不宣称完成服务器真实数据的性能验收。

## 已通过的检查

| 内容 | 证据 |
| --- | --- |
| 默认同构严格回归 | 原 `verify_multi_group_warmup.py --compat-only` 对照冻结 `2fc0e09`：20人/K3/两轮，初始模型、每次本地上传、服务器参数、实际batch和Python/NumPy/Torch CPU/CUDA RNG严格一致，分组指纹相同。 |
| 旧单目标 | `test_target_projection.py` 14项通过，覆盖原Avg/Projection数学、下载、本地CE/F、buffers、只读评价与保存。 |
| 容量 | 默认20人按id%5分配，每档4人；每人三个模型同档并跨轮固定。实际conv2/fc1/fc2秩与原rank规则一致；参数数目见实现说明，.90档没有被擅自压成所谓90%参数量。 |
| 优化器/训练 | 下载分解、移设备后重建SGD；参数对象ID严格一致，fc1低秩因子实际更新，原F损失为正。K个模型、优化器和新鲜上传均检查；重复训练及缺列拒绝。 |
| 上传与checkpoint | 独立恢复普通模型逐参数复算对应列上传；恢复和聚合不改变普通checkpoint SHA256。保存的客户端预测模型仍含低秩U/V及自己的容量信息，服务器文件为完整母模型。 |
| 首轮时序 | 实际入口检查20×3训练任务全部完成、各上传轮次为0后才调用一次分组，随后同轮聚合；无额外预热/训练。 |
| 相似度 | 普通post-local低秩模型恢复后，完整named_parameters含fc3、别名去重。真实宽度CNN逐参数float64 Gram独立复算，与K列余弦均值相符；独立归约顺序容差仅1e-12。未改投影epsilon或聚合容差。 |
| 原群组核 | 每轮每列用当前完整母模型及该列恢复上传直接调用原 `aggregate_group_updates`，对照新服务器更新所有参数逐元素相等，最大参数误差0。 |
| 配对分组 | K3 Projection发布首轮自动分组，K3 Avg-only完成同样首轮任务后复用。首轮上传/随机状态严格一致，组文件完全相同；context拒绝旧同构及不同容量映射文件。 |
| 观察隔离 | K3两轮、pre测试和SVD距离观察同时开/关：每次普通低秩模型、恢复上传、聚合、实际batch、分组和最终RNG严格一致。 |
| 加权与时序 | 非均衡训练/测试样本数；独立按固定g(i)复算全员、三个组及五档容量的correct/samples。best为同轮全员曲线最大值并选最早轮次，tail按评价点。 |
| 稀疏及强制末轮 | K1、20人、gr2实际3轮、eg3：评价loop为0和2，末轮2不是间隔倍数仍强制评价；训练任务60，每人每轮一次5epoch。 |
| 导出 | 两种K3模式及K1的CSV/JSON/H5/manifest主序列与summary一致。实际K服务器及全部客户端×K低秩checkpoint均保存重载，参数与预测严格一致；加载结构与实际容量匹配。 |

实际main入口四个合成用例：

| 用例 | N/K | 完成轮数 | 每模型epoch | 训练任务数 |
| --- | --- | ---: | ---: | ---: |
| group_projection_softmax | 20/3 | 2 | 5 | 120 |
| group_avg_only，同分组文件 | 20/3 | 2 | 5 | 120 |
| Projection关闭pre及SVD距离观察 | 20/3 | 2 | 5 | 120 |
| K1全员group_avg_only | 20/1 | 3 | 5 | 60 |

所有实际入口都使用原宽度 `n_kernels=16`、100类模型和五档容量，没有用减宽网络替代上述smoke。辅助单元使用同一个Hyper_CNN类的小宽度模型来减少重复SVD开销。合成分片只在验收目录中生成，pat元数据核验明确记录观测标签不足20类的限制，不将其冒充真实pat20数据。

## 测试与复验

40项相关单元/集成测试通过：新增SVD 9项，原多群组9项、预热8项、单目标Projection 14项；另有上述实际main验收和冻结源码对照。通过 `git diff --check`、Python编译及三命令文件 `bash -n`。

本机原始轻量报告位于：

- `system/temp/svd_acceptance_20261023/acceptance_report.json`：四用例PASS、逐列误差0、相似矩阵容差、观察隔离。
- 同目录各用例 `verification.json`：逐客户端/模型参数指纹、实际batch与RNG、分组和指标。
- `avg_all/verification.json`：补验 `eval_gap=3 / evaluation_loop_rounds=[0,2] / last_loop_forced=true`。
- `system/temp/svd_homogeneous_regression_20261023/acceptance_report.json`：冻结 `2fc0e09` 的严格回归PASS。

正常验收工具在模型重载验证后自动清理合成模型、保留轻量报告；单独补验K1的内部worker运行保留了其预测模型供核对。临时参照源码副本约17.7MB，其递归清理被自动审批以 `blocked by policy` 拒绝，已保留在本机验收目录，不纳入Git提交。所有验收目录与正式runtime目录均按专属路径忽略，避免误提交模型/数据。

绘图文件 `system/plot_hyperparameter_sensitivity.py` SHA256保持 `90FF3A5C9FF8615CE28953D89598A6295AD6B513BA235293309E6875D429F95C`。正式命令与参数/产物说明见 [multi_group_svd.md](multi_group_svd.md)。未对真实准确率作改善结论。
