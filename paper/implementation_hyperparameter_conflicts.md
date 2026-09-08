# Implementation Details 审计冲突

审计基于 `main` 分支提交 `08d7383`。机器可读结果位于
`paper/implementation_hyperparameter_audit/`。

1. **解析器默认值不是论文主实验值。** `main.py` 默认是 2000 轮、1 个本地 epoch、batch size 10、2 个客户端和学习率 0.01；启动记录显式覆盖为 100/5/16/20/0.005。
2. **训练并未完整固定全局 seed=0。** 当前只激活了 `torch.manual_seed(0)`；完整 `set_seed(0)` 被注释，而客户端选择和上传抽样分别使用 NumPy 与 Python `random`。
3. **学习率版本不能靠方法名自动区分。** `--use_asymmetric_lr` 默认是 1。省略该参数时执行非对称学习率；matched learning rates 必须显式传入 0。
4. **ResNet-18 的有效默认不是 original ordered。** `SVD_resnet.py` 的构造函数默认虽为 `original`，但 `main.py` 显式传入默认值为 `dynamic_capacity` 的共享参数。要运行 original 必须显式传参。
5. **ResNet-18 正则系数与聚合温度记录不统一。** 当前启动记录同时存在 `regular_lamda=1e-3, tau=1` 和 `regular_lamda=1e-4, tau=5`，应以最终结果 JSON/manifest 中保存的 `args` 为准。
6. **参与率实验命令没有保存在当前 launch 文件。** 参与率 0.2/0.4/0.8、500 轮是论文实验协议，不是解析器默认值，需要通过对应结果 JSON 核实。
7. **循环存在一次未评估的额外更新。** FedCLIP 使用 `range(global_rounds+1)`；设置 100 时报告 100 个更新后评估点，但最后还执行一次不进入结果曲线的更新，最终导出模型因此与最后评估状态不同。
8. **硬件和 CUDA 环境信息缺失。** 仓库只固定 Python 3.11 和 PyTorch 2.0.1，未持久化主服务器 GPU、CPU、内存、操作系统、CUDA 与 cuDNN 版本。
9. **不能声称所有基线都是原样官方代码或经过统一验证集调参。** README 只能确认仓库为 HtFLlib/PFLlib-compatible；仓库没有验证集和统一超参数搜索记录。
10. **红色提升值的参照方法未编码。** 该值由论文排版阶段计算，最终表注需人工说明其相对最佳基线还是指定基线。
11. **`launch.json` 是混合历史草稿。** 其中仍有当前 `main.py` 不接受的 `staged` 模式，以及其他分支遗留的聚合参数，不能将整个文件直接视为可执行的最终实验清单。

建议在论文最终定稿前，从用于主表的每个 JSON/manifest 中导出 `args`，尤其核对
`use_asymmetric_lr`、`rank_dropout_mode`、`regular_lamda`、`aggregate_tau`、
`join_ratio`、`global_rounds` 和 `times`。
