# 本地分类头 L2 验收记录

日期：2026-10-20。实现起点：`target_proj / 9a9e4c1`。

仅运行合成短程测试；未修改环境、生成/重划真实数据、启动正式训练或推送。真实 CIFAR-100 准确率收益尚未验证。原绘图文件未修改，也不纳入本次提交。

## 环境

使用已有 `D:/miniconda/envs/htfllib/python.exe`：Python 3.11.14、PyTorch 2.0.1+cu117（CUDA build 11.7）、NumPy 1.26.4、SciPy 1.16.3。CUDA 可用，但本批训练数值验收使用 **CPU**，不据此宣称完成 CUDA 内核数值验收；隔离检查同时比较可用 CUDA RNG 状态。

环境原有 transformers / fused-window / TypedStorage 提示仍存在，本批 CNN 测试成功，无依赖安装或版本修改。

## 验收结果

| 检查 | 结果与证据 |
| --- | --- |
| 零系数严格回归 | `test_target_head_l2_runtime.py` 分别加载 `git show 9a9e4c1` 的客户端、服务器源码作为参照；与新版本未传、显式0对比，20客户端、2轮、joint5、异构五种秩，逐客户端上传、聚合参数、接收顺序、实际训练 batch 及 Python/NumPy/Torch CPU/CUDA RNG 全部严格一致。参数误差为0（完整 state_dict 指纹一致）。 |
| 观察隔离 | 同一非零系数 `.001` 开/关专项观察后，同样的逐客户端参数、服务器参数、实际 batch、接收顺序和 RNG 全部严格一致；观察前后所有 checkpoint 文件 SHA256 不变。 |
| 头 L2 梯度 | 独立反传精确得到 `fc3.weight.grad=lambda_h*W`，bias及骨干梯度为空；与CE/F组合反传时，仅该weight梯度增加这一项。别名引用身份一致，未重复计算。 |
| joint与F路径 | `_objective()` 仍仅返回CE/F；联合训练中头、骨干均变化，原加权F非零。完整参数集合统一裁剪10，SGD无weight decay。 |
| 损失均值 | 独立记录尾批及不同样本数的实际batch值，复算CE、加权F、加权L2和总损失；按真实样本访问数加权相符。 |
| 分类头生命周期 | 各客户端第一轮头等于共同服务器初始化；下一轮下载后仍逐元素等于各自上一轮头，weight/bias均被检查。每人每轮只调用一次训练，5epoch，无额外训练。 |
| endpoint只读 | 使用含混合模块模式和故意在推理时修改buffer的模型检查：正常模型参数、buffers、模块模式、checkpoint及RNG不变。CE-only观察不调用F或头L2目标。 |
| 原宽度实际入口 | `test_target_head_l2_cli.py` 通过真实 `main.py`、20客户端、原 `n_kernels=16` 和五种低秩容量运行2轮、每人每轮5epoch，100类head；`.001` 应用于所有客户端。 |
| 导出一致 | 实际入口及运行时测试检查40条客户端、2条C0专项记录，CSV/JSON/H5数值一致、主结果summary一致；manifest含系数和20人最后轮训练统计。最终普通C0及聚合骨干＋本地头模型能重载并输出100类；服务器骨干文件明确不能独立推理。日志ZIP包含专项CSV，不含`.pt`。 |
| 入口及非法值 | main真实parser和启动器验证合法配置、转发系数/观察开关，拒绝负数、NaN、Inf、shared/FedRep/Projection/反馈/群组/DWA/meta/非零Prox等未支持的非零组合。非零实验保留显式实验名称，目录带系数。 |
| 旧模式回归 | 原本地头、Prox、Avg/Projection/ProjectionSoftmax/ReLU聚合、FedRep schedule的受影响单元/集成测试通过；未改算法核。 |

运行时严格对照为降低反复SVD成本使用同一 `Hyper_CNN_512` 的 `n_kernels=4`，保留512→100分类头、五种容量及完整训练生命周期；另有上表**原宽度 `n_kernels=16` 的真实main入口**验收，未将减宽测试冒充正式模型。所有测试分片、checkpoint和模型产物均置于自动清理的临时目录，不覆盖真实数据/正式结果。

## 测试清单

新测试8项、相关回归55项，共63项通过：

| 文件 | 测试数 |
| --- | ---: |
| `tests/test_target_head_l2.py` | 5 |
| `tests/test_target_head_l2_runtime.py` | 1（内部含五组两轮流程） |
| `tests/test_target_head_l2_cli.py` | 2（含真实main两轮及多个parser子用例） |
| `tests/test_target_local_head.py` | 8 |
| `tests/test_target_prox.py` | 11 |
| `tests/test_projection_weighting.py` | 15 |
| `tests/test_target_projection.py` | 14 |
| `tests/test_fedrep_schedule.py` | 7 |

原Prox/Projection旧测试夹具缺失 `algorithm`，另有一个服务器非法参数夹具缺失 `model_family`，会在既有裁剪/完整模型校验中触发AttributeError；已核对起点源码具有同样缺失。本次仅补齐这两个测试文件中的必要配置，未修改生产裁剪规则、算法、断言或误差容限。

另通过 `bash -n system/target_head_l2_commands.sh` 和 `git diff --check`。绘图文件 `system/plot_hyperparameter_sensitivity.py` SHA256 保持 `90FF3A5C9FF8615CE28953D89598A6295AD6B513BA235293309E6875D429F95C`。

复验命令见 [target_head_l2.md](target_head_l2.md)。三个31轮正式参数入口仅已写入命令数组，未启动；已有 `10-15-03.zip` 两组对照未重跑。本批验收只支持实现正确性，不支持准确率提升的结论。
