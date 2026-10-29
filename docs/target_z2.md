# 旧单目标 FedTargetProj：Z2 特征正则

新增独立 `target_z2` 参数，保留旧单目标模型构造、初始化／种子、容量、SVD下载、恢复上传及聚合核。每客户端每轮仍只训练一个普通低秩模型。

## 使用范围与参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `--target_z2_lambda` | 0 | 有限非负系数，0直接走旧路径 |
| `--target_z2_start_round` | 2 | 完成训练次数编号，必须>=2；2对应loop_round=1 |
| `--target_z2_feature` | base_output | `base_output / classifier_input` |
| `--target_z2_diagnostics` | 1 | 0关闭逐客户端训练日志行，不改变目标、训练或主指标 |

非零只支持 FedTargetProj 的 `avg / softmax_only / projection_softmax`、`Decom_CNN-5-512`、共享头、joint、完整本地数据与全参与。Prox、头L2、FedRep、反馈、group、DWA、meta、完整CNN控制入口、断点／预训练／慢客户端／额外采集等未验证组合明确拒绝。零系数不限制原算法及组合，不增加路径后缀、原型操作或额外前向。

`main.py` 和 `run_target_proj.py` 均支持、校验并转发参数。原启动器省略新参数时保留旧命令；显式零或非默认起始轮等也正确转发。

## 目标与轮次

普通训练中使用 `CE + 原加权F + effective_lambda * features.square().mean()`，同时按 batch 和实际特征维度取均值，无额外1/2、归一化、分类筛选或目标尺度。全部客户端均生效。

首轮始终普通CE＋F；第二轮起按显式 `current_round` 启用，不额外预热。`_objective()` 仍只返回CE和加权F，不修改评价或其他训练路径。

复用多群组的 `Features / z2_loss` 纯特征接口和数学定义，不启用 `multi_proto_mode`，不调用原型采集／合并／通信。`base_output` 是原base的有符号输出，在head的原地ReLU前clone并保留计算图；`classifier_input`是实际fc3输入。一次原forward同时取得logits和特征，hook正常／异常退出均移除。

总损失一次反向，然后按原全部可训练参数、原norm上限裁剪，再执行原无动量／无weight decay的SGD。Z2直接给骨干梯度，分类器继续通过CE更新，不把Z2实现为分类器L2或F正则。

## 初始化、聚合与数据

使用原 `Model_Distribe` 和原服务器恢复，不换成多群组的初始化器、不变更内部manual_seed或ratio状态。policy记录实际初始服务器SHA、模型构造公式、各客户端容量／参数量、训练和测试样本数、实际分片路径及文件SHA。文件指纹读取不额外解码数据或遍历sampler；客户端模型元数据观察保护RNG。

Avg按原训练样本量权重。两个Softmax正式命令均固定C0=.125／helpers=.875，原始global-delta余弦评分、温度.20及内部归一化不变；Projection继续以普通C0 global delta逐helper投影。未重放历史权重CSV。同一上传快照下两种Softmax的逐客户端权重完全一致。

历史约55%的结果只作参考，不宣称本次初始化／随机轨迹已经与该历史运行严格对齐。

## 输出与主指标

日志独立于模型目录，均带模式及Z2配置后缀：

- `target_z2_clients.csv/json`：loop_round、completed_train_count、client_id、请求／有效lambda、特征位置、训练CE、加权F、原始／加权Z2、总损失、epoch、batch、样本访问量和耗时。
- `target_z2.jsonl`：逐客户端持续训练记录。
- `target_z2_primary.csv/json`：C0普通post-local与独立的聚合后下载准确率。
- `target_z2_summary.json`：Final、Best及最早轮次、Last10／Last20和实际点数。
- H5 `target_projection/target_z2`：policy、客户端JSON行、主指标JSON与数值曲线、summary。
- 最终 `manifest.json`：配置、实际初始化与容量／数据信息、C0普通预测checkpoint及同一主summary。

训练平均损失按实际SGD样本访问加权，不是endpoint评价。关闭diagnostics后，逐客户端记录为空，配置及C0主曲线仍保存。

准确率保持仓库fraction单位，需百分比时乘100。主指标始终是本轮C0普通5epoch结束、聚合之前的checkpoint。聚合后指标来自正常SVD下载当前服务器模型，不做额外微调；不能将同一行post-local／聚合后之差解释为本地微调收益。

仅对**正系数新实验**，H5根`rs_test_acc`和manifest的`best_accuracy`统一导出C0主指标。旧诊断曲线保存在`legacy_server_diagnostic_test_acc`，旧best保存在`legacy_server_diagnostic_best_accuracy`并标明原all-client evaluate()时序；`server.rs_test_acc`训练状态及原评价调用均不改变。旧`rs_train_loss`仍是原评价CE＋F，不称为Z2总训练损失。省略／零系数的旧导出完全不变。

`Client_0_model.pt`是普通post-local预测模型，`Server_model.pt`是聚合后共享模型，两者用途明确。仅修改导出标识，不增加测试遍历。普通下发、恢复、额外评价继续沿用旧checkpoint、buffers和RNG保护。

## 三条完整命令

从服务器`system/`使用 [target_z2_commands.sh](../system/target_z2_commands.sh)。文件含完整 `declare -a COMMANDS` 三条字符串，没有待填环境变量，只定义数组，不自动训练。

公共参数：Cifar100/pat_20、N20、C0、seed0、Decom_CNN-5-512、全参与、SGD .005、batch16、local5、F=.001、clip10、共享头／joint、Prox0、Z2=1、第二轮启用、base_output、gr100实际101轮、每轮评价。GPU示例0/1/2可替换。

输出为 `target_z2_runs/{模式}_z2_lam1_start2_base_output/{checkpoints,logs,h5,final_models}`，各组独立。Projection显式Prox0沿用现有零mu对照的命名／只读诊断后缀，实际损失没有近端项；不更改旧Prox判定规则。

## 短程验收

```powershell
# 仓库根目录，使用已有环境
& D:\miniconda\envs\htfllib\python.exe -m unittest discover -s tests -p test_target_z2.py -v
& D:\miniconda\envs\htfllib\python.exe -m unittest discover -s tests -p test_target_z2_runtime.py -v
# Windows HDF5需要短输出路径；目录必须尚不存在
& D:\miniconda\envs\htfllib\python.exe system/verify_target_z2.py --output D:/z2_NEW --jobs 2
```

服务器`system/`可运行 `python verify_target_z2.py --output temp/target_z2_NEW --jobs 2`。工具只在新目录生成微型合成数据，不访问或重划真实分片；三种模式各运行两轮、N20、原宽度异构CNN、local5，另做Projection日志关闭配对。验证后清理自己目录内的模型二进制，保留日志和收据。

详见 [验收报告](target_z2_acceptance.md)。本地合成验收不代表真实涨点，也不启动101轮正式训练。
