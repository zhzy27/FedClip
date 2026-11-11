# fc1尾部因子可训练性：离线A/B

入口为`system/fc1_tail_diagnostic.py`，仅在离线副本上训练，不导入或调用服务器训练／聚合循环。旧模型、客户端训练函数、联邦入口及其默认值均未修改。

## 来源与编号

使用同一FA02导出目录的三个必需文件：`Server_model_0.pt`、`Global_class_sample_latest_generated.packet`、`manifest.json`。传`--source-run`时只在该目录的`final_models`下查找唯一服务器文件；有多个／缺少文件立即失败，不猜测其他实验。也可用`--server-file`直接指定服务器文件，其原型包和manifest必须相邻。

默认要求来源完成101次聚合，且为K1、完整服务器、full_avg、global_last_epoch／class_sample／raw_mean／base_output MSE。使用R101**最新生成**的原型，明确标为“基于R101服务器的离线R102本地训练”。不使用R101客户端模型继续训练，也不误用R101实际训练使用的R100原型。测试分片不参与优化。

检查服务器角色、轮次、模型结构、容量分配、原型包schema、采集与使用轮次、原型完整指纹、源manifest声明、全体训练／测试分片文件及解码指纹、初始化指纹；还按源seed／构造／恢复方法重建S0核验其哈希。原型包中的`source_server_sha256`属于R101训练前的S100，不应拿它与聚合后S101比较；原型与该运行的绑定通过manifest的`latest_generated_library`完整记录及包哈希检查。旧导出格式没有S101文件哈希声明，不能凭原型的S100哈希额外证明最终S101文件内容；本工具记录实际输入S101文件与状态哈希供追溯。

仅读取已有NPZ和config，不生成真实数据。默认从源配置选择`.35/.90`两档，原id%5分配下分别为`2,7,12,17`和`0,5,10,15`。可通过`--client-ids 2,7`选择子集。实际秩取原`FactorizedLinear`构造规则；原宽度下最低档120、.35档280、.90档720，未把比例当参数保留率。

## A/B与训练

每客户端只执行一次原`decom_larger_model()`。离线作用域内捕获**这一次**fc1 SVD的P／σ／Qh，原返回值不变；B深拷贝A。前120槽位、bias和其他层不动，只调整尾部：`a=max(sqrt(σ_j),sqrt(σ_120))`，`U_j=a P_j`、`V_j=(σ_j/a)Qh_j`。零奇异值仍用同次SVD的单位向量；若下限本身为0，则σ/a采用有限0约定，不虚造正下限。

训练前核对恢复fc1矩阵、完整训练／测试集logits、argmax以及相同首batch的CE、加权F、原始MSE和总损失。默认数值容差`atol=1e-6, rtol=1e-5`，记录实际最大误差；超出或argmax变化明确失败。

训练直接调用未修改的`utils.single_avg_proto.train`；CE／F／MSE、一次联合backward、全可训练参数裁剪和无动量SGD全部沿用。上一轮原型库在A/B间冻结共用。原训练函数仍收最后epoch统计，但离线新包直接丢弃，不替换输入库、不反馈服务器。两分支各新建优化器、恢复同一训练RNG；复用原`single-avg-proto-v1`的seed／client／loop=101数据流，完整比较稳定样本索引，SGD步数为`epochs*ceil(N_train/batch)`。

参数默认继承源manifest，可显式配置epochs、batch、lr、F／原型系数、clip及seed，记录请求值与生效值。本入口要求原型系数大于0；无F／无原型的CE梯度隔离使用独立数值测试，不冒充本次配对配置。正式命令显式指定原公共配置。`-did`在导入Torch前仅暴露请求的物理GPU；旧训练恢复函数因此也不会初始化其他显卡。新增探针仅保护实际模型设备的CUDA状态。CPU通过`--device cpu`选择。

## 如何读诊断

- `first_batch_CE_signal.csv/json`：第一次真实训练前向中，用`autograd.grad(CE, fc1_output, retain_graph=True)`读取dy，计算`G=dy.T@fc1_input`，再投影到下载时固定P／Q尾部。不会执行第二次模型前向，也不写实际参数`.grad`。这是**分类信号**，不含F／MSE。
- `first_step_gradients.csv/json`：实际CE＋F＋MSE反向的尾部U／V裁剪前后梯度、总范数和全模型裁剪系数。参数hook只读；实际clip调用未替换。先观察实际裁剪后梯度，再观察原optimizer.step后的矩阵变化。
- `fixed_slot_updates.csv/json`：第1步、epoch1、epoch5（其他预算则末epoch）上的固定前部／尾部乘积变化。用实际float32参数的私有float64 CPU副本做矩阵乘法和差值，减少观察中的舍入掩盖；不改训练dtype。另在实际U@V矩阵上重新SVD记录前120／其余奇异值平方能量及比例。固定因子槽位变化与重新排序的谱能量不同，不能互相替代。
- `endpoints.csv/json`：训练前、末epoch的train/test CE及准确率；末epoch还用SVD最佳rank120 fc1的**直接dense推理**，仅替换fc1.weight，其bias、其他层和分类头保持不变。不是简单把下载时120以外槽位清零，不是重新压缩整个模型。
- `client_pairs.csv/json`：逐客户端A/B、测试计数、尾部更新、单层截秩掉点、实际步骤及样本访问量。`capacity_summary.csv/json`按档位累加测试correct/samples；矩阵变化另用客户端均值，明确不是准确率。
- `local_training`与`batch_order`：原训练过程CE／F／MSE／总损失，实际batch、epoch、访问量、索引哈希及训练RNG哈希；时间标为包含在线诊断的训练墙钟时间，另列诊断耗时。谱SVD和截秩评价是额外计算，未声称计算成本与原训练相同。

准确率为fraction，CE为每样本交叉熵；没有测试驱动的epoch／模型选择。默认只保存诊断表与manifest，原训练函数产生的临时checkpoint会在作用域结束清理；`--save-models`才保存带offline标记的A/B预测模型。可选`--historical-client-dir`只读原R101客户端检查点做单独历史评价，不作为下发起点。

只读评价和谱观察保护参数、buffers、模块模式和训练RNG；临时hook、损失包装与optimizer.step观察在异常时清理。输入模型、原型及manifest结束时复核文件哈希。离线进程读取源manifest的Torch deterministic、cuDNN benchmark／deterministic和TF32实际设置，不自行换成另一套GPU数值协议；来源若开启非确定性CUDA内核，不能据CPU逐元素验收承诺CUDA逐位复现。

## 验收与运行

[验收报告与日志](fc1_tail_diagnostic_acceptance.json)：39项CPU检查中38项通过、1项旧CUDA专用检查跳过；另完成真实入口、原宽度模型的离线A/B与冻结旧函数配对验收。真实FA02 R101模型／原型和pat20分片不在本机，未执行真实准确率诊断。

[完整单条命令数组](../system/fc1_tail_diagnostic_commands.sh)从服务器`system/`运行，GPU直接写0、可修改，输出到独立`fc1_tail_diagnostic_runs/FA02_R101_offline_R102_floor120_20261111_v1`。数组仅定义，不自动执行。

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH='tests'
python -m unittest -v test_fc1_tail_diagnostic
python system/verify_fc1_tail_diagnostic.py --output E:/fc1_tail_unique
```

验收工具先用真实main产生N20/K1两轮合成FA02来源，再以.90/.35客户端各一人为原宽度模型离线A/B smoke，保持每分支5epoch；其来源编号为R2→离线R3，**不是用户真实R101结果**。另对冻结`97dc436`原本地函数／公共loader比较A参数、损失、原型包与RNG，并核对诊断开关、批次、重载及输入错配失败。

判读优先看裁剪后的实际固定槽位矩阵更新。只有A尾部有CE信号但实际更新小，且B在同预算下增大更新并维持到末epoch，才支持因子尺度限制本地学习；准确率及单层截秩掉点再回答这些更新是否有分类价值。此处不能证明跨联邦轮次能够保留，也不能用合成数据证明真实准确率收益。
