# K1纯低秩Avg：每轮fc1尾部因子尺度调整

`--multi_fc1_tail_scale_strength`默认0，有限非负。0直接执行原`decom_larger_model(rate)`，不创建回调、读取奇异值或进行乘积核验。正值c只在显式K1全员Avg机制入口生效：

```
FedMultiGroupProj --multi_avg_proto_ablation --multi_svd -nsm 1
--multi_group_mode group_avg_only --multi_group_sizes 20
--multi_svd_aggregation full_avg --multi_proto_mode none --multi_proto_lambda 0
```

第一版不支持K3、rank_partition、任何原型模式、Projection、私有头、非joint、Prox、Z2、头L2、replay、课程或KD。非零值明确拒绝这些组合；默认0不扩大旧配置限制。c=1是本次正式B；其他有限非负c可配置，计算若溢出／出现非有限值会失败，不暗自钳制。

## 下发与数值定义

原`Decom_LINEAR`增加可选的`svd_observer`，`Hyper_CNN_512.decom_larger_model`增加仅针对fc1的可选入口。缺省None时，原SVD、乘法、参数复制和随机操作顺序保留。

启用后，每次客户端下载完整服务器模型，仍执行原conv2／fc1／fc2的三个SVD。fc1完成平衡因子与bias复制后，用**同一次SVD**的P、σ、Qh进行调整，未额外分解、重排或从训练后的恢复矩阵反推方向。

最低秩来自`factorized_linear_rank(in_features,out_features,minimum_assigned_rate)`；它就是原`FactorizedLinear`构造的`max(1,round(rate*min(in,out)))`规则，未新增另一种取整。最低档取实际客户端容量分配，不取未分配的列表项。原宽度800→2000及五档容量下，实际秩为720、296、280、200、120。

对零基槽位`[r_min,r_client)`：

```
a_j = max(sqrt(sigma_j), c*sqrt(sigma[r_min-1]))
U[:,j] = a_j * P[:,j]
V[j,:] = (sigma_j/a_j) * Qh[j,:]
```

前r_min槽位、fc1.bias、conv2／fc2和其余参数不动。σ=0且下限>0时，使用同次SVD单位P列，U非零、V为0；下限与σ均为0时，两侧为0。没有除以原零因子范数，也不虚造正下限。最低实际秩客户端整个调整分支跳过，保持原平衡下发。

训练开始前，模型已迁移到实际训练设备，优化器才根据当前参数创建。每轮重新执行本轮服务器矩阵的SVD及尺度规则，未跨轮固定最初的方向或因子。SGD、CE＋原恢复权重F、全参数裁剪、恢复完整上传和样本量full_avg均未修改；原型不采集、不编码、不下发。没有新增SGD步、epoch或模型训练前向。

数学纯函数位于`utils/fc1_tail_factors.py`，在线与离线B共用尺度计算；验收另用冻结17e5ebb离线B的独立原公式作对照，避免只与自身比较。因子按旧实现所在的实际设备／dtype写回；旧分解在GPU服务器上执行SVD后仍创建CPU低秩模块，随后按原流程迁移，不额外创建随机头。

## 观察、保存与指标

开启`multi_svd_diagnostics=1`时，在私有float64 CPU副本上核验调整前后fc1乘积（atol=1e-6、rtol=1e-5），并记录误差及成本；不是额外模型SVD。0时省去这项矩阵核验，尺度公式不变。初始logits、CE/F与参数量通过独立验收检查，正式流程不为此额外前向。

新增`fc1_tail_scale_downloads.csv/json`和每轮持续`fc1_tail_scale.jsonl`，纳入已有H5／日志ZIP。每客户端每轮记录：

- 请求／实际强度、容量等级及实际fc1秩、最低实际秩。
- 尾部起止槽位、数量、下限活跃槽位数；最低档明确记录跳过原因。
- σ_min、实际因子下限、尾部零奇异值数、零下限与有限处理约定。
- 原模型下发分解次数1、fc1实际SVD次数1、额外SVD次数0；调整回调次数与SVD次数分开命名。
- fc1／整个客户端参数量、下载模型与对应完整服务器指纹、耗时及乘积核验误差。

配置、manifest和H5记录`fc1_tail_scale`策略及统计；B客户端预测checkpoint记录本轮下发凭据，B服务器checkpoint记录**客户端下发策略**，不表示服务器做过尺度操作。旧checkpoint缺该字段意味着旧的尺度0；默认0不向客户端预测checkpoint插入新调整字段。

沿用`--multi_rank_spectrum 1`：默认采样完成训练次数1、10、50和末轮101，观察C0下载分解后、普通post-local后及服务器聚合后的实际矩阵谱。读取`rank_spectrum`中`layer=fc1`，按最低实际秩120区分奇异值平方能量。它是重新排序后的矩阵谱，不是下载时固定槽位乘积变化；不会把尾部能量增大自动解释为分类价值。

主指标仍为全20客户端原分类头、普通post-local的`sum(correct)/sum(test_samples)`，按同一轮总体曲线计算Final、最早Best、Last10／Last20；保留训练前和五档容量指标。不存在C0特殊聚合质量，不做截秩推理；单层截秩仍使用独立离线工具。

## 正式数组与FA01复用

[两条完整COMMANDS](../system/fc1_tail_scale_commands.sh)从服务器`system/`使用，数组只定义、不执行：

1. B（GPU0）：strength=1，优先运行。
2. A（GPU1）：strength=0；若既有FA01严格匹配，可复用并不重复启动。

公共参数为pat20、N20/K1、seed0、原五档容量、low_rank_recovered .15初始化、lr .005、batch16、local5、F .001、clip10、full_avg、原型none/0、gr100实际101轮、每轮全员评价。两组均开启相同原谱诊断。新批次根为`fc1_tail_scale_runs/pat20_seed0_gr100_20261111_v1/`，checkpoint、logs、H5、final_models独立。

复用A前核对完整命令、生效配置、数据／初始化指纹、全员普通post-local评价、实际101轮口径、F／裁剪及采样协议。不能拿FA02原型结果、rank_partition结果或仅C0评价结果当A。若历史A缺谱数据，可复用准确率，但不能补造谱观察或称观察成本一致。

## 合成验收

[验收报告](multi_fc1_tail_scale_acceptance.json)：71项CPU检查中70项通过、1项旧CUDA专用检查跳过；另完成五个20客户端、原宽度模型、两轮真实入口场景。日志和合成产物放在E盘，不占真实数据／正式输出目录。

`tests/test_multi_fc1_tail_scale.py`检查冻结旧模型初始值／RNG、尺度0直接路径、原五档整数秩、同SVD与冻结离线B逐元素一致、恢复矩阵／logits／参数量、零奇异值／下限、下一轮新服务器与新优化器。

`system/verify_multi_fc1_tail_scale.py`通过真实main运行原宽度CNN、20客户端、两轮、每客户端每轮5epoch的合成数据；覆盖B、0、缺省、冻结17e5ebb旧客户端／服务器、诊断关闭。核对120次原层SVD、200次SGD、普通Avg核、全员不等测试样本加权、最早Best和模型重载；最低档8次下发跳过，其他32次调整。合成小数据的SGD步数不是2250张真实数据下的步数，正式预算按实际DataLoader计数。

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:PYTHONPATH='tests'
python -m unittest -v test_multi_fc1_tail_scale test_fc1_tail_diagnostic
python system/verify_multi_fc1_tail_scale.py --output E:/fc1_tail_scale_unique
```

CPU合成验收仅证明实现，不证明真实准确率改善或CUDA逐位复现。应以同轮全员准确率曲线比较A/B，下载／训练／聚合后的尾部谱用于解释；即使本地学习增强，也不能预先声称跨轮必然保留。
