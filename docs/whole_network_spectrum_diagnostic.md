# 四组离线对照：整网容量与聚合谱修复

入口 `system/whole_network_spectrum_diagnostic.py`，不需要检查点或S101。所有模型来自本次实际随机初始化，只有一次模拟模型Avg，随后四组各自独立训练。未修改联邦入口、模型定义、SVD/恢复、聚合或本地损失实现。

## 正式命令

从服务器 `system/` 目录运行（GPU直接改 `-did`，输出目录必须不存在）：

```bash
/data/zh/miniconda3/envs/htfllib/bin/python -u whole_network_spectrum_diagnostic.py -did 0 --device cuda --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --num-clients 20 --class-per-client 20 --client-id 0 --seed 0 --capacity-ratios 0.15,0.25,0.35,0.90 --epochs 5 --batch-size 16 --test-batch-size 16 --lr 0.005 --f-lambda 0.001 --grad-clip 10 --threads 4 --atol 1e-5 --rtol 1e-5 --diagnostics 1 --output-dir whole_network_spectrum_runs/pat20_C0_seed0_local5_20261114_v1
```

[同一条命令的数组文件](../system/whole_network_spectrum_diagnostic_commands.sh)仅定义，不自动启动。CPU可传 `--device cpu`。支持客户端ID、seed、已有数据目录、客户端数/cpc、容量列表及本地超参数。容量列表为至少两档、无重复、有限且在(0,1)，按升序确定性使用：最小档是E3，最大档是母模型和E1/E2/E4。默认4档如下；不将秩比例解释成参数保留比例。

| 层 | .15 | .25 | .35 | .90 |
| --- | ---: | ---: | ---: | ---: |
| conv2 | 12 | 20 | 28 | 72 |
| fc1 | 120 | 200 | 280 | 720 |
| fc2 | 77 | 128 | 179 | 461 |

本工具只读取已有CIFAR100/RGB32/100类pat NPZ。公共reader/auditor核验实际config、客户端数、标签统计、shape/dtype、文件和解码指纹；不依据目录名猜测，不下载/划分/改写数据。审计全分片，SGD只使用所选客户端的原训练集，端点评价使用其原训练/测试集。

## 共同起点与一次模拟Avg

1. 在原CPU seed-fork作用域，用原 `Hyper_CNN_512(ratio_LR=.90)` 构造**一份**真实低秩初始化，保留原始U/V作E4。私有副本调用原 `recover_larger_model()`，规范为完整母模型S0；不重新初始化完整层。
2. 原 `decom_larger_model()` 对S0的conv2/fc1/fc2各做一次SVD，捕获该次P/S/Qh。四档均从同一套方向和高档平衡因子前缀截断，按各层构造函数得到实际整数秩；不会每档重新SVD。
3. 四档分别调用原恢复函数取得完整named_parameters。用原 `aggregate_group_updates(..., group_avg_only)` 对完整参数等权Avg。它们是4个**虚拟容量档上传**，各权重1/4，不是20个真实客户端，也不使用C0数据量作模拟权重。此阶段SGD步数为0。
4. 对Avg结果通过原下发路径再做每层一次SVD，得到E1平衡因子，并捕获该次P/S/Qh。E2复制E1，使用**这同一组P/Qh**、将保留范围的S替换为聚合前实测S，再按原平方根乘法构造两侧因子。没有旧1.5上限，也不拟合/估计原谱。E4始终保留第1步的原始随机因子，未被平衡SVD覆盖。
5. E3在相同seed/原生构造顺序下独立构造.15因子。所有组的conv1、fc3、各层bias和buffers都复制同一份原始大档公共参数；**不复制E3的U/V**。因此E3的因子是原生小档初始化，而共同参数是按实验要求对齐的值，并非整份未经对齐的独立小网。

卷积矩阵保持原 `[out*K,in*K]` 与恢复时的reshape/permute。参数按身份/去重named_parameters处理，模块别名不会重复聚合。默认4档公共参数在Avg后逐元素等于原值；对于可配置的非二次幂档数，纯浮点Avg可能产生末位舍入，统一公共参数时将其复制回共同初值并记录每个参数的实际变化，分解层矩阵保持真实Avg结果。

## 衰减与修复校验

对实际整数秩 `r_i`，位置j的理论衰减为 `count(r_i >= j)/档数`。默认各层四段为1/.75/.5/.25，理想补偿为1、4/3、2、4；若不同容量取整成相同秩，按真实贡献者数量计算，不能套用四段常数。

训练前验证：

- 实际Avg后的逐秩谱，与聚合前实测谱乘上述衰减吻合。
- 实际Avg矩阵，与独立公式 `P diag(S * attenuation) Qh` 吻合。
- E2各层恢复矩阵，与原始E4恢复矩阵接近；记录最大绝对误差、相对最大误差、相对Frobenius误差。
- 所有训练/测试样本的E2/E4初始logits接近，记录相同误差指标和argmax差异数，不强制近似相同值的argmax必须一致。

默认atol/rtol均1e-5，判据明确为“全张量最大误差 ≤ atol+rtol×参考最大幅值，且L2误差 ≤ atol+rtol×参考L2”，不是逐元素allclose。超出时在任何SGD前失败，不静默放宽。未保留秩之外的微小谱能量可能是float32恢复/累加误差，另行标记，不当作有效容量。

新离线进程统一使用float32模型/原SVD/原恢复，CPU构造；仅误差与观察用float64。四组训练都禁用TF32、关闭cuDNN benchmark并设置cuDNN deterministic，设置仅作用于此新进程，写入manifest，不改联邦默认值或安装环境。CUDA命令只暴露指定物理GPU。

## 本地训练与端点

每组全新无动量、无weight decay SGD，直接调用未改的 `utils.single_avg_proto.train`：原CE＋F.001，原全模型裁剪10，joint训练骨干与100类分类头，无原型、Prox、KD、私有头、额外微调。

每组从同一训练RNG快照开始，原私有 `single-avg-proto-v1:seed:loop0:client` loader保证相同batch顺序。核验初始/最终RNG和稳定样本索引哈希，保留实际epoch、batch、访问量、参数量、训练过程CE/加权F/总损失、裁剪前范数和实际裁剪比例。

固定epoch0/1/5（配置其他epoch预算时为0/1/末epoch）的完整train/test纯CE、correct/samples和准确率。epoch0缓存被复用作初始函数检查，未增加第二套数据采样。只读评价保护参数、buffers、模式和RNG；优化器step包装只计数并在异常时恢复。原训练器的临时checkpoint退出后清理，恢复上传不再进入任何服务器聚合。

## 结果文件

默认只保存CSV/JSON/Markdown，无模型或逐样本特征。

| 文件 | 内容 |
| --- | --- |
| `manifest`、`data_manifest` | 请求/实际配置、数据/初始化/模拟上传与聚合矩阵/学生指纹、检查误差、预算与来源 |
| `initial_norms` | 各虚拟容量档、四个学生的实际秩、U/V范数、恢复矩阵范数、bias及参数量；float64观察与原float32恢复乘积分开 |
| `spectra` | 逐秩谱：聚合前、Avg后原谱、E2修复后的**实际矩阵谱**，默认另有E1/E3/E4实际初始谱 |
| `spectral_boundaries` | 每个实际容量边界的相邻sigma、左/右与右/左之比、是否仍在保留范围 |
| `spectral_interval_energy` | 各实际秩区间sigma平方能量、贡献档数、理论衰减与补偿倍率 |
| `repair_gains` | 各保留秩的实测目标、Avg后sigma和实际补偿；gain_cap=null |
| `E1_E2_spectral_gap` | E1/E2实际逐秩sigma差与保留范围内实际倍率，不只列目标曲线 |
| `endpoints`、`local_training`、`batch_order` | 各组端点、原训练统计/裁剪/成本、批次指纹 |
| `comparisons`、`summary.md` | E2−E1与E4−E3的epoch0/1/5训练/测试CE和准确率差；E2−E4作为表示对照 |

准确率用fraction，摘要差值转百分点；CE为nats/sample。谱能量是sigma平方，因子范数是参数范数，两者不互换。`--diagnostics 0`减少可选细节谱，仍执行强制构造验证及端点评价；SGD与RNG不变。

## 验收与解释边界

[验收报告](whole_network_spectrum_diagnostic_acceptance.json)记录CPU数值检查与真实入口合成smoke。可在独立新目录复现：

```powershell
D:/miniconda/envs/htfllib/python.exe system/verify_whole_network_spectrum_diagnostic.py --output E:/whole_network_unique
```

真实pat20不在本机，未执行真实四组结果。合成只证明时序、公式、隔离和导出。E2−E1检验在相同大档参数量下恢复谱的作用；E4−E3检验原生因子初始化、公共参数对齐后的整网增大；E2/E4初始函数近似相同，但因子表示不同，不要求之后SGD一致，也不能把差异都解释成“容量”或跨轮聚合能力。
