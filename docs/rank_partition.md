# K1 CNN按秩分区归一化聚合

本实现借鉴[raFLoRA论文§5、Algorithm 1](https://arxiv.org/html/2602.13486v1#S5)的按秩分区、对有效贡献者重新归一化思想。论文为Fei Wu、Jia Hu、Geyong Min、Shiqiang Wang的 *Preventing Rank Collapse in Federated Low-Rank Adaptation with Client Heterogeneity*，固定来源版本arXiv:2602.13486v1（2026-02-13）。本次是按用户给定公式独立实现的CNN适配，未复制外部代码。

当前模型没有LoRA或预训练底座：分区对象是整个分解层的训练后权重。服务器保持完整CNN，不增加论文LoRA协议中的服务器低秩因子重构／截断步骤。不能称为完整raFLoRA复现，也不能据此声称已经防止秩坍塌或改善准确率。

## 入口与兼容

新增`--multi_svd_aggregation full_avg/rank_partition`，默认`full_avg`。默认分支继续直接调用原`aggregate_group_updates`，原下载、训练、恢复上传、原型、评价和随机数协议不变。

`rank_partition`首版仅支持：`FedMultiGroupProj --multi_avg_proto_ablation --multi_svd -nsm 1 --multi_group_mode group_avg_only`、全参与、唯一群组为全部客户端、shared head/joint。原型关闭或使用`global_last_epoch + class_sample + raw_mean + base_output + mse`。拒绝`global_pre_train`、K3、Projection、私有头等未验收组合；不替用户改原型时序或聚合策略。新核需要conv2/fc1/fc2真实因子，因此新核／谱诊断不接受ratio=1的dense客户端；旧`full_avg`默认不增加该限制。

## 分区核

实现位于`system/utils/multi_rank_partition.py`。逐层读取实际模块整数`rank`，得到去重边界`0 < r1 < ... < rH`。区间`[left,right)`仅由`rank >= right`的客户端贡献，使用原训练分片样本数`n_i / sum_eligible(n)`加权**各客户端自身的因子乘积**，最后把所有分区矩阵相加。

- 因子直接取普通post-local预测模型的`conv_u/conv_v`或`weight_u/weight_v`；不SVD上传矩阵、不重排因子、不先平均U/V。
- `conv2`沿用原`reconstruct_full_weight/Recover_COV`的排列：`A[out*k,in*k] -> reshape(out,k,in,k).permute(0,2,1,3)`。
- 只替换conv2/fc1/fc2的完整权重矩阵。conv1、fc3、全部bias及其他未分解参数复用原全员Avg结果，只处理一次去重`named_parameters`，兼容base/head别名。
- 相同实际秩的层直接复用旧Avg的结果与计算顺序，包含单客户端情形。日志标记`uniform_rank_legacy_fallback`；其分区计算计时为0表示复用，旧核成本仍计入总聚合耗时。
- 服务器结果不截断。不同客户端训练后方向不一定对齐，分区和的代数秩可以高于最大单客户端秩；下一轮仍由原SVD下载重新排序并截取各容量前缀。

本次继续使用原完整母模型下载和恢复后上传缓存，额外读取驻留的训练后因子；不改造通信格式，不声称低秩通信或内存预算已对齐。

原型RP02保持最新单模型协议：只收最后一个本地epoch已有前向的原始ReLU前特征，类别sum/count按类别样本数合并，下一轮冻结使用，从第二次训练起加入MSE。RP01无原型。模型分区归一化用原客户端总训练样本数，与原型类别计数完全独立。

## 谱诊断

| 参数 | 默认与含义 |
| --- | --- |
| `--multi_rank_spectrum` | `rank_partition`默认开启，旧`full_avg`默认关闭；显式0/1 |
| `--multi_rank_spectrum_rounds` | 省略时完成训练次数1、10、50中在预算内的点，加末轮；显式逗号整数列表也自动包含末轮 |
| `--multi_rank_spectrum_client_id` | 默认0，可配置合法客户端ID，只影响只读诊断 |

非法／重复／超预算抽样轮次报错；关闭诊断时不能传入无效的轮次或非默认诊断客户端。正式gr100会采样第1、10、50、101次训练。

每个抽样点分别读取：C0实际下载／分解后、C0普通post-local后、服务器聚合后。每层用全体客户端实际`rank`的最小值切分**实际矩阵奇异值平方能量**，记录区间内／外能量、总能量、Frobenius范数、矩阵形状和观察时序。卷积谱使用原UV二维排列，不能用另一种卷积展平替代。

诊断只对私有float64 CPU矩阵调用`svdvals`，不修改模型、模式、buffers或梯度；保护Python/NumPy/Torch CPU及实际模型设备的CUDA随机状态。没有额外训练／测试数据遍历。耗时单列，不计为SGD epoch。

`factor_position_block_frobenius_norm`是归一化分区矩阵P_h的范数；`spectral_energy_*`是实际权重矩阵的奇异值平方能量。训练后分区可能不正交，不能把前者平方相加当成后者。当前谱是CNN**权重矩阵**的谱，并非论文LoRA增量的谱。

## 日志、保存与主指标

新增`svd_aggregation_protocol.csv/json`明确记录方式，另有`rank_partitions.csv/json`、`rank_spectrum.csv/json`、`rank_aggregation.csv/json`及逐轮`rank_aggregation.jsonl`，纳入H5和日志ZIP。每分区记录层、边界、实际秩、贡献者、各自原训练样本数、分区质量／权重和、范数及耗时。

原标量`aggregation_weights`只适用于conv1、fc3、bias等非分解参数；分解层必须读`rank_partitions`。旧完整上传的group-reference范数仍只是诊断，不冒充新分区聚合结果。manifest、H5及模型checkpoint均记录`svd_aggregation`实际方式、公式与抽样设置。

主指标不变：各客户端普通post-local模型的测试correct总和／test_samples总和，再对同轮总体曲线取best及最早轮次；保留final、last10/20、逐客户端及下载后训练前指标。服务器聚合后模型不替代主指标。

## 命令

从服务器`system/`目录读取[`rank_partition_commands.sh`](../system/rank_partition_commands.sh)。文件只定义数组，无执行循环。

- `COMMANDS`：RP01（rank_partition，无原型）与RP02（rank_partition，最新类别样本数原型），GPU0/1。
- `OPTIONAL_CONTROL_COMMANDS`：两条匹配的full_avg对照，GPU2/3。按需复制到调度器，不自动重跑已有对照。

四组均为原五档容量、恢复初始化.15、CIFAR-100/pat20、N20/K1、seed0、lr=.005、batch16、每模型5epoch、F=.001、clip10、gr100实际101次训练、逐轮评价。输出根为`rank_partition_runs/pat20_seed0_gr100_20261110_v1/`，四个子目录及日志/checkpoint/H5/final_models各自独立。

复用full_avg结果前应核对初始化／数据指纹、容量、训练预算、F／裁剪和原型协议。历史`client_equal`原型结果不是RP02的直接对照。可选对照显式开启相同谱诊断；旧匹配结果若缺谱记录，可以复用准确率，但不能补造诊断或声称观察成本相同。

## 验收与边界

[验收报告](rank_partition_acceptance.json)：CPU两组回归共61项通过、1项旧CUDA专用检查跳过；另完成7个真实main两轮合成场景。未将此前并行验证的内存不足失败计作通过，也未宣称完成GPU验收。完整日志与合成fixture保留在`E:/rp1110cpu`，大体积临时模型在重载检查后清理。

`tests/test_rank_partition.py`包含不等样本量手算、单贡献者、卷积排列、实际取整、uniform／单客户端精确旧核、bias／别名、输入不变和谱能量。`system/verify_rank_partition.py`通过真实main运行原宽度CNN的N20/K1两轮合成用例，包含两条新路径、诊断关闭、full_avg及冻结7fd0685服务器参照，检查上传／原型／聚合／批次／RNG、导出与21模型重载。

```powershell
$env:PYTHONPATH='tests'
python -m unittest -v test_rank_partition
python system/verify_rank_partition.py --output E:/rank_partition_unique --jobs 1
```

默认串行CPU验收；验收子进程隐藏无关CUDA设备以控制本机内存，正式GPU命令与训练随机数协议不变。所有fixture仅写到新建独立临时目录，不读取或改写真实分片。合成验收仅证明实现，真实准确率收益尚未验收。
