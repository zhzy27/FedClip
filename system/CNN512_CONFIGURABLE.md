# 可配置完整CNN比较入口

`--cnn512_control`现在表示共同的完整模型协议，而非固定pat20/20人/seed0/local5。
旧命令、默认模型参数、聚合公式和seed0训练轨迹保持兼容。旧的两份control/population命令文件保留。

当前支持Cifar100、100类、RGB 3×32×32、同构完整`CNN-512`。不使用SVD、容量分配或恢复。
保持全参与、fresh run、shared head、joint、CE-only、SGD momentum0/weight_decay0。
部分参与/drop/slow、resume、early-stop、学习率衰减、F/Prox、meta/head诊断/反馈仍明确拒绝，
因为当前比较入口尚未实现这些协议的状态与预算对齐。

## 参数

| 参数 | 生效规则 |
|---|---|
| `--data-root/-data/--data-subdir` | 精确定位已有分片；不按目录名猜划分、不生成/转换数据 |
| `--data-formats` | 缺省比较协议npz_only，也可显式指定pkl_first/pkl_only；两算法共用同一解码/选择策略 |
| `-pt pat/dir` | 与真实config.json一致；显式`--data-expected-partition`不能与之冲突 |
| `-nc` | 正整数，必须匹配真实config.num_clients |
| `-cpc` | pat下为1–100，校验元数据和实际标签分配；dir下不作固定类别数约束 |
| `--data-expected-alpha` | 可选有限正数，必须等于config实际alpha；不依据文件夹中的a0p5等文字判断 |
| `--seed` | 0到2**32-1；控制公共初始化及Python/NumPy/Torch CPU/CUDA训练随机流 |
| `--target_client_id` | 0到nc-1；用于单目标参考/自身质量及可选辅助计数，不能改变Few算法目标 |
| `-lr` | 有限且大于0 |
| `-lbs/-ls` | 正整数；ls是每模型epoch |
| `-gr` | 非负整数；循环0..gr，实际gr+1次训练/聚合 |
| `-eg` | 正整数，可大于gr；loop%eg==0或末轮评价，三算法一致 |
| `--local_grad_clip` | 有限非负；**0明确关闭**；正数为总梯度范数上限 |
| `--control_test_batch_size` | 正整数，缺省16；测试shuffle=False/drop_last=False/workers0 |
| `-nsm/-smu` | Few K正整数，smooth_mu有限正数；不更改双层权重或模型选择公式 |
| `--projection_self_weight` | 单目标ProjectionSoftmax自身质量[0,1]；省略沿原训练样本占比 |
| `--target_group_ids` | 群组模式有效非空、无重复、ID不越界，规范化排序；不必包含目标客户端 |
| `--target_group_weight` | 群内质量[0,1]；省略按群内训练样本占全体的比例；正群外质量但无群外成员时报错 |
| `--projection_softmax_tau` | 独立的单目标helper cosine softmax除数，有限正数，缺省.2 |
| `--group_softmax_tau` | 独立的群外helper cosine softmax除数，有限正数，缺省.2 |
| `--control_eval_scope c0/all/group` | 缺省c0为兼容名称，实际读取target_client_id；all为全员，group为声明群组，均按测试样本量加权 |
| `--control_preview` | 审计指定数据并打印请求/生效设置、预算及评价轮次，不构造模型、不训练 |

两个tau参数不复用其他算法`--temperature`。非默认tau只允许用于相应cosine softmax模式，
不会被其他模式静默忽略。与原实现同样的稳定softmax、投影系数/epsilon、global-delta和累加顺序保留。

裁剪缺省语义：比较模式省略仍为10（包含Few），非比较TargetProj省略仍为10，
非比较原生FedFewS省略仍不裁剪。旧兼容入口`run_fedfews.py`没有改变。
关闭时JSON的`gradient_norm_cap=null`，H5使用cap=0且`gradient_clipping_enabled=False`，不表示把梯度裁为零。
比较模式的Few强制使用`--seed`对应的controlled协议；原非比较original随机协议不变。
客户端内部seed0保留以兼容旧生命周期，但公共模型独立按seed生成、构造完客户端后再次重置全部训练随机流。
相同seed/结构的不同算法初始参数和训练初始随机状态相同；改变seed会改变二者。

## 数据核验能力

比较数据必须提供config.json中的num_clients、num_classes和pat/dir划分类型。
每个实际读取文件保存路径、格式、SHA256、解码后x/y指纹、dtype、样本量、标签统计。

pat先核对config.class_per_client（若存在），再核对每客户端train+test的实际标签集合：
超过cpc直接报错；若配置含`Size of samples for labels in clients`，逐项核对实际合并计数。
缺少分配元数据、实际观察类别少于cpc时，标为`partial`并说明不能证明未观测的分配，
不把文件夹名当证据。dir记录真实alpha，不对每人类别数作pat检查。
不匹配时只报错，保留已有数据，不下载、重划分或改写。
比较模式Dir输出路径中的alpha标签也取真实审计值，不沿用无关的旧`-dir_alpha`默认值。

## 评价和导出

Avg/Few通常使用`--control_eval_scope all`：普通本地训练后全员correct之和除以test_samples之和。
Few仍按训练集终点CE选择部署模型，不按测试成绩选模型。群组同样对声明成员求加权准确率。
比较不同评价人群时不能称为同一对象；群组结果仍仅为阶段性代理。

`primary_metrics.csv/json`保存已评价轮次，`primary_summary.json`报告Final、同轮平均序列的Best、
最早Best loop/完成次数、末10/20**次评价**及对应实际loop列表。末轮始终评价。
未评价轮也正常训练/聚合，不读取旧准确率填充新轮。群组每轮权重/投影诊断保留在CSV/JSON，
H5的`target_group/all_aggregation_rounds`与已评价成员数据分轴保存。

`local_training.*`按每轮每客户端记录实际访问量、batch/optimizer_steps、耗时、每模型epoch和模型数。
Avg/群组每人遍历local_epochs次，Few每人K×local_epochs次，预算差异如实保留。
manifest/H5保存请求值与生效值；从真实数据解析的缺省自身/群内质量另记为生效值，
不改写原算法中None参数的缺省计算路径。最终轮数/评价次数/总访问量/总步数从实际记录归约。

target=0时兼容`c0_*`；非零目标写`target_<id>_*`，不会伪装成C0。
目标不在群组内时不额外测试它，`target_auxiliary.json`明确标记未观测，不混用其他人的计数。
主指标CSV/JSON/H5及最终manifest一致；分类器共享并包含在训练/聚合/保存中。

## 本批pat_10正式命令

从服务器system目录使用`cnn512_pat10_commands.sh`的完整两项`declare -a COMMANDS`。
20人、Cifar100/pat_10、cpc10、完整CNN-512、seed0、lr.005、batch16、local5、gr100=101次、
clip10、每轮全员评价；Few K3/.01。输出avg与fedfews独立，GPU编号可替换。

```bash
export PAT10_DATA_ROOT=/data/zh/AllRepo/FedClip/dataset
export PAT10_RUN_ROOT=/data/zh/AllRepo/FedClip/fedfews_native_runs/pat10/run01
source cnn512_pat10_commands.sh
# COMMANDS仅定义，不会启动。PREVIEW_COMMANDS为同配置的无训练审计入口。
```

注意这是**20客户端、每人10类**的本仓库pat_10，不是此前FedFew原复现用的10客户端pat10。

## 短程复验

```bash
# system目录，输出必须全新；仅隔离合成fixture，--compact减少重复导出模型占用。
python verify_configurable_control.py --case default --reference-commit 7e236dd --compact --output ../fedfews_native_runs/check_default
python verify_configurable_control.py --case pat10 --compact --output ../fedfews_native_runs/check_pat10
python verify_configurable_control.py --case custom --compact --output ../fedfews_native_runs/check_custom
python verify_configurable_control.py --case dir --compact --output ../fedfews_native_runs/check_dir
```

default从Git导出不可修改的旧入口，比较逐本地/聚合参数哈希、实际batch和RNG轨迹。
其他case覆盖非默认seed/N/lr/batch/epoch/K/mu/target/tau/质量、关闭裁剪、稀疏及末轮评价、
不等测试样本量、输出与模型重载。合成验收不表示真实准确率改善。
