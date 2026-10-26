# 离线三模型蒸馏诊断

独立入口：`system/offline_distillation.py`。联邦 `main.py`、服务器、客户端正常训练入口与默认参数均不接入此实验。最终只导出一个学生，不将离线结果反馈服务器。

## 模型与损失

输入为三个同客户端、同完成轮次、同结构和容量的 `ordinary_post_local_low_rank` 预测检查点。直接调用现有 `utils.multi_group_svd.load_checkpoint` 加载低秩状态；训练过程中不调用恢复、SVD、分类头重置。A₀由固定分组/检查点归属声明确定，也可显式给出并核对。`--model-ids` 指定三个文件的模型ID；先重排出A₀，剩余按该列表顺序为B₀、C₀。因此并列规则始终为A₀、B₀、C₀。

|模式|训练目标|教师|
|---|---|---|
|`continue_ce`|CE＋原加权F正则|无蒸馏项|
|`kd_mean`|CE＋原加权F＋λKD·T²·KL(q∥p学生)|B₀/C₀在温度T下的概率均值|
|`kl_dp`|同上|T=1时真实类别概率最大的冻结A₀/B₀/C₀|

KL按类别求和、按batch样本平均；`mean_raw_kl`不含λ或T²，`mean_weighted_kd`包含且只包含一次λ·T²。F正则复用 `clientTargetProj._objective`，未更改其CE/F接口。骨干、低秩因子与fc3均参与训练。每种模式新建同配置SGD。`--kd-lambda 0`在反向传播时直接使用原 `CE+F` 表达式。

KL+DP选择使用原训练集标签，仅在训练集上执行；A₀固定不随学生更新。选择A₀时仍蒸馏其完整类别分布。没有优势阈值、正确性门控、Top-K、EMA、额外正则或测试选教师。

参考：[FantasticGains的main_distillation.py](https://github.com/ExplainableML/FantasticGains/blob/main/main_distillation.py)。此处依据用户指定公式独立实现其逐样本选择/保留机制，并接入本项目CE＋F目标；没有复制其训练框架，也不声称完整复现论文的训练配置。

## 数据、输入与兼容性

可直接读取ZIP而无需解压，也可传已解压目录：

```bash
--bundle ../offline_inputs/C0_three_models_R201_20261025_214112_203491.zip
```

按精确成员 `models/Client_ID_model_K.pt` 读取，不搜索其他检查点。也可使用：

```bash
--checkpoints /path/model_0.pt /path/model_1.pt /path/model_2.pt \
--model-ids 0 1 2 --client-id 0 --assigned-model-id 0 --completed-round 201
```

显式模型文件不含分组文件时，从检查点的固定归属声明核验；可另外传 `--group-file`。检查客户端/模型ID、普通低秩预测角色、完成轮次与loop编号、实际层秩/参数量、模型构造配置、state名称/dtype/shape、别名一致性、有限性、分组指纹及容量协议。互不兼容时在训练前失败。

默认读取 `--data-root ../dataset --dataset Cifar100 --data-subdir pat_20` 下该客户端的 `train/ID.npz`、`test/ID.npz`，或显式 `--train-file/--test-file/--data-config`。复用公共NPZ reader及float32/int64解码，保持顺序，不做增强/归一化/重新划分。首版支持原RGB32、100类CIFAR-100 pat分片。

包内分组有全数据指纹时，训练默认通过现有 `common.client_data.audit_image_clients` 审计原目录的全部分片，与该指纹核对；也可用 `--data-manifest /original/run/data_manifest.json` 提供原审计清单，随后核对实际读取的train/test文件哈希、解码指纹与样本数。原审计清单必须来自这次源实验，不能用另一批数据生成的清单冒充。目录缺失或指纹不符会报错，不合成/寻找替代分片。

旧检查点没有显式语义类别名单：保持原0–99整数标签顺序，不映射类别；manifest明确记录“语义类名不能独立核验”。若检查点/数据有 `class_order`，必须一致；`--class-order`可指定JSON类别顺序声明，但不能补造原检查点中不存在的证据。

只评价原检查点、不训练，可显式使用包内测试集；本模式允许原训练集缺失：

```bash
python offline_distillation.py \
  --bundle ../offline_inputs/C0_three_models_R201_20261025_214112_203491.zip \
  --client-id 0 --assigned-model-id 0 --completed-round 201 \
  --native-test-only --test-from-bundle --device cpu \
  --output-dir offline_distillation_runs/native_R201
```

## 随机流与缓存

教师均eval/无梯度，参数、buffers与模式全程检查不变。教师缓存、额外评价、模型加载受Python/NumPy/Torch CPU/CUDA随机状态保护。评价使用模型私有副本，保护原参数和buffers。

所有准备工作完成后设置一次实验seed，每个独立模式恢复完全相同的训练起始随机状态。DataLoader为shuffle=True、drop_last=False、workers=0，无增强；保存每epoch每batch的稳定原始样本索引及最终随机状态摘要。`--mode all`顺序运行三组；正式三个独立进程也按相同规则初始化，不继承上一组的结束随机状态。

缓存原始logits及固定T=1选择，键包含三个教师检查点/状态指纹、训练数据指纹、预处理、类顺序、设备、Torch版本、backend与前向batch大小；不按温度或KD系数重复缓存。原子发布，校验索引、内容指纹与选择，拒绝损坏缓存。`--verify-cache`额外完整前向复核，固定atol=1e-6、rtol=1e-5并要求选择完全相同。GPU编号属于缓存键，跨设备可能生成独立缓存；额外成本如实计入。

默认使用确定性算法与cuDNN设置，配置和实际backend均写入manifest。可配置 `--deterministic 0`；三组比较必须一致。不更改环境依赖。

## 正式五epoch命令

完整三条命令位于 [offline_distillation_commands.sh](../system/offline_distillation_commands.sh)，默认GPU示例0/1/2。文件只定义数组，不运行训练。在服务器 `system/`：

```bash
# 先将原ZIP放到该路径，或者设置为服务器上已有ZIP的绝对路径。
export DISTILL_BUNDLE=../offline_inputs/C0_three_models_R201_20261025_214112_203491.zip
export DISTILL_DATA_ROOT=../dataset
export DISTILL_RUN_TAG=R201_extra5_seed0_T2_KD1
source offline_distillation_commands.sh
declare -p COMMANDS
```

交给已有调度器时使用此COMMANDS数组。输出目录必须不存在，以免覆盖已完成实验；重复运行更换 `DISTILL_RUN_TAG`。客户端/归属可设置 `DISTILL_CLIENT_ID`、`DISTILL_ASSIGNED_MODEL_ID`；其他轮次/模型ID/路径及全部训练参数均有CLI入口。

三组均额外训练5epoch、batch16、SGD .005/momentum0/weight_decay0、原F系数.001、clip10、T2、λKD1、seed0。未训练A₀单独写baseline，三组epoch0严格核对。主对照固定epoch5，**不按测试准确率选epoch/T/系数**。

全部完成后合并独立运行：

```bash
python compare_offline_distillation.py \
  "${DISTILL_RUN_ROOT}/continue_ce" "${DISTILL_RUN_ROOT}/kd_mean" "${DISTILL_RUN_ROOT}/kl_dp" \
  --output-dir "${DISTILL_RUN_ROOT}/comparison"
```

该工具核对输入/数据/初始化/配置/实际批次/训练RNG/backend一致，拒绝错误配对。`--mode all`入口也自动输出同预算对照表。

## 输出及口径

根目录：`manifest.json`、`baseline.json`、`training.log`、`teacher_cache.json`、`teacher_selection.csv`、`teacher_selection_summary.json`、`comparison.csv/json/md`。

每种模式独立子目录：`metrics.csv/json`（epoch0与每epoch）、`batch_order.json`、`test_predictions.json`、`summary.json`、`manifest.json`、`Student_A.pt`。输入文件只读，最终每组仅一个学生文件；role为`offline_distilled_student`，用现有 `utils.multi_group_svd.load_checkpoint` 按原结构加载，不能把它冒充源联邦ordinary checkpoint。

- endpoint的train/test CE仅为原始交叉熵，准确率内部fraction；对照表转percent，变化单位为百分点。
- 训练平均CE、加权F、rawKL、加权KD及总损失按实际样本访问量加权，独立于endpoint CE。
- 原错后对/原对后错用同一测试集原A₀预测作事后分析，不参与教师选择或优化。
- 保存选择A/B/C互斥数量/比例，同时保存B>A、C>A和外部并集；不会把B/C的非互斥计数相加当作最终选择数量。
- 保存各epoch真实样本访问量、优化步数、去重参数量、训练/评价耗时；teacher_cache另记教师前向次数、耗时、缓存总耗时与字节；模型/输出存储独立记录。continue_ce也生成统一教师诊断，辅助成本单列，不加入其训练损失。
- 原训练样本数若为2250，则5epoch每模式11250次训练样本访问、705个优化步（ceil(2250/16)×5）。不与原联邦201轮混为同一预算。

## 可重复短程验收

```bash
python -m unittest discover -s ../tests -p test_offline_distillation.py -v
python verify_offline_distillation.py --output-dir temp/offline_distill_smoke_NEW --epochs 2
```

第二条使用隔离的合成数据、真实原宽度Hyper_CNN_512、rank .9、100类，实际调用CLI，分别验收正KD与零KD三组、模型重载/对照/输入不变。默认核验后删除自己生成的合成模型与缓存二进制，保留所有可读报告；`--keep-models`可保留。它不读取/改写原训练数据或模型。

验收及真实未训练检查点结果见 [offline_distillation_acceptance.md](offline_distillation_acceptance.md)。合成验收不能证明真实蒸馏有效；本机缺少原pat_20训练分片，未训练真实5epoch，也未验证2250训练图上的655/1595选择计数。
