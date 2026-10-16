# FedFewS 原生基线入口

基于 FedClip `c73ec5b`，提供 `system/main.py -algo FedFewS`。这是原版标准
FedFewS 的工程接入，使用普通 CNN，不含低秩适配，也不安排低秩版本。
`system/run_fedfews.py` 及整个兼容模块未改动，已有长实验继续使用原路径。

## 算法与公共设施

* 新增 `flcore/clients/clientfedfews.py` 和 `flcore/servers/serverfedfews.py`，分别直接
  继承仓库 Client/Server，不继承或调用 vendor 的训练实现。
* 模型明确选择 `-m FedAvgCNN-plain`，构造 `common.models.FedAvgCNN(3,100,1600)`，
  924,708 参数。旧 `FedavgCNN` 的 Decom 映射不变。单个公共初始化 CNN 深拷贝为 K 个服务器；
  客户端仍保留源代码的 `torch.manual_seed(0)` 和 K 个模型/K 个 SGD/调度器的生命周期。
* CE 是唯一训练目标。SGD momentum=0、weight_decay=0；不调用 F、裁剪、Prox、SVD、
  Projection、私有头或 meta。`smooth_mu` 独立于旧 `mu`。
* 聚合损失是最后一 epoch 中的样本加权 batch CE；部署选择另在 eval 模式遍历训练集，
  以终点 CE 最小的模型测试。没有按测试准确率选择或集成。
* 全参与、固定 K 顺序训练；先选模型，全部测试，再全部 train_metrics，然后接收、聚合。
  NumPy 选择顺序与 Python random 接收顺序保留。上传 IDs 对齐损失、模型和样本权重。
* 损失先乘一次样本占比，再执行源 NumPy/SciPy logsumexp 双层权重。全部差分先相对旧服务器
  计算，再按原接收顺序 `add_`；没有按模型重新归一化质量或改成完整模型 Avg。

基类仅新增窄钩子：Client 可显式接收一个初始内存模型；Server 可提供客户端构造参数和
结果文件名前缀。默认分支不变。FedFewS 避免 Model_Distribe、隐含随机模型初始化及单模型
checkpoint 生命周期；其 K 模型保持在内存，Parameter 对象不因下载而替换。

FedFewS 不采用 CLIP 的默认线程控制；可用 `--fedfews_torch_threads` 明确指定，否则保留
环境默认值。主入口先完成原生类/基类导入，再建立该算法 RNG 状态，避免惰性导入插在
种子设置与 CNN 初始化之间。旧算法的线程、种子和训练分支保持原样。

来源：pgg3/FedFew `1b29dec70b66d89b956d90e0da024897f698897a`，MIT，copyright Ping Guo。
完整许可证为 `system/common/LICENSE.fedfew`。工程行为的固定参照是 FedClip `c73ec5b`
的兼容实现。新的原生实现有适配记录和数值对照，不声称文件体与原仓库逐字相同。

## 参数及数据

在 `system` 目录运行。核心新增参数：

```text
-algo FedFewS -m FedAvgCNN-plain -nsm 3 -smu .01
--data-root /existing/FedFew/PFLlib/dataset
-data Cifar100/noniid_dir_20_a0p5
--data-formats pkl_first
--data-expected-alpha .1 --data-expected-partition dir
--fedfews_rng original|controlled
--fedfews_control_seed 0
--fedfews_torch_threads 2                 # 可选；正式数组不强加线程数
--fedfews_test_batch_size 256             # 可选；带 suffix 默认256
--fedfews_no_cache                        # 可选；默认沿用原版缓存
--fedfews_preview                         # 只读审计，不构造/训练模型
```

`-nsm/--num_server_models` 默认3；`-smu/--smooth_mu` 默认.01。
首次支持标准模式、CIFAR-100、100类、全参与、无 drop/slow/early-stop/decay/resume。
不支持的协议明确拒绝，Rep 参数不开放。共享配置中的旧 `mu` 不控制本算法。

也可将数据字符串写为 `-data Cifar100 --data-subdir noniid_dir_20_a0p5`；不要同时在
`-data` 和 `--data-subdir` 重复给 suffix。格式可明确指定 `pkl_first / npz_only / pkl_only`。
未指定格式的 FedFewS 默认 pkl 优先、npz 回退；不搜索其他分片或大小写变体。
旧算法不传 data-root 时仍走原路径/原 npz 策略。

服务端统计、客户端缓存/训练/评价和审计统一调用 `utils.data_utils` 的显式数据适配及
`common.client_data`，底层共用 `common.data_io`。源 FedFew 列表图像的 numpy_stack
解码策略保留，float32 x/int64 y、不重排、不变换。config.json 原始字节、实际 alpha、
每文件 SHA256、解码内容顺序哈希、样本数和标签统计保存到日志。

**本次 Dirichlet 的真实 alpha 按用户指定为 .1，目录名字仍为 a0p5。** 正式数组显式
要求 config.json 中 alpha=.1，不根据名字或旧生成器常量猜测。不修改兼容入口已有的
命令/配置，不修改任何真实分片。pat10 单独要求 partition=pat 和10客户端，记录实际 alpha。

四类 loader 均沿用源 shuffle=True、drop_last=False、workers=0、全局 Torch RNG，无
额外 Generator 或 sampler 遍历。训练/部署选择用 -lbs，测试/train_metrics 对带 suffix
的数据默认256。`K=3, local_epochs=1` 表示每客户端每轮数据遍历3次。

original：与兼容入口一样，主进程 Torch 重设为0；不新设 Python/NumPy 种子；每次客户端
构造内部 Torch 重设0仍保留。controlled：先固定 Python/NumPy/Torch 到指定 control seed，
再按原规则 Torch 重设0，保留客户端内部重设。它是受控比较，不声称重放未记录 RNG 的历史曲线。
不修改 cuDNN/TF32/确定性 flags。初始化前 RNG 和结束 RNG 分别保存。

## 正式命令与短程验收

完整两项 `declare -a COMMANDS`、预览和真实验收数组见
[`system/fedfews_native_commands.sh`](../system/fedfews_native_commands.sh)。与旧兼容命令文件
分开。两项都是 `-gr 2000`，实际2001次训练/聚合；普通 CNN、K3、每模型1 epoch、batch50、
lr .005、全参与、eval_gap10、最后强制评价、times1/prev0。GPU示例2/3可替换。

```bash
cd /data/zh/AllRepo/FedClip/system
export FEW_NATIVE_PYTHON=python
export FEW_NATIVE_DATA_ROOT=/data/zh/AllRepo/FedFew/PFLlib/dataset
export FEW_NATIVE_RUN_ROOT=/data/zh/AllRepo/FedClip/fedfews_native_runs/check_01
export FEW_NATIVE_GPU_DIR=2
export FEW_NATIVE_GPU_PAT=3
source fedfews_native_commands.sh

eval "${PREVIEW_COMMANDS[0]}"             # 真实 .1 数据只读核对
eval "${PREVIEW_COMMANDS[1]}"             # pat10 只读核对
eval "${SYNTHETIC_ACCEPTANCE_COMMANDS[0]}" # 20人合成 CPU 对照
eval "${REAL_CPU_ACCEPTANCE_COMMANDS[0]}"  # 真实 .1 数据3次训练
eval "${REAL_CPU_ACCEPTANCE_COMMANDS[1]}"  # 真实 pat10 数据3次训练
# 计划使用 CUDA 时，在相同服务器/数据上另外验收：
eval "${REAL_CUDA_ACCEPTANCE_COMMANDS[0]}"
eval "${REAL_CUDA_ACCEPTANCE_COMMANDS[1]}"
```

数组只定义命令，不启动训练。检查报告后，长实验由用户/原调度器选择 `COMMANDS[0]` 或
`COMMANDS[1]`。正式训练不需要外部 FedFew、source-root 或 YAML。验收工具需要本仓库 Git
历史中的 c73ec5b，以 `git archive` 导出固定代码；请保留该提交，不使用缺历史的源码压缩包。

`verify_fedfews_native.py` 在独立进程运行固定兼容参照与真正 `main.py`。参照使用导出的
reader/common/CNN/vendor 算法，不引用当前原生训练类；原生端仅由验收工具安装冻结的观察器，
观察真实 sampler，不回放 loader。失败返回非零，生成 FAIL/ERROR 报告及必要的首次差异状态。
`--reference-repeat` 是 c73 自身重放诊断，报告明确写非原生验收。
CPU默认 atol=rtol=0；CUDA数组沿用1e-6/1e-5，未放宽。真实验收详见
[本机报告](fedfews_native_acceptance_20261016.md)，本机默认 CUDA 尚未通过，不能忽略失败。

## 指标、目录及模型

主指标始终是选中普通 post-local 模型后的全客户端测试样本量加权 fraction。
CSV/JSON summary 用百分比。final、最早 best、last10/last20 都注明实际评价点数；
eval_gap10 时尾部是最后10/20次评价，不是训练轮次。
通用 average_data 不再处理 FedFewS，避免混淆评价对象。

每个配置的独立根目录按命令提供：

```text
<config>/logs/train.log                         全过程控制台
<config>/logs/run_<run_id>/manifest.json         实际配置/环境/来源/RNG/预算/路径
<config>/logs/run_<run_id>/data_manifest.json    文件与解码指纹、真实alpha
<config>/logs/run_<run_id>/data_config.json      原config.json字节
<config>/logs/run_<run_id>/rounds.json[l]        全轮权重、顺序、服务器间距离/收到质量
<config>/logs/run_<run_id>/clients.csv           两类CE、逐客户端权重、选择及correct/samples
<config>/logs/run_<run_id>/evaluations.{csv,json}
<config>/logs/run_<run_id>/summary.json          主指标汇总，单位和评价对象明确
<config>/logs/run_<run_id>/*.h5                  公共H5的日志镜像及FedFew详细数据
<config>/logs.zip                               公共日志ZIP，不含模型
<config>/h5/<structured path>/*.h5               原始H5
<config>/checkpoints/<dataset>/FedFewS/<run_id>/FedFewS_final.pt
<config>/checkpoints/<dataset>/FedFewS/<run_id>/FedFewS_rng_before_initialization.pt
<config>/final_models/<structured path>/runs/<run_id>/
  FedFewS_final.pt, manifest.json, fedfews_manifest.json, RNG文件
```

逐客户端同时记录 epochs_per_model、model_count、data_epochs、train_sample_visits、
train_batch_count。H5含两类CE、α/w、实际更新权重、收到总质量、上传IDs、选择分布和测试计数。
`rounds.json` 让逐轮距离诊断也进入公共 ZIP。没有为提取C0/七人组再测试：
`utils.fedfews_reporting.collected_accuracy(evaluation, [0,1,2,3,4,6,7])` 直接计算已收集计数，
返回fraction，可使用保存的 evaluations.json。

`FedFewS_final.pt` 是一个有明确 K 索引的 bundle，含 K 个实际聚合服务器、每客户端 K 个
普通 post-local 模型、部署索引、K优化器/调度器和结束 RNG；不是未参与训练的单模板。
部署模型例子：

```python
state = torch.load(path, map_location="cpu", weights_only=False)
client = state["clients"][0]
model = FedAvgCNN(in_features=3, num_classes=100, dim=1600)
model.load_state_dict(client["models"][client["best_model_idx"]])
model.eval()
```

不支持自动 resume；保存完整状态供审计，不静默用不完整状态续训。每个进程使用独立日志和
模型根，重复实验内部也有独立 run_id。绘图改动不纳入此实现提交。
