# 同构完整 CNN 公共对照

> 参数化入口已开放；本文件保留历史默认配置说明。可调参数、数据核验和新pat_10命令见 [CNN512_CONFIGURABLE.md](CNN512_CONFIGURABLE.md)。

这是一套显式启用的工程对照：`--cnn512_control`。三组都使用原有
`CNN-512`（不是 `CNN-5-512`）：3×32×32，卷积16/32、kernel5，
FC 800→2000→512→100，共2,691,860个可训练参数。分类器参与训练、上传、评分和聚合。

`common.models.CNN_512` 是从原 `flcore.trainmodel.models` 移出的同一实现。
原 `BaseHeadCNN/CNN_512` 导入路径仍为别名，可解析旧完整模型checkpoint。
`create_full_model/initial_full_model` 是两算法共同的构造入口；不添加低秩属性，
不调用容量分配、SVD、分解或恢复。完整模型下发使用逐名称/形状校验后复制参数，
保留本地buffers；该CNN实际没有buffers。

## 固定协议

三组均为现有Cifar100/pat_20、20客户端全参与、C0、seed0、SGD .005、
momentum0/weight_decay0、batch16、每模型5 epoch、每轮评价、无学习率衰减。
`-gr 100` 仍执行loop 0–100，共101次训练/聚合。F/Prox关闭、shared head、joint训练；
不增加C0训练、不启用反馈、meta或FedRep。

训练loader：shuffle=True/drop_last=False/workers=0；测试loader：batch16、
shuffle=False/drop_last=False/workers=0。FedFewS部署选择及原train_metrics仍使用原训练集遍历，
不以测试结果选择模型，不额外测试C0。各算法训练/选择遍历数不同，后续批次不要求强行一致。

`--local_grad_clip` 是可选正数，缺省时TargetProj继续为10、原生FedFewS继续不裁剪。
新对照显式固定为10。旧兼容入口 `run_fedfews.py` 和 `fedfews_compat` 未修改，
普通CNN复现默认仍为原测试shuffle和原随机协议。

初始化从CPU seed0构造一份公共模型，复制给服务器及全部客户端；FedFewS的K=3服务器
也由该模型复制。构造公共初值使用隔离的CPU RNG，客户端内部重设种子后，
新对照两算法均将Python/NumPy/Torch CPU/CUDA训练随机流重设为seed0。
不额外更改TF32、cuDNN或确定性backend设置，实际配置记入manifest；CPU验收不代表CUDA逐位确定性。

Avg保持原样本量参数加权计算顺序。ProjectionSoftmax继续调用原完整global-delta核，
C0=.125/helper=.875、投影前余弦、temperature=.2。FedFewS继续K3、smooth_mu=.01，
末epoch过程CE作为聚合损失，训练终点CE用于部署选择，原α/w和增量更新均保留。

**训练预算并不相等**：Avg/Projection每客户端每轮5次数据遍历；FedFewS为3×5=15次。
本次只统一模型及每模型本地协议，不将每模型epoch除以3。

## 启动

在服务器 `FedClip/system/` 目录，使用
[cnn512_control_commands.sh](cnn512_control_commands.sh) 中的完整三项 `declare -a COMMANDS`。
脚本只定义数组，不执行训练。GPU默认0/1/2，可通过CONTROL_GPU_AVG/PROJ/FEW替换。

```bash
# 可先设置已有数据的根目录；它下面必须直接包含 Cifar100/pat_20。
export CONTROL_DATA_ROOT=/data/zh/AllRepo/FedClip/dataset
export CONTROL_RUN_ROOT=/data/zh/AllRepo/FedClip/fedfews_native_runs/cnn512_control/run01
source cnn512_control_commands.sh
# 将 COMMANDS 交给现有调度器；本文件和脚本均不自动启动。
```

每组分别使用 `avg/`、`projection/`、`fedfews/` 下独立的checkpoints/logs/h5/final_models。
数据显式定位，npz_only，不搜索其他分片、不生成/改写数据。`data_manifest.json`保存config、
实际文件路径/SHA256、解码后x/y指纹、dtype、样本量和标签分布。
`comparison_manifest.json`保存初始模型SHA256、参数形状、各初始副本哈希、有效配置和环境。
各组初始哈希和完整数据manifest应相同（算法配置本身则不同）。

## 主指标和文件

统一主指标为C0 ordinary post-local：Avg/Projection为普通5epoch后的checkpoint；
FedFewS为K模型各5epoch后按训练终点CE选择的post-local模型。FedFewS直接提取已采集的C0计数。

- `logs/train.log`、`logs/run_<id>/`、旁边的 `logs.zip`：可单独下载，不含模型。
- `c0_post_local.csv/json`：loop_round、round/完成次数、correct/samples、百分比准确率。
- `c0_summary.json`：final、最早best及loop/完成次数、last10/last20和实际点数；单位percent。
- `local_training.csv/json`：每客户端每轮CE、每模型local_epochs、model_count、
  processed_train_samples（实际样本访问量）、train_batches（每batch恰一次SGD，即优化步数）、耗时。
- H5的 `cnn512_control`：相同C0序列、汇总和training诊断；`initial_model.pt`保存在模型目录，
  包含结构参数和初始state_dict。TargetProj的新对照通用rs_test_acc记录C0，不再额外进行旧的全员评价；
  原模式默认路径的全员评价不受影响。
- 原Projection权重/余弦/冲突/删除诊断照常导出。FedFewS仍保留全体样本量加权指标、逐客户端结果、
  两类CE、模型选择、α/w、接收质量和服务器间距离；其原 `summary.json` 是**全体**指标，
  比较C0须读取 `c0_summary.json`。

TargetProj最终 `Server_model.pt/Client_i_model.pt` 为可直接加载的完整CNN模型。
FedFewS最终 `FedFewS_final.pt` 含实际训练的K服务器、每客户端K模型及部署索引和优化器状态；
`model/model_family/num_classes`反映真实CNN。可用
`common.models.create_full_model(SimpleNamespace(model_family='CNN-512',num_classes=100))`
创建结构，再加载bundle中指定state_dict；不要将初始模板作为训练后的模型。

## 短程验收

在现有环境、仓库system目录：

```bash
# 只在全新output目录生成隔离的合成fixture，不触碰正式dataset。
python verify_cnn512_control.py --output ../fedfews_native_runs/cnn512_accept_cpu --device cpu
# GPU上也可运行相同的入口/导出smoke；不作为CPU逐位误差验收替代。
python verify_cnn512_control.py --output ../fedfews_native_runs/cnn512_accept_cuda --device cuda --device-id 0
```

三真实main入口各执行两轮、20人、每模型5epoch，合成样本量非均衡，校验初始模型/数据一致、
无低秩调用、真实训练预算、C0采集、CSV/JSON/H5、最终模型重新加载。
数值公式、原构造/旧checkpoint兼容、同批次且实际触发裁剪的逐元素SGD一致性在
`tests/test_cnn512_control.py`验收。完整CPU报告见仓库根目录
`CNN512_CONTROL_ACCEPTANCE.md`。

本机没有真实pat_20分片，未报告真实准确率或CUDA精确等价；不据合成smoke判断效果。
