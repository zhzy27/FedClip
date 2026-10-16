# FedFew公共模块接入验收（基于189bacd，2026-10-16）

公共化代码已完成；CPU和旧FedTargetProj路径验收通过。**本机默认CUDA配置未通过，
外部原版自身重放也未通过；cuDNN确定性诊断通过，但不能替代默认配置验收。**
没有真实α=.5分片，本次没有启动真实短程或正式长实验，没有修改依赖环境或推送。

## 公共实现与保留差异

| 模块 | 接入内容 | 保留的调用方语义 |
|---|---|---|
| `common/data_io.py` | 明确文件解析、图像float32/标签int64解码 | FedClip原根目录/partition/alpha/cpc、仅npz；FedFew独立根目录、pkl优先/npz回退 |
| `common/data_loading.py` | 可注入loader类的构造工厂 | FedClip测试shuffle=False；FedFew测试shuffle=True；原batch、drop_last=False、workers=0、无独立Generator |
| `common/models.py` | 唯一FedAvgCNN类体，仅依赖Torch | 默认值、参数名/顺序、构造顺序、初始化、forward不变 |
| `common/metrics.py` | 已采集correct/samples及评价序列的纯计算 | 准确率fraction；汇总声明fraction/percent；FedFew保留percent尾部均值的原计算顺序 |

列表图像保留两种显式策略：FedClip仍用原Tensor(list)，FedFew仍用原
stack(from_numpy)。两边文本/Shakespeare与few-shot分支未公共化或改写。
缓存仍由调用方控制。审计和迁移训练使用同一个选定文件解析/解码路径。
没有新增增强、归一化、排序、分区搜索或数据划分。

`flcore.trainmodel.models.FedAvgCNN`及`fedfews_compat.vendor.models.FedAvgCNN`
是公共类的兼容导出。已实际保存旧类路径的完整模型checkpoint，然后在恢复新别名后
成功加载并核对forward；新公共路径也可加载。FedFew导入小CNN不会导入整个模型库或
torchvision。许可证保存在`common/LICENSE.fedfew`，来源和适配详见
[`PROVENANCE.md`](../system/fedfews_compat/PROVENANCE.md)。

算法专属文件`clientfedfews.py`、`serverfedfews.py`、FedTargetProj客户端/服务器算法
文件及`reference.py`未修改。K模型/K优化器、两类CE、部署选择、双层权重、增量聚合、
RNG协议及内部种子重设不变。旧FedTargetProj的算法专属指标汇总暂未替换。

## 参照与环境

外部参照仍为`D:\All_Git_Repo\FedFew_reference`，提交
`1b29dec70b66d89b956d90e0da024897f698897a`，工作区干净。它继续执行自己的reader、
CNN、基类loader及算法，不经过公共reader/model/loader。测试会核对参照函数来源，
并在公共实现被替换为报错函数时执行参照端的读取、训练、选择、测试和train_metrics。
导出观察器可共用，但总体准确率会与外部算法独立计算的`rs_test_acc`逐值核对。

本机使用既有`D:\miniconda\envs\htfllib\python.exe`：Python3.11.14，
PyTorch2.0.1+cu117、CUDA11.7、cuDNN8500、NumPy1.26.4、SciPy1.16.3、
torchvision0.15.2+cu117、scikit-learn1.7.2、h5py3.15.1、PyYAML6.0.3。
GPU是RTX3070 Ti Laptop。没有安装、升级或复制环境。

默认测量设置：deterministic_algorithms=False、cudnn.deterministic=False、
cudnn.benchmark=False、matmul TF32=False、cuDNN TF32=True；两端intra-op线程2。
不是对另一台服务器/软件栈的验收。

## 外部原代码与迁移端短程对照

全部使用实际普通FedAvgCNN、100类、K=3、SGD .005、每模型1 epoch。
合成图像包含数组和numpy数组列表，训练样本数为`7+3*i`、测试为`3+3*i`。
部分客户端同时放置有效pkl与内容不同的旧npz，验证格式策略。合成config的alpha为null，
不冒充真实Dirichlet分片。未改写任何用户真实数据。

| 配置 | 客户端/实际训练次数 | 观察点/比较参数元素 | 最大参数绝对误差 | 结果 |
|---|---|---|---:|---|
| CPU，原版→公共化 | 20 / 4 | 705 / 235,800,540 | 0 | PASS，atol=rtol=0 |
| 默认CUDA，原版→公共化 | 20 / 3 | 499 / 177,543,936 | 3.166906535625458e-4 | FAIL |
| 默认CUDA，原版→原版重放 | 20 / 3 | 499 / 177,543,936 | 3.2417895272374153e-4 | FAIL |
| cuDNN确定性诊断，原版→公共化 | 20 / 3 | 499 / 177,543,936 | 0 | PASS，仅诊断配置 |

CPU最大标量误差也为0。覆盖每客户端/K模型训练后参数、聚合用训练过程CE、部署终点CE、
α/w、模型选择、测试正确数/样本数、服务器参数、实际批次和RNG。评价loop为0/2/3，
loop1不评价，loop3验证末轮强制评价。

所有CUDA检查保持原atol=1e-6、rtol=1e-5，**没有放宽**。评价loop0/2，loop1不评价。
默认CUDA首次记录到的超差是event27，第0轮Client17的Model2训练结束后：
conv1.weight约2.31508e-5、conv2.weight约7.94e-5。该时点记录的聚合损失仍相等，
初始化、数据指纹、批次和全部RNG哈希一致。后续出现损失、权重及离散部署选择差异。
这里的首次观测是客户端训练端点，尚未定位到具体minibatch或CUDA算子。

默认CUDA最大参数差位置为event350、`model_2.fc1.0.weight`平坦索引30367；原版重放
最大差位置为event370、同名参数索引126353。默认两次检查的最大标量差3.7037037来自
百分比准确率字段，不应解释成CE误差。

原版自身同样失败，说明公共化不是产生该偏差的必要条件。仅对两个诊断端同时设置
`cudnn.deterministic=True`后，参数及标量误差均为0，支持默认cuDNN执行路径重复性是
重要原因；尚未确认具体kernel或完整误差传播机制。**这不是默认CUDA已经通过的结论。**

工具新增`--reference-repeat`，报告明确标为原版自身重复性；新增
`--diagnostic-cudnn-deterministic`仅用于短程诊断，报告的`acceptance_scope`明确标记，
进程内设置在finally中恢复。正式入口不接受此诊断参数，默认backend没有切换。
超差返回非零；首个参数超差的双端状态另存`first_parameter_mismatch.pt`。

## 公共部件与旧路径回归

最终公共部件8项测试通过：普通npz/pkl/两格式并存、列表/数组、原文件SHA与解码顺序/dtype、
FedClip不读pkl、不跨分片回退、few-shot及文本分支、缓存开关、四阶段loader批次与RNG、
Recorder注入、CNN初值/forward/构造后RNG与旧checkpoint、纯指标/非法输入及诊断设置恢复。
FedFew既有9项专项继续通过。

专门的FedTargetProj回归从Git `189bacd`加载旧reader、Client/Server基类及算法类，
对照公共化后的代码。使用真正的临时npz文件和内容不同的pkl干扰文件，20客户端、
非均衡7/8/9训练样本、异构低秩CNN、每轮5 epoch，分别执行Avg和12.5% ProjectionSoftmax
各两轮。每轮上传与服务器各state_dict张量的SHA256、训练样本权重、实际batch索引、
Python/NumPy/Torch CPU/CUDA状态和最终指标汇总全部一致。额外采样器观察不重复遍历。

该测试首版曾因第一次尚未初始化CUDA RNG、第二次已初始化而使状态表示不同；在两端
开始前统一初始化可见CUDA RNG后通过。只调整了测试准备，没有修改生产训练代码。
所选混合回归套件151项通过（包含当时7项公共测试及9项FedFew测试）；最后新增的
诊断设置恢复测试在公共8项专项中另行通过。正常入口两轮smoke也完成CSV/JSON/H5、
公共模型K状态、manifest与RNG导出；预览未创建训练输出目录。

## 服务器下一步与命令

从FedClip根目录使用已有环境，设置实际路径和新输出根目录：

```bash
export FEW_PYTHON=python
export FEW_SOURCE_ROOT=/data/zh/AllRepo/FedFew_reference
export FEW_DATA_ROOT=/data/zh/AllRepo/FedFew/PFLlib/dataset
export FEW_OUTPUT_ROOT=/data/zh/AllRepo/FedClip/fedfews_runs/shared_20261016
source system/fedfews_commands.sh
eval "${PREVIEW_COMMANDS[0]}"
eval "${REAL_ACCEPTANCE_COMMANDS[0]}"  # 实际alpha=.5、20人、3次训练，默认GPU设置
```

预览打印实际数据目录、选中文件/哈希、格式策略、公共reader/model、四阶段loader参数。
严格CPU替代入口是`REAL_CPU_ACCEPTANCE_COMMANDS`，保持相同数据/训练参数、容差0。
若GPU失败，先用实际对应配置检查原版重放，不要扩大容差或拿确定性诊断替代默认验收。
`REFERENCE_REPEAT_COMMANDS`与`DIAGNOSTIC_CUDA_ACCEPTANCE_COMMANDS`提供合成诊断入口。

**当前可进入服务器真实分片短程验收；不建议直接开始默认GPU长实验。**
用户在计划运行的实际数据、环境与backend设置下确认通过后，才手动执行
`COMMANDS[0]`：完整参数仍是20人、100类、CNN、K3、μ=.01、lr=.005、batch50、
每模型1 epoch、全参与、eval_gap10、gr2000（实际2001次），无drop/slow/few-shot/衰减。
完整数组及可选pat10入口在[`system/fedfews_commands.sh`](../system/fedfews_commands.sh)。

本地机器可读报告和日志副本位于`fedfews_runs/shared_validation_20261016/`（不提交），
原始大张量诊断位于`D:\temp\fedfews_shared_*_20261016`；正常smoke输出在
`fedfews_runs/shared_entry_20261016/`。训练日志仍与模型分目录保存。

绘图文件SHA256仍为`A066FD21F6DF5DADC2DF1BBAF31C247D7C8326348C36F793B3B1FB7C0A175E8A`，
不纳入本次提交。公共化没有建立普通CNN与异构低秩CNN之间的公平性能比较：数据协议、
训练预算及评价对象的统一仍留待下一阶段，本次不据此判断准确率优劣。
