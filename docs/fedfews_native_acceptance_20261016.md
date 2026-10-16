# FedFewS 原生接入验收，2026-10-16

基于 `c73ec5b` 完成原版普通 CNN FedFewS 原生入口。**CPU短程通过；本机默认CUDA未通过，
固定兼容版自身CUDA重放也未通过。** 不将后者写成原生已通过CUDA验收。服务器真实分片
仍需复验。本次未改环境、未重新生成真实数据、未推送、未启动正式长实验。

## 固定参照与工程差异

参照严格为 Git 提交 `c73ec5bcbc6a603915957f45a5f20343c516274f` 的
`system/fedfews_compat` 和 `system/common`，由验收工具 `git archive` 导出到每次独立输出。
不是使用当前工作区兼容算法与自身比较，不需要外部FedFew仓库。当前兼容入口及模块没有改动。
验收工具新导出时另记录 `reference_source_sha256.json`；引用提交和完整 job/实际 main argv
写入报告。参照与原生在独立进程运行，参照使用导出版本自己的模型、数据及训练/聚合代码。

原生新增 Client/Server 直接继承 FedClip 基类；不继承或调用 vendor 训练。窄初始化钩子避免
Model_Distribe、SVD和隐式模型构造。普通CNN、K深拷贝、每客户端内部Torch重设0、K优化器
绑定、两类CE、选择顺序、随机接收顺序、样本权重只乘一次及旧服务器差分增量全部保留。
原生复用公共reader/loader/CNN/纯指标及运行目录、H5、最终导出、日志ZIP。

实际main smoke曾发现公共H5文件名直接包含带 `/` 的dataset，导致尝试创建不存在的目录。
已增加默认保持原字符串的文件名前缀钩子，只在FedFewS用安全前缀；这是保存适配，不改变算法。
验收工具首次直接调用build_server也遗漏了调用方的数据根环境准备，现先使用固定参照的原审计
函数准备/核对数据。上述失败记录保留，不把未完成导出的早期运行当作通过证据。

## 环境与协议

既有 `D:\miniconda\envs\htfllib\python.exe`，Python3.11.14，Torch2.0.1+cu117、
CUDA11.7、cuDNN8500、NumPy1.26.4、SciPy1.16.3、h5py3.15.1、torchvision0.15.2+cu117。
GPU RTX3070 Ti Laptop，两个worker都是 intra-op2、interop8。
deterministic_algorithms=False、cudnn.deterministic=False、benchmark=False、matmul TF32=False、
cuDNN TF32=True，没有为通过而修改backend。正式数组不强制线程数，记录实际环境。

受控验收固定Python/NumPy为0；Torch按参照先设置control seed再重设0，保留各客户端内部重设。
每个worker的模型初始化、调用顺序、真正sampler批次和每个观察点Python/NumPy/Torch CPU/CUDA
状态都记录。原生端通过 `runpy` 执行真正 `main.py` 的参数解析、注册、普通CNN分支、训练、
日志捕获、H5/模型保存和ZIP；观察器只包装真实调用，不回放数据。

## 合成短程结果

使用实际924,708参数 FedAvgCNN、100类、K3、每模型1epoch、SGD .005、batch4。
不等样本量为训练 `7+3*i`、测试 `3+3*i`，数组/列表两种图像输入，部分有效pkl旁有内容
不同的旧npz。合成config alpha=null，不冒充真实Dirichlet或真实pat10。
每次实际训练3次，loop0与2评价，loop1不评价，末轮强制评价。

| 对照 | 客户端数 | 观察点/参数元素 | 最大参数绝对误差 | 最大标量误差 | 结论 |
|---|---:|---:|---:|---:|---|
| c73兼容→原生，CPU，dir05路径入口 | 20 | 499 / 177,543,936 | 0 | 0 | PASS，atol=rtol=0 |
| c73兼容→原生，CPU，pat10路径入口 | 10 | 259 / 94,320,216 | 0 | 0 | PASS，atol=rtol=0 |
| c73兼容→原生，默认CUDA | 20 | 499 / 177,543,936 | 3.2410980202257633e-4 | .00482177734375 | FAIL |
| c73兼容→c73兼容自身重放，默认CUDA | 20 | 499 / 177,543,936 | 2.008196897804737e-4 | .008691787719726562 | FAIL，非原生验收 |

CPU覆盖初始参数/固定输入forward/构造后RNG、每客户端/K模型训练结果、最后epoch聚合CE、
部署终点CE、α/w、上传ID映射、聚合参数、部署选择、correct/samples、真正batch索引及RNG。
另外直接比较实际保存的所有K服务器/K客户端张量、优化器和调度器状态；审计也逐文件比较
config字节、真实alpha、文件选择/原文件/解码内容/顺序指纹。

CUDA未放宽原 atol=1e-6、rtol=1e-5。原生对照首次超差在event213：loop1、Client11训练结束，
Model0 conv1.weight约2.73436e-6；最大差位置event370、Model2 fc1.weight、索引126353。
固定c73自身重放首次超差event237：loop1、Client19训练结束，Model1 conv1.weight约9.86271e-6；
最大差event269、Model0 fc1.weight、索引287952。两组均无RNG或loader差异。
标量误差是Comparison汇总多类标量的最大值，不将其自动解释成CE误差。

原生迁移不是默认CUDA偏差出现的必要条件；固定参照自身已经有重复性问题，与之前报告一致。
本次没有进一步定位CUDA算子，也不据此排除所有可能差异或给出默认CUDA通过结论。
FAIL返回非零并保存首次超差双方状态。若用户在服务器相同环境上复验仍FAIL，应先处理数值
重复性/定位差异，不通过放宽容差跳过。无需重新采集/生成数据。

## 无观察器与旧路径检查

额外直接启动不挂观察器的实际 `main.py`，10客户端/3次训练，同一批数据/受控初始化。
所有K服务器、K客户端参数、部署索引、优化器、调度器及结束Python/NumPy/Torch CPU/CUDA
状态与被观察的原生运行逐元素一致。H5 correct/samples独立重算主指标一致；检查JSON/CSV、
实际权重H5、最终bundle导出和日志ZIP。ZIP包含H5/rounds.json，不包含任何pt模型。

58项测试通过：FedFew相关22项（含原兼容15项、新原生7项），公共部件/旧Projection/日志/
保存隔离/客户端初始化35项，以及异构低秩旧路径运行1项。
旧路径运行针对Avg及12.5% ProjectionSoftmax，20客户端、真实临时npz、每轮5epoch、各两轮；
旧数据内容、实际batch、每轮上传/服务器参数哈希、随机流和汇总一致，约199秒。
公共工厂默认值、npz旧策略、旧checkpoint类路径仍通过检查。
特别保持新final_model_root参数在旧入口缺省时不增加字段，避免改变已有Prox/head命名逻辑。

Bash数组语法和Python编译通过。用户原绘图文件SHA256仍为
`A066FD21F6DF5DADC2DF1BBAF31C247D7C8326348C36F793B3B1FB7C0A175E8A`，保留其暂存状态，
不纳入本次提交。兼容入口和模块对c73没有diff。

本机原始报告（Git忽略的产物）分别在：

```text
fedfews_runs/native_cpu20_final/acceptance_report.json
fedfews_runs/native_pat10_cpu/acceptance_report.json
fedfews_runs/native_cuda20/acceptance_report.json
fedfews_runs/native_cuda20_c73_repeat/acceptance_report.json
fedfews_runs/native_unobserved10/                 不挂观察器的main输出
fedfews_runs/native_final_exports10/             最终字段/初始化RNG/预算/ZIP复核
```

## 服务器真实验收与正式运行

本机没有用户真实CIFAR-100分片。真实alpha按本次说明为.1，目录仍为a0p5；两组真实命令见
[`完整命令数组`](../system/fedfews_native_commands.sh)，从system目录执行。
先 `PREVIEW_COMMANDS`、再真实 `REAL_CPU_ACCEPTANCE_COMMANDS` / `REAL_CUDA_ACCEPTANCE_COMMANDS`。
兼容参照无须外部仓库，工具从本仓库固定Git提交导出。正式main训练不需要Git参照、source-root或YAML。

固定c73 CUDA自身重放的复验方式（独立输出目录；只做3次训练）：

```bash
python verify_fedfews_native.py --output-dir /path/to/new/c73_cuda_repeat \
  --num-clients 20 --rounds 3 --eval-gap 10 --batch-size 4 --device cuda \
  --device-id 2 --threads 2 --atol 1e-6 --rtol 1e-5 --reference-repeat
```

正式两项COMMANDS均已提供，仍由用户启动。不宣称合成smoke证明准确率提升，也不以历史
final41.30%作为正确性判据。CPU工程行为已验收；**计划使用的服务器/真实数据/CUDA协议尚待
验收，当前不能宣称GPU正式长实验的数值一致性已通过。**
