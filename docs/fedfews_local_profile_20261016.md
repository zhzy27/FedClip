# FedFew 用户本地配置修正与验收（2026-10-16）

基于 FedClip `4220b61`，按用户新提供的服务器 `git diff HEAD` 修正迁移配置。
该 diff 没有算法修改；未提供实际异常堆栈，因此这里只确认并修复代码中可复现的配置冲突，
不声称已经定位服务器所有报错。未推送、未启动长实验、未修改依赖或真实数据。

## 已确认的问题及修正

1. 原迁移的两个 base YAML 仍使用 `cifar100`、GPU 0，与用户实际使用的 `Cifar100`、GPU 2
   不同。现在均按用户版本设置。Linux 路径大小写仍严格匹配，不自动搜索另一目录。
2. 原验收要求合并后的源 YAML 与迁移 YAML 完全相等。用户的改动会触发
   `Upstream base + algorithm YAML merge differs`。现在显式记录差异、有效默认值及双方
   短程参数；只允许数据集名称大小写和 GPU ID 的执行配置差异，且必须验证双方实际收到
   相同 dataset/device。训练、模型、K、smooth_mu 等其他差异仍报错，短程覆盖也不能掩盖
   原默认协议不同。失败前保存 `configuration_alignment.json`。
3. 原验收固定读取 dir05 的源 YAML。现在支持 `--config`，pat10 会读取自身的独立源配置。
4. 预览也使用解析后的 GPU ID，包括仅在 YAML 指定、以及 `--device_id=...` 的情况。
5. 启动数组默认 `FEW_SOURCE_ROOT=/data/zh/AllRepo/FedFew`，指向实际修改过的工作区；
   `FEW_GPU_ID=2`，可覆盖。默认输出根新增 `local_profile_20261016`，不覆盖旧结果。

用户另修改了生成器的 `alpha=.1 -> .5`。本次没有复制、导入或执行生成器，仅静态读取
其常量与 SHA256 作为来源证据。**已有分片的真实 alpha 仍以 config.json 为准**；生成器
改成 .5 不会改变已有 .1 数据。正式 dir05 命令继续要求 `--expected-alpha .5` 与
`--expected-partition dir`。不重新生成、不篡改 config.json。

`source_manifest.py` 保留官方提交的原始哈希，另列用户报告的 profile。运行时继续记录
实际源仓库提交、dirty 状态、文件哈希；用户展示的两个未跟踪文件内容尚未检查。
外部参照继续使用自身 reader、CNN、Client/Server 与算法，未改接公共实现。
K 模型/优化器、两类 CE、双层权重、增量聚合、DataLoader 和随机协议没有改动。
FedTargetProj 和用户绘图文件未修改。

## 本机验证

使用原有 `D:\miniconda\envs\htfllib\python.exe`，Python 3.11.14、Torch 2.0.1+cu117、
NumPy 1.26.4、SciPy 1.16.3，未安装依赖。完整版本保存在各报告 environment 字段。

独立参照来源为 `D:\All_Git_Repo\FedFew_reference`，官方提交
`1b29dec70b66d89b956d90e0da024897f698897a`，该工作区保持干净。
修改版参照是从此处复制所需源码到独立临时目录后，仅应用用户报告的三处改动；来源和
副本路径保存在 `reference_fixture.json`。不是对服务器完整工作区的直接验收。

| CPU 短程配置 | 客户端 / 实际训练次数 | 观察点 | 参数最大绝对误差 | 标量最大绝对误差 |
|---|---:|---:|---:|---:|
| 用户修改版 dir05 配置 vs 迁移 | 20 / 3 | 499 | 0 | 0 |
| 干净官方 dir05 配置，显式对齐路径/GPU | 3 / 3 | 91 | 0 | 0 |
| 用户修改版 pat10 配置入口 | 3 / 3 | 91 | 0 | 0 |

全部为不等样本量合成图像、实际普通 FedAvgCNN、K=3、每模型 1 epoch、batch4。
pat10 行只验收配置入口及执行路径，不冒充真实十人 pat10 数据。
每次均 atol=rtol=0；loop0/2评价，loop1不评价，覆盖末轮强制评价。
20 人测试比较 177,543,936 个参数元素，外部参照训练约 23.61 秒、迁移约 29.70 秒，
包含记录与保存开销。损失、选择、α/w、参数、实际批次与 RNG 全部一致。

37 项测试通过：新增 profile 6 项、FedFew 既有 9 项、公共部件 8 项、旧 Projection 14 项。
新增测试覆盖官方/修改版配置、两组默认值、错误协议仍拒绝、生成器不执行且不推进 RNG、
以及生成器 .5 不能覆盖实际数据 .1。Bash 数组语法检查通过。
另执行不启动训练的预览，确认默认 dataset=Cifar100/...、device_id=2、
CUDA_VISIBLE_DEVICES=2、source generator alpha=.5；没有创建训练输出目录。

本次原始证据位于（Git 忽略的独立产物）：

```text
fedfews_runs/local_profile_20261016_validation/
  reference_fixture.json
  preview.json
  modified_source_cpu20/{acceptance_report.json,configuration_alignment.json,data_manifest.json}
  pristine_cpu3/{acceptance_report.json,configuration_alignment.json}
  modified_pat10_cpu3/{acceptance_report.json,configuration_alignment.json}
```

本机没有真实 α=.5 分片；真实数据与服务器 CUDA 验收仍待执行。本次配置修正不解决、
也没有重新验收此前默认 CUDA 的重复性问题，详见
[公共化验收报告](fedfews_shared_acceptance_20261016.md)。不把 CPU 通过或确定性诊断通过
写成默认 CUDA 已通过。

## 服务器命令

在 FedClip 根目录使用原环境，替换实际路径即可。不要 reset 或覆盖原 FedFew 工作区。

```bash
export FEW_PYTHON=python
export FEW_SOURCE_ROOT=/data/zh/AllRepo/FedFew
export FEW_DATA_ROOT=/data/zh/AllRepo/FedFew/PFLlib/dataset
export FEW_GPU_ID=2
export FEW_OUTPUT_ROOT=/data/zh/AllRepo/FedClip/fedfews_runs/local_profile_check_01
source system/fedfews_commands.sh

eval "${PREVIEW_COMMANDS[0]}"             # 不训练，核对路径、真实 alpha 和来源
eval "${SYNTHETIC_ACCEPTANCE_COMMANDS[0]}" # 20人 CPU 合成短程，严格零误差
eval "${REAL_CPU_ACCEPTANCE_COMMANDS[0]}"  # 真实 .5 分片，3次训练，CPU 严格验收
# 或在计划使用的 CUDA 环境做实际验收，容差未放宽：
eval "${REAL_ACCEPTANCE_COMMANDS[0]}"
```

完整 2000 配置轮次命令仍在 `COMMANDS`；仅定义数组不会训练。先检查真实验收报告及
数据指纹，再由用户决定启动。可选 pat10 的 `PAT10_ACCEPTANCE_COMMANDS` 和
`PAT10_COMMANDS` 也已保留，待首组验收后使用。
