# 可配置完整CNN入口验收（2026-10-18）

基于`7e236dd`。将固定实验值改为合法范围校验，保留算法公式、旧默认和真实数据。
不改环境、不生成真实数据、不启动正式训练、不推送；绘图文件未修改。

## 代码与约束

开放分片目录/N/cpc/seed/target/lr/batch/epoch/gr/eg/clip/test batch/K/mu/自身及群内质量。
独立`projection_softmax_tau`和`group_softmax_tau`缺省仍.2，计算与导出共用实参。
仍只支持已适配的Cifar100/100类/RGB32/full CNN-512、共享头、joint CE、全参与fresh run，
不支持的部分参与、resume、slow/drop、F/Prox/meta等明确拒绝。

seed实际控制公共模型初始化及训练前Python/NumPy/Torch CPU/CUDA重置；公共Client内部seed0不改，
保留旧生命周期，构造完毕后用配置seed重置训练流。原Few original入口不变。
裁剪0明确关闭，两客户端与独立无裁剪SGD逐参数一致；省略仍保留各旧路径缺省。

数据按指定路径/格式读取，核对config真实N/100类/partition、显式expected alpha。
pat核对cpc元数据及实际合并标签数/分配统计；缺少分配证据时记录partial能力，
不靠目录名推断。dir不施加固定每人类别数。文件和解码指纹继续保存。

评价调度统一为`loop%eval_gap==0 or loop==global_rounds`；末轮强制评价。
非评价轮正常训练/聚合，训练预算每轮记录，不复用旧轮准确率。
target非零正确导出；群组不含target时只标记未观测，不额外测试或借用其他计数。
实际配置、数据决定的缺省质量、预算/完成轮数/评价次数/采样量/步数写入请求/生效manifest和H5。

## 真实main入口合成验收

全部采用已有htfllib（Python3.11.14、Torch2.0.1+cu117、NumPy1.26.4、SciPy1.16.3），CPU2线程。
未安装或升级依赖。合成fixture与正式dataset完全隔离，模型导出验收用同卷硬链接减少重复副本。

| case | 覆盖与结果 |
|---|---|
| default | 20人、pat20、seed0、local5、K3/.01、clip10，Avg/群组/Few各两轮；当前与Git冻结`7e236dd`独立进程参照全部PASS |
| pat10 | 20人、每人10类、原公共lr/batch/local/clip/K，Avg/Few各两轮，全员加权及保存重载PASS |
| custom | 4人、seed7、target2、lr.012、batch4、local2、K2/.025、clip0、test batch3，四轮、eg2；Avg/群组/单目标Projection/Few全部PASS |
| dir | 4人、真实alpha.5、seed11、target3、lr.007、batch3、local1、K2/.04、clip2.5；cpc99不套pat检查；eg3大于gr1，Avg/Few末轮评价PASS |

default参照直接从Git archive导出源码，未把参照替换成迁移实现。
三算法合计129个初始化/本地/聚合观察事件：所有本地/服务器参数SHA256、主指标、
实际batch索引顺序及每事件Python/NumPy/Torch CPU/已初始化CUDA RNG指纹完全相同，
最终随机状态也完全相同。没有放宽参数误差容差。

每个case不同算法的初始化参数和训练初始RNG指纹一致；不同seed的初始化确实不同：

- seed0：`4e45a927113fb911515381893caca5578bc807181092572447927a98f51b058c`
- seed7：`c3b2e8b0a1aa...`（完整值见机器报告）
- seed11：`d754317ab327...`（完整值见机器报告）

custom群组为`[0,1,3]`，target2不在群内、群内质量省略按真实训练样本量计算、group tau=.65；
单目标自身.30、projection tau=.45。稀疏评价实际loop为`[0,2,3]`，
四轮训练预算全部存在，最后10/20按三次评价计算。
独立从逐客户端correct/samples复算主序列，核对同轮均值Best/最早轮次、CSV/JSON/H5和最终manifest。
所有case均验证实际训练访问量/优化步数与生效配置一致、实际模型及Few K bundle重新加载成功。

## 测试与预览

新可配置测试8项通过；原CNN/Few/群组/无投影等相关测试41项通过，共49项；
公共reader/model/loader/metrics回归8项通过；旧异构低秩Avg/Projection的20人两轮
与189bacd参数/批次/RNG严格回归通过，总计58项相关测试通过。

新增测试覆盖关闭裁剪的完整CNN单步SGD、seed生效、动态质量、目标ID、温度、
数据错配/缺失证据、非法值、Dirichlet忽略cpc、两条正式命令参数一致。
实际main.py的Avg和Few `--control_preview`均成功审计隔离pat10分片：
seed9、lr.012、batch4、local2、clip0、eg2、测试batch3，Few K2/.03；
不构造模型、不训练，打印预算分别2和4及评价loop `[0,2,3]`。

## 报告与命令

机器报告保存在忽略目录：

- `fedfews_native_runs/configurable_default_v2/acceptance_report.json`
- `fedfews_native_runs/configurable_pat10/acceptance_report.json`
- `fedfews_native_runs/configurable_custom_final/acceptance_report.json`
- `fedfews_native_runs/configurable_dir_final/acceptance_report.json`

各子目录`current/<algo>/verification.json`包含事件、批次、主指标、数据清单及日志路径；
default另含冻结reference报告。早期一次稀疏群组smoke发现空成员CSV写入问题，已修复并重跑通过。

新参数/数据核验/调度/导出说明：`system/CNN512_CONFIGURABLE.md`。
正式完整两项`declare -a COMMANDS`：`system/cnn512_pat10_commands.sh`。
20客户端pat_10、seed0、lr.005、batch16、local5、gr100=101轮、clip10，Few K3/.01；
独立目录/GPU，仅定义命令，不启动。原pat20命令文件保持原样。

本地缺少真实pat_10/20分片，因此不宣称真实性能或CUDA逐位数值已验收；
合成smoke用于实现与协议正确性，不作精度提升结论。
