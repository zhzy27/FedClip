# FedMultiGroupProj 首版实现与短程验收

基于`a529cd7`，独立原生算法入口。未安装/升级环境，未生成/修改真实数据，
未推送或启动正式长训练；绘图文件SHA256保持原值。

## 实现

- `main.py -algo FedMultiGroupProj`：当前完整CNN-512、共享分类器、CE SGD，K/N/数据/seed等可配置。
- `clientMultiGroupProj.py`：K模型及各自优化器，分别训练local_epochs；无Few权重或CE部署选择。
- `serverMultiGroupProj.py`：首轮训练更新确定固定分组，之后每列独立调用原单组聚合核。
- `multi_group_clustering.py`：同列global delta余弦的K列平均、容量约束核球面聚类、独立随机流和规范编号。
- `multi_group_reporting.py`：按固定归属提取全部客户端普通post-local计数，样本量加权，统一导出。
- 原Few/Avg/单组算法训练和聚合文件未改；公共改动限于新入口派发与日志ZIP纳入jsonl。

群内/群外默认.50/.50、tau.2；avg_only实际为1/0，训练任务不减少。
K=3默认20人7/7/6，所有人每轮训练三个不同模型各5 epoch，合计15次数据遍历。
分组不读取测试计数或标签重叠。载入已有分组时核对数据/初始化/协议/成员指纹，
直接保留来源映射，同时单独报告当前首轮上传是否逐位一致；不因GPU数值差异换组。

## 实际main入口合成验收

环境：已有htfllib，Python3.11.14，Torch2.0.1+cu117，NumPy1.26.4，SciPy1.16.3；CPU2线程。
只在忽略目录`fedfews_native_runs/`下建立隔离合成分片。

| case | 训练与评价 | 结果 |
|---|---|---|
| default_projection | 20人/K3/seed0/local5，两轮，每轮60份上传 | PASS，120个真实模型训练任务 |
| default_avg_only | 相同配置，复用Projection的分组文件 | PASS，仍120个任务；首轮上传、分组、指纹完全相同 |
| custom_projection | 5人/K2/seed7，显式2,3→规范3,2；lr.012、batch2、local2、clip0、beta.65、tau.4；三轮eg2 | PASS，30个任务；实际评价loop0/2，末轮强制评价 |
| custom_avg_only | 同上、复用同一分组 | PASS，首轮上传和分组精确一致 |
| custom_no_pre | 关闭微调前观察，复用同一分组 | PASS，与custom_projection全部轮次上传、服务器参数、最终RNG完全一致 |
| metadata_final_utf8 | 最终导出元数据版，5人/K2/一轮、eg3 | PASS，CSV/JSON/H5/manifest及模型重载；Windows无外部UTF8环境变量也能完成导出 |

验收工具逐次检查：

1. 每个客户端实际下载的每列参数SHA与对应聚合前服务器相同。
2. 每列重新独立调用旧`aggregate_group_updates`构造对照；全部参数逐元素相等，最大参数误差0。
3. 聚合前后普通上传的参数SHA不变，聚合不推进RNG。
4. 每任务实际样本访问和SGD步数与各客户端真实样本数/epochs/batch一致。
5. 测试样本数不等，独立用correct/samples重算全员及每组指标；Best为同轮全员平均的跨点评价最大值，最早并列正确。
6. 完整模型没有进入Model_Distribe；构造调用被验收钩子设为失败。
7. CSV/JSON/H5/manifest主序列与summary一致；K服务器和每客户端K模型均strict加载，固定输入预测逐元素一致。

默认初始化参数SHA256：
`4e45a927113fb911515381893caca5578bc807181092572447927a98f51b058c`，
与原Avg/Few/单组同构入口一致。seed7为`c3b2e8b0a1aac6e67880a37d6b7b9d36865c289c271b18ab2ab4598ec2c12959`。
默认分组指纹为`f64f70b1c19151819e5a75e51adbacdfec65b1a0d1e4b87d59f3936f87d148b7`；
该分组仅来自合成fixture，不用于真实数据实验。

机器报告：

- `fedfews_native_runs/multi_group_acceptance_v2/acceptance_report.json`
- 各case子目录`verification.json`（包含参数/RNG事件、分组、指标与日志路径）
- `metadata_final_utf8/verification.json`
- `default_entry_preview.json`（新算法缺省配置与已有数据审计，不训练）

## 旧路径回归

`verify_configurable_control.py --case default --reference-commit a529cd7 --compact`通过。
独立从Git导出冻结参照，20客户端、每模型local5、两轮，Avg/单组Projection/Few三条真实入口对照：
初始化、本地/服务器参数SHA、真实batch顺序、全部事件RNG、最终RNG及主指标完全一致。
机器报告：`fedfews_native_runs/multi_group_legacy_regression/acceptance_report.json`。

共63项相关单元测试通过：

- 新多群组9项（解析列聚合、退化/并列平衡聚类、分组校验、K任务/优化器、含BN的状态保护、不等样本指标、正式命令）。
- CNN与可配置入口15项、旧群组核12项、原生Few7项、旧population6项。
- 公共reader/model/loader/metrics8项、日志ZIP6项。

本机NumPy BLAS/linalg测试入口曾触发已有DLL异常；独立小例子的范数/内积校验使用等价逐元素运算。
运行算法的Gram归约使用Torch，SciPy Hungarian已在全部短程中验证；没有修改依赖或替换算法近似。
发现并修复了新入口Windows GBK重定向时旧公共导出打印emoji引发异常的问题；UTF8设置仅作用于新入口，旧Few设置保持不变。

## 交付及限制

完整两个COMMANDS数组项：`system/multi_group_commands.sh`；
参数、固定分组复用、日志与模型加载说明：`system/MULTI_GROUP.md`；
合成复验入口：`system/verify_multi_group.py`。
均不自动启动正式训练。默认gr100实际101轮、每客户端每轮15次数据遍历，与旧单模型Avg的5次预算区分。

本地没有真实pat20分片，本次不声称真实准确率改善或CUDA数值已验收。
自动审批策略阻止删除本次已验收的临时`.pt`文件；权重副本仍在上述两个合成验收目录，未清理既有文件。
