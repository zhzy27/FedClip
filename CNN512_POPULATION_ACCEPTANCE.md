# 群组比较协议验收（2026-10-17）

基于`7f72099`，新增显式`control_eval_scope`，旧缺省c0。
三正式实验仅为Avg/all20、GroupProjection/fixed7 beta=.50、FedFewS/all20。
七人指标是阶段性代理，**不是全20人的同一评价对象，也不是完整方法结果**。
未推送、未启动正式训练、未改环境或真实数据、绘图文件未修改。

## 主要修改

- 共享计数归约与统一primary导出。Avg每轮普通post-local评价20人；C0从同一计数提取。
  FedFewS直接使用其已有全员部署模型测试计数，训练终点CE选择、损失/α/w和更新不变。
- 完整CNN的group_projection_softmax路径不再读取ratio_LR、不分解、不恢复；
  下载和独立聚合后评价按参数名/形状复制完整参数，保留本地buffers。
- 群组原聚合核未改。只有显式新group协议将primary、summary、best及H5主标记切为
  测试样本加权；旧群组缺省仍宏平均。
- `primary_metrics.csv/json`、`primary_summary.json`、`client_metrics.csv/json`、H5主组与
  最终manifest使用一致scope/成员/权重/percent主指标。各组同时保留C0附加结果及各自原诊断。
- local_training明确样本访问量、optimizer_steps、data_epochs；预算5/5/15数据epoch。

## 本地结果

现有htfllib环境：Python3.11.14、Torch2.0.1+cu117，CPU2线程。
未安装/升级依赖、未改CUDA/backend默认配置。

| 检查 | 结果 |
|---|---|
| 新人口/指标/状态/命令测试 | 6项通过 |
| 原CNN完整模型对照单元测试 | 7项通过 |
| 原生FedFewS默认/算法/数据/RNG测试 | 7项通过 |
| 原群组公式/边界/状态/宏平均/启动器测试 | 12项通过 |
| 旧异构低秩三群组模式20人两轮及私有评价保护 | 2项通过 |
| 旧异构低秩Avg/Projection与189bacd数值/采样/RNG参照 | 1项通过 |
| 新三个真实main入口 | 每组20人、两轮、每模型5epoch，全部PASS |
| 旧三个c0真实入口 | 每组20人、两轮，全部PASS |

共35项相关测试通过。新smoke使用2/3/4个train/test样本的不等样本量fixture，
独立从client_metrics.csv正确数/样本数复算全20人或固定七人指标，核对：

- 同轮均值的best、最早best_loop_round/completed_count、final、last10/last20。
- CSV/JSON/H5及最终manifest完全对应同一主序列和主summary。
- 群组H5主标记为sample-weighted，不输出伪造的member_low_rank_ratio。
- 群内/群外实际质量为.50/.50；群内没有投影；群外使用原公式。
- 全部客户端每轮仅一次5epoch；FedFewS每客户端K3各5epoch。
- 完整模型、K服务器/客户端bundle实际保存并可重新加载。
- TargetProj的Model_Distribe和恢复入口设为失败哨兵，完整CNN运行未触发。

三组初始化参数SHA256和所有40个分片文件/解码后张量指纹完全相同：

`4e45a927113fb911515381893caca5578bc807181092572447927a98f51b058c`

此外，对此前7f72099交付时保留的CPU报告做独立回归：
三个c0序列及summary逐字段相等；Avg/Projection的完整逐轮聚合诊断history相等；
FedFewS的全部轮次损失、α/w、收发顺序、服务器距离、部署CE/选择及测试/训练指标相等，
只排除耗时。参数与RNG的严格旧路径回归另由上述异构参照及FedFewS测试覆盖。

测试还验证不等样本量会使宏平均与样本加权均值不同、不能平均个人历史best，
并在反复best时选择最早轮次；真实完整CNN额外评价前后checkpoint SHA256、
参数、buffers、模块状态和Python/NumPy/Torch CPU/已初始化CUDA RNG均不变。
`aggregate_group_updates`的AST与7f72099完全相同，原解析投影测试通过。

机器可读产物（忽略目录，不纳入提交）：

- `fedfews_native_runs/group_control_final_cpu/acceptance_report.json`
- `fedfews_native_runs/group_control_c0_regression_cpu/acceptance_report.json`
- `fedfews_native_runs/group_control_c0_regression_cpu/behavior_regression.json`

本次E盘未挂载，在D盘验收；使用验收专属硬链接避免重复模型副本占用，
正式导出和训练不使用此优化。早期一次smoke发现独立复算时percent乘除顺序相差float64一ULP，
复算统一先归约fraction再转percent后重跑通过；未放宽模型参数容差。

## 命令与使用边界

三个完整`declare -a COMMANDS`见`system/cnn512_population_commands.sh`，从system目录运行。
每组gr100=101轮，独立名称、checkpoint/logs/H5/final_models，GPU可替换；
不包含额外正式消融，旧`cnn512_control_commands.sh`逐字保持不变。
说明、文件位置、scope和单位见`system/CNN512_POPULATION.md`。

```bash
# system目录，必须全新输出目录；只生成隔离合成fixture
python verify_cnn512_control.py --population-protocol --compact --output ../fedfews_native_runs/population_accept_cpu --device cpu
python verify_cnn512_control.py --compact --output ../fedfews_native_runs/c0_regression_cpu --device cpu
```

本机无真实pat_20数据，未据合成准确率声明性能提升，未宣称CUDA精确数值已验收。
