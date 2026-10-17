# CNN-512群组比较协议

基于`7f72099`新增显式`--control_eval_scope c0/all/group`，缺省仍是c0。
旧单目标命令文件`cnn512_control_commands.sh`未改；原低秩模式及FedFewS复现默认协议保留。

本批**只有三个正式实验**：

| 算法 | 主评价范围 | 聚合与训练预算 |
|---|---|---|
| Avg | 全部20客户端 | 原训练样本量Avg；每人5数据epoch |
| Group ProjectionSoftmax | 固定七人0,1,2,3,4,6,7 | 组内.50/组外.50；每人5数据epoch |
| FedFewS | 全部20客户端 | 原K3/smooth_mu=.01；各模型5epoch，每人15数据epoch |

**评价对象不同**。七人群组是尚未完成的完整方法的阶段性代理，不是完整方法结果，
也不把七人指标宣称为全20人的同一评价对象。启动日志和manifest明确记录这一点。

公共设置继承上一版：同构完整CNN-512、共享分类器、joint、Cifar100/pat_20原NPZ、
seed0完整随机流控制、20人全参与、SGD .005/momentum0/weight_decay0、batch16、
每模型5epoch、裁剪10、F/Prox关闭、无衰减、无额外微调。
`-gr 100`执行loop0–100，共101次训练/聚合，每轮评价。

## 流程与指标

Avg在20人都完成本轮普通训练后、聚合前，仅测试每人的当前post-local checkpoint一次，
收集correct/test_samples。额外观察保护checkpoint、模型/buffers和Python/NumPy/Torch随机流。
C0直接从同一份计数提取，不重复测试C0的post-local模型。原聚合后C0评价仍是单独诊断，
不替代本轮post-local主指标。

FedFewS不改变选择、训练或聚合：每人训练K模型，按**训练集终点CE**选部署模型，
沿原顺序测试全部部署模型。从已有计数提取全员指标，不增加C0测试、不使用测试表现选模型。

群组完整CNN路径直接按参数名/形状复制服务器参数，上传直接读取完整参数。
聚合后评价把完整服务器参数复制到独立评价模型；普通本地checkpoint和buffers不被覆盖。
元数据标为full_CNN，没有ratio_LR，不伪造低秩属性。
群组原聚合核`aggregate_group_updates`未改：组内训练样本量归一化参考delta；
组外逐helper投影；投影前余弦、temperature=.2；组内.50/组外.50。

每轮主指标为**指定范围的sum(correct)/sum(test_samples)**，再转percent导出。
主比较结果取这条每轮序列的best，同时报告最早best轮次、final、last10/last20及点数。
不平均各成员各自历史best。成员间worst/std、宏平均、聚合后准确率是附加诊断。
旧群组协议的默认宏平均口径不变；只有本批显式group评价协议将summary/best/H5主指标切为样本加权。

## 三项完整启动数组

在服务器`FedClip/system/`目录，使用
[`cnn512_population_commands.sh`](cnn512_population_commands.sh)的完整`declare -a COMMANDS`。
它只定义三条命令，不执行训练；GPU默认0/1/2可替换。

```bash
export POP_DATA_ROOT=/data/zh/AllRepo/FedClip/dataset
export POP_RUN_ROOT=/data/zh/AllRepo/FedClip/fedfews_native_runs/cnn512_population/run01
source cnn512_population_commands.sh
# 将数组交给已有调度器。
```

目录分别为`avg_all20`、`group7_beta0p50`、`fedfews_all20`，每组独立checkpoint/logs/H5/final_models。
FedFewS命令没有任何target_group参数；群组命令没有projection_self_weight。
原初始化factory和NPZ指纹审计继续复用。三组data_manifest和initial SHA256应相同。

## 文件与单位

- `logs/train.log`及`logs.zip`：日志可单独下载，不含模型。
- `logs/run_<id>/primary_metrics.csv/json`：每轮correct/test_samples、百分比主指标、
  loop/完成次数、scope、成员IDs、test_sample_count加权方式和单位。
- `primary_summary.json`：百分比final/best/最早best_loop_round与best_completed_train_count、
  last10/last20和实际点数；以它作为本批主比较入口。
- `client_metrics.csv/json`：每轮所评价成员的correct/test_samples及百分比post-local准确率。
- `local_training.csv/json`：全20人的实际访问量、train_batches/optimizer_steps、耗时、
  每模型local_epochs、model_count、总data_epochs。
- H5根`primary_metrics`：同一主序列/summary、成员ID和逐成员计数矩阵；
  根attrs标记scope/成员/加权/percent，`cnn512_control`主序列也指向同一人口范围。
- 最终`manifest.json`和`comparison_manifest.json`包含相同`primary_summary`和范围标识。
- `c0_*`仍是**C0附加结果**。FedFewS原`summary.json/evaluations/clients`及模型选择、α/w、
  接收质量继续保留；群组原`target_group_*`保留成员、聚合后、worst/std、权重/投影诊断。
  原群组细表准确率用fraction；canonical `primary_*`统一percent，H5各层属性区分单位。

## 复验

```bash
# system目录，仅隔离合成短程，不改写真实数据。
python verify_cnn512_control.py --population-protocol --compact --output ../fedfews_native_runs/population_accept_cpu --device cpu
# 既有c0协议短程回归（不是新增正式消融）
python verify_cnn512_control.py --compact --output ../fedfews_native_runs/c0_regression_cpu --device cpu
```

每个入口20人、两轮、每模型5epoch；不等train/test样本量。工具独立按CSV计数复算主指标、
最早best/尾均值，核对CSV/JSON/H5/manifest、组内外质量、模型重新加载、数据/初始化一致。
`--compact`仅验收时对同磁盘重复导出模型使用硬链接，减少临时空间；不改正式保存或训练逻辑。
正式三条命令没有此验收参数。输出目录必须全新。

本地无真实pat_20数据，不据合成smoke声称性能改善。CPU短程不替代服务器真实数据或CUDA数值验收。
