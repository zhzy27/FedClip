# 固定七人目标群体对照

G=[0,1,2,3,4,6,7] 是预先指定的诊断组，不按测试成绩选成员，
不声称它是最优集群。只维护一套共享服务器模型。群外13人正常下载、
训练5 epochs和上传，连 group_avg_only 都不跳过任何客户端。

三个模式的普通训练、额外评价和观察顺序相同，仅聚合规则不同：

|模式|聚合|主指标|
|---|---|---|
|group_avg_all|全部20人原样本量权重 Avg|七人普通 post-local 宏平均|
|group_avg_only|群内样本量归一化 Avg|同上|
|group_projection_softmax|群内参考、群外 Projection + 原余弦 softmax|同上|

使用原完整训练集，不启用 meta 留出、guidance、权重学习、EMA、
Top-K、逐层投影或群体质量扫描。原单目标模式、12.5%自身质量入口和结果文件不修改。

## 公式与边界

群内归一化样本权重为 a_i=n_i/sum_G(n)，参考为
Delta_G=sum_G(a_i*(W_i_post-W_global))。一个投影系数覆盖包括分类头的所有
recovered full-W named_parameters，不直接平均异构U/V，不聚合buffers。
仅群外负内积更新减去 dot/(norm(Delta_G)^2+1e-12)*Delta_G；
群内成员即使与群体参考冲突也不会被投影。未调用旧单目标投影核伪装群体。

群外分数用投影前 cos(Delta_j,Delta_G)，稳定 exp((cos-max_cos)/.2)
归一化为 q；不乘样本量评分。群内质量 beta*a_i，群外质量 (1-beta)*q_j。
最后为 W_global 加实际加权 delta，再减群外加权投影修正。
两种Avg对照使用直接加权参数求和，group_avg_all保持原Avg顺序与算术，逐元素一致。

--target_group_ids 不传时仅在群体模式启用固定七人；显式列表校验非空、
无重复、整数且在[0,num_clients)，升序保存。原单目标入口不传参数时保持原行为。
显式向单目标模式传group IDs会报错，防止误以为评价了群体。

--target_group_weight 只允许 group_projection_softmax，需有限且在[0,1]。
未传时按 n_group/n_all 取值，当前等样本为 .35，群外 .65。
group_avg_only固定群内质量1；group_avg_all保留原样本质量。
未从单目标12.5%推导35%的最优性。首批正式实验不显式指定beta，不做扫描。
群内训练样本总量必须正。

参考平方范数恰为0时：不投影，余弦0，群外q均匀。
近零但非零时继续使用原epsilon规则，同时记录reference_epsilon_dominated，
不跳过投影、不换方向或成员。beta=0/1时零质量角色的有效数量、min/max权重均为0。
beta=1仍可记录群外几何投影诊断，但其加权删除量为0。
无群外成员且默认质量时beta精确1；显式请求正群外质量却没有群外成员时拒绝。
样本比例最后一位舍入不会使全成员默认群体误报缺少helper。

## 评价与保护

每轮（round=loop_round+1）：

1. 20人正常从同一服务器按容量SVD下发，并各训练正常5 epochs。
2. 观察七人的已保存普通post-local低秩checkpoint。
3. 恢复full-W、计算群体参考/原始评分/投影，更新单一服务器。
4. 在私有模型副本上执行相同SVD下发、复制参数、保留各成员本地buffers，再评价七人。

步骤4不调用写checkpoint的client.set_parameters，也不改写或恢复普通checkpoint。
所有额外评价、模型恢复和诊断保护Python/NumPy/Torch CPU/CUDA随机状态。
服务器只更新named_parameters；本地/服务器buffers保留。
无测试成绩进入权重或成员计算。为保持比较不受早停等干扰，群体模式拒绝
auto_break、few_shot、train_slow、非每轮评价或meta训练划分。

主曲线group_post_local_acc为七人客户端宏平均；另标明sample-weighted曲线。
每轮记录最差成员和总体标准差(ddof=0)，每个成员的容量比、参数数目及准确率。
final是最后一次普通训练；last10是最后10次（不足10次取已有次数）；
best先计算每轮群体均值，再选其最大值与最早轮次，不平均各人的不同轮次best。
每个成员另有独立final/last10/best及最早best轮次，便于后处理C0-3和C4/C6/C7。

时序为 group_local(t) -> group_agg(t) -> group_local(t+1)。
本行权重影响下一轮初始化，不能把同一行local与agg差值称为本地微调收益。

## 正式启动

完整 declare -a COMMANDS 数组在 system/target_group_commands.sh。
GPU0/1/2和所有checkpoint/H5/final目录独立，文件只定义数组，不自动执行。
公共配置 Cifar100/pat_20，20人全参与，seed0，Decom_CNN-5-512，
lr=.005、batch16、local5、Frobenius=.001、-gr100；实际101次正常训练/聚合。
正式模型容量仍按现有.9/.37/.35/.25/.15循环和n_kernels=16配置。

从system/可以先查看启动器命令，不会训练或创建实验目录：

~~~bash
python run_target_proj.py --modes group_avg_all group_avg_only group_projection_softmax \
  --target_group_ids 0,1,2,3,4,6,7 --rounds 100 --seeds 0 \
  --device-ids 0 1 2 --parallel --dry-run
~~~

去掉--dry-run才会启动完整实验；本次没有执行该操作。
如果显式指定beta，仅选择group_projection_softmax；和Avg模式一起传会拒绝，
也不允许用projection_self_weight控制群体质量。

## 日志与结果

直接COMMANDS启动：checkpoint及逐轮文件位于
system/temp/group7/<mode>/Cifar100/FedTargetProj/<run_id>/。
启动器启动：system/target_proj_runs/<timestamp>/<mode>_group0_1_2_3_4_6_7/，
其中train.log记录stdout，command.json记录完整命令，checkpoint在checkpoints子目录下的实际run路径。
启动时打印具体metrics_dir和H5路径。所有run目录独立，不覆盖旧单目标文件。

- target_group_metrics.csv：每轮macro/weighted local与agg、worst/std、beta/实际质量、
  参考范数、冲突、删除范数、三个角色的有效数量/min/max权重。
- target_group_members.csv：各成员逐轮普通local与agg准确率、test/train样本数、容量及参数数。
- target_group_clients.csv：20人的群内角色、原样本权重、实际alpha、群外原余弦、
  稳定softmax分数/q、冲突、投影系数、更新/删除范数。
- target_group_metrics.json与target_group_matrices.json：配置、规范化IDs、时序、
  群体及各成员summary、每轮群内cosine/conflict矩阵和七人更新范数。无Client0主指标字段。
- H5：target_group分组，member_*矩阵[round,7]、client_*矩阵[round,20]、
  群内矩阵[round,7,7]及具名group曲线。没有target_projection分组。
- final_models对应run：普通20人及共享服务器checkpoint、group日志复制件、
  包含群体summary和primary_accuracy_metric的manifest。

scope明确为原始全模型global delta；matrix中同时保存投影epsilon、tau、读取顺序。
raw_conflict_with_reference仅是原几何诊断；conflict是实际被投影的群外更新。
Avg模式不投影，群内所有模式系数与删除量均0。

## 本地验证边界

解析测试覆盖三种模式、质量/规则退化、群内冲突豁免、群外正负投影、
head、零/近零参考、无群外/零质量、非法参数和上传、输入/RNG不变及统计时序。
两轮实际Hyper_CNN_512烟测使用20人、固定7人、100类head、原五种容量比例、
5个正常local epochs、batch16和原SGD/Frobenius/clip规则，覆盖三模式及CSV/JSON/H5/final。
为了缩短本地测试时间，烟测使用合成train16/test4、n_kernels=4；
正式入口仍是CIFAR-100和原n_kernels=16，不据烟测宣称群体性能已经改善。
烟测校验三模式第一次聚合前普通上传数值与随机状态一致；
每人正常训练/下发各2次；所有额外观察保持普通checkpoint字节与buffers不变。

本批新增14项测试通过，并完成191项旧回归检查（含独立模块发现的既有测试），
合计205项相关检查通过。覆盖旧Avg/Projection/自身权重、softmax、逐层、
APA、DWA、meta冻结、post-local、存储隔离和启动器；CLI帮助、非法参数及dry-run通过。
实现保留在本地target_proj，不自动启动正式实验或推送远程。
