# 群组优先、组外补齐的类别原型

入口仍为 `main.py -algo FedMultiGroupProj --multi_svd`。原型是独立的本地损失开关，不改变固定K、分组、参数下发、Projection/Avg核、部署或评价规则。默认none/λ0，原模型forward、参数名称及state_dict结构均未修改。

## 使用范围及参数

|参数|默认|含义|
|---|---:|---|
|`--multi_proto_mode`|none|none/group_priority|
|`--multi_proto_lambda`|0|有限非负系数|
|`--multi_proto_start_round`|2|人类轮次，2表示loop=1训练前；必须≥2|
|`--multi_proto_batch_size`|16|独立统计采集batch|
|`--multi_proto_eps`|1e-12|样本/最终原型单位化及余弦epsilon，有限正数|
|`--multi_proto_diagnostics`|1|0只关闭额外的客户端/容量方向余弦表，不改变原型、采集或训练|

支持当前全员、无预热、首轮full_weights固定分组的SVD流程，两种原聚合模式 `group_projection_softmax/group_avg_only` 均可用。原框架模型、数据、部分参与/续训等约束继续保留。拒绝与任何在线KD模式、非零Prox/头L2、本地分类头、FedRep组合。正λ的开始轮次必须落在实际gr+1预算内。none配非零λ明确报错。

K、C、D、分组人数和容量均读实际配置/模型；D从实际fc3.in_features取得，不另建完整编码器。当前框架支持的模型仍为原Decom_CNN-5-512/CIFAR-100。与旧实验一样，异构低秩特征仅近似处于同一母模型坐标，**不宣称各容量坐标严格一致**。

## 一轮的严格边界

首轮：完全旧下载/SVD→全部N×K普通训练→恢复完整上传→确定/读取分组→原聚合；不采集原型。

从配置的开始轮次起：

1. 全部N个客户端完成全部K列原SVD下载，保存实际低秩下载状态指纹。
2. **在任何客户端SGD之前**，逐客户端、逐模型列用完整原训练集采集一次。独立loader：不shuffle、不drop_last、workers0，无增强；eval/no_grad，保护Python/NumPy/Torch CPU/CUDA随机状态。临时副本为该客户端真实低秩模型，不恢复/重分解、不冻结真实学生。
3. 采集完全部N×K后才合并K套原型并发给全部客户端；再次核验所有模型未变化、没有upload_round，构成全体SGD屏障。
4. 所有客户端按原model_id顺序、原shuffle和每模型local_epochs训练全部K列，每列使用对应P_g。正常恢复上传和原参数聚合。

样本顺序、标签和数量对照原数据审计指纹；索引检查确保每模型每原训练样本恰好前向一次。关闭/λ0/开始前跳过所有采集、通信及特征hook，训练走原CE＋F。

每轮重新生成，不读历史原型、不做EMA；本轮无效类别不会继承上轮值。

## 特征、来源和合并

使用临时fc3 pre-hook获取该线性层**实际输入**。原Hyper_CNN_512的head在base后还有ReLU，因此该值是ReLU(base(x))。统计和学生损失都走原forward；学生一次forward同时供CE与特征损失。hook及时移除，不存入模型或checkpoint。

每张图h先按max(norm(h),eps)单位化为u，然后累加float64类别总和s，真实标签计数n为int64。每客户端上传K×C×D的s与一份C维n，不发送样本级特征。模型列之间不混合。

每个g/c先检查群内n是否有正值：有则只使用群内，完全忽略组外同类；没有才用组外。q=所选来源的总和s/总样本数n，最后p=q/max(norm(q),eps)。不是客户端类别均值单位化后等权平均。

无样本或合并均值norm(q)≤eps时无效，p置零、valid=False；非有限特征/统计直接失败。群内有样本但向量为零时，仍不改用组外，记录candidate_source=group、source=invalid及原因。

## 学生目标与梯度

`CE + 原加权F + λproto * mean_valid(1-cos(h,p_label))`。全部客户端、全部K列生效，不限定所属模型。有效样本仅指标签有有效原型，整batch无效返回可微零项；零特征不会产生NaN。

原型detach；原型项仅经fc3之前的特征回传骨干，不给fc3参数产生直接梯度；分类器继续由原CE训练。CE/F复用 `clientTargetProj._objective`，一次总损失backward后执行原梯度裁剪和无动量SGD，无额外epoch/step。

日志的raw/weighted prototype与total_loss均为实际batch目标按SGD样本访问量加权的平均；batch内部原型项先除以该batch有效样本数。有效样本访问比例独立记录，不能把训练过程均值当作endpoint CE。

## 统计包与额外成本

真实执行in-process传包边界：8字节小端header长度＋UTF8 JSON header＋连续原始CPU tensor字节，schema=1。统计为float64 sums/int64 counts；下发为float64 prototypes/bool valid。包中含形状、dtype、客户端/模型下载指纹、轮次、数据与分组信息，客户端实际decode后创建detached库。

统计上传字节逐客户端计算为 `len(packet)`；下发的是一个完整K库包给每个客户端，合计N×len(library_packet)。这是真正序列化包的大小，不是估算张量数，也不声称测量TCP/IP等物理网络流量。各成本与原完整模型上下行分开。

采集forward的样本数/batch、采集/合并/分发耗时与SGD访问量/优化步数分别记录；不把一次统计采集称为新增训练epoch。

## 输出与诊断

使用原独立日志/CSV/JSON/H5/最终模型目录及日志ZIP，新增：

- `prototype_sources`：每轮、每列、每类的group/outside/invalid，candidate来源、客户端ID/数量、样本数、单位化前的**均值q范数**、无效原因、分组/采集/类库指纹。
- `prototype_collection`：每客户端/列的归属、容量、实际低秩下载状态及完整服务器指纹、采集样本/前向batch、零特征计数/耗时。与sources按轮次、列及source_client_ids关联，追溯每类来源模型。
- `prototype_uploads/prototype_rounds`：实际包字节、上传及广播总量、全体屏障、有效/无效类别数、分阶段成本。
- `prototype_training`：每客户端每列请求/有效配置、来源轮次、类库指纹、CE/F/raw/weighted prototype/total、有效样本比例及原SGD预算。
- `prototype_client_directions/prototype_capacity_directions`：同列同类本地unit-feature均值与最终原型的余弦，保留来源角色/样本数；容量汇总按样本数合并，范围是该容量所有有该类的客户端（含组外）。没有用U/V范数代替特征尺度。

以上有CSV/JSON及对应H5 JSON表，`class_prototypes.jsonl`在采集、合并及每客户端训练完成时持续追加。必要来源/成本不因方向诊断开关关闭而省略。

运行/最终manifest与实际模型payload含class_prototypes协议。最新实际类库另存 `Prototype_library.packet` 到checkpoint与最终模型目录，可用 `utils.multi_group_proto.decode_packet(bytes,'library')`读取；明确来自最后一次训练**之前**的下载状态。客户端预测仍只加载原普通低秩模型，不需要类库。日志ZIP不含这个包、逐样本特征或模型副本。

## 四条31轮命令

完整变量定义和 `declare -a COMMANDS` 见 [multi_group_proto_commands.sh](../system/multi_group_proto_commands.sh)。服务器system目录中，把它放到原调度器COMMANDS配置处或在那里source，避免被旧数组覆盖：

```bash
export PROTO_RUN_TAG=K3_original_capacity_proto31_seed0
export PROTO_DATA_ROOT=../dataset
source multi_group_proto_commands.sh
declare -p COMMANDS
```

这里只定义命令，不执行训练。四组为无原型、λ=.01/.1/1，GPU示例0/1/2/3；全部gr30实际31轮、原容量、low_rank_recovered初始化秩.15、每模型5epoch、SGD .005/batch16/clip10/F .001、群内.50、softmax温度.20、开始轮次2，其他协议一致。目录默认 `system/multi_proto_runs/$PROTO_RUN_TAG/`，按系数独立。

对照首轮原子发布共享分组，其余三组完成相同首轮后有限等待读取。不因原型参数不同造成分组不兼容；也不会接受容量/初始化/数据不同的旧分组。可先设 `PROTO_GROUP_FILE=/absolute/original/fixed_groups.json` 和 `PROTO_REUSE_GROUP_FILE=1`，让四组都读已有兼容分组。重复批次用新tag，并清空或重设同shell里派生的ROOT/GROUP变量。

主结果仍是固定所属模型的全员测试样本量加权post-local序列，保留pre、群组/容量/逐成员以及Final/最早Best/Last10/20。原型数据和诊断不使用测试信息，不自动选择λ或延长预算。

## 验收与限制

```bash
# 仓库根目录
python -m unittest discover -s tests -p test_multi_group_proto.py -v
python system/verify_multi_group_proto.py --output system/temp/multi_proto_acceptance_NEW --jobs 2
```

后者使用独立合成目录、原宽度SVD CNN、20客户端/K3/每模型5epoch/两轮，覆盖固定5d6103a旧train参照、零λ、Projection、仅群内Avg和关闭方向日志。参照需要Git对象5d6103a。`--audit-only`仅重读现有报告/各格式；默认核验后清理自己目录内的合成模型，`--keep-models`保留。

验收结果见 [multi_group_class_prototypes_acceptance.md](multi_group_class_prototypes_acceptance.md)。本地没有真实pat_20训练数据；合成结果不代表真实准确率收益。没有启动31轮真实训练、200轮实验或推送，没有修改环境和绘图。

新增的原始特征 MSE、无锚点 Z2 和 ReLU 前特征对照见 [独立损失比较说明](multi_group_proto_loss_comparison.md)。本页原 Cosine 默认及 31 轮命令保留。
