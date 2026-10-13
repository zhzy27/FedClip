# Self12.5% ProjectionSoftmax + 低秩本地近端约束

本次仅在FedTargetProj的projection_softmax中加入本地近端项。
非零mu要求C0、projection_self_weight=.125、原完整训练集；
不用于DWA、APA、群体或meta模式。服务器仍是原global-delta全模型Projection，
只修正冲突helpers，评分仍用投影前余弦、tau=.2；C0质量.125、helpers质量.875。
模型正常上传、恢复、共享服务器下发、评价与原算法相同。

独立参数 --target_prox_mu 默认0，--target_prox_scope 默认helpers，另支持all。
mu需有限且非负。旧 -mu 完全没有连接到这个新功能；旧参数默认值不会启用它。
helpers只约束C1-19；all同时约束C0。非零mu的错误模式/自身权重/目标/holdout组合在启动前拒绝。

## 参照与目标

每次本地训练加载本轮真实下载的低秩模型，在任何optimizer step前，
按named_parameters中的requires_grad参数建立detach().clone()锚点。
锚点包含低秩U/V、分类头和其他可训练参数，不包含buffers或冻结参数，
5个epoch内不重置、不随SGD变化，下一轮重新建立。
不使用上一轮post-local、C0训练后权重或服务器未分解full-W作为参照。

每batch目标 CE + regular_lamda*Frobenius + mu/2*sum((theta-anchor)^2)；
参数平方差求和，不按参数数量平均。当前参数可微，anchor无梯度。
三项一起backward后沿用总梯度norm10的裁剪和原SGD lr=.005，无新增epoch。
这是低秩坐标约束，不能解释为恢复full-W后的参数距离。

mu=0和helpers中的C0直接使用原train路径，不建立训练锚点、不调用prox loss。
原_objective和train_metrics仍是CE+原Frobenius，测试准确率不加入近端项。
启用本批实验的日志观察器用CPU只读加载本轮普通训练前后checkpoint来测量距离，
保护Python/NumPy/Torch CPU/CUDA随机流；不在训练GPU上保留额外诊断模型，
不改写checkpoint或buffers。零mu对照也可报告结束距离，但训练本身没有新Tensor算术。
未传近端参数时不运行该观察器，也不新增本批目录标签或指标文件。

## 结果与日志

主要结果仍是普通C0的5epoch post-local：
final、best/最早轮次、last10、last20及对应计数；不足10/20次时使用已有次数。
同轮聚合后准确率仍作为后续初始化的观察项，权重/余弦/投影删除及有效helpers诊断未替换。
所有这些结果必须在相同的201轮预算下比较，不能直接把新201轮和旧101轮结果解释为近端收益。
已有完全相同配置的201轮无prox对照可以复用，但不同预算的旧结果仅作历史参考。

本批实验名、checkpoint根目录、metrics日志目录、H5目录、最终模型根目录
都会带prox_<scope>_mu<value>标签。显式mu0也有独立标签，避免对照相互覆盖。
日志沿用独立training_logs和自动ZIP，不和模型文件混放。

- target_prox_local.csv/json：每round/client分别记录CE、已乘lambda的Frobenius、
  已乘mu/2的prox平均损失、实际启用标志、effective mu、低秩结束距离/平方距离、
  最终prox值、普通训练秒数、epoch/batch/sample数量及参数总数。
- target_proj_metrics.csv/json：原C0和聚合指标，新增mu、scope、启用人数、last20。
- projection_softmax_clients.csv/matrices.json：原全模型评分、权重和投影诊断。
- H5的target_projection/proximal_local_training：按round/client的损失、低秩距离、
  时间矩阵；target_projection属性包含mu/scope/实际启用IDs和last20汇总。

global-delta中的delta_norm是recovered full-W对聚合前服务器的距离。
low_rank_start_distance是训练低秩参数相对本轮真实起点的距离。
两者属于不同空间与不同参照，不能混为同一指标或直接比较大小。
prox平均值是各batch更新前损失的样本加权平均；end_proximal_loss另算最终距离对应值。

## 五组正式启动参数

完整declare -a COMMANDS数组：system/target_prox_commands.sh。
从system/使用原run_now式runner手动启动；source文件只定义数组，不会训练。
五组全部helpers，mu=0/.001/.01/.1/1，seed0、Cifar100/pat20、20人全参与、
Decom_CNN-5-512、batch16、lr=.005、local5、Frobenius=.001、-gr200（实际201轮）。
GPU0-4仅示例，请替换成空闲设备。checkpoint/H5/final/log根目录全部独立。
不自动推送或运行正式长实验。

启动器同样支持参数，可以先检查命令：

~~~bash
python run_target_proj.py --modes projection_softmax --projection_self_weight .125 \
  --target_prox_mu .01 --target_prox_scope helpers --rounds 200 --seeds 0 \
  --device-id 0 --dry-run
~~~

helpers扫描完成后，若手动选择同系数检查all，可把scope改为all。
以下仅演示接口，不说明.01是已验证最佳系数：

~~~bash
python run_target_proj.py --modes projection_softmax --projection_self_weight .125 \
  --target_prox_mu .01 --target_prox_scope all --rounds 200 --seeds 0 \
  --device-id 0 --dry-run
~~~

## 验证范围

小网络检查起点prox=0、梯度mu*(theta-anchor)、不归一化、
head/U/V覆盖、冻结参数/buffers排除、锚点独立且每轮重建、
helpers下C0与显式零mu的参数/RNG完全一致、旧-mu无作用和原评价目标不变。
短轮实际异构Hyper_CNN_512覆盖普通旧对照、显式mu0、helpers正mu、all正mu，
检查正常SVD下载后锚点匹配、每人本地5epochs、上传/聚合/日志/H5及buffer保护。
烟测使用合成数据和较小卷积通道数，仅验证机制，不声称CIFAR100目标准确率改善。
具体为20客户端、100类head、原五种容量比例、train16/test4、n_kernels=4、两轮，
每轮每人执行5个完整epochs。新机制与旧模式共163项相关检查通过（含12项新增测试）。
显式零mu与旧对照在两轮全部20人上传参数、服务器聚合参数上逐元素完全一致，
训练随机流一致；正mu的helpers和all都完成正常训练与导出。
完整201轮正式扫描尚未启动。
