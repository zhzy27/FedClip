# 单目标 Z2 验收报告

参照提交 `89d442c`。本次不修改旧初始化、SVD或聚合算法；不启动正式长训练、不推送、不安装依赖。绘图保持原状。

## 新增检查

- `tests/test_target_z2.py`：8项通过，日志 `system/temp/target_z2_unit_20261029_final.log`。
- `tests/test_target_z2_runtime.py`：三种聚合各跑固定89d442c参照、未传参数、显式零系数，共9个两轮轨迹。N5覆盖五档容量、小宽度低秩CNN、每模型1epoch；本地参数、恢复上传、聚合、实际batch、RNG与summary严格一致。日志 `system/temp/target_z2_runtime_20261029.log`。

检查正负原始特征未被head原地ReLU改写、Z2值及梯度等于直接平方均值、骨干得到梯度而fc3无直接Z2梯度、一次batch一次forward、首轮保持旧路径、hook异常清理、日志开关不改变客户端结果。`_objective()`仍为CE/F双返回，评价不加入Z2。固定冲突快照的Softmax权重严格相同，投影／无投影结果确有差别。主程序和启动器的合法／非法配置及新参数转发通过检查。

## 原宽度真实入口合成验收

`system/verify_target_z2.py` 使用实际`main.py`，原宽度16的Hyper_CNN_512、原五档容量、原Model_Distribe初始化、20客户端、两轮、local5、batch16、F=.001、clip10。仅合成train2/3张、test1/2/3/4张，不读取真实数据。三模式和Projection diagnostics=0，共4个案例。

最终运行输出 `D:/z2_1029/acceptance_report.json`，状态PASS、退出码0。轻量收据另存 `system/temp/target_z2_verified_20261029/`。每例40个真实本地训练任务、200个SGD batch、500次训练样本访问，21个普通客户端／服务器模型保存重载通过。

- 三种模式实际初始服务器参数SHA及数据文件SHA一致；初始化协议没有替换为多群组版本。
- 第一轮有效lambda=0；第二轮20个客户端全部lambda=1。每客户端每轮只训练一次5epoch，实际forward数逐batch核对一致。
- Avg独立样本量加权和、Softmax及Projection原核结果逐参数误差0。同一实际上传快照，两种Softmax权重严格一致。
- 普通上传checkpoint在恢复／聚合后评价前后hash不变，日志开启／关闭的完整上传、聚合、batch、RNG、C0summary严格一致。
- CSV／JSON／JSONL／H5／manifest的训练记录、主指标和summary一致。H5根主曲线为C0 post-local，旧诊断另存；最终预测模型逐元素与当前checkpoint一致并复现forward。
- 日志ZIP不含模型或原型／数据包。

首轮验收脚本曾使用过长的Windows输出路径，触发HDF5路径长度限制；ZIP查找也需按实际带Z2后缀的日志根目录定位。修正验收路径查找并在短目录完整重跑，不修改训练数学或放宽参数误差。第一次目录 `system/temp/target_z2_acceptance_20261029/` 仅保留失败日志，模型二进制已清理。

损失总和核对允许float32加法／归约顺序舍入，使用 `8*float32_epsilon*max(1,abs(total_loss))`；只作用于日志中的分别归约项之和，不放宽参数、上传、聚合及RNG的严格相等。

## 旧模式回归与限制

**142项旧测试全部PASS**，受影响旧单目标、Projection、Softmax、Prox、local head、FedRep、启动器以及多群组SVD/KD/Z2结果见 `system/temp/target_z2_regression_20261029.log`。旧多群组`Features/z2_loss`代码没有修改。

短目录验收及第一次失败目录中的模型二进制均已清理，最终短目录保留约3.3MiB日志、数据与结果，未删除用户原数据或模型。

本地缺少真实pat_20训练分片，仅完成CPU合成验收，不判断真实准确率，也不宣称与历史约55%的初始化／随机轨迹严格对齐。

[实现及日志说明](target_z2.md) · [三条完整COMMANDS](../system/target_z2_commands.sh)。原绘图SHA256保持 `90ff3a5c9ff8615ce28953d89598a6295ad6b513ba235293309e6875d429f95c`。
