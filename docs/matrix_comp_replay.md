# 从 H R5 上传回放两轮：子组尾部矩阵补偿

这是独立离线诊断，不修改联邦入口、`all_tail` 或默认 Avg。入口 `system/matrix_comp_replay.py` 同时协调 H、Hplus、matrix 三支，确保同一客户端、同一轮使用相同训练批次和随机状态。**R6 和 R7 各执行完整的本地5 epoch**，之后结束，不生成 R8。

## 来源与时序

只读 H 的 R5 `Server_model_0.pt`、相邻 `manifest.json` 和同轮20份普通 `Client_i_model_0.pt`。可指定 `--source-run` 或 `--server-file`，客户端目录不同则用 `--clients-dir`。不存在时直接报错，不搜索其他轮次或用恢复上传冒充低秩预测模型。

校验 K1/N20、完整服务器/普通低秩客户端角色、完成轮次5、ID、实际容量与结构、`.90`恢复初始化及其初始权重指纹、数据文件和解码指纹、原CE+F.001、SGD.005/batch16/local5/clip10、共享头/joint/无原型。

兼容原 H 无原型正式入口的最终导出：`single_avg_prototype_usage.downloaded_model`可以为null，顶层可以没有数据指纹。已有的顶层和下载上下文指纹分别严格核对，两者同时存在时均须匹配。缺失直接绑定的客户端仅允许来自服务器所在的同一最终导出目录，并核对初始化、容量协议、恢复参数schema、分组指纹与分组协议；相邻服务器manifest的K1分组、训练数据和初始化上下文也必须一致。显式`--clients-dir`指定其他目录时，缺失直接绑定的文件仍报错。

旧格式的接受还必须等待20份上传集合复算Avg通过。其结论仅为“同一导出的上传集合复算通过，数据独立通过服务器清单核验”，不补写或假装检查点自身具有数据指纹。不能由此证明缺失字段的历史文件曾自行保存了训练数据身份。复制其他运行的不同权重即使元数据相容，也会在复算Avg时失败。

原函数私有恢复客户端模型，使用原 `aggregate_group_updates/group_avg_only` 按原样本量、确定顺序重算全员Avg，逐参数核对保存的 R5 服务器。默认 `atol=rtol=1e-5`，同时限制全局最大误差与Frobenius误差（判据写入报告）。重算时用已保存S5作为delta-Avg算术锚点；这与原S4锚点的浮点累加可能略异，实际误差必须通过上述检查，绝不自动放宽。

R6三支从同一份已核验的R5上传产生普通Avg。R7各自从本分支R6实际上传产生普通Avg。每次：

- H：普通Avg母模型，普通SVD。
- Hplus：普通Avg母模型，现有`all_tail`，不加1.5倍率上限。
- matrix：普通Avg母模型加下述矩阵补偿，完整矩阵统一普通SVD，再按容量截断。

R7额外只读评价各分支**当前干预之前、由自己的上传生成的普通Avg母模型**的普通SVD下载；H直接复用epoch0计数，其他两支只做推理，不训练反事实模型。该对照排除当轮再次修复的直接影响，保留各分支此前的服务器轨迹差异。

## 矩阵补偿

`utils/matrix_tail_compensation.py`只处理去重后的`conv2/fc1/fc2.weight`。卷积矩阵排列完全沿用原恢复的`[out*k,in*k]`规则。其他层、分类头和所有bias保持全员Avg。

每层按实际整数秩升序划段（fc1为120/200/280/296/720）。Wavg的最低档左右奇异子空间为初始前缀。每段从**原始完整恢复上传**算符合容量的子组样本量归一化Avg B，按要求双侧去掉已经选出的左右方向，对残差做SVD，取至多原定段宽的有效分量。T使用这些残差分量自己的奇异值和方向。所有`(1-q)T`加到同一个原始Wavg上，不将前段补偿写入后段子组Avg。

计算使用CPU float64，**默认数值秩阈值改为`max(shape) × eps(float64) × sigma_max(B)`**，其中B仍是投影前子组矩阵。最终阈值为`max(atol, rtol × scale)`；`--matrix-rank-rtol/atol`保留显式覆盖。上传dtype不再决定有效秩阈值。日志单列上传dtype、计算dtype/epsilon、实际阈值、计划/数值/实取秩、缺口和最后一个入选奇异值；原上传精度阈值及其会保留的数值秩仅作参考，不参与选择。不足不填噪声、不向下一段借位。QR仅维护实际选中方向的正交投影基，不改变T。完成后转换回母模型dtype，完整SVD可让补偿方向跨越任何原容量边界；不额外限制到最大客户端秩。

## 配对与预算

复用原本地训练器、原F、一次全模型裁剪和新建SGD。批次复用原`single_avg_proto.loader_seed(seed,client,loop)`（R6 loop5、R7 loop6）。每个客户端/轮次用稳定用途标识产生独立训练RNG，并在三支间恢复；不宣称恢复了历史源进程RNG。三支核对实际逐epoch批次及结束RNG指纹。模型构造、补偿和只读评价不推进训练随机流。

每支每轮的实际下载各执行一次原三层SVD，各容量取共同因子前缀，保持同母模型的方向一致。R6三支9次，R7三支及matrix的额外普通对照12次，合计21次原下发层SVD；Hplus的普通对照直接取修复时同次SVD保存的原谱因子。两轮三支共120个本地任务；每任务5 epoch，训练访问量和步数按真实分片记录。残差SVD、边界观测SVD和反事实推理是额外诊断成本，分开记录；不是训练epoch或额外SGD。

## 输出

新根目录下`H/`、`Hplus/`、`matrix/`各自有CSV/JSON及manifest；根目录保存配对比较与曲线：

- `source_validation.json`：20份客户端来源、直接绑定/旧格式兼容的ID列表、实际缺失/null字段、恢复上传指纹及重算Avg逐参数误差。Avg失败也保存差异，不先训练回放。
- `endpoints`、`cohort_endpoints`：每客户端epoch0–5纯CE/正确数/样本数，以及全员、C0、原五档固定成员加权汇总。
- `epoch_training`、`local_training`：训练过程CE/F/总损失、裁剪比例、真实预算和耗时，区别于endpoint CE。
- `comparisons`：主比较`matrix_minus_Hplus`、补充`matrix_minus_H`，含相对epoch0的差值变化；R7实际下载对自己普通母模型的瞬时差值。
- `compensation_segments`：子组成员、原训练质量q、权重、投影前/后能量、计划/数值/实取秩、阈值及补偿范数。
- `capacity_boundaries`、`compensation_spectrum`：全部容量边界两侧谱与比值、左右前缀子空间重合度、退化边界标记和完整逐秩谱。
- `batch_fingerprints`、`all_tail_downloads`、`aggregations`、`costs`：配对、旧修谱实际倍率、原核及成本记录。

准确率内部为fraction，图表为percent，差值文字为百分点。子空间重合度是`||U_before[:r].T U_after[:r]||F²/r`，符号/子空间内旋转不变；近重根边界单独标记，避免过度解释。

默认仅保存诊断表和曲线，临时低秩checkpoint和完整上传由拥有生命周期的临时目录清理。可用`--temp-root`指定大空间盘。输入源文件和数据末尾重新验哈希。

## 启动

从服务器`system/`使用`matrix_comp_replay_commands.sh`中的完整数组，只定义命令，不自动执行。默认来源H目录沿用五轮实验；若实际路径不同，显式替换`--source-run`。输出必须为尚不存在的独立目录。

```bash
declare -a COMMANDS=(
  "/data/zh/miniconda3/envs/htfllib/bin/python -u matrix_comp_replay.py -did 0 --device cuda --source-run svd_five_round_runs/pat20_seed0_init090_R5_20261115_v1/H --data-root ../dataset --dataset Cifar100 --data-subdir pat_20 --target-client-id 0 --threads 4 --output-root matrix_comp_replay_runs/pat20_S5_R6_R7_seed0_20261116_v2"
)
```

这是一条协调三支的命令，避免三个独立进程失去配对或重复读取来源。比较目录仍互相独立。真实R5产物不在本机，验收报告仅包含合成机制和真实结构检查，不代表真实准确率收益。未启动上述服务器实验。

本次兼容修复验收使用原`svd_five_round_commands.sh`的H命令，通过实际`main.py`在合成数据上完成R1–R5，再原样读取正式导出的服务器和20个客户端文件；不再用离线构造器补齐缺失字段。五轮正式训练代码、导出代码和后续R6/R7配对流程没有修改。`matrix_comp_replay_acceptance.json`保留6e2c732历史报告，本次结果另存`matrix_comp_replay_fix_acceptance.json`。
