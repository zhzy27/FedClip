# 离线整网扫秩与可配置中间宽度

`system/svd_capacity_sweep.py`只做推理，不调用训练或服务器聚合，不导出替换预测模型。输入是普通post-local低秩预测checkpoint，不是完整服务器权重或恢复上传缓存。第一批原生容量要求`.90`，RGB32、100类、原卷积通道和512维分类特征；不支持的结构明确失败。

## 输入与来源核验

从服务器`system/`运行`svd_capacity_sweep_commands.sh`中的两条完整命令。数组只定义，不自动执行。L/H来源目录根据`11-17.tar.gz`内的实际训练日志核对：

- L：`svd_first_band_runs/pat20_seed0_init090_R100_20261117_v1/L`
- H：`svd_first_band_runs/pat20_seed0_init090_R100_20261117_v1/H`

默认指定R100和客户端`0,5,10,15`，读取`Client_i_model_0.pt`。也可用`--completed-round`/`--source-round`、`--client-ids`、`--model-id`指定。优先同目录文件、其次`final_models/`、再检查运行目录中的其他同名文件，**必须匹配指定轮次和角色**；相同优先级存在内容不同的候选时要求显式指定。缺失会逐一列出预期文件及发现的实际轮次，不用最后一轮冒充指定轮次。

显式文件示例：`--client-ids 0,5 --checkpoint 0=/path/Client_0_model_0.pt --checkpoint 5=/path/Client_5_model_0.pt --manifest /path/manifest.json`。只测一人时可直接传`--checkpoint /path/file.pt`。通常manifest与最终导出checkpoint相邻；非相邻时必须显式提供，所有选中模型须绑定相同内容的运行manifest。

核对客户端、列号、post-local角色、完成轮次、结构、实际容量、初始化/分组/容量协议（源文件已提供时），严格加载原state_dict。数据通过公共NPZ reader和完整运行清单审计：config文件、原分片SHA256、解码x/y指纹、样本数均须匹配。可迁移`--data-root`、`--data-subdir`，不改数据、不根据目录名推断划分。旧checkpoint缺少逐客户端数据绑定时，如实记录`independent_same_run_manifest_audit`；已有的顶层/下载上下文数据指纹逐一核验，绝不把缺失字段伪装成直接绑定。

## 扫描及统计口径

1. 首先评价原始`.90`checkpoint，作为准确率参照。
2. 在副本上调用原恢复函数及原下发分解，每层`conv2/fc1/fc2`各做**一次SVD**。卷积排列、平方根因子和bias处理完全复用现有代码；记录同一组U/S/V的指纹。
3. 六档均从这一次谱取因子前缀，不再次SVD、不修谱、不重新初始化头。请求列表可通过`--rank-rates`改变，首版范围`0<ratio<=.90`；重新分解的`.90`始终作为数值往返对照保留。
4. 全段`eval/no_grad`，保护输入模型及RNG；实际逐样本访问顺序记录并核对，全部档位测试同一份数据各一次。默认batch16、workers0、不打乱。

`native .90`和`resvd .90`分别输出，不把数值往返结果冒充原准确率。记录logits最大绝对误差、相对Frobenius误差及预测变化数，默认`torch.allclose(atol=1e-4,rtol=1e-4)`，可显式配置；失败保留报告，不自动放宽。预测变化数按实记录，近并列logits可能在数值容差内改变argmax。

CUDA离线扫描显式使用`torch.linalg.svd(..., driver="gesvd")`，CPU保持默认驱动。驱动仅在`shared_download`内按实际分解矩阵的设备选择，三层各分解一次，六档仍共用前缀；不通过关闭TF32或放宽容差规避往返误差。`manifest.json.svd_driver`和`shared_svd.json`的每客户端/每层记录实际驱动（CPU为null）及SVD设备。超差时除JSON外，异常消息也列出最大误差、相对误差、预测变化数和容差。

底层`Hyper_CNN_512.decom_larger_model`、`Decom_COV`和`Decom_LINEAR`新增可选末尾参数`svd_driver=None`。所有原训练调用保持缺省值，`torch.linalg.svd`不额外收到`driver`关键字，原默认算法及RNG保留；没有全局替换SVD。L/H原CUDA命令无需新增驱动选项。若先前失败的输出目录已存在，仍拒绝覆盖，应另指定空输出目录，保留原失败报告。

参数量从实际实例`model.parameters()`去重计数，包含U/V、全部bias、未分解层和分类头，不等于通信量或峰值内存。宽度2000的验收值为：

| 秩率 | 全模型参数量 |
| --- | ---: |
| .90 | 3,246,372 |
| .60 | 2,181,764 |
| .45 | 1,649,460 |
| .35 | 1,295,428 |
| .25 | 941,396 |
| .15 | 587,364 |

准确率内部单位为fraction，掉点是相对**本来源原始checkpoint**的百分点差；负掉点表示提升。汇总固定同一客户端集合，使用`sum(correct)/sum(samples)`。`--max-drop-pp .5`只作为报告参考线，不训练、不调参。输出已测档位中满足参考线的最小参数量、相邻已测点及全部跨线区间；不假设掉点单调，也不声称未测中间档可行。

## 输出与配对比较

两条命令分别输出到`svd_capacity_sweep_runs/pat20_R100_20261125_v1/{L,H}/`，已存在目录会拒绝覆盖：

- `client_results.csv/json`：来源、轮次、客户端、native/resvd、实际三层秩、宽度、完整参数量、correct/samples、纯CE、准确率、掉点及样本顺序指纹。
- `cohort_results.csv/json`、`summary.json`：同客户端集合的测试样本量加权汇总，以及逐客户端/整体的参考线判断。
- `parameters_vs_drop.png`：实际参数量—掉点，单人曲线及整体曲线；native单独用菱形标示。
- `roundtrip_checks.json`、`shared_svd.json`：往返误差、预测改变数、源模型/RNG保护及三层共同谱证据。
- `manifest.json`、`data_manifest.json`：请求/有效配置、源文件/代码/环境、数据和初始化来源、宽度、输入文件未变核验。不保存模型副本。

两次完成后可只读合并L/H，客户端集合、完成轮次和数据指纹不一致会拒绝：

```bash
python svd_capacity_sweep.py --device cpu --compare-results svd_capacity_sweep_runs/pat20_R100_20261125_v1/L svd_capacity_sweep_runs/pat20_R100_20261125_v1/H --output-dir svd_capacity_sweep_runs/pat20_R100_20261125_v1/paired
```

生成双来源曲线、原始汇总和`paired_comparisons.csv/json`（H减L的同档准确率、原生准确率及掉点差）。各来源仍以自己的native为掉点基准。

## 为缩宽重训预留的参数

`Hyper_CNN_512(..., hidden_width=1536)`同时设置fc1输出与fc2输入。主入口用`--cnn_hidden_width 1536`，默认2000。这个选项不会缩窄卷积、fc1输入800、fc2输出512或分类头；后续真正重训应使用独立目录。本次未启动缩宽正式训练。

公共构造、主入口模型描述、容量协议、有效配置及模型保存均贯穿实际宽度。非默认宽度写入`model_kwargs.hidden_width`；旧checkpoint缺省按2000严格加载。默认构造顺序、参数名称、state_dict、前向及构造后的RNG与751adbd相同；不在旧默认模型kwargs/容量协议中额外插入字段。

2000与1536均高于原秩计算的较小矩阵维度，因此六档三层实际秩一致。`.90`的参数量分别3,246,372与2,697,924。其他任意宽度未必保持秩不变，必须以实际层rank为准。扫描从checkpoint配置读宽度，不靠额外CLI覆盖；训练好的宽版/窄版可用同一入口扫描。

## 验收范围

详见`docs/svd_capacity_sweep_acceptance.json`。包括实际RGB32/100类模型的计数、两宽度秩和重建、冻结751adbd默认构造/前向/RNG对照、错误轮次/角色/数据绑定拒绝、非均衡加权及非单调参考线案例；两个宽度均通过真实main.py合成训练/保存/加载和真实扫秩CLI。合成数据只在独立临时目录，未修改真实分片。

复查命令：`python -m unittest discover -s tests -p test_svd_capacity_sweep.py -v`；入口验收：`python system/verify_svd_capacity_sweep.py --output /independent/empty/directory`。本机没有真实L/H checkpoint，尚未产出真实掉点结果。

CUDA精度修复验收见`docs/svd_capacity_sweep_cuda_acceptance.json`。复用已经由真实main.py生成的两种宽度合成checkpoint，对CPU/CUDA分别调用原扫描CLI，均导出native加六档及配对图。可复查：`python system/verify_svd_capacity_sweep.py --source-fixture /previous/synthetic/entry --output /new/empty/cuda_results --sweep-device cuda`（CPU改为`cpu`）。该命令只重跑离线推理，不重训源checkpoint。

本机RTX 3070 Ti Laptop GPU上，2000/1536宽度的CUDA往返最大误差分别为`3.69549e-6`/`2.56300e-6`，预测变化均为0；容差仍为`1e-4`。CPU结果与1d28412逐字段相同（新增驱动元数据引起的共享谱收据哈希除外），完整U/S/V指纹相同；另对冻结1d28412原模型分解、CE+F/裁剪/SGD及恢复上传参数/RNG做严格回归。合成验收不代表真实容量—准确率结论。未修改环境、真实数据或绘图，未启动正式训练。
