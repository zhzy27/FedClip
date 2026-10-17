# CNN-512公共对照验收（2026-10-17）

基于target_proj / `2af9cd2`。不改环境，不推送，不启动正式训练；
`system/plot_hyperparameter_sensitivity.py`未修改、未纳入本次提交。

## 实现与边界

- 共享原CNN-512：`common.models.CNN_512`，2,691,860参数，含共享分类器。
  原类路径兼容；FedFewS和TargetProj由同一个full-model factory创建初值。
- `--cnn512_control`显式统一协议；完整TargetProj支持avg/projection_softmax，
  下载按参数名/形状复制，上传直接完整参数，完全绕过Model_Distribe/SVD/恢复。
- 原Projection核未修改。FedFewS标准损失、两层权重、选择和增量聚合未修改；
  CNN-512新增按名称校验的更新分支，参数计算和接收顺序保留。
- 可选`--local_grad_clip`，旧TargetProj缺省10、旧FedFewS缺省不裁剪；新对照10。
  新对照测试batch16/shuffleFalse，原FedFewS测试shuffleTrue和original协议保留。
- 新主指标均为C0普通post-local；FedFewS从已采集计数提取，保留原全员主序列和诊断。
  新对照TargetProj通用H5序列改记C0，不额外执行旧全员评价；旧路径不变。
- 新对照预算：Avg/Projection每人5数据epoch；FedFewS每人3×5=15。
  统一模型和每模型训练协议，不声称计算预算相等。

## 环境

现有`D:\miniconda\envs\htfllib\python.exe`，Python3.11.14，PyTorch2.0.1+cu117，
CUDA runtime11.7/cuDNN8500，NumPy1.26.4，SciPy1.16.3。
验收为CPU、2线程；未更改依赖或CUDA/backend默认设置。
可选模型包的既有PyTorch版本提示不影响本次CNN路径。

## 验收结果

| 验收 | 结果 |
|---|---|
| `test_cnn512_control.py` | 7/7通过 |
| `test_fedfews_native.py` | 7/7通过 |
| `test_shared_primitives.py` | 8/8通过 |
| `test_shared_target_runtime.py` | 1/1通过；20人异构低秩Avg/Projection，两轮，与189bacd参照上传、服务器、批次和RNG一致 |
| 兼容FedFewS c73ec5b → 当前原生普通CNN | 20人、非均衡、K3、三轮；覆盖跳过评价与末轮强制评价；499事件PASS；最大参数/标量误差均0；RNG/批次验收通过 |
| 三个新真实main入口 | 各20人、两轮、每模型5epoch，非均衡合成样本；全部PASS |
| Bash启动数组语法和diff空白检查 | 通过 |

新单元测试验证：原CNN构造/参数名/形状/值、固定输入forward和构造后RNG逐元素一致；
旧checkpoint全局类路径可解析；隔离seed0 factory不推进随机流；同批次且实际触发裁剪时，
TargetProj普通训练和FedFewS K模型的单次SGD与独立标准SGD逐元素一致；
Avg解析加权和、含冲突的Projection公式、FedFewS按名称完整模型分支的双层权重增量公式；
非法配置拒绝，旧默认不启用新配置。

新smoke把TargetProj的Model_Distribe和恢复入口替换为失败哨兵，全程未触发。
三组真实主入口都完成C0评价、CSV/JSON/H5导出及最终完整模型重新加载。
每客户端每轮实际训练batch/样本访问量检查通过；FedFewS每模型5epoch、K3。

三个运行的初始参数SHA256完全相同：

`4e45a927113fb911515381893caca5578bc807181092572447927a98f51b058c`

三组所有40个train/test分片的路径、文件hash、标签分布、解码后张量指纹及格式策略一致。
共同模型所有初始化副本（含FedFewS三服务器和60客户端模型）均校验同一hash。

机器可读报告（本地忽略目录，未纳入Git）：

- `fedfews_native_runs/cnn512_control_final_cpu/acceptance_report.json`
- `fedfews_native_runs/cnn512_legacy_regression/acceptance_report.json`

前两次开发smoke定位过CLI拼写和C0通用H5保存触发条件，均修复后重新验收；
最终报告只引用`cnn512_control_final_cpu`。失败smoke目录保留。

## 复验与正式命令

```bash
# 仓库根目录，现有环境
python -m unittest discover -s tests -p test_cnn512_control.py -v
python -m unittest discover -s tests -p test_fedfews_native.py -v
python -m unittest discover -s tests -p test_shared_primitives.py -v
python -m unittest discover -s tests -p test_shared_target_runtime.py -v

# system目录，全新的输出目录；仅隔离合成smoke
python verify_cnn512_control.py --output ../fedfews_native_runs/cnn512_accept_cpu --device cpu
python verify_fedfews_native.py --output-dir ../fedfews_native_runs/cnn512_plain_regression --num-clients 20 --rounds 3 --eval-gap 10 --batch-size 4 --local-epochs 1 --device cpu --threads 2 --atol 0 --rtol 0
```

正式三项完整`declare -a COMMANDS`见 `system/cnn512_control_commands.sh`，
三组均gr100，分别独立checkpoint/logs/H5/final_models。说明与文件语义见
`system/CNN512_CONTROL.md`。脚本只定义命令数组，不会自动启动。

本机无真实`dataset/Cifar100/pat_20`分片，因此未验收真实数据性能、未测CUDA逐位误差，
也未据合成准确率声称任何算法改善。服务器运行时须核对各组data_manifest及初始化hash。
