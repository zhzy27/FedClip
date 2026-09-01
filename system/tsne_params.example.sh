# 受 Git 管理的默认值与参数说明；实际实验配置请改 tsne_params.local.sh。
# 新增参数时，旧 local 文件缺少的参数会自动采用这里的默认值。
# 相对模型/输出路径以 system 目录为基准。

# ==================== 1. 最常改：模型、数据与客户端 ====================
# 指定具体 run 目录（直接包含 Client_<id>_model.pt）。多组消融时建议填写。
# 留空或指定含 runs/ 的分组目录时，自动选修改时间最新的 run，不是最高精度。
MODEL_DIR="${MODEL_DIR:-}"

# 数据划分必须与训练一致，即使指定 MODEL_DIR 也不会自动读取这些参数。
DATASET="${DATASET:-Cifar100}"                    # Cifar10 / Cifar100 / TinyImagenet；自动对应10/100/200类
ALGORITHM="${ALGORITHM:-FedCLIP}"                # 方法名与训练保存目录一致
# 模型族对照（列顺序：CIFAR CNN | Tiny CNN | ResNet）：
# FedCLIP：Decom_CNN-5-512 | Decom_CNN-5-512 | Decom_resnet18_5
# FedSPU ：SPU_CNN1 | SPU_CNN1-tiny | SPU_ResNet18_1
# PFedAFM：CNN-5-512-AFM | CNN-5-512-AFM-tiny | ResNet18-5-AFM
# FedKD/FedTGP/FedRE/FedGH/LG-FedAvg/FedProto/FML/FD/FedGen/FedMRL：
#         CNN-5-512 | CNN-5-512-tiny | ResNet18-5
MODEL_FAMILY="${MODEL_FAMILY-Decom_resnet18_5}"
PARTITION="${PARTITION:-dir}"                    # dir：Dirichlet；pat：病理；exdir：扩展Dirichlet
DIR_ALPHA="${DIR_ALPHA:-1.0}"                    # dir/exdir 系数，如0.1、0.5、1.0
CLASS_PER_CLIENT="${CLASS_PER_CLIENT:-20}"       # pat/exdir 每客户端类别数
MODEL_SOURCE="${MODEL_SOURCE:-client}"          # client：本地训练后模型；server：服务器模型
SPLIT="${SPLIT:-test}"                          # train / test；此版本不支持 both
# best=从全部客户端中选所选 train/test 准确率最高者；沿用绘图样本上限，并列选较小编号。
CLIENT_IDS="${CLIENT_IDS-0}"                    # "best"、单个 "0"、列表 "0,5,10"、范围 "0-19"；"" 表示全部
AUTO_BEST_CLIENT="${AUTO_BEST_CLIENT:-0}"        # 1=按特征分离度选一个；CLIENT_IDS=best 时此开关不生效

# ==================== 2. 特征与 t-SNE 设置 ====================
# classifier_input=分类器实际输入（CIFAR CNN 会包含 head 中的 ReLU）；raw_base=旧脚本的 base 原始输出。
FEATURE_SPACE="${FEATURE_SPACE:-classifier_input}"
PCA_COMPONENTS="${PCA_COMPONENTS:-50}"          # t-SNE 前的 PCA 维数；0=关闭 PCA
REPRESENTATION_DIAGNOSTICS="${REPRESENTATION_DIAGNOSTICS:-1}" # 1=输出完整表征诊断，0=关闭
DIAGNOSTIC_KNN_K="${DIAGNOSTIC_KNN_K:-5}"       # leave-one-out k-NN 的 k

# ==================== 3. 样本量与 t-SNE 设置 ====================
# 0=不限；正数=每客户端batch上限；-1=旧默认（未指定客户端时40，否则不限）。
MAX_BATCHES="${MAX_BATCHES:-0}"
# 每客户端样本上限，0=不限；与 MAX_BATCHES 先到者为准，不是类别平衡采样。
MAX_SAMPLES_PER_CLIENT="${MAX_SAMPLES_PER_CLIENT:-0}"
PERPLEXITY="${PERPLEXITY:-30}"                   # 邻域尺度，如15/30/50；应小于样本数，不是类别数
MAX_ITER="${MAX_ITER:-1000}"                     # t-SNE 最大迭代数，至少250；不是通信轮数
TSNE_LR="${TSNE_LR:-200}"                        # t-SNE 学习率，正数，不接受 auto
SEED="${SEED:-0}"                               # 比较不同实验时固定

# ==================== 4. 图形与输出 ====================
# 留空自动保存至 T-SNE-legacy/；同目录重跑会覆盖同名 PNG/PDF/CSV。
OUTPUT_DIR="${OUTPUT_DIR:-}"
POINT_SIZE="${POINT_SIZE:-18}"                   # 散点大小
POINT_ALPHA="${POINT_ALPHA:-0.7}"                # 透明度0~1，不是Dirichlet系数
SHOW_LEGEND="${SHOW_LEGEND:-1}"                  # 1=显示图例，0=隐藏
MAX_LEGEND_CLASSES="${MAX_LEGEND_CLASSES:-20}"   # 超过此类别数就不显示图例
SAVE_EXCEL="${SAVE_EXCEL:-0}"                    # 1=额外保存xlsx，0=不保存

# ==================== 5. 通常不变：训练配置与目录匹配 ====================
NUM_CLIENTS="${NUM_CLIENTS:-20}"                 # 训练时客户端总数
JOIN_RATIO="${JOIN_RATIO:-1.0}"                  # 训练时参与率，仅用于匹配目录
NIID="${NIID:-1}"                               # 1=非IID，0=IID；与训练一致
FINAL_MODEL_ROOT="${FINAL_MODEL_ROOT:-./final_models}"  # MODEL_DIR 留空时的查找根目录

# ==================== 6. 特征分离度评分（CLIENT_IDS=best 时不使用） ====================
SELECTION_SCORE="${SELECTION_SCORE:-silhouette}"        # silhouette=轮廓系数；separation=类间/类内距离比
SELECTION_METRIC="${SELECTION_METRIC:-euclidean}"        # euclidean/cosine，仅用于轮廓系数
SELECTION_MAX_BATCHES="${SELECTION_MAX_BATCHES:-40}"     # 评分时每客户端batch上限，0=不限
SELECTION_MAX_SAMPLES="${SELECTION_MAX_SAMPLES:-1200}"   # 轮廓系数采样上限，0=不限

# ==================== 7. 运行设置 ====================
DEVICE="${DEVICE:-cuda:0}"                       # 指定 CUDA 设备或 cpu
BATCH_SIZE="${BATCH_SIZE:-16}"                   # 特征提取batch大小
PYTHON_BIN="${PYTHON_BIN:-python}"
