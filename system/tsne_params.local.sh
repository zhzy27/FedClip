# 本机实验参数，不受 Git 管理；同步到服务器时手动传这个文件。
# 运行：bash system/run_tsne.sh；最新选项说明见 tsne_params.example.sh。
# 保留 ${NAME:-value} 写法可继续用同名环境变量临时覆盖。

# 模型、数据与客户端（数据划分必须与训练一致）
MODEL_DIR="${MODEL_DIR:-}"
DATASET="${DATASET:-Cifar100}"                    # 自动确定类别数
ALGORITHM="${ALGORITHM:-FedCLIP}"
# 模型族：CIFAR CNN | Tiny CNN | ResNet
# FedCLIP：Decom_CNN-5-512 | Decom_CNN-5-512 | Decom_resnet18_5
# FedSPU：SPU_CNN1 | SPU_CNN1-tiny | SPU_ResNet18_1
# PFedAFM：CNN-5-512-AFM | CNN-5-512-AFM-tiny | ResNet18-5-AFM
# 其他常用方法：CNN-5-512 | CNN-5-512-tiny | ResNet18-5
MODEL_FAMILY="${MODEL_FAMILY-Decom_resnet18_5}"
PARTITION="${PARTITION:-dir}"                    # dir / pat / exdir
DIR_ALPHA="${DIR_ALPHA:-1.0}"
CLASS_PER_CLIENT="${CLASS_PER_CLIENT:-20}"
MODEL_SOURCE="${MODEL_SOURCE:-client}"          # client / server
SPLIT="${SPLIT:-test}"                          # train / test
CLIENT_IDS="${CLIENT_IDS-0}"                    # 单个、逗号列表、范围；空字符串表示全部
AUTO_BEST_CLIENT="${AUTO_BEST_CLIENT:-0}"        # 1=按特征分离度自动选择

# 样本量与 t-SNE
MAX_BATCHES="${MAX_BATCHES:-0}"                 # 0=不限
MAX_SAMPLES_PER_CLIENT="${MAX_SAMPLES_PER_CLIENT:-0}"  # 0=不限
PERPLEXITY="${PERPLEXITY:-30}"
MAX_ITER="${MAX_ITER:-1000}"
TSNE_LR="${TSNE_LR:-200}"
SEED="${SEED:-0}"

# 图形与输出
OUTPUT_DIR="${OUTPUT_DIR:-}"
POINT_SIZE="${POINT_SIZE:-18}"
POINT_ALPHA="${POINT_ALPHA:-0.7}"
SHOW_LEGEND="${SHOW_LEGEND:-1}"
MAX_LEGEND_CLASSES="${MAX_LEGEND_CLASSES:-20}"
SAVE_EXCEL="${SAVE_EXCEL:-0}"

# 训练目录匹配
NUM_CLIENTS="${NUM_CLIENTS:-20}"
JOIN_RATIO="${JOIN_RATIO:-1.0}"
NIID="${NIID:-1}"
FINAL_MODEL_ROOT="${FINAL_MODEL_ROOT:-./final_models}"

# 自动选择客户端（开启时生效）
SELECTION_SCORE="${SELECTION_SCORE:-silhouette}"
SELECTION_METRIC="${SELECTION_METRIC:-euclidean}"
SELECTION_MAX_BATCHES="${SELECTION_MAX_BATCHES:-40}"
SELECTION_MAX_SAMPLES="${SELECTION_MAX_SAMPLES:-1200}"

# 运行设置
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"
