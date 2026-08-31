#!/usr/bin/env bash
# 启动 T-SNE-Cifar-legacy-compatible.py：原始特征，不加文本锚点、不做归一化。
# 图左上角自动标注所选 train/test 绘图样本的准确率；多客户端按样本数汇总。
# 运行：bash system/run_tsne.sh；预览命令：bash system/run_tsne.sh --dry-run
# 修改下方配置即可，也支持同名环境变量覆盖；相对路径均以 system 目录为基准。
set -euo pipefail

# ==================== 1. 最常改：模型、数据与客户端 ====================
# 指定具体 run 目录（直接包含 Client_<id>_model.pt）。多组消融时建议填写。
# 留空或指定含 runs/ 的分组目录时，自动选修改时间最新的 run，不是最高精度。
MODEL_DIR="${MODEL_DIR:-}"

# 数据划分必须与训练一致，即使指定 MODEL_DIR 也不会自动读取这些参数。
DATASET="${DATASET:-Cifar100}"                    # Cifar10 / Cifar100 / TinyImagenet；自动对应10/100/200类
ALGORITHM="${ALGORITHM:-FedCLIP}"               # 改方法时同时改下面的模型族，名称与训练保存目录一致
# 模型族对照（main.py / .vscode/launch.json；列顺序：CIFAR CNN | Tiny CNN | ResNet）：
# FedCLIP：Decom_CNN-5-512 | Decom_CNN-5-512 | Decom_resnet18_5
# FedSPU ：SPU_CNN1 | SPU_CNN1-tiny | SPU_ResNet18_1
# PFedAFM：CNN-5-512-AFM | CNN-5-512-AFM-tiny | ResNet18-5-AFM
# FedKD/FedTGP/FedRE/FedGH/LG-FedAvg/FedProto/FML/FD/FedGen/FedMRL：
#         CNN-5-512 | CNN-5-512-tiny | ResNet18-5
MODEL_FAMILY="${MODEL_FAMILY-Decom_resnet18_5}"
PARTITION="${PARTITION:-dir}"                    # dir：Dirichlet；pat：病理；exdir：扩展Dirichlet
DIR_ALPHA="${DIR_ALPHA:-1.0}"                    # dir/exdir 系数，如0.1、0.5、1.0
CLASS_PER_CLIENT="${CLASS_PER_CLIENT:-20}"       # pat/exdir 每客户端类别数，如Cifar100 pat20填20

MODEL_SOURCE="${MODEL_SOURCE:-client}"          # client：本地训练后模型；server：服务器模型
SPLIT="${SPLIT:-test}"                          # train / test；此版本不支持 both
# 单个 "0"、列表 "0,5,10"、范围 "0-19"；"" 表示全部客户端。
CLIENT_IDS="${CLIENT_IDS-0}"
# 1=从 CLIENT_IDS 中按特征分离度选一个（不是最高准确率）；0=不自动选。
AUTO_BEST_CLIENT="${AUTO_BEST_CLIENT:-0}"

# ==================== 2. 样本量与 t-SNE 设置 ====================
# 0=不限；正数=每客户端batch上限；-1=旧默认（未指定客户端时40，否则不限）。
MAX_BATCHES="${MAX_BATCHES:-0}"
# 每客户端样本数上限，0=不限；与 MAX_BATCHES 先到者为准，不是类别平衡采样。
MAX_SAMPLES_PER_CLIENT="${MAX_SAMPLES_PER_CLIENT:-0}"
PERPLEXITY="${PERPLEXITY:-30}"                  # 邻域尺度，如15/30/50；应小于样本数，不是类别数
MAX_ITER="${MAX_ITER:-1000}"                    # t-SNE 最大迭代数，至少250；不是通信轮数
TSNE_LR="${TSNE_LR:-200}"                       # t-SNE 学习率，正数，不接受 auto
SEED="${SEED:-0}"                              # 比较不同实验时固定随机种子

# ==================== 3. 图形与输出 ====================
# 留空自动保存至 T-SNE-legacy/ 下；同目录重复运行会覆盖同名 PNG/PDF/CSV。
OUTPUT_DIR="${OUTPUT_DIR:-}"
POINT_SIZE="${POINT_SIZE:-18}"                  # 散点大小，如18/36
POINT_ALPHA="${POINT_ALPHA:-0.7}"               # 透明度0~1，不是Dirichlet系数
SHOW_LEGEND="${SHOW_LEGEND:-1}"                 # 1=显示图例，0=隐藏
MAX_LEGEND_CLASSES="${MAX_LEGEND_CLASSES:-20}"  # 超过此类别数就不显示图例
SAVE_EXCEL="${SAVE_EXCEL:-0}"                   # 1=额外保存xlsx，0=不保存

# ==================== 4. 通常不变：训练配置与目录匹配 ====================
NUM_CLIENTS="${NUM_CLIENTS:-20}"                # 训练时客户端总数
JOIN_RATIO="${JOIN_RATIO:-1.0}"                 # 训练时参与率，仅用于匹配目录
NIID="${NIID:-1}"                              # 1=非IID，0=IID；与训练一致
FINAL_MODEL_ROOT="${FINAL_MODEL_ROOT:-./final_models}"  # MODEL_DIR 留空时的查找根目录

# ==================== 5. 自动选客户端的评分细节（开启时才生效） ====================
# silhouette=轮廓系数；separation=最小类中心距离/平均类内距离。
SELECTION_SCORE="${SELECTION_SCORE:-silhouette}"
SELECTION_METRIC="${SELECTION_METRIC:-euclidean}"        # euclidean/cosine，仅用于轮廓系数
SELECTION_MAX_BATCHES="${SELECTION_MAX_BATCHES:-40}"     # 评分时每客户端batch上限，0=不限
SELECTION_MAX_SAMPLES="${SELECTION_MAX_SAMPLES:-1200}"    # 轮廓系数采样上限，0=不限

# ==================== 6. 运行设置 ====================
DEVICE="${DEVICE:-cuda:0}"                      # 指定 CUDA 设备或 cpu
BATCH_SIZE="${BATCH_SIZE:-16}"                  # 特征提取batch大小
PYTHON_BIN="${PYTHON_BIN:-python}"

# ==================== 以下为启动逻辑，一般不用改 ====================
DRY_RUN=0
case "${1-}" in
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h)
        printf 'Usage: bash %s [--dry-run]\nEdit settings in this file or override them with environment variables.\n' "$0"
        exit 0
        ;;
esac
if (( $# != 0 )); then
    printf 'Unknown argument: %s. Use --dry-run or edit the settings above.\n' "$1" >&2
    exit 2
fi

# 类别总数由数据集确定，无需手动配置。
case "$DATASET" in
    Cifar10) NUM_CLASSES=10 ;;
    Cifar100) NUM_CLASSES=100 ;;
    TinyImagenet) NUM_CLASSES=200 ;;
    *)
        printf 'Unsupported DATASET: %s. Choose Cifar10, Cifar100 or TinyImagenet.\n' "$DATASET" >&2
        exit 2
        ;;
esac

for name in AUTO_BEST_CLIENT SHOW_LEGEND SAVE_EXCEL; do
    if [[ "${!name}" != 0 && "${!name}" != 1 ]]; then
        printf '%s must be 0 or 1 (got %s).\n' "$name" "${!name}" >&2
        exit 2
    fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR"
cmd=(
    "$PYTHON_BIN" -u "$SCRIPT_DIR/T-SNE-Cifar-legacy-compatible.py"
    --model-dir "$MODEL_DIR"
    --final-model-root "$FINAL_MODEL_ROOT"
    --model-family "$MODEL_FAMILY"
    --algorithm "$ALGORITHM"
    --model-source "$MODEL_SOURCE"
    --dataset "$DATASET"
    --num-classes "$NUM_CLASSES"
    --num-clients "$NUM_CLIENTS"
    --join-ratio "$JOIN_RATIO"
    --niid "$NIID"
    --partition "$PARTITION"
    --dir-alpha "$DIR_ALPHA"
    --class-per-client "$CLASS_PER_CLIENT"
    --split "$SPLIT"
    --client-ids "$CLIENT_IDS"
    --selection-score "$SELECTION_SCORE"
    --selection-metric "$SELECTION_METRIC"
    --selection-max-batches "$SELECTION_MAX_BATCHES"
    --selection-max-samples "$SELECTION_MAX_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --max-batches "$MAX_BATCHES"
    --max-samples-per-client "$MAX_SAMPLES_PER_CLIENT"
    --device "$DEVICE"
    --seed "$SEED"
    --perplexity "$PERPLEXITY"
    --tsne-lr "$TSNE_LR"
    --max-iter "$MAX_ITER"
    --point-size "$POINT_SIZE"
    --alpha "$POINT_ALPHA"
    --max-legend-classes "$MAX_LEGEND_CLASSES"
    --output-dir "$OUTPUT_DIR"
)
if [[ "$AUTO_BEST_CLIENT" == 1 ]]; then
    cmd+=(--auto-best-client)
fi
if [[ "$SHOW_LEGEND" == 1 ]]; then
    cmd+=(--show-legend)
else
    cmd+=(--no-show-legend)
fi
if [[ "$SAVE_EXCEL" == 1 ]]; then
    cmd+=(--save-excel)
else
    cmd+=(--no-save-excel)
fi

printf 'Working directory: %s\nCommand:' "$SCRIPT_DIR"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi
exec "${cmd[@]}"
