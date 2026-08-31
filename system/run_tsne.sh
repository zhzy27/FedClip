#!/usr/bin/env bash
# 启动 T-SNE-Cifar-legacy-compatible.py：原始特征，不加文本锚点、不做归一化。
# 图左上角自动标注所选 train/test 绘图样本的准确率；多客户端按样本数汇总。
# 运行：bash system/run_tsne.sh；预览命令：bash system/run_tsne.sh --dry-run
# 实验参数放在 tsne_params.local.sh；本文件只负责加载、校验与启动。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/tsne_params.local.sh"
DRY_RUN=0
while (( $# > 0 )); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --config)
            if (( $# < 2 )) || [[ -z "$2" || "$2" == --* ]]; then
                printf '%s\n' '--config requires a settings file path.' >&2
                exit 2
            fi
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            printf 'Usage: bash %s [--config FILE] [--dry-run]\n' "$0"
            printf '%s\n' \
                'Default settings: system/tsne_params.local.sh (Git-ignored, sync manually).' \
                'Defaults and option descriptions: system/tsne_params.example.sh (Git-tracked).' \
                'CLIENT_IDS=best selects the highest accuracy on the chosen train/test plotting samples.' \
                '--config paths are relative to your current directory.' \
                'Model/output paths are relative to system/.' \
                'Environment overrides remain supported by the supplied settings file.'
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s. Use --help.\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$CONFIG_FILE" ]]; then
    printf 'Settings file not found: %s\n' "$CONFIG_FILE" >&2
    printf 'Sync your local settings manually, or create them from the example:\n  cp %q %q\n' \
        "$SCRIPT_DIR/tsne_params.example.sh" "$CONFIG_FILE" >&2
    exit 2
fi
# 在切换工作目录前解析 --config；带空格或中文的路径保持为一个参数。
CONFIG_FILE="$(cd -- "$(dirname -- "$CONFIG_FILE")" && pwd)/$(basename -- "$CONFIG_FILE")"
source "$CONFIG_FILE"
# 只补充 local 中缺少的参数，使新增选项可以兼容旧实验配置。
source "$SCRIPT_DIR/tsne_params.example.sh"

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

printf 'Settings file: %s\nWorking directory: %s\nCommand:' "$CONFIG_FILE" "$SCRIPT_DIR"
printf ' %q' "${cmd[@]}"
printf '\n'
if [[ "$DRY_RUN" == 1 ]]; then
    exit 0
fi
exec "${cmd[@]}"
