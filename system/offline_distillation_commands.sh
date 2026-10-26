#!/usr/bin/env bash
# Source from FedClip/system, then use the existing scheduler's COMMANDS array.
# This file only defines commands. It never launches training.
DISTILL_RUN_TAG="${DISTILL_RUN_TAG:-R201_extra5_seed0_$(date +%Y%m%d_%H%M%S)}"
DISTILL_BUNDLE="${DISTILL_BUNDLE:-../offline_inputs/C0_three_models_R201_20261025_214112_203491.zip}"
DISTILL_DATA_ROOT="${DISTILL_DATA_ROOT:-../dataset}"
DISTILL_RUN_ROOT="${DISTILL_RUN_ROOT:-offline_distillation_runs/${DISTILL_RUN_TAG}}"
DISTILL_CACHE_DIR="${DISTILL_CACHE_DIR:-${DISTILL_RUN_ROOT}/teacher_cache}"
DISTILL_CLIENT_ID="${DISTILL_CLIENT_ID:-0}"
DISTILL_ASSIGNED_MODEL_ID="${DISTILL_ASSIGNED_MODEL_ID:-0}"

declare -a COMMANDS=(
"python -u offline_distillation.py --bundle '${DISTILL_BUNDLE}' --client-id ${DISTILL_CLIENT_ID} --assigned-model-id ${DISTILL_ASSIGNED_MODEL_ID} --model-ids 0 1 2 --completed-round 201 --data-root '${DISTILL_DATA_ROOT}' --dataset Cifar100 --data-subdir pat_20 --mode continue_ce --epochs 5 --batch-size 16 --test-batch-size 16 --teacher-batch-size 16 --lr 0.005 --momentum 0 --weight-decay 0 --f-regular 1 --f-lambda 1e-3 --grad-clip 10 --temperature 2 --kd-lambda 1 --seed 0 --device cuda --device-id 0 --threads 2 --deterministic 1 --verify-cache --cache-dir '${DISTILL_CACHE_DIR}' --output-dir '${DISTILL_RUN_ROOT}/continue_ce'"
"python -u offline_distillation.py --bundle '${DISTILL_BUNDLE}' --client-id ${DISTILL_CLIENT_ID} --assigned-model-id ${DISTILL_ASSIGNED_MODEL_ID} --model-ids 0 1 2 --completed-round 201 --data-root '${DISTILL_DATA_ROOT}' --dataset Cifar100 --data-subdir pat_20 --mode kd_mean --epochs 5 --batch-size 16 --test-batch-size 16 --teacher-batch-size 16 --lr 0.005 --momentum 0 --weight-decay 0 --f-regular 1 --f-lambda 1e-3 --grad-clip 10 --temperature 2 --kd-lambda 1 --seed 0 --device cuda --device-id 1 --threads 2 --deterministic 1 --verify-cache --cache-dir '${DISTILL_CACHE_DIR}' --output-dir '${DISTILL_RUN_ROOT}/kd_mean'"
"python -u offline_distillation.py --bundle '${DISTILL_BUNDLE}' --client-id ${DISTILL_CLIENT_ID} --assigned-model-id ${DISTILL_ASSIGNED_MODEL_ID} --model-ids 0 1 2 --completed-round 201 --data-root '${DISTILL_DATA_ROOT}' --dataset Cifar100 --data-subdir pat_20 --mode kl_dp --epochs 5 --batch-size 16 --test-batch-size 16 --teacher-batch-size 16 --lr 0.005 --momentum 0 --weight-decay 0 --f-regular 1 --f-lambda 1e-3 --grad-clip 10 --temperature 2 --kd-lambda 1 --seed 0 --device cuda --device-id 2 --threads 2 --deterministic 1 --verify-cache --cache-dir '${DISTILL_CACHE_DIR}' --output-dir '${DISTILL_RUN_ROOT}/kl_dp'"
)

# After all three independent runs finish (not executed by sourcing this file):
# python compare_offline_distillation.py "${DISTILL_RUN_ROOT}/continue_ce" "${DISTILL_RUN_ROOT}/kd_mean" "${DISTILL_RUN_ROOT}/kl_dp" --output-dir "${DISTILL_RUN_ROOT}/comparison"
