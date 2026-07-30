#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/new_log}"
CSV_ROOT="${CSV_ROOT:-${REPOSITORY_ROOT}/projection_csv_logs}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${SCRIPT_DIR}/new_log_archives}"
ALLOW_INCOMPLETE=0

usage() {
    cat <<'EOF'
用法：
  bash pack_projection_logs.sh [选项] [日志名称或路径 ...]

不传日志名称时，默认整理 new_log/ 下的全部日志。每个压缩包包含：
  1 个训练日志 + 1 个客户端诊断 CSV + 1 个方向诊断 CSV
  如果本次运行生成了跨层客户端诊断，还会包含：
  1 个 projection_cross_layer_client_diagnostics.csv

选项：
  --log-dir DIR       日志根目录，默认 system/new_log
  --csv-dir DIR       CSV 根目录，默认 <仓库根>/projection_csv_logs
  --output-dir DIR    压缩包目录，默认 system/new_log_archives
  --allow-incomplete  允许打包没有出现 All done! 的未完成/异常日志
  -h, --help          显示帮助

示例：
  bash pack_projection_logs.sh
  bash pack_projection_logs.sh 20260712_170731_01_FedCLIP_Cifar100_dir0.5
  bash pack_projection_logs.sh --output-dir ./archives 20260712_170731_01_FedCLIP_Cifar100_dir0.5

也可以用环境变量 LOG_ROOT、CSV_ROOT、ARCHIVE_ROOT 修改默认目录。
EOF
}

error() {
    printf '❌ %s\n' "$*" >&2
}

absolute_file_path() {
    local path="$1"
    local directory
    directory="$(cd -- "$(dirname -- "${path}")" && pwd -P)" || return 1
    printf '%s/%s\n' "${directory}" "$(basename -- "${path}")"
}

declare -a selectors=()
while (($# > 0)); do
    case "$1" in
        --log-dir)
            if (($# < 2)); then
                error "--log-dir 缺少目录参数。"
                exit 2
            fi
            LOG_ROOT="$2"
            shift 2
            ;;
        --log-dir=*)
            LOG_ROOT="${1#*=}"
            shift
            ;;
        --csv-dir)
            if (($# < 2)); then
                error "--csv-dir 缺少目录参数。"
                exit 2
            fi
            CSV_ROOT="$2"
            shift 2
            ;;
        --csv-dir=*)
            CSV_ROOT="${1#*=}"
            shift
            ;;
        --output-dir)
            if (($# < 2)); then
                error "--output-dir 缺少目录参数。"
                exit 2
            fi
            ARCHIVE_ROOT="$2"
            shift 2
            ;;
        --output-dir=*)
            ARCHIVE_ROOT="${1#*=}"
            shift
            ;;
        --allow-incomplete)
            ALLOW_INCOMPLETE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            selectors+=("$@")
            break
            ;;
        -*)
            error "未知参数：$1"
            usage >&2
            exit 2
            ;;
        *)
            selectors+=("$1")
            shift
            ;;
    esac
done

if [[ ! -d "${LOG_ROOT}" ]]; then
    error "日志目录不存在：${LOG_ROOT}"
    exit 1
fi
if [[ ! -d "${CSV_ROOT}" ]]; then
    error "CSV 目录不存在：${CSV_ROOT}"
    exit 1
fi

LOG_ROOT="$(cd -- "${LOG_ROOT}" && pwd -P)"
CSV_ROOT="$(cd -- "${CSV_ROOT}" && pwd -P)"
mkdir -p -- "${ARCHIVE_ROOT}" || exit 1
ARCHIVE_ROOT="$(cd -- "${ARCHIVE_ROOT}" && pwd -P)"

declare -a log_entries=()
failed=0

if ((${#selectors[@]} == 0)); then
    while IFS= read -r -d '' entry; do
        if [[ "${entry}" == "${ARCHIVE_ROOT}" ]]; then
            continue
        fi
        case "${entry}" in
            *.tar.gz|*.tgz|*.zip)
                continue
                ;;
        esac
        log_entries+=("${entry}")
    done < <(find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 \
        \( -type f -o -type d \) -print0)
else
    for selector in "${selectors[@]}"; do
        if [[ -e "${selector}" ]]; then
            log_entries+=("${selector}")
        elif [[ -e "${LOG_ROOT}/${selector}" ]]; then
            log_entries+=("${LOG_ROOT}/${selector}")
        elif [[ -e "${LOG_ROOT}/${selector}.log" ]]; then
            log_entries+=("${LOG_ROOT}/${selector}.log")
        else
            error "找不到指定日志：${selector}"
            ((failed += 1))
        fi
    done
fi

if ((${#log_entries[@]} == 0)); then
    error "没有找到可整理的日志。"
    exit 1
fi

RESOLVED_LOG=""
RUN_PREFIX=""

resolve_log_entry() {
    local entry="$1"
    local base parent stem
    local -a candidates=()

    if [[ -f "${entry}" ]]; then
        RESOLVED_LOG="$(absolute_file_path "${entry}")" || return 1
        base="$(basename -- "${RESOLVED_LOG}")"
        parent="$(dirname -- "${RESOLVED_LOG}")"
        if [[ "${base}" == "train.log" || "${base}" == "run.log" || "${base}" == "output.log" ]] \
            && [[ "${parent}" != "${LOG_ROOT}" ]]; then
            RUN_PREFIX="$(basename -- "${parent}")"
        else
            stem="${base}"
            stem="${stem%.log}"
            stem="${stem%.out}"
            stem="${stem%.txt}"
            RUN_PREFIX="${stem}"
        fi
        return 0
    fi

    if [[ ! -d "${entry}" ]]; then
        error "日志条目既不是文件也不是目录：${entry}"
        return 1
    fi

    RUN_PREFIX="$(basename -- "${entry}")"
    if [[ -f "${entry}/train.log" ]]; then
        RESOLVED_LOG="$(absolute_file_path "${entry}/train.log")" || return 1
        return 0
    fi

    while IFS= read -r -d '' candidate; do
        candidates+=("${candidate}")
    done < <(find "${entry}" -mindepth 1 -maxdepth 1 -type f \
        \( -name '*.log' -o -name '*.out' \) -print0)

    if ((${#candidates[@]} == 0)); then
        while IFS= read -r -d '' candidate; do
            candidates+=("${candidate}")
        done < <(find "${entry}" -mindepth 1 -maxdepth 1 -type f -print0)
    fi

    if ((${#candidates[@]} != 1)); then
        error "${entry} 中无法唯一确定训练日志（找到 ${#candidates[@]} 个候选）。"
        return 1
    fi

    RESOLVED_LOG="$(absolute_file_path "${candidates[0]}")" || return 1
}

extract_csv_references() {
    local log_file="$1"
    local label="$2"
    awk -v label="${label}" '
        index($0, label) {
            value = substr($0, index($0, label) + length(label))
            sub(/^[[:space:]]*/, "", value)
            sub(/\r$/, "", value)
            if (!seen[value]++) {
                print value
            }
        }
    ' "${log_file}"
}

RESOLVED_CSV=""
resolve_csv_reference() {
    local reference="$1"
    local basename_reference

    if [[ -f "${reference}" ]]; then
        RESOLVED_CSV="$(absolute_file_path "${reference}")" || return 1
        return 0
    fi

    basename_reference="$(basename -- "${reference}")"
    if [[ -f "${CSV_ROOT}/${basename_reference}" ]]; then
        RESOLVED_CSV="$(absolute_file_path "${CSV_ROOT}/${basename_reference}")" || return 1
        return 0
    fi

    return 1
}

process_log_entry() {
    local entry="$1"
    local archive_path temporary_archive
    local client_csv direction_csv cross_layer_client_csv=""
    local client_key direction_key cross_layer_client_key archive_listing
    local expected_member_count=3
    local -a client_references=()
    local -a direction_references=()
    local -a cross_layer_client_references=()
    local -a archive_members=()
    local -a tar_arguments=()

    RESOLVED_LOG=""
    RUN_PREFIX=""
    if ! resolve_log_entry "${entry}"; then
        return 1
    fi
    if [[ ! -s "${RESOLVED_LOG}" ]]; then
        error "训练日志为空，跳过：${RESOLVED_LOG}"
        return 1
    fi

    archive_path="${ARCHIVE_ROOT}/${RUN_PREFIX}.tar.gz"
    if [[ -e "${archive_path}" ]]; then
        printf '⏭️  已存在，跳过：%s\n' "${archive_path}"
        return 10
    fi
    if ((ALLOW_INCOMPLETE == 0)) && ! grep -Fq "All done!" "${RESOLVED_LOG}"; then
        printf '⏭️  日志尚未正常结束，跳过：%s（需要强制打包时传 --allow-incomplete）\n' "${RESOLVED_LOG}"
        return 11
    fi

    mapfile -t client_references < <(
        extract_csv_references "${RESOLVED_LOG}" "Projection 客户端诊断 CSV:"
    )
    mapfile -t direction_references < <(
        extract_csv_references "${RESOLVED_LOG}" "Projection 方向诊断 CSV:"
    )
    mapfile -t cross_layer_client_references < <(
        extract_csv_references "${RESOLVED_LOG}" "Projection 跨层客户端诊断 CSV:"
    )

    if ((${#client_references[@]} != 1 || ${#direction_references[@]} != 1)); then
        error "${RESOLVED_LOG} 中 CSV 路径不是唯一一对（客户端 ${#client_references[@]} 个，方向 ${#direction_references[@]} 个）。"
        return 1
    fi
    if ((${#cross_layer_client_references[@]} > 1)); then
        error "${RESOLVED_LOG} 中跨层客户端诊断 CSV 路径不唯一（找到 ${#cross_layer_client_references[@]} 个）。"
        return 1
    fi

    if ! resolve_csv_reference "${client_references[0]}"; then
        error "找不到客户端诊断 CSV：${client_references[0]}"
        return 1
    fi
    client_csv="${RESOLVED_CSV}"
    if [[ ! -s "${client_csv}" ]]; then
        error "客户端诊断 CSV 为空：${client_csv}"
        return 1
    fi

    if ! resolve_csv_reference "${direction_references[0]}"; then
        error "找不到方向诊断 CSV：${direction_references[0]}"
        return 1
    fi
    direction_csv="${RESOLVED_CSV}"
    if [[ ! -s "${direction_csv}" ]]; then
        error "方向诊断 CSV 为空：${direction_csv}"
        return 1
    fi
    if [[ "${client_csv}" == "${direction_csv}" ]]; then
        error "两条日志记录指向同一个 CSV，拒绝重复打包：${client_csv}"
        return 1
    fi

    client_key="$(basename -- "${client_csv}")"
    client_key="${client_key%_projection_client_diagnostics.csv}"
    direction_key="$(basename -- "${direction_csv}")"
    direction_key="${direction_key%_projection_direction_diagnostics.csv}"
    if [[ ! "${client_key}" =~ ^[0-9]{8}_[0-9]{6}_[0-9]{6}$ ]] \
        || [[ ! "${direction_key}" =~ ^[0-9]{8}_[0-9]{6}_[0-9]{6}$ ]]; then
        error "CSV 文件名不符合当前时间戳命名规则：$(basename -- "${client_csv}") / $(basename -- "${direction_csv}")"
        return 1
    fi
    if [[ "${client_key}" != "${direction_key}" ]]; then
        error "两份 CSV 的时间戳前缀不一致，拒绝混合打包：${client_key} / ${direction_key}"
        return 1
    fi

    if ((${#cross_layer_client_references[@]} == 1)); then
        if ! resolve_csv_reference "${cross_layer_client_references[0]}"; then
            error "找不到跨层客户端诊断 CSV：${cross_layer_client_references[0]}"
            return 1
        fi
        cross_layer_client_csv="${RESOLVED_CSV}"
        if [[ ! -s "${cross_layer_client_csv}" ]]; then
            error "跨层客户端诊断 CSV 为空：${cross_layer_client_csv}"
            return 1
        fi
        if [[ "${cross_layer_client_csv}" == "${client_csv}" ]] \
            || [[ "${cross_layer_client_csv}" == "${direction_csv}" ]]; then
            error "跨层客户端诊断 CSV 与已有 CSV 重复，拒绝打包：${cross_layer_client_csv}"
            return 1
        fi

        cross_layer_client_key="$(basename -- "${cross_layer_client_csv}")"
        cross_layer_client_key="${cross_layer_client_key%_projection_cross_layer_client_diagnostics.csv}"
        if [[ ! "${cross_layer_client_key}" =~ ^[0-9]{8}_[0-9]{6}_[0-9]{6}$ ]]; then
            error "跨层客户端诊断 CSV 文件名不符合当前时间戳命名规则：$(basename -- "${cross_layer_client_csv}")"
            return 1
        fi
        if [[ "${cross_layer_client_key}" != "${client_key}" ]]; then
            error "跨层客户端诊断 CSV 的时间戳前缀不一致，拒绝混合打包：${client_key} / ${cross_layer_client_key}"
            return 1
        fi
        expected_member_count=4
    fi

    temporary_archive="${ARCHIVE_ROOT}/.${RUN_PREFIX}.tar.gz.tmp.$$.$RANDOM"
    tar_arguments=(
        -czf "${temporary_archive}"
        -C "$(dirname -- "${RESOLVED_LOG}")" "$(basename -- "${RESOLVED_LOG}")"
        -C "$(dirname -- "${client_csv}")" "$(basename -- "${client_csv}")"
        -C "$(dirname -- "${direction_csv}")" "$(basename -- "${direction_csv}")"
    )
    if [[ -n "${cross_layer_client_csv}" ]]; then
        tar_arguments+=(
            -C "$(dirname -- "${cross_layer_client_csv}")"
            "$(basename -- "${cross_layer_client_csv}")"
        )
    fi
    if ! tar "${tar_arguments[@]}"; then
        rm -f -- "${temporary_archive}"
        error "压缩失败：${RUN_PREFIX}"
        return 1
    fi

    if ! archive_listing="$(tar -tzf "${temporary_archive}")"; then
        rm -f -- "${temporary_archive}"
        error "无法校验压缩包：${RUN_PREFIX}"
        return 1
    fi
    mapfile -t archive_members <<< "${archive_listing}"
    if ((${#archive_members[@]} != expected_member_count)); then
        rm -f -- "${temporary_archive}"
        error "压缩包内容不是预期的 ${expected_member_count} 个文件，已取消：${RUN_PREFIX}"
        return 1
    fi

    if ! mv -- "${temporary_archive}" "${archive_path}"; then
        rm -f -- "${temporary_archive}"
        error "无法保存压缩包：${archive_path}"
        return 1
    fi

    printf '✅ 已打包：%s\n' "${archive_path}"
    printf '   log: %s\n' "${RESOLVED_LOG}"
    printf '   csv: %s\n' "${client_csv}"
    printf '   csv: %s\n' "${direction_csv}"
    if [[ -n "${cross_layer_client_csv}" ]]; then
        printf '   csv: %s\n' "${cross_layer_client_csv}"
    fi
    return 0
}

packed=0
skipped=0
for entry in "${log_entries[@]}"; do
    if process_log_entry "${entry}"; then
        ((packed += 1))
    else
        status=$?
        if ((status == 10 || status == 11)); then
            ((skipped += 1))
        else
            ((failed += 1))
        fi
    fi
done

printf '\n整理完成：成功 %d，跳过 %d，失败 %d。\n' "${packed}" "${skipped}" "${failed}"
printf '压缩包目录：%s\n' "${ARCHIVE_ROOT}"

if ((failed > 0)); then
    exit 1
fi
