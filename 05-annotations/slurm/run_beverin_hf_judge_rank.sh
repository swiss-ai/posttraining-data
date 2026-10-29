#!/usr/bin/env bash

set -euo pipefail

: "${REPO_ROOT:?}"
: "${MIXTURE_INPUT:?}"
: "${MIXTURE_OUTPUT:?}"
: "${JUDGE_MODEL_ID:?}"
: "${JUDGE_MODEL_REVISION:?}"
: "${SLURM_PROCID:?}"
: "${SLURM_NTASKS:?}"

accelerator_visibility="${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
if [[ -z "${accelerator_visibility}" ]]; then
    echo "Rank ${SLURM_PROCID} has no assigned AMD GPU; check --gpus-per-task=1." >&2
    exit 2
fi

# CSCS exposes the assigned AMD device through ROCR_VISIBLE_DEVICES. Normalize
# it to HIP_VISIBLE_DEVICES, matching the current Beverin ROCm guidance.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES}"
    unset ROCR_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
fi

printf -v shard_name "shard-%05d-of-%05d" "${SLURM_PROCID}" "${SLURM_NTASKS}"
shard_output="${MIXTURE_OUTPUT}/${shard_name}"
taxonomy_path="${MIXTURE_TAXONOMY:-${REPO_ROOT}/05-annotations/taxonomies/swe_terminal_tool_use_v0.2.0.yaml}"

arguments=(
    python "${REPO_ROOT}/05-annotations/quantify_sft_mixture.py"
    "${MIXTURE_INPUT}"
    --output "${shard_output}"
    --taxonomy "${taxonomy_path}"
    --output-format "${MIXTURE_OUTPUT_FORMAT:-jsonl}"
    --rows-per-shard "${MIXTURE_ROWS_PER_SHARD:-50000}"
    --judge-provider huggingface
    --judge-model "${JUDGE_MODEL_ID}"
    --judge-hf-revision "${JUDGE_MODEL_REVISION}"
    --judge-hf-device cuda
    --judge-hf-dtype bfloat16
    --judge-hf-max-input-tokens "${JUDGE_MAX_INPUT_TOKENS:-8192}"
    --judge-max-output-tokens "${JUDGE_MAX_OUTPUT_TOKENS:-512}"
    --judge-batch-size "${JUDGE_BATCH_SIZE:-2}"
    --judge-hf-local-files-only
    --num-input-shards "${SLURM_NTASKS}"
    --input-shard-index "${SLURM_PROCID}"
)

if [[ -n "${MIXTURE_LIMIT_PER_SHARD:-}" ]]; then
    arguments+=(--limit "${MIXTURE_LIMIT_PER_SHARD}")
fi

echo "rank=${SLURM_PROCID}/${SLURM_NTASKS} rocm_device=${accelerator_visibility} output=${shard_output}"
"${arguments[@]}"
