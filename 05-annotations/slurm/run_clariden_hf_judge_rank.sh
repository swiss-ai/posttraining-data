#!/usr/bin/env bash

set -euo pipefail

: "${REPO_ROOT:?}"
: "${MIXTURE_INPUT:?}"
: "${MIXTURE_OUTPUT:?}"
: "${JUDGE_MODEL_ID:?}"
: "${JUDGE_MODEL_REVISION:?}"
: "${SLURM_PROCID:?}"
: "${SLURM_NTASKS:?}"
: "${CLARIDEN_UENV_LABEL:?}"
: "${CLARIDEN_UENV_IMAGE_ID:?}"
: "${CLARIDEN_UENV_SQUASHFS_SHA256:?}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "Rank ${SLURM_PROCID} has no assigned GPU; check --gpus-per-task=1." >&2
    exit 2
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
    --runtime-metadata "cluster=clariden"
    --runtime-metadata "accelerator=gh200"
    --runtime-metadata "uenv_label=${CLARIDEN_UENV_LABEL}"
    --runtime-metadata "uenv_image_id=${CLARIDEN_UENV_IMAGE_ID}"
    --runtime-metadata "uenv_squashfs_sha256=${CLARIDEN_UENV_SQUASHFS_SHA256}"
)

if [[ -n "${MIXTURE_LIMIT_PER_SHARD:-}" ]]; then
    arguments+=(--limit "${MIXTURE_LIMIT_PER_SHARD}")
fi

echo "rank=${SLURM_PROCID}/${SLURM_NTASKS} cuda=${CUDA_VISIBLE_DEVICES} output=${shard_output}"
"${arguments[@]}"
