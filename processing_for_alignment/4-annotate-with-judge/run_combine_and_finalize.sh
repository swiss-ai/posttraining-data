#!/bin/bash
set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────
SCRIPT_DIR="$SCRATCH/posttraining-data/processing_for_alignment/4-annotate-with-judge"
DATASETS_DIR="$SCRATCH/posttraining-data/processing_for_alignment/datasets"

ANNOTATIONS_DIR="${DATASETS_DIR}/MaxMin-Filtered-Ref-Completions-Annotated"
COMBINED_DIR="${DATASETS_DIR}/MaxMin-Filtered-Ref-Completions-Combined"
LOGPROBS_DIR="${DATASETS_DIR}/MaxMin-Filtered-Logprobs/merged"
FINAL_DIR="${DATASETS_DIR}/MaxMin-Filtered-Final"
# ──────────────────────────────────────────────────────────────────

echo "=== Step 1: Combine annotated completion splits ==="
echo "  Input:  ${ANNOTATIONS_DIR}/completion_*/"
echo "  Output: ${COMBINED_DIR}"
python -u "${SCRIPT_DIR}/combine_annotations.py" \
    --annotations-dir "${ANNOTATIONS_DIR}" \
    --output-dir "${COMBINED_DIR}"

echo ""
echo "=== Step 2: Finalize dataset with logprobs + QRPO rewards ==="
echo "  Annotations: ${COMBINED_DIR}"
echo "  Logprobs:    ${LOGPROBS_DIR}"
echo "  Output:      ${FINAL_DIR}"
python -u "${SCRIPT_DIR}/finalize_dataset.py" \
    --annotations-path "${COMBINED_DIR}" \
    --logprobs-path "${LOGPROBS_DIR}" \
    --output-dir "${FINAL_DIR}"

echo ""
echo "=== Done ==="
echo "Final dataset saved to: ${FINAL_DIR}"
