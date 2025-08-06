#!/bin/bash

# Simple working conversion script

INPUT_DIR="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat"
CONVERT_SCRIPT="./07-dataset-aggregation/convert_old_to_new_format.py"
PYTHON_ENV="./venv/bin/python"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Arrays for tracking
SUCCESSFUL=()
FAILED=()
SKIPPED=()

echo "Starting conversions..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Process each directory
cd "$INPUT_DIR"
for dataset in */; do
    dataset_name="${dataset%/}"
    echo "Processing: $dataset_name"
    
    input_path="$INPUT_DIR/$dataset_name"
    output_path="$OUTPUT_DIR/$dataset_name"
    
    # Skip if already exists
    if [[ -d "$output_path" ]]; then
        echo "  SKIP: Already exists"
        SKIPPED+=("$dataset_name")
        continue
    fi
    
    # Skip decontamination_prompts folder
    if [[ "$dataset_name" == "decontamination_prompts" ]]; then
        echo "  SKIP: decontamination_prompts folder"
        SKIPPED+=("$dataset_name")
        continue
    fi
    
    # Skip if no dataset_dict.json
    if [[ ! -f "$input_path/dataset_dict.json" ]]; then
        echo "  SKIP: No dataset_dict.json"
        SKIPPED+=("$dataset_name")
        continue
    fi
    
    # Convert
    echo "  Converting..."
    cd /iopsstor/scratch/cscs/schlag/post-training-scripts
    if "$PYTHON_ENV" "$CONVERT_SCRIPT" "$input_path" "$output_path" --no-validate; then
        echo "  SUCCESS"
        SUCCESSFUL+=("$dataset_name")
    else
        echo "  FAILED"
        FAILED+=("$dataset_name")
    fi
    
    cd "$INPUT_DIR"
    echo ""
done

# Summary
echo "SUMMARY:"
echo "Successful: ${#SUCCESSFUL[@]}"
echo "Failed: ${#FAILED[@]}"
echo "Skipped: ${#SKIPPED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "Failed datasets:"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
fi