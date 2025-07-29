#!/bin/bash

data_root_folder="/capstor/store/cscs/swissai/infra01/posttrain_data"
dataset_name="tulu-3-sft-mixture"
decontamination_prompts_path="${data_root_folder}/04_decontaminated/decontamination_prompts"
input_path="${data_root_folder}/03_license_filtered/${dataset_name}"
export PROJECT_ROOT_AT="${HOME}/projects/post-training-scripts/04-decontamination"

# if decontamination_prompts_path does not exist, then create it
if [ ! -d "${decontamination_prompts_path}" ]; then
  echo "Decontamination prompts not found at: ${decontamination_prompts_path}"
  echo "Running Python command..."

  python gather_decontamination_prompts --output "${decontamination_prompts_path}"
else
  echo "Decontamination prompts already exists: ${decontamination_prompts_path}"
fi

python $PROJECT_ROOT_AT/decontamination.py \
  "${input_path}" \
  --output "${data_root_folder}/04_decontaminated/${dataset_name}" \
  --decontamination_prompts "${decontamination_prompts_path}" \
  --tokenizer_name "alehc/swissai-tokenizer" \
  --report_path "${input_path}/contamination_reports" \
  --ngram_length 8 \
  --diff_threshold 0.5 \
  --num_proc 10
