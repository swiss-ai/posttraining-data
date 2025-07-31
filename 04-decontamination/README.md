# Decontamination

Cross-checks prompts in the given dataset with the prompts in the evaluation benchmarks.
If significant overlap is detected, the prompt is removed from the training data

## Usage

### Python interactive session
Decontamination contains two steps. See below the commands, however, they are also provided in `run_decontamination.sh`
so you can update paths and names there and run with the `bash` command.
Additionally, the `unattended_run_decontamination.sh` can be used with the `sbatch` command that automatically calls the `run_decontamination.sh` script.

1. Gathering the prompts from the benchmarks. This is already done and unless you included new benchmarks, you can omit this step.
```python
    python gather_decontamination_prompts --output "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"
```
2. Execute the decontamination
```python
export PROJECT_ROOT_AT="${HOME}/projects/post-training-scripts/04-decontamination"
data_root_folder="/capstor/store/cscs/swissai/infra01/posttrain_data"
decontamination_prompts_path="${data_root_folder}/04_decontaminated/decontamination_prompts"

dataset_name="tulu-3-sft-mixture"
input_path="${data_root_folder}/03_license_filtered/${dataset_name}"
python $PROJECT_ROOT_AT/decontamination.py \
  "${input_path}" \
  --output "${data_root_folder}/04_decontaminated/${dataset_name}" \
  --decontamination_prompts "${decontamination_prompts_path}" \
  --tokenizer_name "alehc/swissai-tokenizer" \
  --report_path "${input_path}/contamination_reports" \
  --ngram_length 8 \
  --diff_threshold 0.5 \
  --num_proc 10 \
  --cache_dir "benchmark_cache"
```
This script loads the dataset from the shared `03_license_filtered` directory, saves the decontamination reports under this dataset's directory,
and writes the decontaminated dataset to the `04_decontaminated` directory.

### Using the SLURM submission script

For easier job submission, use the `submit_decontamination.sh` wrapper script:

```bash
./04-decontamination/submit_decontamination.sh <input_dataset_path> <output_dataset_path>
```

Example:
```bash
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/EuroBlocks-SFT-Synthetic-1124" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/EuroBlocks-SFT-Synthetic-1124"
```

This script:
- Validates the input dataset exists
- Creates a timestamped SLURM job script
- Submits the job with appropriate resources (288 CPUs, 16 parallel processes)
- Uses the shared benchmark cache at `/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache`
- Saves contamination reports in the output directory
- Updates the dataset metadata with a processing log entry
- Prints the `tail -f` command to monitor job progress

The decontamination process:
1. Tokenizes training prompts and computes n-grams (default: 8-grams)
2. Compares against 400+ evaluation benchmark prompts
3. Identifies contaminated samples using sequence matching (default threshold: 0.5)
4. Filters out contaminated conversations from all dataset splits
5. Saves the cleaned dataset with updated metadata

## Examples

```bash
# AceReason-1.1-SFT
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/AceReason-1.1-SFT" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/AceReason-1.1-SFT"

# EuroBlocks-SFT-Synthetic-1124
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/EuroBlocks-SFT-Synthetic-1124" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/EuroBlocks-SFT-Synthetic-1124"

# Llama-Nemotron-Post-Training-Dataset-science-chat-safety
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/Llama-Nemotron-Post-Training-Dataset-science-chat-safety" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Llama-Nemotron-Post-Training-Dataset-science-chat-safety"

# smoltalk
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk"

# smoltalk2
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2"

# The-Tome
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/The-Tome" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/The-Tome"

# tulu-3-sft-mixture
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/tulu-3-sft-mixture" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/tulu-3-sft-mixture"

# Commercial-Flan-Collection-Chain-Of-Thought
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-Chain-Of-Thought" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-Chain-Of-Thought"

# Commercial-Flan-Collection-Flan-2021
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-Flan-2021" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-Flan-2021"

# Commercial-Flan-Collection-SNI
./04-decontamination/submit_decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-SNI" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-SNI"
```