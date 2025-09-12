# Decontamination

Cross-checks prompts in the given dataset with the prompts in the evaluation benchmarks.
If significant overlap is detected, the prompt is removed from the training data

## Available Scripts

### Python Scripts
- **`decontamination.py`** - Main decontamination script with n-gram overlap detection
- **`decontamination-parallel.py`** - High-performance parallel decontamination for large datasets
- **`gather-decontamination-prompts.py`** - Gather evaluation benchmark prompts for reference
- **`list-benchmarks.py`** - List available benchmarks for parallel processing
- **`merge-decontamination-reports.py`** - Merge contamination reports from parallel jobs

### Shell Scripts
- **`submit-decontamination.sh`** - SLURM submission wrapper for single-threaded decontamination
- **`submit-parallel-decontamination.sh`** - SLURM submission wrapper for parallel decontamination

## Usage

### Python interactive session
Decontamination contains two steps:

1. Gathering the prompts from the benchmarks. This is already done and unless you included new benchmarks, you can omit this step.
```bash
    python gather-decontamination-prompts.py --output "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"
```
2. Execute the decontamination
```bash
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

For easier job submission, use the `submit-decontamination.sh` wrapper script:

```bash
./04-decontamination/submit-decontamination.sh <input_dataset_path> <output_dataset_path>
```

Example:
```bash
./04-decontamination/submit-decontamination.sh \
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
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/AceReason-1.1-SFT" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/AceReason-1.1-SFT"

# EuroBlocks-SFT-Synthetic-1124
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/EuroBlocks-SFT-Synthetic-1124" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/EuroBlocks-SFT-Synthetic-1124"

# Llama-Nemotron-Post-Training-Dataset-science-chat-safety
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/Llama-Nemotron-Post-Training-Dataset-science-chat-safety" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Llama-Nemotron-Post-Training-Dataset-science-chat-safety"

# smoltalk
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk"

# smoltalk2 (individual splits after dataset splitting)
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-aya_dataset_Qwen3_32B_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-aya_dataset_Qwen3_32B_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-multi_turn_reasoning_if_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-multi_turn_reasoning_if_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-OpenThoughts3_1.2M_no_think_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-OpenThoughts3_1.2M_no_think_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-OpenThoughts3_1.2M_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-OpenThoughts3_1.2M_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-s1k_1.1_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-s1k_1.1_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_everyday_convs_reasoning_Qwen3_32B_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_everyday_convs_reasoning_Qwen3_32B_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_multilingual_8languages_lang_5_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_multilingual_8languages_lang_5_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_multilingual8_Qwen3_32B_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_multilingual8_Qwen3_32B_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_smollm3_everyday_conversations_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_everyday_conversations_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_smollm3_smol_magpie_ultra_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_smol_magpie_ultra_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_smollm3_smol_rewrite_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_smol_rewrite_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_smollm3_smol_summarize_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_smol_summarize_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_smollm3_systemchats_30k_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_smollm3_systemchats_30k_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-smoltalk_systemchats_Qwen3_32B_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-smoltalk_systemchats_Qwen3_32B_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-table_gpt_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-table_gpt_no_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-table_gpt_Qwen3_32B_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-table_gpt_Qwen3_32B_think"

./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/smoltalk2-tulu_3_sft_personas_instruction_following_no_think" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/smoltalk2-tulu_3_sft_personas_instruction_following_no_think"

# The-Tome
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/The-Tome" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/The-Tome"

# tulu-3-sft-mixture
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/tulu-3-sft-mixture" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/tulu-3-sft-mixture"

# Commercial-Flan-Collection-Chain-Of-Thought
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-Chain-Of-Thought" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-Chain-Of-Thought"

# Commercial-Flan-Collection-Flan-2021
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-Flan-2021" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-Flan-2021"

# Commercial-Flan-Collection-SNI
./04-decontamination/submit-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/Commercial-Flan-Collection-SNI" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/Commercial-Flan-Collection-SNI"

# muri-it
./04-decontamination/submit-decontamination.sh ~/store/posttrain_data/02_standardised/muri-it ~/store/posttrain_data/04_decontaminated/muri-it
```

## Parallel Processing (Recommended for Large Datasets)

For faster processing of the 400+ evaluation benchmarks, use the parallel decontamination system:

### Usage

```bash
./04-decontamination/submit-parallel-decontamination.sh <input_dataset> <output_dataset> [chunk_size] [max_parallel_jobs]
```

**Parameters:**
- `input_dataset`: Path to input dataset directory 
- `output_dataset`: Path for final output dataset directory
- `chunk_size`: Benchmarks per parallel job (default: 20)
- `max_parallel_jobs`: Maximum parallel jobs (default: 20)

### Examples

```bash
# Default: 20 benchmarks per job, max 20 parallel jobs (~20x speedup)
./04-decontamination/submit-parallel-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/tulu-3-sft-mixture" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/tulu-3-sft-mixture"

# Custom: 25 benchmarks per job, max 16 parallel jobs  
./04-decontamination/submit-parallel-decontamination.sh \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/03_license_filtered/The-Tome" \
  "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/The-Tome" \
  25 16
```

### How It Works

1. **Job Array Submission**: Automatically splits 400+ benchmarks across parallel SLURM jobs
2. **Independent Processing**: Each job processes assigned benchmark subset using existing decontamination logic
3. **Shared Caching**: All jobs use shared benchmark n-gram cache for efficiency
4. **Automatic Merging**: Final job combines all contamination reports and creates filtered dataset
5. **Fault Tolerance**: Individual job failures don't affect other jobs

### Performance Benefits

- **~20x faster**: Parallel processing vs sequential (with default settings)
- **Resource efficiency**: Better CPU utilization across cluster
- **Scalability**: Easy to adjust parallelization level
- **Memory efficient**: Each job loads training data only once

### Monitoring

```bash
# Monitor parallel jobs
squeue -j <parallel_job_id>

# Watch progress  
watch -n 5 'ls /path/to/reports/*.completed 2>/dev/null | wc -l; echo "/ <num_jobs> completed"'

# View job logs
tail -f slurm_logs/pdecontam_<dataset>_<timestamp>_*.out

# View merge log  
tail -f slurm_logs/merge_<dataset>_<timestamp>.out
```

### Utilities

```bash
# List available benchmarks
python 04-decontamination/list-benchmarks.py --prompts-path /path/to/decontamination_prompts

# Count benchmarks and show job estimation
python 04-decontamination/list-benchmarks.py --prompts-path /path/to/decontamination_prompts --chunk-size 20

# Manual merge (if needed)
python 04-decontamination/merge-decontamination-reports.py \
  /path/to/input/dataset \
  /path/to/output/dataset \
  /path/to/parallel_reports
```
