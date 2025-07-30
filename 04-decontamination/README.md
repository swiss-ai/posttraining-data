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

## Performance Optimizations

### Benchmark N-gram Caching
The script now caches pre-computed n-grams for each benchmark to avoid recomputation on subsequent runs. Cache files are stored in the `benchmark_cache` directory (configurable with `--cache_dir`) and are keyed by:
- Benchmark name
- Tokenizer name  
- N-gram length

On the first run, n-grams are computed and cached. Subsequent runs will load from cache, significantly reducing processing time for the 60+ evaluation benchmarks.