# Post-Training Data Processing Pipeline

`posttraining-data` contains scripts organized into eight stages for processing Hugging Face datasets. Select and configure the stages required by your data recipe; there is no single command that runs all eight. It was used to prepare Apertus' post-training data and notably its [SFT mixture](https://huggingface.co/datasets/swiss-ai/apertus-sft-mixture). More information can be found in the [Apertus tech report](https://github.com/swiss-ai/apertus-tech-report).

## Pipeline Stages

The pipeline consists of the following self-contained stages:
1. **01-hf-download**: Downloads HuggingFace datasets with metadata tracking → produces saved HF Dataset or DatasetDict
2. **02-standardisation**: Converts datasets to unified chat format → produces HF DatasetDict  
3. **03-license-based-filtering**: Removes samples with licensing restrictions → produces HF DatasetDict
4. **04-decontamination**: Filters training candidates against evaluation benchmark references → produces HF DatasetDict
5. **05-annotations**: Adds LLM-based classifications and language detection → produces HF DatasetDict
6. **06-field-based-filtering**: General field analysis and filtering → produces HF DatasetDict
7. **07-dataset-aggregation**: Combines datasets into mixtures; separate scripts normalize or linearize legacy data
8. **08-judge-evaluation**: Evaluates datasets with LLM judges.

A few additional running scripts and miscellaneous commands are also provided in `examples`. 

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Native Apertus 2 processing

The separate [apertus2-processing](apertus2-processing/README.md) project checks
and tokenizes datasets in the native Apertus 2 format on Slurm clusters. It writes
an exact per-sample check report, or Megatron indexed datasets with optional loss
weights, and resumes interrupted jobs. It uses its own locked environment.

The numbered stages do not produce the native format, and `linearise-dataset.py`
is not needed for it. For data from this pipeline, complete the curation and
mixture selection your recipe requires, then map the selected branches to native
conversations with a [mapping script](apertus2-processing/README.md#mapping-a-dataset).
See the [native format](apertus2-processing/README.md#native-format) and how to
[check and tokenize](apertus2-processing/README.md#check-and-tokenize).
