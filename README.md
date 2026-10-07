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

The isolated [apertus2-processing](apertus2-processing/README.md) project checks
native conversation corpora, tokenizes HF/JSONL inputs, and exports HF, Parquet or
Megatron indexed datasets. It uses its own locked environment and supports
resumable shard jobs.

For existing pipeline data, complete the curation and mixture selection required
by your recipe, then convert selected branches to native Apertus JSON before
checking and tokenization. Stages 02 and 07 do not produce that native format;
`linearise-dataset.py` is a legacy path and is not required for the new encoder.
Already-native datasets can start directly with the native commands.

See [prerequisites and stage ordering](apertus2-processing/README.md#before-you-start),
[exact input layouts](apertus2-processing/README.md#native-input-format), and
[check/tokenize commands](apertus2-processing/README.md#check-then-tokenize).
