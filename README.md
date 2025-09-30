# Post-Training Data Processing Pipeline

`posttraining-data` is a turn-key 7-stage pipeline for processing HuggingFace datasets into training-ready format. It was used to prepare Apertus' post-training data and notably its [SFT mixture](https://huggingface.co/datasets/swiss-ai/apertus-sft-mixture). More information can be found in the [Apertus tech report](https://github.com/swiss-ai/apertus-tech-report).

## Pipeline Stages

The pipeline consists of the following self-contained stages:
1. **01-hf-download**: Downloads HuggingFace datasets with metadata tracking → produces HF DatasetDict
2. **02-standardisation**: Converts datasets to unified chat format → produces HF DatasetDict  
3. **03-license-based-filtering**: Removes samples with licensing restrictions → produces HF DatasetDict
4. **04-decontamination**: Removes contaminated samples from evaluation sets → produces HF DatasetDict
5. **05-annotations**: Adds LLM-based classifications and language detection → produces HF DatasetDict
6. **06-field-based-filtering**: General field analysis and filtering → produces HF DatasetDict
7. **07-dataset-aggregation**: Combines multiple datasets into training mixtures → produces HF Dataset ready for training
8. **08-judge-evaluation**: Evaluates datasets with LLM judges.

A few additional running scripts and miscellaneous commands are also provided in `examples`. 

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
