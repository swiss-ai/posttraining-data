# Post-Training Data Processing Pipeline

A 7-stage pipeline for processing HuggingFace datasets into training-ready format.

## Pipeline Stages

1. **01-hf-download**: Downloads HuggingFace datasets with metadata tracking → produces HF DatasetDict
2. **02-standardisation**: Converts datasets to unified chat format → produces HF DatasetDict  
3. **03-license-based-filtering**: Removes samples with licensing restrictions → produces HF DatasetDict
4. **04-decontamination**: Removes contaminated samples from evaluation sets → produces HF DatasetDict
5. **05-annotations**: Adds LLM-based classifications and language detection → produces HF DatasetDict
6. **06-field-based-filtering**: General field analysis and filtering → produces HF DatasetDict
7. **07-dataset-aggregation**: Combines multiple datasets into training mixtures → produces HF Dataset ready for training

## Data Storage

Pipeline data is mirrored at: `/capstor/store/cscs/swissai/infra01/posttrain_data`

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```