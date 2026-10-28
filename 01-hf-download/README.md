# HuggingFace Dataset Download

This folder contains scripts for downloading and generating datasets from HuggingFace Hub.

## Scripts

### `hf-download.py`
Downloads datasets from HuggingFace Hub with compatibility checks and metadata tracking.

This script downloads HuggingFace datasets and stores them locally with metadata including version information, commit hash, and dataset details. It includes a compatibility check for the `datasets` library version (requires 3.3.2 for compatibility with current docker images).

### `generate-charter-qa.py`
Generates prompt-completion examples demonstrating awareness of [Swiss AI Charter](https://github.com/swiss-ai/apertus-tech-report/blob/main/Apertus_Tech_Report.pdf) principles. Creates diverse Q&A pairs covering all sections of the charter for training data.

## Usage

### hf-download.py Arguments

- `dataset_name`: Name of the dataset to download (e.g., 'allenai/tulu-3-sft-mixture')
- `--download-folder`: Folder to download the dataset to (required)
- `--subset`: Dataset subset/configuration name (optional)
- `--split`: Dataset split to download (e.g., 'train', 'test', 'validation') (optional)

## Examples

```bash
python 01-hf-download/hf-download.py allenai/tulu-3-sft-mixture --download-folder data/01-hf-data
python 01-hf-download/hf-download.py HuggingFaceTB/smoltalk --download-folder data/01-hf-data --subset all
python 01-hf-download/hf-download.py HuggingFaceTB/smoltalk2 --download-folder data/01-hf-data --subset SFT
python 01-hf-download/hf-download.py arcee-ai/The-Tome --download-folder data/01-hf-data
python 01-hf-download/hf-download.py utter-project/EuroBlocks-SFT-Synthetic-1124 --download-folder data/01-hf-data
python 01-hf-download/hf-download.py nvidia/Llama-Nemotron-Post-Training-Dataset --download-folder data/01-hf-data
```