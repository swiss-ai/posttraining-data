# Post-training data processing pipeline

## Structure
This pipline processes raw dataset to a common schema and carries out necessary filtering before training.
The datasets are saved at the subdirectories of `/capstor/store/cscs/swissai/infra01/posttrain_data`.

The code is structured into 5 steps
1. `01-hf-download`: Downloads the datasets from the Huggingface website and saves them to the `01_raw_hf_data` folder
2. `02-standardisation`: Standardise the schema of the datasets to a common format. Loads data from `01_raw_hf_data` and saves to `02_standardised`.
3. `03-field-based-filtering`: Filters out fields that are not needed or unwanted (e.g. due to license issues). Loads data from `02_standardised` and saves to `03_license_filtered`.
4. `04-decontamination`: Removes data samples that have a significant overlap with any of the benchmarks to avoid contamination and unreliable evaluation results. Loads data from `03_license_filtered` and saves to `04_decontaminated`.
5. `05-annotations`: Annotates each data sample based on quality and whether it is an assistant response to remove unwanted artifacts and improve the data quality. Loads data from `04_decontaminated` and saves to `05_filtered`.
6. `06-dataset-aggregation`: Combines subsets from multiple datasets into a single mix for training. Loads data from `05_filtered` and saves to `06_sft_mixtures` or `07_alignment_mixtures` depending on the purpose of the dataset mix.
Please read the `READ.me` documentation within each folder for further information.

## Setup
Create virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```