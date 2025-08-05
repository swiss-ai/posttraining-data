# Dataset Aggregation
This script is meant to aggregate subsets of various datasets into a single data mix that can be used for training.

## Format Conversion

Convert datasets from old format (string content) to new format (parts structure):

```bash
# Convert dataset to directory (preserves dataset name)
python convert_old_to_new_format.py ../data/02-standardised/tulu-3-sft-mixture ../data/converted/

# Convert with specific output name
python convert_old_to_new_format.py input_dataset output_dataset

# Convert with validation disabled (faster)
python convert_old_to_new_format.py input_dataset output_dataset --no-validate

# Convert only first N samples (for testing)
python convert_old_to_new_format.py input_dataset output_dataset --sample 100
```

## Usage
To add a new dataset, first update the `data-mixtures.yaml` file as follows
```yaml
new-data-mix-name:
  - dataset_path: "/path/to/first/input/dataset"
    filters:
      - field: field-name-1
        values:
          - value-to-keep-1
          - value-to-keep-2
          - ...
      - field: field-name-2
        values:
          - value-to-keep-1
          - value-to-keep-2
          - ...
      - ...
  - dataset_path: "/path/to/second/input/dataset"
    filters:
      - field: field-name-1
        values:
          - ...
      - ...
  - ...
```

Then you can generate the new mix as