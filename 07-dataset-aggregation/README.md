# Dataset Aggregation
This script is meant to aggregate subsets of various datasets into a single data mix that can be used for training.

**Format Support**: The aggregation script operates on dataset-level fields rather than message content.

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
## Concatenating datasets (`concatenate-datasets.py`)

Concatenates standardised datasets into one mix. Passing the same input path more than once upsamples it.

```bash
python 07-dataset-aggregation/concatenate-datasets.py \
  /path/to/04_decontaminated/dataset-a /path/to/04_decontaminated/dataset-b /path/to/04_decontaminated/dataset-a \
  -o /path/to/mix --num-proc 16
```

Before concatenating, two checks stop the script with a list of the offending inputs:

- **Decontamination:** every input's `dataset_metadata.json` must contain a successful decontamination
  entry (`04-decontamination`) covering every benchmark of the prompt set
  (`--decontamination-prompts`, default: the standard set on capstor).
- **conversation_ids:** no empty IDs, no duplicates within an input, and no IDs shared between different
  inputs (repeating the same path for upsampling is allowed).

`--skip-decontamination-check` and `--skip-id-check` disable these checks for legacy datasets (e.g. the
v1.0 / v1.5 outputs, which predate them).
