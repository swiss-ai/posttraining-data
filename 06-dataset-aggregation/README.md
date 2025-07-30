# Dataset Aggregation
This script is meant to aggregate subsets of various datasets into a single data mix that can be used for training.

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