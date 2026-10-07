# Dataset Aggregation

These scripts assemble datasets in the repository's legacy branched format
(`system_prompt`, `initial_prompt`, `available_functions`, and
`conversation_branches`). Run the commands below from the repository root in
the legacy pipeline environment.

For native Apertus 2 checking and tokenization, see
[Before you start](../apertus2-processing/README.md#before-you-start).
Neither aggregation nor the legacy linearizer produces native Apertus JSON.

## Create a filtered mixture

Add a named mixture to `07-dataset-aggregation/data-mixtures.yaml`:

```yaml
my-mixture:
  - dataset_path: "/path/to/curated-dataset"
    filters:
      - field: dataset_source
        values: ["selected-source"]
  - dataset_path: "/path/to/another-dataset"
    filters: []
```

Then run:

```bash
python 07-dataset-aggregation/create-mixture.py \
  --mixture_name my-mixture \
  --config_path 07-dataset-aggregation/data-mixtures.yaml \
  --output data/07-mixtures/my-mixture
```

The script combines **all splits** of each input, applies the configured
filters, harmonizes fields, and saves one HF Dataset. Supply only the intended
training splits; validation and test splits are otherwise included too.
Harmonization can replace `original_metadata` with an `original_id` field, so
apply metadata-dependent filters before that step and retain source datasets
for provenance.

## Concatenate datasets

For concatenation without mixture configuration:

```bash
python 07-dataset-aggregation/concatenate-datasets.py \
  data/06-filtered/first data/06-filtered/second \
  --output data/07-mixtures/combined
```

This also combines all input splits. By default it normalizes the legacy
schema, removes extra fields, serializes metadata as JSON strings, and saves
one Dataset. Use `--as-datasetdict` for a DatasetDict with a `train` split, or
`--no-normalize` when input schemas already match and normalization is not
wanted. Inspect the resulting schema before downstream use.

## Legacy SFT linearization

For the existing legacy training format:

```bash
python 07-dataset-aggregation/linearise-dataset.py \
  data/07-mixtures/my-mixture data/07-linearised/my-mixture \
  --training-type sft
```

This selects the first conversation branch and produces a `messages` column
with structured `content`, `parts`, and `blocks`. It retains
`conversation_id`, `dataset_source`, `original_metadata`, and
`created_timestamp`, but removes other original columns. Complete filtering
that depends on those fields before linearization. Only SFT is implemented.

For native Apertus 2, use an explicit legacy-to-native conversion before this
linearization step instead. That adapter is not provided here; branch
selection, tool associations, turn boundaries, and metadata preservation need
an explicit policy. A legacy `messages` column cannot be passed directly to
`apertus-data` as native JSON.
