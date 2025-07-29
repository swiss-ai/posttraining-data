# Field-Based Dataset Filtering

Analyzes and filters chat format datasets by field values. Use dot notation for nested fields (e.g., `original_metadata.category`).

## Tools

- **field_filter.py**: General-purpose field analysis and filtering
- **license_filter.py**: Pre-configured license-based filtering for specific datasets

## Usage

### Show Dataset Schema
```bash
# Show all fields and types
python 05-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture

# Show schema from first 10,000 samples only (faster for large datasets)
python 05-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --max-samples 10000
```

### Analyze Specific Field
```bash
# Show value statistics for a specific field
python 05-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --field original_metadata.category

# Analyze field from first 10,000 samples only (faster for large datasets)
python 05-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --field original_metadata.category --max-samples 10000
```

### Filter Datasets
```bash
# Keep samples with specific values
python 05-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k \
  --output data/03-filtered/smoltalk-numina

# Keep samples with multiple values
python 05-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k smol-magpie-ultra \
  --output data/03-filtered/smoltalk-numina-magpie

# Remove samples with specific values  
python 05-field-based-filtering/field_filter.py data/02-standardised/dataset \
  --field dataset_source \
  --remove-values test-data \
  --output data/03-filtered/dataset-clean
```

## Example Workflow
```bash
# 1. Show dataset schema
python 05-field-based-filtering/field_filter.py data/02-standardised/smoltalk

# 2. Analyze specific field (with sample limit for speed)
python 05-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source --max-samples 10000

# 3. Filter based on field values (processes all samples)
python 05-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k smol-magpie-ultra \
  --output data/03-filtered/smoltalk-numina-magpie
```

## License Filtering

The `license_filter.py` tool provides pre-configured filtering rules for removing samples with licensing restrictions from specific datasets.

### Supported Datasets
- **tulu-3-sft-mixture**: Removes ai2-adapt-dev sources with licensing issues
- **smoltalk**: Excludes openhermes-100k, longalign, explore-instruct-rewriting
- **smoltalk2**: Removes problematic splits and concatenates remaining data
- **The-Tome**: Filters out infini-instruct, ultrainteract, qwen2-magpie sources
- **AceReason-1.1-SFT**: Removes leetcode samples
- **Llama-Nemotron-Post-Training-Dataset**: Keeps only cc-by-4.0 and odc-by licenses
- **EuroBlocks-SFT-Synthetic-1124**: No filtering needed

### Usage
```bash
# Apply license filtering with default output path
python 05-field-based-filtering/license_filter.py data/02-standardised/tulu-3-sft-mixture

# Specify custom output path
python 05-field-based-filtering/license_filter.py data/02-standardised/smoltalk \
  --output data/03-license-filtered/smoltalk

# List all configured filters
python 05-field-based-filtering/license_filter.py --list-filters
```

### Output
Filtered datasets are saved to `data/03-license-filtered/` by default, maintaining:
- DatasetDict format for consistency
- Complete metadata tracking with processing history
- Sample removal statistics