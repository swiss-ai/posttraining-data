# 05-field-based-filtering: General Field Analysis and Filtering

Analyzes and filters chat format datasets by field values. Use dot notation for nested fields (e.g., `original_metadata.category`).

**Note**: License-based filtering has been moved to `03-license-based-filtering/` for better pipeline organization.

## Tool

- **field_filter.py**: General-purpose field analysis and filtering

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

## Related Tools

For license-based filtering with pre-configured rules for specific datasets, see `03-license-based-filtering/`.