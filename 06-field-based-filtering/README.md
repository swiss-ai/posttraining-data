# 06-field-based-filtering: General Field Analysis and Filtering

Analyzes, filters, and splits chat format datasets by field values. Use dot notation for nested fields (e.g., `original_metadata.category`).

**Note**: License-based filtering has been moved to `03-license-based-filtering/` for better pipeline organization.

## Tool

- **field_filter.py**: General-purpose field analysis, filtering, and dataset splitting

## Usage

### Show Dataset Schema
```bash
# Show all fields and types
python 06-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture

# Show schema from first 10,000 samples only (faster for large datasets)
python 06-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --max-samples 10000
```

### Analyze Specific Field
```bash
# Show value statistics for a specific field
python 06-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --field original_metadata.category

# Analyze field from first 10,000 samples only (faster for large datasets)
python 06-field-based-filtering/field_filter.py data/02-standardised/tulu-3-sft-mixture --field original_metadata.category --max-samples 10000
```

### Filter Datasets
```bash
# Keep samples with specific values
python 06-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k \
  --output data/03-filtered/smoltalk-numina

# Keep samples with multiple values
python 06-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k smol-magpie-ultra \
  --output data/03-filtered/smoltalk-numina-magpie

# Remove samples with specific values  
python 06-field-based-filtering/field_filter.py data/02-standardised/dataset \
  --field dataset_source \
  --remove-values test-data \
  --output data/03-filtered/dataset-clean
```

### Split Datasets by Field Values
```bash
# Basic split - creates separate datasets for each field value
python 06-field-based-filtering/field_filter.py data/02-standardised/mydataset \
  --field original_metadata.category \
  --split \
  --output data/06-split-by-category/
# Creates: mydataset-math/, mydataset-science/, mydataset-history/, etc.

# Split with custom prefix
python 06-field-based-filtering/field_filter.py data/02-standardised/mydataset \
  --field status \
  --split "filtered-" \
  --output data/06-custom-splits/
# Creates: mydataset-filtered-active/, mydataset-filtered-inactive/, etc.

# Split boolean field
python 06-field-based-filtering/field_filter.py data/02-standardised/conversations \
  --field has_code \
  --split "code-" \
  --output data/06-code-splits/
# Creates: conversations-code-True/, conversations-code-False/
```

## Example Workflows

### Filtering Workflow
```bash
# 1. Show dataset schema
python 06-field-based-filtering/field_filter.py data/02-standardised/smoltalk

# 2. Analyze specific field (with sample limit for speed)
python 06-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source --max-samples 10000

# 3. Filter based on field values (processes all samples)
python 06-field-based-filtering/field_filter.py data/02-standardised/smoltalk \
  --field original_metadata.source \
  --keep-values numina-cot-100k smol-magpie-ultra \
  --output data/03-filtered/smoltalk-numina-magpie
```

### Splitting Workflow
```bash
# 1. Analyze field to see all unique values
python 06-field-based-filtering/field_filter.py data/02-standardised/mydataset \
  --field original_metadata.domain

# 2. Split dataset by domain values
python 06-field-based-filtering/field_filter.py data/02-standardised/mydataset \
  --field original_metadata.domain \
  --split \
  --output data/06-domain-splits/
# Creates separate datasets for each domain: mydataset-math/, mydataset-science/, etc.

# 3. Further process individual domain datasets as needed
python some-other-tool.py data/06-domain-splits/mydataset-math/
```

## Related Tools

For license-based filtering with pre-configured rules for specific datasets, see `03-license-based-filtering/`.