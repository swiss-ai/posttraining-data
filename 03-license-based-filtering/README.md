# 03-license-based-filtering: License-Based Dataset Filtering

Removes samples from datasets based on known licensing issues or problematic sources. Each dataset has custom filtering rules based on licensing restrictions and data quality concerns.

## Available Scripts

- **`license_filter.py`**: For old format datasets (string content)
- **`license_filter_newformat.py`**: For new format datasets (parts structure)

Both scripts use identical filtering logic since they operate on top-level fields like `original_metadata` which are the same in both formats.

## Usage

### Apply License Filtering

#### Old Format Datasets
```bash
# Apply license filtering with default output path
venv/bin/python 03-license-based-filtering/license_filter.py data/02-standardised/tulu-3-sft-mixture

# Specify custom output path  
venv/bin/python 03-license-based-filtering/license_filter.py data/02-standardised/smoltalk \
  --output data/03-license-filtered/smoltalk

# Use chunked processing for large datasets
venv/bin/python 03-license-based-filtering/license_filter.py data/02-standardised/smoltalk2 \
  --chunk-size 50000
```

#### New Format Datasets
```bash
# Apply license filtering with default output path
venv/bin/python 03-license-based-filtering/license_filter_newformat.py data/02-standardised-newformat/tulu-3-sft-mixture

# Specify custom output path  
venv/bin/python 03-license-based-filtering/license_filter_newformat.py data/02-standardised-newformat/smoltalk \
  --output data/03-license-filtered-newformat/smoltalk

# Use chunked processing for large datasets
venv/bin/python 03-license-based-filtering/license_filter_newformat.py data/02-standardised-newformat/smoltalk2 \
  --chunk-size 50000
```

### List Available Filters
```bash
# Works with both scripts
venv/bin/python 03-license-based-filtering/license_filter.py --list-filters
venv/bin/python 03-license-based-filtering/license_filter_newformat.py --list-filters
```

### Force Format for Custom Filtering
```bash
# Use another dataset's filter configuration (old format)
venv/bin/python 03-license-based-filtering/license_filter.py data/02-standardised/custom-dataset \
  --force-format smoltalk

# Use another dataset's filter configuration (new format)
venv/bin/python 03-license-based-filtering/license_filter_newformat.py data/02-standardised-newformat/custom-dataset \
  --force-format smoltalk
```

## Supported Datasets

### Standard Source Exclusion
- **tulu-3-sft-mixture**: Removes ai2-adapt-dev sources with licensing issues
- **tulu-3-sft-olmo-2-mixture-0225**: Removes olmo_hardcoded and ai2-adapt-dev sources with licensing issues
- **smoltalk**: Excludes openhermes-100k, longalign, explore-instruct-rewriting
- **The-Tome**: Filters out infini-instruct, ultrainteract, qwen2-magpie sources
- **AceReason-1.1-SFT**: Removes leetcode samples

### License-Based Inclusion
- **Llama-Nemotron-Post-Training-Dataset**: Keeps only cc-by-4.0 and odc-by licenses

### Complex Split Handling
- **smoltalk2**: Removes problematic splits and concatenates remaining data with source augmentation

### No Filtering Required
- **EuroBlocks-SFT-Synthetic-1124**: No filtering needed

## Output

Filtered datasets are saved to:
- **Old format**: `data/03-license-filtered/` by default
- **New format**: `data/03-license-filtered-newformat/` by default

Both maintain:
- DatasetDict format for consistency with pipeline expectations
- Complete metadata tracking with processing history
- Sample removal statistics and reasoning

## Memory Efficiency

For large datasets (>500k samples), the tool automatically uses chunked processing to manage memory usage efficiently, similar to the standardisation pipeline.