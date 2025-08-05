# 03-license-based-filtering: License-Based Dataset Filtering

Removes samples from datasets based on known licensing issues or problematic sources. Each dataset has custom filtering rules based on licensing restrictions and data quality concerns.

## Usage

### Apply License Filtering
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

### List Available Filters
```bash
venv/bin/python 03-license-based-filtering/license_filter.py --list-filters
```

### Force Format for Custom Filtering
```bash
# Use another dataset's filter configuration
venv/bin/python 03-license-based-filtering/license_filter.py data/02-standardised/custom-dataset \
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

Filtered datasets are saved to `data/03-license-filtered/` by default, maintaining:
- DatasetDict format for consistency with pipeline expectations
- Complete metadata tracking with processing history
- Sample removal statistics and reasoning

## Memory Efficiency

For large datasets (>500k samples), the tool automatically uses chunked processing to manage memory usage efficiently, similar to the standardisation pipeline.