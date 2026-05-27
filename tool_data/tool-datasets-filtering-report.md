# Linearised Dataset Validation & Filtering Report

I inspected the tool call datasets automatically and looking at samples. TLDR: we have some samples filtered out from Tucan but the rest seems correct.

## Validation Checks

Each linearised dataset was validated against the Apertus format specification. The checks cover:

- **Top-level fields** — presence of conversation_id, dataset_source, messages
- **System/developer/user/assistant message structure** — required fields, content types, block ordering
- **Tool definitions** — name, description, parameters schema
- **Tool calls & outputs** — valid JSON arguments, call names match declared tools, call/output pairing
- **Role sequence** — system first, developer early, no consecutive same-role messages

Issues are classified as errors (structural violations) or warnings (suspicious but valid).

## Validation Results

### Toucan-1.5M — 119,287 samples, 0 errors

| Warning Code             | Occurrences | Description                        |
| ------------------------ | ----------- | ---------------------------------- |
| tool_empty_description   | 9,076       | tool definition with empty desc    |
| call_unknown_tool        | 1,371       | call name not in declared tools    |
| call_arguments_json      | 1,072       | call arguments not valid JSON      |
| empty_outputs            | 84          | tool_outputs with only placeholders|

We tolerate empty tool descriptions as long as the tool calling is done correclty.

### EnvScaler-SFT-Traj-9K — 9,022 samples, 0 errors

Clean.

### OpenSeeker-v1-Data — 4,949 samples, 0 errors

Clean.

## Filtering Applied

Toucan-1.5M was filtered for malformed tool calls and empty tool outputs:

| Code                  | Unique Samples | Description                        |
| --------------------- | -------------- | ---------------------------------- |
| call_unknown_tool     | 768            | calls to undeclared tools          |
| call_arguments_json   | 904            | arguments not valid JSON           |
| empty_outputs         | 68             | tool_outputs with only placeholders|

Result: 119,287 -> 117,586 samples (1,701 removed, some overlap between codes).
Output: /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M_filtered

No filtering was applied to EnvScaler-SFT-Traj-9K or OpenSeeker-v1-Data.

## Reproducing

All commands run from the repo root. Data lives under `/capstor/store/cscs/swissai/infra01/tmp_data/`.

### 1. Validation

```bash
# Toucan
python tool_data/validate-linearised.py \
  /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M \
  --report-json tool_data/validation-results/toucan_report.json \
  --flagged-ids tool_data/validation-results/toucan_flagged.json

# EnvScaler
python tool_data/validate-linearised.py \
  /capstor/store/cscs/swissai/infra01/tmp_data/EnvScaler-SFT-Traj-9K \
  --report-json tool_data/validation-results/envscaler_report.json \
  --flagged-ids tool_data/validation-results/envscaler_flagged.json

# OpenSeeker
python tool_data/validate-linearised.py \
  /capstor/store/cscs/swissai/infra01/tmp_data/OpenSeeker-v1-Data \
  --report-json tool_data/validation-results/openseeker_report.json \
  --flagged-ids tool_data/validation-results/openseeker_flagged.json
```

### 2. Filtering

```bash
python tool_data/filter-flagged.py \
  /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M \
  --flagged-ids tool_data/validation-results/toucan_flagged.json \
  --remove-codes call_unknown_tool call_arguments_json empty_outputs \
  --output /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M_filtered
```

### Output Files

Result JSONs are in `tool_data/validation-results/`:

- `*_report.json` — validation summary (error/warning counts per split)
- `*_flagged.json` — mapping of warning codes to conversation_ids
- `flagged_samples.json` — 5 full samples per filtered category for manual inspection
