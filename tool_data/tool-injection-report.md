# Tool Definition Injection Report

## Overview

Injected tool definitions into non-tool-calling SFT samples to create "tools-aware but not tool-calling" training data. This teaches the model when NOT to call tools when not necessary.
We augment as many samples as we have in the toolcalling datasets (See section "Tool Sources")

## Datasets

| Role | Path | Samples |
|------|------|---------|
| **Target (input)** | `/iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/v1p5-mix-v1-23-05-cleaned-linearised-fixed` | 3,529,012 |
| **Output** | `/capstor/store/cscs/swissai/infra01/tmp_data/v1p5-mix-v1-23-05-cleaned-linearised-fixed_injected_tool_calls` | 3,529,012 |

### Tool Sources

Sources for tool definitions

| Dataset | Path | Tool sets extracted |
|---------|------|---------------------|
| Toucan-1.5M_filtered | `/capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M_filtered` | 117,586 |
| EnvScaler-SFT-Traj-9K | `/capstor/store/cscs/swissai/infra01/tmp_data/EnvScaler-SFT-Traj-9K` | 9,022 |
| OpenSeeker-v1-Data | `/capstor/store/cscs/swissai/infra01/tmp_data/OpenSeeker-v1-Data` | 4,949 |
| **Total pool** | | **131,557** |

## Results

- **Augmented samples**: 131,557
- **Eligible candidates**: 2,533,174 / 3,529,012 (71.8%)
- **Seed**: 42
- **Date**: 2026-05-27

## Algorithm

1. **Extract tool definitions** from each tool source dataset — collect the `(tools, formatted_tools)` string pair from every developer message that has non-empty tools. This yields a pool of 131,557 tool set entries.

2. **Sample tool sets** from the pool without replacement (131,557 = full pool).

3. **Find eligible targets** in the SFT mix — samples where:
   - The developer message has empty `tools`
   - No assistant message contains actual `tool_calls` or `tool_outputs` blocks with non-empty content (this excludes `display_answers` blocks from verifiable-responses datasets like OpenMathReasoning, medmcqa, etc.)

4. **Randomly select** 131,557 eligible indices and pair each with a sampled tool set.

5. **Augment** selected samples and save the full dataset (augmented + unchanged).

## Mechanical Injection

In the linearised format, every sample has a developer message:

```json
{
  "role": "developer",
  "content": {
    "tools": "",
    "has_thinking": false,
    "formatted_tools": ""
  }
}
```

For each augmented sample, the script modifies **only** the developer message content:

- `tools` is set to the source sample's `tools` string (JSON-serialized list of tool definitions with name, description, parameters)
- `formatted_tools` is set to the source sample's `formatted_tools` string (TypeScript-type rendering of the tool schemas)
- `has_thinking` is **preserved** from the original sample (not overwritten)

All other messages (system, user, assistant) are left untouched. The assistant messages contain no tool calls — the model sees available tools but correctly chooses not to use them.

## Validation

After saving, the script reloads the output dataset from disk and performs round-trip validation:

- Total sample count preserved (3,529,012)
- Augmented samples have the correct `tools` and `formatted_tools` from their paired source
- `has_thinking` unchanged for all augmented samples
- Spot-check of up to 50,000 non-augmented samples confirms they are identical to input

## Reproducibility

The augmented indices are deterministic given seed=42: the script uses `random.Random(42)` for all sampling operations. Re-running with the same seed and inputs produces the same set of augmented samples.

### Full run

```bash
python tool_data/inject-tool-definitions.py \
    /iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/v1p5-mix-v1-23-05-cleaned-linearised-fixed \
    --tool-sources \
        /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M_filtered \
        /capstor/store/cscs/swissai/infra01/tmp_data/EnvScaler-SFT-Traj-9K \
        /capstor/store/cscs/swissai/infra01/tmp_data/OpenSeeker-v1-Data \
    --output /capstor/store/cscs/swissai/infra01/tmp_data/v1p5-mix-v1-23-05-cleaned-linearised-fixed_injected_tool_calls \
    --seed 42 \
    --save-indices
```

### Dry run (inspect which samples would be modified without writing a dataset)

```bash
python tool_data/inject-tool-definitions.py \
    /iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/v1p5-mix-v1-23-05-cleaned-linearised-fixed \
    --tool-sources \
        /capstor/store/cscs/swissai/infra01/tmp_data/Toucan-1.5M_filtered \
        /capstor/store/cscs/swissai/infra01/tmp_data/EnvScaler-SFT-Traj-9K \
        /capstor/store/cscs/swissai/infra01/tmp_data/OpenSeeker-v1-Data \
    --seed 42 \
    --dry-run \
    --dry-run-output injection_dry_run.json
```

### Cross-check samples between original and augmented datasets

```bash
python tool_data/cross-check-injection.py injection_dry_run.json \
    --original /iopsstor/scratch/cscs/hyukhymenko/sft-1.1-mixes/v1p5-mix-v1-23-05-cleaned-linearised-fixed \
    --augmented /capstor/store/cscs/swissai/infra01/tmp_data/v1p5-mix-v1-23-05-cleaned-linearised-fixed_injected_tool_calls \
    --num-samples 5 \
    --output injection_comparison.json
```

## Scripts

- `tool_data/inject-tool-definitions.py` — main injection script
- `tool_data/cross-check-injection.py` — cross-check utility
