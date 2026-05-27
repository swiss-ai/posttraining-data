# Linearised Dataset Validation Report

**Generated**: 2025-05-26
**Validator**: `07-dataset-aggregation/validate-linearised.py`
**Datasets validated**: EnvScaler, OpenSeeker, Toucan

---

## Summary

| Dataset | Samples | Errors | Samples with Warnings | Warning Count |
|---------|--------:|-------:|----------------------:|--------------:|
| EnvScaler SFT Traj 9K | 9,022 | 0 | 1 | 1 |
| OpenSeeker | 4,949 | 0 | 0 | 0 |
| Toucan 1.5M | 119,287 | 0 | 78,359 | 200,584 |
| **Total** | **133,258** | **0** | **78,360** | **200,585** |

All three datasets pass validation with **zero errors**. Warnings are non-blocking quality signals.

---

## Warning Breakdown by Dataset

### EnvScaler SFT Traj 9K

| Warning Code | Count |
|-------------|------:|
| `thoughts_after_response` | 1 |

### OpenSeeker

No warnings.

### Toucan 1.5M

| Warning Code | Count |
|-------------|------:|
| `calls_after_response` | 188,981 |
| `tool_empty_description` | 9,076 |
| `call_unknown_tool` | 1,371 |
| `call_arguments_json` | 1,072 |
| `empty_outputs` | 84 |

---

## Detailed Findings

### 1. `calls_after_response` (Toucan — 188,981 occurrences)

**What it detects**: A tool call block appears after a response block within the same assistant message.

**Root cause**: Toucan's function-calling conversations use a pattern where the assistant first provides a natural-language response and then issues one or more tool calls in the same turn. This is a deliberate conversational design — the assistant explains its intent before executing the tool call.

**Training impact**: Low risk. This ordering is common in production tool-use agents. Many samples have multiple tool calls after a response, inflating the count (188,981 occurrences across 78,359 samples). The pattern is consistent and intentional.

**Real example** (`Toucan_1.5M_be9e49f2de05`, message index 3):
```
assistant message blocks:
  [0] response:    "I'll help you find rhyming options for your birthday poem!
                    Let me start by getting rhyming words for \"smile\" and
                    counting the syllables in your current line."
  [1] tool_calls:  lyrical-mcp-find_rhymes({"input_word": "smile"})
  [2] tool_outputs: {"1_syllable": ["aisle","bile","bille",...]}
  [3] tool_calls:  lyrical-mcp-count_syllables({"input_string": "With joy that leaps from every smile"})
  [4] tool_outputs: "9"
  [5] response:    "Great! Here's what I found: ## Rhyming Options for \"smile\": ..."
```
The assistant explains intent (block 0), then executes tool calls (blocks 1-4), then summarises results (block 5).

### 2. `tool_empty_description` (Toucan — 9,076 occurrences)

**What it detects**: A declared tool has an empty or missing `description` field.

**Root cause**: Some Toucan samples declare tools with only a `name` and `parameters` schema but no `description`. This likely comes from source datasets where tool descriptions were optional or stripped during conversion.

**Training impact**: Medium. Models trained on these samples may learn that tool descriptions are optional, which could reduce the quality of tool selection in ambiguous cases. However, the tool name and parameter schema still provide strong signal.

**Real example** (`Toucan_1.5M_bfde81ab1e50`, message index 1):
```
Developer message declares 5 tools. 3 have empty descriptions:

  name: pubmed-mcp-server-search_pubmed_key_words
  description: ''
  parameters: [key_words, num_results]

  name: pubmed-mcp-server-search_pubmed_advanced
  description: ''
  parameters: [term, title, author, journal, start_date]

  name: pubmed-mcp-server-get_pubmed_article_metadata
  description: ''
  parameters: [pmid]
```

### 3. `call_unknown_tool` (Toucan — 1,371 occurrences)

**What it detects**: The assistant calls a tool name that was not declared in the developer message's tool list.

**Root cause**: These fall into two categories:
1. **Misspelled or reformatted names** — e.g., the declared tool is `get_weather` but the call uses `getWeather`.
2. **Hallucinated tools** — the assistant invents a tool that was never provided.

**Training impact**: Medium-high. This teaches the model to call tools outside the declared set, which is undesirable. These samples should be reviewed. However, 1,371 out of 119,287 samples (1.1%) is a small fraction.

**Real example** (`Toucan_1.5M_5ecf98bdcfde`, message index 3):
```
Declared tools (4):
  - shadcn/ui-component-reference-server-get_component_details
  - shadcn/ui-component-reference-server-get_component_examples
  - shadcn/ui-component-reference-server-list_shadcn_components
  - shadcn/ui-component-reference-server-search_components

Called tool (NOT in declared list):
  name: search_components
  args: {"query": "form stepper multi-step animated transitions"}
```
The assistant dropped the `shadcn/ui-component-reference-server-` prefix, calling `search_components` instead of the declared `shadcn/ui-component-reference-server-search_components`.

### 4. `call_arguments_json` (Toucan — 1,072 occurrences)

**What it detects**: The `arguments` field of a tool call is not valid JSON.

**Root cause**: The arguments contain malformed JSON — truncated strings, unquoted keys, trailing commas, or mixed formats. This appears to come from source data where tool call arguments were not properly serialized.

**Training impact**: Medium-high. Invalid JSON arguments will teach the model to produce malformed tool calls. These samples are candidates for filtering or repair.

**Real example** (`Toucan_1.5M_9a50fda6ebd0`, message index 3):
```
Tool: two-truths-and-a-twist-create_round
Arguments (634 chars, invalid JSON):
  {"category": "Gaming & Technology", "question": "According to current
  Steam data, which surprising fact reveals how player preferences are
  shifting in 2024?", "trivia_1": "The Strategy genre has only 1 game
  but nearly 380,000 players - that's more players per game than ANY
  other genre on Steam", "trivia_2": \"Action games dominated with
  over 1.35 million active players, ...\"  ...}

JSON parse error at char 309: Expecting value
  → Escaped quotes (\") inside an already-quoted JSON string value
```

### 5. `empty_outputs` (Toucan — 84 occurrences)

**What it detects**: A `tool_outputs` block contains outputs where every entry has an empty `output` string.

**Root cause**: The tool was called and returned an empty response. This can happen with side-effect-only tools (e.g., `send_email`) or when the source data did not capture the tool output.

**Training impact**: Low. 84 occurrences is negligible. Empty outputs for side-effect tools are semantically valid.

**Real example** (`Toucan_1.5M_430a1f87bc59`, message index 3):
```
Preceding tool call:
  name: votars-mcp-Votars fetch a specific transcript
  args: {"id": 98765}

Tool output:
  name: ""
  output: ""    ← empty
```

### 6. `thoughts_after_response` (EnvScaler — 1 occurrence)

**What it detects**: A thinking/thoughts block appears after a response block within the same assistant message.

**Root cause**: A single sample (`EnvScaler_SFT_Traj_9K_5b5735c7ebc0`) has the assistant produce a response followed by additional thinking. This is likely a conversion artefact.

**Training impact**: Low. One sample out of 9,022. The thinking-after-response pattern contradicts the expected flow (think first, then respond). Could be filtered or left as noise.

**Real example** (`EnvScaler_SFT_Traj_9K_5b5735c7ebc0`, message index 5):
```
assistant message blocks:
  [0] thoughts:  "Okay, let's see. The user asked for the current status of
                  EQ004 and the full definition of the SFT-NIGHT shift. I
                  already called get_equipment_status ..."
  [1] response:  "The current status of equipment EQ004 is **under maintenance**.
                  Regarding the SFT-NIGHT shift definition, here are the details:
                  - **Shift ID**: S..."
  [2] thoughts:  "Okay, let me go through the user's request step by step. ..."   ← AFTER response
  [3] response:  "The current status of equipment **EQ004** is **under
                  maintenance**. The **SFT-NIGHT shift** details are as follows:
                  - **Start Time**: 22:00 ..."
```
The assistant thinks, responds, then thinks again and responds again — a duplicated think→respond cycle within one message.

---

## Recommendations

| Priority | Action | Affected Samples |
|----------|--------|-----------------|
| Consider | Review `call_unknown_tool` samples for filtering | ~1,371 (Toucan) |
| Consider | Review `call_arguments_json` samples for repair or filtering | ~1,072 (Toucan) |
| Low | Optionally filter `thoughts_after_response` sample | 1 (EnvScaler) |
| Info | `calls_after_response` is intentional pattern — no action needed | 78,359 (Toucan) |
| Info | `tool_empty_description` is cosmetic — tools still have name+params | 9,076 (Toucan) |
| Info | `empty_outputs` is expected for side-effect tools | 84 (Toucan) |

---

## Exported Report Files

| File | Contents |
|------|----------|
| `/tmp/envscaler_report.json` | EnvScaler validation summary (counts by error/warning code) |
| `/tmp/envscaler_flagged.json` | EnvScaler flagged conversation IDs by code |
| `/tmp/openseeker_report.json` | OpenSeeker validation summary |
| `/tmp/openseeker_flagged.json` | OpenSeeker flagged conversation IDs by code |
| `/tmp/toucan_report.json` | Toucan validation summary |
| `/tmp/toucan_flagged.json` | Toucan flagged conversation IDs by code |

These JSON files were generated using:
```bash
python validate-linearised.py <dataset_path> --report-json /tmp/<name>_report.json --flagged-ids /tmp/<name>_flagged.json
```

To inspect specific flagged samples:
```bash
python validate-linearised.py <dataset_path> --dump-code <warning_code> --limit 5
```
