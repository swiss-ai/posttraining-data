# Tool Data

Scripts and reports for tool-calling data: validating linearised datasets, filtering flagged samples, injecting tool definitions into non-tool-calling samples, and cross-checking results.

## Scripts

| Script                                 | Purpose                                                                                           |
|----------------------------------------|---------------------------------------------------------------------------------------------------|
| `validate-linearised.py`              | Validate linearised datasets against the full Apertus format spec (general structure + tool use). |
| `filter-flagged.py`                   | Remove samples by validation error codes (uses flagged IDs from `validate-linearised.py`).        |
| `inject-tool-definitions.py`          | Inject tool definitions into non-tool-calling SFT samples ("tools-aware but not tool-calling").   |
| `cross-check-injection.py`            | Compare N random samples between original and augmented datasets to verify injection correctness. |

### validate-linearised.py — details

Checks the full linearised format, not only tool-related fields. Validation covers:

- **General structure**: required fields, message roles, role ordering, consecutive roles
- **System/developer messages**: presence, position, content types, duplicates
- **User messages**: parts structure, text types
- **Assistant messages**: block structure, block ordering, response/thoughts content
- **Tool definitions**: JSON validity, required fields (name, description, parameters), parameter schemas, formatted_tools consistency
- **Tool calls**: name presence, arguments validity, calls to undeclared tools (`call_unknown_tool`), call/output count mismatches
- **Tool outputs**: orphaned outputs, missing fields, type checks
- **Thinking**: has_thinking flag consistency with actual thoughts blocks

Each check has a unique code (e.g. `call_unknown_tool`, `empty_response`, `consecutive_roles`) classified as error `[E]` or warning `[W]`. Outputs:

- `--report-json`: per-code counts and per-sample issue details
- `--flagged-ids`: `{code: [conversation_id, ...]}` mapping for use with `filter-flagged.py`
- `--dump-code <code>`: print all samples matching a specific error/warning code

## Reports

| File                        | Contents                                                                              |
|-----------------------------|---------------------------------------------------------------------------------------|
| `tool-injection-report.md`  | Documents the injection run: datasets, algorithm, results, and reproduction commands. |
| `tool-datasets-validation-report.md`   | Validation results for tool-calling datasets (error/warning counts).          |
| `tool-datasets-validation-report.html` | HTML version of the validation report.                                        |
| `tool-datasets-filtering-report.md`  | Report on which samples were filtered and why.                                 |
| `validation-results/`       | Per-dataset validation outputs (flagged IDs, reports).                                |

## Typical Workflow

### 1. Validate a linearised dataset

```bash
python validate-linearised.py /path/to/linearised-dataset \
    --report-json validation-report.json \
    --flagged-ids flagged.json
```

### 2. Inspect specific errors

```bash
# Dump all samples with a specific error code
python validate-linearised.py /path/to/dataset --dump-code call_unknown_tool
```

### 3. Filter out problematic samples

```bash
python filter-flagged.py /path/to/dataset \
    --flagged-ids flagged.json \
    --remove-codes call_unknown_tool empty_response \
    --output /path/to/filtered-dataset
```

### 4. Inject tool definitions into non-tool-calling samples

```bash
# Full run
python inject-tool-definitions.py /path/to/sft-mix \
    --tool-sources /path/to/Toucan /path/to/EnvScaler /path/to/OpenSeeker \
    --output /path/to/output \
    --seed 42 --save-indices

# Dry run (preview only, no dataset written)
python inject-tool-definitions.py /path/to/sft-mix \
    --tool-sources /path/to/Toucan /path/to/EnvScaler /path/to/OpenSeeker \
    --seed 42 --dry-run --dry-run-output dry_run.json
```

### 5. Cross-check injection results

```bash
python cross-check-injection.py dry_run.json \
    --original /path/to/original \
    --augmented /path/to/augmented \
    --num-samples 5 \
    --output comparison.json
```
