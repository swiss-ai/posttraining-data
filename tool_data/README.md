# Tool Data

Scripts and reports used to validate and fix the tool-calling data of the Apertus 1.5 text SFT mix,
on the linearised format (`linearisation-format.md`).

| Script | Purpose |
|---|---|
| `validate-linearised.py` | Check a linearised dataset against the format spec; writes issue counts and flagged IDs per code |
| `filter-flagged.py` | Remove samples flagged with given codes |
| `strip-display-answers.py` | Remove undeclared `display_answers` call/output blocks created from `verifiable-responses` |
| `inject-tool-definitions.py` | Add tool definitions to non-tool samples, so the model learns when not to call a tool |
| `cross-check-injection.py` | Compare random samples before and after injection |
| `find-display-answers-datasets.py` | Diagnostic: count `display_answers` calls per source, declared or not |
| `analyze-undeclared-tool-usage.py` | Diagnostic: statistics on calls to undeclared tools |
| `analyze-unknown-tool-corrections.py` | Diagnostic: what follows a call to an unknown tool: retry, other tool, or answer |

Tool datasets (Toucan, EnvScaler, OpenSeeker):

```bash
python tool_data/validate-linearised.py TOOLS --report-json report.json --flagged-ids flagged.json
python tool_data/filter-flagged.py TOOLS --flagged-ids flagged.json --remove-codes call_unknown_tool call_arguments_json empty_outputs --output TOOLS_FILTERED
```

SFT mix:

```bash
python tool_data/validate-linearised.py MIX --report-json report.json --flagged-ids flagged.json
python tool_data/strip-display-answers.py MIX --flagged-ids flagged.json --output FIXED
python tool_data/inject-tool-definitions.py FIXED --tool-sources TOOLS_FILTERED ... --seed 42 --dry-run --dry-run-output indices.json
python tool_data/inject-tool-definitions.py FIXED --tool-sources TOOLS_FILTERED ... --seed 42 --output INJECTED
python tool_data/cross-check-injection.py indices.json --original FIXED --augmented INJECTED --output check.json
```

Diagnostics (not needed to reproduce the outputs):

```bash
# why call_unknown_tool samples are removed rather than repaired
python tool_data/analyze-unknown-tool-corrections.py TOOLS --flagged-ids flagged.json
# which tools and sources tool_use_without_declarations comes from
python tool_data/analyze-undeclared-tool-usage.py MIX --flagged-ids flagged.json --output undeclared.json
# display_answers per source, declared or not: why only undeclared ones are stripped
python tool_data/find-display-answers-datasets.py MIX --output display_answers.json
```
