# Coding SFT recipe

Revision-pinned download, source-column unification, exact/fuzzy deduplication and
benchmark decontamination using the repository's existing stages:

| Stage | Implementation | Output |
| --- | --- | --- |
| [01: download](../../01-hf-download/code-sft/) | Manifest resolution, source download, evaluation references | Pinned catalog and unified Parquet |
| [02: standardisation](../../02-standardisation/code-sft/) | Column mappings, chat-role normalization, source adapters | Unified Parquet with original rows and provenance |
| [04: decontamination](../../04-decontamination/code-sft/) | Global exact/optional fuzzy training dedup; evaluation matching | Clean Parquet, decisions and removal reports |

The runner here connects these stages and supplies their import paths. It does not
run license filtering, annotation, mixture selection or judge evaluation. This
recipe's Parquet schema differs from the legacy HF DatasetDict conversation schema;
use the appropriate conversion before passing its output to other numbered scripts.

## Environment

Run from this directory:

```bash
uv sync --python 3.11 --locked
uv run python pipeline.py --help
uv run pytest -q
```

The recipe has its own environment because it needs newer datasets/PyArrow packages
than some legacy stages. Linux is required for the dedup engine's fork workers and
advisory output locks. Set `HF_HOME` for the download cache and `HF_TOKEN` for gated
sources. Access restrictions remain failures; no dataset is silently counted as
downloaded.

## Download and normalize

Edit the example manifest for your selected sources and code-selection policies,
then resolve source files at immutable dataset revisions:

```bash
uv run python pipeline.py resolve --manifest manifest.example.yaml \
  --output /path/to/catalog
uv run python pipeline.py download --catalog /path/to/catalog/catalog.json \
  --output /path/to/unified --workers 16
```

The resolver records source-card evidence and unresolved repositories. The downloader
uses file-level checkpoints, writes per-source completion/failure markers and exits
nonzero for incomplete runs. Source-specific code-selection policies are explicit
in the catalog; excluded non-code/unknown row counts appear in completion reports.
`code_source` preserves every row in an already scoped coding source. Do not apply
this policy to mixed-domain releases without declaring the intended scope.

For already downloaded JSONL/Parquet files, map source columns explicitly:

```bash
uv run python pipeline.py unify --input /path/to/source.jsonl \
  --repo example/source --revision source-version --output /path/to/unified \
  --column-map column-map.example.json
```

The JSON map is source name to canonical name, for example
`{"task_text": "prompt", "completion_text": "response"}`. Aliases such as
`instruction`, `question`, `output`, `answer`, `conversations`, `human` and `gpt`
are normalized as well. Unavailable values stay missing; conflicting explicit
mappings fail. The original input object remains in `source_record_json`, alongside
repository/revision/file/row provenance. The local unifier performs no domain
filtering and refuses an existing output source directory; use a fresh directory
after a failed local conversion.

## Deduplicate and filter evaluation overlap

```bash
uv run python pipeline.py benchmarks --output /path/to/evaluation --workers 8
uv run python pipeline.py dedup --input /path/to/unified \
  --benchmarks /path/to/evaluation --output /path/to/dedup \
  --categories code_sft code_agentic --workers 16 --memory-limit 32GB --no-fuzzy
```

Evaluation references remain separate from training inputs. Their revision lock is
under `01-hf-download/code-sft/`. Benchmark preparation excludes evaluation training
splits according to the manifest, and dedup refuses incomplete reference coverage.

Exact sample dedup preserves case and code whitespace, normalizing line endings.
It includes reasoning, tools and tests in sample fingerprints. Benchmark matching
also checks normalized text equality and containment (minimum 80 characters).
`--fuzzy` adds LSH candidate generation with token-Jaccard verification; it can miss
near matches and is not a semantic-equivalence check. `--report-only` calculates
decisions without materializing clean data. Optional `--exclusions PATH` reads
`{"datasets": [{"repo": "owner/name", "reason": "..."}]}` and records omissions.

Clean shards are under `/path/to/dedup/clean/`; `report.json` summarizes reasons and
per-source counts, while `_work/` retains fingerprint/decision evidence. Run
identity includes input snapshots and matching settings. Resume with identical
inputs/settings; choose a fresh output for a changed run.

## Debug partition

```bash
sbatch debug.sh dedup --input /path/to/unified --benchmarks /path/to/evaluation \
  --output /path/to/dedup --categories code_sft code_agentic --workers 16 --no-fuzzy
```

The example launcher requests `infra01`, partition `debug`, 16 CPUs and one hour.
It expects this directory as the submission directory and an environment available
on compute nodes. `CODE_SFT_PYTHON` can select another Python environment. Resubmit
resumable stages explicitly if the allocation expires.

## Validation and origin

The code is adapted from [User3574/code-data-processing](https://github.com/User3574/code-data-processing/tree/be3ecc1),
limited to these three stages. Historical audit utilities, inventories, native
conversion, runtime archives and corpus artifacts are not included. The adapter
registry uses explicit catalog policies and surviving source adapters; it has not
been validated over the entire production corpus after restructuring.

Tests cover source preservation, column mapping, role aliases, source selection,
SQL/instruction/benchmark adapters, exact/fuzzy matching and resume checks. A local
fixture test runs unification through materialized dedup output, checking that one
duplicate and one benchmark match are removed. Tests do not download the full corpus.
