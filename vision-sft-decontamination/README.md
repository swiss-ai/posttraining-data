# vision-sft-decontamination

Per-dataset pipeline for decontaminating vision-SFT corpora against the
swiss-ai text-benchmark set. Built as a sibling of `04-decontamination` because
vision SFT ships in many native shapes (parquet, JSON, JSONL, HF arrow, CSV)
with image bytes embedded; this pipeline strips media, dedups prompts, and
runs benchmark overlap matching only on the unique prompt set.

## Why a separate pipeline

- Vision SFT is hundreds of GB to TB per dataset, mostly image bytes. The text
  fraction we actually care about is a tiny slice.
- Prompts are heavily templated — many datasets reuse the same skeleton with
  numbers or names swapped. Without dedup we score the same prompt thousands
  of times.
- The canonical step-4 path expects a standardised HF `DatasetDict` from step 2
  and writes a filtered DatasetDict copy of the training corpus. For vision
  SFT we want neither: no standardised intermediate (it would copy the text
  again) and no filtered output (we just want IDs + benchmark reports).
- We need adapters per dataset family, not a one-size loader.

## Three-stage pipeline

1. **Dedup** — `dedup/compute-<dataset>.py` reads the raw shards, extracts
   `(id, prompt, response)` via the dataset's native schema, writes a tiny
   per-shard parquet of three hash columns:
   - `prompt_hash` — lowercase + whitespace-collapse + strip `<image>` →
     verbatim duplicates
   - `prompt_skeleton_hash` — additionally maps numeric runs to `<N>` and
     bbox-like coord lists to `<BBOX>` → catches "Provide a short description
     for the given region. [x,y,x,y]" style clusters
   - `prompt_prefix8_hash` — hash of the first 8 normalised tokens → catches
     slot-filled stems like "What is the main focus of the …"

   `dedup/aggregate-clusters.py` (polars) collapses per-shard outputs to
   cluster tables. Output: `clusters_prompt.parquet`,
   `clusters_prompt_skeleton.parquet`, `clusters_prompt_prefix8.parquet` plus
   `summary.json`.

2. **Prepare decontam input** — `dedup/prepare-decontam-input.py` loads the
   chosen cluster table (default `prompt_prefix8`), filters source shards to
   the representative IDs only, and writes a single
   `decontam-input.parquet` in `(id, conversations=[{from:"human", value:...}])`
   shape. This is the only on-disk text copy the pipeline produces; it never
   touches image bytes.

3. **Decontaminate** — `decontaminate.py` (patched fork of
   `04-decontamination/decontamination.py`) consumes the prepare output via
   its new parquet-direct adapter and writes per-benchmark
   `contamination_report.json` files plus a combined `summary.json`. The fork
   adds:
   - `--input-format sharegpt-vision` reader that streams via
     `pyarrow.ParquetFile.iter_batches` to avoid the multi-chunk nested-column
     bug on large parquets
   - `--exclude-benchmarks` substring filter (case-insensitive)
   - in-memory `.map()` for the standardised path (no HF cache copy)
   - no filtered DatasetDict written; final output is `summary.json` keyed by
     `conversation_id` plus per-benchmark JSON reports

## Folder map

```
decontaminate.py                       # patched fork; new parquet-direct adapter
dedup/
  README.md                            # authoring guide for compute-*.py scripts
  _hash_utils.py                       # shared regex + hash helpers
  compute-<dataset>.py × 27            # per-dataset extractors
  aggregate-clusters.py                # polars merge over per-shard outputs
  prepare-decontam-input.py            # rep extraction
  submit-dedup-array.sbatch            # parameterised SLURM array
  submit-aggregate.sbatch              # polars aggregate as a sbatch
  submit-prepare-decontam.sbatch       # prepare-decontam-input as a sbatch
  submit-decontam-pipeline.sbatch      # prepare + decontam in one job
  submit-decontam-innovator.sbatch     # Innovator-specific (10h budget, full corpus)
  launch-all.sh                        # submit dedup arrays for all registered datasets
  launch-decontam-all.sh               # submit full decontam pipeline for all
smoke/
  smoke-innovator.{py,sbatch}          # 1-shard parquet stripping smoke
  smoke-decontam.sbatch                # 8-gram / 0.5 smoke
  smoke-decontam-13gram07.sbatch       # 13-gram / 0.7 smoke
  convert_innovator_smoke.py           # ad-hoc smoke conversion to standardised
```

## Dataset coverage

Per-dataset adapters are registered in `dedup/launch-all.sh` and
`dedup/launch-decontam-all.sh` (`DS_SCRIPT`, `DS_GLOB`, `DS_FOLDER`).

### parquet, ShareGPT-style `(id, conversations=[{from,value}], …)`
- Innovator-VL-Instruct-46M

### parquet, text-pair (`question`, `answer`, etc.)
- ChartVerse-SFT-1.8M, VDR_Cooking_Recipes, path-vqa, pixmo-ask-model-anything,
  pixmo-cap-qa, pixmo-point-explanations, Common-O, MathNet, MMFineReason,
  BigEarthNet (text), EO-Data1.5M

### parquet, conversations + metadata
- RadImageNet-VQA, Omnimodal-Agent-SFT-2K

### parquet, multi-QA per row
- Molmo2-MultiImageQA

### parquet, JSON-encoded messages string
- nemotron archive/, nemotron swissai/

### JSONL with `{role, content}`
- TCM-Instruction-Tuning-ShizhenGPT

### JSON list with `{id, conversations[{from,value}]}`
- BigData-KSU, CulturalGround, PersonaVLM, PangeaInstruct (master JSON)

### JSONL with `{id, messages:[{role, content:[...]}]}`
- nemotron hf___nvidia___

### JSON list with `(problem, solution / answer)`
- DRIM-VisualReasonHard, OneThinker, SPIQA

### HF arrow with `(id, image, conversations[{from,value}])`
- LLaVA-OneVision-1.5-Instruct-Data (161 sub-datasets), llava_cot_100k

### CSV metadata
- RSRCC (no actual prompt text — included for completeness; effectively a no-op)

## Adding a new dataset

1. Inspect a sample shard to confirm schema and identify the `id` and prompt
   fields.
2. Create `dedup/compute-<name>.py` that exposes:
   ```python
   def iter_rows(src):
       # yield (id_str, prompt_text, response_text) per row in `src`
       ...
   ```
   For minimal boilerplate, import from `_hash_utils` and reuse
   `first_human_assistant_fromvalue` / `first_human_assistant_rolecontent`.
3. Register the dataset in both `dedup/launch-all.sh` (for dedup) and
   `dedup/launch-decontam-all.sh` (for decontam) with its glob, output folder
   name, and compute script.
4. `bash dedup/launch-all.sh <name>` → dedup array
5. `bash dedup/launch-decontam-all.sh <name>` → prepare + decontam (chained)

If the dataset's IDs do not embed shard names, the prepare step automatically
falls back to scanning all source shards and filtering rows by ID-set
membership; nothing else needs to change.

If shard basenames collide across nested subdirectories (e.g.
`country/train-XXX.parquet` for many countries), set `SRC_ROOT` when invoking
`submit-dedup-array.sbatch` so the relative path under `SRC_ROOT` is flattened
into the output filename. This is what LLaVA-OneVision needs because its 161
subfolders share `0.0.0/<hash>/<file>.arrow` filenames.

## Settings used in production

- `--ngram_length 13`
- `--diff_threshold 0.7`
- `--exclude-benchmarks polyglotoxicity,toxicity,polygloto` (case-insensitive
  substring filter; drops both `swiss-ai__polyglotoxicityprompts__*` and
  `ToxicityPrompts__PolygloToxicityPrompts__*`)
- tokenizer `alehc/swissai-tokenizer`
- shared benchmark prompts:
  `/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts`
- shared benchmark n-gram cache:
  `/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache`
  (keyed by `(benchmark, tokenizer, ngram_length)`; populating 13-gram entries
  takes ~30–200 s per benchmark first time, then loads from cache)

## Known false-positive patterns

13-gram / 0.7 is conservative for English-on-English but still produces these
recurring FP families that manual inspection caught:

- **Short non-Latin prompts** — Sinhala / Amharic / Igbo / Marathi benchmark
  prompts share long stretches of byte-fallback tokens with any other
  short non-Latin training prompt, regardless of meaning. Pattern seen
  heavily in CulturalGround.
- **Same template, different numbers** — math_qa / hendrycks_math / MATH-500
  problems where only the constants differ. The matcher treats them as TPs
  because token overlap is high; downstream classifier can split this off
  with a number-presence check (`<dataset>_template_only_ids.txt`).
- **Matrix-literal collisions** — Code-Feedback training rows that contain
  small integer matrices like `[[1,2,3],[4,5,6],[7,8,9]]` match unrelated
  benchmark questions that operate on the same canonical matrix
  (determinant / flatten / kernel).
- **Numeric-string overlap** — long factorial expansions or large numerical
  sequences in training prompts share token runs with benchmarks asking about
  the same digits in a different context (see MathNet `003q`:
  29! expansion matched against `What is the ones digit of 9!?` across ~30
  language variants).

## Outputs per dataset

Layout under `/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/<dataset>/`:

```
shards/                              # per-source-shard hash parquets (dedup output)
clusters_prompt.parquet              # cluster table (verbatim hash)
clusters_prompt_skeleton.parquet     # cluster table (number/bbox normalised)
clusters_prompt_prefix8.parquet      # cluster table (first 8 tokens)
summary.json                         # dedup summary (counts, dedup factors, top clusters)
decontam-input.parquet               # prepare output: (id, conversations) for reps only
decontam-reports-13gram07/
  <benchmark>__contamination_report.json   # one per benchmark, {conv_id: eval_idx}
  summary.json                              # combined; total/per-bench counts, contaminated_ids
```

Filtered shareable ID lists (verbatim + near-verbatim true positives) can be
produced from the reports with the bucket-filter logic in
`filter_buckets_innovator.py`-style scripts. Outputs go to
`/iopsstor/scratch/cscs/schlag/<dataset>_contaminated_ids.txt`
(world-readable).
