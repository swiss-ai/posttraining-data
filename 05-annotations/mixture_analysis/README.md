# SFT mixture annotation and quantification

This package implements the deterministic and optional semantic stages of the
SWE/Terminal/Tool Use measurement pipeline. It reads source records without
modifying them and writes compact sidecar annotations keyed by stable record ID.

## Current scope

Pipeline version `0.2.0` provides:

- validation of the versioned taxonomy YAML;
- streaming JSON, JSONL, and Parquet input;
- support for both OpenAI-style messages and native Apertus 2 `system`/`items`
  conversations;
- optional semantic judges for taxonomy-declared unresolved dimensions, using either
  the CSCS OpenAI-compatible endpoint or a locally loaded Hugging Face model;
- conservative deterministic extraction of structured task labels, programming
  languages, tools, terminal commands, record structure, and selected SWE
  properties;
- sharded Parquet or JSONL sidecar output with checkpoints and safe resume;
- record-level summary statistics and optional token statistics when token counts
  are present on every input record, including per-source-dataset contributions;
- complete run identity containing source paths, file metadata, taxonomy hash,
  taxonomy version, classifier version, Git commit, and exact classifier-code
  hash.

The default compact taxonomy is `swe_terminal_tool_use_v0.2.0.yaml`. It reports
exactly five axes: expert domain, programming language, and one five-label
category axis each for SWE, Terminal, and Tool Use. Programming language uses
deterministic extraction; the other four axes may be resolved by the semantic
judge. The earlier detailed `v0.1.0` taxonomy remains available by passing
`--taxonomy` explicitly, so existing experiments are not overwritten.

Missing deterministic evidence remains `unknown`; it is not treated as
`none_of_the_above`. Licence eligibility and final mixture selection remain
external inputs.

## Usage

Use Python 3.10 or newer in the project environment. Parquet input/output requires
the repository's `pyarrow` dependency.

```bash
python 05-annotations/quantify_sft_mixture.py \
  /path/to/parquet/archive \
  --output /path/to/annotations \
  --output-format parquet
```

For a small development run using the bundled previews:

```bash
python 05-annotations/quantify_sft_mixture.py \
  dataset_website/reports/examples/b-mc2__sql-create-context-d7104aa6.json \
  dataset_website/reports/examples/Lite-Coder__LiteCoder-Terminal-SFT-c36db113.json \
  --output /tmp/sft-mixture-annotations \
  --output-format jsonl \
  --limit 2
```

The output directory contains:

```text
run.json                 immutable run identity
part-*.parquet|jsonl     compact per-record sidecars
summary.json             aggregate record/token statistics
_SUCCESS.json            completion marker
```

Rerunning the identical command returns the completed result. Changed inputs,
taxonomy content, or processing options require a new output directory.
Objects without a supported SFT content field are rejected, which prevents a
manifest such as `index.json` from silently becoming a training record.

## Reproducible development subsets

Use [`prepare_mixture_subset.py`](../prepare_mixture_subset.py) to materialize a
small JSONL input from standardized Parquet shards. It ranks source files and
records by SHA-256 using an explicit seed, caps records per file to improve
source diversity, and can require declared per-record licenses. The adjacent
manifest records the source revision and file hashes, selected record IDs,
selection parameters, output hash, and subset-builder code hash.

The 80-record CommitPackFT Clariden debug input was created with:

```bash
python 05-annotations/prepare_mixture_subset.py \
  /capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft/code_sft/bigcode/commitpackft \
  --output data/debug-subsets/commitpackft-debug-80-seed-20261029.jsonl \
  --records 80 \
  --records-per-file 4 \
  --seed apertus-mixture-debug-2026-10-29 \
  --allow-license mit \
  --allow-license apache-2.0 \
  --allow-license bsd-2-clause \
  --allow-license bsd-3-clause \
  --allow-license isc
```

Subset preparation requires the repository-pinned PyArrow dependency. Generated
subsets belong under the ignored `data/` directory; the builder refuses to
overwrite an existing subset or manifest.

## Optional semantic judge

### Local Hugging Face model

The local provider downloads or reads an HF model, loads it through
`AutoModelForCausalLM`, batches unresolved records on the GPU, and performs
greedy generation. It does not need a CSCS inference key. Public HF models need
no HF token; gated or private models may require `HF_TOKEN` and acceptance of
their licence terms.

Run this on an allocated compute node, not a login node. The environment must
provide a GPU-compatible PyTorch build plus the repository's `transformers`
dependency. The default `cuda` mode targets one GPU. `auto` can distribute a
model across GPUs but additionally requires `accelerate`.

```bash
python 05-annotations/quantify_sft_mixture.py \
  /path/to/parquet/archive \
  --output /path/to/local-judge-annotations \
  --output-format parquet \
  --judge-provider huggingface \
  --judge-model YOUR_HF_MODEL_ID \
  --judge-hf-revision YOUR_PINNED_COMMIT \
  --judge-hf-device cuda \
  --judge-hf-dtype bfloat16 \
  --judge-batch-size 4 \
  --limit 100
```

Use an instruction-tuned causal language model that can reliably emit JSON.
Model architecture, licence, memory fit, chat-template behavior, and judgement
quality still need validation; “Hugging Face model” does not mean every model on
the Hub is suitable. Remote model code is disabled unless
`--judge-hf-trust-remote-code` is explicitly supplied. For a pre-downloaded
model or an offline compute node, pass `--judge-hf-local-files-only` and point
`--judge-model` at its local directory.

The immutable run identity records the requested and resolved HF revisions,
PyTorch and Transformers versions, dtype, device strategy, prompt hash, token
limits, thinking mode, and generation settings. Start with `--limit` and adjust
batch size based on measured GPU memory and throughput.

### Clariden Qwen3-14B experiment

The first pinned local-judge preset is
[`model_presets/qwen3_14b_clariden.yaml`](../model_presets/qwen3_14b_clariden.yaml).
It uses `Qwen/Qwen3-14B` at commit
`40c069824f4251a91eefaf281ebe4c544efd3e18` in BF16. The roughly 29.6 GB
checkpoint fits on one GH200, so the Clariden job runs four independent model
replicas per node instead of splitting this relatively small model across all
GPUs. Each Slurm rank gets one GPU and a disjoint deterministic input shard.
Qwen thinking mode is disabled because this classifier requires compact JSON and
uses deterministic greedy decoding.

The execution environment is pinned to
`pytorch/v2.9.1:v2@clariden%gh200`, image ID
`c05f143d5fbf092714650697c821ee9fcf3d00c3580d56dabeacb2370a824b0f`.
The batch shell deliberately stays in Clariden's host environment. The preload
and inference `srun` steps each activate that exact uenv image and its `default`
view explicitly. This keeps Slurm's control process on the host OpenSSL while
the compute processes use the pinned PyTorch stack. The label, image ID, and
SquashFS digest are written to each shard's `run.json` as runtime metadata.

The bundled Slurm job defaults to one node (four replicas) and accepts an
`sbatch --nodes=N` override up to four nodes. For the initial smoke test, use
JSON or JSONL so the stock PyTorch uenv needs no Parquet extension:

```bash
cd /path/to/posttraining-data
mkdir -p slurm_logs

sbatch --account=YOUR_PROJECT \
  --export=ALL,MIXTURE_INPUT=/absolute/path/input.jsonl,MIXTURE_OUTPUT=/absolute/path/qwen3-14b-smoke,MIXTURE_LIMIT_PER_SHARD=25 \
  05-annotations/slurm/clariden_qwen3_14b_judge.sbatch
```

This classifies at most 100 records in total: 25 records on each of four ranks.
Remove `MIXTURE_LIMIT_PER_SHARD` after inspecting the smoke-test sidecars. The
model is downloaded once to the shared `HF_HOME` cache (default:
`$SCRATCH/huggingface`) before ranks start, then loaded offline by every rank.
No Hugging Face token is needed for this public model. A private or gated model
would require `HF_TOKEN`.

For a minimal test in Clariden's `debug` partition, override the same job's
resource directives at submission time. This profile uses one node, one GPU,
one model replica, at most four records, and 45 of the partition's 90 permitted
node-minutes:

```bash
cd /path/to/posttraining-data
mkdir -p slurm_logs

export DEBUG_INPUT=/absolute/path/input.jsonl
export DEBUG_OUTPUT=/absolute/path/qwen3-14b-debug-001

sbatch --account=YOUR_PROJECT \
  --partition=debug \
  --time=00:45:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-task=1 \
  --cpus-per-task=72 \
  --export=ALL,MIXTURE_INPUT="$DEBUG_INPUT",MIXTURE_OUTPUT="$DEBUG_OUTPUT",MIXTURE_LIMIT_PER_SHARD=4,JUDGE_BATCH_SIZE=1 \
  05-annotations/slurm/clariden_qwen3_14b_judge.sbatch
```

Use a new output directory for the debug run. If an earlier job uses the same
output, cancel it or let it finish rather than allowing both jobs to write there.
The model checkpoint is roughly 29.6 GB; a cold Hugging Face cache makes the
first debug run slower because the checkpoint must be downloaded before model
loading. Clariden permits at most one running debug job and two submitted debug
jobs per user, so this profile is for a smoke test rather than production data.

Debug job `3788684` used the earlier job-level uenv view and failed before
classification because the view's OpenSSL shadowed the host library needed by
Slurm's site-policy plugin. The job produced no sidecar or success marker. The
per-step activation described above is the permanent fix; do not restore
job-level `#SBATCH --view=default` to this launcher.

Use `--nodes=4` on the `sbatch` command only after the one-node test; that starts
16 replicas and creates 16 disjoint output directories. This initial modulo
sharding makes every rank scan the input stream and is suitable for validation,
but large production runs should consume physically pre-sharded input to avoid
repeating input I/O. Parquet input/output additionally requires a `pyarrow`
overlay on the uenv.

Clariden references: [cluster](https://docs.cscs.ch/clusters/clariden/),
[one rank per GH200 GPU](https://docs.cscs.ch/running/slurm/#one-rank-per-gpu),
and [PyTorch uenv](https://docs.cscs.ch/software/ml/pytorch/).

### Clariden Phi-4 compact-taxonomy comparison

The second pinned candidate is `microsoft/phi-4` at commit
`2db69c1c3e91a05d2c64a3185acfbaf36f744e25`, using the MIT-licensed 14B
checkpoint and compact taxonomy `v0.2.0`. Its preset is
[`model_presets/phi4_14b_clariden.yaml`](../model_presets/phi4_14b_clariden.yaml).
The job uses four independent BF16 replicas on one Clariden node and records the
same immutable uenv identity as the Qwen experiment.

For a controlled A/B run on the fixed 100-record Glaive subset:

```bash
sbatch --account=YOUR_PROJECT \
  --partition=debug \
  --time=00:30:00 \
  --export=ALL,MIXTURE_INPUT="$PWD/data/debug-subsets/glaive-code-assistant-v2-debug-100-seed-20261029.jsonl",MIXTURE_OUTPUT=/absolute/new/output/phi4-glaive-100-v020 \
  05-annotations/slurm/clariden_phi4_14b_judge.sbatch
```

Do not compare these results directly to the old Qwen `v0.1.0` output as a
model-only experiment: both model and taxonomy changed. A strict model A/B must
rerun Qwen on the same input with taxonomy `v0.2.0`; the human-labelled gold set
is still required to determine which model is actually more accurate.

### Beverin MI300A experiment

Beverin uses AMD accelerators, so the Clariden CUDA uenv job cannot be submitted
there unchanged. The Beverin preset
[`model_presets/qwen3_14b_beverin.yaml`](../model_presets/qwen3_14b_beverin.yaml)
uses the same pinned Qwen3-14B judge but targets the `mi300` partition through a
pinned ROCm 7.2 SGLang container. One node has four MI300A APUs; the job launches
four independent model replicas, one per APU, and partitions records across the
four ranks. PyTorch on ROCm intentionally continues to use the `torch.cuda` API,
so `--judge-hf-device cuda` is correct on this platform.

The repository currently lives on MLP-only Iopsstor. Stage the code and smoke
test input to a filesystem visible on Beverin, such as Capstor, before logging
in to Beverin:

```bash
rsync -a --exclude data/ --exclude venv/ \
  /iopsstor/scratch/cscs/$USER/posttraining-data/ \
  /capstor/scratch/cscs/$USER/posttraining-data/
```

On Beverin, prepare the pinned ROCm container once:

```bash
cd /capstor/scratch/cscs/$USER/posttraining-data
bash 05-annotations/slurm/prepare_beverin_rocm_container.sh
```

The helper starts a one-node MI300 allocation because Podman requires temporary
overlay storage on node-local `/dev/shm`; Beverin home storage does not provide
the required extended attributes. If the desired Slurm account is not your
default association, prefix the command with
`BEVERIN_ACCOUNT=YOUR_BEVERIN_PROJECT`. Only the completed SquashFS image is
persisted to Capstor.

Then submit the one-node smoke test:

```bash
export MIXTURE_INPUT=/capstor/scratch/cscs/$USER/data/sample.jsonl
export MIXTURE_OUTPUT=/capstor/scratch/cscs/$USER/apertus-mixture/qwen3-14b-beverin-smoke
export MIXTURE_LIMIT_PER_SHARD=25

mkdir -p slurm_logs "$MIXTURE_OUTPUT"
sbatch --account=YOUR_BEVERIN_PROJECT --nodes=1 \
  05-annotations/slurm/beverin_qwen3_14b_judge.sbatch
```

This processes at most 100 records across four replicas. The script permits an
override up to `--nodes=4`, which launches 16 replicas. Input, output, repository,
container, and model cache paths must be under the directories mounted by
[`beverin_sglang_rocm.toml`](../slurm/beverin_sglang_rocm.toml). As with the
Clariden smoke test, use JSON/JSONL unless the selected container has been
extended with the repository's pinned `pyarrow` dependency.

Beverin references: [cluster and partitions](https://docs.cscs.ch/clusters/beverin/),
[MI300A hardware](https://docs.cscs.ch/alps/hardware/#amd-mi300a-gpu-nodes), and
[ROCm inference containers](https://docs.cscs.ch/tutorials/ml/vllm-sglang-amdgpus/).

### CSCS managed endpoint

Deterministic processing does not require an API key. Live semantic judging does:
CSCS requires a bearer key in every inference request. Access is granted through
an inference resource attached to an active CSCS project. Once that resource
exists, a project member can create their own key in the Inference API UI.

The pipeline reads the key from `CSCS_INFERENCE_API_KEY`; it never accepts the
secret as a command-line option or writes it to run metadata. Keep it in a
password manager or a temporary shell/secret environment, not in the repository.

```bash
export CSCS_INFERENCE_API_KEY="..."

python 05-annotations/quantify_sft_mixture.py \
  /path/to/parquet/archive \
  --output /path/to/judged-annotations \
  --output-format parquet \
  --judge-provider cscs \
  --judge-model YOUR_EXPLICIT_MODEL_ID \
  --judge-concurrency 8

unset CSCS_INFERENCE_API_KEY
```

The model ID is deliberately explicit because the models available to a key can
change. Use the CSCS Inference API UI or authenticated `/v1/models` endpoint to
choose it. Start with `--limit` to validate prompts and estimate project-credit
usage before a large run.

For either provider, only `unknown` dimensions declared by the selected
taxonomy's `semantic_judge.dimensions` are replaced. Successful responses are
taxonomy- and applicability-validated and carry the provider, model
ID/revision, prompt version, response hash, and token usage. Invalid or
exhausted requests receive the reportable `error` status. The judge prompt
treats record contents as untrusted data.

CSCS documentation: <https://docs.cscs.ch/services/inference/api/>

## Annotation strategy

The classifier only uses evidence that can be recovered reproducibly from the
record:

- explicit source metadata and labels;
- structured tool definitions, calls, and results;
- explicit command fields and shell-call arguments;
- code-fence language tags, file extensions, and strong SQL syntax;
- repository, patch, diff, file, reasoning, and message structure.

Every dimension stores its status, labels, method, confidence, and evidence in
`classifications_json`; frequently aggregated labels are also written as native
list columns.

## Next stages

1. Review the compact experimental taxonomy `0.2.0`.
2. Build a representative human-labelled gold set.
3. Measure the semantic judge against the gold set and revise its prompt.
4. Select and pin the production judge model and operating parameters.
5. Run over the licence-eligible selected mixture.
6. Join native-token and loss-token counts and publish the final report.
