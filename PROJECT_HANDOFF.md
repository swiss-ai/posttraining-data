# Apertus post-training data and SFT mixture quantification handoff

Last updated: 2026-10-29

## Purpose of this document

This document summarizes the project context, the main decisions from the work so far,
the implementation that has been added locally, the CSCS execution setup, and the
remaining decisions. It is intended to let a new collaborator or a future session
continue without reconstructing the full conversation.

For the detailed 95-dataset audit, also read
[`dataset_website/CONTEXT_HANDOFF.md`](dataset_website/CONTEXT_HANDOFF.md). For usage
of the mixture-classification implementation, read
[`05-annotations/mixture_analysis/README.md`](05-annotations/mixture_analysis/README.md).

## 1. Project introduction

Apertus is the Swiss AI open large-language-model project. This repository,
`posttraining-data`, prepares datasets for post-training rather than pretraining. Its
main pipeline downloads source datasets, converts them to common structures, filters
them, annotates them, aggregates the retained data, and evaluates it.

The repository describes eight main stages:

1. `01-hf-download`: download and pin Hugging Face datasets.
2. `02-standardisation`: normalize heterogeneous datasets.
3. `03-license-based-filtering`: apply approved license filters.
4. `04-decontamination`: remove overlap with evaluation data.
5. `05-annotations`: add classifications and other annotations.
6. `06-field-based-filtering`: filter using record fields and annotations.
7. `07-dataset-aggregation`: assemble the final training mixture.
8. `08-judge-evaluation`: assess data with LLM judges.

The current assignment belongs mainly in stage 5, with its final reports informing
stage 7. It measures what capabilities and task types are actually present in the SFT
records. It does not itself decide whether a dataset is legally eligible or what its
final sampling weight should be.

## 2. Assigned problem

The issue asks for a reproducible way to quantify the data mixture. The requested axes
include task domains such as bug fixing or test creation, programming languages such as
Python or Rust, and tools such as Git or grep. It must cover:

- Gyms and SFT data;
- SWE, Terminal, and Tool Use experts;
- categories that may differ between the three experts; and
- current mixture statistics kept in the issue or another agreed report location.

The responsibility discussed here is **SFT SWE/Terminal**. Gyms SWE/Terminal and
Gyms/SFT Tool Use have other named owners, although the taxonomy and pipeline support
overlapping Tool Use labels so that shared records do not lose information.

The intended result is not merely a list of datasets. It is a versioned program that
classifies records, records how each label was obtained, and aggregates the results
with explicit denominators.

## 3. High-level mental model

The fundamental unit is a **training record**, not a whole dataset. A dataset can
contain millions of heterogeneous records, so assigning one label to its name would
hide the internal mixture.

```text
licensed and selected source records
             |
             v
  deterministic record inspection
             |
             +---- labels with strong structural evidence
             |
             +---- unresolved semantic dimensions
                              |
                              v
                   validated LLM classifier
                              |
                              v
                    per-record sidecars
                              |
                              v
             aggregate counts and percentages
```

Dataset-level statements are then aggregates of the record-level annotations. For
example: 35% of records are SWE, 18% are bug fixing, 27% contain Python, and 12% invoke
a shell tool. Because dimensions are multi-label, such percentages can overlap and do
not always sum to 100%.

The pipeline stores compact **sidecar annotations** rather than copying every original
record. A sidecar contains a stable record identifier, labels, evidence, versions, and
provenance. The source data remains unchanged.

For large results, Parquet is the preferred format. Parquet is a typed, compressed,
column-oriented file format: it is much more efficient than JSON for scanning selected
columns and calculating statistics over millions of rows.

## 4. Taxonomy decisions

The first draft taxonomy is
[`05-annotations/taxonomies/swe_terminal_tool_use_v0.1.0.yaml`](05-annotations/taxonomies/swe_terminal_tool_use_v0.1.0.yaml).
It is YAML because humans can review it while programs can parse and validate it. It is
not only explanatory documentation: the pipeline loads it, validates emitted labels
against it, includes its version and SHA-256 hash in run identity, and uses its label
definitions in the semantic-judge prompt.

The top-level dimensions are:

- `expert_domains`: which specialist capabilities the record trains: SWE, Terminal,
  Tool Use, or none of these.
- `task_domains`: what the user is trying to accomplish, such as bug fixing, feature
  implementation, test creation, text-to-SQL, code generation, or technical Q&A.
- `programming_languages`: languages materially involved in the task or answer,
  including shell and SQL.
- `work_context`: where the work occurs, such as a standalone snippet, one file, a
  repository, a terminal environment, or a database schema.
- `interaction_styles`: the shape of the example, such as a direct answer, multi-turn
  conversation, reasoning trace, agent trajectory, or terminal trajectory.
- `swe`: SWE-only details such as change scope and affected artifact types.
- `terminal`: terminal-only activities and observed executable names.
- `tool_use`: model-visible tool types, available and invoked tool names, and trajectory
  properties.

These dimensions answer different questions. For example, a repository bug fix can
have `expert_domains=[swe, terminal, tool_use]`, `task_domains=[bug_fixing]`,
`work_context=[repository, terminal_environment]`, and
`interaction_styles=[agent_trajectory, terminal_trajectory]`.

An "expert" is a capability stream or specialization, not a dataset. A record may
belong to several expert domains. If a record is both SWE and Tool Use, the annotation
contains both expert labels and, where evidence exists, both the SWE-specific and
Tool-Use-specific subdimensions.

The taxonomy is currently a draft at version `0.1.0`. Taxonomy changes must use review
and explicit versioning; silently changing labels after a run would make statistics
non-comparable.

## 5. How classification is divided

Three complementary methods are expected.

### Deterministic rules

Use deterministic extraction when a fact can be recovered reliably from structure or
syntax. Examples include:

- a structured `tool_calls` object proves an invoked model-visible tool;
- a tool declaration provides the available tool name;
- a fenced block marked `python` or a `.py` filename supports the Python label;
- a unified diff supports patch/change structure;
- an explicit command field can yield `git`, `grep`, or `pytest`;
- multiple user/assistant turns establish a multi-turn interaction;
- source metadata can provide a documented task label or language.

Deterministic rules should be conservative. A prose mention of `grep` does not prove
that grep was exposed as a model-visible tool. If the model calls a general `bash` tool
and executes grep inside it, `bash` is the tool and `grep` is a terminal command.

### LLM semantic classification

Some intent labels require reading the task. Distinguishing a requested bug fix from a
new feature, or an investigation from an implementation, usually needs semantic
classification. An instruction-tuned LLM can resolve these dimensions after the
deterministic pass.

The LLM judge is not accepted merely because it can emit labels. It must use a pinned
model and prompt, emit taxonomy-valid JSON, and be evaluated against a human-labelled
gold set. Unsupported cases remain `unknown`; `unknown` is not the same as
`none_of_the_above`.

### Human review

Humans should label a small representative and difficult subset, not tens of millions
of records manually. This gold set is used to refine the taxonomy, measure judge
quality, identify systematic errors, and approve production settings. Human overrides
should remain explicit and auditable.

## 6. What “version the pipeline” means

Versioning is broader than storing a taxonomy version string. Reproducibility requires
versioning or pinning:

- taxonomy definitions;
- classifier code and deterministic rules;
- prompt text and output schema;
- model ID and exact model revision;
- generation parameters;
- source-dataset revisions;
- processing configuration;
- output schema and report logic; and
- ideally, the repository commit that ties these together.

The implementation also hashes the executable classifier source. This matters for a
dirty worktree: a Git commit alone would not identify local edits.

## 7. Dataset discovery and license audit completed so far

The local `dataset_website` work contains an inventory and surface audit of exactly 95
coding-related datasets:

| Inventory category | Datasets | Interpretation |
| --- | ---: | --- |
| `code_sft` | 42 | Non-agentic or ordinary supervised examples |
| `code_agentic` | 38 | Agent, terminal, or tool-oriented collections |
| `benchmarks` | 15 | Evaluation-oriented tasks, normally not training data |

The canonical working audit is
[`dataset_website/dataset_metadata_audit.csv`](dataset_website/dataset_metadata_audit.csv).
It covers license, conversation shape, chat-template evidence, system prompts,
harnesses, tool usage, agent traces, thinking representation, and synthetic provenance.
The local previews are examples, not full-corpus scans.

Current audit status is preliminary:

- all 95 rows are still marked `in_progress`;
- 77 have a declared license in the surface audit;
- 18 have missing, ambiguous, conflicting, or otherwise unclear license status;
- an initial conservative proposal excluded 21 datasets and left 74 candidates;
- that 74-dataset result has not been approved as a legal determination or materialized
  as a final candidate-mix CSV; and
- repository-derived and mixed datasets may still require row/subset-level upstream
  license filtering even when their dataset-level metadata looks acceptable.

The rough proposed policy was to exclude non-commercial, ShareAlike, copyleft,
undeclared, and materially conflicting cases from the first mix. CC-BY and ODC-By were
not automatically excluded, but they carry attribution requirements. This is a
metadata-based screening proposal, not legal advice or a finalized policy.

### Work that can proceed before license policy is final

License eligibility and mixture measurement should remain separate inputs. While the
license decision is being completed, it is safe and useful to:

- review and revise the taxonomy;
- implement deterministic parsers against local previews or approved development data;
- define the annotation and report schemas;
- create a small gold-labelled sample without publishing it as a training mix;
- validate the semantic judge;
- test streaming, sharding, resume, and run identity; and
- prepare cluster execution scripts.

A final full-corpus run and published mixture statistics must specify the approved
eligible dataset list and exact source revisions.

## 8. Relevant branch survey

The checkout is currently on `main` at commit `574e7c5`. At the time of this handoff:

- `origin/feat/apertus2-data-processing` contains the standalone Apertus 2 mapping,
  tokenization, inspection, reporting, and cluster-processing framework. It has work
  not present in `main` and has also diverged from newer `main` changes.
- `origin/feat/apertus2-tool-use-mapper` builds on that processing branch and adds the
  `apertus_tool_use.py` mapper plus tests. It also has work not present in `main`.
- `origin/feat/sft-curation` has no commits that are currently ahead of `main`; its work
  is already an ancestor of the current main history.

The tool-use mapper is not a general data-mixture classifier. It:

- reads `swiss-ai/apertus-tool-use` JSONL conversations;
- maps OpenAI-style messages and tool definitions to native Apertus 2 conversations;
- converts function arguments to compact JSON strings;
- rejects assistant steps with neither text nor tool calls;
- optionally rejects samples above a pinned-tokenizer length limit;
- preserves selected metadata; and
- writes one native sample per source conversation under a thinking convention.

That mapper solves source-to-native-format conversion for one collection. The mixture
measurement work solves record classification and aggregate reporting across many
datasets. The two can eventually be composed, but expanding only the mapper would not
satisfy the quantification issue.

## 9. What was added in the current worktree

The current changes are local and uncommitted. They should be reviewed before being
placed on a feature branch or committed.

### Taxonomy and command-line pipeline

- [`05-annotations/taxonomies/swe_terminal_tool_use_v0.1.0.yaml`](05-annotations/taxonomies/swe_terminal_tool_use_v0.1.0.yaml)
  defines the draft dimensions, labels, evidence contract, and reporting contract.
- [`05-annotations/quantify_sft_mixture.py`](05-annotations/quantify_sft_mixture.py)
  is the command-line entry point.
- [`05-annotations/prepare_mixture_subset.py`](05-annotations/prepare_mixture_subset.py)
  creates seeded, hash-ranked JSONL development subsets from Parquet shards and writes
  a source/output manifest.
- [`05-annotations/mixture_analysis/`](05-annotations/mixture_analysis/) contains
  taxonomy loading, input/output, deterministic annotation, optional semantic judges,
  pipeline orchestration, and summaries.
- [`05-annotations/tests/test_mixture_analysis.py`](05-annotations/tests/test_mixture_analysis.py)
  tests the main contracts and representative inputs.

The implementation currently supports:

- streaming JSON, JSONL, and Parquet inputs;
- OpenAI-style messages and native Apertus 2 `system`/`items` conversations;
- conservative deterministic extraction;
- optional CSCS OpenAI-compatible or local Hugging Face semantic judging;
- deterministic modulo input sharding across independent ranks;
- sharded JSONL or Parquet output;
- checkpoints and safe resume;
- stable record identifiers and input hashes;
- record and optional token statistics by dataset and taxonomy dimension; and
- immutable run identity including taxonomy hash, code hash, Git commit, sources,
  judge configuration, and model revision.

The output directory contains:

```text
run.json                 run identity and reproducibility metadata
part-*.parquet|jsonl     per-record sidecar annotations
summary.json             aggregate statistics
_SUCCESS.json            completion marker
```

Rerunning an identical completed job returns its result. Reusing an output directory
with different inputs, taxonomy content, or relevant options is rejected.

### Optional semantic judge

The CSCS provider uses the OpenAI-compatible inference API. Deterministic processing
does not need an API key, but live CSCS inference does. The secret is read only from
`CSCS_INFERENCE_API_KEY`; it is not accepted on the command line or written into run
metadata.

Because no CSCS inference key was available, a local Hugging Face provider was added.
It loads an instruction model directly on allocated GPUs, applies the model chat
template, batches unresolved records, performs greedy generation, validates the JSON,
and records model/prompt provenance. Public models need no Hugging Face token; gated or
private models can require `HF_TOKEN` and prior license acceptance.

The first experiment uses:

```text
model:    Qwen/Qwen3-14B
revision: 40c069824f4251a91eefaf281ebe4c544efd3e18
dtype:    bfloat16
license:  Apache-2.0
thinking: disabled for compact deterministic JSON output
```

The BF16 checkpoint is roughly 29.6 GB, so it fits as one model replica on one GH200 or
MI300A. The production judge is not yet approved: this model and prompt still need
measurement against the human-labelled gold set.

## 10. Clariden execution design

The Clariden job uses the PyTorch `v2.9.1` uenv and one Qwen replica per GH200:

- one to four nodes;
- four Slurm ranks per node;
- one GPU and one independent model replica per rank; and
- a disjoint deterministic record shard per rank.

Relevant files:

- [`05-annotations/model_presets/qwen3_14b_clariden.yaml`](05-annotations/model_presets/qwen3_14b_clariden.yaml)
- [`05-annotations/slurm/clariden_qwen3_14b_judge.sbatch`](05-annotations/slurm/clariden_qwen3_14b_judge.sbatch)
- [`05-annotations/slurm/run_clariden_hf_judge_rank.sh`](05-annotations/slurm/run_clariden_hf_judge_rank.sh)

One-node smoke-test pattern:

```bash
cd /path/to/posttraining-data
mkdir -p slurm_logs

sbatch --account=YOUR_PROJECT \
  --export=ALL,MIXTURE_INPUT=/absolute/path/input.jsonl,MIXTURE_OUTPUT=/absolute/path/qwen3-14b-smoke,MIXTURE_LIMIT_PER_SHARD=25 \
  05-annotations/slurm/clariden_qwen3_14b_judge.sbatch
```

With four ranks and a per-shard limit of 25, this processes at most 100 records. Start
with one node, inspect the annotations, and only then consider up to four nodes. The
current modulo sharding is suitable for experiments but makes every rank scan the input;
large production runs should use physically pre-sharded input to avoid repeated I/O.

Clariden's `debug` partition can run a smaller version of the same job by overriding
the submission directives: one node, one task, one GPU, `MIXTURE_LIMIT_PER_SHARD=4`,
and a 45-minute wall time. This consumes 45 of the permitted 90 node-minutes and avoids
starting four Qwen replicas for a tiny functional test. Always use a new output path so
the debug job cannot race an earlier normal-partition job. The exact command is kept in
the mixture-analysis README.

The first debug run, Slurm job `3788684`, failed before classification. A job-level
PyTorch uenv view changed `LD_LIBRARY_PATH`, causing the local `srun` process to load the
uenv OpenSSL instead of the host version required by Clariden's Slurm site-policy
plugin. The permanent design now leaves the batch shell on host libraries and activates
the immutable uenv separately for each compute step. The pinned environment is
`pytorch/v2.9.1:v2@clariden%gh200`, image ID
`c05f143d5fbf092714650697c821ee9fcf3d00c3580d56dabeacb2370a824b0f`, and its label,
ID, and SquashFS digest are included in every `run.json`.

After the one-record job succeeded, an 80-record CommitPackFT subset was materialized
from 20 source Parquet files using seed `apertus-mixture-debug-2026-10-29`, with at most
four records per file and declared permissive per-record licenses. Its JSONL SHA-256 is
`b78acb097087e52e640b8061915fe32af9f437c90ffe73e93470ccc57a7546dd`; the adjacent
manifest contains all source hashes and selected record identities. Clariden job
`3789167` runs this subset as four independent ranks, 20 records per GPU, batch size 2.

## 11. Beverin execution design and troubleshooting

Beverin has AMD MI300A accelerators, so the Clariden CUDA uenv cannot be reused as-is.
The Beverin setup uses:

- the `mi300` Slurm partition;
- one to four nodes;
- four MI300A APUs and four independent ranks per node;
- one model replica per APU;
- a pinned ROCm 7.2 SGLang container; and
- the PyTorch `torch.cuda` API, which remains the correct API name under ROCm.

Relevant files:

- [`05-annotations/model_presets/qwen3_14b_beverin.yaml`](05-annotations/model_presets/qwen3_14b_beverin.yaml)
- [`05-annotations/slurm/beverin_sglang_rocm.toml`](05-annotations/slurm/beverin_sglang_rocm.toml)
- [`05-annotations/slurm/prepare_beverin_rocm_container.sh`](05-annotations/slurm/prepare_beverin_rocm_container.sh)
- [`05-annotations/slurm/beverin_qwen3_14b_judge.sbatch`](05-annotations/slurm/beverin_qwen3_14b_judge.sbatch)
- [`05-annotations/slurm/run_beverin_hf_judge_rank.sh`](05-annotations/slurm/run_beverin_hf_judge_rank.sh)

The repository is on MLP Iopsstor, which is not the filesystem used by Beverin. Stage
the code and a small input to Capstor first:

```bash
rsync -a --exclude data/ --exclude venv/ \
  /iopsstor/scratch/cscs/$USER/posttraining-data/ \
  /capstor/scratch/cscs/$USER/posttraining-data/
```

On Beverin, prepare the container once:

```bash
cd /capstor/scratch/cscs/$USER/posttraining-data
BEVERIN_ACCOUNT=infra01 bash 05-annotations/slurm/prepare_beverin_rocm_container.sh
```

The first version failed with `lsetxattr ... operation not supported` because rootless
Podman attempted to use overlay storage under the Vadret home filesystem. The helper
was updated to acquire an MI300 allocation and place temporary Podman and Enroot storage
under node-local `/dev/shm`. Only the completed SquashFS image is persisted to Capstor.

Submit a one-node smoke test with:

```bash
export MIXTURE_INPUT=/capstor/scratch/cscs/$USER/data/sample.jsonl
export MIXTURE_OUTPUT=/capstor/scratch/cscs/$USER/apertus-mixture/qwen3-14b-beverin-smoke
export MIXTURE_LIMIT_PER_SHARD=25

mkdir -p slurm_logs "$MIXTURE_OUTPUT"
sbatch --account=infra01 --nodes=1 \
  05-annotations/slurm/beverin_qwen3_14b_judge.sbatch
```

### Beverin account and primary group

SSH authentication, the Unix primary group, and the Slurm charging account are related
but not identical settings.

- `id -gn "$USER"` prints the Unix primary project group. If it still prints the old
  project `a0152`, changing `$USER` is not the solution.
- Confirm membership with `id -Gn "$USER" | tr ' ' '\n'`.
- To make `infra01` the persistent primary CSCS project, select it as **Primary project**
  in the CSCS user portal, then fully log out and reconnect. This is an account-wide
  setting, not Beverin-only.
- `newgrp infra01` changes only the current shell and is not the persistent account
  setting.
- Confirm the Beverin Slurm association separately:

  ```bash
  sacctmgr show assoc \
    where user="$USER" cluster=beverin \
    format=Cluster,Account,Partition,QOS
  ```

- An explicit `sbatch --account=infra01`, `salloc --account=infra01`, or
  `srun --account=infra01` is authoritative for that command even if the Unix primary
  group still needs updating.

## 12. Statistics the pipeline can produce

The report contract requires explicit denominators and reports unknown/error states.
The intended statistics include:

- total records and, when present for every record, source tokens and training-loss
  tokens;
- contribution of each source dataset;
- counts and percentages for every expert domain;
- intersections such as SWE + Terminal or SWE + Tool Use;
- task-domain distributions such as bug fixing, feature work, test creation, terminal
  execution, SQL, and standalone code generation;
- programming-language distributions;
- work-context and interaction-style distributions;
- SWE change scopes and artifact types;
- terminal activities and command names;
- available versus invoked tool names and tool types; and
- `unknown`, `ambiguous`, `not_applicable`, and `error` rates.

Record percentages answer “how many examples?” Token percentages answer “how much
training mass?” Both are useful because a small number of long agent trajectories can
account for a large share of tokens.

## 13. End-to-end plan

The agreed sequence is:

1. Finalize the license policy and produce an approved eligible source manifest.
2. Pin dataset repositories, revisions, splits, and record identity rules.
3. Review and version taxonomy `0.1.0`.
4. Run deterministic classification on representative samples and inspect unknowns.
5. Build a stratified human-labelled gold set.
6. Evaluate the semantic judge per dimension and important subgroup; revise the prompt
   or taxonomy where errors are systematic.
7. Pin the approved model, prompt, parameters, and classifier code.
8. Run the complete eligible SFT mixture with scalable physical input sharding.
9. Join or calculate native-token and training-loss-token counts.
10. Publish the reproducible report and update the issue with the latest statistics,
    source manifest, taxonomy version, and run identity.

## 14. Contribution relative to existing work

Existing repository code already provided the general eight-stage preparation pipeline,
several annotation utilities, tool-data validation utilities, and dataset-specific
standardizers. The unmerged Apertus 2 branches add scalable native-format mapping and a
tool-use mapper.

The work summarized here contributes a different missing layer:

- a shared record-level taxonomy for SWE, Terminal, and Tool Use;
- definitions separating experts, task intent, work context, and interaction shape;
- a reproducible sidecar annotation and reporting contract;
- conservative deterministic classification across heterogeneous record formats;
- CSCS and local-HF semantic-judge backends;
- model, prompt, source, taxonomy, and code provenance;
- streaming/sharded execution and resume behavior; and
- Clariden and Beverin smoke-test architectures.

This directly supports the requested SFT data-preprocessing slides as both current work
and planned work: source selection/license filtering comes first, normalization and
structural extraction follow, semantic classification handles unresolved intent, and
the resulting annotations quantify the retained mixture before aggregation.

## 15. Current validation state

At the last local validation, all 21 unit tests in
`05-annotations/tests/test_mixture_analysis.py` passed. Shell syntax checks also passed
for the cluster scripts. This validates the software contracts exercised by the tests;
it does not validate the semantic accuracy of Qwen3-14B, the final taxonomy, the legal
eligibility list, or full-scale throughput.

The current working tree also contains uncommitted modifications to `.gitignore` and
`05-annotations/README.md`, plus the new taxonomy, package, tests, presets, and Slurm
files. Preserve and review these changes before rebasing, switching branches, or
committing. `dataset_website/` is currently ignored by Git, so its audit artifacts and
handoff are local unless intentionally moved or tracked elsewhere.

## 16. Open decisions

The following are intentionally unresolved:

1. Who approves the final license policy and the eligible dataset/row manifest?
2. Which benchmarks, if any, may be converted to training data without contaminating
   evaluation?
3. Which taxonomy labels should be merged, split, or clarified after human review?
4. What gold-set size and sampling strata are sufficient?
5. What accuracy thresholds must the semantic judge meet per dimension?
6. Does Phi-4 or Qwen3-14B perform better on a human-labelled compact-taxonomy gold set?
7. Should annotations be applied before or after Apertus 2 native-format mapping?
8. How will production input be physically sharded to avoid repeated scans?
9. Where should published summaries and historical run artifacts live?
10. Which feature branch should receive this work, and how should it be reconciled with
    the two unmerged Apertus 2 processing branches?

## 17. Recommended next action

Do not start with the full tens-of-millions-record run. First review the taxonomy with
the issue owners, select a small stratified sample from license-uncontroversial datasets,
label a gold subset, and run the one-node Qwen smoke test. The result should be inspected
for label validity, unknown rate, semantic mistakes, GPU memory, and throughput. That
evidence will determine whether the taxonomy and judge are ready for a larger pilot.

## 18. Compact taxonomy and Phi-4 comparison (2026-10-29)

The first Glaive run showed that asking Qwen3-14B to choose from the broad `v0.1.0`
taxonomy produced systematic semantic errors, especially treating standalone coding as
SWE. A new experimental `swe_terminal_tool_use_v0.2.0.yaml` was therefore added without
overwriting `v0.1.0`.

The compact version reports only five axes:

- `expert_domains`: `swe`, `terminal`, `tool_use`, or exclusive `none_of_the_above`;
- `programming_languages`, extracted deterministically and kept outside the LLM prompt;
- five `swe_categories`: bug fixing, feature implementation, testing, refactoring, and
  build/dependency/configuration;
- five `terminal_categories`: filesystem inspection, text search/processing, version
  control, build/test/debug, and environment/system operations; and
- five `tool_use_categories`: shell execution, file read/search, file editing,
  code/build/test execution, and external-service interaction.

The taxonomy loader, deterministic classifier, semantic prompt and applicability
validation, dynamic sidecar schema, and summary reporter now support either taxonomy.
The semantic prompt for `v0.2.0` contains 19 labels and enforces
`not_applicable` for an expert-specific category when its expert is absent.

The pinned alternative model is `microsoft/phi-4` at Hugging Face commit
`2db69c1c3e91a05d2c64a3185acfbaf36f744e25` (MIT, advertised 14B class,
14,659,507,200 exact parameters). The Clariden preset and launcher are
`05-annotations/model_presets/phi4_14b_clariden.yaml` and
`05-annotations/slurm/clariden_phi4_14b_judge.sbatch`.

Clariden debug job `3789352` was submitted under account `infra01` for the same fixed
100-record Glaive subset (SHA-256
`3041c97cba7b8a5ba9f5d73b95ff62454b53f89b7e3cf691aa2177e0dec83351`). Its output
directory is
`/iopsstor/scratch/cscs/zakariyyay/apertus-mixture/phi4-14b-glaive-debug-100-taxonomy-v020`.
At handoff-writing time, Slurm's controller/database queries were timing out, so the job
state and results still need checking. The old Qwen output used taxonomy `v0.1.0`; a
strict model comparison requires rerunning Qwen on the same subset with `v0.2.0`, then
evaluating both against human labels rather than comparing raw distributions alone.

