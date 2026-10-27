# Dataset catalog

Apertus 2 SFT datasets: who prepared them, how, and where they live. `/capstor/store` is
durable; scratch (`/iopsstor/scratch/...`) is purged after 14 idle days. Statistics come
from `tokens/summary.json` or `apertus-data inspect`. State as of 2026-10-26.

## Overview

| Dataset | Prepared by | Train docs | Tokens | Status |
|---|---|---|---|---|
| [Safety r15 + PersonaFactory v12](#safety-r15--personafactory-v12) | Hanna Yukhymenko | 342,174 | 0.53B | re-tokenize |
| [Instruction following v1](#instruction-following-v1) | Hanna Yukhymenko | 144,162 | 0.16B | re-tokenize |
| [Meditron medical SFT v1](#meditron-medical-sft-v1) | Xavier Theimer-Lienhard | 385,959 | 0.30B | re-tokenize |
| [Nemotron-SFT-Math-v4](#nemotron-sft-math-v4) | Raphael Kreft | 545,408 | 7.17B | re-tokenize; on scratch |
| [Tool use v2](#tool-use-v2) | Ramon Kaspar | 1,340,741 | 5.69B | re-tokenize |
| [Terminal-Bench trajectories v1](#terminal-bench-trajectories-v1) | Ramon Kaspar | 16,117 | 0.23B | re-tokenize |
| [Factuality SFT v0.1](#factuality-sft-v01) | Oscar Wohlfahrt | 10,818 | 0.005B | re-tokenize |
| [Math + STEM v2](#math--stem-v2) (?) | Antonio Lopardo | 5,087,134 | 55.2B | inclusion open |
| [Code SFT v1](#code-sft-v1) | Roman Machacek | 48,208,784 | — | tokenize |

Token counts are for the longest build (64k caps for tool use and terminal).

## Requirements

- Tokenizer: apertus-common `345e6d2` with `Apertus_2_instruct` from apertus-omni-tokenizer
  `3752b12` (NFC normalizer). Every build above used `8bcc096` / `cc727cc` and must be
  re-tokenized.
- Thinking effort: `medium` in every sample.
- Empty thinks: every assistant output starts with a think, empty where the sample or
  turn has no reasoning.

## Dataset Details

### Safety r15 + PersonaFactory v12

- Provider: Hanna Yukhymenko, 2026-10-12; mapper `code/safety_personafactory.py` in the share
- Content: PersonaFactory v12 persona chats (236,191) and safety SFT r15 (105,983), shuffled;
  no reasoning, so every think is empty
- Format: `medium`; no loss weights; check 0 failing rows; licence internal
- Length: p50 1,057, p99 8,322, max 17,730; 1.1% over 8k

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/users/hyukhymenko/shared/apertus2-safety-personafactory-v2/tokens/tokens` |
| Native, check | `.../apertus2-safety-personafactory-v2/{native,check}` |
| Raw | `/iopsstor/scratch/cscs/hyukhymenko/safety_sft/data/sft_r15so_trl.jsonl`; `/capstor/store/cscs/swissai/infra01/reasoning/users/hyukhymenko/personafactory-release-20260922/data` (ask Hanna) |

### Instruction following v1

- Provider: Hanna Yukhymenko, 2026-10-26; mapper and verification in `code/` of the share
- Content: the `instruction_following` domain of the Apertus 1.5 SFT mix (hub `d38f5b9`):
  MiniMax-M2.7 verified single-turn (54,771), Dolci IF 200k (38,822) and 70k balanced
  (13,989), multiturnIF_LSAIE (28,522), IF 12k multi-turn verified (8,058); all synthetic,
  decontaminated per Hanna
- Cleaning: 22 duplicates, 138 truncated-reasoning replies and 16 rows with invalid
  injected tools dropped; 55 replies with leaked `</think>` split into think and reply
- Format: `medium`; think-first, empty except 618 conversations with source reasoning
  (sample check: 3,000 rows, every burst think-led); 3.9% carry injected unrelated tools to
  teach non-use; no loss weights; check 0 failing rows
- Length: p50 647, p99 5,721, max 37,795; 0.2% over 8k

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/users/hyukhymenko/shared/apertus2-apertus15-if-v1/tokens/tokens` |
| Native, check | `.../apertus2-apertus15-if-v1/{native,check}` |
| Raw | `swiss-ai/Apertus-v1.5-SFT-mix` (infra01 mirror in `datasets-hf-private/`) |

### Meditron medical SFT v1

- Provider: Xavier Theimer-Lienhard, 2026-10-19; remapped to `medium` by Raphael the same
  day; mapper `apertus2_native_mapper.py` in the native folder
- Content: single-turn medical Q&A answered by gpt-oss-120b: `curated_qa` (MedMCQA, MedQA,
  HealthSearchQA, LiveQA; 177,386), `synthetic_qa` (184,531), `synthetic_moove` clinical
  vignettes (24,042); multiple-choice answers carry the option letter as a claim
- Format: `medium`, all thinks empty; no loss weights; check 0 failing rows
- Licence: MIT and CC BY 4.0 sources plus own synthetic data; licence sheet row pending
- Decontaminated by the provider against medical and post-training evals. 147,092 MedMCQA
  rows overlap the Apertus 1.5 mix; the provider recommends dropping the 1.5 copies
- Length: p50 487, max 8,843; needs at least 8k (a 4k cut loses 16% of the vignettes)

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/datasets/meditron-medical-sft-v1-native/megatron/tokens` |
| Native, check | `.../meditron-medical-sft-v1-native/{native,check}` |
| Raw | `/capstor/store/cscs/swissai/infra01/datasets/meditron-medical-sft-v1` |

### Nemotron-SFT-Math-v4

- Provider: Raphael Kreft, 2026-10-19; mapper `mappers/nemotron_math.py`
- Content: AoPS and Math StackExchange problems solved by DeepSeek-V4-Pro; `cot` (285,516)
  and `tir` with Python tool rounds (259,892); the `\boxed{}` answer is a claim
- Format: `medium` (source generated in high mode); source reasoning as thinks, empty
  thinks only between some `tir` tool rounds; no loss weights; check 0 failing rows
- Licence: per row, CC BY 4.0 (AoPS) or CC BY-SA 4.0 (Math StackExchange). Antonio's sheet
  keeps only the CC BY rows of this release; the SA rows probably need filtering
- Length: p50 5,456, p99 90,589, max 205,116; 23.7% over 16k, so cap before mixing
- Open: all stages are on scratch (last read 2026-10-22, purge from about 2026-11-05) and
  need moving to the store; AoPS overlap with STEM `nemotron_math_v2` not checked

| Stage | Path |
|---|---|
| Tokens | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/nemotron-math-tokens/tokens` |
| Native, check | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/nemotron-math-{native,check}` |
| Raw | `nvidia/Nemotron-SFT-Math-v4` on Hugging Face |

### Tool use v2

- Provider: Ramon Kaspar, 2026-10-20 (WIP); mapper `mappers/apertus_tool_use.py` on
  `feat/apertus2-tool-use-mapper` at `4504aae`
- Content: tool-calling and agentic conversations from 15 releases (xLAM, Nemotron agentic
  v2, Toucan, EnvScaler, ToolACE, Glaive, OpenSeeker, LiteResearcher, Jupyter agent,
  When2Call); one sample per conversation, every turn trained
- Format: `medium`; 30% of the thinks empty; loss weights written (user and tool results
  untrained); check 0 failing rows; decontaminated against `decon_refs/v2`
- Licence: per row, CC BY 4.0, Apache 2.0 or MIT
- Builds: 64k (1,340,741 docs, 5.69B tokens) and 8k (1,203,776 docs, 3.42B tokens); longer
  samples dropped. The 8k build loses almost all of EnvScaler, Nemotron search and OpenSeeker
- Validation: 2,941 docs (64k), 2,167 (8k)

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/datasets_tokenized/tool_use_apertus_v2/sft/tool_use/v2/{8k,64k}/{train,validation}/tokens` |
| Native, check | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/apertus2/tool_use/v2/{8k,64k}/{native,check}/{train,validation}` |
| Raw | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/mixes/tool_use/v2` |

### Terminal-Bench trajectories v1

- Provider: Ramon Kaspar, 2026-10-20 (WIP, will grow); same mapper as tool use
- Content: Terminal-Bench-style trajectories in the Terminus-2 format (command JSON in the
  reply, terminal output as user messages) from `recursive_task_synthesis`, `termgrade`
  and `recursive_task_synthesis_glm`; at most 3 runs per task
- Format: `medium`; 61% of the thinks empty; loss weights written (terminal output
  untrained); check 0 failing rows; decontaminated against `decon_refs/v2`
- Licence: per row, CC BY 4.0 or Apache 2.0
- Builds: 64k (16,117 docs, 234M tokens) and 8k (5,270 docs, 30M tokens; keeps 32%)
- Open: Terminus-2 versus native tool calls

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/datasets_tokenized/tool_use_apertus_v2/sft/terminal/v1/{8k,64k}/{train,validation}/tokens` |
| Native, check | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/apertus2/terminal/v1/{8k,64k}/{native,check}/{train,validation}` |
| Raw | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/mixes/terminal/v1` |

### Factuality SFT v0.1

- Provider: Oscar Wohlfahrt (#1011), 2026-10-21; mapper `factuality_sft/1` in
  swiss-ai/factuality-sft-data at `042b164`; larger versions to follow
- Content: closed-book NQ answers (5,321), grounded SQuAD 2.0 answers and abstentions
  (4,459), multi-hop MuSiQue (1,038) with passages as RAG documents or attachments
- Format: `medium`, all thinks empty; no loss weights; check 0 failing rows
- Licence: not stated
- Notes: `eval_questions.jsonl` holds the factuality eval questions for decontaminating
  the whole mix; length max 5,753
- Raw: `/capstor/store/cscs/swissai/infra01/datasets/factuality-sft-v0`

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/datasets/factuality-sft-v0.1-apertus2/tokens/tokens` |
| Native, check | `.../factuality-sft-v0.1-apertus2/{native,check}` |

### Math + STEM v2

- Inclusion: open (?). The thinking effort is mixed per row, against the `medium`
  requirement; decontamination against the Apertus eval suite is still running; Davit
  says Antonio will provide a new initial STEM set
- Provider: Antonio Lopardo, 2026-10-20, announced on
  [apertus-program #1197](https://github.com/swiss-ai/apertus-program/issues/1197#issuecomment-6314466921);
  sources in the [STEM Expert data sheet](https://docs.google.com/spreadsheets/d/1mRXLV7aqNoLp4yDp0s9v39aD66f_VCQhJmw_upbwd8Y/edit);
  code github.com/AntonioLopardo/stem-sft (private)
- Format: `low`/`medium`/`high` per row (`proofs_v3` and most of `ot3_math` are `high`);
  loss weights written; check 0 failing rows; samples contradicting the reference dropped;
  exact dedup; capped at 64k (longer dropped)
- Decontamination: verbatim matches of 80+ characters only
- Overlaps: `synthetic2` is also in Code SFT v1; `nemotron_math_v2` shares AoPS problems with
  Nemotron-SFT-Math-v4; `llmjp_stem` and `osr2` lack a subject filter

| Set | Version | Source | Teacher | Licence | Train docs | Tokens |
|---|---|---|---|---|---|---|
| `nemotron_math_v2` | v0.4 | Nemotron-Math-v2, AoPS CC BY rows | gpt-oss-120b | CC BY 4.0 | 1,969,777 | 18.60B |
| `ot3_math` | v0.4 | Dolci-Think-SFT-32B, OpenThoughts3 math | QwQ-32B | ODC-BY per sheet, Apache 2.0 per README | 720,882 | 15.34B |
| `osr2` | v0.5 | OpenScienceReasoning-2 | DeepSeek-R1-0528 | CC BY 4.0 | 1,598,971 | 10.00B |
| `proofs_v3` | v0.1 | Nemotron-Math-Proofs-v3-SFT | DeepSeek-V4-Pro | CC BY 4.0 | 237,081 | 9.57B |
| `synthetic2` | v0.4 | SYNTHETIC-2-SFT-verified | DeepSeek-R1-0528 | Apache 2.0 | 53,606 | 0.55B |
| `science_v1_mcq` | v0.5 | Nemotron-Science-v1 | gpt-oss-120b | CC BY 4.0 | 173,908 | 0.52B |
| `llmjp_stem` | v0.4 | llm-jp-4.1-thinking-sft-data | gpt-oss-120b | CC BY 4.0 | 280,825 | 0.30B |
| `science_v1_rqa` | v0.4 | Nemotron-Science-v1 | gpt-oss-120b | CC BY 4.0 | 52,084 | 0.30B |

Validation: 5,195 docs in total.

| Stage | Path |
|---|---|
| Tokens | `/capstor/store/cscs/swissai/infra01/datasets/stem_apertus_v2/sft/apertus2/latest/<set>/tokens/{train,validation}/tokens` |
| Native, check | `.../latest/<set>/{native,check}/{train,validation}` |

### Code SFT v1

- Provider: Roman Machacek (SWE/Terminal/Tool Use team, #1271), 2026-10-22; scripts in
  `/iopsstor/scratch/cscs/rmachace/data/apertus2_jobs`
- Content: 76 Hugging Face code releases. `code_sft` (40 sets, 45.36M rows): single-turn
  programming, largest CodeX-7M, UltraData-SFT, Ling-Coder, OpenCodeInstruct, opc-sft.
  `code_agentic` (36 sets, 2.85M rows): SWE and terminal trajectories with native tool
  calls (Open-SWE-Traces, SWE-Zero, Nemotron-SFT-SWE, AgentTrove, Mocha, Sera)
- Format: `medium`, think-first with empty thinks (checked on sample shards); no loss
  weights yet; check 0 failing rows on all 48.2M rows
- Decontamination: against 15 code benchmarks (HumanEval, MBPP, LiveCodeBench, APPS,
  CodeContests, BigCodeBench, MultiPL-E, McEval, SWE-bench variants); exact containment of
  80+ characters or 5-token Jaccard ≥ 0.80; 834,430 rows removed (1.7%); no dedup
- Licence: per set in `code_posttrain/openai_sft/dataset_inventory.csv`. About 1.5M rows
  are questionable: CC BY-NC (KodCode, 893k), CC BY-NC-SA (CodeExercise), CC BY-SA
  (McEval-Instruct), OpenRAIL-M (Code-Evol-Instruct), unknown (Sera, Mocha, alquistcoder,
  Mixture-of-Thoughts, SYNTHETIC-2 SFT and others; 569k)
- Access: final path confirmed by Roman; move and permissions done 2026-10-26, all 78,125
  shards present and group-readable (466 GB `code_sft`, 109 GB `code_agentic`)
- Open: not tokenized; 48M rows need per-set sampling

| Stage | Path |
|---|---|
| Native | `/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft/decontaminated/datasets/{code_sft,code_agentic}/<org>/<set>/shard-*/native.parquet` |
| Check | `.../code_posttrain/sft/decontaminated/check` |
| Raw | `.../code_posttrain/sft/{code_sft,code_agentic}`; OpenAI-format export in `.../code_posttrain/openai_sft` |
