# Dataset catalog

Apertus 2 SFT datasets: who prepared them, how, and where they live. `/capstor/store` is
durable; scratch (`/iopsstor/scratch/...`) is purged after 14 idle days. Statistics come
from `tokens/summary.json` or `apertus-data inspect`. State as of 2026-10-28.

## Overview

Datasets for the SFT mix:

| Dataset | Domain | Prepared by | Train docs | Tokens | Tokens ≤16k | Status |
|---|---|---|---|---|---|---|
| [Safety r15 + PersonaFactory v12](#safety-r15--personafactory-v12) | Safety, persona | Hanna Yukhymenko | 342,174 | 0.53B | 0.53B | v0.1 tokenized |
| [Instruction following v1](#instruction-following-v1) | Instruction following | Hanna Yukhymenko | 144,162 | 0.16B | 0.16B | v0.1 tokenized |
| [Meditron medical SFT v1](#meditron-medical-sft-v1) | Medical | Xavier Theimer-Lienhard | 385,959 | 0.30B | 0.30B | v0.1 tokenized |
| [Tool use v2](#tool-use-v2) | Tool use | Ramon Kaspar | 1,340,741 | 5.69B | 4.41B | v0.1 tokenized |
| [Terminal-Bench trajectories v1](#terminal-bench-trajectories-v1) | Terminal agents | Ramon Kaspar | 16,117 | 0.23B | 0.10B | v0.1 tokenized |
| [Factuality SFT v0.1](#factuality-sft-v01) | Factuality | Oscar Wohlfahrt | 10,818 | 0.005B | 0.005B | v0.1 tokenized |
| [STEM SFT v3](#stem-sft-v3) | Math, science | Davit Melikidze | 501,902 | 1.68B | 1.68B | v0.1 tokenized |
| **Total without code** | | | **2,741,873** | **8.59B** | **7.18B** | |
| [Code SFT v1](#code-sft-v1) | Code, SWE agents | Roman Machacek | 48,208,784 | ~398B | ~158B | sample, then tokenize |

Tokens are for the v0.1 builds, the train split of every dataset except code tokenized
2026-10-28 with the pinned tokenizer (job 3785881) into
`/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/<name>/{full,16k}`; tool use and
terminal from their 64k native builds. ≤16k is what `cap --max-tokens 16384` keeps. Code
counts are estimates from about 1,000 rows per set tokenized with the current tokenizer
(`/iopsstor/scratch/cscs/rkreft/apertus2-processing/code-lengths-16k.json`): `code_sft`
~295B (~151B ≤16k), `code_agentic` ~103B (~7B ≤16k).

Development datasets, not in the mix:

| Dataset | Domain | Prepared by | Train docs | Tokens | Tokens ≤16k | Status |
|---|---|---|---|---|---|---|
| [Nemotron-SFT-Math-v4](#nemotron-sft-math-v4) | Math | Raphael Kreft | 545,408 | 7.17B | 2.12B | re-tokenized 2026-10-28; on scratch |

## Requirements

- Tokenizer: apertus-common `494c2be` with `Apertus_2_instruct` from apertus-omni-tokenizer
  `b801ff1` on `main` (NFC normalizer; same files as `3752b12`). The providers' own builds
  used `8bcc096` / `cc727cc`; use the v0.1 builds and Nemotron's re-tokenized build instead.
- Thinking effort: `medium` in every sample.
- Empty thinks: every assistant output starts with a think, empty where the sample or
  turn has no reasoning.
- Loss weights: no native row of any dataset here carries a `loss_weight` annotation (scan
  of all rows, 2026-10-28), so the trainer defaults apply and every dataset is tokenized
  without `--loss-weights`. The old tool use and terminal weights equal those defaults.

## Dataset Details

### Safety r15 + PersonaFactory v12

- Provider: Hanna Yukhymenko, 2026-10-12; mapper `code/safety_personafactory.py` in the share
- Content: PersonaFactory v12 persona chats (236,191) and safety SFT r15 (105,983), shuffled;
  no reasoning, so every think is empty
- Format: `medium`; no loss weights; check 0 failing rows; licence internal
- Length: p50 1,057, p99 8,322, max 17,730; 1.1% over 8k

| Stage | Path |
|---|---|
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/safety-personafactory-v2/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/users/hyukhymenko/shared/apertus2-safety-personafactory-v2/tokens/tokens` |
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
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/if-v1/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/users/hyukhymenko/shared/apertus2-apertus15-if-v1/tokens/tokens` |
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
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/meditron-medical-v1/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/datasets/meditron-medical-sft-v1-native/megatron/tokens` |
| Native, check | `.../meditron-medical-sft-v1-native/{native,check}` |
| Raw | `/capstor/store/cscs/swissai/infra01/datasets/meditron-medical-sft-v1` |

### Tool use v2

- Provider: Ramon Kaspar, 2026-10-20 (WIP); mapper `mappers/apertus_tool_use.py` on
  `feat/apertus2-tool-use-mapper` at `4504aae`
- Content: tool-calling and agentic conversations from 15 releases (xLAM, Nemotron agentic
  v2, Toucan, EnvScaler, ToolACE, Glaive, OpenSeeker, LiteResearcher, Jupyter agent,
  When2Call); one sample per conversation, every turn trained
- Format: `medium`; 30% of the thinks empty; no `loss_weight` annotations (the old build's
  weights equal the trainer defaults); check 0 failing rows; decontaminated against
  `decon_refs/v2`
- Licence: per row, CC BY 4.0, Apache 2.0 or MIT
- Builds: 64k (1,340,741 docs, 5.69B tokens) and 8k (1,203,776 docs, 3.42B tokens); longer
  samples dropped. The 8k build loses almost all of EnvScaler, Nemotron search and OpenSeeker
- Validation: 2,941 docs (64k), 2,167 (8k)

| Stage | Path |
|---|---|
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/tool-use-v2/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/datasets_tokenized/tool_use_apertus_v2/sft/tool_use/v2/{8k,64k}/{train,validation}/tokens` |
| Native, check | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/apertus2/tool_use/v2/{8k,64k}/{native,check}/{train,validation}` |
| Raw | `/capstor/store/cscs/swissai/infra01/datasets/tool_use_apertus_v2/sft/mixes/tool_use/v2` |

### Terminal-Bench trajectories v1

- Provider: Ramon Kaspar, 2026-10-20 (WIP, will grow); same mapper as tool use
- Content: Terminal-Bench-style trajectories in the Terminus-2 format (command JSON in the
  reply, terminal output as user messages) from `recursive_task_synthesis`, `termgrade`
  and `recursive_task_synthesis_glm`; at most 3 runs per task
- Format: `medium`; 61% of the thinks empty; no `loss_weight` annotations (terminal output
  is user input, untrained by default); check 0 failing rows; decontaminated against
  `decon_refs/v2`
- Licence: per row, CC BY 4.0 or Apache 2.0
- Builds: 64k (16,117 docs, 234M tokens) and 8k (5,270 docs, 30M tokens; keeps 32%)
- Open: Terminus-2 versus native tool calls

| Stage | Path |
|---|---|
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/terminal-v1/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/datasets_tokenized/tool_use_apertus_v2/sft/terminal/v1/{8k,64k}/{train,validation}/tokens` |
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
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/factuality-v0.1/{full,16k}/tokens` |
| Tokens, old tokenizer | `/capstor/store/cscs/swissai/infra01/datasets/factuality-sft-v0.1-apertus2/tokens/tokens` |
| Native, check | `.../factuality-sft-v0.1-apertus2/{native,check}` |

### STEM SFT v3

- Provider: Davit Melikidze with Antonio Lopardo's stem-sft pipeline; v3 2026-10-28
  (announced on Slack), after v1 (2026-10-27, Slack DM). Replaces Math + STEM v2
  (`stem_apertus_v2/sft/apertus2/latest`). v2 (11:19) has the same rows; v4 (18:58, not
  announced) drops `deepmath` and 3,885 excluded ids (423,254 rows, 1.39B tokens)
- Content: seven sets, 182,134 science and 319,768 math rows. A claim holds the boxed
  answer or the answer letter; 1,662 rows have none
- Format: `medium` in every row; the source effort is kept in `source_thinking` (none
  233,853, low 121,858, medium 138,370, high 7,821). The four v1 sets have source reasoning;
  `openmathreasoning`, `deepmath` and `openr1_math_default` have none. 30,313
  `nemotron_math_v2` rows call Python. No loss weights; the provider's check reports 0
  failing rows; v3 checked here with 0 failing rows of 501,902 (2026-10-29)
- Decontamination: stage 04 against 595 benchmarks (8-grams, threshold 0.5); the v1 sets
  lost 5,146 rows, the new sets were decontaminated separately with the same settings
- Tokenized 2026-10-28 (v0.1): 1,675,513,933 tokens, p50 3,080, p90 6,208, max 8,192 (the
  provider caps at 8,192, so the 16k build is identical)
- Open: no validation split (use `apertus-data split`); `synthetic2` is also in Code SFT v1

| Set | Source | Teacher | Licence | Train docs | Tokens |
|---|---|---|---|---|---|
| `nemotron_math_v2` | Nemotron-Math-v2 (`math.medium`) | gpt-oss-120b | CC BY 4.0 | 70,631 | 0.32B |
| `science_v1_mcq` | Nemotron-Science-v1 MCQ | gpt-oss-120b | CC BY 4.0 | 139,263 | 0.25B |
| `science_v1_rqa` | Nemotron-Science-v1 RQA | gpt-oss-120b | CC BY 4.0 | 42,871 | 0.16B |
| `synthetic2` | SYNTHETIC-2-SFT-verified | DeepSeek-R1-0528 | Apache 2.0 | 15,284 | 0.07B |
| `openmathreasoning` | OpenMathReasoning | | not checked | 117,839 | 0.46B |
| `deepmath` | DeepMath | | not checked | 74,763 | 0.27B |
| `openr1_math_default` | OpenR1-Math (`default`) | | not checked | 41,251 | 0.15B |

Token counts per set come from the provider's `stats.json`.

| Stage | Path |
|---|---|
| Tokens v0.1 | `/iopsstor/scratch/cscs/rkreft/v0_1_chonk_sft_tokenized/stem-v3/{full,16k}/tokens` |
| Native | `/capstor/store/cscs/swissai/infra01/datasets/experts/STEM/apertus2_stem_sft_v3/train` (`stats.json` one level up) |
| Check | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/stem-sft-v3-check` |

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
- Licence: per set in `code_posttrain/openai_sft/dataset_inventory.csv`; matches the
  Hugging Face licence field of all 76 sets (checked 2026-10-28). About 1.27M rows are
  questionable: CC BY-NC (KodCode, 893k), CC BY-NC-SA (CodeExercise, 27k), CC BY-SA
  (McEval-Instruct, 36k), OpenRAIL-M (Code-Evol-Instruct, 4k), none declared (Mocha,
  Mixture-of-Thoughts, SYNTHETIC-2 SFT, Terminal-Lego, LiteCoder World Model; 306k). The
  inventory's "unknown" Sera (ODC-By, "intended for research and educational use") and
  alquistcoder (MIT for synthetic content, public subsets under their own licences) state a
  licence in their README text; 264k rows to check
- Access: final path confirmed by Roman; move and permissions done 2026-10-26, all 78,125
  shards present and group-readable (466 GB `code_sft`, 109 GB `code_agentic`)
- Open: not tokenized; 48M rows need per-set sampling

| Stage | Path |
|---|---|
| Native | `/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft/decontaminated/datasets/{code_sft,code_agentic}/<org>/<set>/shard-*/native.parquet` |
| Check | `.../code_posttrain/sft/decontaminated/check` |
| Raw | `.../code_posttrain/sft/{code_sft,code_agentic}`; OpenAI-format export in `.../code_posttrain/openai_sft` |

## Development Datasets

### Nemotron-SFT-Math-v4

- Provider: Raphael Kreft, 2026-10-19; mapper `mappers/nemotron_math.py`. Own development
  set for this pipeline, not part of the SFT mix
- Content: AoPS and Math StackExchange problems solved by DeepSeek-V4-Pro; `cot` (285,516)
  and `tir` with Python tool rounds (259,892); the `\boxed{}` answer is a claim
- Format: `medium` (source generated in high mode); source reasoning as thinks, empty
  thinks only between some `tir` tool rounds; no loss weights; check 0 failing rows
- Licence: per row, CC BY 4.0 (AoPS) or CC BY-SA 4.0 (Math StackExchange). Antonio's sheet
  keeps only the CC BY rows of this release; the SA rows probably need filtering
- Tokenized 2026-10-28 with apertus-common `494c2be` and tokenizer `b801ff1`: 7,166,294,899
  tokens. Length: p50 5,456, p90 35,745, p99 90,589, max 205,116; 23.7% over 16k. The 16k cap
  keeps 415,906 documents and 2,124,700,666 tokens (p50 3,772, p99 15,798)
- Open: all stages are on scratch (last read 2026-10-28, purge from about 2026-11-11)

| Stage | Path |
|---|---|
| Tokens, ≤16k | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/nemotron-math-tokens-16k/tokens` |
| Tokens | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/nemotron-math-tokens/tokens` |
| Native, check | `/iopsstor/scratch/cscs/rkreft/apertus2-processing/nemotron-math-{native,check}` |
| Raw | `nvidia/Nemotron-SFT-Math-v4` on Hugging Face |
