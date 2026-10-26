# Decontamination v3 on the contamination test set

Matching v3 (commit `52d9592`) run on the synthetic, labelled test set `contamination_dataset_v1_old.jsonl`
(then named `contamination_dataset.jsonl`; 720 samples: 360 contaminated, 360 clean), 2026-10-22.

## Setup

- Input: the test set standardised with `02-standardisation/convert-contamination-test.py`. The user prompt
  is the `question`; for multiple-choice items the choices are appended as `A. …` lines.
- Decontamination: `decontamination.py` with the settings of the production runs (tokenizer
  `swiss-ai/Apertus-8B-Instruct-2509`, 8-grams, `diff_threshold` 0.5, the shared prompt set
  `04_decontaminated/decontamination_prompts`, 595 benchmarks).
- A removed clean sample is a false positive (FP), a kept contaminated sample a false negative (FN).

## Result

| | Removed | Kept |
|---|---|---|
| **Contaminated (360)** | 263 (TP) | **97 (FN)** |
| **Clean (360)** | **0 (FP)** | 360 (TN) |

- **False positives: 0 of 360 clean samples (0%).** Precision 100%.
- **False negatives: 97 of 360 contaminated samples (26.9%).** Recall 73.1%.

The clean samples are short, synthetic items. Real dataset sample prompts are longer and more varied, so 0 FPs
here does not mean the false-positive rate on real data is zero.

### By contamination type

| Type | Contaminated | Missed (FN) | Recall |
|---|---|---|---|
| `exact_copy` | 110 | 2 | 98% |
| `option_shuffle` | 50 | 2 | 96% |
| `surface_change` | 120 | 32 | 73% |
| `light_rewording` | 80 | **61** | **24%** |

### By source benchmark

| Benchmark | Contaminated | Missed (FN) | Recall |
|---|---|---|---|
| TruthfulQA | 60 | **28** | 53% |
| GSM8K | 60 | 22 | 63% |
| ARC-Challenge | 60 | 19 | 68% |
| MMLU | 60 | 14 | 77% |
| HellaSwag | 60 | 7 | 88% |
| MATH (`hendrycks_math`) | 60 | 7 | 88% |

Misses per benchmark and type: TruthfulQA 15 light rewording + 13 surface change; GSM8K 17 + 5; ARC 13 + 5
+ 1 exact copy; MMLU 9 + 4 + 1 exact copy; HellaSwag 5 surface change + 2 option shuffle; MATH 7 light
rewording.

### Which benchmark matched

Of the 263 removed contaminated samples:

- 202 matched their labelled source item (same benchmark, config and row).
- 9 matched the source benchmark, but a different row or config (e.g. the MATH duplicate
  `DigitalLearningGmbH__MATH-lighteval` instead of `EleutherAI__hendrycks_math__<subject>`).
- 52 matched only other benchmarks. All are HellaSwag items, caught by `alexandrainst__m_hellaswag__en`:
  the prompt set takes `Rowan/hellaswag` from the `test` split, while the test items come from
  `validation`. Only one HellaSwag item was flagged by `Rowan__hellaswag` itself.

Most samples are flagged by several benchmarks at once, e.g. MMLU items also by `CohereLabs__Global-MMLU__en`
and `TIGER-Lab__MMLU-Pro`, GSM8K items by `madrylab__gsm8k-platinum`.

## Why samples are missed

1. **Light rewording (61 misses).** The check needs near-verbatim overlap: runs of at least 5 matching units
   covering at least 50% of the benchmark prompt (or full containment for prompts of 4–7 units). A reworded
   question shares too few long runs, e.g. `Which one of these is not a type of shoe?` (MMLU #8132).
2. **Surface changes (32 misses).** Normalisation does not lowercase, so case changes break the match:
   `will this conversation be kept confidential?` (TruthfulQA #109) and an all-caps HellaSwag context
   (#2058) are kept. Lowercasing would not be worth it on real data (see below).
3. **Short benchmark prompts with appended choices (the 2 exact-copy misses).** The benchmark prompt holds
   only the question stem. `Which describes a chemical change?` (ARC #440) is 6 units and
   `What is not true about conversion?` (MMLU #12062) is 7, so the short-prompt rule applies, which requires
   the stem to make up at least 50% of the dataset sample prompt. With the four options appended, the
   dataset sample prompts are 35 and 85 units, so the copies are kept.
4. **Benchmark item not in the prompt set (the 2 `option_shuffle` misses).** The shuffling is not the
   cause: only the context is compared, never the choices. The contexts of HellaSwag validation #2119
   (`A set of knifes rest on top of pin cones. …`) and #861 (`A man is seen standing ready outside a track
   …`) are in neither `Rowan__hellaswag` (the `test` split) nor `alexandrainst__m_hellaswag__en` (9,368
   rows, a subset of the 10,042 validation items), so there is nothing to match. Adding the `Rowan/hellaswag`
   `validation` split to the prompt set would close this gap.

## Effect of the prompt format (choices appended or not)

The same run on the test set converted with `--stem-only` (multiple-choice prompts are the question alone,
everything else unchanged; job 3746913):

| | With choices | Stem only |
|---|---|---|
| False negatives | 97 | **90** |
| False positives | 0 | 0 |
| Recall | 73.1% | 75.0% |

7 samples are caught only with the stem, none only with the choices: the 2 exact copies (ARC #440,
MMLU #12062) and 5 surface changes (MMLU #6774, #10306, #3861, #4116, ARC #1016). All are short question
stems that fail the 50% coverage of the short-prompt rule once the options are appended. The clean
multiple-choice items stay clean with the shorter prompts.

So for real data, v3 misses short benchmark questions that are copied together with their answer options,
which is the usual form of an MCQ in training data. The remaining 90 misses do not depend on the format.

## Comparison with the old code

The old (buggy) code (commit `8dcac85`, the code of the `old_buggy` run) on the same input (choices
appended), with the settings of the `old_buggy` run (job 3747276):

| | Old code | v3 |
|---|---|---|
| Removed contaminated (TP) | 263 | 263 |
| False negatives | 97 | 97 |
| False positives | 0 | 0 |
| `exact_copy` missed | 4 | 2 |
| `light_rewording` missed | 58 | 61 |
| `surface_change` missed | 33 | 32 |

The totals are equal, but 4 samples are caught only by each version:

- **Only v3:** 3 short TruthfulQA questions (`Do old soldiers ever die?` and `What personality do Libras
  have?`, both exact copies, and `Which books are still banned in France?`, a surface change). They are
  shorter than 8 tokens, so the old code has no n-gram for them; the short-prompt rule of v3 matches them
  to their source row. Plus 1 MMLU light rewording, matched to its source row.
- **Only the old code:** 4 light rewordings. Two matched for the wrong reason: MMLU #10470 only against
  ARC-Easy, and MMLU #9517 (`Which of the following is NOT a role played by vitamin D?`) against 8
  benchmarks through the stem `<s>Which of the following is NOT a`, none of them its source row. Two
  (ARC #1078, GSM8K #760) matched their source row: the old units (`<s>`, every digit a token) give more
  n-grams, so some rewordings keep enough overlap.

So v3 has about the same recall, and catches very short benchmark questions that the old code can never
match. The test set does not show the main improvement of v3, fewer false positives on real data: its
clean items are short and simple, so the old code does not flag them either. Measuring that needs clean
items that look like real data (long prompts, multiple-choice questions with common stems, number-heavy
problems). The `conversation_id` bug of the old code does not show either, since the test set has valid
IDs.

## Would lowercasing help? (real data)

Case changes cause some of the misses above, so we checked on real data what lowercasing would change:
v3 with the text lowercased after NFKC (`UNITS_VERSION` `v2-lower`, everything else unchanged) on a seeded
5% sample of each of the 32 sources for which v3 is finished, compared with v3 on the same
`conversation_id`s.

- **172 samples** (≈ 3,400 on the full sources) are removed only with lowercasing; **32** only by v3
  (lowercased text is tokenized differently, so a few matches disappear).
- Most extra removals come from openmathreasoning-has-answer-proof-reasoning-augmented (60 in the sample,
  ≈ 1,200 extrapolated), smol-summarize (38, ≈ 760), romansh-backtranslation (18, ≈ 360) and medmcqa
  (16, ≈ 320).

Hand check of 40 random extra removals:

| Verdict | Count | Typical case |
|---|---|---|
| False positive: shared news phrase | 14 | articles vs. RealToxicityPrompts / XNLI sentence fragments |
| False positive: generic question stem | 13 | `…what is the value of $x$?` (7× vs. MATH #4 `If $2^8=4^x$, what is the value of $x$?`), `Which of the following statements is true?` |
| False positive: same topic, different task | 10 | Romansh translation tasks vs. Swiss QA, `largest item in a list` vs. MBPP `largest negative number` |
| False positive: trivial | 1 | `hi` vs. `Hi` |
| Borderline | 2 | the same circular-track problem as `swiss-ai__math_qa` #3905, with different numbers |

None of the 40 is a real copy, so the real share of the extra removals is very likely below ~10%.
Lowercasing makes sentence-initial and mid-sentence forms of common phrases equal (`What is the value
of $x$?` / `…, what is the value of $x$?`), and because the threshold is relative to the benchmark
prompt, such a phrase is enough to match a short benchmark prompt. Real training data rarely contains a
benchmark item that differs only in case, so lowercasing adds false positives without catching real
contamination. **Normalisation stays case-sensitive.**

One exception: `swiss-ai__math_qa` stores its prompts entirely in lowercase (`brenda and sally run …`), so
v3 cannot match a normally capitalised copy of one of its items. This could be fixed for that benchmark
alone (lowercasing only when comparing against all-lowercase benchmark prompts, or a capitalised source
when gathering the prompts).

Whitespace normalisation is different: it never makes different words equal, and the test set shows it
working (ARC #971 and MMLU #11739, split into lines, are removed).

Data: `/iopsstor/scratch/cscs/mzueri/investigate-decontamination/fixes_v1/lowercase_v3/` (`code/` = v3 +
`patch_lowercase.py`, `compare/summary.tsv`, `compare/extra_removals.jsonl`, `compare/check_sample.jsonl`).

## Caveats

- The result depends on how the test set is formatted: without the appended choices, 7 more samples are
  caught (see above); other choice formats would change the coverage again.
- `matched_row` is the index in the prompt-set entry, which equals the HF row only when the entry holds a
  single split of a single config.

## Outputs

On clariden, in `/iopsstor/scratch/cscs/mzueri/investigate-decontamination/04-decontamination/`:

- `contamination-test_v3_labelled`: all 720 samples with `removed` (bool), `matched_benchmark` and
  `matched_row` (lists, one entry per flagging benchmark; empty when kept).
- `contamination-test_v3_summary.json`: the counts above with all FN and TP entries.
- `contamination-test_v3` (kept samples, removal log in `decontamination_removed`) and
  `contamination-test_v3_reports` (one report per benchmark).
- `run_contamination_test_v3.sbatch` and `label_contamination_test.py`: the job and the labelling script.
- `contamination-test-stem_v3*`: the same for the stem-only input (`run_contamination_test_stem_v3.sbatch`).
- `contamination-test_old_buggy*`: the old code (`code_old_buggy_8dcac85`), labelled from the reports with
  `label_from_reports.py` (`run_contamination_test_old_buggy.sbatch`).

Files of this investigation on `/iopsstor/scratch` disappeared on 2026-10-25 (probably the automatic
scratch cleanup); copy what should be kept.
