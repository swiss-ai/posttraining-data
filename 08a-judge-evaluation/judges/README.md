# Judges

## Overview

| #  | Name | LLM | Scoring Methodology |
|----|--------|-------|---------|
| 00 | 00-ActiveUF | Qwen3-235B-A22B | Mean of scores from judges 01-04 |
| 01 | 01-ActiveUF-Helpfulness | Qwen3-235B-A22B | Logprobs-based scoring |
| 02 | 02-ActiveUF-Instruction_Following | Qwen3-235B-A22B | Logprobs-based scoring |
| 03 | 03-ActiveUF-Honesty | Qwen3-235B-A22B | Logprobs-based scoring |
| 04 | 04-ActiveUF-Truthfulness  | Qwen3-235B-A22B | Logprobs-based scoring |
||
| 10 | 10-General_Quality | Qwen3-235B-A22B | Logprobs-based scoring |
| 11 | 11-SwissAI_Charter | Qwen3-235B-A22B | Logprobs-based scoring |
||
| 20 | 20-Qwen3.5_35B-Helpfulness | Qwen3.5-35B-A3B-FP8 | Logprobs-based scoring |
| 21 | 21-Qwen3.6_27B-Helpfulness | Qwen3.6-27B | Logprobs-based scoring |
||
| 30 | 30-ArenaHard-regex | Qwen3-235B-A22B | Regex-based score extraction, with random as fallback |
| 31 | 31-Qwen3.6_27B-ArenaHard-regex | Qwen3.6-27B | Regex-based score extraction, with random as fallback |
| 32 | 32-Qwen3.6_27B-ArenaHard | Qwen3.6-27B | Logprobs-based scoring |

## Adding a new judge

1. Add `judges/NN.py` and set `name = "NN"`.
2. Usually copy `01.py` and adjust prompts and any overrides; import shared pieces from `activeuf` (`extract_score_distribution`, prompts, `model`, server fields, etc.) as needed.
3. The module must expose **snake_case** names used by `src.judge` and `run_judge.py`: e.g. `model`, `system_prompt_for_judge`, `user_prompt_for_judge` (with `{prompt}` and `{response}` in the user template), `temperature`, `max_tokens`, `logprobs`, `top_logprobs`, `scoring_range`, `concurrent`, and for the launcher `slurm_nodes`, `workers`, `nodes_per_worker`, `dp_size`, `tp_size`, `framework`, `disable_ocf`.
4. Implement `extract_score_distribution(response, scoring_range)`; the first argument is the **full** chat completion `response` from the API (see `activeuf` for logprob-based scoring, `12.py` for text/regex). Optional: `get_score_from_distribution` for downstream metrics.

**Skeleton** (re-exporting ActiveUF-style defaults):

```python
import activeuf

name = "06"
model = activeuf.model
# … copy serving / hyperparameters from 01.py or activeuf as needed …
system_prompt_for_judge = activeuf.system_prompt
user_prompt_for_judge = activeuf.helpfulness_user_prompt
extract_score_distribution = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution
```

