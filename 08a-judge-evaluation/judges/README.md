# Judges

## Overview

| Judge | Description | Model | Scoring |
|-------|--------|-------|---------|
| 00 | ActiveUF | Qwen3-235B-A22B | 1-5, Mean of scores from judges 01-04 |
| 01 | Helpfulness | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 02 | Instruction Following | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 03 | Honesty | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 04 | Truthfulness  | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 05 | Swiss AI Charter compliance | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 11 | General quality | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 12 | ArenaHard Judge | Qwen3-235B-A22B | 1-5, Discrete |
| 21 | Helpfulness | Qwen3.6-27B | 1-5, ActiveUF-style logprobs |
| 22 | Helpfulness | Qwen3.5-35B-A3B-FP8 | 1-5, ActiveUF-style logprobs |
| 23 | ArenaHard Judge | Qwen3.6-27B | 1-5, Discrete |
| 24 | ArenaHard Judge | Qwen3.6-27B | 1-5, ActiveUF-style logprobs |


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

