# Judges

## Overview

| Judge | Description | Model | Scoring |
|-------|--------|-------|---------|
| 01 | Helpfulness | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 02 | Instruction Following | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 03 | Honesty | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 04 | Truthfulness  | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 05 | Swiss AI Charter compliance | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 11 | General quality | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |
| 12 | ArenaHard Judge | Qwen3-235B-A22B | 1-5, Discrete |
| 13 | ArenaHard Judge | Qwen3-235B-A22B | 1-5, ActiveUF-style logprobs |

## Adding a new judge

Copy the template below to a new file (e.g. `judges/06.py`) and fill in prompts and scoring.

Required exports: `MODEL`, `TEMPERATURE`, `MAX_TOKENS`,
`SYSTEM_PROMPT_FOR_JUDGE`, `USER_PROMPT_FOR_JUDGE`, `extract_score_distribution`.
Optional: `SCORING_RANGE` when using the ActiveUF-style logprob helper.

```python
from src.utils import extract_score_distribution_like_activeuf

MODEL = "your-serving-model-id"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = ["1", "2", "3", "4", "5"]

SYSTEM_PROMPT_FOR_JUDGE = """..."""

USER_PROMPT_FOR_JUDGE = """...

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""


def extract_score_distribution(res, scoring_range):
    return ...

def get_score_from_distribution(score_distribution):
    return ...
```

