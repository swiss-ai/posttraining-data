Copy the code below to a new file (e.g. judges/02.py) and fill in prompts and scoring.

Required exports (names matter): MODEL, TEMPERATURE, MAX_TOKENS,
SYSTEM_PROMPT_FOR_JUDGE, USER_PROMPT_FOR_JUDGE, extract_score_distribution.
Optional: SCORING_RANGE when using the ActiveUF-style logprob helper below.

```{python}
# Loaded by src/judge.py with src/ on sys.path — use `utils`, not `src.utils`.
from utils import extract_score_distribution_like_activeuf

MODEL = "your-serving-model-id"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = [1, 2, 3, 4, 5]

SYSTEM_PROMPT_FOR_JUDGE = """You are an impartial judge. Output only one token: an integer score from the allowed set. No other text."""

USER_PROMPT_FOR_JUDGE = """Rate the assistant response for the user task.

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""


def extract_score_distribution(res, scoring_range):
    return extract_score_distribution_like_activeuf(res, scoring_range)
```