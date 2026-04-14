"""
ActiveUF judge for aspect instruction following
"""

from src.utils import extract_score_distribution_like_activeuf, get_score_from_distribution_like_activeuf

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = ["1", "2", "3", "4", "5"]

# TODO: import these from a file instead because these are reused a lot
SYSTEM_PROMPT_FOR_JUDGE = """You will be doing an Instruction Following Assessment of an AI assistant response.

Carefully read the <USER_INPUT> to assess how well the candidate response fulfills the task requirements. If the input includes a conversation history, the response must align with that context as well as the final instruction.

**Scoring**: Rate the text on a scale of 1 to 5:
1. **Irrelevant**: No alignment.
2. **Partial Focus**: Addresses one aspect poorly.
3. **Partial Compliance**:
    - (1) Meets goal or restrictions, neglecting other.
    - (2) Acknowledges both but slight deviations.
4. **Almost There**: Near alignment, minor deviations.
5. **Comprehensive Compliance**: Fully aligns, meets all requirements.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

def extract_score_distribution(res, scoring_range):
    return extract_score_distribution_like_activeuf(res, scoring_range)

def get_score_from_distribution(score_distribution: dict[str, float]) -> float:
    return get_score_from_distribution_like_activeuf(score_distribution)