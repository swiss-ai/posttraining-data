"""
ActiveUF judge for aspect truthfulness
"""

from src.utils import extract_score_distribution_like_activeuf, get_score_from_distribution_like_activeuf

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = ["1", "2", "3", "4", "5"]

# TODO: import these from a file instead because these are reused a lot
SYSTEM_PROMPT_FOR_JUDGE = """You will be doing a Truthfulness and Hallucination Assessment of an AI assistant response.

Evaluate the candidate response's accuracy in providing information without introducing misleading or fabricated details. 

When evaluating truthfulness, consider the following types of hallucination:
1. **Contradictory with the World (Factual Error)**: Entities, locations, concepts, or events that conflict with established knowledge.
2. **Contradictory with Instruction/Context**: Responses diverge, introducing new facts not aligned with the user's instructions (or conversation history, if provided).
3. **Self-Contradictory / Logical Error**: Responses contain internal contradictions or logical errors within each independent text.

Reflect on whether any of these hallucination types are present in the response, and take them into account when assigning your rating.

**Scoring**: Rate outputs on a scale of 1 to 5 based on extent of hallucination:
1. **Completely Hallucinated**: Entirely unreliable due to hallucinations.
2. **Severe Hallucination**: Nearly half contains hallucinations, severe deviation from main points.
3. **Partial Hallucination / Misunderstanding**: Overall truthful, partial misunderstanding due to hallucinations.
4. **Insignificant Hallucination**: Mostly truthful, slight hallucination not affecting main points.
5. **No Hallucination**: Free of hallucinations.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{completion}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

def extract_score_distribution(res, scoring_range):
    return extract_score_distribution_like_activeuf(res, scoring_range)

def get_score_from_distribution(score_distribution: dict[str, float]) -> float:
    return get_score_from_distribution_like_activeuf(score_distribution)