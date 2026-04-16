"""
Aspectless judge, just general quality. Prompts are only minimally modified from those for activeuf judges.
"""

from src.utils import extract_score_distribution_like_activeuf, get_score_from_distribution_like_activeuf

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = ["1", "2", "3", "4", "5"]
LOGPROBS = True
TOP_LOGPROBS = 20

# TODO: import these from a file instead because these are reused a lot
SYSTEM_PROMPT_FOR_JUDGE = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response. You'll receive an input with two sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response. 

Carefully read the provided input to understand the task, then assess how good the response is at addressing the user input prompt. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

USER_PROMPT_FOR_JUDGE = """You will be doing a Quality Assessment of an AI assistant response.

Score on a scale of 1 to 5 based on response quality:
1. **Very Poor**: The response fails to address the user's prompt.
2. **Poor**: The response partially addresses the user's prompt but has notable shortcomings.
3. **Acceptable**: The response adequately addresses the user's prompt.
4. **Good**: The response thoroughly addresses the user's prompt.
5. **Excellent**: The response exceptionally addresses the user's prompt.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

def extract_score_distribution(res, scoring_range):
    return extract_score_distribution_like_activeuf(res, scoring_range)

def get_score_from_distribution(score_distribution: dict[str, float]) -> float:
    return get_score_from_distribution_like_activeuf(score_distribution)