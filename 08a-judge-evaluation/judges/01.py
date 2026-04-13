"""
ActiveUF judge for aspect helpfulness
"""

from utils import extract_score_distribution_like_activeuf

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TEMPERATURE = 0.0
MAX_TOKENS = 1

SCORING_RANGE = ["1", "2", "3", "4", "5"]

# TODO: import these from a file instead because these are reused a lot
SYSTEM_PROMPT_FOR_JUDGE = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response based on a given criteria. You'll receive an input with two sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response. 

Carefully read the provided input to understand the task, then assess how well the response fulfills the criteria requirements. If conversation history is present, ensure the response aligns with it; otherwise, evaluate based solely on the instruction. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

USER_PROMPT_FOR_JUDGE = """You will be doing an Informativeness / Helpfulness Assessment of an AI assistant response.

Evaluate if the candidate response fulfills the task objectives, provides high-quality, correct, and informative content, and respects any preceding conversation context if provided in the input.

Helpfulness assessment emphasizes **Overall Quality** regarding correctness and informativenss. 

**Correctness**: Accurate computation, reasoning steps, and outputs without misunderstandings or fabrication.

When assessing informativeness, consider the following aspects:
1. **Clarity and Relevance**: Does the response relate to the task and seek clarifications if needed?
2. **Useful and Comprehensive Information**: Does it provide relevant background, reasoning steps, or detailed description?
3. **Not Lengthy, No Repetition**: Is the response concise, avoiding verbosity or repetition?

Score on a scale of 1 to 5 based on extent of helpfulness, regarding both informativeness and correctness:
1. **Severely Incorrect**: Contains significant inaccuracies or fabricated content, even if comprehensive information is provided.
2. **Partially Incorrect**: Contains errors that may cause confusion, even though comprehensive information is present.
3. **Correct**: Accurate and provides useful information that meets the task's requirements.
4. **Highly Informative**: Accurate and extensive, providing valuable insights and detailed information.
5. **Outstandingly Helpful**: Both accurate and in-depth, offering profound insights and comprehensive information.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

def extract_score_distribution(res, scoring_range):
    return extract_score_distribution_like_activeuf(res, scoring_range)