"""
Aspectless judge, just general quality. Prompts are only minimally modified from those for activeuf judges.
"""

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TEMPERATURE = 0.0
MAX_TOKENS = 4096

SCORING_RANGE = ["1", "2", "3", "4", "5"]
LOGPROBS = False
TOP_LOGPROBS = None

SYSTEM_PROMPT_FOR_JUDGE = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt displayed below. You will be given the assistant's answer. Your job is to evaluate how good the response is at addressing the user input prompt.

Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging the assistant's answer. Then consider if the assistant's answer is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's answer when needed. Finally, identify any missing important information in the assistant's answer that would be beneficial to include when responding to the user prompt. 

After providing your explanation, you must output only one of the following choices as your final verdict with a label:
1. The response is very poor: [[1]]
2. The response is poor: [[2]]
3. The response is acceptable: [[3]]
4. The response is good: [[4]]
5. The response is excellent: [[5]]

Example output: "My final verdict is the response is good: [[4]]."""

USER_PROMPT_FOR_JUDGE = """Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

def extract_score_distribution(res, scoring_range):
    import re

    pattern = r"\[\[(" + "|".join(map(re.escape, scoring_range)) + r")\]\]"
    try:
        match = re.search(pattern, res.choices[0].message.content).group(1)
        return {score: 1.0 if score == match else 0.0 for score in scoring_range}
    except Exception as e:
        print(f"⚠️ Warning: Failed to extract score from distribution, returning uniform distribution. Error: {e}")
        return {score: 0.0 for score in scoring_range}
    
def get_score_from_distribution(score_distribution: dict[str, float]) -> float | None:
    if sum(score_distribution.values()) == 0:
        return None
    return float(max(score_distribution.keys(), key=lambda x: score_distribution[x]))