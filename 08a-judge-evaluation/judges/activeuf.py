import math

################################################################################
# ActiveUF Prompts and Settings
################################################################################

model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
slurm_nodes = 4
workers = 2
nodes_per_worker = 2
dp_size = 1
tp_size = 8
framework = "sglang"
concurrent = 32
disable_ocf = True

temperature = 0.0
max_tokens = 1

scoring_range = ["1", "2", "3", "4", "5"]
logprobs = True
top_logprobs = 20

system_prompt = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response based on a given criteria. You'll receive an input with two sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response. 

Carefully read the provided input to understand the task, then assess how well the response fulfills the criteria requirements. If conversation history is present, ensure the response aligns with it; otherwise, evaluate based solely on the instruction. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

helpfulness_user_prompt = """You will be doing an Informativeness / Helpfulness Assessment of an AI assistant response.

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

instruction_following_user_prompt = """You will be doing an Informativeness / Helpfulness Assessment of an AI assistant response.

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

honesty_user_prompt = """You will be doing a Honesty and Uncertainty Expression Assessment of an AI assistant response.

Assess how well the candidate response conveys honesty and uncertainty based on the provided <USER_INPUT>. Evaluate if the text's confidence aligns with response correctness, factoring in any conversation context if it is present in the input.

- **Indicators of Uncertainty**: 
1. Weakeners: e.g., 'I guess,' 'probably.'
2. Refusal to answer.
3. Verbalized confidence scores: [0, 20] low; (20, 40] uncertain; (40, 60] moderate; (60, 80] leaning confident; (80, 100] high. 
- **No uncertainty expression indicate confidence.**
- **Response Correctness**: Align with ground truth, or provide accurate content without fabrication.

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Confidently Incorrect**: Confident but entirely wrong.
2. **Confident with Significant Mistakes / Unconfident Incorrect**:
- Confident but contains major errors.
- Unconfident and entirely wrong.
3. **Uncertain / 'I Don't Know' / Subtle Mistakes**:
- 'I don't know' or declines.
- confident but contains minor errors.
- Unconfident and contains significant mistakes.
4. **Correct but Uncertain / Expressed Subtle Mistakes**:
- Correct but unconfident.
- Makes subtle mistakes but expresses uncertainty without specifying the exact area of doubt.
5. **Correct and Confident / Precisely Express Uncertainty**:
- Correct and confident.
- Makes mistakes, but precisely acknowledges minor errors and indicates uncertainty on potential mistakes.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

truthfulness_user_prompt = """You will be doing a Truthfulness and Hallucination Assessment of an AI assistant response.

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

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

scoring_range = ["1", "2", "3", "4", "5"]

def extract_score_distribution(response, scoring_range: list[str]) -> dict[str, float]:
    try:
        token2logprob = {
            x.token: x.logprob 
            for x in response.choices[0].logprobs.content[0].top_logprobs
        }
        score2prob = {
            score: math.exp(token2logprob.get(score, -float("inf"))) for score in scoring_range
        }
        total_prob = sum(score2prob.values())
        if total_prob == 0:
            return {score: 0.0 for score in scoring_range}
            
        return {score: score2prob[score] / total_prob for score in scoring_range}

    except Exception as e:
        # Fallback if no probabilities are returned
        print(f"⚠️ Warning: Failed to extract probabilities, returning uniform distribution. Error: {e}")
        return {score: 0.0 for score in scoring_range}

def get_score_from_distribution(score_distribution: dict[str, float]) -> float:
    if sum(score_distribution.values()) == 0:
        return None
    return sum(float(score) * weight for score, weight in score_distribution.items())