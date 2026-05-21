from judges import activeuf

name = "24"
model = "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B"

slurm_nodes = 1
workers = 1
nodes_per_worker = 1
dp_size = 1
tp_size = 4
framework = "vllm"
concurrent = activeuf.concurrent
disable_ocf = activeuf.disable_ocf

temperature = activeuf.temperature

# Phase 1: generate judge's own answer (long; needs full answer)
compute_own_answer = True
max_tokens_own_answer = 4096
own_answer_system_prompt = "You are a knowledgeable and helpful assistant. Answer the following user prompt."

# Phase 2: only needs to output the verdict (short)
max_tokens = activeuf.max_tokens

scoring_range = activeuf.scoring_range
logprobs = activeuf.logprobs
top_logprobs = activeuf.top_logprobs

system_prompt_for_judge = """You will be given a user prompt and the answer that you gave to this prompt. Please act as an impartial judge and evaluate the quality of a response provided by an AI assistant. Consider if it is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Also consider creativity and novelty when appropriate, and identify any missing important information.

Below is a scoring rubric from 1 to 5:
1. The response is very poor
2. The response is poor
3. The response is acceptable
4. The response is good
5. The response is excellent

Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

# Note: no {prompt} here — the prompt is already in the conversation as turn 1
user_prompt_for_judge = """Here is another AI assistant's response to the same prompt:

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

extract_score_distribution = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution
