from judges import activeuf

name = "10-General-Quality"
model = activeuf.model

slurm_nodes = activeuf.slurm_nodes
workers = activeuf.workers
nodes_per_worker = activeuf.nodes_per_worker
dp_size = activeuf.dp_size
tp_size = activeuf.tp_size
framework = activeuf.framework
concurrent = activeuf.concurrent
disable_ocf = activeuf.disable_ocf

temperature = activeuf.temperature
max_tokens = activeuf.max_tokens

scoring_range = activeuf.scoring_range
logprobs = activeuf.logprobs
top_logprobs = activeuf.top_logprobs

system_prompt_for_judge = """You are an impartial judge. Your role is to critically evaluate the quality of an AI assistant response. You'll receive an input with two sections, enclosed in tags: <USER_INPUT>...</USER_INPUT> for the task instructions (and any accompanying context, if applicable), and <ASSISTANT_RESPONSE_TO_EVALUATE>...</ASSISTANT_RESPONSE_TO_EVALUATE> for the AI assistant's response. 

Carefully read the provided input to understand the task, then assess how good the response is at addressing the user input prompt. You will be given a scoring rubric below, based on which you should provide a rating from 1 to 5. Your output should only be an integer from 1 to 5. Do not output any additional text or explanations."""

user_prompt_for_judge = """You will be doing a Quality Assessment of an AI assistant response.

Score on a scale of 1 to 5 based on response quality:
1. **Very Poor**: The response fails to address the user's prompt.
2. **Poor**: The response partially addresses the user's prompt but has notable shortcomings.
3. **Acceptable**: The response adequately addresses the user's prompt.
4. **Good**: The response thoroughly addresses the user's prompt.
5. **Excellent**: The response exceptionally addresses the user's prompt.

Here are the user inputs and the AI assistant response to evaluate:

<USER_INPUT>{prompt}</USER_INPUT>

<ASSISTANT_RESPONSE_TO_EVALUATE>{response}</ASSISTANT_RESPONSE_TO_EVALUATE>"""

extract_score_distribution = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution