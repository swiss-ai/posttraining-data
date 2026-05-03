from judges import activeuf

name = "22"
model = "/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-35B-A3B-FP8"

slurm_nodes = 1
workers = 1
nodes_per_worker = 1
dp_size = 1
tp_size = 4
framework = "vllm"
concurrent = activeuf.concurrent
disable_ocf = activeuf.disable_ocf

temperature = activeuf.temperature
max_tokens = activeuf.max_tokens

scoring_range = activeuf.scoring_range
logprobs = activeuf.logprobs
top_logprobs = activeuf.top_logprobs

system_prompt_for_judge = activeuf.system_prompt
user_prompt_for_judge = activeuf.helpfulness_user_prompt

extract_score_distribution = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution