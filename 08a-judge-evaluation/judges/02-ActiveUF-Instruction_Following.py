from judges import activeuf

name = "02-ActiveUF-Instruction_Following"
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

system_prompt_for_judge = activeuf.system_prompt
user_prompt_for_judge = activeuf.instruction_following_user_prompt

extract_score_distribution = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution