# Judges

## Overview

| #  | Name | LLM | Scoring methodology |
|----|------|-----|---------------------|
| 00 | 00-ActiveUF | Qwen3-235B-A22B | Mean of scores from judges 01–04 |
| 01 | 01-ActiveUF-Helpfulness | Qwen3-235B-A22B | Logprobs-based scoring |
| 02 | 02-ActiveUF-Instruction_Following | Qwen3-235B-A22B | Logprobs-based scoring |
| 03 | 03-ActiveUF-Honesty | Qwen3-235B-A22B | Logprobs-based scoring |
| 04 | 04-ActiveUF-Truthfulness | Qwen3-235B-A22B | Logprobs-based scoring |
| 10 | 10-General_Quality | Qwen3-235B-A22B | Logprobs-based scoring |
| 11 | 11-SwissAI_Charter | Qwen3-235B-A22B | Logprobs-based scoring |
| 20 | 20-Qwen3.5_35B-Helpfulness | Qwen3.5-35B-A3B-FP8 | Logprobs-based scoring |
| 21 | 21-Qwen3.6_27B-Helpfulness | Qwen3.6-27B | Logprobs-based scoring |
| 30 | 30-ArenaHard-regex | Qwen3-235B-A22B | Regex extraction, random fallback |
| 31 | 31-Qwen3.6_27B-ArenaHard-regex | Qwen3.6-27B | Regex extraction, random fallback |
| 32 | 32-Qwen3.6_27B-ArenaHard | Qwen3.6-27B | Two-phase: own answer then logprobs |

## Judging methodolodgies

**Single-turn (all except judge 32)**: Messages are sent to the judge only once per input prompt.

**Multi-turn (judge 32)**: The judge is prompted to generate its own answer to the prompt in the first phase, then to judge the quality of a given answer while taking its own answer into consideration in the second phase.

## Scoring methodologies

**Logprobs-based** (judges 01–04, 10, 11, 20, 21, 32): the model generates one token; its logprobs over the scoring tokens (`["1"…"5"]`) are softmax-normalised into a probability distribution. `get_score_from_distribution` returns the weighted mean.

**Regex-based** (judges 30–31): the model generates a full text response; the score is extracted with the pattern `\[\[([1-5])\]\]`. Returns 1.0 on match, or a random value from `scoring_range` as fallback.

**Composite** (judge 00): loads the `score_distribution` columns produced by judges 01–04, averages the numeric scores, and re-emits a single value.

## Judge module interface

Each judge is a plain Python file. `src/judge.py` and `init_judge_server.py` import it dynamically; every name below must be a module-level variable or function.

### Required fields

| Name | Type | Purpose |
|------|------|---------|
| `name` | `str` | Short identifier, e.g. `"06"` |
| `model` | `str` | Model name passed to the OpenAI-compatible server |
| `system_prompt_for_judge` | `str` | System prompt sent to the judge model |
| `user_prompt_for_judge` | `str` | User prompt template; must contain `{prompt}` and `{response}` placeholders (single-turn judges) or only `{response}` (two-phase judges where turn 1 is the prompt) |
| `temperature` | `float` | Sampling temperature (use `0.0` for deterministic scoring) |
| `max_tokens` | `int` | `1` for logprob-based judges, larger for text-generation judges |
| `logprobs` | `bool` | Whether to request token logprobs from the API |
| `top_logprobs` | `int \| None` | Number of top logprobs to return (e.g. `20`); `None` when `logprobs=False` |
| `scoring_range` | `list[str]` | Token strings the judge scores over, e.g. `["1","2","3","4","5"]` |
| `concurrent` | `int` | Max concurrent API requests (32 is typical) |

### Serving fields (read by `init_judge_server.py`)

| Name | Type | Purpose |
|------|------|---------|
| `slurm_nodes` | `int` | Total SLURM nodes for the serving job |
| `workers` | `int` | Number of vLLM/SGLang worker processes |
| `nodes_per_worker` | `int` | Nodes allocated per worker |
| `dp_size` | `int` | Data-parallel degree |
| `tp_size` | `int` | Tensor-parallel degree |
| `framework` | `str` | `"sglang"` or `"vllm"` |
| `disable_ocf` | `bool` | Disable OCF scheduling (set `True` for large models) |

### Required functions

```python
def extract_score_distribution(response, scoring_range) -> dict[str, float]:
    """
    response: full chat completion object from the OpenAI API
    scoring_range: list of score token strings, e.g. ["1","2","3","4","5"]
    Returns a dict mapping each token string to a probability (need not sum to 1).
    On failure, return a uniform or zero distribution rather than raising.
    """
```

### Optional fields and functions

| Name | Default | Purpose |
|------|---------|---------|
| `get_score_from_distribution(dist) -> float \| None` | weighted mean | Converts a score distribution dict to a single float for metric computation |
| `compute_own_answer` | `False` | Set `True` to enable phase-1 own-answer generation |
| `own_answer_system_prompt` | `None` | System prompt for phase-1 generation (falls back to `system_prompt_for_judge`) |
| `max_tokens_own_answer` | `max_tokens` | Token budget for phase-1 generation |

## Adding a new judge

1. **Pick a number** — use the next available two-digit prefix in the appropriate group (00–09 ActiveUF variants, 10–19 general quality, 20–29 small-model helpfulness, 30–39 ArenaHard-style).

2. **Create `judges/NN-My-Judge.py`** with `name = "NN-My-Judge"`.

3. **Fill in the interface** — either import shared pieces from `activeuf` or write from scratch. See the skeleton below.

4. **Implement `extract_score_distribution`** — for logprob judges, delegate to `activeuf.extract_score_distribution`; for text judges, write a regex extractor.

5. **Test** — run on a small benchmark (e.g. 10 rows) before submitting a full job:

   ```bash
   # Start a server manually, then:
   python -m src.judge \
       --input-dir benchmarks/JudgeBench-gpt/1-reformatted \
       --output-dir /tmp/test-judge \
       --judge-cfg-path judges/NN-My-Judge.py \
       --server-url http://localhost:30000
   ```

### Skeleton — logprob judge re-using ActiveUF infrastructure

```python
import activeuf

name = "06-MyJudge"

# Serving (copy from activeuf or override for a different model)
model             = activeuf.model
slurm_nodes       = activeuf.slurm_nodes
workers           = activeuf.workers
nodes_per_worker  = activeuf.nodes_per_worker
dp_size           = activeuf.dp_size
tp_size           = activeuf.tp_size
framework         = activeuf.framework
disable_ocf       = activeuf.disable_ocf

# Generation
temperature  = 0.0
max_tokens   = 1
logprobs     = True
top_logprobs = 20
concurrent   = 32
scoring_range = ["1", "2", "3", "4", "5"]

# Prompts — swap in your own or reuse activeuf's
system_prompt_for_judge = activeuf.system_prompt
user_prompt_for_judge   = activeuf.helpfulness_user_prompt  # must contain {prompt} and {response}

# Scoring functions
extract_score_distribution  = activeuf.extract_score_distribution
get_score_from_distribution = activeuf.get_score_from_distribution
```

### Skeleton — text/regex judge

```python
import re
import activeuf

name = "33-MyRegexJudge"

model             = activeuf.model
slurm_nodes       = activeuf.slurm_nodes
workers           = activeuf.workers
nodes_per_worker  = activeuf.nodes_per_worker
dp_size           = activeuf.dp_size
tp_size           = activeuf.tp_size
framework         = activeuf.framework
disable_ocf       = activeuf.disable_ocf

temperature   = 0.0
max_tokens    = 4096
logprobs      = False
top_logprobs  = None
concurrent    = 32
scoring_range = ["1", "2", "3", "4", "5"]

system_prompt_for_judge = "..."
user_prompt_for_judge   = "... {prompt} ... {response} ..."  # must contain {prompt} and {response}


def extract_score_distribution(response, scoring_range):
    text = response.choices[0].message.content or ""
    m = re.search(r"\[\[([1-5])\]\]", text)
    if m:
        return {s: (1.0 if s == m.group(1) else 0.0) for s in scoring_range}
    # Fallback: uniform distribution signals "no score extracted"
    return {s: 1.0 / len(scoring_range) for s in scoring_range}


def get_score_from_distribution(dist):
    # For regex judges: return the argmax token as a float, or None if uniform
    total = sum(dist.values())
    if total == 0:
        return None
    best = max(dist, key=dist.get)
    # Return None if all values equal (no clear winner)
    if len(set(round(v, 9) for v in dist.values())) == 1:
        return None
    return float(best)
```