import math
from pathlib import Path
import importlib.util

import httpx
from openai import AsyncOpenAI


def create_async_openai_client(
    base_url: str,
    max_connections: int,
    api_key: str = "EMPTY",
    timeout_s: float = 7200.0,
) -> tuple[AsyncOpenAI, httpx.AsyncClient]:
    """Build an AsyncOpenAI client backed by a shared httpx.AsyncClient pool."""
    n = max(1, max_connections)
    limits = httpx.Limits(
        max_connections=n,
        max_keepalive_connections=n,
    )
    http_client = httpx.AsyncClient(
        limits=limits,
        timeout=httpx.Timeout(timeout_s),
    )
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=http_client,
    )
    return client, http_client

def load_module(module_path: str):
    """Load a Python module from a file path."""
    path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def stringify_prompt(prompt: str | list[dict]) -> str:
    """Converts a single-turn or multi-turn conversation into a string.
    
    Each prompt should either already be a string or a list of dicts with 'role', 'content' keys.
    """
    if isinstance(prompt, str):
        return prompt.strip()

    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("Prompt is empty")

        if len(prompt) == 1:
            return prompt[0].get('content', '').strip()

        stringified_prompt = "### CONVERSATION HISTORY ###\n"
        for turn in prompt[:-1]:
            role = turn.get('role', '').upper()
            content = turn.get('content', '').strip()
            stringified_prompt += f"[{role}]: {content}\n\n"

        stringified_prompt += "### FINAL INSTRUCTION ###\n"
        stringified_prompt += prompt[-1].get('content', '').strip()
        return stringified_prompt

    raise ValueError(f"Invalid prompt type: {type(prompt)}")

def extract_score_distribution_like_activeuf(res, scoring_range: list[str]) -> dict[str, float]:
    try:
        token2logprob = {
            x.token: x.logprob 
            for x in res.choices[0].logprobs.content[0].top_logprobs
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

