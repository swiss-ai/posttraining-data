"""
Filter verifiable datasets with pass@k labels from a served model.

This script processes datasets in the standardised format, runs the user prompt against
an OpenAI-compatible endpoint, labels pass@1..k, and writes a filtered dataset that
keeps unsolved samples unchanged while keeping solved samples with reasoning removed.

Implements windowed generation with degeneration detection to handle long sequences efficiently.

Usage:
    python filter-verifiable-datasets.py \
        --input /path/to/dataset \
        --output /path/to/output \
        --url http://localhost:30000 \
        --model-name apertus
"""
import json
import asyncio
import argparse
import fcntl
import os
import pickle
import re
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime, UTC
from dataclasses import dataclass, field
from collections import Counter

import aiohttp
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk

# python3 dev/posttraining-data/02-standardisation/filter-verifiable-datasets-labeled.py --input data/sft_1.1/deepmath/ --output data/sft_1.1/deepmath-filtered/ --labeled-output data/sft_1.1/deepmath-labeled/ --model-name apertus --concurrency 6400 --max-new-tokens 4096 --url http://172.28.44.144:30000 --k 4 --temperature 0.7

# =============================================================================
# Degeneration Detection Constants
# =============================================================================

DEGENERATION_THRESHOLD = 0.9  # If 1 - TTR > this, consider it degeneration
MAX_NEW_TOKENS_WINDOW_SIZE = 1024  # Generate in windows of this size
API_ERROR_PREVIEW_CHARS = 500
API_ERROR_LOG_LIMIT = 20
RESPONSE_PREVIEW_CHARS = 2000
RETRY_HTTP_STATUSES = {429, 502, 503, 504}
MAX_API_RETRIES = 4
DEFAULT_CHECKPOINT_INTERVAL = 2000
SAMPLE_INFO_CACHE_VERSION = 2
_api_error_log_count = 0


def log_api_error(sample_idx: int, error_message: str) -> None:
    """Print a capped preview of API errors while the job is still running."""
    global _api_error_log_count
    if _api_error_log_count >= API_ERROR_LOG_LIMIT:
        return

    _api_error_log_count += 1
    print(f"[api-error sample_idx={sample_idx}] {error_message}", flush=True)
    if _api_error_log_count == API_ERROR_LOG_LIMIT:
        print("[api-error] further API error previews suppressed; summary will be printed later", flush=True)


def compute_token_ttr(sequence: list[int], n: int = 1) -> float:
    """Compute Type-Token Ratio for a sequence of token IDs."""
    if len(sequence) < n:
        return 0.0

    ngrams = []
    for i in range(len(sequence) - n + 1):
        ngram = tuple(sequence[i : i + n])
        ngrams.append(ngram)

    if len(ngrams) > 0:
        return len(set(ngrams)) / len(ngrams)

    return 0.0


def check_degeneration(output_ids: list[int]) -> bool:
    """Check if output is degenerating (too many repeated tokens)."""
    return (1 - compute_token_ttr(output_ids)) > DEGENERATION_THRESHOLD


# =============================================================================
# Grammar and Tool Definitions (from dataset.py)
# =============================================================================

ANSWERS_TOOL = {
    "name": "display_answers",
    "description": "Display the answers to the user",

    "parameters": {
        "type": "object",
        "properties": {
            "answers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The answers to the user",
            },
        },
        "required": ["answers"],
    },
}

OPENAI_ANSWERS_TOOL = {
    "type": "function",
    "function": ANSWERS_TOOL,
}

ANSWERS_TOOL_GRAMMAR = """%llguidance {}
start: TEXT? tool_calls <|assistant_end|>

tool_calls: <|tools_prefix|> %json {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "display_answers": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["answers"]
            }
        },
        "required": ["display_answers"]
    }
} <|tools_suffix|>

TEXT: /(.|\n)+/
"""


# =============================================================================
# Scorer
# =============================================================================

def strip_latex_answer_wrappers(expr: str) -> str:
    """Remove common answer-only LaTeX wrappers without changing the answer."""
    expr = str(expr).strip()

    changed = True
    while changed:
        changed = False
        if len(expr) >= 2 and expr.startswith("$") and expr.endswith("$"):
            expr = expr[1:-1].strip()
            changed = True
        if expr.startswith("\\(") and expr.endswith("\\)"):
            expr = expr[2:-2].strip()
            changed = True
        if expr.startswith("\\[") and expr.endswith("\\]"):
            expr = expr[2:-2].strip()
            changed = True

    boxed_match = re.fullmatch(r"\\boxed\{(.+)\}", expr)
    if boxed_match:
        expr = boxed_match.group(1).strip()

    expr = re.sub(r"\\text\{([^{}]*)\}", r"\1", expr)
    expr = expr.replace("\\%", "%")
    return expr.strip()


def parse_number(s: str) -> float | None:
    """Parse common numeric answer formats, including LaTeX fractions."""
    s = strip_latex_answer_wrappers(s).replace(",", "").strip()
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()

    patterns = [
        r"^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$",
        r"^\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$",
    ]
    for pattern in patterns:
        match = re.match(pattern, s)
        if match:
            try:
                num = float(match.group(1))
                denom = float(match.group(2))
                if denom != 0:
                    value = num / denom
                    return value / 100 if is_percent else value
            except ValueError:
                pass

    mixed_patterns = [
        r"^(-?\d+)\s+(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)$",
        r"^(-?\d+)\\frac\{(\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}$",
    ]
    for pattern in mixed_patterns:
        match = re.match(pattern, s)
        if match:
            try:
                whole = float(match.group(1))
                num = float(match.group(2))
                denom = float(match.group(3))
                if denom != 0:
                    value = whole + (num / denom if whole >= 0 else -num / denom)
                    return value / 100 if is_percent else value
            except ValueError:
                pass

    try:
        value = float(s)
        return value / 100 if is_percent else value
    except (ValueError, TypeError):
        return None


def normalize_expression(expr: str) -> str:
    """Normalize a mathematical expression for comparison."""
    expr = strip_latex_answer_wrappers(expr)
    # Remove whitespace
    expr = re.sub(r'\s+', '', expr)
    # Normalize common LaTeX patterns
    expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', expr)
    expr = expr.replace('\\left', '')
    expr = expr.replace('\\right', '')
    expr = expr.replace('\\cdot', '*')
    expr = expr.replace('\\times', '*')
    expr = expr.replace('\\div', '/')
    expr = expr.replace('^', '**')
    # Remove braces used for grouping in LaTeX
    expr = re.sub(r'[{}]', '', expr)
    # Normalize parentheses
    expr = expr.replace('[', '(').replace(']', ')')
    return expr.lower()


def parse_fraction(s: str) -> float | None:
    """Parse a string as a fraction (e.g., '3/4' or '\\frac{3}{4}')."""
    return parse_number(s)


def parse_answer_candidates(answer: Any) -> list[str]:
    """Return alternate acceptable answers from strings or JSON-list strings."""
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(item) for item in answer if item is not None and str(item).strip()]
    if isinstance(answer, tuple):
        return [str(item) for item in answer if item is not None and str(item).strip()]

    text = str(answer).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = [
                    str(item) for item in parsed
                    if item is not None and str(item).strip()
                ]
                if candidates:
                    return candidates
        except json.JSONDecodeError:
            pass

    return [text]


def standalone_judge_single(ground_truth: str, prediction: str, precision: float = 1e-8) -> bool:
    """Judge one ground-truth string against one prediction string."""
    gt = strip_latex_answer_wrappers(ground_truth)
    pred = strip_latex_answer_wrappers(prediction)

    # Exact match
    if gt == pred:
        return True

    # Case-insensitive match
    if gt.lower() == pred.lower():
        return True

    # MCQ match
    if gt.upper() in ["A", "B", "C", "D", "E"]:
        if pred.upper().startswith(gt.upper()):
            return True

    # Numerical match (handles floats and integers with commas)
    gt_num = parse_number(gt)
    pred_num = parse_number(pred)
    if gt_num is not None and pred_num is not None:
        if abs(gt_num - pred_num) <= precision * max(1, abs(gt_num)):
            return True

    # Fraction match
    gt_frac = parse_fraction(gt)
    pred_frac = parse_fraction(pred)
    if gt_frac is not None and pred_frac is not None:
        if abs(gt_frac - pred_frac) <= precision * max(1, abs(gt_frac)):
            return True

    # Try fraction vs number comparison
    if gt_frac is not None:
        try:
            pred_num = float(pred.replace(",", ""))
            if abs(gt_frac - pred_num) <= precision * max(1, abs(gt_frac)):
                return True
        except (ValueError, TypeError):
            pass
    if pred_frac is not None:
        try:
            gt_num = float(gt.replace(",", ""))
            if abs(gt_num - pred_frac) <= precision * max(1, abs(gt_num)):
                return True
        except (ValueError, TypeError):
            pass

    # Normalized expression match (for simple mathematical expressions)
    gt_norm = normalize_expression(gt)
    pred_norm = normalize_expression(pred)
    if gt_norm == pred_norm:
        return True

    return False


def standalone_judge(ground_truth: Any, prediction: Any, precision: float = 1e-8) -> bool:
    """Judge if prediction matches any accepted ground truth."""
    gt_candidates = parse_answer_candidates(ground_truth)
    pred_candidates = parse_answer_candidates(prediction)
    for gt in gt_candidates:
        for pred in pred_candidates:
            if standalone_judge_single(gt, pred, precision):
                return True
    return False


class AutoScoringJudge:
    """Simplified scoring judge for evaluating model outputs against ground truth."""

    def __init__(self, precision: float = 1e-8):
        self.precision = precision

    def judge(self, ground_truth: str, prediction: str) -> bool:
        """Judge if prediction matches ground truth."""
        return standalone_judge(ground_truth, prediction, self.precision)


def get_scorer() -> AutoScoringJudge:
    return AutoScoringJudge()


# =============================================================================
# Tool Call Extraction (from rollout_tasks.py)
# =============================================================================

def extract_answer_from_tool_call(output_text: str) -> str | None:
    """
    Extract answer from tool call in model output.

    Looks for display_answers tool calls in the format:
    <|tools_prefix|>[{"display_answers": {"answers": ["..."]}}]<|tools_suffix|>

    Returns:
        The first answer string if found, None otherwise.
    """
    if "<|tools_prefix|>" not in output_text:
        return None

    try:
        tool_calls_str = output_text.split("<|tools_prefix|>")[1].split("<|tools_suffix|>")[0]
        tool_calls = json.loads(tool_calls_str)
        for tool_call in tool_calls:
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        return None
    except Exception:
        # Try flexible parsing for malformed JSON
        try:
            tool_call = json.loads(output_text.split("<|tools_prefix|>[")[1])
            if "display_answers" in tool_call:
                arguments = tool_call["display_answers"]
                if "answers" in arguments:
                    answers = arguments["answers"]
                    if answers and len(answers) > 0:
                        return str(answers[0])
        except Exception:
            pass
        return None


def get_chat_completions_url(api_url: str) -> str:
    """Normalize an endpoint URL to the OpenAI-compatible chat completions route."""
    api_url = api_url.rstrip("/")
    if api_url.endswith("/chat/completions"):
        return api_url
    if api_url.endswith("/v1"):
        return f"{api_url}/chat/completions"
    return f"{api_url}/v1/chat/completions"


def parse_api_urls(api_url: str) -> list[str]:
    """Parse one URL or a comma-separated list of OpenAI-compatible base URLs."""
    urls = [url.strip() for url in api_url.split(",") if url.strip()]
    if not urls:
        raise ValueError("--url must contain at least one endpoint")
    return urls


def select_api_url(api_urls: list[str], sample_idx: int) -> str:
    """Pick a stable endpoint for a sample so attempts stay on the same worker."""
    return api_urls[sample_idx % len(api_urls)]


def make_auth_headers(api_key: str | None) -> dict[str, str] | None:
    """Return Authorization headers for OpenAI-compatible gateways when a key is set."""
    if not api_key:
        return None
    return {"Authorization": f"Bearer {api_key}"}


async def check_api_reachable(api_url: str, model_name: str, api_key: str | None, timeout: int = 10) -> None:
    """Fail fast by probing the same OpenAI-compatible chat route used for filtering."""
    chat_url = get_chat_completions_url(api_url)
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.post(chat_url, json=payload, headers=make_auth_headers(api_key)) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {body[:API_ERROR_PREVIEW_CHARS]}")

                try:
                    json.loads(body)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Non-JSON response: {body[:API_ERROR_PREVIEW_CHARS]}") from exc

                print(
                    f"API preflight reached {chat_url} (HTTP {response.status}); starting dataset processing.",
                    flush=True,
                )
    except Exception as exc:
        if isinstance(exc, aiohttp.ClientConnectorError):
            hint = (
                "The endpoint is not reachable from this Slurm allocation. "
                "For the SwissAI public gateway use API_URL=https://api.swissai.cscs.ch/v1 with a bearer token. "
                "For a custom/direct SGLang router, pass a URL that is reachable from the filter job's node "
                "and usually set API_AUTH_MODE=none in the launcher."
            )
        else:
            hint = (
                "For the SwissAI public gateway use API_URL=https://api.swissai.cscs.ch/v1 with a bearer token. "
                "For a custom/direct SGLang router, pass its OpenAI-compatible base URL, for example "
                "http://<router-ip>:30000 or http://<router-ip>:30000/v1."
            )
        raise RuntimeError(
            f"API preflight failed for {chat_url}: {type(exc).__name__}: {exc}. "
            f"{hint}"
        ) from exc


async def get_reachable_api_urls(
    api_urls: list[str],
    model_name: str,
    api_key: str | None,
    timeout: int = 10,
) -> list[str]:
    """Probe configured endpoints and keep the reachable ones."""
    reachable_urls = []
    for api_url in api_urls:
        try:
            await check_api_reachable(api_url, model_name, api_key, timeout=timeout)
            reachable_urls.append(api_url)
        except RuntimeError as exc:
            print(f"Warning: dropping unreachable endpoint {api_url}: {exc}", flush=True)

    if not reachable_urls:
        raise RuntimeError("No configured API endpoints are reachable.")

    if len(reachable_urls) != len(api_urls):
        print(
            f"Using {len(reachable_urls)}/{len(api_urls)} reachable API endpoints.",
            flush=True,
        )
    else:
        print(f"Using {len(reachable_urls)} API endpoint(s).", flush=True)

    return reachable_urls


def extract_answer_from_chat_message(message: dict[str, Any]) -> str | None:
    """Extract the first answer from an OpenAI chat message."""
    tool_calls = message.get("tool_calls") or []
    for tool_call in tool_calls:
        function_call = tool_call.get("function", {})
        if function_call.get("name") != "display_answers":
            continue

        arguments = function_call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        answers = arguments.get("answers", []) if isinstance(arguments, dict) else []
        if answers:
            return str(answers[0])

    content = message.get("content") or ""
    if not isinstance(content, str):
        return None

    tool_answer = extract_answer_from_tool_call(content)
    if tool_answer is not None:
        return tool_answer

    try:
        data = json.loads(content)
        json_answer = extract_answer_from_json_value(data)
        if json_answer is not None:
            return json_answer
    except json.JSONDecodeError:
        pass

    return extract_answer_from_text(content)


def extract_answer_from_json_value(value: Any) -> str | None:
    """Extract display_answers payloads from common OpenAI/SGLang JSON shapes."""
    if isinstance(value, dict):
        answers = value.get("answers")
        if isinstance(answers, list) and answers:
            return str(answers[0])

        display_answers = value.get("display_answers")
        if isinstance(display_answers, dict):
            answer = extract_answer_from_json_value(display_answers)
            if answer is not None:
                return answer

        parameters = value.get("parameters")
        if isinstance(parameters, dict):
            answer = extract_answer_from_json_value(parameters)
            if answer is not None:
                return answer

        function_call = value.get("function")
        if isinstance(function_call, dict):
            arguments = function_call.get("arguments")
            if isinstance(arguments, str):
                try:
                    return extract_answer_from_json_value(json.loads(arguments))
                except json.JSONDecodeError:
                    return None
            if isinstance(arguments, dict):
                return extract_answer_from_json_value(arguments)

    if isinstance(value, list):
        for item in value:
            answer = extract_answer_from_json_value(item)
            if answer is not None:
                return answer

    return None


def extract_balanced_boxed(text: str) -> str | None:
    """Extract the content of the last LaTeX \\boxed{...} expression."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    i = start + len(marker)
    depth = 1
    answer_chars: list[str] = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
            answer_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                answer = "".join(answer_chars).strip()
                return answer or None
            answer_chars.append(char)
        else:
            answer_chars.append(char)
        i += 1

    return None


def extract_answer_from_text(content: str) -> str | None:
    """Best-effort fallback when the endpoint ignores tool_choice."""
    text = content.strip()
    if not text:
        return None

    try:
        json_answer = extract_answer_from_json_value(json.loads(text))
        if json_answer is not None:
            return json_answer
    except json.JSONDecodeError:
        pass

    boxed_answer = extract_balanced_boxed(text)
    if boxed_answer:
        return boxed_answer

    patterns = [
        r"(?is)(?:final\s+answer|answer)\s*(?:is|:)\s*([^\n]+)",
        r"(?is)(?:therefore|thus|so),?\s+(?:the\s+)?answer\s*(?:is|:)\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip().strip(" .。")
            if answer:
                return answer

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = nonempty_lines[-1].strip().strip(" .。")
        if 0 < len(last_line) <= 160:
            return last_line

    if len(text) <= 160:
        return text

    return None


def get_chat_message_preview(message: dict[str, Any]) -> str | None:
    """Keep a bounded response preview in labels so extraction bugs are debuggable."""
    preview: dict[str, Any] = {}
    content = message.get("content")
    if isinstance(content, str) and content:
        preview["content"] = content[:RESPONSE_PREVIEW_CHARS]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        preview["tool_calls"] = tool_calls
    if not preview:
        return None
    return json.dumps(preview, ensure_ascii=False)[:RESPONSE_PREVIEW_CHARS]


# =============================================================================
# Sample Processing
# =============================================================================

@dataclass
class SampleInfo:
    """Information about a sample to be processed."""
    idx: int
    has_reasoning: bool
    expected_answer: str
    messages: list[dict[str, str]]
    original_sample: dict


@dataclass
class FilterStats:
    """Statistics about the filtering process."""
    total_samples: int = 0
    samples_with_reasoning: int = 0
    samples_without_reasoning: int = 0
    correct_without_reasoning: int = 0
    reasoning_removed: int = 0
    failed_requests: int = 0
    invalid_tool_calls: int = 0
    degenerated: int = 0
    clipped_length: int = 0
    timeouts: int = 0
    solved_samples: int = 0
    hard_samples_kept: int = 0
    api_failed_samples: int = 0


def default_labeled_output_path(output_path: str) -> str:
    """Return the default path for the labeled dataset."""
    output_dir = Path(output_path)
    return str(output_dir.with_name(f"{output_dir.name}-labeled"))


def extract_expected_answer_from_metadata(sample: dict[str, Any], initial_prompt: dict[str, Any]) -> str | None:
    """Find dataset-provided verifiable answers outside assistant parts."""
    candidates: list[Any] = []
    candidates.append(sample.get("answer"))

    original_metadata = sample.get("original_metadata", {})
    if isinstance(original_metadata, str):
        try:
            original_metadata = json.loads(original_metadata)
        except json.JSONDecodeError:
            original_metadata = {}
    if isinstance(original_metadata, dict):
        candidates.append(original_metadata.get("verifiable_answer"))
        candidates.append(original_metadata.get("answer"))

    prompt_metadata = initial_prompt.get("metadata", {})
    if isinstance(prompt_metadata, str):
        try:
            prompt_metadata = json.loads(prompt_metadata)
        except json.JSONDecodeError:
            prompt_metadata = {}
    if isinstance(prompt_metadata, dict):
        candidates.append(prompt_metadata.get("verifiable_answer"))
        candidates.append(prompt_metadata.get("answer"))

    for candidate in candidates:
        answer_candidates = parse_answer_candidates(candidate)
        if answer_candidates:
            return str(candidate).strip()
    return None


def extract_sample_info(sample: dict[str, Any], idx: int) -> SampleInfo | None:
    """Extract relevant information from a sample for filtering."""
    # Get initial prompt
    initial_prompt = sample.get("initial_prompt", {})
    if isinstance(initial_prompt, str):
        try:
            initial_prompt = json.loads(initial_prompt)
        except json.JSONDecodeError:
            return None

    user_content = initial_prompt.get("content", "")
    if not user_content:
        return None

    # Get system prompt
    system_prompt = sample.get("system_prompt", {})
    if isinstance(system_prompt, str):
        try:
            system_prompt = json.loads(system_prompt)
        except json.JSONDecodeError:
            system_prompt = {}
    system_content = system_prompt.get("content", "")

    # Get conversation branches
    branches = sample.get("conversation_branches", [])
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except json.JSONDecodeError:
            return None

    if not branches:
        return None

    # Check for reasoning and verifiable answers
    has_reasoning = False
    expected_answer = None

    first_branch = branches[0]
    messages = first_branch.get("messages", [])

    for msg in messages:
        if msg.get("role") == "assistant":
            parts = msg.get("parts", [])
            for part in parts:
                part_type = part.get("type", "")
                if part_type == "thought":
                    has_reasoning = True
                elif part_type == "verifiable-responses":
                    answers = part.get("answers", [])
                    if answers:
                        expected_answer = str(answers[0])

    if expected_answer is None:
        expected_answer = extract_expected_answer_from_metadata(sample, initial_prompt)

    # Skip samples without verifiable answers
    if expected_answer is None:
        return None

    # Build chat messages for the model (without reasoning)
    chat_messages = []
    if system_content:
        chat_messages.append({"role": "system", "content": system_content})
    chat_messages.append({"role": "user", "content": user_content})

    return SampleInfo(
        idx=idx,
        has_reasoning=has_reasoning,
        expected_answer=expected_answer,
        messages=chat_messages,
        original_sample=sample,
    )


def remove_reasoning_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove reasoning (thought) parts from a sample."""
    sample = sample.copy()

    # Get conversation branches
    branches = sample.get("conversation_branches", [])
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except json.JSONDecodeError:
            return sample

    # Process each branch
    new_branches = []
    for branch in branches:
        new_branch = branch.copy()
        messages = branch.get("messages", [])
        new_messages = []

        for msg in messages:
            if msg.get("role") == "assistant":
                new_msg = msg.copy()
                parts = msg.get("parts", [])
                # Filter out thought parts
                new_parts = [p for p in parts if p.get("type") != "thought"]
                new_msg["parts"] = new_parts
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)

        new_branch["messages"] = new_messages
        new_branches.append(new_branch)

    sample["conversation_branches"] = new_branches
    return sample


def build_attempt_labels(
    attempts: list[tuple[str | None, bool, str, str | None] | tuple[str | None, bool, str, str | None, str | None]],
    expected_answer: str,
    k: int,
    scorer: AutoScoringJudge,
) -> tuple[list[dict[str, Any]], dict[int, bool], int | None]:
    """Build reusable pass@k labels from the attempts that were actually run."""
    labeled_attempts: list[dict[str, Any]] = []
    pass_by_k: dict[int, bool] = {}
    first_correct_at: int | None = None
    has_passed = False

    for attempt_idx in range(1, k + 1):
        if attempt_idx <= len(attempts):
            attempt = attempts[attempt_idx - 1]
            predicted_answer, failed, finish_reason, error_message = attempt[:4]
            response_preview = attempt[4] if len(attempt) > 4 else None
            is_correct = (
                not failed
                and predicted_answer is not None
                and scorer.judge(expected_answer, predicted_answer)
            )

            if is_correct and first_correct_at is None:
                first_correct_at = attempt_idx

            has_passed = has_passed or is_correct
            labeled_attempts.append({
                "attempt": attempt_idx,
                "prediction": predicted_answer,
                "failed": failed,
                "finish_reason": finish_reason,
                "error": error_message,
                "correct": is_correct,
            })
            if response_preview is not None:
                labeled_attempts[-1]["response_preview"] = response_preview

        pass_by_k[attempt_idx] = has_passed

    return labeled_attempts, pass_by_k, first_correct_at


def add_filter_labels(
    sample: dict[str, Any],
    status: str,
    k: int,
    original_idx: int | None = None,
    has_reasoning: bool = False,
    expected_answer: str | None = None,
    attempts: list[dict[str, Any]] | None = None,
    pass_by_k: dict[int, bool] | None = None,
    first_correct_at: int | None = None,
) -> dict[str, Any]:
    """Attach filtering labels to a sample."""
    sample = sample.copy()
    pass_by_k = pass_by_k or {pass_idx: False for pass_idx in range(1, k + 1)}

    sample["filter_verifiable_status"] = status
    sample["filter_verifiable_original_idx"] = original_idx
    sample["filter_verifiable_has_reasoning"] = has_reasoning
    sample["filter_verifiable_expected_answer"] = expected_answer
    sample["filter_verifiable_k"] = k
    sample["filter_verifiable_pass_at_k"] = pass_by_k.get(k, False)
    sample["filter_verifiable_first_correct_at"] = first_correct_at
    sample["filter_verifiable_attempts_json"] = json.dumps(attempts or [], ensure_ascii=False)

    for pass_idx in range(1, k + 1):
        sample[f"filter_verifiable_pass_at_{pass_idx}"] = pass_by_k.get(pass_idx, False)

    return sample


def sanitize_for_arrow(value: Any) -> Any:
    """Replace invalid Unicode surrogates before handing rows to PyArrow."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [sanitize_for_arrow(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_for_arrow(item) for item in value)
    if isinstance(value, dict):
        return {
            sanitize_for_arrow(key): sanitize_for_arrow(item)
            for key, item in value.items()
        }
    return value


def build_dataset_dict(samples: list[dict[str, Any]], split_name: str | None) -> DatasetDict:
    """Build a DatasetDict while preserving the original split name when available."""
    dataset = Dataset.from_list([sanitize_for_arrow(sample) for sample in samples])
    if split_name:
        return DatasetDict({split_name: dataset})
    return DatasetDict({"train": dataset})


def save_dataset_dict_atomic(dataset_dict: DatasetDict, output_dir: Path) -> None:
    """Save a HuggingFace dataset directory via a temporary sibling directory."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    stamp = f"{os.getpid()}.{int(datetime.now(UTC).timestamp() * 1_000_000)}"
    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp.{stamp}")
    backup_dir = output_dir.with_name(f".{output_dir.name}.previous.{stamp}")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            dataset_dict.save_to_disk(str(tmp_dir))
            if output_dir.exists():
                output_dir.rename(backup_dir)
            tmp_dir.replace(output_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def build_labeled_samples(
    dataset: Dataset,
    sample_info_by_idx: dict[int, SampleInfo],
    labels_by_idx: dict[int, dict[str, Any]],
    k: int,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Build labeled rows from the current completed-result map."""
    labeled_samples = []
    row_indices = indices if indices is not None else range(len(dataset))
    for idx in row_indices:
        sample = dataset[idx]
        sample_dict = dict(sample)
        info = sample_info_by_idx.get(idx)

        if info is None:
            sample_dict = add_filter_labels(
                sample=sample_dict,
                status="no_verifiable_answer",
                k=k,
                original_idx=idx,
            )
        elif info.has_reasoning:
            label_info = labels_by_idx.get(idx, {})
            status = "processed" if idx in labels_by_idx else "not_processed"
            sample_dict = add_filter_labels(
                sample=sample_dict,
                status=status,
                k=k,
                original_idx=idx,
                has_reasoning=True,
                expected_answer=info.expected_answer,
                attempts=label_info.get("attempts"),
                pass_by_k=label_info.get("pass_by_k"),
                first_correct_at=label_info.get("first_correct_at"),
            )
        elif idx in labels_by_idx:
            label_info = labels_by_idx[idx]
            sample_dict = add_filter_labels(
                sample=sample_dict,
                status="processed_no_reasoning",
                k=k,
                original_idx=idx,
                has_reasoning=False,
                expected_answer=info.expected_answer,
                attempts=label_info.get("attempts"),
                pass_by_k=label_info.get("pass_by_k"),
                first_correct_at=label_info.get("first_correct_at"),
            )
        else:
            sample_dict = add_filter_labels(
                sample=sample_dict,
                status="no_reasoning",
                k=k,
                original_idx=idx,
                has_reasoning=False,
                expected_answer=info.expected_answer,
            )

        labeled_samples.append(sample_dict)

    return labeled_samples


def checkpoint_output_path(labeled_output_dir: Path) -> Path:
    """Return the path used for lightweight in-progress labeled checkpoints."""
    return labeled_output_dir.parent / f"{labeled_output_dir.name}.checkpoint"


def load_labeled_checkpoint(checkpoint_dir: Path, k: int) -> dict[int, dict[str, Any]]:
    """Load completed labels from a previous checkpoint, if it has restart metadata."""
    if not checkpoint_dir.exists():
        return {}

    checkpoint_dataset = load_from_disk(str(checkpoint_dir))
    if hasattr(checkpoint_dataset, "keys"):
        split_name = list(checkpoint_dataset.keys())[0]
        checkpoint_dataset = checkpoint_dataset[split_name]

    labels_by_idx: dict[int, dict[str, Any]] = {}
    missing_original_idx = False
    for row in checkpoint_dataset:
        original_idx = row.get("filter_verifiable_original_idx")
        if original_idx is None:
            missing_original_idx = True
            continue

        attempts_json = row.get("filter_verifiable_attempts_json") or "[]"
        try:
            attempts = json.loads(attempts_json)
        except json.JSONDecodeError:
            attempts = []

        labels_by_idx[int(original_idx)] = {
            "attempts": attempts,
            "pass_by_k": {
                pass_idx: bool(row.get(f"filter_verifiable_pass_at_{pass_idx}", False))
                for pass_idx in range(1, k + 1)
            },
            "first_correct_at": row.get("filter_verifiable_first_correct_at"),
        }

    if missing_original_idx:
        print(
            f"Found checkpoint at {checkpoint_dir}, but it does not contain restart indices; ignoring it.",
            flush=True,
        )
        return {}

    return labels_by_idx


def sample_info_cache_path(input_path: str, labeled_output_path: str | None, cache_path: str | None) -> Path:
    """Return the cache path used for extracted sample metadata."""
    if cache_path:
        return Path(cache_path)
    base_dir = Path(labeled_output_path) if labeled_output_path else Path(input_path)
    return base_dir.parent / f".{Path(input_path).name}.sample-info.pkl"


def save_sample_info_cache(
    cache_path: Path,
    sample_infos: list[SampleInfo],
    total_samples: int,
) -> None:
    """Persist extracted sample metadata so restarts avoid a full dataset scan."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.tmp")
    payload = {
        "version": SAMPLE_INFO_CACHE_VERSION,
        "total_samples": total_samples,
        "sample_infos": [
            {
                "idx": info.idx,
                "has_reasoning": info.has_reasoning,
                "expected_answer": info.expected_answer,
                "messages": info.messages,
            }
            for info in sample_infos
        ],
    }
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)


def load_sample_info_cache(cache_path: Path, total_samples: int) -> list[SampleInfo] | None:
    """Load extracted sample metadata if it matches the current dataset length."""
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        print(f"Warning: could not load sample-info cache {cache_path}: {exc}", flush=True)
        return None

    if payload.get("version") != SAMPLE_INFO_CACHE_VERSION or payload.get("total_samples") != total_samples:
        print(f"Warning: ignoring stale sample-info cache {cache_path}", flush=True)
        return None

    return [
        SampleInfo(
            idx=int(item["idx"]),
            has_reasoning=bool(item["has_reasoning"]),
            expected_answer=str(item["expected_answer"]),
            messages=item["messages"],
            original_sample={},
        )
        for item in payload.get("sample_infos", [])
    ]


# =============================================================================
# Async Processing
# =============================================================================

async def process_single_sample(
    session: aiohttp.ClientSession,
    sample_info: SampleInfo,
    sglang_url: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    timeout: int,
    api_key: str | None,
) -> tuple[int, str | None, bool, str, str | None, str | None]:
    """
    Process a single sample asynchronously with windowed generation and degeneration detection.

    Returns:
        Tuple of (sample_idx, predicted_answer, request_failed, finish_reason, error_message, response_preview)
        finish_reason can be: "stop", "length", "degenerating", "error", "timeout"
    """
    async with semaphore:
        try:
            async def _generate():
                last_error_message = None
                for retry_idx in range(MAX_API_RETRIES + 1):
                    async with session.post(
                        get_chat_completions_url(sglang_url),
                        headers=make_auth_headers(api_key),
                        json={
                            "model": model_name,
                            "messages": sample_info.messages,
                            "max_tokens": max_new_tokens,
                            "temperature": temperature,
                            "tools": [OPENAI_ANSWERS_TOOL],
                            "tool_choice": {
                                "type": "function",
                                "function": {"name": "display_answers"},
                            },
                        },
                    ) as response:
                        if response.status != 200:
                            error_body = (await response.text())[:API_ERROR_PREVIEW_CHARS]
                            error_message = f"HTTP {response.status}: {error_body}"
                            last_error_message = error_message
                            if response.status in RETRY_HTTP_STATUSES and retry_idx < MAX_API_RETRIES:
                                await asyncio.sleep(min(2 ** retry_idx, 8))
                                continue
                            log_api_error(sample_info.idx, error_message)
                            return None, True, "error", error_message, None

                        try:
                            result_data = await response.json()
                        except Exception as exc:
                            error_body = (await response.text())[:API_ERROR_PREVIEW_CHARS]
                            error_message = f"Invalid JSON response: {type(exc).__name__}: {exc}; body={error_body}"
                            log_api_error(sample_info.idx, error_message)
                            return None, True, "error", error_message, None

                        choices = result_data.get("choices", [])
                        if not choices:
                            error_message = f"No choices in response: {json.dumps(result_data)[:API_ERROR_PREVIEW_CHARS]}"
                            log_api_error(sample_info.idx, error_message)
                            return None, True, "error", error_message, None

                        choice = choices[0]
                        finish_reason = str(choice.get("finish_reason", "stop"))
                        message = choice.get("message", {})
                        predicted_answer = extract_answer_from_chat_message(message)
                        response_preview = get_chat_message_preview(message)
                        return predicted_answer, False, finish_reason, None, response_preview

                return None, True, "error", last_error_message or "Request failed after retries", None

            predicted_answer, failed, finish_reason, error_message, response_preview = await asyncio.wait_for(_generate(), timeout=timeout)
            return (sample_info.idx, predicted_answer, failed, finish_reason, error_message, response_preview)

        except asyncio.TimeoutError:
            error_message = f"Request timed out after {timeout}s"
            log_api_error(sample_info.idx, error_message)
            return (sample_info.idx, None, True, "timeout", error_message, None)
        except aiohttp.ClientError as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            log_api_error(sample_info.idx, error_message)
            return (sample_info.idx, None, True, "error", error_message, None)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            log_api_error(sample_info.idx, error_message)
            return (sample_info.idx, None, True, "error", error_message, None)


async def process_sample_with_early_stopping(
    session: aiohttp.ClientSession,
    sample_info: SampleInfo,
    sglang_url: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    timeout: int,
    k: int,
    scorer: AutoScoringJudge,
    api_key: str | None,
) -> tuple[int, list[tuple[str | None, bool, str, str | None, str | None]], bool]:
    """
    Process a sample with early stopping - stop as soon as one attempt is correct.

    Returns:
        Tuple of (sample_idx, attempts_list, found_correct)
        where attempts_list contains results for all attempted generations
    """
    attempts: list[tuple[str | None, bool, str, str | None, str | None]] = []
    found_correct = False

    for attempt_idx in range(k):
        sample_idx, predicted_answer, failed, finish_reason, error_message, response_preview = await process_single_sample(
            session=session,
            sample_info=sample_info,
            sglang_url=sglang_url,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            semaphore=semaphore,
            timeout=timeout,
            api_key=api_key,
        )

        attempts.append((predicted_answer, failed, finish_reason, error_message, response_preview))

        # Check if this attempt is correct - if so, stop early
        if not failed and predicted_answer is not None:
            is_correct = scorer.judge(sample_info.expected_answer, predicted_answer)
            if is_correct:
                found_correct = True
                break

    return (sample_info.idx, attempts, found_correct)


async def filter_samples_async(
    sample_infos: list[SampleInfo],
    sglang_urls: list[str],
    model_name: str,
    max_new_tokens: int = 2048,
    k: int = 1,
    temperature: float = 0.0,
    concurrency: int = 64,
    timeout: int = 300,
    api_key: str | None = None,
    checkpoint_interval: int = 0,
    checkpoint_callback: Any | None = None,
) -> dict[int, tuple[list[tuple[str | None, bool, str, str | None]], bool]]:
    """
    Filter samples using SGLang with concurrent requests, degeneration detection,
    and early stopping (stops generating for a sample once one attempt is correct).

    Returns:
        Dictionary mapping sample_idx to (attempts_list, found_correct)
    """
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    scorer = get_scorer()
    results: dict[int, tuple[list[tuple[str | None, bool, str, str | None]], bool]] = {}
    completed_count = 0

    def make_task(info: SampleInfo) -> asyncio.Task:
        return asyncio.create_task(
            process_sample_with_early_stopping(
                session=session,
                sample_info=info,
                sglang_url=select_api_url(sglang_urls, info.idx),
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                semaphore=semaphore,
                timeout=timeout,
                k=k,
                scorer=scorer,
                api_key=api_key,
            )
        )

    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        with tqdm(total=len(sample_infos), desc="Processing samples") as pbar:
            sample_iter = iter(sample_infos)
            active_tasks: set[asyncio.Task] = set()

            for _ in range(max(concurrency, 1)):
                try:
                    active_tasks.add(make_task(next(sample_iter)))
                except StopIteration:
                    break

            while active_tasks:
                done_tasks, active_tasks = await asyncio.wait(
                    active_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done_tasks:
                    sample_idx, attempts, found_correct = await task
                    results[sample_idx] = (attempts, found_correct)
                    completed_count += 1
                    pbar.update(1)
                    if (
                        checkpoint_interval > 0
                        and checkpoint_callback is not None
                        and completed_count % checkpoint_interval == 0
                    ):
                        checkpoint_callback(results, completed_count)

                    try:
                        active_tasks.add(make_task(next(sample_iter)))
                    except StopIteration:
                        pass

    return results


# =============================================================================
# Main Processing
# =============================================================================

async def filter_dataset_async(
    input_path: str,
    output_path: str,
    labeled_output_path: str | None,
    sglang_url: str,
    model_name: str,
    max_new_tokens: int = 2048,
    k: int = 1,
    temperature: float = 0.0,
    concurrency: int = 64,
    timeout: int = 300,
    api_key: str | None = None,
    process_without_reasoning: bool = False,
    dry_run: bool = False,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    sample_info_cache: str | None = None,
) -> FilterStats:
    """Filter the dataset by keeping samples that are unsolved under pass@k."""
    stats = FilterStats()
    if labeled_output_path is None:
        labeled_output_path = default_labeled_output_path(output_path)

    if k > 1 and temperature == 0.0:
        print("Warning: k > 1 but temperature is 0.0. All attempts will likely be identical.")

    sglang_urls = await get_reachable_api_urls(
        parse_api_urls(sglang_url),
        model_name,
        api_key,
        timeout=min(timeout, 10),
    )

    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)

    # Handle DatasetDict
    split_name = None
    if hasattr(dataset, "keys"):
        split_name = list(dataset.keys())[0]
        print(f"Using split: {split_name}")
        dataset = dataset[split_name]

    stats.total_samples = len(dataset)
    print(f"Loaded {stats.total_samples} samples")

    # Extract sample information
    cache_path = sample_info_cache_path(input_path, labeled_output_path, sample_info_cache)
    cached_sample_infos = load_sample_info_cache(cache_path, stats.total_samples)

    if cached_sample_infos is None:
        print("Extracting sample information...")
        sample_infos: list[SampleInfo] = []
        for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Analyzing samples"):
            sample_dict = dict(sample)
            info = extract_sample_info(sample_dict, idx)

            if info is None:
                continue

            sample_infos.append(info)

        save_sample_info_cache(cache_path, sample_infos, stats.total_samples)
        print(f"Saved sample-info cache to {cache_path}")
    else:
        sample_infos = cached_sample_infos
        print(f"Loaded {len(sample_infos):,} cached sample infos from {cache_path}")

    samples_to_process: list[SampleInfo] = []
    sample_info_by_idx: dict[int, SampleInfo] = {}
    for info in sample_infos:
        sample_info_by_idx[info.idx] = info
        if info.has_reasoning:
            stats.samples_with_reasoning += 1
            samples_to_process.append(info)
        else:
            stats.samples_without_reasoning += 1
            if process_without_reasoning:
                samples_to_process.append(info)

    print(f"Found {stats.samples_with_reasoning} samples with reasoning")
    print(f"Found {stats.samples_without_reasoning} samples without reasoning")
    print(f"Skipped {stats.total_samples - len(sample_infos)} samples (no verifiable answer)")

    if dry_run:
        print("\n[DRY RUN] Would process samples but not saving.")
        return stats

    process_target = "eligible samples"

    labeled_output_dir = Path(labeled_output_path)
    output_dir = Path(output_path)
    checkpoint_dir = checkpoint_output_path(labeled_output_dir)
    resumed_labels_by_idx = load_labeled_checkpoint(checkpoint_dir, k)
    if resumed_labels_by_idx:
        before_resume = len(samples_to_process)
        samples_to_process = [
            info for info in samples_to_process
            if info.idx not in resumed_labels_by_idx
        ]
        print(
            f"Resuming from {checkpoint_dir}: loaded {len(resumed_labels_by_idx):,} completed labels; "
            f"{len(samples_to_process):,}/{before_resume:,} samples left to process.",
            flush=True,
        )

    def save_labeled_checkpoint(
        partial_results: dict[int, tuple[list[tuple[str | None, bool, str, str | None]], bool]],
        completed_count: int,
    ) -> None:
        scorer = get_scorer()
        partial_labels_by_idx: dict[int, dict[str, Any]] = dict(resumed_labels_by_idx)
        processed_indices = sorted(partial_results)
        for idx in processed_indices:
            info = sample_info_by_idx[idx]
            result = partial_results[idx]
            attempts, _found_correct = result
            labeled_attempts, pass_by_k, first_correct_at = build_attempt_labels(
                attempts=attempts,
                expected_answer=info.expected_answer,
                k=k,
                scorer=scorer,
            )
            partial_labels_by_idx[info.idx] = {
                "attempts": labeled_attempts,
                "pass_by_k": pass_by_k,
                "first_correct_at": first_correct_at,
            }

        checkpoint_indices = sorted(partial_labels_by_idx)
        labeled_samples = build_labeled_samples(
            dataset=dataset,
            sample_info_by_idx=sample_info_by_idx,
            labels_by_idx=partial_labels_by_idx,
            k=k,
            indices=checkpoint_indices,
        )
        save_dataset_dict_atomic(build_dataset_dict(labeled_samples, split_name), checkpoint_dir)
        checkpoint_metadata = {
            "checkpoint": True,
            "timestamp": datetime.now(UTC).isoformat(),
            "completed_samples": len(partial_labels_by_idx),
            "completed_this_run": completed_count,
            "remaining_at_start": len(samples_to_process),
            "input_path": str(input_path),
            "labeled_output_path": str(labeled_output_path),
            "checkpoint_path": str(checkpoint_dir),
            "k": k,
            "model_name": model_name,
            "sglang_urls": sglang_urls,
        }
        with open(checkpoint_dir / "filter_checkpoint_metadata.json", "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)
        print(
            f"\nCheckpointed {len(labeled_samples):,} completed labeled rows to {checkpoint_dir} "
            f"after {completed_count:,}/{len(samples_to_process):,} processed samples this run.",
            flush=True,
        )

    if not samples_to_process:
        print("No eligible samples to process. Saving labeled and unchanged filtered datasets.")
        results = {}
    else:
        print(f"\nProcessing {len(samples_to_process)} {process_target} (k={k}, temp={temperature})...")
        results = await filter_samples_async(
            sample_infos=samples_to_process,
            sglang_urls=sglang_urls,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
            concurrency=concurrency,
            timeout=timeout,
            api_key=api_key,
            checkpoint_interval=checkpoint_interval,
            checkpoint_callback=save_labeled_checkpoint,
        )

    # Determine which samples to modify
    print("\nCollecting statistics...")
    indices_to_remove_reasoning: set[int] = set()
    solved_indices: set[int] = set()
    api_failed_indices: set[int] = set()
    labels_by_idx: dict[int, dict[str, Any]] = dict(resumed_labels_by_idx)
    api_error_counts: Counter[str] = Counter()
    scorer = get_scorer()

    for info in tqdm(samples_to_process, desc="Processing results"):
        result = results.get(info.idx)
        if result is None:
            continue

        attempts, found_correct = result
        labeled_attempts, pass_by_k, first_correct_at = build_attempt_labels(
            attempts=attempts,
            expected_answer=info.expected_answer,
            k=k,
            scorer=scorer,
        )
        labels_by_idx[info.idx] = {
            "attempts": labeled_attempts,
            "pass_by_k": pass_by_k,
            "first_correct_at": first_correct_at,
        }

    for idx, label_info in labels_by_idx.items():
        info = sample_info_by_idx.get(idx)
        if info is None:
            continue

        attempts = label_info.get("attempts") or []
        pass_by_k = label_info.get("pass_by_k") or {}

        all_requests_failed = bool(attempts) and all(attempt.get("failed") for attempt in attempts)
        if all_requests_failed:
            api_failed_indices.add(idx)

        for attempt in attempts:
            failed = bool(attempt.get("failed"))
            finish_reason = attempt.get("finish_reason")
            error_message = attempt.get("error")
            predicted_answer = attempt.get("prediction")
            if failed:
                if error_message:
                    api_error_counts[str(error_message)] += 1
                if finish_reason == "timeout":
                    stats.timeouts += 1
                else:
                    stats.failed_requests += 1
                continue

            if finish_reason == "degenerating":
                stats.degenerated += 1
            elif finish_reason == "length":
                stats.clipped_length += 1

            if predicted_answer is None:
                stats.invalid_tool_calls += 1

        if pass_by_k.get(k, False):
            solved_indices.add(idx)
            stats.correct_without_reasoning += 1
            if info.has_reasoning:
                indices_to_remove_reasoning.add(idx)

    stats.reasoning_removed = len(indices_to_remove_reasoning)
    stats.solved_samples = len(solved_indices)
    stats.api_failed_samples = len(api_failed_indices)

    if api_error_counts:
        print("\nAPI/request error summary (top 10):")
        for error_message, count in api_error_counts.most_common(10):
            print(f"  {count:,}x {error_message}")

    # Build the labeled and filtered datasets
    print(f"\nBuilding labeled dataset with pass@1..{k} labels...")
    labeled_samples = build_labeled_samples(
        dataset=dataset,
        sample_info_by_idx=sample_info_by_idx,
        labels_by_idx=labels_by_idx,
        k=k,
    )

    hard_indices = set(labels_by_idx) - solved_indices - api_failed_indices
    retained_indices = hard_indices | solved_indices
    stats.hard_samples_kept = len(hard_indices)

    print(
        f"\nBuilding filtered dataset: keeping {len(hard_indices):,} unsolved samples and "
        f"{len(solved_indices):,} solved samples with reasoning removed, "
        f"excluding {len(api_failed_indices):,} API-failed samples."
    )

    filtered_samples = []
    for idx, sample in enumerate(dataset):
        if idx in retained_indices:
            if idx in indices_to_remove_reasoning:
                filtered_samples.append(remove_reasoning_from_sample(dict(sample)))
            else:
                filtered_samples.append(dict(sample))

    # Save the labeled and filtered datasets
    labeled_dataset_dict = build_dataset_dict(labeled_samples, split_name)
    if filtered_samples:
        filtered_dataset_dict = build_dataset_dict(filtered_samples, split_name)
    else:
        filtered_dataset_dict = DatasetDict({split_name or "train": dataset.select([])})

    save_dataset_dict_atomic(labeled_dataset_dict, labeled_output_dir)
    save_dataset_dict_atomic(filtered_dataset_dict, output_dir)

    # Save metadata
    metadata = {
        "processing_log": [{
            "operation": "filter_verifiable_datasets",
            "script": "filter-verifiable-datasets-labeled.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "labeled_output_path": str(labeled_output_path),
            "model_name": model_name,
            "sglang_url": sglang_url,
            "sglang_urls": sglang_urls,
            "k": k,
            "temperature": temperature,
            "timeout": timeout,
            "process_without_reasoning": process_without_reasoning,
            "stats": {
                "total_samples": stats.total_samples,
                "samples_with_reasoning": stats.samples_with_reasoning,
                "samples_without_reasoning": stats.samples_without_reasoning,
                "correct_without_reasoning": stats.correct_without_reasoning,
                "reasoning_removed": stats.reasoning_removed,
                "solved_samples": stats.solved_samples,
                "hard_samples_kept": stats.hard_samples_kept,
                "api_failed_samples": stats.api_failed_samples,
                "failed_requests": stats.failed_requests,
                "invalid_tool_calls": stats.invalid_tool_calls,
                "degenerated": stats.degenerated,
                "clipped_length": stats.clipped_length,
                "timeouts": stats.timeouts,
                "api_error_summary": dict(api_error_counts.most_common(20)),
            }
        }]
    }

    metadata_file = output_dir / "filter_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    labeled_metadata_file = labeled_output_dir / "filter_metadata.json"
    with open(labeled_metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nLabeled dataset saved to {labeled_output_dir}")
    print(f"Labeled metadata saved to {labeled_metadata_file}")
    print(f"Filtered dataset saved to {output_dir}")
    print(f"Metadata saved to {metadata_file}")

    return stats


def print_stats(stats: FilterStats):
    """Print filtering statistics."""
    print("\n" + "=" * 60)
    print("Filtering Statistics")
    print("=" * 60)
    print(f"Total samples:                 {stats.total_samples:,}")
    print(f"Samples with reasoning:        {stats.samples_with_reasoning:,}")
    print(f"Samples without reasoning:     {stats.samples_without_reasoning:,}")
    print("-" * 60)
    print(f"Correct without reasoning:     {stats.correct_without_reasoning:,}")
    print(f"Solved under pass@k:           {stats.solved_samples:,}")
    print(f"Hard samples kept:             {stats.hard_samples_kept:,}")
    print(f"API-failed samples excluded:   {stats.api_failed_samples:,}")
    print(f"Reasoning removed:             {stats.reasoning_removed:,}")
    print(f"Failed requests:               {stats.failed_requests:,}")
    print(f"Timeouts:                      {stats.timeouts:,}")
    print(f"Invalid tool calls:            {stats.invalid_tool_calls:,}")
    print("-" * 60)
    print(f"Degenerated (early stop):      {stats.degenerated:,}")
    print(f"Clipped (max length):          {stats.clipped_length:,}")

    if stats.samples_with_reasoning > 0:
        removal_rate = stats.reasoning_removed / stats.samples_with_reasoning * 100
        print("-" * 60)
        print(f"Reasoning removal rate:        {removal_rate:.1f}%")

    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter verifiable datasets by keeping samples the served model cannot solve under pass@k"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input dataset (standardised format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered dataset",
    )
    parser.add_argument(
        "--labeled-output",
        type=str,
        default=None,
        help="Output path for labeled pass@k dataset (default: <output>-labeled)",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="SGLang server URL or comma-separated worker URLs (e.g., http://host1:8080/v1,http://host2:8080/v1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name served by the API endpoint",
    )
    parser.add_argument(
        "--model-path",
        dest="model_name",
        type=str,
        default=None,
        help="Deprecated alias for --model-name",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Bearer token for authenticated OpenAI-compatible gateways",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum new tokens for generation (default: 2048)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of concurrent requests (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each attempt (including all windowed generations) (default: 300)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of attempts per sample (pass@k) (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0). Should be > 0 if k > 1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze the dataset without processing or saving",
    )
    parser.add_argument(
        "--process-without-reasoning",
        action="store_true",
        help="Also call the endpoint for samples without thought parts and label pass@k",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=(
            "Save the labeled dataset checkpoint every N completed samples "
            f"(default: {DEFAULT_CHECKPOINT_INTERVAL}; set 0 to disable)"
        ),
    )
    parser.add_argument(
        "--sample-info-cache",
        type=str,
        default=None,
        help="Optional path for cached extracted sample metadata to speed up restarts",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_name is None:
        raise SystemExit("--model-name is required")
    api_key = args.api_key or os.environ.get("CSCS_SERVING_API") or os.environ.get("API_KEY")

    stats = asyncio.run(
        filter_dataset_async(
            input_path=args.input,
            output_path=args.output,
            labeled_output_path=args.labeled_output,
            sglang_url=args.url,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            k=args.k,
            temperature=args.temperature,
            concurrency=args.concurrency,
            timeout=args.timeout,
            api_key=api_key,
            process_without_reasoning=args.process_without_reasoning,
            dry_run=args.dry_run,
            checkpoint_interval=args.checkpoint_interval,
            sample_info_cache=args.sample_info_cache,
        )
    )

    print_stats(stats)


if __name__ == "__main__":
    main()
