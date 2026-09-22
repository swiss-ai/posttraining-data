"""
Filter instruction-following datasets by keeping samples the served model cannot
pass under pass@k according to the local ifevalg verifiers.

The script expects standardised datasets with IFBench-style ground truth, e.g.
`ground_truth={"instruction_id": [...], "kwargs": [...]}` for single-turn rows or
`ground_truth=[{"turn": ..., "instruction_id": [...], "kwargs": [...]}, ...]`
for multi-turn rows.
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import fcntl
import json
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ifevalg.instructions_registry import INSTRUCTION_DICT  # noqa: E402


API_ERROR_PREVIEW_CHARS = 500
API_ERROR_LOG_LIMIT = 20
RETRY_HTTP_STATUSES = {429, 502, 503, 504}
MAX_API_RETRIES = 4
DEFAULT_CHECKPOINT_INTERVAL = 2000
_api_error_log_count = 0


def log_api_error(sample_idx: int, error_message: str) -> None:
    global _api_error_log_count
    if _api_error_log_count >= API_ERROR_LOG_LIMIT:
        return
    _api_error_log_count += 1
    print(f"[api-error sample_idx={sample_idx}] {error_message}", flush=True)
    if _api_error_log_count == API_ERROR_LOG_LIMIT:
        print("[api-error] further API error previews suppressed; summary will be printed later", flush=True)


def parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return default
    return value


def content_from_prompt(prompt: Any) -> str:
    prompt = parse_jsonish(prompt, prompt)
    if isinstance(prompt, dict):
        return str(prompt.get("content") or "")
    if isinstance(prompt, str):
        return prompt
    return ""


def text_from_parts(parts: Any) -> str:
    parts = parse_jsonish(parts, parts)
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    chunks: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            content = part.get("content")
            if content:
                chunks.append(str(content))
        elif isinstance(part, str):
            chunks.append(part)
    return "\n".join(chunks)


def strip_reasoning(response: str) -> str:
    """Best-effort removal of common reasoning wrappers before IF verification."""
    if "<|inner_suffix|>" in response:
        response = response.split("<|inner_suffix|>", 1)[1]
    if "</think>" in response:
        response = response.split("</think>", 1)[1]
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return response.strip()


def sanitize_for_arrow(value: Any) -> Any:
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
    dataset = Dataset.from_list([sanitize_for_arrow(sample) for sample in samples])
    return DatasetDict({split_name or "train": dataset})


def save_dataset_dict_atomic(dataset_dict: DatasetDict, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.with_name(f".{output_dir.name}.lock")
    stamp = f"{os.getpid()}.{int(datetime.now(timezone.utc).timestamp() * 1_000_000)}"
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


@dataclass
class IFSpecGroup:
    turn: int | None
    instruction_ids: list[str]
    kwargs_list: list[dict[str, Any]]


@dataclass
class IFSampleInfo:
    idx: int
    messages: list[dict[str, str]]
    spec_groups: list[IFSpecGroup]
    first_branch_messages: list[dict[str, Any]]
    is_multiturn: bool


@dataclass
class IFFilterStats:
    total_samples: int = 0
    eligible_samples: int = 0
    skipped_no_ground_truth: int = 0
    skipped_bad_ground_truth: int = 0
    failed_requests: int = 0
    verifier_errors: int = 0
    solved_samples: int = 0
    hard_samples_kept: int = 0
    api_failed_samples: int = 0


def normalize_instruction_ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return []


def normalize_kwargs(value: Any, n: int) -> list[dict[str, Any]]:
    if value is None:
        return [{} for _ in range(n)]
    if isinstance(value, str):
        value = parse_jsonish(value, {})
    if isinstance(value, dict):
        return [value for _ in range(n)]
    if isinstance(value, list):
        normalized = [item if isinstance(item, dict) else {} for item in value]
        if len(normalized) < n:
            normalized.extend({} for _ in range(n - len(normalized)))
        return normalized[:n]
    return [{} for _ in range(n)]


def extract_spec_groups(sample: dict[str, Any]) -> list[IFSpecGroup] | None:
    ground_truth = parse_jsonish(sample.get("ground_truth"), None)
    if ground_truth is None:
        original_metadata = parse_jsonish(sample.get("original_metadata"), {})
        if isinstance(original_metadata, dict):
            ground_truth = parse_jsonish(original_metadata.get("ground_truth"), None)

    if ground_truth is None:
        ids = normalize_instruction_ids(sample.get("instruction_id"))
        if not ids:
            return None
        return [IFSpecGroup(turn=None, instruction_ids=ids, kwargs_list=normalize_kwargs(sample.get("kwargs"), len(ids)))]

    if isinstance(ground_truth, dict):
        ids = normalize_instruction_ids(ground_truth.get("instruction_id"))
        if not ids:
            return None
        return [
            IFSpecGroup(
                turn=ground_truth.get("turn"),
                instruction_ids=ids,
                kwargs_list=normalize_kwargs(ground_truth.get("kwargs"), len(ids)),
            )
        ]

    if isinstance(ground_truth, list):
        groups: list[IFSpecGroup] = []
        for item in ground_truth:
            if not isinstance(item, dict):
                continue
            ids = normalize_instruction_ids(item.get("instruction_id"))
            if not ids:
                continue
            groups.append(
                IFSpecGroup(
                    turn=item.get("turn"),
                    instruction_ids=ids,
                    kwargs_list=normalize_kwargs(item.get("kwargs"), len(ids)),
                )
            )
        return groups or None

    return None


def extract_if_sample_info(sample: dict[str, Any], idx: int) -> IFSampleInfo | None:
    spec_groups = extract_spec_groups(sample)
    if not spec_groups:
        return None

    system_content = content_from_prompt(sample.get("system_prompt"))
    user_content = content_from_prompt(sample.get("initial_prompt"))
    if not user_content:
        return None

    branches = parse_jsonish(sample.get("conversation_branches"), [])
    if not isinstance(branches, list):
        branches = []
    first_branch = branches[0] if branches and isinstance(branches[0], dict) else {}
    branch_messages = first_branch.get("messages") or []
    if not isinstance(branch_messages, list):
        branch_messages = []

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})

    return IFSampleInfo(
        idx=idx,
        messages=messages,
        spec_groups=spec_groups,
        first_branch_messages=branch_messages,
        is_multiturn=len(spec_groups) > 1,
    )


def verify_instruction_group(response: str, group: IFSpecGroup) -> tuple[bool, list[dict[str, Any]], int]:
    response = strip_reasoning(response)
    details: list[dict[str, Any]] = []
    all_passed = True
    verifier_errors = 0

    for instruction_id, kwargs in zip(group.instruction_ids, group.kwargs_list):
        passed = False
        error = None
        try:
            instruction_cls = INSTRUCTION_DICT[instruction_id]
            instruction_obj = instruction_cls(instruction_id)
            instruction_obj.build_description(**(kwargs or {}))
            passed = bool(instruction_obj.check_following(response))
        except Exception as exc:
            verifier_errors += 1
            error = f"{type(exc).__name__}: {exc}"
            passed = False

        details.append(
            {
                "instruction_id": instruction_id,
                "passed": passed,
                "error": error,
            }
        )
        if not passed:
            all_passed = False

    return all_passed, details, verifier_errors


def get_chat_completions_url(api_url: str) -> str:
    base = api_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def make_auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def parse_api_urls(value: str) -> list[str]:
    return [url.strip().rstrip("/") for url in value.split(",") if url.strip()]


def select_api_url(api_urls: list[str], sample_idx: int, attempt_idx: int = 0) -> str:
    return api_urls[(sample_idx + attempt_idx) % len(api_urls)]


def extract_chat_content(response_data: dict[str, Any]) -> str | None:
    choices = response_data.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    return content if isinstance(content, str) else None


async def check_api_reachable(api_url: str, model_name: str, api_key: str | None, timeout: int = 10) -> None:
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        try:
            async with session.post(
                get_chat_completions_url(api_url),
                headers=make_auth_headers(api_key),
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                },
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:API_ERROR_PREVIEW_CHARS]
                    raise RuntimeError(f"HTTP {response.status}: {body}")
                print(f"API preflight reached {get_chat_completions_url(api_url)} (HTTP 200); starting dataset processing.")
        except Exception as exc:
            raise RuntimeError(f"Could not reach API at {api_url}: {exc}") from exc


async def get_reachable_api_urls(
    api_urls: list[str],
    model_name: str,
    api_key: str | None,
    timeout: int = 10,
) -> list[str]:
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
        print(f"Using {len(reachable_urls)}/{len(api_urls)} reachable API endpoints.", flush=True)
    else:
        print(f"Using {len(reachable_urls)} API endpoint(s).", flush=True)
    return reachable_urls


async def generate_response(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    sample_idx: int,
    attempt_idx: int,
    sglang_urls: list[str],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    api_key: str | None,
) -> tuple[str | None, bool, str | None]:
    async with semaphore:
        last_error_message = None
        for retry_idx in range(MAX_API_RETRIES + 1):
            try:
                async with session.post(
                    get_chat_completions_url(select_api_url(sglang_urls, sample_idx, attempt_idx + retry_idx)),
                    headers=make_auth_headers(api_key),
                    json={
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                ) as response:
                    if response.status != 200:
                        body = (await response.text())[:API_ERROR_PREVIEW_CHARS]
                        last_error_message = f"HTTP {response.status}: {body}"
                        if response.status in RETRY_HTTP_STATUSES and retry_idx < MAX_API_RETRIES:
                            await asyncio.sleep(min(2 ** retry_idx, 8))
                            continue
                        log_api_error(sample_idx, last_error_message)
                        return None, True, last_error_message

                    try:
                        data = await response.json()
                    except Exception as exc:
                        body = (await response.text())[:API_ERROR_PREVIEW_CHARS]
                        error_message = f"Invalid JSON response: {type(exc).__name__}: {exc}; body={body}"
                        log_api_error(sample_idx, error_message)
                        return None, True, error_message

                    content = extract_chat_content(data)
                    if content is None:
                        error_message = f"No assistant content in response: {json.dumps(data)[:API_ERROR_PREVIEW_CHARS]}"
                        log_api_error(sample_idx, error_message)
                        return None, True, error_message
                    return content, False, None
            except asyncio.TimeoutError:
                error_message = "Request timed out"
                log_api_error(sample_idx, error_message)
                return None, True, error_message
            except aiohttp.ClientError as exc:
                last_error_message = f"{type(exc).__name__}: {exc}"
                if retry_idx < MAX_API_RETRIES:
                    await asyncio.sleep(min(2 ** retry_idx, 8))
                    continue
                log_api_error(sample_idx, last_error_message)
                return None, True, last_error_message
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                log_api_error(sample_idx, error_message)
                return None, True, error_message

        return None, True, last_error_message or "Request failed after retries"


def append_next_user_turn(messages: list[dict[str, str]], branch_messages: list[dict[str, Any]], after_turn: int) -> None:
    user_seen = -1
    for msg in branch_messages:
        role = msg.get("role")
        if role == "user":
            user_seen += 1
            if user_seen == after_turn:
                content = text_from_parts(msg.get("parts")) or str(msg.get("content") or "")
                if content:
                    messages.append({"role": "user", "content": content})
                return


async def run_if_attempt(
    info: IFSampleInfo,
    attempt_idx: int,
    session: aiohttp.ClientSession,
    sglang_urls: list[str],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
    api_key: str | None,
) -> dict[str, Any]:
    messages = [message.copy() for message in info.messages]
    turn_results: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    request_failed = False
    api_error = None
    verifier_errors = 0
    all_passed = True

    for group_idx, group in enumerate(info.spec_groups):
        if group_idx > 0:
            append_next_user_turn(messages, info.first_branch_messages, group_idx - 1)

        response, failed, error_message = await generate_response(
            session=session,
            messages=messages,
            sample_idx=info.idx,
            attempt_idx=attempt_idx + group_idx,
            sglang_urls=sglang_urls,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            semaphore=semaphore,
            api_key=api_key,
        )

        if failed or response is None:
            request_failed = True
            api_error = error_message
            all_passed = False
            turn_results.append(
                {
                    "turn": group.turn,
                    "passed": False,
                    "request_failed": True,
                    "api_error": error_message,
                    "details": [],
                }
            )
            break

        passed, details, error_count = verify_instruction_group(response, group)
        verifier_errors += error_count
        all_passed = all_passed and passed
        clean_response = strip_reasoning(response)
        responses.append({"turn": group.turn, "content": clean_response})
        turn_results.append(
            {
                "turn": group.turn,
                "passed": passed,
                "request_failed": False,
                "api_error": None,
                "details": details,
            }
        )
        messages.append({"role": "assistant", "content": clean_response})

        if not passed:
            break

    return {
        "attempt": attempt_idx,
        "request_failed": request_failed,
        "api_error": api_error,
        "passed": all_passed,
        "verifier_errors": verifier_errors,
        "responses": responses,
        "turn_results": turn_results,
    }


async def process_attempt_batch(
    sample_infos: list[IFSampleInfo],
    attempt_idx: int,
    sglang_urls: list[str],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
    timeout: int,
    api_key: str | None,
) -> dict[int, dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    results: dict[int, dict[str, Any]] = {}

    async def run_one(info: IFSampleInfo) -> tuple[int, dict[str, Any]]:
        attempt_label = await run_if_attempt(
            info=info,
            attempt_idx=attempt_idx,
            session=session,
            sglang_urls=sglang_urls,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            semaphore=semaphore,
            api_key=api_key,
        )
        return info.idx, attempt_label

    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        tasks = [asyncio.create_task(run_one(info)) for info in sample_infos]

        with tqdm(total=len(tasks), desc=f"Generating/verifying IF attempt {attempt_idx}") as pbar:
            for task in asyncio.as_completed(tasks):
                sample_idx, attempt_label = await task
                results[sample_idx] = attempt_label
                pbar.update(1)

    return results


def add_if_filter_labels(
    sample: dict[str, Any],
    status: str,
    k: int,
    original_idx: int,
    attempts: list[dict[str, Any]] | None = None,
    pass_by_k: dict[int, bool] | None = None,
    first_success_at: int | None = None,
) -> dict[str, Any]:
    sample = sample.copy()
    pass_by_k = pass_by_k or {idx: False for idx in range(1, k + 1)}
    sample["filter_if_status"] = status
    sample["filter_if_original_idx"] = original_idx
    sample["filter_if_k"] = k
    sample["filter_if_pass_at_k"] = pass_by_k.get(k, False)
    sample["filter_if_first_success_at"] = first_success_at
    sample["filter_if_attempts_json"] = json.dumps(attempts or [], ensure_ascii=False)
    for pass_idx in range(1, k + 1):
        sample[f"filter_if_pass_at_{pass_idx}"] = pass_by_k.get(pass_idx, False)
    return sample


def build_labeled_samples(
    dataset: Dataset,
    sample_info_by_idx: dict[int, IFSampleInfo],
    attempt_labels_by_idx: dict[int, list[dict[str, Any]]],
    pass_by_k_by_idx: dict[int, dict[int, bool]],
    first_success_at_by_idx: dict[int, int],
    k: int,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    row_indices = indices if indices is not None else range(len(dataset))
    for idx in row_indices:
        sample = dict(dataset[idx])
        info = sample_info_by_idx.get(idx)
        attempts = attempt_labels_by_idx.get(idx, [])
        if info is None:
            status = "not_processable"
        elif attempts:
            status = "processed"
        else:
            status = "not_processed"
        rows.append(
            add_if_filter_labels(
                sample=sample,
                status=status,
                k=k,
                original_idx=idx,
                attempts=attempts,
                pass_by_k=pass_by_k_by_idx.get(idx),
                first_success_at=first_success_at_by_idx.get(idx),
            )
        )
    return rows


def checkpoint_output_path(labeled_output_dir: Path) -> Path:
    return labeled_output_dir.parent / f"{labeled_output_dir.name}.checkpoint"


def load_labeled_checkpoint(
    checkpoint_dir: Path,
    k: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[int, bool]], dict[int, int]]:
    if not checkpoint_dir.exists():
        return {}, {}, {}

    checkpoint_dataset = load_from_disk(str(checkpoint_dir))
    if hasattr(checkpoint_dataset, "keys"):
        split_name = list(checkpoint_dataset.keys())[0]
        checkpoint_dataset = checkpoint_dataset[split_name]

    attempt_labels_by_idx: dict[int, list[dict[str, Any]]] = {}
    pass_by_k_by_idx: dict[int, dict[int, bool]] = {}
    first_success_at_by_idx: dict[int, int] = {}
    missing_original_idx = False

    for row in checkpoint_dataset:
        original_idx = row.get("filter_if_original_idx")
        if original_idx is None:
            missing_original_idx = True
            continue
        idx = int(original_idx)
        attempts = parse_jsonish(row.get("filter_if_attempts_json"), [])
        attempts = attempts if isinstance(attempts, list) else []
        first_success_at = row.get("filter_if_first_success_at")

        is_complete = first_success_at is not None or len(attempts) >= k
        if not is_complete:
            continue

        attempt_labels_by_idx[idx] = attempts
        if first_success_at is not None:
            first_success_at = int(first_success_at)
            first_success_at_by_idx[idx] = first_success_at
            pass_by_k_by_idx[idx] = {pass_idx: first_success_at <= pass_idx for pass_idx in range(1, k + 1)}
        else:
            pass_by_k_by_idx[idx] = {
                pass_idx: bool(row.get(f"filter_if_pass_at_{pass_idx}", False))
                for pass_idx in range(1, k + 1)
            }

    if missing_original_idx:
        print(f"Found checkpoint at {checkpoint_dir}, but it does not contain restart indices; ignoring it.", flush=True)
        return {}, {}, {}

    return attempt_labels_by_idx, pass_by_k_by_idx, first_success_at_by_idx


def default_labeled_output_path(output_path: str) -> str:
    output_dir = Path(output_path)
    return str(output_dir.with_name(f"{output_dir.name}-labeled"))


async def filter_if_dataset_async(
    input_path: str,
    output_path: str,
    labeled_output_path: str | None,
    sglang_url: str,
    model_name: str,
    max_new_tokens: int,
    k: int,
    temperature: float,
    concurrency: int,
    timeout: int,
    api_key: str | None,
    dry_run: bool,
    checkpoint_interval: int,
) -> IFFilterStats:
    if k < 1:
        raise ValueError("--k must be >= 1")
    if labeled_output_path is None:
        labeled_output_path = default_labeled_output_path(output_path)
    if k > 1 and temperature == 0.0:
        print("Warning: k > 1 but temperature is 0.0. All attempts will likely be identical.")

    if dry_run:
        sglang_urls = parse_api_urls(sglang_url)
    else:
        sglang_urls = await get_reachable_api_urls(
            parse_api_urls(sglang_url),
            model_name,
            api_key,
            timeout=min(timeout, 10),
        )

    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    split_name = None
    if hasattr(dataset, "keys"):
        split_name = list(dataset.keys())[0]
        print(f"Using split: {split_name}")
        dataset = dataset[split_name]

    stats = IFFilterStats(total_samples=len(dataset))
    print(f"Loaded {stats.total_samples} samples")

    sample_infos: list[IFSampleInfo] = []
    sample_info_by_idx: dict[int, IFSampleInfo] = {}
    for idx, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Analyzing IF samples"):
        sample_dict = dict(sample)
        if extract_spec_groups(sample_dict) is None:
            stats.skipped_no_ground_truth += 1
            continue
        info = extract_if_sample_info(sample_dict, idx)
        if info is None:
            stats.skipped_bad_ground_truth += 1
            continue
        sample_infos.append(info)
        sample_info_by_idx[idx] = info

    stats.eligible_samples = len(sample_infos)
    print(f"Found {stats.eligible_samples} IF-verifiable samples")
    print(f"Skipped {stats.skipped_no_ground_truth} samples without IF ground truth")
    print(f"Skipped {stats.skipped_bad_ground_truth} samples with malformed IF rows")

    if dry_run:
        print("\n[DRY RUN] Would process samples but not save.")
        return stats

    labeled_output_dir = Path(labeled_output_path)
    output_dir = Path(output_path)
    checkpoint_dir = checkpoint_output_path(labeled_output_dir)
    (
        resumed_attempt_labels_by_idx,
        resumed_pass_by_k_by_idx,
        resumed_first_success_at_by_idx,
    ) = load_labeled_checkpoint(checkpoint_dir, k)

    samples_to_process = sample_infos
    if resumed_attempt_labels_by_idx:
        before_resume = len(samples_to_process)
        samples_to_process = [info for info in samples_to_process if info.idx not in resumed_attempt_labels_by_idx]
        print(
            f"Resuming from {checkpoint_dir}: loaded {len(resumed_attempt_labels_by_idx):,} completed labels; "
            f"{len(samples_to_process):,}/{before_resume:,} samples left to process.",
            flush=True,
        )

    attempt_labels_by_idx: dict[int, list[dict[str, Any]]] = {info.idx: [] for info in samples_to_process}
    attempt_labels_by_idx.update(resumed_attempt_labels_by_idx)
    pass_by_k_by_idx: dict[int, dict[int, bool]] = {info.idx: {} for info in samples_to_process}
    pass_by_k_by_idx.update(resumed_pass_by_k_by_idx)
    first_success_at_by_idx: dict[int, int] = dict(resumed_first_success_at_by_idx)
    solved_indices: set[int] = set(first_success_at_by_idx)
    api_error_counts: Counter[str] = Counter()
    completed_sample_count = 0

    def update_pass_labels(info: IFSampleInfo, completed_attempt_idx: int) -> None:
        first_success_at = first_success_at_by_idx.get(info.idx)
        pass_by_k = pass_by_k_by_idx.setdefault(info.idx, {})
        for pass_idx in range(1, k + 1):
            if first_success_at is not None and first_success_at <= pass_idx:
                pass_by_k[pass_idx] = True
            elif pass_idx <= completed_attempt_idx:
                pass_by_k[pass_idx] = False

    def record_attempt(info: IFSampleInfo, attempt_label: dict[str, Any], attempt_idx: int) -> None:
        if attempt_label.get("request_failed"):
            stats.failed_requests += 1
            if attempt_label.get("api_error"):
                api_error_counts[str(attempt_label["api_error"])] += 1
        stats.verifier_errors += int(attempt_label.get("verifier_errors") or 0)
        if attempt_label.get("passed"):
            solved_indices.add(info.idx)
            first_success_at_by_idx[info.idx] = attempt_idx
        attempt_labels_by_idx[info.idx].append(attempt_label)
        update_pass_labels(info, attempt_idx)

    def save_labeled_checkpoint(completed_count: int) -> None:
        processed_indices = sorted(
            idx for idx, attempts in attempt_labels_by_idx.items()
            if attempts and (first_success_at_by_idx.get(idx) is not None or len(attempts) >= k)
        )
        labeled_samples = build_labeled_samples(
            dataset=dataset,
            sample_info_by_idx=sample_info_by_idx,
            attempt_labels_by_idx=attempt_labels_by_idx,
            pass_by_k_by_idx=pass_by_k_by_idx,
            first_success_at_by_idx=first_success_at_by_idx,
            k=k,
            indices=processed_indices,
        )
        save_dataset_dict_atomic(build_dataset_dict(labeled_samples, split_name), checkpoint_dir)
        metadata = {
            "checkpoint": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_attempt_results": sum(len(attempts) for attempts in attempt_labels_by_idx.values()),
            "completed_samples_this_run": completed_count,
            "remaining_at_start": len(samples_to_process),
            "input_path": str(input_path),
            "labeled_output_path": str(labeled_output_path),
            "checkpoint_path": str(checkpoint_dir),
            "k": k,
            "model_name": model_name,
            "sglang_urls": sglang_urls,
        }
        with open(checkpoint_dir / "filter_checkpoint_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(
            f"\nCheckpointed {len(labeled_samples):,} completed labeled rows to {checkpoint_dir} "
            f"after {completed_count:,} completed samples this run.",
            flush=True,
        )

    print(f"\nGenerating and verifying IF responses for {len(samples_to_process)} samples (k={k}, temp={temperature})...")
    chunk_size = max(checkpoint_interval, concurrency * 4, 1)
    total_chunks = (len(samples_to_process) + chunk_size - 1) // chunk_size
    for chunk_start in range(0, len(samples_to_process), chunk_size):
        chunk = samples_to_process[chunk_start:chunk_start + chunk_size]
        pending_chunk = chunk
        print(f"\nProcessing chunk {chunk_start // chunk_size + 1}/{total_chunks} ({len(chunk)} samples, k={k})...", flush=True)

        for attempt_idx in range(1, k + 1):
            if not pending_chunk:
                for info in chunk:
                    update_pass_labels(info, attempt_idx)
                continue
            results = await process_attempt_batch(
                sample_infos=pending_chunk,
                attempt_idx=attempt_idx,
                sglang_urls=sglang_urls,
                model_name=model_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                concurrency=concurrency,
                timeout=timeout,
                api_key=api_key,
            )
            for info in pending_chunk:
                attempt_label = results.get(info.idx)
                if attempt_label is None:
                    attempt_label = {
                        "attempt": attempt_idx,
                        "request_failed": True,
                        "api_error": "missing attempt result",
                        "passed": False,
                        "verifier_errors": 0,
                        "responses": [],
                        "turn_results": [],
                    }
                record_attempt(info, attempt_label, attempt_idx)
            for info in chunk:
                update_pass_labels(info, attempt_idx)
            pending_chunk = [info for info in pending_chunk if info.idx not in solved_indices]

        completed_sample_count += len(chunk)
        if checkpoint_interval > 0 and completed_sample_count % checkpoint_interval == 0:
            save_labeled_checkpoint(completed_sample_count)

    if checkpoint_interval > 0 and completed_sample_count % checkpoint_interval != 0:
        save_labeled_checkpoint(completed_sample_count)

    stats.solved_samples = len(solved_indices)
    api_failed_indices = {
        idx for idx, attempts in attempt_labels_by_idx.items()
        if attempts and all(attempt.get("request_failed") for attempt in attempts)
    }
    stats.api_failed_samples = len(api_failed_indices)

    if api_error_counts:
        print("\nAPI/request error summary (top 10):")
        for error_message, count in api_error_counts.most_common(10):
            print(f"  {count:,}x {error_message}")

    print(f"\nBuilding labeled dataset with pass@1..{k} labels...")
    labeled_indices = sorted(attempt_labels_by_idx)
    labeled_samples = build_labeled_samples(
        dataset=dataset,
        sample_info_by_idx=sample_info_by_idx,
        attempt_labels_by_idx=attempt_labels_by_idx,
        pass_by_k_by_idx=pass_by_k_by_idx,
        first_success_at_by_idx=first_success_at_by_idx,
        k=k,
        indices=labeled_indices,
    )
    save_dataset_dict_atomic(build_dataset_dict(labeled_samples, split_name), labeled_output_dir)

    hard_indices = [
        idx for idx in labeled_indices
        if not pass_by_k_by_idx.get(idx, {}).get(k, False)
    ]
    stats.hard_samples_kept = len(hard_indices)
    hard_samples = [dict(dataset[idx]) for idx in hard_indices]
    save_dataset_dict_atomic(build_dataset_dict(hard_samples, split_name), output_dir)

    print("\nFiltering complete!")
    print(f"Total samples: {stats.total_samples}")
    print(f"Eligible IF samples: {stats.eligible_samples}")
    print(f"Solved under pass@{k}: {stats.solved_samples}")
    print(f"Hard samples kept: {stats.hard_samples_kept}")
    print(f"API failed samples: {stats.api_failed_samples}")
    print(f"Verifier errors: {stats.verifier_errors}")
    print(f"Labeled output: {labeled_output_dir}")
    print(f"Filtered output: {output_dir}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter IF datasets by local ifevalg pass@k verification")
    parser.add_argument("--input", required=True, help="Input HF dataset path")
    parser.add_argument("--output", required=True, help="Output path for hard/filtered dataset")
    parser.add_argument("--labeled-output", help="Output path for labeled dataset")
    parser.add_argument("--url", required=True, help="Comma-separated OpenAI-compatible base URLs")
    parser.add_argument("--model-name", required=True, help="Model name for chat completions")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    asyncio.run(
        filter_if_dataset_async(
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
            api_key=args.api_key,
            dry_run=args.dry_run,
            checkpoint_interval=args.checkpoint_interval,
        )
    )


if __name__ == "__main__":
    main()
