"""Code-only domain selection based on declared source labels, not prose keywords."""

import re
import json

TERMINAL_SOURCES = {
    "Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k",
    "Lite-Coder/LiteCoder-Terminal-SFT",
    "Lite-Coder/LiteCoder-Terminal-World-Model-SFT",
    "open-thoughts/OpenThoughts-Agent-v1-SFT",
    "ukisai/Qwen3.8-27B-multi-turn-agent-sft",
}

VERSION = "code-only-1"
CODE = {
    "code",
    "coding",
    "programming",
    "code_generation",
    "competitive_programming",
    "software_engineering",
    "software_editing",
    "swe",
    "sql",
    "prime_rl_code",
    "verifiable_code",
    "verifiable_coding",
    "verifiable_coding_problems",
    "verifiable_code_understanding",
    "code_output_prediction",
    "real_world_swe",
    "real_world_software_engineering",
    "code_editing",
    "code_edit",
    "code_understanding",
    "apps",
    "taco",
    "codeforces",
    "codecontests",
    "code_contests",
    "leetcode",
    "opencodereasoning",
    "open_code_reasoning",
    "magicoderevolinstruct",
    "magicoder_evol_instruct",
    "opc_sft_stage2",
    "swe_smith",
    "swe_bench",
    "swebench",
    "swe_rebench",
    "r2e_gym",
    "swe_gym",
    "exercism",
}
NONCODE = {
    "math",
    "mathematics",
    "science",
    "physics",
    "chemistry",
    "biology",
    "general",
    "general_qa",
    "stem",
    "openmathreasoning",
    "numinamath",
    "numinamath_cot",
    "prime_rl_math",
    "verifiable_math",
    "verifiable_math_problems",
}


def label(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def decision(row, entry, file):
    rule = entry.get("code_policy")
    if rule is None or rule == "code_source":
        return "keep"
    if rule == "exclude":
        return "excluded_noncode"
    if rule == "terminal_conversations":
        # These source-selected terminal releases have no per-row domain label.
        # Inspect the actual message container instead of treating missing labels as non-code.
        for key in ("messages", "conversations", "conversation", "trajectory"):
            messages = row.get(key)
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except (ValueError, TypeError):
                    continue
            if isinstance(messages, list) and any(
                isinstance(m, dict)
                and m.get("role", m.get("from", m.get("speaker")))
                in ("assistant", "gpt", "bot")
                and any(
                    m.get(k)
                    for k in ("content", "value", "text", "tool_calls", "function_call")
                )
                for m in messages
            ):
                return "keep"
        return "excluded_unknown"
    domains = [
        label(row[k])
        for k in ("domain", "task_type", "category", "task_category")
        if row.get(k) is not None
    ]
    if any(v in NONCODE for v in domains):
        return "excluded_noncode"
    if any(v in CODE for v in domains):
        return "keep"
    sources = [
        label(row[k])
        for k in (
            "source",
            "dataset",
            "source_dataset",
            "task_source",
            "original_source",
        )
        if isinstance(row.get(k), str)
    ]
    if any(v in NONCODE for v in sources):
        return "excluded_noncode"
    if any(v in CODE for v in sources):
        return "keep"
    task = row.get("task")
    if isinstance(task, str) and re.match(
        r"^(swesmith|swe[-_]?bench|swe[-_]?rebench|r2e[-_]?gym)[-_/]", task.lower()
    ):
        return "keep"
    return "excluded_unknown"


def configure(entry):
    entry = dict(entry)
    repo, original = entry["repo"], entry["category"]
    if original == "benchmarks":
        entry["code_policy_version"] = VERSION
        entry["code_policy"] = "code_source"
        return entry
    entry["category"] = (
        "code_agentic" if original in ("agentic", "code_agentic") else "code_sft"
    )
    entry["code_policy_version"] = VERSION
    entry["code_policy"] = (
        "labels" if original in ("mixed_sft", "agentic") else "code_source"
    )
    if repo in (
        "nvidia/Nemotron-SFT-ARC-AGI-v1",
        "nvidia/Nemotron-Agentic-v1",
        "nvidia/Nemotron-SFT-Agentic-v2",
    ):
        entry["code_policy"] = "exclude"
    if any(
        s in repo
        for s in (
            "Terminal",
            "OpenThoughts-Agent",
            "AgentTrove",
            "Qwen3.8-27B-multi-turn",
        )
    ):
        entry["code_policy"] = "labels"
    if "ColdStartForRL" in repo:
        entry["code_policy"] = "code_source"
    if repo in TERMINAL_SOURCES:
        entry["code_policy"] = "terminal_conversations"
        entry["code_policy_version"] = "terminal-conversations-2"
    from preprocessors import get_processor

    entry = get_processor(repo).configure(entry)
    return entry
