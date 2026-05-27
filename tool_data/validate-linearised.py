#!/usr/bin/env python3
"""
Validate a linearised dataset against the Apertus format specification.

Reports per-sample issues with counts and optional detailed output for
manual inspection. Supports exporting flagged sample IDs grouped by error
code for downstream processing.

Usage:
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --verbose
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --sample 5
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --dump-idx 0
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --dump-code consecutive_roles --limit 5
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --report-json report.json
    python 07-dataset-aggregation/validate-linearised.py data/07-linearised/MyDataset --flagged-ids flagged.json

Checks performed (error = structural violation, warning = suspicious but valid):

  TOP-LEVEL FIELDS
    [E] missing_field             Required field absent (conversation_id, dataset_source, messages)
    [E] empty_messages            messages is not a non-empty list
    [E] invalid_role              Message role not in {system, developer, user, assistant, tool}
    [E] missing_content           Message has no content field

  SYSTEM MESSAGE (role="system")
    [W] system_position           system message is not at index 0
    [W] duplicate_system          More than one system message found
    [E] system_content_type       content is not a dict
    [E] system_no_text            content missing 'text' field
    [E] system_text_type          content.text is not a string

  DEVELOPER MESSAGE (role="developer")
    [W] no_developer              No developer message found
    [W] duplicate_developer       More than one developer message found
    [E] developer_content_type    content is not a dict
    [E] developer_missing         content missing required field (tools, has_thinking, formatted_tools)
    [E] has_thinking_type         has_thinking is not a bool
    [E] thinking_declared_but_absent  has_thinking=True but no thoughts blocks in any assistant message
    [E] thinking_present_but_undeclared  thoughts blocks exist but has_thinking=False
    [W] tool_use_without_declarations  Assistant makes tool calls but no tools declared in developer message
    [E] tools_json                developer.tools is not valid JSON
    [E] tools_not_list            developer.tools does not parse to a list
    [E] tool_not_dict             Tool entry is not a dict
    [E] tool_missing_field        Tool missing required field (name, description, parameters)
    [E] tool_empty_name           Tool name is empty or not a string
    [W] tool_empty_description    Tool description is empty
    [E] tool_description_type     Tool description is not a string
    [E] tool_params_type          Tool parameters is not a dict (should be JSON Schema object)
    [W] tool_params_no_type       Tool parameters dict missing 'type' field
    [E] formatted_tools_type      formatted_tools is not a string
    [E] formatted_tools_empty     Tools declared but formatted_tools is empty
    [W] formatted_tools_missing_name  formatted_tools does not mention a declared tool name

  USER MESSAGE (role="user")
    [E] user_content_type         content is not a dict or string
    [E] user_no_parts             content missing 'parts' field
    [E] user_parts_type           content.parts is not a list
    [W] user_empty_parts          content.parts is an empty list
    [E] user_part_type            Part entry is not a dict
    [E] user_part_bad_type        Part type is not 'text'
    [E] user_part_no_text         Part missing 'text' field
    [E] user_part_text_type       Part text is not a string

  ASSISTANT MESSAGE (role="assistant")
    [E] assistant_content_type    content is not a dict
    [W] assistant_string          content is a plain string (not structured blocks)
    [E] assistant_no_blocks       content missing 'blocks' field
    [E] assistant_blocks_type     blocks is not a list
    [W] assistant_empty_blocks    blocks list is empty

  ASSISTANT BLOCKS (ordering & fields)
    [E] block_not_dict            Block is not a dict
    [E] invalid_block_type        Block type not in {thoughts, tool_calls, tool_outputs, response}
    [E] thoughts_after_calls      thoughts block after tool_calls without tool_outputs in between
    [W] thoughts_after_response   thoughts block after response block (unusual)
    [W] empty_thoughts            thoughts text is empty
    [E] consecutive_calls         Two tool_calls blocks without tool_outputs in between
    [W] calls_after_response      tool_calls after response block (unusual)
    [E] calls_type                calls field is not a list
    [W] empty_calls               tool_calls block has only placeholder calls
    [E] call_not_dict             Call entry is not a dict
    [E] call_no_name              Call missing 'name' field
    [E] call_empty_name           Call name is empty
    [E] call_no_arguments         Call missing 'arguments' field
    [E] call_arguments_type       Call arguments is not a string
    [W] call_arguments_json       Call arguments is not valid JSON
    [W] call_unknown_tool         Call name not found in declared tools
    [E] orphan_outputs            tool_outputs block without preceding tool_calls
    [E] outputs_type              outputs field is not a list
    [W] empty_outputs             tool_outputs block has only placeholder outputs
    [E] output_not_dict           Output entry is not a dict
    [E] output_no_output          Output missing 'output' field
    [E] output_type               Output 'output' field is not a string
    [E] output_name_type          Output 'name' field is not a string
    [W] calls_outputs_count       Mismatch between number of calls and outputs in adjacent blocks
    [E] response_after_calls      response block after tool_calls without tool_outputs
    [W] empty_response            response text is empty
    [W] dangling_calls            Message ends with tool_calls but no tool_outputs

  TOOL MESSAGE (role="tool")
    [W] standalone_tool           Standalone tool message (linearised format embeds in blocks)

  ROLE SEQUENCE
    [W] system_not_first          system message is not first
    [W] developer_late            developer message not at index 0 or 1
    [W] consecutive_roles         Consecutive user or assistant messages
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set

from datasets import load_from_disk, DatasetDict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ROLES = {"system", "developer", "user", "assistant", "tool"}
VALID_BLOCK_TYPES = {"thoughts", "tool_calls", "tool_outputs", "response"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if longer than max_len."""
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ---------------------------------------------------------------------------
# Issue dataclass
# ---------------------------------------------------------------------------

class Issue:
    """Single validation issue."""

    def __init__(self, level: str, code: str, msg: str, loc: str = ""):
        self.level = level  # "error" or "warning"
        self.code = code
        self.msg = msg
        self.loc = loc  # e.g. "messages[2].blocks[1]"

    def __repr__(self):
        prefix = f"[{self.loc}] " if self.loc else ""
        return f"{self.level.upper()}: {prefix}{self.code}: {self.msg}"


# ---------------------------------------------------------------------------
# Per-sample validation
# ---------------------------------------------------------------------------

def validate_sample(sample: Dict[str, Any]) -> List[Issue]:
    """Validate a single linearised sample. Returns a list of issues."""
    issues: List[Issue] = []

    def err(code, msg, loc=""):
        issues.append(Issue("error", code, msg, loc))

    def warn(code, msg, loc=""):
        issues.append(Issue("warning", code, msg, loc))

    # -- 1. top-level fields -----------------------------------------------
    for field in ("conversation_id", "dataset_source", "messages"):
        if field not in sample:
            err("missing_field", f"Missing required top-level field '{field}'")

    if "messages" not in sample:
        return issues

    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        err("empty_messages", "messages must be a non-empty list")
        return issues

    # -- 2. per-message validation -----------------------------------------
    developer_content = None
    developer_count = 0
    system_count = 0
    has_thinking_in_data = False
    has_tool_use_in_data = False
    declared_tool_names: Set[str] = set()
    roles_sequence: List[str] = []

    for i, msg in enumerate(messages):
        loc = f"messages[{i}]"
        role = msg.get("role")
        if role not in VALID_ROLES:
            err("invalid_role", f"Invalid role '{role}'", loc)
            continue
        roles_sequence.append(role)

        content = msg.get("content")
        if content is None:
            err("missing_content", "Message has no content", loc)
            continue

        # -- system --------------------------------------------------------
        if role == "system":
            system_count += 1
            if system_count > 1:
                warn("duplicate_system", "More than one system message found", loc)
            if i != 0:
                warn("system_position", "system message should be first", loc)
            if not isinstance(content, dict):
                err("system_content_type", f"system content should be dict, got {type(content).__name__}", loc)
            elif "text" not in content:
                err("system_no_text", "system content missing 'text' field", loc)
            elif not isinstance(content["text"], str):
                err("system_text_type", f"system.text should be str, got {type(content['text']).__name__}", loc)

        # -- developer -----------------------------------------------------
        elif role == "developer":
            developer_count += 1
            if developer_count > 1:
                warn("duplicate_developer", "More than one developer message found", loc)
            developer_content = content
            if not isinstance(content, dict):
                err("developer_content_type", f"developer content should be dict, got {type(content).__name__}", loc)
                continue
            for f in ("tools", "has_thinking", "formatted_tools"):
                if f not in content:
                    err("developer_missing", f"developer content missing '{f}'", loc)
            # has_thinking type
            if "has_thinking" in content and not isinstance(content["has_thinking"], bool):
                err("has_thinking_type", f"has_thinking should be bool, got {type(content['has_thinking']).__name__}", loc)
            # tools parsing and per-tool validation
            _validate_developer_tools(content, loc, declared_tool_names, issues)

        # -- user ----------------------------------------------------------
        elif role == "user":
            if not isinstance(content, dict):
                if not isinstance(content, str):
                    err("user_content_type", f"user content should be dict or str, got {type(content).__name__}", loc)
                continue
            if "parts" not in content:
                err("user_no_parts", "user content missing 'parts'", loc)
            elif not isinstance(content["parts"], list):
                err("user_parts_type", f"user.parts should be list, got {type(content['parts']).__name__}", loc)
            else:
                if len(content["parts"]) == 0:
                    warn("user_empty_parts", "user content has empty parts list", loc)
                for pi, part in enumerate(content["parts"]):
                    ploc = f"{loc}.parts[{pi}]"
                    if not isinstance(part, dict):
                        err("user_part_type", "Each user part should be a dict", ploc)
                        continue
                    if part.get("type") != "text":
                        err("user_part_bad_type", f"User part type should be 'text', got '{part.get('type')}'", ploc)
                    if "text" not in part:
                        err("user_part_no_text", "User part missing 'text' field", ploc)
                    elif not isinstance(part["text"], str):
                        err("user_part_text_type", f"User part text should be str, got {type(part['text']).__name__}", ploc)

        # -- assistant -----------------------------------------------------
        elif role == "assistant":
            if not isinstance(content, dict):
                if isinstance(content, str):
                    warn("assistant_string", "Assistant uses string content (not structured blocks)", loc)
                else:
                    err("assistant_content_type", f"assistant content should be dict, got {type(content).__name__}", loc)
                continue
            if "blocks" not in content:
                err("assistant_no_blocks", "assistant content missing 'blocks'", loc)
                continue
            blocks = content["blocks"]
            if not isinstance(blocks, list):
                err("assistant_blocks_type", f"blocks should be list, got {type(blocks).__name__}", loc)
                continue
            if len(blocks) == 0:
                warn("assistant_empty_blocks", "assistant has empty blocks list", loc)

            block_issues = validate_assistant_blocks(blocks, loc, declared_tool_names)
            issues.extend(block_issues)

            for block in blocks:
                if isinstance(block, dict):
                    if block.get("type") == "thoughts":
                        has_thinking_in_data = True
                    elif block.get("type") == "tool_calls":
                        calls = block.get("calls", [])
                        if any(isinstance(c, dict) and c.get("name") for c in calls):
                            has_tool_use_in_data = True

        # -- tool ----------------------------------------------------------
        elif role == "tool":
            warn("standalone_tool", "Standalone tool message (linearised format embeds outputs in blocks)", loc)

    # -- 3. cross-message consistency --------------------------------------

    if developer_content is None:
        warn("no_developer", "No developer message found (linearised format always includes one)")
    elif isinstance(developer_content, dict):
        declared_thinking = developer_content.get("has_thinking", False)
        if isinstance(declared_thinking, bool):
            if has_thinking_in_data and not declared_thinking:
                err("thinking_present_but_undeclared",
                    "Assistant has thoughts blocks but developer.has_thinking is False")
            if declared_thinking and not has_thinking_in_data:
                err("thinking_declared_but_absent",
                    "developer.has_thinking is True but no thoughts blocks in any assistant message")

    if has_tool_use_in_data and not declared_tool_names:
        warn("tool_use_without_declarations",
             "Assistant makes tool calls but no tools are declared in the developer message")

    validate_role_sequence(roles_sequence, issues)

    return issues


def _validate_developer_tools(
    content: Dict[str, Any],
    loc: str,
    declared_tool_names: Set[str],
    issues: List[Issue],
):
    """Validate developer.tools, developer.formatted_tools, and each tool definition."""

    def err(code, msg, tloc=""):
        issues.append(Issue("error", code, msg, tloc or loc))

    def warn(code, msg, tloc=""):
        issues.append(Issue("warning", code, msg, tloc or loc))

    tools_str = content.get("tools", "")
    formatted_tools = content.get("formatted_tools", "")

    # formatted_tools type check
    if "formatted_tools" in content and not isinstance(formatted_tools, str):
        err("formatted_tools_type", f"formatted_tools should be str, got {type(formatted_tools).__name__}")

    # No tools declared — nothing more to check
    if not tools_str:
        return

    # Parse tools JSON
    try:
        parsed_tools = json.loads(tools_str)
    except (json.JSONDecodeError, TypeError) as e:
        err("tools_json", f"developer.tools is not valid JSON: {e}")
        return

    if not isinstance(parsed_tools, list):
        err("tools_not_list", "developer.tools should parse to a list")
        return

    # Per-tool validation
    for ti, tool in enumerate(parsed_tools):
        tloc = f"{loc}.tools[{ti}]"
        if not isinstance(tool, dict):
            err("tool_not_dict", "Each tool should be a dict", tloc)
            continue

        # Required fields
        for tf in ("name", "description", "parameters"):
            if tf not in tool:
                err("tool_missing_field", f"Tool missing '{tf}'", tloc)

        # name
        name = tool.get("name")
        if "name" in tool:
            if not name or not isinstance(name, str):
                err("tool_empty_name", "Tool name is empty or not a string", tloc)
            else:
                declared_tool_names.add(name)

        # description
        if "description" in tool:
            desc = tool["description"]
            if not isinstance(desc, str):
                err("tool_description_type", f"Tool description should be str, got {type(desc).__name__}", tloc)
            elif not desc.strip():
                warn("tool_empty_description", "Tool description is empty", tloc)

        # parameters (should be a dict with JSON Schema structure in linearised format)
        if "parameters" in tool:
            params = tool["parameters"]
            if not isinstance(params, dict):
                err("tool_params_type",
                    f"Tool parameters should be a dict (JSON Schema), got {type(params).__name__}", tloc)
            elif "type" not in params:
                warn("tool_params_no_type", "Tool parameters dict missing 'type' field", tloc)

    # formatted_tools — must be non-empty when tools exist, and should mention tool names
    if isinstance(formatted_tools, str):
        if not formatted_tools.strip():
            err("formatted_tools_empty", "Tools declared but formatted_tools is empty")
        elif declared_tool_names:
            for name in declared_tool_names:
                if name not in formatted_tools:
                    warn("formatted_tools_missing_name",
                         f"formatted_tools does not mention declared tool '{name}'")


def validate_assistant_blocks(
    blocks: List[Dict[str, Any]],
    base_loc: str,
    declared_tool_names: Set[str],
) -> List[Issue]:
    """Validate block types, fields, and ordering within one assistant message.

    Returns list of issues.
    """
    issues: List[Issue] = []

    def err(code, msg, loc=""):
        issues.append(Issue("error", code, msg, loc))

    def warn(code, msg, loc=""):
        issues.append(Issue("warning", code, msg, loc))

    saw_response = False
    expects_output = False
    last_calls_count = 0  # number of calls in last tool_calls block (for count matching)

    for bi, block in enumerate(blocks):
        bloc = f"{base_loc}.blocks[{bi}]"

        if not isinstance(block, dict):
            err("block_not_dict", "Block should be a dict", bloc)
            continue

        btype = block.get("type")
        if btype not in VALID_BLOCK_TYPES:
            err("invalid_block_type", f"Invalid block type '{btype}'", bloc)
            continue

        # -- thoughts ------------------------------------------------------
        if btype == "thoughts":
            if expects_output:
                err("thoughts_after_calls",
                    "thoughts block after tool_calls without tool_outputs in between", bloc)
                expects_output = False
            if saw_response:
                warn("thoughts_after_response", "thoughts block after response block", bloc)
            text = block.get("text", "")
            if not text or (isinstance(text, str) and not text.strip()):
                warn("empty_thoughts", "thoughts block has empty text", bloc)

        # -- tool_calls ----------------------------------------------------
        elif btype == "tool_calls":
            if expects_output:
                err("consecutive_calls",
                    "tool_calls block without preceding tool_outputs (consecutive calls)", bloc)
            if saw_response:
                warn("calls_after_response", "tool_calls after response block", bloc)

            calls = block.get("calls", [])
            if not isinstance(calls, list):
                err("calls_type", f"calls should be list, got {type(calls).__name__}", bloc)
                expects_output = False
            else:
                # Detect placeholder (EMPTY_CALLS from lineariser)
                real_calls = [c for c in calls if isinstance(c, dict) and c.get("name")]
                if not real_calls:
                    warn("empty_calls", "tool_calls block has no actual calls (placeholder)", bloc)
                    expects_output = False
                    last_calls_count = 0
                else:
                    expects_output = True
                    last_calls_count = len(real_calls)

                for ci, call in enumerate(calls):
                    cloc = f"{bloc}.calls[{ci}]"
                    if not isinstance(call, dict):
                        err("call_not_dict", "Each call should be a dict", cloc)
                        continue
                    cname = call.get("name", "")
                    if "name" not in call:
                        err("call_no_name", "Call missing 'name'", cloc)
                    elif not cname:
                        # Empty name in placeholder is expected, skip
                        if real_calls:
                            err("call_empty_name", "Call has empty name", cloc)
                    else:
                        if declared_tool_names and cname not in declared_tool_names:
                            warn("call_unknown_tool",
                                 f"Call to '{cname}' not found in declared tools", cloc)

                    if "arguments" not in call:
                        if cname:  # skip check for placeholder
                            err("call_no_arguments", "Call missing 'arguments'", cloc)
                    elif not isinstance(call["arguments"], str):
                        err("call_arguments_type",
                            f"Call arguments should be str, got {type(call['arguments']).__name__}", cloc)
                    elif call["arguments"].strip():
                        try:
                            json.loads(call["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            warn("call_arguments_json",
                                 f"Call arguments not valid JSON: {truncate(call['arguments'], 60)}", cloc)

        # -- tool_outputs --------------------------------------------------
        elif btype == "tool_outputs":
            if not expects_output:
                err("orphan_outputs", "tool_outputs block without preceding tool_calls", bloc)
            expects_output = False

            outputs = block.get("outputs", [])
            if not isinstance(outputs, list):
                err("outputs_type", f"outputs should be list, got {type(outputs).__name__}", bloc)
            else:
                real_outputs = [o for o in outputs
                                if isinstance(o, dict) and (o.get("output") or o.get("name"))]
                if not real_outputs:
                    warn("empty_outputs",
                         "tool_outputs block has no actual outputs (placeholder)", bloc)
                else:
                    # Count match with preceding tool_calls
                    if last_calls_count and len(real_outputs) != last_calls_count:
                        warn("calls_outputs_count",
                             f"tool_calls had {last_calls_count} calls but tool_outputs has "
                             f"{len(real_outputs)} outputs", bloc)

                for oi, out in enumerate(outputs):
                    oloc = f"{bloc}.outputs[{oi}]"
                    if not isinstance(out, dict):
                        err("output_not_dict", "Each output should be a dict", oloc)
                        continue
                    if "output" not in out:
                        err("output_no_output", "Output missing 'output' field", oloc)
                    elif not isinstance(out["output"], str):
                        err("output_type",
                            f"Output should be str, got {type(out['output']).__name__}", oloc)
                    # Validate name field type when present
                    if "name" in out and not isinstance(out["name"], str):
                        err("output_name_type",
                            f"Output name should be str, got {type(out['name']).__name__}", oloc)
            last_calls_count = 0

        # -- response ------------------------------------------------------
        elif btype == "response":
            if expects_output:
                err("response_after_calls",
                    "response block after tool_calls without tool_outputs", bloc)
                expects_output = False
            saw_response = True
            text = block.get("text", "")
            if not text or (isinstance(text, str) and not text.strip()):
                warn("empty_response", "response block has empty text", bloc)

    if expects_output:
        warn("dangling_calls",
             "Assistant message ends with tool_calls but no tool_outputs", base_loc)

    return issues


def validate_role_sequence(roles: List[str], issues: List[Issue]):
    """Check that the message role sequence is reasonable."""
    if not roles:
        return

    if "system" in roles and roles.index("system") != 0:
        issues.append(Issue("warning", "system_not_first",
                            "system message is not the first message"))

    if "developer" in roles:
        dev_idx = roles.index("developer")
        if dev_idx > 1:
            issues.append(Issue("warning", "developer_late",
                                f"developer message at index {dev_idx}, expected 0 or 1"))

    # Consecutive same-role check (excluding system/developer)
    conv_roles = [r for r in roles if r not in ("system", "developer")]
    for i in range(1, len(conv_roles)):
        if conv_roles[i] == conv_roles[i - 1] and conv_roles[i] in ("user", "assistant"):
            issues.append(Issue("warning", "consecutive_roles",
                                f"Consecutive {conv_roles[i]} messages in conversation"))


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------

def validate_dataset(
    dataset, max_samples: Optional[int] = None, verbose: bool = False
) -> Dict[str, Any]:
    """Validate all samples in a dataset split. Returns summary dict.

    Always collects a mapping of issue codes to affected sample IDs,
    regardless of the verbose flag.
    """
    total = len(dataset)
    if max_samples:
        total = min(total, max_samples)

    error_counts: Counter = Counter()
    warning_counts: Counter = Counter()
    samples_with_errors = 0
    samples_with_warnings = 0
    sample_issues: List[Tuple[int, str, List[Issue]]] = []
    # Always collected: code -> [(idx, conversation_id)]
    flagged_by_code: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

    for idx in range(total):
        sample = dataset[idx]
        issues = validate_sample(sample)
        cid = sample.get("conversation_id", f"idx={idx}")

        errors = [i for i in issues if i.level == "error"]
        warnings = [i for i in issues if i.level == "warning"]

        if errors:
            samples_with_errors += 1
        if warnings:
            samples_with_warnings += 1

        for i in issues:
            if i.level == "error":
                error_counts[i.code] += 1
            else:
                warning_counts[i.code] += 1
            flagged_by_code[i.code].append((idx, cid))

        if issues and verbose:
            sample_issues.append((idx, cid, issues))

        if (idx + 1) % 2000 == 0:
            print(f"  validated {idx + 1}/{total}...")

    return {
        "total": total,
        "samples_with_errors": samples_with_errors,
        "samples_with_warnings": samples_with_warnings,
        "error_counts": error_counts,
        "warning_counts": warning_counts,
        "sample_issues": sample_issues,
        "flagged_by_code": dict(flagged_by_code),
    }


# ---------------------------------------------------------------------------
# Sample dumper
# ---------------------------------------------------------------------------

def dump_sample(sample: Dict[str, Any]):
    """Pretty-print a single linearised sample for manual inspection."""
    print(f"conversation_id: {sample.get('conversation_id', 'N/A')}")
    print(f"dataset_source:  {sample.get('dataset_source', 'N/A')}")
    print(f"original_metadata: {json.dumps(sample.get('original_metadata', {}), indent=2)}")
    print()

    messages = sample.get("messages", [])
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", {})

        if role == "system":
            text = content.get("text", "") if isinstance(content, dict) else str(content)
            print(f"[{i}] SYSTEM: {truncate(text, 120)}")

        elif role == "developer":
            if isinstance(content, dict):
                has_t = content.get("has_thinking", False)
                tools_str = content.get("tools", "")
                ft = content.get("formatted_tools", "")
                n_tools = 0
                if tools_str:
                    try:
                        n_tools = len(json.loads(tools_str))
                    except Exception:
                        pass
                print(f"[{i}] DEVELOPER: has_thinking={has_t}, tools={n_tools} defined")
                if ft:
                    print(f"      formatted_tools: {truncate(ft, 80)}")
            else:
                print(f"[{i}] DEVELOPER: {truncate(str(content), 120)}")

        elif role == "user":
            if isinstance(content, dict) and "parts" in content:
                texts = [p.get("text", "") for p in content["parts"] if isinstance(p, dict)]
                combined = " ".join(texts)
                print(f"[{i}] USER: {truncate(combined, 120)}")
            else:
                print(f"[{i}] USER: {truncate(str(content), 120)}")

        elif role == "assistant":
            if isinstance(content, dict) and "blocks" in content:
                blocks = content["blocks"]
                print(f"[{i}] ASSISTANT ({len(blocks)} blocks):")
                for bi, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        print(f"      [{bi}] <invalid>")
                        continue
                    btype = block.get("type", "?")
                    if btype == "thoughts":
                        text = block.get("text", "")
                        print(f"      [{bi}] thoughts: {truncate(text, 80)}")
                    elif btype == "tool_calls":
                        calls = block.get("calls", [])
                        real_calls = [c for c in calls if isinstance(c, dict) and c.get("name")]
                        for c in real_calls:
                            args_preview = truncate(c.get("arguments", ""), 60)
                            print(f"      [{bi}] call: {c['name']}({args_preview})")
                        if not real_calls:
                            print(f"      [{bi}] tool_calls: (empty placeholder)")
                    elif btype == "tool_outputs":
                        outputs = block.get("outputs", [])
                        real_outputs = [o for o in outputs if isinstance(o, dict) and o.get("output")]
                        for o in real_outputs:
                            print(f"      [{bi}] output: {truncate(o['output'], 80)}")
                        if not real_outputs:
                            print(f"      [{bi}] tool_outputs: (empty placeholder)")
                    elif btype == "response":
                        text = block.get("text", "")
                        print(f"      [{bi}] response: {truncate(text, 100)}")
                    else:
                        print(f"      [{bi}] {btype}: ???")
            else:
                print(f"[{i}] ASSISTANT: {truncate(str(content), 120)}")

        elif role == "tool":
            print(f"[{i}] TOOL: {truncate(str(content), 120)}")
        else:
            print(f"[{i}] {role}: {truncate(str(content), 120)}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate a linearised dataset against the Apertus format spec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python validate-linearised.py data/07-linearised/MyDataset\n"
            "  python validate-linearised.py data/07-linearised/MyDataset --verbose\n"
            "  python validate-linearised.py data/07-linearised/MyDataset --dump-idx 42\n"
            "  python validate-linearised.py data/07-linearised/MyDataset --dump-code consecutive_roles --limit 5\n"
            "  python validate-linearised.py data/07-linearised/MyDataset --report-json report.json\n"
            "  python validate-linearised.py data/07-linearised/MyDataset --flagged-ids flagged.json\n"
        ),
    )
    parser.add_argument("input_path", type=str, help="Path to linearised dataset (load_from_disk format)")
    parser.add_argument("--verbose", action="store_true", help="Show per-sample issues")
    parser.add_argument("--sample", type=int, default=None, help="Only validate first N samples")
    parser.add_argument("--split", type=str, default=None, help="Validate a specific split (default: all)")
    parser.add_argument("--dump-idx", type=int, default=None, help="Pretty-print a single sample by index and exit")
    parser.add_argument("--dump-code", type=str, default=None,
                        help="Dump samples flagged with a specific error/warning code")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max samples to show with --dump-code (default: 10)")
    parser.add_argument("--report-json", type=str, default=None,
                        help="Write validation summary as JSON to this path")
    parser.add_argument("--flagged-ids", type=str, default=None,
                        help="Write {code: [conversation_ids]} mapping as JSON to this path")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: path does not exist: {input_path}")
        sys.exit(1)

    print(f"Loading dataset from: {input_path}")
    try:
        dataset = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error: failed to load dataset: {e}")
        sys.exit(1)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    # -- dump-idx mode -----------------------------------------------------
    if args.dump_idx is not None:
        split = args.split or list(dataset.keys())[0]
        if split not in dataset:
            print(f"Error: split '{split}' not found. Available: {list(dataset.keys())}")
            sys.exit(1)
        ds = dataset[split]
        if args.dump_idx >= len(ds):
            print(f"Error: index {args.dump_idx} out of range (split '{split}' has {len(ds)} samples)")
            sys.exit(1)
        sample = ds[args.dump_idx]
        print(f"\n{'='*60}")
        print(f"Sample {args.dump_idx} from split '{split}'")
        print(f"{'='*60}\n")
        dump_sample(sample)
        print(f"\n{'='*60}")
        print("Validation issues:")
        print(f"{'='*60}")
        issues = validate_sample(sample)
        if issues:
            for iss in issues:
                print(f"  {iss}")
        else:
            print("  No issues found.")
        sys.exit(0)

    # -- batch validation --------------------------------------------------
    splits_to_check = [args.split] if args.split else list(dataset.keys())
    all_clean = True
    all_results: Dict[str, Dict[str, Any]] = {}

    for split_name in splits_to_check:
        if split_name not in dataset:
            print(f"Warning: split '{split_name}' not found. Available: {list(dataset.keys())}")
            continue
        ds = dataset[split_name]
        print(f"\nValidating split '{split_name}' ({len(ds)} samples)...")
        result = validate_dataset(ds, max_samples=args.sample, verbose=args.verbose)
        all_results[split_name] = result

        print(f"\n{'='*60}")
        print(f"  Split: {split_name}")
        print(f"  Samples validated: {result['total']}")
        print(f"  Samples with errors:   {result['samples_with_errors']}")
        print(f"  Samples with warnings: {result['samples_with_warnings']}")

        if result["error_counts"]:
            all_clean = False
            print(f"\n  Errors:")
            for code, count in result["error_counts"].most_common():
                print(f"    {code}: {count}")

        if result["warning_counts"]:
            print(f"\n  Warnings:")
            for code, count in result["warning_counts"].most_common():
                print(f"    {code}: {count}")

        if args.verbose and result["sample_issues"]:
            print(f"\n  Per-sample details (first 20):")
            for idx, cid, sample_iss in result["sample_issues"][:20]:
                print(f"\n    [{idx}] {cid}:")
                for iss in sample_iss:
                    print(f"      {iss}")

        print(f"{'='*60}")

    # -- dump-code mode (runs after validation) ----------------------------
    if args.dump_code:
        code = args.dump_code
        for split_name, result in all_results.items():
            flagged = result["flagged_by_code"].get(code, [])
            if not flagged:
                print(f"\nNo samples flagged with '{code}' in split '{split_name}'.")
                continue
            ds = dataset[split_name]
            show_count = min(args.limit, len(flagged))
            print(f"\n{'='*60}")
            print(f"Samples flagged with '{code}' in split '{split_name}' "
                  f"({show_count} of {len(flagged)} shown):")
            print(f"{'='*60}")
            for idx, cid in flagged[:show_count]:
                print(f"\n--- [{idx}] {cid} ---\n")
                dump_sample(ds[idx])
                relevant = [iss for iss in validate_sample(ds[idx]) if iss.code == code]
                if relevant:
                    print(f"  Issues ({code}):")
                    for iss in relevant:
                        print(f"    {iss}")
                print()

    # -- export: report-json -----------------------------------------------
    if args.report_json:
        report = {}
        for split_name, result in all_results.items():
            report[split_name] = {
                "total": result["total"],
                "samples_with_errors": result["samples_with_errors"],
                "samples_with_warnings": result["samples_with_warnings"],
                "error_counts": dict(result["error_counts"]),
                "warning_counts": dict(result["warning_counts"]),
            }
        report_path = Path(args.report_json)
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport written to: {report_path}")
        except OSError as e:
            print(f"Error: could not write report to {report_path}: {e}", file=sys.stderr)

    # -- export: flagged-ids -----------------------------------------------
    if args.flagged_ids:
        flagged_export: Dict[str, Dict[str, List[str]]] = {}
        for split_name, result in all_results.items():
            split_flagged: Dict[str, List[str]] = {}
            for code, entries in result["flagged_by_code"].items():
                split_flagged[code] = [cid for _, cid in entries]
            if split_flagged:
                flagged_export[split_name] = split_flagged
        flagged_path = Path(args.flagged_ids)
        try:
            flagged_path.parent.mkdir(parents=True, exist_ok=True)
            with open(flagged_path, "w") as f:
                json.dump(flagged_export, f, indent=2)
            print(f"\nFlagged IDs written to: {flagged_path}")
        except OSError as e:
            print(f"Error: could not write flagged IDs to {flagged_path}: {e}", file=sys.stderr)

    if all_clean:
        print("\nAll samples passed validation (no errors).")
    else:
        print("\nValidation found errors. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
