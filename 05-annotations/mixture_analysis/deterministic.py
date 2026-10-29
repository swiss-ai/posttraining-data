"""Conservative deterministic annotation of normalized SFT records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections import defaultdict
from pathlib import PurePath
from typing import Any, Iterable

from .taxonomy import Taxonomy


CLASSIFIER_VERSION = "0.2.0"

JSON_FIELDS = (
    "messages_json",
    "tools_json",
    "tests_json",
    "metadata_json",
    "source_record_json",
)

LANGUAGE_ALIASES = {
    "c#": "csharp",
    "csharp": "csharp",
    "c++": "cpp",
    "cpp": "cpp",
    "objective-c++": "objective_cpp",
    "objective_cpp": "objective_cpp",
    "f#": "fsharp",
    "fsharp": "fsharp",
    "bash": "shell",
    "sh": "shell",
    "shell": "shell",
    "zsh": "shell",
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "py": "python",
    "python3": "python",
    "golang": "go",
    "rs": "rust",
    "rb": "ruby",
}

EXTENSION_LANGUAGES = {
    ".adb": "ada",
    ".ads": "ada",
    ".asm": "assembly",
    ".awk": "awk",
    ".c": "c",
    ".h": "c",
    ".cs": "csharp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".clj": "clojure",
    ".coffee": "coffeescript",
    ".cu": "cuda",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".fs": "fsharp",
    ".f90": "fortran",
    ".go": "go",
    ".groovy": "groovy",
    ".hs": "haskell",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".jl": "julia",
    ".kt": "kotlin",
    ".lua": "lua",
    ".mm": "objective_cpp",
    ".ml": "ocaml",
    ".pas": "pascal",
    ".pl": "perl",
    ".php": "php",
    ".ps1": "powershell",
    ".py": "python",
    ".r": "r",
    ".rkt": "racket",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".scm": "scheme",
    ".sh": "shell",
    ".sol": "solidity",
    ".sql": "sql",
    ".swift": "swift",
    ".tcl": "tcl",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".vim": "vimscript",
    ".vb": "visual_basic",
    ".zig": "zig",
}

TASK_ALIASES = {
    "bug_fixing": ("bug_fix", "bugfix", "bug fixing", "defect_fix", "repair_bug"),
    "feature_implementation": ("feature", "feature_implementation", "implement_feature"),
    "debugging_diagnosis": ("debug", "debugging", "diagnosis", "root_cause"),
    "test_creation": ("test_creation", "write_tests", "test_generation", "generate_tests"),
    "test_repair": ("test_repair", "fix_tests", "repair_tests"),
    "refactoring": ("refactor", "refactoring"),
    "performance_optimization": ("optimization", "optimisation", "performance"),
    "code_review": ("code_review", "review_patch", "review"),
    "build_ci_repair": ("build_repair", "ci_repair", "fix_ci", "build_fix"),
    "dependency_configuration": ("dependency", "configuration", "config"),
    "security_remediation": ("security", "vulnerability", "security_fix"),
    "documentation": ("documentation", "docs"),
    "repository_navigation": ("repository_navigation", "repo_navigation", "code_search"),
    "commit_transformation": ("commit", "commitpack", "patch_generation", "diff"),
    "code_generation": ("code_generation", "generate_code", "coding"),
    "code_completion": ("code_completion", "fill_in_middle", "fim"),
    "code_explanation": ("code_explanation", "explain_code"),
    "code_translation": ("code_translation", "translate_code"),
    "code_output_prediction": ("code_output_prediction", "output_prediction"),
    "algorithmic_problem_solving": (
        "competitive_programming",
        "algorithmic",
        "codeforces",
        "leetcode",
        "apps",
    ),
    "text_to_sql": ("text_to_sql", "text2sql", "nl2sql"),
    "sql_reasoning": ("sql_reasoning", "sql_debugging", "sql_optimization"),
    "data_analysis": ("data_analysis", "data_science"),
    "technical_qa": ("technical_qa", "programming_qa", "code_qa"),
    "api_usage_guidance": ("api_usage", "library_usage"),
    "terminal_task_execution": ("terminal_task", "terminal_execution", "terminal"),
    "command_generation": ("command_generation", "shell_command"),
    "terminal_output_prediction": ("terminal_output_prediction", "terminal_world_model"),
    "environment_setup": ("environment_setup", "setup_environment"),
    "system_administration": ("system_administration", "sysadmin"),
}

TOOL_TYPE_PATTERNS = {
    "shell": ("bash", "shell", "terminal", "execute_command", "run_command", "computer"),
    "file_read": ("read_file", "open_file", "view_file", "cat_file"),
    "file_search": ("search", "grep", "find_file", "search_files", "ripgrep"),
    "file_edit": ("edit", "write_file", "str_replace", "apply_patch", "create_file"),
    "version_control": ("git", "version_control", "vcs"),
    "code_execution": ("execute_code", "python", "code_interpreter", "run_code"),
    "test_runner": ("test", "pytest", "run_tests"),
    "build_compiler": ("build", "compile", "compiler"),
    "package_manager": ("package", "pip", "npm", "cargo", "dependency"),
    "browser_web": ("browser", "web", "http", "search_web"),
    "database": ("database", "sql", "query"),
    "notebook": ("notebook", "jupyter"),
}

SHELL_TOOL_NAMES = {"bash", "shell", "terminal", "execute_command", "run_command", "computer"}

COMMAND_ACTIVITIES = {
    "filesystem_navigation": {"cd", "pwd", "ls", "tree", "find"},
    "file_inspection": {"cat", "head", "tail", "less", "more", "stat", "file"},
    "text_search": {"grep", "rg", "ag", "awk"},
    "file_editing": {"sed", "perl", "cp", "mv", "touch", "mkdir", "rm", "patch"},
    "version_control": {"git", "hg", "svn"},
    "build_compilation": {"make", "cmake", "ninja", "gcc", "g++", "clang", "rustc", "go"},
    "test_execution": {"pytest", "unittest", "jest", "mocha", "ctest", "tox", "cargo"},
    "runtime_debugging": {"gdb", "lldb", "pdb", "strace", "ltrace", "valgrind"},
    "package_management": {"pip", "pip3", "uv", "npm", "yarn", "pnpm", "cargo", "apt", "apt-get", "conda"},
    "environment_configuration": {"export", "env", "source", "set", "unset"},
    "container_operations": {"docker", "podman", "kubectl", "apptainer", "singularity"},
    "network_operations": {"curl", "wget", "ssh", "scp", "ping", "nc"},
    "database_cli": {"psql", "mysql", "sqlite3", "redis-cli", "mongosh"},
    "system_administration": {"systemctl", "service", "ps", "top", "kill", "chmod", "chown", "sudo"},
    "data_processing": {"jq", "cut", "sort", "uniq", "tr", "xargs"},
}

PATH_KEY_RE = re.compile(r"(?:^|_)(?:file|path|filename|repo|repository)(?:$|_)", re.I)
CODE_FENCE_RE = re.compile(r"```\s*([A-Za-z0-9_+#.-]+)")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


class RecordFormatError(ValueError):
    """Raised when an input object is not a supported SFT record."""


def parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def unwrap_record(record: dict[str, Any]) -> dict[str, Any]:
    """Accept a unified row or one of the local preview wrapper documents."""
    stored = record.get("stored_sample")
    return dict(stored) if isinstance(stored, dict) else dict(record)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def _walk(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            yield path, child
            yield from _walk(child, path)


def _content(message: dict[str, Any]) -> Any:
    for key in ("content", "value", "text", "message", "observation"):
        if key in message:
            return message[key]
    return None


def _native_messages(owner: dict[str, Any]) -> list[dict[str, Any]]:
    """Represent native Apertus 2 items as OpenAI-like messages for extraction."""
    items = owner.get("items")
    if not isinstance(items, list):
        return []
    messages: list[dict[str, Any]] = []
    pending_reasoning: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type", "")).lower()
        payload = item.get("payload")
        if item_type == "think":
            if isinstance(payload, str):
                pending_reasoning.append(payload)
        elif item_type == "user":
            messages.append({"role": "user", "content": payload})
        elif item_type == "call":
            call = {
                "id": str(item.get("counter", "")),
                "function": {"name": item.get("name"), "arguments": payload},
            }
            # Consecutive native calls before a wait are parallel calls from one turn.
            if messages and messages[-1].get("_native_calls_open"):
                messages[-1]["tool_calls"].append(call)
            else:
                message: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": [call],
                    "_native_calls_open": True,
                }
                if pending_reasoning:
                    message["reasoning_content"] = "\n".join(pending_reasoning)
                    pending_reasoning = []
                messages.append(message)
        elif item_type == "wait":
            if messages:
                messages[-1].pop("_native_calls_open", None)
        elif item_type == "result":
            if messages:
                messages[-1].pop("_native_calls_open", None)
            messages.append(
                {
                    "role": "tool",
                    "name": item.get("name"),
                    "tool_call_id": str(item.get("counter", "")),
                    "content": payload,
                }
            )
        elif item_type in ("reply", "claim"):
            if messages:
                messages[-1].pop("_native_calls_open", None)
            message = {"role": "assistant", "content": payload}
            if pending_reasoning:
                message["reasoning_content"] = "\n".join(pending_reasoning)
                pending_reasoning = []
            messages.append(message)
    for message in messages:
        message.pop("_native_calls_open", None)
    return messages


def _messages(record: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    for owner in (record, source):
        native = _native_messages(owner)
        if native:
            return native
        for key in (
            "messages_json",
            "messages",
            "conversations",
            "conversation",
            "chat_rounds",
            "trajectory",
        ):
            value = parse_json(owner.get(key))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _tool_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    function = tool.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    for key in ("name", "tool_name", "function_name"):
        if tool.get(key):
            return str(tool[key])
    return None


def _available_tools(record: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    for owner in (record, source):
        system = owner.get("system")
        if isinstance(system, dict) and isinstance(system.get("tools"), list):
            return [item for item in system["tools"] if isinstance(item, dict)]
        for key in ("tools_json", "tools", "functions", "available_tools", "available_functions"):
            value = parse_json(owner.get(key))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    result.append({**call, "_message_index": message_index})
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            result.append({"function": function_call, "_message_index": message_index})
        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and part.get("type") in ("function-call", "tool_call"):
                    result.append({**part, "_message_index": message_index})
    return result


def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for message_index, message in enumerate(messages):
        role = str(message.get("role", message.get("from", ""))).lower()
        if role in ("tool", "function", "tool_result", "observation"):
            results.append({**message, "_message_index": message_index})
        parts = message.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if isinstance(part, dict) and part.get("type") in ("function-output", "tool_result"):
                    results.append({**part, "_message_index": message_index})
    return results


def _call_arguments(call: dict[str, Any]) -> Any:
    function = call.get("function")
    if isinstance(function, dict):
        return parse_json(function.get("arguments", function.get("args")))
    return parse_json(call.get("arguments", call.get("args")))


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _command_texts(value: Any, *, allow_plain_string: bool = False) -> Iterable[str]:
    value = parse_json(value)
    if isinstance(value, str):
        # Raw strings are commands only when they came from a structured call
        # argument. Assistant prose is considered only when it decodes to an
        # object containing an explicit command/cmd/script/keystrokes field.
        if allow_plain_string:
            yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in ("command", "cmd", "script", "keystrokes", "input") and isinstance(child, str):
                yield child
            elif key.lower() in ("commands", "actions"):
                yield from _command_texts(child, allow_plain_string=True)
    elif isinstance(value, list):
        for child in value:
            yield from _command_texts(child, allow_plain_string=allow_plain_string)


def _strip_heredoc_bodies(command: str) -> str:
    """Keep heredoc-opening commands but remove their embedded data/code lines."""
    kept: list[str] = []
    delimiter: str | None = None
    heredoc = re.compile(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
    for line in command.splitlines():
        if delimiter is not None:
            if line.strip() == delimiter:
                delimiter = None
            continue
        kept.append(line)
        match = heredoc.search(line)
        if match:
            delimiter = match.group(1)
    return "\n".join(kept)


def extract_command_names(command: str) -> list[str]:
    names: list[str] = []
    cleaned = _strip_heredoc_bodies(command).replace("\n", " ; ")
    try:
        lexer = shlex.shlex(cleaned, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        stream = list(lexer)
    except ValueError:
        stream = cleaned.split()
    segments: list[list[str]] = [[]]
    for token in stream:
        if token and set(token).issubset({";", "&", "|"}):
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    for tokens in segments:
        if not tokens:
            continue
        index = 0
        while index < len(tokens) and ASSIGNMENT_RE.match(tokens[index]):
            index += 1
        while index < len(tokens) and tokens[index] in ("sudo", "env", "command", "time", "nohup"):
            index += 1
            while index < len(tokens) and (tokens[index].startswith("-") or ASSIGNMENT_RE.match(tokens[index])):
                index += 1
        if index < len(tokens):
            name = os.path.basename(tokens[index]).lower()
            if name and not name.startswith(("<", ">")):
                names.append(name)
    return sorted(set(names))


def _normalize_language(value: str, allowed: frozenset[str]) -> str | None:
    normalized = value.strip().lower().replace(" ", "_")
    normalized = LANGUAGE_ALIASES.get(normalized, normalized)
    return normalized if normalized in allowed else None


def _explicit_languages(values: Iterable[tuple[str, Any]], allowed: frozenset[str]) -> tuple[set[str], list[str]]:
    labels: set[str] = set()
    evidence: list[str] = []
    for path, value in values:
        key = re.split(r"[.\[]", path)[-1].rstrip("]").lower()
        if key not in ("language", "languages", "programming_language", "programming_languages"):
            continue
        candidates = value if isinstance(value, list) else re.split(r"[,;/]", str(value))
        for candidate in candidates:
            label = _normalize_language(str(candidate), allowed)
            if label:
                labels.add(label)
                evidence.append(path)
    return labels, evidence


def _language_from_text(texts: Iterable[str], allowed: frozenset[str]) -> tuple[set[str], list[str]]:
    labels: set[str] = set()
    evidence: list[str] = []
    for text in texts:
        for raw in CODE_FENCE_RE.findall(text):
            label = _normalize_language(raw, allowed)
            if label:
                labels.add(label)
                evidence.append(f"code_fence:{raw}")
        if re.search(r"\bCREATE\s+TABLE\b|\bSELECT\s+.+\s+FROM\b", text, re.I | re.S):
            labels.add("sql")
            evidence.append("sql_syntax")
    return labels, evidence


def _language_from_paths(values: Iterable[tuple[str, Any]]) -> tuple[set[str], list[str], set[str]]:
    labels: set[str] = set()
    evidence: list[str] = []
    paths: set[str] = set()
    for path, value in values:
        if not PATH_KEY_RE.search(path) or not isinstance(value, str):
            continue
        suffix = PurePath(value).suffix.lower()
        if suffix in EXTENSION_LANGUAGES:
            labels.add(EXTENSION_LANGUAGES[suffix])
            evidence.append(path)
            paths.add(value)
    return labels, evidence, paths


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _explicit_task_domains(values: Iterable[tuple[str, Any]]) -> tuple[set[str], list[str]]:
    labels: set[str] = set()
    evidence: list[str] = []
    task_keys = {"task", "task_type", "task_category", "domain", "category", "problem_type", "source"}
    for path, value in values:
        key = re.split(r"[.\[]", path)[-1].rstrip("]").lower()
        if key not in task_keys or not isinstance(value, (str, int)):
            continue
        normalized = _normalized_text(value)
        for label, aliases in TASK_ALIASES.items():
            if any(normalized == _normalized_text(alias) for alias in aliases):
                labels.add(label)
                evidence.append(path)
    return labels, evidence


def _tool_type(name: str, description: str = "") -> str:
    haystack = f"{name} {description}".lower()
    for label, patterns in TOOL_TYPE_PATTERNS.items():
        if any(pattern in haystack for pattern in patterns):
            return label
    return "external_api" if description else "other_tool"


def _tool_description(tool: dict[str, Any]) -> str:
    description = tool.get("description")
    if description is None:
        function = tool.get("function")
        if isinstance(function, dict):
            description = function.get("description")
    return str(description) if description is not None else ""


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _classification(labels: Iterable[str], evidence: Iterable[str], *, applicable: bool = True,
                    confidence: str = "high") -> dict[str, Any]:
    unique_labels = sorted(set(labels))
    unique_evidence = sorted(set(evidence))
    if not applicable:
        status = "not_applicable"
    elif unique_labels:
        status = "labeled"
    else:
        status = "unknown"
    return {
        "status": status,
        "labels": unique_labels,
        "method": "deterministic_rule",
        "confidence": confidence if unique_labels else "low",
        "evidence": unique_evidence,
    }


def annotate_record(raw_record: dict[str, Any], taxonomy: Taxonomy) -> dict[str, Any]:
    """Return a compact deterministic sidecar annotation for one source record."""
    record = unwrap_record(raw_record)
    decoded = {key: parse_json(record.get(key)) for key in JSON_FIELDS}
    source = decoded.get("source_record_json")
    source = source if isinstance(source, dict) else {}
    metadata = decoded.get("metadata_json")
    metadata = metadata if isinstance(metadata, dict) else {}
    combined = {**source, **metadata}
    messages = _messages({**record, **decoded}, combined)
    tools = _available_tools({**record, **decoded}, combined)
    calls = _calls(messages)
    results = _tool_results(messages)

    all_values = list(_walk(record)) + list(_walk(combined))
    text_values = [
        value
        for owner in (record, combined)
        for value in (owner.get("prompt"), owner.get("response"), owner.get("reasoning"))
        if isinstance(value, str)
    ]
    text_values.extend(str(value) for message in messages for value in [_content(message)] if isinstance(value, str))
    if not messages and not text_values:
        raise RecordFormatError(
            "record has no supported content (expected native items, messages, prompt, response, or reasoning)"
        )

    languages, language_evidence = _explicit_languages(all_values, taxonomy.labels["programming_languages"])
    path_languages, path_evidence, file_paths = _language_from_paths(all_values)
    text_languages, text_evidence = _language_from_text(text_values, taxonomy.labels["programming_languages"])
    languages.update(path_languages)
    languages.update(text_languages)
    language_evidence.extend(path_evidence)
    language_evidence.extend(text_evidence)

    available_names = sorted({name for name in (_tool_name(tool) for tool in tools) if name})
    invoked_names = sorted({name for name in (_tool_name(call) for call in calls) if name})
    descriptions = {
        name: _tool_description(tool)
        for tool in tools
        if (name := _tool_name(tool))
    }
    tool_types = {_tool_type(name, descriptions.get(name, "")) for name in available_names + invoked_names}

    command_texts: list[str] = []
    for call in calls:
        name = (_tool_name(call) or "").lower()
        if name in SHELL_TOOL_NAMES or _tool_type(name) == "shell":
            command_texts.extend(_command_texts(_call_arguments(call), allow_plain_string=True))
    # Some terminal datasets encode command batches as JSON assistant content, without tool_calls.
    for message in messages:
        if str(message.get("role", message.get("from", ""))).lower() in ("assistant", "gpt", "bot"):
            command_texts.extend(_command_texts(_content(message)))
    command_names = sorted({name for text in command_texts for name in extract_command_names(text)})
    if command_names:
        languages.add("shell")
        language_evidence.append("terminal_commands")

    terminal_activities: set[str] = set()
    for activity, commands in COMMAND_ACTIVITIES.items():
        if commands.intersection(command_names):
            terminal_activities.add(activity)
    if command_names and not terminal_activities:
        terminal_activities.add("command_execution")

    explicit_tasks, task_evidence = _explicit_task_domains(all_values)
    source_name = str(record.get("source_dataset", "")).lower()
    joined_text = "\n".join(text_values)
    if "sql" in languages and re.search(r"\bCREATE\s+TABLE\b", joined_text, re.I) and re.search(
        r"\bSELECT\b", str(record.get("response", combined.get("response", ""))), re.I
    ):
        explicit_tasks.add("text_to_sql")
        task_evidence.append("sql_schema_and_select_target")
    if "terminal-world-model" in source_name or "terminal_world_model" in source_name:
        explicit_tasks.add("terminal_output_prediction")
        task_evidence.append("source_dataset")
    if command_names and not ({"terminal_output_prediction", "command_generation"} & explicit_tasks):
        explicit_tasks.add("terminal_task_execution")
        task_evidence.append("terminal_commands")

    terminal_task_labels = {
        "terminal_task_execution",
        "command_generation",
        "terminal_output_prediction",
        "environment_setup",
        "system_administration",
    }
    terminal_present = bool(
        command_names or "shell" in tool_types or terminal_task_labels.intersection(explicit_tasks)
    )
    tool_present = bool(available_names or invoked_names)
    repository_keys = {
        "repo",
        "repository",
        "base_commit",
        "patch",
        "diff",
        "instance_id",
        "problem_statement",
    }
    repository_evidence = [path for path, value in all_values if path.split(".")[-1] in repository_keys and value]
    swe_task_labels = {
        "feature_implementation",
        "bug_fixing",
        "debugging_diagnosis",
        "test_creation",
        "test_repair",
        "refactoring",
        "performance_optimization",
        "code_review",
        "build_ci_repair",
        "dependency_configuration",
        "security_remediation",
        "documentation",
        "repository_navigation",
        "commit_transformation",
    }
    swe_present = bool(repository_evidence or swe_task_labels.intersection(explicit_tasks))

    expert_domains = []
    expert_evidence = []
    if swe_present:
        expert_domains.append("swe")
        expert_evidence.extend(repository_evidence or ["task_domains"])
    if terminal_present:
        expert_domains.append("terminal")
        expert_evidence.append("terminal_commands_or_shell_tool")
    if tool_present:
        expert_domains.append("tool_use")
        expert_evidence.append("tool_definitions_or_calls")
    if not expert_domains and explicit_tasks:
        expert_domains.append("none_of_the_above")
        expert_evidence.append("explicit_non_target_task_domain")

    work_context = []
    work_evidence = []
    if repository_evidence:
        work_context.append("repository")
        work_evidence.extend(repository_evidence)
    if terminal_present:
        work_context.append("terminal_environment")
        work_evidence.append("terminal_commands_or_shell_tool")
    if "sql" in languages and re.search(r"\bCREATE\s+TABLE\b", joined_text, re.I):
        work_context.append("database_schema")
        work_evidence.append("create_table_syntax")
    if any(path.lower().endswith(".ipynb") for path in file_paths):
        work_context.append("notebook")
        work_evidence.append("ipynb_path")
    if len(file_paths) == 1 and "repository" not in work_context:
        work_context.append("single_file")
        work_evidence.append("one_explicit_file_path")
    elif len(file_paths) > 1 and "repository" not in work_context:
        work_context.append("multi_file")
        work_evidence.append("multiple_explicit_file_paths")
    elif not work_context and text_languages:
        work_context.append("standalone_snippet")
        work_evidence.append("language_tagged_code_fence")

    user_messages = sum(
        str(message.get("role", message.get("from", ""))).lower() in ("user", "human")
        for message in messages
    )
    assistant_messages = sum(
        str(message.get("role", message.get("from", ""))).lower() in ("assistant", "gpt", "bot")
        for message in messages
    )
    reasoning_present = bool(record.get("reasoning")) or any(
        message.get("reasoning_content") or message.get("thought")
        for message in messages
    )
    interaction_styles = []
    interaction_evidence = []
    if user_messages <= 1 and assistant_messages <= 1 and not calls and not terminal_present:
        interaction_styles.append("direct_response")
        interaction_evidence.append("single_user_assistant_exchange")
    if user_messages > 1 or assistant_messages > 1:
        interaction_styles.append("multi_turn_conversation")
        interaction_evidence.append("multiple_user_or_assistant_messages")
    if reasoning_present:
        interaction_styles.append("reasoning_trace")
        interaction_evidence.append("reasoning_or_thought_field")
    if calls or (terminal_present and user_messages + assistant_messages > 2):
        interaction_styles.append("agent_trajectory")
        interaction_evidence.append("action_observation_structure")
    if calls:
        interaction_styles.append("tool_calling_trajectory")
        interaction_evidence.append("structured_tool_calls")
    if terminal_present and user_messages + assistant_messages > 2:
        interaction_styles.append("terminal_trajectory")
        interaction_evidence.append("terminal_commands_and_multiple_messages")
    if "terminal_output_prediction" in explicit_tasks:
        interaction_styles.append("environment_transition_prediction")
        interaction_evidence.append("terminal_output_prediction_task")

    change_scope = []
    if swe_present:
        if not file_paths and not any(label in explicit_tasks for label in swe_task_labels - {"code_review", "repository_navigation"}):
            change_scope.append("no_change")
        elif len(file_paths) <= 1:
            change_scope.append("localized_change")
        elif len(file_paths) > 1:
            change_scope.append("cross_file_change")
    artifact_types = set()
    for path in file_paths:
        lower = path.lower()
        if re.search(r"(?:^|/)(?:test|tests|testing)(?:/|_|\.)", lower):
            artifact_types.add("tests")
        elif PurePath(path).name.lower() in (
            "makefile", "cmakelists.txt", "pyproject.toml", "setup.py", "tox.ini",
            "dockerfile", ".gitlab-ci.yml",
        ) or ".github/workflows/" in lower:
            artifact_types.add("build_configuration")
        elif PurePath(path).name.lower() in (
            "requirements.txt", "package.json", "cargo.toml", "go.mod", "poetry.lock", "uv.lock",
        ):
            artifact_types.add("dependency_manifest")
        elif PurePath(path).suffix.lower() in EXTENSION_LANGUAGES:
            artifact_types.add("source_code")

    parallel = any(
        isinstance(message.get("tool_calls"), list) and len(message["tool_calls"]) > 1
        for message in messages
    )
    tool_properties = set()
    if parallel:
        tool_properties.add("parallel_calls")
    if len(calls) > 1:
        tool_properties.add("iterative_calls")
    if set(available_names).difference(invoked_names):
        tool_properties.add("unused_available_tools")
    if set(invoked_names).difference(available_names) and available_names:
        tool_properties.add("undeclared_tool_call")
    result_text = " ".join(_strings(results)).lower()
    if results and re.search(r"\b(error|failed|exception|not found|traceback)\b", result_text):
        last_error_index = max(result.get("_message_index", -1) for result in results)
        if any(call.get("_message_index", -1) > last_error_index for call in calls):
            tool_properties.add("call_error_recovery")

    legacy_classifications = {
        "expert_domains": _classification(expert_domains, expert_evidence),
        "task_domains": _classification(explicit_tasks, task_evidence, confidence="medium"),
        "programming_languages": _classification(languages, language_evidence, confidence="medium"),
        "work_context": _classification(work_context, work_evidence, confidence="medium"),
        "interaction_styles": _classification(interaction_styles, interaction_evidence),
        "swe.change_scope": _classification(change_scope, ["file_paths", "task_domains"], applicable=swe_present, confidence="medium"),
        "swe.artifact_types": _classification(artifact_types, file_paths, applicable=swe_present, confidence="medium"),
        "terminal.activities": _classification(terminal_activities, command_names, applicable=terminal_present),
        "terminal.command_names": _classification(
            command_names,
            [f"shell_command[{index}]" for index in range(len(command_texts))],
            applicable=terminal_present,
        ),
        "tool_use.tool_types": _classification(tool_types, available_names + invoked_names, applicable=tool_present),
        "tool_use.available_tool_names": _classification(available_names, ["tools"], applicable=tool_present),
        "tool_use.invoked_tool_names": _classification(invoked_names, ["tool_calls"], applicable=tool_present),
        "tool_use.trajectory_properties": _classification(tool_properties, ["tool_calls", "tool_results"], applicable=tool_present),
    }

    if "swe_categories" in taxonomy.labels:
        # v0.2 keeps deterministic signals but exposes only the compact axes.
        # When the expert itself is unresolved, its category remains unresolved
        # too so that the semantic judge can decide both in one request.
        expert_unresolved = not expert_domains

        swe_category_sources = {
            "bug_fixing": {"bug_fixing"},
            "feature_implementation": {"feature_implementation"},
            "testing": {"test_creation", "test_repair"},
            "refactoring": {"refactoring"},
            "build_dependency_configuration": {
                "build_ci_repair",
                "dependency_configuration",
            },
        }
        swe_categories = {
            category
            for category, sources in swe_category_sources.items()
            if sources.intersection(explicit_tasks)
        }
        terminal_category_sources = {
            "filesystem_inspection": {"filesystem_navigation", "file_inspection"},
            "text_search_processing": {"text_search", "data_processing"},
            "version_control": {"version_control"},
            "build_test_debug": {
                "build_compilation",
                "test_execution",
                "runtime_debugging",
            },
            "environment_system_operations": {
                "package_management",
                "environment_configuration",
                "container_operations",
                "network_operations",
                "database_cli",
                "system_administration",
            },
        }
        terminal_categories = {
            category
            for category, sources in terminal_category_sources.items()
            if sources.intersection(terminal_activities)
        }
        tool_category_sources = {
            "shell_execution": {"shell"},
            "file_read_search": {"file_read", "file_search"},
            "file_editing": {"file_edit"},
            "code_build_test_execution": {
                "code_execution",
                "test_runner",
                "build_compiler",
            },
            "external_service_interaction": {
                "browser_web",
                "database",
                "external_api",
            },
        }
        tool_categories = {
            category
            for category, sources in tool_category_sources.items()
            if sources.intersection(tool_types)
        }
        classifications = {
            "expert_domains": legacy_classifications["expert_domains"],
            "programming_languages": legacy_classifications["programming_languages"],
            "swe_categories": _classification(
                swe_categories,
                task_evidence,
                applicable=expert_unresolved or swe_present,
                confidence="medium",
            ),
            "terminal_categories": _classification(
                terminal_categories,
                command_names,
                applicable=expert_unresolved or terminal_present,
                confidence="medium",
            ),
            "tool_use_categories": _classification(
                tool_categories,
                available_names + invoked_names,
                applicable=expert_unresolved or tool_present,
                confidence="medium",
            ),
        }
    else:
        classifications = legacy_classifications

    for dimension, classification in classifications.items():
        if dimension in taxonomy.labels:
            taxonomy.validate_labels(dimension, classification["labels"])
        if classification["status"] not in taxonomy.statuses:
            raise ValueError(f"invalid status for {dimension}: {classification['status']}")

    source_tokens = record.get("source_tokens", record.get("token_count"))
    training_loss_tokens = record.get("training_loss_tokens", record.get("loss_token_count"))
    input_hash = record_hash(record)
    record_id = record.get(
        "id", record.get("record_id", combined.get("id", combined.get("record_id")))
    )
    return {
        "record_id": str(record_id) if record_id is not None else input_hash,
        "source_dataset": str(
            record.get(
                "source_dataset",
                record.get(
                    "dataset_source",
                    record.get(
                        "source",
                        combined.get(
                            "source_dataset",
                            combined.get("dataset_source", combined.get("source", "")),
                        ),
                    ),
                ),
            )
        ),
        "source_revision": str(record.get("source_revision", "")),
        "source_split": str(record.get("source_split", "")),
        "source_row_index": _optional_int(record.get("source_row_index")),
        "taxonomy_version": taxonomy.version,
        "classifier_version": CLASSIFIER_VERSION,
        "input_hash": input_hash,
        "source_tokens": _optional_int(source_tokens),
        "training_loss_tokens": _optional_int(training_loss_tokens),
        "classifications": classifications,
    }
