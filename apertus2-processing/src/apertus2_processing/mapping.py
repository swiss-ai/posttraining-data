"""Build native Apertus 2 conversations from source rows and save them with `datasets`.

A mapper is a script with `to_native(row) -> Conversation`, built with `from_chat` for
OpenAI-style message lists or the `apertus-common` builder otherwise, plus `main()` using
`mapper_parser` and `run_mapper`. The driver applies the thinking convention, optionally
checks each conversation, skips rows whose mapping raised `ValueError`, and writes
`conversation_json`. See README.md, "Writing a mapper".
"""

import argparse
import functools
import json
import sys
from collections import Counter
from pathlib import Path

from apertus_common import (
    AdapterError,
    Conversation,
    InputMessage,
    MappingPolicy,
    OutputMessage,
    Think,
    Wait,
    checkers_for,
    from_openai,
)
from datasets import Dataset, Features, List, Value

from ._run import check_conversation, issue_counts, write_json
from .report import FAILING

SUMMARY = "mapping.json"
# Project convention for think items; the profile itself makes think optional.
# "think-first": every output burst starts with a think, empty when the source has none.
# "source": think items only where the source has reasoning.
CONVENTIONS = ("think-first", "source")
DEFAULT_CONVENTION = "think-first"  # See README.md, "Thinking convention".
SIDE_COLUMNS = {
    "conversation_json": Value("string"),
    "skip": Value("string"),
    "failed": Value("bool"),
    "issues": List({"rule": Value("string"), "severity": Value("string")}),
}


def apply_convention(conversation, convention=DEFAULT_CONVENTION):
    """Insert empty think items where the convention requires one; waits are untouched."""
    if convention != "think-first":
        return conversation
    items, previous = [], None
    for item in conversation.items:
        burst_start = isinstance(item, OutputMessage) and (
            previous is None or isinstance(previous, InputMessage | Wait)
        )
        if burst_start and not isinstance(item, Think):
            items.append(Think(payload=""))
        items.append(item)
        previous = item
    if len(items) == len(conversation.items):
        return conversation
    return Conversation(system=conversation.system, items=items)


def is_turn_based(messages):
    """True if, after an optional system message, user and assistant alternate."""
    roles = [message["role"] for message in messages]
    roles = roles[1:] if roles[:1] == ["system"] else roles
    return bool(roles) and roles == ["user", "assistant"] * (len(roles) // 2)


def _message(message):
    """Drop what `datasets` adds and the library rejects: `None` fields (every optional
    struct field is materialised), `name` on tool messages, and empty `content` on
    tool-call turns, which would otherwise become an empty reply before the call."""
    empty_call = message.get("tool_calls") and not message.get("content")
    return {
        k: v
        for k, v in message.items()
        if v is not None and k != "name" and not (k == "content" and empty_call)
    }


def _tool(tool):
    """Parse `parameters` stored as a JSON string; an empty one means no schema."""
    function = dict(tool["function"])
    if isinstance(function.get("parameters"), str):
        function["parameters"] = (
            json.loads(function["parameters"]) if function["parameters"] else None
        )
    return {**tool, "function": function}


def from_chat(messages, *, tools=None, system=None, merge_system=False):
    """Map OpenAI-style messages (and function tools) to a conversation with waits.

    `system` supplies `SystemPrompt` defaults as a dict or instance, such as `identity`,
    `thinking`, and `environment`; source system messages become `behavior`. Several
    system or `developer` messages require `merge_system=True`, which joins them. Rows
    the library cannot map raise `AdapterError`. Dataset quirks are normalised first:
    `None` fields and tool-message `name`s are dropped, empty `content` on tool-call
    turns is dropped, and tool `parameters` given as JSON strings are parsed.
    """
    messages = [_message(m) for m in messages]
    tools = [_tool(t) for t in tools] if tools else None
    policy = MappingPolicy(merge_system_messages=merge_system)
    return from_openai(messages, tools=tools, system_defaults=system, policy=policy).value


@functools.cache
def _checkers(loss_weights):
    return checkers_for("full", allow_loss_weights=loss_weights, training=True)


def _map_batch(batch, *, to_native, convention, check, strict, loss_weights):
    keys = list(batch.keys())
    out = {column: [] for column in SIDE_COLUMNS}
    for values in zip(*(batch[key] for key in keys), strict=True):
        skip, failed, issues, raw = None, False, [], None
        try:
            conversation = apply_convention(
                to_native(dict(zip(keys, values, strict=True))), convention
            )
        except (ValueError, AdapterError) as error:  # Other exceptions are mapper bugs.
            skip = (error.reason if isinstance(error, AdapterError) else str(error))[:120]
        else:
            if check:
                ok, found = check_conversation(conversation, _checkers(loss_weights))
                failed = not ok
                issues = [{"rule": i["rule"], "severity": i["severity"]} for i in found]
                if strict and failed:
                    skip = "check/" + next(i["rule"] for i in found if i["severity"] in FAILING)
            if skip is None:
                raw = conversation.to_json()
        out["conversation_json"].append(raw)
        out["skip"].append(skip)
        out["failed"].append(failed)
        out["issues"].append(issues)
    return out


def map_dataset(
    dataset,
    to_native,
    *,
    num_proc=1,
    keep_columns=(),
    convention=DEFAULT_CONVENTION,
    check=False,
    strict=False,
    loss_weights=False,
    limit=None,
):
    """Map every row with `to_native`; return (native dataset, summary).

    Rows whose mapping raises `ValueError` or `AdapterError` are skipped and counted by
    reason; other exceptions propagate. With `check`, each conversation is checked inline
    and rule counts are summarised; with `strict`, failing rows are skipped as well. The
    `datasets` cache is disabled: it ignores library upgrades.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown thinking convention {convention!r}")
    missing = [name for name in keep_columns if name not in dataset.column_names]
    clashing = [name for name in keep_columns if name in SIDE_COLUMNS]
    if missing or clashing:
        raise ValueError(f"keep_columns missing {missing} or reserved {clashing}")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    kept = {name: dataset.features[name] for name in dataset.column_names if name in keep_columns}
    features = Features(kept | SIDE_COLUMNS)
    if len(dataset):
        native = dataset.map(
            _map_batch,
            batched=True,
            num_proc=min(num_proc, len(dataset)) if num_proc > 1 else None,
            remove_columns=[name for name in dataset.column_names if name not in kept],
            features=features,
            load_from_cache_file=False,
            fn_kwargs={
                "to_native": to_native,
                "convention": convention,
                "check": check or strict,
                "strict": strict,
                "loss_weights": loss_weights,
            },
        )
    else:  # map would never call the function and would drop the side columns
        native = Dataset.from_dict({name: [] for name in features}, features=features)
    skips = native["skip"]
    skipped = Counter(filter(None, skips))
    summary = {
        "rows": len(native),
        "written": len(native) - sum(skipped.values()),
        "skipped": dict(sorted(skipped.items())),
        "convention": convention,
        "check": None,
    }
    if check or strict:
        issues = native["issues"]
        counts = Counter((i["rule"], i["severity"]) for found in issues for i in found)
        summary["check"] = {
            "rows_with_issues": sum(map(bool, issues)),
            "failed_rows": sum(native["failed"]),
            "issues": issue_counts(counts),
        }
    if skipped:
        native = native.select([i for i, skip in enumerate(skips) if skip is None])
    return native.remove_columns([c for c in SIDE_COLUMNS if c != "conversation_json"]), summary


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def mapper_parser(description):
    """Argument parser with the options every mapper shares; add dataset-specific ones."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("output", help="Directory for the saved native HF dataset")
    parser.add_argument("--num-proc", type=_positive, default=1, help="Worker processes")
    parser.add_argument("--limit", type=_positive, help="Map only the first N rows")
    parser.add_argument(
        "--check", action="store_true", help="Check each conversation and summarise rule counts"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Also skip conversations that fail the check"
    )
    parser.add_argument(
        "--loss-weights", action="store_true", help="Checked conversations carry loss weights"
    )
    parser.add_argument(
        "--thinking-convention",
        choices=CONVENTIONS,
        default=DEFAULT_CONVENTION,
        help="How think items are added; see README.md",
    )
    return parser


def run_mapper(dataset, to_native, args, *, keep_columns=()):
    """Map, save to `args.output` with `mapping.json` inside it, report; return the summary."""
    native, summary = map_dataset(
        dataset,
        to_native,
        num_proc=args.num_proc,
        keep_columns=keep_columns,
        convention=args.thinking_convention,
        check=args.check,
        strict=args.strict,
        loss_weights=args.loss_weights,
        limit=args.limit,
    )
    output = Path(args.output)
    native.save_to_disk(str(output))
    write_json(output / SUMMARY, summary)
    skipped = sum(summary["skipped"].values())
    print(f"{summary['written']} native conversations written to {output}; skipped {skipped}")
    for reason, count in summary["skipped"].items():
        print(f"  skipped {count}: {reason}")
    if summary["check"]:
        for issue in summary["check"]["issues"]:
            print(f"  {issue['severity']} {issue['count']}: {issue['rule']}")
    if not summary["written"]:
        print("error: no conversations written", file=sys.stderr)
        sys.exit(2)
    return summary
