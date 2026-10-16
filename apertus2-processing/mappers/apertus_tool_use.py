"""Map the SFT mixes of swiss-ai/apertus-tool-use to native Apertus 2 conversations.

A mix file has one OpenAI-style conversation per line, with `reasoning_content`. A
conversation without reasoning gets `thinking: low` and gives one sample. Any other
conversation gets `thinking: medium` and gives one sample per user message: the earlier turns
have no thinks, as the harness may remove them, and loss weight 0. A sample is skipped if its
conversation has an assistant step without text and without calls, or if it is longer than
`--max-tokens`. Check and tokenize the output with `--loss-weights`.

    uv run --no-sync python mappers/apertus_tool_use.py OUT --input MIX/train.jsonl --check
"""

import functools
import json

from apertus_common import Conversation, OutputMessage, Wait, load_encoding
from datasets import Dataset

from apertus2_processing.mapping import apply_convention, from_chat, mapper_parser, run_mapper

META_COLUMNS = ["teacher", "subset", "task", "language", "license"]
KEEP_COLUMNS = ["id", "source", "turn", *META_COLUMNS]


def _has_reasoning(record):
    return any((message.get("reasoning_content") or "").strip() for message in record["messages"])


def read(paths):
    """One row per sample, with the position of its line, so the rows stay small."""
    rows = {column: [] for column in [*KEEP_COLUMNS, "path", "offset"]}
    for path in map(str, paths):
        offset = 0
        with open(path, "rb") as file:
            for line in file:
                record = json.loads(line)
                user_messages = sum(message["role"] == "user" for message in record["messages"])
                meta = record["meta"]
                for turn in range(max(user_messages, 1) if _has_reasoning(record) else 1):
                    row = {"id": record["id"], "source": meta["source"], "turn": turn}
                    row |= {column: meta.get(column) for column in META_COLUMNS}
                    row |= {"path": path, "offset": offset}
                    for column, value in row.items():
                        rows[column].append(value)
                offset += len(line)
    return Dataset.from_dict(rows)


def _record(row):
    """The line of the row. A changed file stops the run: `ValueError` would skip the row."""
    with open(row["path"], "rb") as file:
        file.seek(row["offset"])
        line = file.readline()
    try:
        record = json.loads(line)
    except ValueError:
        record = None
    if not record or record.get("id") != row["id"]:
        raise RuntimeError(f"{row['path']} changed after it was read: no line of {row['id']}")
    return record


def _compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _text_arguments(message):
    calls = [
        {
            **call,
            "function": {**call["function"], "arguments": _compact(call["function"]["arguments"])},
        }
        for call in message.get("tool_calls") or []
    ]
    return {**message, "tool_calls": calls} if calls else message


def _without_reasoning(messages):
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]


def _untrained(item):
    has_loss = isinstance(item, OutputMessage | Wait)
    return item.model_copy(update={"loss_weight": 0}) if has_loss else item


def _turn(messages, tools, turn):
    starts = [i for i, message in enumerate(messages) if message["role"] == "user"]
    if not starts:
        raise ValueError("no user message")
    start, end = [*starts, len(messages)][turn : turn + 2]
    # TODO: keep the thinks of earlier turns if Apertus 2 is served with them; Terminus-2
    # drops them unless run with `interleaved_thinking=true`.
    earlier = _without_reasoning(messages[:start])
    conversation = from_chat(
        [*earlier, *messages[start:end]], tools=tools, system={"thinking": "medium"}
    )
    system, items = conversation.system, conversation.items
    last_user = max(i for i, item in enumerate(items) if item.type == "user")
    trained = apply_convention(Conversation(system=system, items=items[last_user + 1 :])).items
    return Conversation(system=system, items=[*map(_untrained, items[: last_user + 1]), *trained])


@functools.cache
def _encoding(tokenizer):
    return load_encoding(tokenizer)


def _sample(row):
    record = _record(row)
    for message in record["messages"]:
        if (
            message["role"] == "assistant"
            and not (message.get("content") or "").strip()
            and not message.get("tool_calls")
        ):
            raise ValueError("assistant step without text and without calls")
    messages = [_text_arguments(message) for message in record["messages"]]
    tools = record["tools"]
    if _has_reasoning(record):
        return _turn(messages, tools, row["turn"])
    return from_chat(_without_reasoning(messages), tools=tools, system={"thinking": "low"})


def to_native(row, max_tokens=None, tokenizer=None):
    sample = _sample(row)
    if max_tokens and len(_encoding(tokenizer).encode_conversation(sample)) > max_tokens:
        raise ValueError(f"sample longer than {max_tokens} tokens")
    return sample


def main():
    parser = mapper_parser(__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", required=True, help="Mix files (JSON Lines)")
    parser.add_argument("--max-tokens", type=int, help="Skip samples longer than this many tokens")
    parser.add_argument(
        "--tokenizer", help="Tokenizer directory; counts the tokens for --max-tokens"
    )
    parser.set_defaults(thinking_convention="source", loss_weights=True)
    args = parser.parse_args()
    if args.thinking_convention != "source":
        parser.error("this mapper applies think-first itself, to the trained turn only")
    if args.max_tokens and not args.tokenizer:
        parser.error("--max-tokens needs --tokenizer")
    run_mapper(
        read(args.input),
        functools.partial(to_native, max_tokens=args.max_tokens, tokenizer=args.tokenizer),
        args,
        keep_columns=KEEP_COLUMNS,
    )


if __name__ == "__main__":
    main()
