"""Map the SFT mixes of swiss-ai/apertus-tool-use to native Apertus 2 conversations.

Each conversation of a mix file (OpenAI messages with `reasoning_content`) gives one sample
with the thinks of all its turns, at the standard `medium` thinking. A sample is skipped if
an assistant step has neither text nor calls, or if it is longer than `--max-tokens` with the
empty thinks of the thinking convention.

    uv run --no-sync python mappers/apertus_tool_use.py OUT --input MIX/train.jsonl --check
"""

import functools
import json

from apertus_common import load_encoding
from datasets import Dataset

from apertus2_processing.mapping import (
    DEFAULT_CONVENTION,
    apply_convention,
    from_chat,
    mapper_parser,
    run_mapper,
)

META_COLUMNS = ["teacher", "subset", "task", "language", "license"]
KEEP_COLUMNS = ["id", "source", *META_COLUMNS]


def read(paths):
    """One row per conversation, with the position of its line, so the rows stay small."""
    rows = {column: [] for column in [*KEEP_COLUMNS, "path", "offset"]}
    for path in map(str, paths):
        offset = 0
        with open(path, "rb") as file:
            for line in file:
                record = json.loads(line)
                meta = record["meta"]
                row = {"id": record["id"], "source": meta["source"]}
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


@functools.cache
def _encoding(tokenizer):
    return load_encoding(tokenizer)


def to_native(row, max_tokens=None, tokenizer=None, convention=DEFAULT_CONVENTION):
    record = _record(row)
    for message in record["messages"]:
        if (
            message["role"] == "assistant"
            and not (message.get("content") or "").strip()
            and not message.get("tool_calls")
        ):
            raise ValueError("assistant step without text and without calls")
    messages = [_text_arguments(message) for message in record["messages"]]
    sample = from_chat(messages, tools=record["tools"])
    # The driver adds the empty thinks after this function, so count them here.
    written = apply_convention(sample, convention)
    if max_tokens and len(_encoding(tokenizer).encode_conversation(written)) > max_tokens:
        raise ValueError(f"sample longer than {max_tokens} tokens")
    return sample


def main():
    parser = mapper_parser(__doc__.splitlines()[0])
    parser.add_argument("--input", nargs="+", required=True, help="Mix files (JSON Lines)")
    parser.add_argument("--max-tokens", type=int, help="Skip samples longer than this many tokens")
    parser.add_argument(
        "--tokenizer", help="Tokenizer directory; counts the tokens for --max-tokens"
    )
    args = parser.parse_args()
    if args.max_tokens and not args.tokenizer:
        parser.error("--max-tokens needs --tokenizer")
    run_mapper(
        read(args.input),
        functools.partial(
            to_native,
            max_tokens=args.max_tokens,
            tokenizer=args.tokenizer,
            convention=args.thinking_convention,
        ),
        args,
        keep_columns=KEEP_COLUMNS,
    )


if __name__ == "__main__":
    main()
