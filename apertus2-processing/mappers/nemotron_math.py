"""Map nvidia/Nemotron-SFT-Math-v4 to native Apertus 2 conversations with answer claims.

Rows are single math problems answered by DeepSeek-V4-Pro in high reasoning mode, as
OpenAI-style messages: the user turn (problem plus the boxed-answer instruction), then
assistant turns, and in the `tir` subset optional rounds of `stateful_python_code_exec`
calls and results. `reasoning_content` becomes `think`, tool rounds become `call` and
`result`, and `thinking` is `high`. The final reply's last `\\boxed{...}` becomes a
`claim` with the bare answer, so it is gradable without parsing the reply; the reply
keeps its box. Rows without a boxed answer are skipped. `--limit` applies after
`--subset`.

    uv run --no-sync python mappers/nemotron_math.py OUT --input DIR --num-proc 8 --check

`--input` is the directory of `hf download --repo-type dataset nvidia/Nemotron-SFT-Math-v4`
(README.md, "Input layout"). For Alps, use the container launch command in README.md.
"""

from apertus_common import Claim, Conversation
from datasets import load_dataset

from apertus2_processing.mapping import from_chat, mapper_parser, run_mapper

KEEP_COLUMNS = ["uuid", "subset", "source", "license", "expected_answer"]


def boxed_answer(text):
    """Content of the last `\\boxed{...}` in `text`, braces balanced; None if absent."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, opening = 0, start + len("\\boxed")
    for position in range(opening, len(text)):
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        if depth == 0:
            return text[opening + 1 : position].strip()
    return None


def to_native(row):
    final = row["messages"][-1]
    answer = boxed_answer(final["content"] or "") if final["role"] == "assistant" else None
    if not answer:
        raise ValueError("no boxed answer in final reply")
    tools = row["tools"] or None
    conversation = from_chat(row["messages"], tools=tools, system={"thinking": "high"})
    *head, reply, wait = conversation.items
    if reply.type != "reply":
        raise ValueError("final message is not a reply")
    return Conversation(
        system=conversation.system, items=[*head, Claim(payload=answer), reply, wait]
    )


def main():
    parser = mapper_parser(__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Downloaded dataset directory")
    parser.add_argument("--subset", choices=("cot", "tir"), help="Map only this subset")
    args = parser.parse_args()
    source = load_dataset("parquet", data_files=f"{args.input}/data/*.parquet", split="train")
    if args.subset:
        source = source.filter(
            lambda s: s == args.subset, input_columns="subset", num_proc=args.num_proc
        )
    run_mapper(source, to_native, args, keep_columns=KEEP_COLUMNS)


if __name__ == "__main__":
    main()
