"""Map HuggingFaceH4/no_robots chats to native Apertus 2 conversations.

Accept an optional system message followed by alternating user/assistant pairs.
Set thinking to the profile's standard `medium`, map the source system message to
`behavior`, and end each assistant answer with a wait. A single-turn chat becomes
system, user, reply, wait. Skip non-alternating rows (in no_robots, user lines
mislabelled as assistant answers).

Run from apertus2-processing on a laptop or compute node, never a cluster login
node. `--num-proc` controls filtering and mapping workers (default 1):

    uv run --no-sync python mappers/no_robots.py /data/no_robots-native --num-proc 8

For Alps, use the container and scratch-environment launch command in README.md.

Save an HF dataset with native JSON strings in `conversation_json` and the source
`prompt_id`, ready for `apertus-data prepare`. See README.md for the full workflow.
"""

import argparse

from apertus_common import Conversation, SystemPrompt
from datasets import load_dataset


def split_system(messages):
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return None, messages


def is_turn_based(messages):
    """True if, after an optional system message, user and assistant alternate."""
    roles = [message["role"] for message in split_system(messages)[1]]
    return bool(roles) and roles == ["user", "assistant"] * (len(roles) // 2)


def to_native(messages):
    if not is_turn_based(messages):
        raise ValueError("roles must alternate user, assistant after an optional system message")
    behavior, messages = split_system(messages)
    builder = Conversation.builder().system(
        SystemPrompt.build(thinking="medium", behavior=behavior)
    )
    for user, assistant in zip(messages[0::2], messages[1::2], strict=True):
        builder.user(user["content"]).reply(assistant["content"]).wait()
    return builder.build().to_json()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="Directory for the saved native HF dataset")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--num-proc", type=int, default=1, help="Worker processes")
    args = parser.parse_args()
    if args.num_proc < 1:
        parser.error("--num-proc must be positive")
    source = load_dataset("HuggingFaceH4/no_robots", split=args.split)
    # The datasets cache key ignores the apertus-common version, so a cached result
    # could hold an older native format; the mapping takes seconds, so always rerun it.
    turn_based = source.filter(
        lambda row: is_turn_based(row["messages"]),
        num_proc=args.num_proc,
        load_from_cache_file=False,
    )
    native = turn_based.map(
        lambda row: {"conversation_json": to_native(row["messages"])},
        remove_columns=["prompt", "messages", "category"],
        num_proc=args.num_proc,
        load_from_cache_file=False,
    )
    native.save_to_disk(args.output)
    skipped = len(source) - len(native)
    print(f"{len(native)} native conversations written to {args.output}; skipped {skipped}")


if __name__ == "__main__":
    main()
