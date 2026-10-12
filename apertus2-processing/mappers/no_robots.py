"""Map HuggingFaceH4/no_robots chats to native Apertus 2 conversations.

Accept an optional system message followed by alternating user/assistant pairs. The
source system message becomes `behavior`, thinking stays at the standard `medium`, and
each assistant answer ends with a wait. Under the default thinking convention every
answer also starts with an empty think (see README.md, "Writing a mapper"). Rows whose
roles do not alternate (in no_robots, user lines mislabelled as assistant answers) are
skipped and counted in `mapping.json`.

    uv run --no-sync python mappers/no_robots.py /data/no_robots-native --num-proc 8 --check

For Alps, use the container and scratch-environment launch command in README.md.
"""

from datasets import load_dataset

from apertus2_processing.mapping import from_chat, is_turn_based, mapper_parser, run_mapper


def to_native(row):
    if not is_turn_based(row["messages"]):
        raise ValueError("roles must alternate user, assistant after an optional system message")
    return from_chat(row["messages"])


def main():
    parser = mapper_parser(__doc__.splitlines()[0])
    parser.add_argument("--split", default="train", choices=("train", "test"))
    args = parser.parse_args()
    source = load_dataset("HuggingFaceH4/no_robots", split=args.split)
    run_mapper(source, to_native, args, keep_columns=["prompt_id"])


if __name__ == "__main__":
    main()
