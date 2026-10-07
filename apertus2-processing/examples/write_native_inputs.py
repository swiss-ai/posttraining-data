"""Write a small native conversation as bare JSONL, wrapped JSONL and saved HF data.

This is an input-format example, not a legacy dataset converter.
"""

import argparse
import json
from pathlib import Path

from apertus_common import Conversation, SystemPrompt
from datasets import Dataset, DatasetDict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for example inputs")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    conversation = (
        Conversation.builder()
        .system(SystemPrompt.build(behavior="Be concise."))
        .user("Hello")
        .reply("Hi")
        .wait()
        .build(check="full")
    )
    native_json = conversation.to_json()
    record = {"record_id": "example-1", "conversation_json": native_json}
    args.output.mkdir(parents=True)
    (args.output / "native.jsonl").write_text(native_json + "\n", encoding="utf-8")
    (args.output / "wrapped.jsonl").write_text(
        json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    DatasetDict(
        {"train": Dataset.from_dict({key: [value] for key, value in record.items()})}
    ).save_to_disk(str(args.output / "hf"))
    print(f"Native example inputs written to {args.output}")


if __name__ == "__main__":
    main()
