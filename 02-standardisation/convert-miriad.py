#!/usr/bin/env python3
"""
Convert miriad/miriad-4.4M into the unified chat format.

MIRIAD already has a QA-style schema:
  - qa_id
  - question
  - answer
  - paper_id, paper_url, paper_title
  - passage_text, passage_position
  - year, venue, specialty

The source dataset is large, so this converter streams from Hugging Face and
samples 100k rows by default.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


SRC = "miriad/miriad-4.4M"
DEFAULT_HF_HOME = "/iopsstor/scratch/cscs/hyukhymenko/.cache/huggingface"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def compact_text(value: Any) -> str:
    return " ".join(clean_text(value).split())


def conversation_id(sample: Dict[str, Any]) -> str:
    qa_id = clean_text(sample.get("qa_id"))
    if qa_id:
        return f"miriad_4_4m_{qa_id}"
    digest = hashlib.sha256(
        f"{sample.get('question', '')}\n{sample.get('answer', '')}".encode()
    ).hexdigest()[:16]
    return f"miriad_4_4m_{digest}"


def format_prompt(sample: Dict[str, Any], include_passage_context: bool) -> str:
    question = clean_text(sample.get("question"))
    if not include_passage_context:
        return question

    passage = clean_text(sample.get("passage_text"))
    if not passage:
        return question

    return (
        "Use the passage below to answer the question.\n\n"
        f"Passage:\n{passage}\n\n"
        f"Question:\n{question}"
    ).strip()


def original_metadata(sample: Dict[str, Any], keep_passage_text: bool) -> Dict[str, Any]:
    metadata = {
        "qa_id": sample.get("qa_id"),
        "paper_id": sample.get("paper_id"),
        "paper_url": sample.get("paper_url"),
        "paper_title": sample.get("paper_title"),
        "passage_position": sample.get("passage_position"),
        "year": sample.get("year"),
        "venue": sample.get("venue"),
        "specialty": sample.get("specialty"),
    }
    if keep_passage_text:
        metadata["passage_text"] = sample.get("passage_text")
    return metadata


def convert_sample(
    sample: Dict[str, Any],
    include_passage_context: bool = False,
    keep_passage_text: bool = True,
    add_verifiable_response: bool = False,
) -> Optional[Dict[str, Any]]:
    question = clean_text(sample.get("question"))
    answer = clean_text(sample.get("answer"))
    if not question or not answer:
        return None

    parts: list[Dict[str, Any]] = [
        {
            "type": "response",
            "content": answer,
            "metadata": {},
        }
    ]
    if add_verifiable_response:
        parts.append(
            {
                "type": "verifiable-responses",
                "answers": [answer],
            }
        )

    return {
        "conversation_id": conversation_id(sample),
        "dataset_source": SRC,
        "original_metadata": original_metadata(sample, keep_passage_text=keep_passage_text),
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": format_prompt(sample, include_passage_context=include_passage_context),
            "metadata": {
                "paper_id": sample.get("paper_id"),
                "paper_title": sample.get("paper_title"),
                "year": sample.get("year"),
                "venue": sample.get("venue"),
                "specialty": sample.get("specialty"),
            },
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": parts,
                    }
                ]
            }
        ],
    }


def load_existing_metadata(output_path: Path) -> Optional[Dict[str, Any]]:
    meta_file = output_path / "dataset_metadata.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def save_dataset_and_metadata(
    dataset_dict: DatasetDict,
    output_path: Path,
    args: argparse.Namespace,
    stats: Dict[str, int],
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_path))

    metadata = load_existing_metadata(output_path) or {}
    metadata.setdefault("processing_log", []).append(
        {
            "operation": "convert_miriad",
            "script": "convert-miriad.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "input_path": args.input,
            "output_path": str(output_path),
            "hf_home": os.environ.get("HF_HOME"),
            "split": args.split,
            "sample_size": args.sample_size,
            "shuffle": not args.no_shuffle,
            "shuffle_buffer_size": args.shuffle_buffer_size,
            "seed": args.seed,
            "include_passage_context": args.include_passage_context,
            "keep_passage_text": not args.drop_passage_text,
            "add_verifiable_response": args.add_verifiable_response,
            "stats": stats,
            "description": "Converted a 100k sample of MIRIAD QA data to unified chat format",
        }
    )
    metadata.setdefault("format", "chat_format_v1")
    metadata.setdefault("source_dataset", SRC)
    metadata.setdefault(
        "conversion_details",
        {
            "conversation_type": "medical_research_question_answering",
            "sampling": "streaming approximate shuffle, then take sample_size",
            "question_field": "question",
            "answer_field": "answer",
            "format": "new_chat_format_with_parts",
        },
    )

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


def iter_input(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.input:
        print(f"Loading dataset from disk: {args.input}")
        loaded = load_from_disk(args.input)
        if isinstance(loaded, DatasetDict):
            if args.split not in loaded:
                raise ValueError(f"Split {args.split!r} not found in {args.input}")
            return iter(loaded[args.split])
        return iter(loaded)

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    print(f"Streaming {SRC} from Hugging Face with HF_HOME={os.environ['HF_HOME']}")
    data = load_dataset(SRC, split=args.split, streaming=True)
    if not args.no_shuffle:
        data = data.shuffle(buffer_size=args.shuffle_buffer_size, seed=args.seed)
    return iter(data)


def convert_dataset(args: argparse.Namespace) -> tuple[DatasetDict, Dict[str, int]]:
    converted = []
    stats = {
        "seen_samples": 0,
        "converted_samples": 0,
        "skipped_empty_question": 0,
        "skipped_empty_answer": 0,
    }

    for sample in iter_input(args):
        stats["seen_samples"] += 1
        if stats["seen_samples"] % 10000 == 0:
            print(
                f"Seen {stats['seen_samples']} rows, "
                f"converted {len(converted)}/{args.sample_size}"
            )

        if not clean_text(sample.get("question")):
            stats["skipped_empty_question"] += 1
            continue
        if not clean_text(sample.get("answer")):
            stats["skipped_empty_answer"] += 1
            continue

        converted_sample = convert_sample(
            sample,
            include_passage_context=args.include_passage_context,
            keep_passage_text=not args.drop_passage_text,
            add_verifiable_response=args.add_verifiable_response,
        )
        if converted_sample is None:
            continue
        converted.append(converted_sample)

        if len(converted) >= args.sample_size:
            break

    stats["converted_samples"] = len(converted)
    print(f"Converted {stats['converted_samples']} samples")
    print(f"Seen {stats['seen_samples']} source rows")
    print(f"Skipped empty questions: {stats['skipped_empty_question']}")
    print(f"Skipped empty answers: {stats['skipped_empty_answer']}")

    return DatasetDict({"train": Dataset.from_list(converted)}), stats


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a 100k sample of MIRIAD QA to chat format")
    parser.add_argument("-i", "--input", default=None, help="Optional local Dataset/DatasetDict path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument("--split", default="train", help="Split to stream/load. Defaults to train.")
    parser.add_argument("--sample-size", type=int, default=100000, help="Number of converted samples to save")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for streaming shuffle")
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100000,
        help="Buffer size for approximate streaming shuffle",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Take the first sample-size rows instead of using streaming shuffle",
    )
    parser.add_argument(
        "--include-passage-context",
        action="store_true",
        help="Include passage_text in the user prompt before the question",
    )
    parser.add_argument(
        "--drop-passage-text",
        action="store_true",
        help="Do not store passage_text in original_metadata",
    )
    parser.add_argument(
        "--add-verifiable-response",
        action="store_true",
        help="Add the full free-form answer as a verifiable-responses part",
    )
    return parser.parse_args()


def main() -> None:
    args = cli()
    output_path = Path(args.output)

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    dataset_dict, stats = convert_dataset(args)
    save_dataset_and_metadata(dataset_dict, output_path, args, stats)


if __name__ == "__main__":
    main()
