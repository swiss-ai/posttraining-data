#!/usr/bin/env python3
"""
Convert openlifescienceai/medmcqa into the unified chat format for
explanation-style SFT.

MedMCQA rows contain:
  - question
  - opa/opb/opc/opd
  - cop: correct option index, with ClassLabel names a/b/c/d
  - exp: explanation text
  - subject_name, topic_name, choice_type

The HF test split has cop=-1 and blank explanations, so this converter uses
train+validation by default and skips unlabeled rows.
"""

import argparse
import hashlib
import html
import json
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


SRC = "openlifescienceai/medmcqa"
DEFAULT_HF_HOME = "/iopsstor/scratch/cscs/hyukhymenko/.cache/huggingface"
LABELS = ["A", "B", "C", "D"]
OPTION_FIELDS = ["opa", "opb", "opc", "opd"]
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
GLUED_URL_TLD_RE = re.compile(r"\.(?:com|org|net|gov|edu)([A-Z][A-Za-z].*)$")
BROKEN_PAGE_RE = re.compile(
    r"\b(?:404\s*(?:error)?\s*[:\-]?\s*(?:page\s+)?not\s+found|page\s+not\s+found)\b",
    flags=re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"\s+")


def stable_rng(sample: Dict[str, Any]) -> random.Random:
    seed_material = str(sample.get("id") or sample.get("question") or "")
    seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
    return random.Random(seed)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(value)).strip()


def remove_url(match: re.Match[str]) -> str:
    """Drop URLs while preserving text accidentally glued after a bare domain."""
    url = match.group(0)
    glued = GLUED_URL_TLD_RE.search(url)
    if glued:
        return f" {glued.group(1)}"
    return " "


def clean_medmcqa_text(value: Any) -> str:
    """Clean MedMCQA text without changing meaningful medical wording.

    Plain "not found" can be a valid part of a medical question, so this only
    removes explicit broken-page boilerplate when it appears with page/404
    context. It also strips HTML and URL residue observed in explanations.
    """
    text = clean_text(value)
    if not text:
        return ""

    text = html.unescape(text)
    # Some rows contain malformed entities such as "Hodgkin&;s".
    text = text.replace("&;", "'")
    text = text.replace("&nbsp;", " ")
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(remove_url, text)
    text = BROKEN_PAGE_RE.sub(" ", text)
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return clean_text(text)


def option_label(cop: int) -> str:
    return LABELS[cop]


def option_text(sample: Dict[str, Any], cop: int) -> str:
    return clean_medmcqa_text(sample.get(OPTION_FIELDS[cop], ""))


def format_question(sample: Dict[str, Any]) -> str:
    question = clean_medmcqa_text(sample.get("question", ""))
    options = [
        f"{label}. {clean_medmcqa_text(sample.get(field, ''))}"
        for label, field in zip(LABELS, OPTION_FIELDS)
    ]
    return "\n".join([question, "", *options]).strip()


def response_templates() -> list[str]:
    return [
        "The correct answer is {label}. {answer}.\n\n{explanation}",
        "{label}. {answer}\n\n{explanation}",
        "Answer: {label}. {answer}\n\n{explanation}",
        "I would choose {label}. {answer}.\n\n{explanation}",
        "The best choice is {label}: {answer}.\n\n{explanation}",
        "{label} is correct: {answer}.\n\n{explanation}",
        "It is {label}. {answer}.\n\n{explanation}",
        "{label}. {answer}\n\nWhy: {explanation}",
        "The answer is {label}: {answer}.\n\nHere is the reasoning: {explanation}",
        "Correct option: {label}. {answer}\n\n{explanation}",
    ]


def format_response(sample: Dict[str, Any], cop: int, keep_no_explanation: bool) -> str:
    label = option_label(cop)
    answer = option_text(sample, cop)
    explanation = clean_medmcqa_text(sample.get("exp"))

    if not explanation and keep_no_explanation:
        answer_only_templates = [
            "The correct answer is {label}. {answer}.",
            "{label}. {answer}",
            "Answer: {label}. {answer}",
            "The best choice is {label}: {answer}.",
        ]
        return stable_rng(sample).choice(answer_only_templates).format(
            label=label,
            answer=answer,
        )

    return stable_rng(sample).choice(response_templates()).format(
        label=label,
        answer=answer,
        explanation=explanation,
    )


def verifiable_answers(label: str, answer: str) -> list[str]:
    return [
        label,
        label.lower(),
        answer,
        f"{label}. {answer}",
        f"{label}: {answer}",
    ]


def conversation_id(sample: Dict[str, Any]) -> str:
    sample_id = sample.get("id")
    if sample_id:
        return f"openlifescienceai_medmcqa_{sample_id}"
    digest = hashlib.sha256(format_question(sample).encode()).hexdigest()[:16]
    return f"openlifescienceai_medmcqa_{digest}"


def convert_sample(sample: Dict[str, Any], keep_no_explanation: bool = False) -> Optional[Dict[str, Any]]:
    cop = sample.get("cop")
    if cop is None or int(cop) < 0 or int(cop) >= len(LABELS):
        return None
    cop = int(cop)

    explanation = clean_medmcqa_text(sample.get("exp"))
    if not explanation and not keep_no_explanation:
        return None

    label = option_label(cop)
    answer = option_text(sample, cop)
    response = format_response(sample, cop, keep_no_explanation=keep_no_explanation)

    return {
        "conversation_id": conversation_id(sample),
        "dataset_source": SRC,
        "original_metadata": {
            "id": sample.get("id"),
            "cop": cop,
            "correct_label": label,
            "correct_answer": answer,
            "choice_type": sample.get("choice_type"),
            "subject_name": sample.get("subject_name"),
            "topic_name": sample.get("topic_name"),
            "exp": explanation,
        },
        "created_timestamp": datetime.now(UTC).isoformat(),
        "system_prompt": {"content": "", "metadata": {}},
        "initial_prompt": {
            "role": "user",
            "content": format_question(sample),
            "metadata": {
                "subject_name": sample.get("subject_name"),
                "topic_name": sample.get("topic_name"),
                "choice_type": sample.get("choice_type"),
            },
        },
        "available_functions": [],
        "conversation_branches": [
            {
                "messages": [
                    {
                        "role": "assistant",
                        "parts": [
                            {
                                "type": "response",
                                "content": response,
                                "metadata": {},
                            },
                            {
                                "type": "verifiable-responses",
                                "answers": verifiable_answers(label, answer),
                            },
                        ],
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
            "operation": "convert_medmcqa_explanation",
            "script": "convert-medmcqa-explanation.py",
            "timestamp": datetime.now(UTC).isoformat(),
            "input_path": args.input,
            "output_path": str(output_path),
            "hf_home": os.environ.get("HF_HOME"),
            "splits": args.splits,
            "num_processes": args.num_proc,
            "limit": args.limit,
            "keep_no_explanation": args.keep_no_explanation,
            "stats": stats,
            "description": "Converted MedMCQA to explanation-style chat SFT format",
        }
    )
    metadata.setdefault("format", "chat_format_v1")
    metadata.setdefault("source_dataset", SRC)
    metadata.setdefault(
        "conversion_details",
        {
            "conversation_type": "medical_multiple_choice_explanation",
            "question_format": "question plus four labeled answer options",
            "response_format": "diversified answer plus explanation templates",
            "cleaning": "strips HTML tags, URLs, malformed entities, and explicit 404/page-not-found boilerplate",
            "verifiable_responses": "correct label and answer text variants",
            "format": "new_chat_format_with_parts",
        },
    )

    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset saved to {output_path}")
    print(f"Metadata saved to {metadata_file}")


def load_input(args: argparse.Namespace) -> Dataset:
    if args.input:
        print(f"Loading dataset from disk: {args.input}")
        loaded = load_from_disk(args.input)
        if isinstance(loaded, DatasetDict):
            datasets = [loaded[split] for split in args.splits if split in loaded]
        else:
            datasets = [loaded]
    else:
        os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
        print(f"Loading {SRC} from Hugging Face with HF_HOME={os.environ['HF_HOME']}")
        datasets = [load_dataset(SRC, split=split) for split in args.splits]

    if not datasets:
        raise ValueError(f"No requested splits were found: {args.splits}")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def convert_dataset(data: Dataset, args: argparse.Namespace) -> tuple[DatasetDict, Dict[str, int]]:
    if args.limit and args.limit > 0:
        data = data.select(range(min(args.limit, len(data))))
        print(f"Limited to {len(data)} samples")

    converted = []
    stats = {
        "input_samples": len(data),
        "converted_samples": 0,
        "skipped_unlabeled": 0,
        "skipped_no_explanation": 0,
    }

    for i, sample in enumerate(data):
        if i % 10000 == 0:
            print(f"Processing sample {i}/{len(data)}")

        cop = sample.get("cop")
        if cop is None or int(cop) < 0 or int(cop) >= len(LABELS):
            stats["skipped_unlabeled"] += 1
            continue
        if not clean_medmcqa_text(sample.get("exp")) and not args.keep_no_explanation:
            stats["skipped_no_explanation"] += 1
            continue

        converted_sample = convert_sample(sample, keep_no_explanation=args.keep_no_explanation)
        if converted_sample is not None:
            converted.append(converted_sample)

    stats["converted_samples"] = len(converted)
    print(f"Converted {stats['converted_samples']} samples")
    print(f"Skipped unlabeled: {stats['skipped_unlabeled']}")
    print(f"Skipped without explanation: {stats['skipped_no_explanation']}")

    return DatasetDict({"train": Dataset.from_list(converted)}), stats


def parse_splits(raw_splits: Iterable[str]) -> list[str]:
    splits = []
    for item in raw_splits:
        splits.extend(part.strip() for part in item.split(",") if part.strip())
    return splits


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MedMCQA to explanation-style SFT format")
    parser.add_argument("-i", "--input", default=None, help="Optional local DatasetDict path")
    parser.add_argument("-o", "--output", required=True, help="Output directory path")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="HF/local splits to convert. Defaults to train validation. Comma-separated values are allowed.",
    )
    parser.add_argument("--num-proc", type=int, default=8, help="Reserved for metadata consistency")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of input samples before filtering")
    parser.add_argument(
        "--keep-no-explanation",
        action="store_true",
        help="Keep rows with missing exp as answer-only samples. By default they are skipped.",
    )
    return parser.parse_args()


def main() -> None:
    args = cli()
    args.splits = parse_splits(args.splits)
    output_path = Path(args.output)

    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            sys.exit(0)

    data = load_input(args)
    print(f"Loaded {len(data)} input samples from splits: {args.splits}")

    dataset_dict, stats = convert_dataset(data, args)
    save_dataset_and_metadata(dataset_dict, output_path, args, stats)


if __name__ == "__main__":
    main()
