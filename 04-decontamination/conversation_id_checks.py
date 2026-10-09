"""Shared conversation_id validation for decontamination jobs.

Decontamination removes samples by conversation_id, and contamination reports are
JSON files whose keys are always strings. Empty, duplicated or non-string IDs make
that filtering silently remove nothing (or everything), so we refuse to run on them.
IDs are assigned during standardisation (02-standardisation/conversation_ids.py).
"""

from collections import Counter
from typing import Sequence


def validate_conversation_ids(conversation_ids: Sequence, dataset_path: str) -> None:
    """Raise ValueError if any conversation_id is non-string, empty or duplicated."""
    non_string = [cid for cid in conversation_ids if not isinstance(cid, str)]
    empty = sum(1 for cid in conversation_ids if isinstance(cid, str) and not cid.strip())
    duplicates = {cid: n for cid, n in Counter(conversation_ids).items() if n > 1}

    problems = []
    if non_string:
        types = sorted({type(cid).__name__ for cid in non_string})
        problems.append(f"{len(non_string)} non-string IDs (types: {', '.join(types)})")
    if empty:
        problems.append(f"{empty} empty IDs")
    if duplicates:
        examples = ", ".join(repr(cid) for cid in list(duplicates)[:5])
        problems.append(f"{len(duplicates)} duplicated IDs covering "
                        f"{sum(duplicates.values())} rows (e.g. {examples})")
    if problems:
        raise ValueError(
            f"Invalid conversation_id values in {dataset_path}: " + "; ".join(problems)
            + ". Re-run the dataset's 02-standardisation converter so that every sample "
              "has a unique, non-empty string conversation_id."
        )
