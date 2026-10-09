"""Shared conversation_id assignment for standardisation scripts.

Every sample gets a unique, non-empty string ID of the form
``<dataset_name>_<split>_<row:07d>`` (e.g. ``gsm8k_train_0000042``).
Decontamination (04) removes samples by this ID and refuses to run on empty,
duplicated or non-string IDs, so assign it once here, after any filtering the
converter does, and never recompute it later in the pipeline.

Use the exact dataset/variant name (normally the output folder name) so that
different variants of one source (e.g. OpenMathReasoning outputs) never share IDs.
A pre-existing ID (e.g. a source row id or a problem hash) is kept in
``original_metadata["original_conversation_id"]``.
"""

import json
from typing import Any, Dict, List


def make_conversation_id(dataset_name: str, split: str, row_idx: int) -> str:
    if not dataset_name or not split:
        raise ValueError("dataset_name and split must be non-empty")
    return f"{dataset_name}_{split}_{row_idx:07d}"


def _keep_original_id(sample: Dict[str, Any]) -> None:
    old_id = sample.get("conversation_id")
    if old_id is None or str(old_id) == "":
        return
    metadata = sample.get("original_metadata")
    if isinstance(metadata, dict):
        metadata["original_conversation_id"] = str(old_id)
    elif isinstance(metadata, str):
        try:
            parsed = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict):
            parsed["original_conversation_id"] = str(old_id)
            sample["original_metadata"] = json.dumps(parsed, ensure_ascii=False)
    elif metadata is None:
        sample["original_metadata"] = {"original_conversation_id": str(old_id)}


def assign_conversation_ids(samples: List[Dict[str, Any]], dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Set conversation_id on a list of converted samples in place (and return it)."""
    for row_idx, sample in enumerate(samples):
        _keep_original_id(sample)
        sample["conversation_id"] = make_conversation_id(dataset_name, split, row_idx)
    return samples


def add_conversation_ids(dataset, dataset_name: str, split: str, num_proc: int = None):
    """Return a HF Dataset with conversation_id set from the row index.

    Any previous ID is kept in original_metadata["original_conversation_id"] when
    original_metadata is a dict or a JSON string. To keep the Arrow schema uniform,
    previous IDs are only kept if every row has one.
    """
    def _assign(sample, row_idx):
        sample = dict(sample)
        _keep_original_id(sample)
        return {
            "conversation_id": make_conversation_id(dataset_name, split, row_idx),
            "original_metadata": sample.get("original_metadata"),
        }

    columns = dataset.column_names
    keep_previous = (
        "original_metadata" in columns
        and "conversation_id" in columns
        and all(cid is not None and str(cid) != "" for cid in dataset["conversation_id"])
    )
    if not keep_previous:
        return dataset.map(
            lambda _, row_idx: {"conversation_id": make_conversation_id(dataset_name, split, row_idx)},
            with_indices=True, num_proc=num_proc, desc="Assigning conversation_ids",
        )
    return dataset.map(_assign, with_indices=True, num_proc=num_proc, desc="Assigning conversation_ids")
