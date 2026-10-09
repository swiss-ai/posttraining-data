"""Read a merged check report."""

import json
from collections import defaultdict
from pathlib import Path

FAILING = {"error", "unevaluated"}


def issue_rows(run, *, failed_only=False):
    """Map each input source to the sorted rows of its samples with issues.

    Rows are zero-based dataset indices (for an HF dataset, within the checked split).
    With `failed_only`, keep only samples that failed: those with an error or an
    unevaluated finding. Warnings alone do not fail a sample.
    """
    rows = defaultdict(set)
    with open(Path(run) / "issues.jsonl", encoding="utf-8") as report:
        for line in report:
            issue = json.loads(line)
            if not failed_only or issue["severity"] in FAILING:
                rows[issue["source"]].add(issue["row"])
    return {source: sorted(found) for source, found in rows.items()}
