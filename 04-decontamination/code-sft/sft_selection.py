"""Explicit user exclusions apply even when archived dataset directories remain."""

import json
from pathlib import Path


def user_exclusions(path=None):
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    entries = json.loads(path.read_text())["datasets"]
    if not isinstance(entries, list) or any(
        not isinstance(e, dict)
        or not isinstance(e.get("repo"), str)
        or "/" not in e["repo"]
        for e in entries
    ):
        raise ValueError("Invalid explicit SFT exclusions")
    return entries
