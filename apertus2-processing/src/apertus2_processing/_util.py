"""Stable identities, atomic manifests and checksum validation."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def checksum(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def inventory(root):
    root = Path(root)
    if root.is_file():
        return [{"path": root.name, "size": root.stat().st_size, "sha256": checksum(root)}]
    return [
        {"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": checksum(p)}
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts
        )
    ]


def library_identity():
    import apertus_common

    root = Path(apertus_common.__file__).parent
    return {
        "version": importlib.metadata.version("apertus-common"),
        "sources": [
            {"path": str(p.relative_to(root)), "sha256": checksum(p)}
            for p in sorted(root.rglob("*.py"))
        ],
    }


def runtime_identity():
    return {
        name: importlib.metadata.version(name)
        for name in ("datasets", "pyarrow", "pydantic", "jsonschema", "tokenizers", "transformers")
    }


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(canonical(value) + "\n")
            target.flush()
            os.fsync(target.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def verify_files(root, files):
    root = Path(root).resolve()
    for entry in files:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or unsafe output file: {entry['path']}")
        if path.stat().st_size != entry["size"] or checksum(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
