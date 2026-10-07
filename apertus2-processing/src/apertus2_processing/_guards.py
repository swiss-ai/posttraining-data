"""Prepare content identities once; check inexpensive immutable-input guards in jobs."""

from pathlib import Path

from ._util import checksum, library_identity, runtime_identity


def paths(root):
    root = Path(root)
    if root.is_file():
        return [(root.name, root)]
    return [
        (str(p.relative_to(root)), p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in p.relative_to(root).parts
        )
    ]


def snapshot(root):
    result = []
    for name, path in paths(root):
        before = path.stat()
        sha = checksum(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"input changed while fingerprinting: {path}")
        result.append(
            {"path": name, "size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": sha}
        )
    return result


def verify_snapshot(root, files, *, full=False):
    current = paths(root)
    if [name for name, _ in current] != [entry["path"] for entry in files]:
        raise ValueError(f"input file inventory changed: {root}")
    for (_, path), expected in zip(current, files, strict=True):
        stat = path.stat()
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"input size/mtime changed: {path}")
        if full and checksum(path) != expected["sha256"]:
            raise ValueError(f"input checksum changed: {path}")


def verify_inputs(config, *, full=False, source_indices=None):
    seen = set()
    sources = (
        config["sources"]
        if source_indices is None
        else [config["sources"][i] for i in source_indices]
    )
    for source in sources:
        if source["path"] not in seen:
            verify_snapshot(source["path"], source["files"], full=full)
            seen.add(source["path"])
    if config["tokenizer"] is not None:
        verify_snapshot(config["tokenizer"], config["artifact"], full=full)


def verify_runtime(config):
    if library_identity() != config["library"] or runtime_identity() != config["runtime"]:
        raise ValueError("library or runtime differs from prepared run")
    current = [
        {"path": p.name, "sha256": checksum(p)} for p in sorted(Path(__file__).parent.glob("*.py"))
    ]
    if current != config["pipeline"]:
        raise ValueError("pipeline code differs from prepared run")
