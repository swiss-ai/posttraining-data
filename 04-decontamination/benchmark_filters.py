"""Shared benchmark filtering for decontamination jobs."""

import os
from typing import List, Sequence, Tuple


DEFAULT_EXCLUDED_BENCHMARK_PATTERNS = ()


def _normalize_pattern(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def get_excluded_benchmark_patterns() -> Tuple[str, ...]:
    """Return normalized benchmark-name patterns to skip."""
    patterns = list(DEFAULT_EXCLUDED_BENCHMARK_PATTERNS)
    env_patterns = os.environ.get("DECONTAM_EXCLUDE_BENCHMARK_PATTERNS", "")
    patterns.extend(pattern.strip() for pattern in env_patterns.split(",") if pattern.strip())
    return tuple(_normalize_pattern(pattern) for pattern in patterns if pattern)


def filter_benchmark_names(benchmark_names: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Filter benchmarks that should not participate in decontamination."""
    patterns = get_excluded_benchmark_patterns()
    kept = []
    excluded = []
    for benchmark_name in benchmark_names:
        normalized_name = _normalize_pattern(benchmark_name)
        if any(pattern in normalized_name for pattern in patterns):
            excluded.append(benchmark_name)
        else:
            kept.append(benchmark_name)
    return kept, excluded
