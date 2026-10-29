#!/usr/bin/env python3
"""Deterministically annotate and quantify SWE/Terminal/Tool Use SFT records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mixture_analysis.judge import (
    DEFAULT_BASE_URL,
    HuggingFaceJudgeSettings,
    HuggingFaceSemanticJudge,
    JudgeConfigurationError,
    JudgeSettings,
    SemanticJudge,
)
from mixture_analysis.pipeline import run
from mixture_analysis.taxonomy import load_taxonomy


DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomies") / "swe_terminal_tool_use_v0.2.0.yaml"


def _parse_runtime_metadata(values: list[str], parser: argparse.ArgumentParser) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            parser.error("--runtime-metadata must use KEY=VALUE")
        key, value = item.split("=", 1)
        if not key or not value:
            parser.error("--runtime-metadata requires a non-empty key and value")
        if key in metadata:
            parser.error(f"duplicate --runtime-metadata key: {key}")
        metadata[key] = value
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", help="Parquet/JSON files or directories")
    parser.add_argument("--output", type=Path, required=True, help="New sidecar output directory")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--output-format", choices=("parquet", "jsonl"), default="parquet")
    parser.add_argument("--rows-per-shard", type=int, default=50_000)
    parser.add_argument("--input-batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Development-only maximum record count")
    parser.add_argument(
        "--num-input-shards",
        type=int,
        default=1,
        help="Deterministically split the global input row stream (default: 1)",
    )
    parser.add_argument(
        "--input-shard-index",
        type=int,
        default=0,
        help="Zero-based shard assigned to this process (default: 0)",
    )
    parser.add_argument(
        "--judge-model",
        help="Enable the semantic judge with this explicit model ID",
    )
    parser.add_argument(
        "--judge-provider",
        choices=("cscs", "huggingface"),
        default="cscs",
        help="Judge backend (default: cscs)",
    )
    parser.add_argument("--judge-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge-api-key-env", default="CSCS_INFERENCE_API_KEY")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        help="Generation batch size (default: 32 for CSCS, 4 for Hugging Face)",
    )
    parser.add_argument("--judge-timeout", type=float, default=90.0)
    parser.add_argument("--judge-max-input-chars", type=int, default=30_000)
    parser.add_argument("--judge-max-output-tokens", type=int, default=1_000)
    parser.add_argument(
        "--judge-retries",
        type=int,
        help="Retry count (default: 3 for CSCS, 0 for local Hugging Face)",
    )
    parser.add_argument("--judge-hf-revision", help="Pinned HF revision or commit hash")
    parser.add_argument(
        "--judge-hf-device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
    )
    parser.add_argument(
        "--judge-hf-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--judge-hf-max-input-tokens", type=int, default=8_192)
    parser.add_argument("--judge-hf-trust-remote-code", action="store_true")
    parser.add_argument("--judge-hf-local-files-only", action="store_true")
    parser.add_argument(
        "--runtime-metadata",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Stable execution-environment identity to record in run.json (repeatable)",
    )
    args = parser.parse_args()
    taxonomy = load_taxonomy(args.taxonomy)
    runtime_metadata = _parse_runtime_metadata(args.runtime_metadata, parser)
    judge = None
    if args.judge_model:
        batch_size = args.judge_batch_size
        if batch_size is None:
            batch_size = 4 if args.judge_provider == "huggingface" else 32
        retries = args.judge_retries
        if retries is None:
            retries = 0 if args.judge_provider == "huggingface" else 3
        if args.judge_provider == "huggingface":
            try:
                judge = HuggingFaceSemanticJudge(
                    taxonomy,
                    HuggingFaceJudgeSettings(
                        model=args.judge_model,
                        revision=args.judge_hf_revision,
                        batch_size=batch_size,
                        max_input_chars=args.judge_max_input_chars,
                        max_input_tokens=args.judge_hf_max_input_tokens,
                        max_output_tokens=args.judge_max_output_tokens,
                        device=args.judge_hf_device,
                        dtype=args.judge_hf_dtype,
                        trust_remote_code=args.judge_hf_trust_remote_code,
                        local_files_only=args.judge_hf_local_files_only,
                        retries=retries,
                    ),
                )
            except JudgeConfigurationError as exc:
                parser.error(str(exc))
        else:
            api_key = os.environ.get(args.judge_api_key_env)
            if not api_key:
                parser.error(
                    f"CSCS judge requires an API key in {args.judge_api_key_env}"
                )
            judge = SemanticJudge(
                taxonomy,
                JudgeSettings(
                    model=args.judge_model,
                    base_url=args.judge_base_url,
                    concurrency=args.judge_concurrency,
                    batch_size=batch_size,
                    timeout_seconds=args.judge_timeout,
                    max_input_chars=args.judge_max_input_chars,
                    max_output_tokens=args.judge_max_output_tokens,
                    retries=retries,
                ),
                api_key=api_key,
            )
    result = run(
        args.input,
        args.output,
        taxonomy,
        output_format=args.output_format,
        rows_per_shard=args.rows_per_shard,
        input_batch_size=args.input_batch_size,
        limit=args.limit,
        judge=judge,
        num_input_shards=args.num_input_shards,
        input_shard_index=args.input_shard_index,
        runtime_metadata=runtime_metadata,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
