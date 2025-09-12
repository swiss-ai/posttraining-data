#!/usr/bin/env python3
"""
Merge decontamination reports from parallel jobs and create final filtered dataset.

This script combines contamination reports from multiple parallel decontamination jobs
and applies the final filtering to create a clean dataset.
"""

import os
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from datasets import load_from_disk, Dataset, DatasetDict


def load_existing_metadata(input_path: Path) -> Optional[Dict[str, Any]]:
    """Load existing dataset metadata if it exists."""
    metadata_file = Path(input_path) / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_dataset_and_metadata(train_data, output_path: Path, input_path: Path,
                              contaminated_ids: set, processed_benchmarks: list,
                              tokenizer_name: str, ngram_length: int, diff_threshold: float,
                              parallel_job_count: int):
    """Save filtered dataset and update metadata with processing log."""
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    print(f"Saving filtered dataset to {output_path}...")
    train_data.save_to_disk(str(output_path))

    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}

    # Calculate samples removed per split if DatasetDict
    samples_by_split = {}
    total_samples_after = 0

    if hasattr(train_data, 'keys'):  # DatasetDict
        for split_name, split_data in train_data.items():
            samples_by_split[split_name] = len(split_data)
            total_samples_after += len(split_data)
    else:  # Single Dataset
        total_samples_after = len(train_data)

    # Add processing log entry
    processing_entry = {
        "operation": "parallel_decontamination",
        "script": "merge_decontamination_reports.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokenizer_name": tokenizer_name,
        "ngram_length": ngram_length,
        "diff_threshold": diff_threshold,
        "benchmarks_processed": len(processed_benchmarks),
        "benchmark_names": processed_benchmarks,
        "contaminated_samples_removed": len(contaminated_ids),
        "samples_after_filtering": total_samples_after,
        "parallel_jobs_used": parallel_job_count,
        "decontamination_success": True
    }

    # Add split information if DatasetDict
    if samples_by_split:
        processing_entry["samples_by_split"] = samples_by_split

    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)

    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def collect_contamination_reports(reports_dir: Path) -> Dict[str, Dict]:
    """Collect all contamination reports from parallel jobs."""
    reports_dir = Path(reports_dir)
    if not reports_dir.exists():
        raise FileNotFoundError(f"Reports directory does not exist: {reports_dir}")

    # Find all contamination report files
    report_files = list(reports_dir.glob("*__contamination_report.json"))

    if not report_files:
        print(f"Warning: No contamination reports found in {reports_dir}")
        return {}

    print(f"Found {len(report_files)} contamination report files")

    all_reports = {}

    for report_file in sorted(report_files):
        try:
            with open(report_file, 'r') as f:
                report = json.load(f)

            # Extract benchmark name from filename
            benchmark_name = report_file.stem.replace("__contamination_report", "")
            all_reports[benchmark_name] = report

            if report:
                print(f"  - {benchmark_name}: {len(report)} contaminated samples")
            else:
                print(f"  - {benchmark_name}: no contamination")

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load report {report_file}: {e}")
            continue

    return all_reports


def merge_contamination_reports(all_reports: Dict[str, Dict]) -> tuple:
    """Merge all contamination reports and return combined results."""
    contaminated_ids = set()
    contamination_summary = []
    processed_benchmarks = []

    for benchmark_name, report in all_reports.items():
        processed_benchmarks.append(benchmark_name)
        contaminated_ids = contaminated_ids.union(set(report.keys()))

        if len(report) > 0:
            contamination_summary.append((benchmark_name, len(report)))

    return contaminated_ids, contamination_summary, processed_benchmarks


def validate_parallel_completion(reports_dir: Path, expected_jobs: Optional[int] = None) -> bool:
    """Validate that all parallel jobs completed successfully."""
    reports_dir = Path(reports_dir)

    # Count completion markers
    completion_files = list(reports_dir.glob("job_*.completed"))
    completed_jobs = len(completion_files)

    if expected_jobs:
        if completed_jobs < expected_jobs:
            print(f"Warning: Only {completed_jobs}/{expected_jobs} parallel jobs completed")
            missing_jobs = []
            for job_id in range(expected_jobs):
                marker_file = reports_dir / f"job_{job_id}.completed"
                if not marker_file.exists():
                    missing_jobs.append(job_id)

            if missing_jobs:
                print(f"Missing completion markers for jobs: {missing_jobs[:10]}")
                if len(missing_jobs) > 10:
                    print(f"... and {len(missing_jobs) - 10} more")

            return False
        else:
            print(f"All {expected_jobs} parallel jobs completed successfully")
            return True
    else:
        print(f"Found completion markers for {completed_jobs} parallel jobs")
        return completed_jobs > 0


def main():
    parser = argparse.ArgumentParser(
        description="Merge decontamination reports from parallel jobs and create final filtered dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge reports and create filtered dataset
  python merge_decontamination_reports.py \\
    /path/to/input/dataset \\
    /path/to/output/dataset \\
    /path/to/parallel_reports
  
  # With specific parameters
  python merge_decontamination_reports.py \\
    /path/to/input/dataset \\
    /path/to/output/dataset \\
    /path/to/parallel_reports \\
    --tokenizer_name "alehc/swissai-tokenizer" \\
    --expected-jobs 20
        """
    )

    parser.add_argument(
        "input_dataset",
        type=str,
        help="Path to input dataset directory"
    )
    parser.add_argument(
        "output_dataset",
        type=str,
        help="Path for final filtered dataset directory"
    )
    parser.add_argument(
        "reports_directory",
        type=str,
        help="Directory containing contamination reports from parallel jobs"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="alehc/swissai-tokenizer",
        help="Tokenizer name used for decontamination (for metadata)"
    )
    parser.add_argument(
        "--ngram_length",
        type=int,
        default=8,
        help="N-gram length used for decontamination (for metadata)"
    )
    parser.add_argument(
        "--diff_threshold",
        type=float,
        default=0.5,
        help="Difference threshold used for decontamination (for metadata)"
    )
    parser.add_argument(
        "--expected-jobs",
        type=int,
        help="Expected number of parallel jobs (for validation)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if some parallel jobs are missing"
    )

    args = parser.parse_args()

    # Validate input paths
    input_path = Path(args.input_dataset)
    if not input_path.exists():
        print(f"Error: Input dataset does not exist: {input_path}")
        return 1

    reports_dir = Path(args.reports_directory)
    if not reports_dir.exists():
        print(f"Error: Reports directory does not exist: {reports_dir}")
        return 1

    output_path = Path(args.output_dataset)

    print(f"=== MERGING DECONTAMINATION REPORTS ===")
    print(f"Input dataset:    {input_path}")
    print(f"Reports directory: {reports_dir}")
    print(f"Output dataset:   {output_path}")
    print("=" * 50)

    # Validate parallel job completion
    if not args.force:
        completion_ok = validate_parallel_completion(reports_dir, args.expected_jobs)
        if not completion_ok and args.expected_jobs:
            print("\nError: Not all parallel jobs completed successfully.")
            print("Use --force to proceed anyway, or check failed jobs and rerun.")
            return 1

    # Collect all contamination reports
    print(f"\n=== COLLECTING CONTAMINATION REPORTS ===")
    try:
        all_reports = collect_contamination_reports(reports_dir)
    except Exception as e:
        print(f"Error collecting reports: {e}")
        return 1

    if not all_reports:
        print("Error: No valid contamination reports found")
        return 1

    # Merge contamination results
    print(f"\n=== MERGING RESULTS ===")
    contaminated_ids, contamination_summary, processed_benchmarks = merge_contamination_reports(all_reports)

    print(f"Total benchmarks processed: {len(processed_benchmarks)}")
    print(f"Total contaminated samples to remove: {len(contaminated_ids)}")

    if contamination_summary:
        print(f"\nTop benchmarks with contamination:")
        contamination_summary.sort(key=lambda x: x[1], reverse=True)
        for benchmark, count in contamination_summary[:10]:  # Show top 10
            print(f"  {benchmark}: {count} samples")
        if len(contamination_summary) > 10:
            print(f"  ... and {len(contamination_summary) - 10} more benchmarks")

    # Load and filter training data
    print(f"\n=== FILTERING TRAINING DATA ===")
    print(f"Loading training data from: {input_path}")

    try:
        train_data = load_from_disk(str(input_path))
    except Exception as e:
        print(f"Error loading training data: {e}")
        return 1

    # Get original sample count
    if hasattr(train_data, 'keys'):  # DatasetDict
        original_samples = sum(len(split) for split in train_data.values())
        print(f"Original dataset: {original_samples} samples across {len(train_data)} splits")
    else:  # Single Dataset
        original_samples = len(train_data)
        print(f"Original dataset: {original_samples} samples")

    # Apply filtering
    print("Applying contamination filtering...")

    if hasattr(train_data, 'keys'):  # DatasetDict
        train_data = DatasetDict({
            k: v.filter(lambda x: x["conversation_id"] not in contaminated_ids)
            for k, v in train_data.items()
        })
        final_samples = sum(len(split) for split in train_data.values())
    else:  # Single Dataset
        train_data = train_data.filter(lambda x: x["conversation_id"] not in contaminated_ids)
        final_samples = len(train_data)

    removed_samples = original_samples - final_samples
    print(f"Removed {removed_samples:,} contaminated samples ({removed_samples / original_samples * 100:.2f}%)")
    print(f"Final dataset: {final_samples:,} samples")

    # Save filtered dataset with metadata
    print(f"\n=== SAVING FILTERED DATASET ===")
    parallel_job_count = len(all_reports)  # Estimate based on number of unique reports

    try:
        save_dataset_and_metadata(
            train_data,
            output_path,
            input_path,
            contaminated_ids,
            processed_benchmarks,
            args.tokenizer_name,
            args.ngram_length,
            args.diff_threshold,
            parallel_job_count
        )
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return 1

    print(f"\n=== MERGE COMPLETE ===")
    print(f"✅ Successfully merged {len(all_reports)} contamination reports")
    print(f"✅ Filtered dataset saved to: {output_path}")
    print(f"✅ Removed {removed_samples:,} contaminated samples from {len(processed_benchmarks)} benchmarks")

    return 0


if __name__ == "__main__":
    exit(main())
