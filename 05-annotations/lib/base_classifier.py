#!/usr/bin/env python3
"""
Base Classifier Framework

Provides a common base class for all LLM-based classification tasks, eliminating
code duplication and providing consistent behavior across all classifiers.
"""

import os
import json
import time
import asyncio
import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

from datasets import load_from_disk, Dataset
from tqdm import tqdm

from llm_classifier import LLMClassifier, RAMP_UP_INITIAL, RAMP_UP_INCREMENT, RAMP_UP_INTERVAL


class BaseClassifier(ABC):
    """
    Base class for all LLM-based classification tasks.
    
    Handles common functionality:
    - Dataset loading and processing
    - Progress tracking and resumption
    - Chunk-based processing
    - Metadata management
    - Error handling
    - Incremental saving
    """
    
    def __init__(self, classifier_name: str, template_filename: str, 
                 valid_categories: List[str], description: str):
        """
        Initialize the base classifier.
        
        Args:
            classifier_name: Name for metadata field (e.g., "refusal_classification")
            template_filename: Prompt template filename in prompts/ directory
            valid_categories: List of valid classification categories
            description: Human-readable description for help text
        """
        self.classifier_name = classifier_name
        self.template_filename = template_filename
        self.valid_categories = valid_categories
        self.description = description
        
        # Load template path
        self.template_path = Path(__file__).parent.parent / "prompts" / template_filename
        
        # Will be set during processing
        self.llm_classifier: Optional[LLMClassifier] = None
        self.prompt_template: Optional[str] = None
    
    @abstractmethod
    def collect_tasks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract classification tasks from a sample.
        
        This method must be implemented by subclasses to define:
        - Which messages/fields to classify
        - How to extract data for template placeholders
        - What context to include
        
        Args:
            sample: Chat format sample
            
        Returns:
            List of task dictionaries with:
            - "sample": Original sample reference
            - Other keys matching template placeholders
        """
        pass
    
    @abstractmethod
    def apply_results(self, tasks: List[Dict[str, Any]], results: List[Any], 
                     model: str) -> List[Dict[str, Any]]:
        """
        Apply classification results back to samples.
        
        This method must be implemented by subclasses to define:
        - Where to store metadata in the sample structure
        - How to structure the classification metadata
        - How to handle multiple tasks per sample
        
        Args:
            tasks: Original classification tasks
            results: Classification results from LLM
            model: Model name used for classification
            
        Returns:
            List of updated samples with classification metadata applied
        """
        pass
    
    def get_argument_parser(self) -> argparse.ArgumentParser:
        """
        Create standard argument parser for classification scripts.
        
        Subclasses can override this to add custom arguments.
        
        Returns:
            Configured ArgumentParser
        """
        script_name = f"classify_{self.classifier_name.replace('_classification', '')}.py"
        
        parser = argparse.ArgumentParser(
            description=f"{self.description}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
Examples:
  # Basic classification
  python {script_name} data/02-standardised/dataset-name \\
    --output data/04-annotations/dataset-name-{self.classifier_name.replace('_classification', '')}

  # Resume interrupted processing
  python {script_name} data/02-standardised/dataset-name \\
    --output data/04-annotations/dataset-name-{self.classifier_name.replace('_classification', '')} \\
    --resume

  # High concurrency processing
  python {script_name} data/02-standardised/dataset-name \\
    --output data/04-annotations/dataset-name-{self.classifier_name.replace('_classification', '')} \\
    --concurrent 300 --chunk-size 50000
            """
        )
        
        parser.add_argument(
            "input_path",
            help="Path to input dataset directory (chat format)"
        )
        parser.add_argument(
            "--output", "-o",
            required=True,
            help="Path to output dataset directory"
        )
        parser.add_argument(
            "--model",
            default="Qwen/Qwen3-32B",
            help="Model to use for classification (default: Qwen/Qwen3-32B)"
        )
        parser.add_argument(
            "--concurrent",
            type=int,
            default=50,
            help="Number of concurrent API requests (default: 50)"
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=10000,
            help="Number of samples to process per chunk (default: 10000)"
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from previous interrupted processing"
        )
        parser.add_argument(
            "--restart",
            action="store_true",
            help="Clear any existing progress and restart from beginning"
        )
        parser.add_argument(
            "--disable-adaptive",
            action="store_true",
            help="Disable adaptive concurrency (use fixed concurrency)"
        )
        
        return parser
    
    def save_progress(self, output_path: Path, completed_chunks: List[int], 
                     model: str, total_chunks: int):
        """Save progress to allow resuming interrupted processing."""
        progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
        progress_data = {
            "model": model,
            "classifier": self.classifier_name,
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "timestamp": datetime.now().isoformat()
        }
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self, output_path: Path, model: str) -> List[int]:
        """Load progress from previous run."""
        progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
        
        if not progress_file.exists():
            return []
        
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            if (progress_data.get("model") == model and 
                progress_data.get("classifier") == self.classifier_name):
                return progress_data.get("completed_chunks", [])
        except Exception as e:
            print(f"Warning: Failed to load progress file: {e}")
        
        return []
    
    def cleanup_progress(self, output_path: Path, model: str):
        """Remove progress file after successful completion."""
        progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
        if progress_file.exists():
            progress_file.unlink()
    
    def load_dataset(self, input_path: Path) -> Dataset:
        """Load and prepare dataset for processing."""
        print(f"Loading dataset from: {input_path}")
        dataset = load_from_disk(str(input_path))
        
        # Handle DatasetDict vs single Dataset
        if hasattr(dataset, 'keys'):
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                dataset = dataset['train']
                print(f"Using 'train' split")
            else:
                first_split = available_splits[0]
                dataset = dataset[first_split]
                print(f"Using '{first_split}' split")
        
        print(f"Dataset size: {len(dataset):,} samples")
        return dataset
    
    def initialize_classifier(self, model: str, concurrent: int, adaptive: bool) -> LLMClassifier:
        """Initialize the LLM classifier."""
        api_key = os.getenv("SWISSAI_API") or os.getenv("SWISSAI_API_KEY")
        if not api_key:
            raise ValueError("SWISSAI_API or SWISSAI_API_KEY environment variable is required")
        
        classifier = LLMClassifier(api_key, model, concurrent=concurrent, adaptive=adaptive)
        
        
        # Load prompt template
        self.prompt_template = classifier.load_prompt_template(self.template_path)
        
        return classifier
    
    def save_incremental_results(self, output_path: Path, samples: List[Dict[str, Any]], 
                               chunk_idx: int, input_metadata: Dict[str, Any]):
        """Save intermediate results for incremental processing."""
        output_path.mkdir(parents=True, exist_ok=True)
        chunk_file = output_path / f"chunk_{chunk_idx:06d}.json"
        
        with open(chunk_file, 'w') as f:
            json.dump({
                "chunk_index": chunk_idx,
                "sample_count": len(samples),
                "samples": samples,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    def analyze_incremental_results(self, output_path: Path) -> Tuple[List[int], int]:
        """
        Analyze existing incremental chunk files without loading/deleting them.
        
        Returns:
            Tuple of (completed_chunk_indices, total_samples_in_incremental_files)
        """
        chunk_files = sorted(output_path.glob("chunk_*.json"))
        completed_chunks = []
        total_samples = 0
        corrupted_files = []
        
        for chunk_file in chunk_files:
            try:
                # Extract chunk index from filename
                chunk_name = chunk_file.stem  # e.g., "chunk_000003"
                if chunk_name.startswith("chunk_"):
                    chunk_idx = int(chunk_name.split("_")[1])
                    
                    # Verify file is not empty
                    if chunk_file.stat().st_size == 0:
                        print(f"⚠️ Warning: Chunk file {chunk_file.name} is empty, will reprocess")
                        corrupted_files.append(chunk_file)
                        continue
                    
                    # Validate JSON structure and count samples
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        
                        # Validate expected structure
                        if not isinstance(chunk_data, dict):
                            print(f"⚠️ Warning: Chunk file {chunk_file.name} has invalid structure, will reprocess")
                            corrupted_files.append(chunk_file)
                            continue
                        
                        if "samples" not in chunk_data or not isinstance(chunk_data["samples"], list):
                            print(f"⚠️ Warning: Chunk file {chunk_file.name} missing valid samples, will reprocess")
                            corrupted_files.append(chunk_file)
                            continue
                        
                        sample_count = chunk_data.get("sample_count", len(chunk_data.get("samples", [])))
                        if sample_count == 0:
                            print(f"⚠️ Warning: Chunk file {chunk_file.name} has no samples, will reprocess")
                            corrupted_files.append(chunk_file)
                            continue
                        
                        # File is valid, add to completed chunks
                        completed_chunks.append(chunk_idx)
                        total_samples += sample_count
                        
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Chunk file {chunk_file.name} contains invalid JSON: {e}")
                print(f"  This chunk will be reprocessed")
                corrupted_files.append(chunk_file)
            except Exception as e:
                print(f"⚠️ Warning: Failed to analyze chunk file {chunk_file.name}: {e}")
                print(f"  This chunk will be reprocessed")
                corrupted_files.append(chunk_file)
        
        # Optionally remove corrupted files to avoid confusion
        if corrupted_files:
            print(f"\n⚠️ Found {len(corrupted_files)} corrupted chunk file(s)")
            for corrupted_file in corrupted_files:
                try:
                    corrupted_file.unlink()
                    print(f"  Removed corrupted file: {corrupted_file.name}")
                except Exception as e:
                    print(f"  Failed to remove corrupted file {corrupted_file.name}: {e}")
        
        return sorted(completed_chunks), total_samples

    def load_incremental_results(self, output_path: Path, cleanup_files: bool = True) -> List[Dict[str, Any]]:
        """
        Load and combine all incremental chunk results.
        
        Args:
            output_path: Path containing chunk files
            cleanup_files: Whether to delete chunk files after loading (default: True)
        """
        chunk_files = sorted(output_path.glob("chunk_*.json"))
        all_samples = []
        
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r') as f:
                    chunk_data = json.load(f)
                    all_samples.extend(chunk_data["samples"])
            except Exception as e:
                print(f"Warning: Failed to load chunk file {chunk_file}: {e}")
        
        # Clean up chunk files after successful load (if requested)
        if cleanup_files:
            for chunk_file in chunk_files:
                try:
                    chunk_file.unlink()
                except Exception:
                    pass
        
        return all_samples
    
    def update_metadata(self, input_path: Path, output_path: Path, processing_stats: Dict[str, Any]):
        """Update dataset metadata with processing information."""
        # Load original metadata
        metadata_file = input_path / "dataset_metadata.json"
        original_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                original_metadata = json.load(f)
        
        # Create processing log entry
        processing_entry = {
            "operation": self.classifier_name,
            "script": f"classify_{self.classifier_name.replace('_classification', '')}.py",
            "timestamp": datetime.now().isoformat(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            **processing_stats
        }
        
        # Update metadata
        updated_metadata = {
            **original_metadata,
            "processing_log": original_metadata.get("processing_log", []) + [processing_entry]
        }
        
        # Save updated metadata
        output_metadata_file = output_path / "dataset_metadata.json"
        with open(output_metadata_file, 'w') as f:
            json.dump(updated_metadata, f, indent=2)
    
    async def process_dataset(self, input_path: Path, output_path: Path, model: str,
                            concurrent: int = 50, chunk_size: int = 10000,
                            resume: bool = False, restart: bool = False,
                            disable_adaptive: bool = False) -> Dict[str, Any]:
        """
        Main dataset processing pipeline.
        
        Args:
            input_path: Path to input dataset
            output_path: Path to output dataset
            model: Model to use for classification
            concurrent: Starting number of concurrent requests
            chunk_size: Number of samples to process per chunk
            resume: Resume from previous run
            restart: Clear progress and restart
            disable_adaptive: Disable adaptive concurrency
            
        Returns:
            Dictionary with processing statistics
        """
        # Load dataset
        dataset = self.load_dataset(input_path)
        
        # Initialize classifier
        self.llm_classifier = self.initialize_classifier(model, concurrent, not disable_adaptive)
        
        # Display configuration
        if not disable_adaptive:
            print(f"🔄 Adaptive concurrency enabled (starting: {concurrent})")
        else:
            print(f"⚡ Fixed concurrency mode: {concurrent} requests")
        
        # Handle progress management
        total_chunks = (len(dataset) + chunk_size - 1) // chunk_size
        
        if restart:
            self.cleanup_progress(output_path, model)
            completed_chunks = []
            print("Restarting from beginning (clearing any previous progress)")
        elif resume:
            if output_path.exists():
                # Analyze existing incremental files to determine what's already completed
                print("Analyzing previous incremental results...")
                completed_chunks, existing_sample_count = self.analyze_incremental_results(output_path)
                if completed_chunks:
                    print(f"Found {len(completed_chunks)} completed chunks with {existing_sample_count:,} samples")
                    print(f"Will resume from chunk {max(completed_chunks) + 1 if completed_chunks else 0}")
                else:
                    # No incremental files found, check progress file
                    completed_chunks = self.load_progress(output_path, model)
            else:
                completed_chunks = self.load_progress(output_path, model)
            
            if completed_chunks:
                print(f"Resuming from previous run: {len(completed_chunks)}/{total_chunks} chunks completed")
            else:
                print("No previous progress found, starting from beginning")
        else:
            completed_chunks = []
        
        # Processing statistics
        total_classifications = 0
        successful_classifications = 0
        failed_classifications = 0
        total_sample_errors = 0
        
        print(f"Processing dataset in {total_chunks} chunks of {chunk_size:,} samples...")
        
        # Process dataset in chunks
        for chunk_idx, chunk_start in enumerate(range(0, len(dataset), chunk_size)):
            # Skip completed chunks if resuming
            if resume and chunk_idx in completed_chunks:
                print(f"Skipping completed chunk {chunk_idx + 1}/{total_chunks}")
                continue
            
            chunk_end = min(chunk_start + chunk_size, len(dataset))
            chunk = [dataset[i] for i in range(chunk_start, chunk_end)]
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{total_chunks}: samples {chunk_start+1}-{chunk_end}")
            
            # Collect classification tasks from this chunk
            chunk_tasks = []
            sample_errors = []
            for i, sample in enumerate(tqdm(chunk, desc="Collecting tasks")):
                try:
                    tasks = self.collect_tasks(sample)
                    if isinstance(tasks, list):
                        chunk_tasks.extend(tasks)
                    elif tasks is not None:
                        chunk_tasks.append(tasks)
                except Exception as e:
                    sample_errors.append((chunk_start + i, str(e)))
                    continue
            
            if sample_errors:
                print(f"Warning: Failed to process {len(sample_errors)} samples in this chunk:")
                for sample_idx, error in sample_errors[:3]:
                    print(f"  Sample {sample_idx}: {error}")
                if len(sample_errors) > 3:
                    print(f"  ... and {len(sample_errors) - 3} more errors")
            
            if not chunk_tasks:
                print("No classification tasks found in this chunk")
                # Save chunk even if no classification tasks (to preserve all samples)
                self.save_incremental_results(output_path, chunk, chunk_idx, {})
                
                # Verify the chunk file was saved properly before marking as complete
                chunk_file = output_path / f"chunk_{chunk_idx:06d}.json"
                try:
                    if chunk_file.exists() and chunk_file.stat().st_size > 0:
                        with open(chunk_file, 'r') as f:
                            chunk_data = json.load(f)
                            if (isinstance(chunk_data, dict) and "samples" in chunk_data):
                                completed_chunks.append(chunk_idx)
                                self.save_progress(output_path, completed_chunks, model, total_chunks)
                                print(f"✓ Chunk {chunk_idx + 1}/{total_chunks} (no tasks) saved and verified")
                            else:
                                print(f"⚠️ Warning: Chunk {chunk_idx + 1} file has invalid structure")
                    else:
                        print(f"⚠️ Warning: Chunk {chunk_idx + 1} file not saved properly")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to verify chunk {chunk_idx + 1}: {e}")
                
                continue
            
            print(f"Found {len(chunk_tasks)} items to classify")
            
            # Classify the tasks
            print("Sending classification requests...")
            results = await self._classify_tasks(chunk_tasks)
            
            # Count results
            chunk_successful = sum(1 for r in results if r.success)
            chunk_failed = sum(1 for r in results if not r.success)
            
            total_classifications += len(results)
            successful_classifications += chunk_successful
            failed_classifications += chunk_failed
            total_sample_errors += len(sample_errors)
            
            # Display chunk metrics
            chunk_metrics = self.llm_classifier.metrics.get_metrics()
            print(f"Chunk results: {chunk_successful} successful, {chunk_failed} failed")
            print(f"Chunk metrics: {chunk_metrics['requests_per_minute']:.1f} req/min, "
                  f"{chunk_metrics['success_rate']*100:.1f}% success rate")
            
            # Apply results to samples
            print("Applying classification results...")
            updated_samples = self.apply_results(chunk_tasks, results, model)
            
            # Add samples that weren't classified
            updated_sample_ids = {s["conversation_id"] for s in updated_samples}
            for sample in chunk:
                if sample["conversation_id"] not in updated_sample_ids:
                    updated_samples.append(sample)
            
            # Always save chunk results immediately
            self.save_incremental_results(output_path, updated_samples, chunk_idx, {})
            
            # Verify the chunk file was actually written successfully before marking as complete
            chunk_file = output_path / f"chunk_{chunk_idx:06d}.json"
            try:
                if chunk_file.exists() and chunk_file.stat().st_size > 0:
                    # Verify the file contains valid JSON with expected structure
                    with open(chunk_file, 'r') as f:
                        chunk_data = json.load(f)
                        if (isinstance(chunk_data, dict) and 
                            "samples" in chunk_data and 
                            isinstance(chunk_data["samples"], list) and
                            len(chunk_data["samples"]) > 0):
                            # File is valid, mark chunk as complete
                            completed_chunks.append(chunk_idx)
                            self.save_progress(output_path, completed_chunks, model, total_chunks)
                            print(f"✓ Chunk {chunk_idx + 1}/{total_chunks} saved and verified successfully")
                        else:
                            print(f"⚠️ Warning: Chunk {chunk_idx + 1} file has invalid structure, not marking as complete")
                            print(f"  Will need to reprocess this chunk on next run")
                else:
                    print(f"⚠️ Warning: Chunk {chunk_idx + 1} file not saved properly or is empty, not marking as complete")
                    print(f"  Will need to reprocess this chunk on next run")
            except (json.JSONDecodeError, Exception) as e:
                print(f"⚠️ Warning: Failed to verify chunk {chunk_idx + 1} file: {e}")
                print(f"  Chunk not marked as complete, will need to reprocess on next run")
        
        # Final dataset creation - always load from incremental files
        print(f"\nCombining incremental results...")
        all_samples = self.load_incremental_results(output_path, cleanup_files=False)  # Don't cleanup yet!
        
        print(f"\nCreating output dataset with {len(all_samples):,} samples...")
        try:
            output_dataset = Dataset.from_list(all_samples)
            
            # Save dataset
            output_path.mkdir(parents=True, exist_ok=True)
            output_dataset.save_to_disk(str(output_path))
            
            # Only clean up chunks AFTER successful save
            print("Cleaning up intermediate chunk files...")
            chunk_files = sorted(output_path.glob("chunk_*.json"))
            for chunk_file in chunk_files:
                try:
                    chunk_file.unlink()
                except Exception:
                    pass
        except Exception as e:
            print(f"\n❌ ERROR: Failed to create final dataset: {e}")
            print(f"Chunk files preserved in {output_path} for debugging")
            raise
        
        # Update metadata
        processing_stats = {
            "model": model,
            "api_provider": "swiss_ai",
            "samples_processed": len(dataset),
            "samples_with_errors": total_sample_errors,
            "items_classified": total_classifications,
            "classification_success": successful_classifications,
            "classification_failed": failed_classifications,
            "total_tokens_used": self.llm_classifier.total_tokens_used,
            "concurrent": concurrent,
            "adaptive_enabled": not disable_adaptive,
            "chunk_size": chunk_size
        }
        
        self.update_metadata(input_path, output_path, processing_stats)
        
        # Cleanup progress file on successful completion
        self.cleanup_progress(output_path, model)
        
        # Display final results
        classification_type = self.classifier_name.replace('_classification', '').upper()
        print(f"\n=== {classification_type} CLASSIFICATION COMPLETED ===")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Model: {model}")
        print(f"Samples processed: {len(dataset):,}")
        if total_sample_errors > 0:
            print(f"Samples with errors: {total_sample_errors:,}")
        print(f"Items classified: {total_classifications:,}")
        print(f"Success rate: {successful_classifications/total_classifications*100:.1f}%" if total_classifications > 0 else "Success rate: N/A")
        print(f"Total tokens used: {self.llm_classifier.total_tokens_used:,}")
        
        # Final performance metrics
        final_metrics = self.llm_classifier.metrics.get_metrics()
        print(f"\n=== FINAL PERFORMANCE METRICS ===")
        print(f"Average requests per minute: {final_metrics['requests_per_minute']:.1f}")
        print(f"Overall success rate: {final_metrics['success_rate']*100:.1f}%")
        print(f"Total failed requests: {final_metrics['failed_requests']}")
        if final_metrics['total_requests'] > 0:
            print(f"Total requests tracked: {final_metrics['total_requests']}")
        
        return processing_stats
    
    async def _classify_tasks(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Classify a list of tasks using the appropriate API."""
        # Check if we can use the batch API or need flexible API
        if all(isinstance(task.get("question"), str) and isinstance(task.get("answer"), str) 
               for task in tasks):
            # Use legacy batch API
            return await self.llm_classifier.classify_batch(
                tasks, self.prompt_template, self.valid_categories
            )
        else:
            # Use flexible API with adaptive concurrency and ramp-up
            return await self._classify_tasks_flexible_adaptive(tasks)
            
            return results
    
    async def _classify_tasks_flexible_adaptive(self, tasks: List[Dict[str, Any]]) -> List[Any]:
        """Flexible API with adaptive concurrency and ramp-up using task set control loop."""
        # Reset adaptation timer for new chunk to allow stabilization
        self.llm_classifier.last_adaptation = time.time()
        self.llm_classifier.stability_zone_entered = False  # Reset stability zone flag for new chunk
        
        # Set up ramp-up for this chunk
        ramp_up_target = self.llm_classifier.current_concurrent
        if ramp_up_target > RAMP_UP_INITIAL:
            self.llm_classifier.current_concurrent = RAMP_UP_INITIAL
            is_ramping_up = True
            last_ramp_up_increase = time.time()
            print(f"🔺 Starting ramp-up: {RAMP_UP_INITIAL} → {ramp_up_target} concurrent requests")
        else:
            is_ramping_up = False
        
        in_flight_tasks = set()
        work_queue = list(enumerate(tasks))  # Keep track of original indices
        results = [None] * len(tasks)  # Pre-allocate results list
        
        # Progress tracking
        completed_count = 0
        pbar = tqdm(total=len(tasks), desc="Classifying",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]')
        
        while work_queue or in_flight_tasks:
            # Start new tasks up to current limit
            while len(in_flight_tasks) < self.llm_classifier.current_concurrent and work_queue:
                idx, task = work_queue.pop(0)
                
                # Create task with index tracking
                flight_task = asyncio.create_task(self._classify_with_index_flexible(idx, task))
                in_flight_tasks.add(flight_task)
            
            # Wait for first completion or timeout
            if in_flight_tasks:
                done, in_flight_tasks = await asyncio.wait(
                    in_flight_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0  # Check every second
                )
                
                # Process completed tasks
                for task in done:
                    idx, result = await task
                    results[idx] = result
                    completed_count += 1
                    pbar.update(1)
                    
                    # Handle API failures immediately (but not during ramp-up)
                    # Only reduce concurrency for API errors, not parsing errors
                    if not result.success and result.api_error and not is_ramping_up:
                        # Calculate remaining tasks for stability zone check
                        total_remaining = len(work_queue) + len(in_flight_tasks)
                        await self._handle_failure_immediate(total_remaining)
                
                # Display metrics periodically
                current_time = time.time()
                if current_time - self.llm_classifier.last_metrics_display >= self.llm_classifier.metrics_display_interval:
                    metrics = self.llm_classifier.metrics.get_adaptive_metrics()
                    error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
                    postfix = {
                        'req/min': f"{metrics['requests_per_minute']:.1f}",
                        'success': f"{metrics['success_rate']*100:.1f}%",
                        'failed': metrics['failed_requests'],
                        'avg_dur': f"{metrics.get('avg_latency_seconds', 0):.1f}s",
                        'error_rate': f"{error_rate*100:.1f}%",
                        'concurrent': self.llm_classifier.current_concurrent
                    }
                    pbar.set_postfix(postfix)
                    self.llm_classifier.last_metrics_display = current_time
                
                # Check for ramp-up increase
                if is_ramping_up:
                    current_time = time.time()
                    if current_time - last_ramp_up_increase >= RAMP_UP_INTERVAL:
                        old_concurrent = self.llm_classifier.current_concurrent
                        self.llm_classifier.current_concurrent = min(
                            self.llm_classifier.current_concurrent + RAMP_UP_INCREMENT,
                            ramp_up_target
                        )
                        if self.llm_classifier.current_concurrent != old_concurrent:
                            print(f"🔺 Ramping up: {old_concurrent} → {self.llm_classifier.current_concurrent}")
                        last_ramp_up_increase = current_time
                        
                        if self.llm_classifier.current_concurrent >= ramp_up_target:
                            is_ramping_up = False
                            print(f"🔺 Ramp-up complete: {self.llm_classifier.current_concurrent} concurrent requests")
                            # Reset adaptation timer after ramp-up completes
                            self.llm_classifier.last_adaptation = time.time()
            
            # Check for periodic adjustment (but not during ramp-up)
            if not is_ramping_up:
                # Consider both queued and in-flight tasks for total remaining
                total_remaining = len(work_queue) + len(in_flight_tasks)
                self._check_concurrency_adjustment(total_remaining)
        
        pbar.close()
        return results
    
    async def _handle_failure_immediate(self, remaining_tasks: int = 0):
        """Immediately reduce concurrency on any failure."""
        # Master control: skip ALL adjustments if in stability zone
        if self.llm_classifier.stability_zone_entered:
            return  # Stability zone active - preserve learned optimal concurrency
        
        old_concurrent = self.llm_classifier.current_concurrent
        self.llm_classifier.current_concurrent = max(
            self.llm_classifier.current_concurrent - 0, 
            self.llm_classifier.min_concurrent
        )
        
        if self.llm_classifier.current_concurrent < old_concurrent:
            print(f"🔻 Reducing concurrency: {old_concurrent} → {self.llm_classifier.current_concurrent} (immediate failure response)")
    
    def _check_concurrency_adjustment(self, remaining_tasks: int = 0):
        """Check if 60s have passed and adjust concurrency based on metrics."""
        if not self.llm_classifier.adaptive_enabled:
            return
            
        now = time.time()
        time_since_last = now - self.llm_classifier.last_adaptation
        
        if time_since_last < self.llm_classifier.adaptation_interval:
            return  # Too soon to adapt
        
        # Check if we should enter stability zone (≤500 tasks remaining)
        if not self.llm_classifier.stability_zone_entered and remaining_tasks <= 500 and remaining_tasks > 0:
            print(f"⏸️  Entered stability zone: skipping concurrency adjustments for final {remaining_tasks} tasks")
            self.llm_classifier.stability_zone_entered = True
            return  # Skip this and all future adjustments for this chunk
        
        # Master control: skip ALL adjustments if in stability zone
        if self.llm_classifier.stability_zone_entered:
            return  # Stability zone active - preserve learned optimal concurrency
            
        metrics = self.llm_classifier.metrics.get_adaptive_metrics()
        error_rate = metrics['failed_requests'] / max(metrics['total_requests'], 1)
        old_concurrent = self.llm_classifier.current_concurrent
        
        # Periodic optimization based on error rate
        if error_rate <= 0.01 and metrics.get('avg_latency_seconds', 0) < 20.0:  # Only increase on <1% error rate and low latency
            # Explore higher concurrency (+20)
            self.llm_classifier.current_concurrent += 20
            print(f"🔺 Increasing concurrency: {old_concurrent} → {self.llm_classifier.current_concurrent} ({error_rate*100:.1f}% error rate, avg {metrics.get('avg_latency_seconds', 0):.1f}s)")
        else:
            # Fall back to calculated optimal
            optimal = max(metrics['optimal_concurrent'], self.llm_classifier.min_concurrent)
            self.llm_classifier.current_concurrent = min(optimal, self.llm_classifier.current_concurrent)  # Don't increase if errors
            if self.llm_classifier.current_concurrent < old_concurrent:
                print(f"🔻 Reducing to optimal: {old_concurrent} → {self.llm_classifier.current_concurrent} (error rate: {error_rate*100:.1f}%)")
        
        self.llm_classifier.last_adaptation = now
    
    async def _classify_with_index_flexible(self, idx: int, task: Dict[str, Any]) -> Tuple[int, Any]:
        """Classify single task using flexible API and return with its index."""
        # Extract template data (remove non-template keys)
        template_data = {k: v for k, v in task.items() 
                       if k not in ["sample"]}
        try:
            result = await asyncio.wait_for(
                self.llm_classifier.classify_single_flexible(
                    template_data, self.prompt_template, self.valid_categories
                ),
                timeout=60.0  # 60s max per request
            )
            return idx, result
        except asyncio.TimeoutError:
            # Import ClassificationResult for timeout handling
            from llm_classifier import ClassificationResult
            # Create timeout error result
            timeout_result = ClassificationResult(
                classification=self.valid_categories[-1] if self.valid_categories else "error",
                reasoning="Request timed out after 60 seconds",
                success=False,
                error="Request timeout",
                tokens_used=0,
                api_error=True  # Treat as API error to trigger concurrency reduction
            )
            return idx, timeout_result
    
    def run_classification(self, args: Optional[argparse.Namespace] = None):
        """
        Main entry point for classification scripts.
        
        Args:
            args: Parsed command line arguments (if None, will parse from sys.argv)
        """
        if args is None:
            parser = self.get_argument_parser()
            args = parser.parse_args()
        
        # Display configuration
        classification_type = self.classifier_name.replace('_classification', '').title()
        print(f"=== {classification_type} Classification Configuration ===")
        print(f"  Input: {args.input_path}")
        print(f"  Output: {args.output}")
        print(f"  Model: {args.model}")
        print(f"  Concurrency: {args.concurrent}")
        print(f"  Chunk size: {args.chunk_size}")
        print("=" * 50)
        
        # Validate input path
        input_path = Path(args.input_path)
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return 1
        
        output_path = Path(args.output)
        
        # Validate resume/restart flags
        if args.resume and args.restart:
            print("Error: Cannot use both --resume and --restart flags")
            return 1
        
        # Check if output exists and handle accordingly
        if output_path.exists():
            progress_file = output_path / f".progress_{args.model.replace('/', '_').replace('-', '_')}.json"
            
            if progress_file.exists():
                if not args.resume and not args.restart:
                    print(f"Previous processing detected for model '{args.model}'")
                    print("Use --resume to continue or --restart to start over")
                    return 1
            elif not args.restart:
                print(f"Output directory already exists: {output_path}")
                print("Use --restart to overwrite existing output")
                return 1
        
        # Run classification
        try:
            asyncio.run(self.process_dataset(
                input_path, output_path, args.model,
                args.concurrent, args.chunk_size,
                args.resume, args.restart, args.disable_adaptive
            ))
            return 0
        except KeyboardInterrupt:
            print("\nClassification interrupted by user")
            return 1
        except Exception as e:
            print(f"Error: {e}")
            return 1