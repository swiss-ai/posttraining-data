#!/usr/bin/env python3
"""
Refusal Classification Script

Classifies assistant messages in chat format datasets to identify refusal responses 
where the assistant declines to provide information or assistance due to safety, 
ethical, or capability constraints.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from datasets import load_from_disk, Dataset
from tqdm import tqdm

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from llm_classifier import LLMClassifier, extract_message_context, should_classify_message


def collect_classification_tasks(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Collect all assistant messages that need refusal classification from a sample.
    
    Args:
        sample: Chat format sample
        
    Returns:
        List of classification tasks with message and context
    """
    tasks = []
    
    # Build conversation message list for context extraction
    conversation_messages = []
    
    # Add system prompt if present
    if sample.get("system_prompt"):
        conversation_messages.append({
            "role": "system",
            "content": sample["system_prompt"]["content"]
        })
    
    # Add initial prompt
    if sample.get("initial_prompt"):
        conversation_messages.append({
            "role": sample["initial_prompt"]["role"],
            "content": sample["initial_prompt"]["content"]
        })
    
    # Add messages from all conversation branches
    for branch in sample.get("conversation_branches", []):
        for message in branch.get("messages", []):
            conversation_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
            
            # Check if this assistant message needs classification
            if should_classify_message(message, ["assistant"]):
                context = extract_message_context(message, conversation_messages)
                
                tasks.append({
                    "sample": sample,
                    "message": message,
                    "question": context,
                    "answer": message["content"]
                })
    
    return tasks


def save_progress(output_path: Path, completed_chunks: List[int], model: str, total_chunks: int):
    """Save progress to allow resuming interrupted processing."""
    progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
    progress_data = {
        "model": model,
        "total_chunks": total_chunks,
        "completed_chunks": completed_chunks,
        "timestamp": datetime.now().isoformat()
    }
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)


def load_progress(output_path: Path, model: str) -> List[int]:
    """Load progress from previous run."""
    progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
    
    if not progress_file.exists():
        return []
    
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        
        if progress_data.get("model") == model:
            return progress_data.get("completed_chunks", [])
    except Exception as e:
        print(f"Warning: Failed to load progress file: {e}")
    
    return []


def cleanup_progress(output_path: Path, model: str):
    """Remove progress file after successful completion."""
    progress_file = output_path / f".progress_{model.replace('/', '_').replace('-', '_')}.json"
    if progress_file.exists():
        progress_file.unlink()


def apply_classification_results(tasks: List[Dict[str, Any]], results: List[Any], model: str) -> List[Dict[str, Any]]:
    """
    Apply classification results to the original samples.
    
    Args:
        tasks: Original classification tasks
        results: Classification results from LLM
        model: Model name used for classification
        
    Returns:
        List of updated samples
    """
    # Group tasks by sample
    sample_updates = {}
    
    for task, result in zip(tasks, results):
        sample_id = task["sample"]["conversation_id"]
        if sample_id not in sample_updates:
            sample_updates[sample_id] = {
                "sample": task["sample"],
                "updates": []
            }
        
        # Create classification metadata
        classification_metadata = {
            "classification": result.classification,
            "reasoning": result.reasoning,
            "success": result.success,
            "timestamp": result.timestamp,
            "model": model
        }
        
        if result.error:
            classification_metadata["error"] = result.error
        
        sample_updates[sample_id]["updates"].append({
            "message": task["message"],
            "metadata": classification_metadata
        })
    
    # Apply updates to samples
    updated_samples = []
    for sample_data in sample_updates.values():
        sample = sample_data["sample"].copy()
        
        # Apply metadata updates to messages
        for update in sample_data["updates"]:
            target_message = update["message"]
            classification_metadata = update["metadata"]
            
            # Find and update the message in conversation branches
            for branch in sample.get("conversation_branches", []):
                for i, message in enumerate(branch.get("messages", [])):
                    if (message["content"] == target_message["content"] and 
                        message["role"] == target_message["role"]):
                        
                        # Initialize metadata if not present
                        if "metadata" not in branch["messages"][i]:
                            branch["messages"][i]["metadata"] = {}
                        
                        # Add refusal classification metadata
                        branch["messages"][i]["metadata"]["refusal_classification"] = classification_metadata
                        break
        
        updated_samples.append(sample)
    
    return updated_samples


async def classify_dataset(input_path: Path, output_path: Path, model: str, 
                         concurrent: int = 50, chunk_size: int = 10000, resume: bool = False, restart: bool = False,
                         disable_adaptive: bool = False):
    """
    Classify refusal responses in a chat format dataset.
    
    Args:
        input_path: Path to input dataset
        output_path: Path to output dataset
        model: Model to use for classification
        concurrent: Starting number of concurrent requests
        chunk_size: Number of samples to process per chunk
        disable_adaptive: Disable adaptive concurrency (use fixed concurrency)
    """
    # Load dataset
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
    
    # Initialize classifier
    api_key = os.getenv("SWISSAI_API") or os.getenv("SWISSAI_API_KEY")
    if not api_key:
        raise ValueError("SWISSAI_API or SWISSAI_API_KEY environment variable is required")
    
    classifier = LLMClassifier(api_key, model, concurrent=concurrent, adaptive=not disable_adaptive)
    
    # Check compute nodes and warn if concurrency is low
    try:
        compute_nodes = await classifier.get_model_compute_nodes()
        recommended_min = compute_nodes * 50
        
        if concurrent < recommended_min:
            print(f"⚠️  WARNING: Found {compute_nodes} compute nodes for model '{model}'")
            print(f"   Current concurrency ({concurrent}) is below recommended minimum ({recommended_min})")
            print(f"   This may result in slower processing. Consider using --concurrent {recommended_min} or higher.")
            print()
        
    except Exception as e:
        # Don't fail the whole script if node detection fails
        pass
    
    if not disable_adaptive:
        print(f"🔄 Adaptive concurrency enabled (starting: {concurrent})")
    else:
        print(f"⚡ Fixed concurrency mode: {concurrent} requests")
    
    # Load prompt template
    prompt_template_path = Path(__file__).parent / "prompts" / "refusal.txt"
    prompt_template = classifier.load_prompt_template(prompt_template_path)
    
    valid_categories = ["refusal", "no_refusal"]
    
    # Calculate total chunks and handle progress
    total_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    
    if restart:
        # Clear any existing progress
        cleanup_progress(output_path, model)
        completed_chunks = []
        print("Restarting from beginning (clearing any previous progress)")
    elif resume:
        completed_chunks = load_progress(output_path, model)
        if completed_chunks:
            print(f"Resuming from previous run: {len(completed_chunks)}/{total_chunks} chunks completed")
        else:
            print("No previous progress found, starting from beginning")
    else:
        completed_chunks = []
    
    # Process dataset in chunks
    all_updated_samples = []
    total_classifications = 0
    successful_classifications = 0
    failed_classifications = 0
    total_sample_errors = 0
    
    print(f"Processing dataset in {total_chunks} chunks of {chunk_size:,} samples...")
    
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
                tasks = collect_classification_tasks(sample)
                chunk_tasks.extend(tasks)
            except Exception as e:
                sample_errors.append((chunk_start + i, str(e)))
                # Continue processing other samples instead of failing entire chunk
                continue
        
        if sample_errors:
            print(f"Warning: Failed to process {len(sample_errors)} samples in this chunk:")
            for sample_idx, error in sample_errors[:3]:  # Show first 3 errors
                print(f"  Sample {sample_idx}: {error}")
            if len(sample_errors) > 3:
                print(f"  ... and {len(sample_errors) - 3} more errors")
        
        if not chunk_tasks:
            print("No assistant messages found in this chunk")
            # Add samples without modifications
            all_updated_samples.extend(chunk)
            continue
        
        print(f"Found {len(chunk_tasks)} assistant messages to classify")
        
        # Classify the tasks
        print("Sending classification requests...")
        
        results = await classifier.classify_batch(
            chunk_tasks, 
            prompt_template, 
            valid_categories
        )
        
        # Count results
        chunk_successful = sum(1 for r in results if r.success)
        chunk_failed = sum(1 for r in results if not r.success)
        
        total_classifications += len(results)
        successful_classifications += chunk_successful
        failed_classifications += chunk_failed
        total_sample_errors += len(sample_errors)
        
        # Display chunk metrics
        chunk_metrics = classifier.metrics.get_metrics()
        print(f"Chunk results: {chunk_successful} successful, {chunk_failed} failed")
        print(f"Chunk metrics: {chunk_metrics['requests_per_minute']:.1f} req/min, "
              f"{chunk_metrics['success_rate']*100:.1f}% success rate")
        
        # Apply results to samples
        print("Applying classification results...")
        updated_samples = apply_classification_results(chunk_tasks, results, model)
        
        # Add any samples that weren't modified (no assistant messages)
        updated_sample_ids = {s["conversation_id"] for s in updated_samples}
        for sample in chunk:
            if sample["conversation_id"] not in updated_sample_ids:
                updated_samples.append(sample)
        
        all_updated_samples.extend(updated_samples)
        
        # Save progress after successful chunk
        completed_chunks.append(chunk_idx)
        save_progress(output_path, completed_chunks, model, total_chunks)
    
    # Create output dataset
    print(f"\nCreating output dataset with {len(all_updated_samples):,} samples...")
    output_dataset = Dataset.from_list(all_updated_samples)
    
    # Save dataset
    output_path.mkdir(parents=True, exist_ok=True)
    output_dataset.save_to_disk(str(output_path))
    
    # Load and update metadata
    metadata_file = input_path / "dataset_metadata.json"
    original_metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            original_metadata = json.load(f)
    
    # Create processing log entry
    processing_entry = {
        "operation": "refusal_classification",
        "script": "classify_refusal.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model": model,
        "api_provider": "swiss_ai",
        "samples_processed": len(dataset),
        "samples_with_errors": total_sample_errors,
        "messages_classified": total_classifications,
        "classification_success": successful_classifications,
        "classification_failed": failed_classifications,
        "total_tokens_used": classifier.total_tokens_used,
        "concurrent": concurrent,
        "adaptive_enabled": not disable_adaptive,
        "chunk_size": chunk_size
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
    
    # Cleanup progress file on successful completion
    cleanup_progress(output_path, model)
    
    print(f"\n=== REFUSAL CLASSIFICATION COMPLETED ===")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model}")
    print(f"Samples processed: {len(dataset):,}")
    if total_sample_errors > 0:
        print(f"Samples with errors: {total_sample_errors:,}")
    print(f"Messages classified: {total_classifications:,}")
    print(f"Success rate: {successful_classifications/total_classifications*100:.1f}%" if total_classifications > 0 else "Success rate: N/A")
    print(f"Total tokens used: {classifier.total_tokens_used:,}")
    
    # Final performance metrics
    final_metrics = classifier.metrics.get_metrics()
    print(f"\n=== FINAL PERFORMANCE METRICS ===")
    print(f"Average requests per minute: {final_metrics['requests_per_minute']:.1f}")
    print(f"Overall success rate: {final_metrics['success_rate']*100:.1f}%")
    print(f"Total failed requests: {final_metrics['failed_requests']}")
    if final_metrics['total_requests'] > 0:
        print(f"Total requests tracked: {final_metrics['total_requests']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Classify refusal responses in chat format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify refusals in a dataset
  python classify_refusal.py data/02-standardised/tulu-3-sft-mixture \\
    --output data/05-annotations/tulu-3-sft-mixture-refusal

  # Resume interrupted processing
  python classify_refusal.py data/02-standardised/smoltalk \\
    --output data/05-annotations/smoltalk-refusal \\
    --resume

  # Restart from beginning (clear progress)
  python classify_refusal.py data/02-standardised/smoltalk \\
    --output data/05-annotations/smoltalk-refusal \\
    --restart

  # Use different model and concurrency settings
  python classify_refusal.py data/02-standardised/smoltalk \\
    --output data/05-annotations/smoltalk-refusal \\
    --model "Qwen/Qwen3-32B" \\
    --concurrent 100 \\
    --chunk-size 5000
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
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model to use for classification (default: meta-llama/Llama-3.3-70B-Instruct)"
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
    
    args = parser.parse_args()
    
    print("=== Refusal Classification Configuration ===")
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
        sys.exit(1)
    
    output_path = Path(args.output)
    
    # Validate resume/restart flags
    if args.resume and args.restart:
        print("Error: Cannot use both --resume and --restart flags")
        sys.exit(1)
    
    # Check if output exists and handle accordingly
    if output_path.exists():
        progress_file = output_path / f".progress_{args.model.replace('/', '_').replace('-', '_')}.json"
        
        if progress_file.exists():
            if not args.resume and not args.restart:
                print(f"Previous processing detected for model '{args.model}'")
                print("Use --resume to continue or --restart to start over")
                sys.exit(1)
        elif not args.restart:
            print(f"Output directory already exists: {output_path}")
            print("Use --restart to overwrite existing output")
            sys.exit(1)
    
    # Run classification
    try:
        asyncio.run(classify_dataset(
            input_path, 
            output_path, 
            args.model,
            args.concurrent,
            args.chunk_size,
            args.resume,
            args.restart,
            args.disable_adaptive
        ))
    except KeyboardInterrupt:
        print("\nClassification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()