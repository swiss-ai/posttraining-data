#!/usr/bin/env python3
"""
Language annotation script for new chat format datasets (with parts structure).

This script adds FastText-based language detection to response parts in conversations,
storing the primary language, confidence score, and top-3 language predictions
in each response part's metadata. Works with the new chat format using parts structure.
"""

import sys
import json
import argparse
import urllib.request
import tempfile
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm

try:
    import fasttext
    import numpy as np
    
    # Simple targeted fix for numpy 2.0 compatibility
    # Replace the problematic line: np.array(probs, copy=False) -> np.asarray(probs)
    original_array = np.array
    def numpy_array_fix(*args, **kwargs):
        if 'copy' in kwargs and kwargs['copy'] is False:
            # Use np.asarray instead of np.array(copy=False)
            return np.asarray(args[0])
        return original_array(*args, **kwargs)
    
    np.array = numpy_array_fix
    
except ImportError:
    print("FastText not found. Install with: pip install fasttext-wheel")
    sys.exit(1)


def download_fasttext_model(model_path: Path) -> bool:
    """
    Download FastText language identification model if not exists.
    
    Args:
        model_path: Path where to store the model
        
    Returns:
        True if model ready, False if failed
    """
    if model_path.exists():
        print(f"Using existing FastText model: {model_path}")
        return True
    
    print("Downloading FastText language identification model...")
    try:
        # Create directory if it doesn't exist
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the model (176 languages supported)
        url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, model_path)
        
        print(f"Model downloaded successfully: {model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def classify_language_fasttext(model, text: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Classify language of text using FastText model.
    
    Args:
        model: Loaded FastText model
        text: Text to classify
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with classification results
    """
    try:
        # Preprocess text - remove newlines that cause issues
        text_clean = text.replace('\n', ' ').replace('\r', ' ').strip()
        if not text_clean:
            return {
                "primary_language": "unknown",
                "primary_confidence": 0.0,
                "top_languages": []
            }
        
        # Truncate very long texts to avoid memory issues
        if len(text_clean) > 10000:
            text_clean = text_clean[:10000]
        
        # Get predictions
        predictions = model.predict(text_clean, k=top_k)
        languages = predictions[0]
        scores = predictions[1]
        
        # Extract top languages
        top_languages = []
        lang_list = languages if isinstance(languages, (list, tuple)) else [languages]
        score_list = scores if hasattr(scores, '__len__') and not isinstance(scores, str) else [scores]
        
        for i in range(min(top_k, len(lang_list))):
            if i < len(lang_list):
                lang = lang_list[i]
                # Remove __label__ prefix
                lang_clean = str(lang).replace('__label__', '')
                
                # Get confidence score - handle numpy scalar conversion
                if i < len(score_list):
                    score_val = score_list[i]
                    try:
                        confidence = float(score_val.item()) if hasattr(score_val, 'item') else float(score_val)
                    except Exception:
                        # Fallback for problematic conversions
                        try:
                            confidence = float(str(score_val))
                        except:
                            confidence = 0.0
                else:
                    confidence = 0.0
                
                top_languages.append({
                    "language": lang_clean,
                    "confidence": confidence
                })
        
        # Primary language is the first one
        primary_lang = top_languages[0]["language"] if top_languages else "unknown"
        primary_conf = top_languages[0]["confidence"] if top_languages else 0.0
        
        return {
            "primary_language": primary_lang,
            "primary_confidence": primary_conf,
            "top_languages": top_languages
        }
        
    except Exception as e:
        # More detailed error logging for debugging
        error_msg = str(e)
        if "copy" in error_msg.lower():
            # Known numpy 2.0 compatibility issue
            return {
                "primary_language": "numpy_error",
                "primary_confidence": 0.0,
                "top_languages": [],
                "error": "numpy compatibility issue"
            }
        else:
            print(f"FastText classification error: {error_msg}")
            return {
                "primary_language": "error",
                "primary_confidence": 0.0,
                "top_languages": [],
                "error": error_msg
            }




def annotate_message_parts(message: Dict[str, Any], fasttext_model, top_k: int = 3) -> Dict[str, Any]:
    """
    Add language annotation to response parts in a message (new format with parts structure).
    
    CRITICAL: Never modifies input data in-place to prevent Arrow corruption.
    
    Args:
        message: Message dictionary with parts structure
        fasttext_model: Loaded FastText model
        top_k: Number of top language predictions
        
    Returns:
        Updated message with language annotations in response parts
    """
    import copy
    
    # Create deep copy to avoid in-place modification (prevents Arrow corruption)
    message_copy = copy.deepcopy(message)
    
    # Process parts if they exist
    if "parts" in message_copy and isinstance(message_copy["parts"], list):
        for i, part in enumerate(message_copy["parts"]):
            # Only annotate response-type parts that have non-empty content
            if (isinstance(part, dict) and 
                part.get("type") == "response"):
                
                content = part.get("content", "").strip()
                
                # Skip if content is empty after stripping whitespace
                if not content:
                    continue
                
                # Classify language
                language_result = classify_language_fasttext(fasttext_model, content, top_k)
                
                # Add to part metadata (safe because we're modifying the copy)
                if "metadata" not in message_copy["parts"][i]:
                    message_copy["parts"][i]["metadata"] = {}
                
                # Ensure consistent schema for top_languages
                top_languages = language_result["top_languages"]
                if not top_languages:
                    top_languages = [{"language": "unknown", "confidence": 0.0}]
                
                message_copy["parts"][i]["metadata"]["language_classification"] = {
                    "primary_language": language_result["primary_language"],
                    "primary_confidence": language_result["primary_confidence"],
                    "top_languages": top_languages,
                    "timestamp": datetime.now().isoformat(),
                    "model": "fasttext-lid.176",
                    "top_k": top_k
                }
    
    return message_copy


def annotate_sample(sample: Dict[str, Any], fasttext_model, top_k: int = 3) -> Dict[str, Any]:
    """
    Add language annotations to all messages in a sample (new format).
    
    Args:
        sample: Chat format sample (may have parts structure in messages)
        fasttext_model: Loaded FastText model
        top_k: Number of top language predictions
        
    Returns:
        Updated sample with language annotations where content exists
    """
    # Create a deep copy to avoid in-place modification (critical for Arrow compatibility)
    import copy
    sample_copy = copy.deepcopy(sample)
    
    # Annotate system prompt if it has content (direct content field)
    if "system_prompt" in sample_copy and sample_copy["system_prompt"]:
        content = sample_copy["system_prompt"].get("content", "").strip()
        if content:
            # Classify language
            language_result = classify_language_fasttext(fasttext_model, content, top_k)
            
            # Add to metadata
            if "metadata" not in sample_copy["system_prompt"]:
                sample_copy["system_prompt"]["metadata"] = {}
            
            # Ensure consistent schema for top_languages
            top_languages = language_result["top_languages"]
            if not top_languages:
                top_languages = [{"language": "unknown", "confidence": 0.0}]
            
            sample_copy["system_prompt"]["metadata"]["language_classification"] = {
                "primary_language": language_result["primary_language"],
                "primary_confidence": language_result["primary_confidence"],
                "top_languages": top_languages,
                "timestamp": datetime.now().isoformat(),
                "model": "fasttext-lid.176",
                "top_k": top_k
            }
    
    # Annotate initial prompt if it has content (direct content field)
    if "initial_prompt" in sample_copy and sample_copy["initial_prompt"]:
        content = sample_copy["initial_prompt"].get("content", "").strip()
        if content:
            # Classify language
            language_result = classify_language_fasttext(fasttext_model, content, top_k)
            
            # Add to metadata
            if "metadata" not in sample_copy["initial_prompt"]:
                sample_copy["initial_prompt"]["metadata"] = {}
            
            # Ensure consistent schema for top_languages
            top_languages = language_result["top_languages"]
            if not top_languages:
                top_languages = [{"language": "unknown", "confidence": 0.0}]
            
            sample_copy["initial_prompt"]["metadata"]["language_classification"] = {
                "primary_language": language_result["primary_language"],
                "primary_confidence": language_result["primary_confidence"],
                "top_languages": top_languages,
                "timestamp": datetime.now().isoformat(),
                "model": "fasttext-lid.176",
                "top_k": top_k
            }
    
    # Annotate conversation branches (messages may have parts structure)
    if "conversation_branches" in sample_copy:
        for branch in sample_copy["conversation_branches"]:
            if "messages" in branch:
                for i, message in enumerate(branch["messages"]):
                    # Check if message has parts (new format) or direct content
                    if "parts" in message:
                        # New format with parts - use parts annotation
                        branch["messages"][i] = annotate_message_parts(message, fasttext_model, top_k)
                    elif "content" in message:
                        # Some messages might have direct content - annotate if non-empty
                        content = message.get("content", "").strip()
                        if content:
                            # Classify language
                            language_result = classify_language_fasttext(fasttext_model, content, top_k)
                            
                            # Add to metadata
                            if "metadata" not in branch["messages"][i]:
                                branch["messages"][i]["metadata"] = {}
                            
                            # Ensure consistent schema for top_languages
                            top_languages = language_result["top_languages"]
                            if not top_languages:
                                top_languages = [{"language": "unknown", "confidence": 0.0}]
                            
                            branch["messages"][i]["metadata"]["language_classification"] = {
                                "primary_language": language_result["primary_language"],
                                "primary_confidence": language_result["primary_confidence"],
                                "top_languages": top_languages,
                                "timestamp": datetime.now().isoformat(),
                                "model": "fasttext-lid.176",
                                "top_k": top_k
                            }
    
    return sample_copy


def annotate_dataset_streaming(dataset, fasttext_model, top_k: int = 3, chunk_size: int = 10000):
    """
    Annotate dataset using streaming approach with chunking for memory efficiency.
    
    Args:
        dataset: HuggingFace Dataset
        fasttext_model: Loaded FastText model
        top_k: Number of top language predictions
        chunk_size: Number of samples to process in each chunk
        
    Returns:
        Annotated Dataset
    """
    print(f"Annotating dataset in chunks of {chunk_size:,} samples using streaming approach...")
    total_samples = len(dataset)
    total_messages = 0
    
    chunk_datasets = []
    
    # Process dataset in chunks
    with tqdm(total=total_samples, desc="Annotating", unit="samples") as pbar:
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_samples = []
            chunk_messages = 0
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                sample = dataset[idx]
                
                # Count messages before annotation
                message_count = 0
                if sample.get("system_prompt"):
                    message_count += 1
                if sample.get("initial_prompt"):
                    message_count += 1
                for branch in sample.get("conversation_branches", []):
                    message_count += len(branch.get("messages", []))
                
                # Annotate sample
                annotated_sample = annotate_sample(sample, fasttext_model, top_k)
                chunk_samples.append(annotated_sample)
                chunk_messages += message_count
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(messages=f"{total_messages + chunk_messages:,}")
            
            # Create chunk dataset
            if chunk_samples:
                chunk_dataset = Dataset.from_list(chunk_samples)
                chunk_datasets.append(chunk_dataset)
                total_messages += chunk_messages
                
                # Clear chunk samples from memory
                del chunk_samples
                gc.collect()
    
    print(f"Annotation complete: {total_samples:,} samples, {total_messages:,} messages")
    
    if not chunk_datasets:
        return Dataset.from_list([])
    
    # Combine using concatenate_datasets for memory efficiency
    print("Combining chunk datasets...")
    final_dataset = concatenate_datasets(chunk_datasets)
    
    # Clear chunk datasets from memory
    del chunk_datasets
    gc.collect()
    
    return final_dataset, total_messages


def annotate_dataset(dataset, fasttext_model, top_k: int = 3, chunk_size: int = 10000):
    """
    Annotate entire dataset using specified parameters with memory-efficient chunking.
    
    Args:
        dataset: HuggingFace Dataset
        fasttext_model: Loaded FastText model
        top_k: Number of top language predictions
        chunk_size: Number of samples to process in each chunk
        
    Returns:
        Tuple of (annotated_dataset, total_messages)
    """
    # For very large datasets (>100k samples), use streaming approach
    if len(dataset) > 100000:
        return annotate_dataset_streaming(dataset, fasttext_model, top_k, chunk_size)
    
    # For smaller datasets, process in chunks but keep in memory
    print(f"Annotating dataset in chunks of {chunk_size:,} samples...")
    annotated_samples = []
    total_messages = 0
    total_samples = len(dataset)
    
    # Process dataset in chunks
    with tqdm(total=total_samples, desc="Annotating", unit="samples") as pbar:
        for chunk_start in range(0, total_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_samples)
            chunk_samples = []
            chunk_messages = 0
            
            # Process chunk
            for idx in range(chunk_start, chunk_end):
                sample = dataset[idx]
                
                # Count messages before annotation
                message_count = 0
                if sample.get("system_prompt"):
                    message_count += 1
                if sample.get("initial_prompt"):
                    message_count += 1
                for branch in sample.get("conversation_branches", []):
                    message_count += len(branch.get("messages", []))
                
                # Annotate sample
                annotated_sample = annotate_sample(sample, fasttext_model, top_k)
                chunk_samples.append(annotated_sample)
                chunk_messages += message_count
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(messages=f"{total_messages + chunk_messages:,}")
            
            # Add chunk to results
            annotated_samples.extend(chunk_samples)
            total_messages += chunk_messages
            
            # Force garbage collection after each chunk
            gc.collect()
    
    print(f"Annotation complete: {len(annotated_samples):,} samples, {total_messages:,} messages")
    
    # Create new dataset from annotated samples
    return Dataset.from_list(annotated_samples), total_messages


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


def save_dataset_and_metadata(dataset, output_path: Path, input_path: Path, total_messages: int):
    """Save annotated dataset with metadata."""
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    print(f"Saving annotated dataset to {output_path}...")
    dataset.save_to_disk(str(output_path))
    
    # Load or create metadata
    metadata = load_existing_metadata(input_path) or {}
    
    # Add processing log entry
    processing_entry = {
        "operation": "language_annotation",
        "script": "language_annotate_newformat.py",
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_messages_annotated": total_messages,
        "model": "fasttext-lid.176",
        "top_k": 3,
        "annotation_success": True,
        "format": "new_chat_format_with_parts"
    }
    
    if "processing_log" not in metadata:
        metadata["processing_log"] = []
    metadata["processing_log"].append(processing_entry)
    
    # Save updated metadata
    metadata_file = output_path / "dataset_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add FastText language detection to new chat format datasets (with parts structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate new format dataset
  venv/bin/python 05-annotations/language_annotate_newformat.py data/02-standardised-newformat/dataset data/05-annotations/dataset-lang
  
  # With custom chunk size for large datasets
  venv/bin/python 05-annotations/language_annotate_newformat.py data/04-decontaminated-newformat/dataset data/05-annotations/dataset-lang --chunk-size 5000
  
  # Test with OLMO prompts-only dataset
  venv/bin/python 05-annotations/language_annotate_newformat.py /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated_newformat/olmo-2-0325-32b-preference-mix-promptsOnly data/05-annotations/
        """
    )
    
    parser.add_argument(
        "input_path",
        help="Path to input new chat format dataset directory (with parts structure)"
    )
    parser.add_argument(
        "output_path", 
        help="Path for output annotated dataset"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top language predictions to store (default: 3)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of samples to process in each chunk for memory efficiency (default: 10000)"
    )
    
    return parser.parse_args()


def main():
    """Main annotation function."""
    args = parse_arguments()
    
    # Validate input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output_path)
    
    # Set up FastText model
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / "lid.176.bin"
    
    if not download_fasttext_model(model_path):
        print("Failed to download FastText model")
        sys.exit(1)
    
    # Load FastText model
    print("Loading FastText model...")
    try:
        fasttext_model = fasttext.load_model(str(model_path))
        print("FastText model loaded successfully")
        
        # Test the model with a simple example
        test_result = classify_language_fasttext(fasttext_model, "Hello world, this is a test.")
        print(f"Model test: {test_result}")
        if test_result["primary_language"] == "error":
            print("FastText model test failed - there may be compatibility issues")
            
    except Exception as e:
        print(f"Failed to load FastText model: {e}")
        sys.exit(1)
    
    # Load dataset  
    print(f"Loading dataset from: {input_path}")
    try:
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
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    
    # Annotate dataset using chunked processing
    print(f"Annotating dataset with language detection (top-{args.top_k} predictions)...")
    print(f"Using chunk size: {args.chunk_size:,} samples")
    
    annotated_dataset, total_messages = annotate_dataset(
        dataset, fasttext_model, args.top_k, args.chunk_size
    )
    
    # Save annotated dataset
    save_dataset_and_metadata(annotated_dataset, output_path, input_path, total_messages)
    
    print(f"\nLanguage annotation completed!")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Samples: {len(annotated_dataset):,}")
    print(f"Messages annotated: {total_messages:,}")
    print(f"Model: fasttext-lid.176 (176 languages)")
    print(f"Top-K predictions: {args.top_k}")


if __name__ == "__main__":
    main()