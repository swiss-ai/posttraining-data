#!/usr/bin/env python3
"""
Filter a HuggingFace dataset by language using fasttext language identification.

Classifies the initial prompt and all model responses, keeping only rows
where the detected language matches the specified language.
"""

import argparse
from datasets import load_from_disk
import fasttext
from huggingface_hub import hf_hub_download
import numpy as np

# Global model cache for multiprocessing (loaded once per worker)
_model_cache = {}


def get_language_model():
    """Get or load the fasttext language identification model (cached per process)."""
    if "model" not in _model_cache:
        model_path = hf_hub_download(
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin"
        )
        model = fasttext.load_model(model_path)
        
        # Monkey-patch predict to fix NumPy 2.0 compatibility issue
        # See: https://numpy.org/devdocs/numpy_2_0_migration_guide.html
        def patched_predict(text, k=1, threshold=0.0, on_unicode_error='strict'):
            predictions = model.f.predict(text, k, threshold, on_unicode_error)
            return predictions
        
        model.predict = patched_predict
        _model_cache["model"] = model
    
    return _model_cache["model"]


def detect_language(text: str) -> str:
    """Detect the language of a text and return the language label."""
    if not text or not text.strip():
        return None
    # Clean text for fasttext (remove newlines)
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    model = get_language_model()
    prediction = model.predict(clean_text)
    # Returns format like '__label__eng_Latn'
    return prediction[0][1].replace("__label__", "")


def extract_texts_to_classify(row: dict) -> list[str]:
    """Extract all texts that need language classification from a row."""
    texts = []
    
    # Extract initial prompt content
    if row.get("initial_prompt") and row["initial_prompt"].get("content"):
        texts.append(row["initial_prompt"]["content"])
    
    # Extract all response parts from conversation branches
    for branch in row.get("conversation_branches", []):
        for message in branch.get("messages", []):
            # Only classify assistant responses
            if message.get("role") == "assistant":
                for part in message.get("parts", []):
                    if part.get("type") == "response" and part.get("content"):
                        texts.append(part["content"])
    
    return texts


def filter_by_language(row: dict, target_lang: str) -> bool:
    """Check if all texts in a row match the target language."""
    texts = extract_texts_to_classify(row)
    
    if not texts:
        # No texts to classify, skip this row
        return False
    
    for text in texts:
        detected_lang = detect_language(text)
        if detected_lang is None:
            # Empty text, skip
            continue
        if detected_lang != target_lang:
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Filter a HuggingFace dataset by language using fasttext."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the HuggingFace dataset (local path)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language code to keep (e.g., 'eng_Latn' for English, 'fra_Latn' for French)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for filtered dataset (default: <dataset>-<lang>)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=16,
        help="Number of processes for parallel filtering (default: 16)"
    )
    args = parser.parse_args()

    # Set default output path
    output_path = args.output or f"{args.dataset.rstrip('/')}-{args.lang}"

    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)

    # Pre-load model in main process to download it once
    print("Loading fasttext language identification model...")
    get_language_model()

    print(f"Filtering for language: {args.lang}")
    print(f"Original dataset size: {len(dataset)}")

    # Filter the dataset - model is loaded lazily in each worker
    filtered_dataset = dataset.filter(
        lambda row: filter_by_language(row, args.lang),
        desc=f"Filtering for {args.lang}",
        num_proc=args.num_proc
    )

    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Kept {len(filtered_dataset) / len(dataset) * 100:.2f}% of rows")

    print(f"Saving filtered dataset to {output_path}...")
    filtered_dataset.save_to_disk(output_path)
    print("Done!")


if __name__ == "__main__":
    main()
