"""
decontamination_common.py
─────────────────────────
N-gram matching, verification and benchmark n-gram caching used by decontamination-parallel.py, and the
metadata loading shared with merge-decontamination-reports.py.
"""

import difflib
import hashlib
import json
import os
import pickle
import tempfile
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Optional

from matching_units import UNITS_VERSION, to_units


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


def get_ngrams(tokens, n):
    """All n-grams of tokens: one per start position, the last one starts at len(tokens) - n."""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def process_tokens(x, ngram_length):
    # "tokens" are the units of matching_units.py; the name is kept since the cache uses it
    return {
        "ngram": get_ngrams(x, ngram_length),
        "tokens": x
    }


shared_eval_ngram_to_eval_idx = None
shared_eval_ngram_to_eval_idx_key_set = None
def init_get_eval_match_indices(ngram_to_eval_idx, ):
    global shared_eval_ngram_to_eval_idx, shared_eval_ngram_to_eval_idx_key_set
    shared_eval_ngram_to_eval_idx = ngram_to_eval_idx
    shared_eval_ngram_to_eval_idx_key_set = set(ngram_to_eval_idx.keys())


def get_eval_match_indices(train_sample):
    shared_ngrams = train_sample["ngram"] & shared_eval_ngram_to_eval_idx_key_set
    if len(shared_ngrams) == 0:
        return tuple([])
    matches = []
    for ngram in shared_ngrams:
        matches.extend(shared_eval_ngram_to_eval_idx.get(ngram))
    return tuple(set(matches))


shared_eval_ngrams_dict = None
def init_contamination_worker(eval_ngrams_dict):
    """Initialize worker process with shared eval_ngrams dictionary."""
    global shared_eval_ngrams_dict
    shared_eval_ngrams_dict = eval_ngrams_dict


def verify_contamination_worker_optimized(args_tuple):
    """
    Optimized worker function for parallel contamination verification using shared memory.

    Args:
        args_tuple: (train_idx, eval_indices, train_tokens, threshold, train_conversation_id)

    Returns:
        tuple: (train_conversation_id, eval_idx, match_ratio, train_tokens_full, eval_tokens_full) if contaminated, None otherwise
    """
    train_idx, eval_indices, train_tokens, threshold, train_conversation_id = args_tuple

    for eval_idx in eval_indices:
        eval_tokens = shared_eval_ngrams_dict[eval_idx]["tokens"]
        matcher = difflib.SequenceMatcher(None, train_tokens, eval_tokens, autojunk=False)
        matching_blocks = matcher.get_matching_blocks()
        match_length = sum([x.size if x.size >= 5 else 0 for x in matching_blocks])
        del matcher, matching_blocks

        match_ratio = match_length / len(eval_tokens)
        if match_ratio >= threshold:
            return (train_conversation_id, eval_idx, match_ratio, train_tokens, eval_tokens)

    return None


def get_benchmark_cache_key(benchmark_name, tokenizer_name, ngram_length):
    """Generate cache key for benchmark n-grams."""
    key_string = f"{benchmark_name}_{tokenizer_name}_{ngram_length}_{UNITS_VERSION}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_benchmark_cache_path(cache_dir, cache_key):
    """Get cache file path for benchmark n-grams."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"benchmark_ngrams_{cache_key}.pkl")


def load_cached_benchmark_ngrams(cache_path):
    """Load cached benchmark n-grams if they exist."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, EOFError, IOError) as e:
            print(f"Warning: Failed to load cache from {cache_path}: {e}. Recomputing.")
    return None


def save_benchmark_ngrams_to_cache(ngrams, lookup_table, cache_path):
    """Save benchmark n-grams to cache."""
    try:
        cache_data = {
            'ngrams': ngrams,
            'lookup_table': lookup_table
        }
        cache_dir = os.path.dirname(cache_path) or "."
        with tempfile.NamedTemporaryFile(mode='wb', dir=cache_dir, delete=False) as f:
            pickle.dump(cache_data, f)
            temp_path = f.name
        os.replace(temp_path, cache_path)
        print(f"  Cached n-grams to {cache_path}")
    except (pickle.PickleError, IOError) as e:
        print(f"Warning: Failed to save cache to {cache_path}: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


def compute_benchmark_ngrams(eval_dataset, tokenizer, ngram_length, num_proc):
    """Compute n-grams for a benchmark dataset."""
    print(f"  Tokenizing {len(eval_dataset)} evaluation prompts...")
    eval_data_tokens = to_units(tokenizer, [str(p) for p in eval_dataset["prompt"][:]])
    print(f"  Generating n-grams...")
    with Pool(num_proc) as p:
        eval_ngrams = p.map(partial(process_tokens, ngram_length=ngram_length), eval_data_tokens)
    del eval_data_tokens

    print(f"  Building n-gram lookup table...")
    eval_ngram_to_eval_idx = defaultdict(list)
    append = eval_ngram_to_eval_idx.__getitem__
    for idx, s in enumerate(eval_ngrams):
        for element in s["ngram"]:
            append(element).append(idx)
    print(f"  Created lookup table with {len(eval_ngram_to_eval_idx)} unique n-grams")

    return eval_ngrams, eval_ngram_to_eval_idx
