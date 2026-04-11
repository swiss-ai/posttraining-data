"""
Generate max-min preference pairs from the combined annotated dataset.

Reads the new dataset format where each row has an 'annotations' column
containing a JSON array of {model, response, detailed_annotations, final_score}.
Picks the best and worst scoring models per row to form preference pairs.

Token filtering: applies a chat template via a tokenizer and excludes any
prompt+completion that exceeds MAX_TOKENS. If ALL completions exceed the limit
for a given row, falls back to picking best/worst without the token filter.
"""

import json
import os

import numpy as np
from datasets import DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_PATH = "/iopsstor/scratch/cscs/smarian/datasets/apertus/aya_dataset/annotation-merged-formatted"
OUTPUT_PATH = "/iopsstor/scratch/cscs/smarian/datasets/apertus/aya_dataset/MaxMin"
MODEL_NAME_OR_PATH = "/iopsstor/scratch/cscs/smarian/cache/hf_home/hub/models--swiss-ai--Apertus-8B-Instruct-2509/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364"
MAX_TOKENS = 4096
EXCLUDED_MODELS = {}
NUM_PROC = min(os.cpu_count() or 4, 288)

PROMPT_COLUMN_NAME = "prompt"
REMOVE_LAST_MESSAGE = False

print(f"Using {NUM_PROC} processes.")
print(f"Excluding models: {EXCLUDED_MODELS or 'none'}")
print(f"Tokenizer: {MODEL_NAME_OR_PATH}")
print(f"Max tokens: {MAX_TOKENS}")

# ---------------------------------------------------------------------------
# Load tokenizer and dataset
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)

dataset = load_from_disk(DATASET_PATH)
if "train" in dataset:
    dataset = dataset["train"]

print(f"Loaded dataset: {len(dataset)} rows, columns: {dataset.column_names}")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def count_tokens(prompt_msgs, response_text):
    """Apply chat template to prompt + response and return token count."""
    messages = prompt_msgs + [{"role": "assistant", "content": str(response_text)}]
    return len(tokenizer.apply_chat_template(messages, tokenize=True))


def extract_maxmin(batch):
    out = {
        "prompt_id": [],
        "chosen": [],
        "rejected": [],
        "chosen_model": [],
        "rejected_model": [],
        "chosen_score": [],
        "rejected_score": [],
    }

    for prompt, prompt_id, annotations_json in zip(
        batch[PROMPT_COLUMN_NAME], batch["prompt_id"], batch["annotations"]
    ):
        annotations = json.loads(annotations_json)

        # Filter out excluded models and None/empty responses
        valid = [
            a
            for a in annotations
            if a["model"] not in EXCLUDED_MODELS
            and a.get("response") is not None
            and str(a["response"]).strip() != ""
        ]
        if len(valid) < 2:
            continue

        # Extract prompt messages (keep only role and content keys)
        if REMOVE_LAST_MESSAGE:
            raw_msgs = (
                prompt[:-1] if isinstance(prompt, list) else json.loads(prompt)[:-1]
            )
        else:
            raw_msgs = prompt
        prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]

        # Count tokens for each valid completion
        token_counts = [count_tokens(prompt_msgs, a["response"]) for a in valid]

        # Filter by token count
        within_limit = [
            (a, tc) for a, tc in zip(valid, token_counts) if tc <= MAX_TOKENS
        ]

        # If all exceed the limit, fall back to unfiltered
        if len(within_limit) < 2:
            candidates = valid
        else:
            candidates = [a for a, _ in within_limit]

        scores = np.array([a["final_score"] for a in candidates])
        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))

        # Skip ties (same model is best and worst)
        if best_idx == worst_idx:
            continue

        best = candidates[best_idx]
        worst = candidates[worst_idx]

        out["prompt_id"].append(prompt_id)
        out["chosen"].append(
            prompt_msgs + [{"role": "assistant", "content": best["response"]}]
        )
        out["rejected"].append(
            prompt_msgs + [{"role": "assistant", "content": worst["response"]}]
        )
        out["chosen_model"].append(best["model"])
        out["rejected_model"].append(worst["model"])
        out["chosen_score"].append(float(best["final_score"]))
        out["rejected_score"].append(float(worst["final_score"]))

    return out


output_features = Features(
    {
        "prompt_id": Value("string"),
        "chosen": [{"role": Value("string"), "content": Value("string")}],
        "rejected": [{"role": Value("string"), "content": Value("string")}],
        "chosen_model": Value("string"),
        "rejected_model": Value("string"),
        "chosen_score": Value("float64"),
        "rejected_score": Value("float64"),
    }
)

print("Extracting max-min preference pairs ...", flush=True)
processed = dataset.map(
    extract_maxmin,
    batched=True,
    batch_size=2000,
    num_proc=NUM_PROC,
    remove_columns=dataset.column_names,
    features=output_features,
    desc="Extracting maxmin pairs",
)

processed = processed.filter(lambda x: x["prompt_id"] is not None)

print(f"Created maxmin dataset with {len(processed)} examples.")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
DatasetDict({"train_split": processed}).save_to_disk(OUTPUT_PATH)
print(f"Saved to {OUTPUT_PATH}")
