"""
Generate max-min preference pairs from the combined annotated dataset.

Reads the combined dataset where each row has a 'prompt' column (chosen[:-1])
and a 'model_evaluations' column containing an array of
{model, response, detailed_annotations, final_score}.
Picks the best and worst scoring models per row to form preference pairs.

Token filtering: applies a chat template via a tokenizer and excludes any
prompt+completion that exceeds MAX_TOKENS. If ALL completions exceed the limit
for a given row, falls back to picking best/worst without the token filter.
"""

import argparse
import os

import numpy as np
from datasets import DatasetDict, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------- Trinity-Mini Phi-4-mini-instruct
parser = argparse.ArgumentParser()
parser.add_argument("--exclude_models", nargs="*", default=[], help="Model names to exclude")
parser.add_argument("--dataset_path", default="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/combined_annotated_new2")
parser.add_argument("--output_path", default="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/aMaxMin_4096")
parser.add_argument("--tokenizer", default="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364")
parser.add_argument("--max_tokens", type=int, default=4096)
args = parser.parse_args()

DATASET_PATH = args.dataset_path
OUTPUT_PATH = args.output_path
MODEL_NAME_OR_PATH = args.tokenizer
MAX_TOKENS = args.max_tokens
EXCLUDED_MODELS = set(args.exclude_models)
NUM_PROC = min(os.cpu_count() or 4, 288)

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

    for prompt_msgs_raw, prompt_id, evaluations in zip(
        batch["prompt"], batch["prompt_id"], batch["model_evaluations"]
    ):
        # Filter out excluded models and None/empty responses
        valid = [
            a for a in evaluations
            if a["model"] not in EXCLUDED_MODELS
            and a.get("response") is not None
            and str(a["response"]).strip() != ""
        ]
        if len(valid) < 2:
            continue

        # Keep only role and content keys from prompt
        prompt_msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs_raw]

        # Sort by score: descending for best-first, ascending for worst-first
        sorted_by_score = sorted(valid, key=lambda a: a["final_score"], reverse=True)

        # Find best within token limit (search from highest score down)
        best = None
        for a in sorted_by_score:
            if count_tokens(prompt_msgs, a["response"]) <= MAX_TOKENS:
                best = a
                break

        # Find worst within token limit (search from lowest score up)
        worst = None
        for a in reversed(sorted_by_score):
            if count_tokens(prompt_msgs, a["response"]) <= MAX_TOKENS:
                worst = a
                break

        # Fallback: if fewer than 2 passed token filter, use unfiltered extremes
        if best is None or worst is None or best is worst:
            best = sorted_by_score[0]
            worst = sorted_by_score[-1]

        # Skip if same model ended up as both best and worst
        if best["model"] == worst["model"]:
            continue

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


output_features = Features({
    "prompt_id": Value("string"),
    "chosen": [{"role": Value("string"), "content": Value("string")}],
    "rejected": [{"role": Value("string"), "content": Value("string")}],
    "chosen_model": Value("string"),
    "rejected_model": Value("string"),
    "chosen_score": Value("float64"),
    "rejected_score": Value("float64"),
})

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
