"""
Stage 3: Compute reference log-probabilities for chosen and rejected.

Uses the exact same logprob computation as TRL's DPOTrainer:
  - Prefix-subtraction tokenization (tokenize prompt, tokenize full chat, slice)
  - Binary completion mask from prefix length (0=prompt, 1=completion)
  - Right-padding
  - selective_log_softmax (row-by-row for bfloat16 stability)
  - Sum of per-token logprobs over completion tokens

Features:
  - Partition/subpartition parallelism (across nodes and GPUs within a node)
  - Checkpoint/resume with save_interval
  - GPU assignment via CUDA_VISIBLE_DEVICES per subpartition

Input:  HuggingFace dataset with 'chosen' and 'rejected' columns
        (lists of {role, content} message dicts, where the last message
        is the assistant completion and all preceding messages are the prompt).
Output: Checkpoint datasets with added columns:
        - ref_chosen_logprob, chosen_length
        - ref_rejected_logprob, rejected_length

Usage (single GPU):
    python compute_logprobs.py \
        --dataset-path /path/to/dataset \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model \
        --partition-start 0 --partition-end 1000 \
        --num-gpus-per-node 1

Usage (multi-GPU via SLURM, one task per subpartition):
    Automatically reads SLURM_PROCID from environment.
    python compute_logprobs.py \
        --dataset-path /path/to/dataset \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model \
        --partition-start 0 --partition-end 8192 \
        --num-gpus-per-node 4 \
        --tensor-parallel-size 1
"""

import argparse
import copy
import logging
import math
import os
from pathlib import Path

import datasets
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_logprobs_single(model, input_ids, loss_mask, temperature=1.0):
    """Compute summed logprobs for a single sequence. Returns (logprob_sum, num_completion_tokens)."""
    input_ids = input_ids.unsqueeze(0).to(model.device)
    attention_mask = (input_ids != 0).long()  # not used for single unpadded seq, but required by model
    attention_mask = torch.ones_like(input_ids)
    loss_mask = loss_mask.unsqueeze(0).to(model.device)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :] / temperature
        labels = input_ids[:, 1:].clone()
        loss_mask = loss_mask[:, 1:].bool()

        labels[~loss_mask] = 0
        per_token_logps = torch.gather(
            logits.float().log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        per_token_logps[~loss_mask] = 0

    return per_token_logps.sum().item(), loss_mask.sum().item()


def compute_logprobs_for_row(model, row, tokenizer, max_seq_len, batch_size, temperature=1.0, debug=False):
    """
    Compute logprobs for chosen and rejected completions in a single row.

    Tokenizes exactly like the trainer (TRL's PreferenceTrainer):
      1. apply_chat_template(tokenize=False) to get text for prompt, chosen, rejected
      2. Tokenize prompt, chosen, rejected text separately
      3. Concatenate prompt_ids + completion_ids
      4. Loss mask = [0]*prompt_len + [1]*completion_len
      5. Forward pass, log_softmax + gather, sum over completion tokens
    """
    # Step 1: Extract prompt messages (same as TRL's maybe_extract_prompt)
    prompt_messages = row["chosen"][:-1]

    # Step 2: Apply chat template using TEXT-SLICING (same as TRL v0.25.1's apply_chat_template)
    # TRL templates the full conversation, then slices by string length to get completion text.
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=False,
    )
    prompt_chosen_text = tokenizer.apply_chat_template(
        row["chosen"], tokenize=False,
    )
    prompt_rejected_text = tokenizer.apply_chat_template(
        row["rejected"], tokenize=False,
    )
    chosen_text = prompt_chosen_text[len(prompt_text):]
    rejected_text = prompt_rejected_text[len(prompt_text):]

    # Step 3: Tokenize separately (same as trainer's tokenize_row)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]

    # Step 4: Concatenate (same as trainer's concatenated_forward)
    chosen_full_ids = prompt_ids + chosen_ids
    rejected_full_ids = prompt_ids + rejected_ids

    # Check max sequence length
    max_len = max(len(chosen_full_ids), len(rejected_full_ids))
    if max_len > max_seq_len:
        raise ValueError(
            f"Sequence length {max_len} exceeds max_seq_len {max_seq_len}."
        )

    # Build loss masks: 0 for prompt, 1 for completion
    prompt_len = len(prompt_ids)
    chosen_loss_mask = [0] * prompt_len + [1] * len(chosen_ids)
    rejected_loss_mask = [0] * prompt_len + [1] * len(rejected_ids)

    assert len(chosen_ids) > 0, "Chosen completion is empty"
    assert len(rejected_ids) > 0, "Rejected completion is empty"

    if debug:
        logger.info("=" * 60)
        logger.info("[SCRIPT DEBUG] Row debug info (trainer-matching tokenization)")
        logger.info(f"  prompt tokens: {prompt_len}")
        logger.info(f"  chosen completion tokens: {len(chosen_ids)}")
        logger.info(f"  rejected completion tokens: {len(rejected_ids)}")
        logger.info(f"  chosen total tokens: {len(chosen_full_ids)}")
        logger.info(f"  rejected total tokens: {len(rejected_full_ids)}")
        logger.info(f"  chosen first 20 token IDs: {chosen_full_ids[:20]}")
        logger.info(f"  chosen last 20 token IDs: {chosen_full_ids[-20:]}")
        logger.info(f"  chosen completion first 20 IDs: {chosen_ids[:20]}")
        logger.info(f"  chosen completion last 20 IDs: {chosen_ids[-20:]}")
        logger.info(f"  chosen completion first 20 decoded: {tokenizer.decode(chosen_ids[:20])!r}")

    # Step 5: Forward pass for each sequence separately (avoids padding issues)
    chosen_input_ids = torch.tensor(chosen_full_ids)
    rejected_input_ids = torch.tensor(rejected_full_ids)
    chosen_mask = torch.tensor(chosen_loss_mask)
    rejected_mask = torch.tensor(rejected_loss_mask)

    chosen_logp, chosen_len = _compute_logprobs_single(
        model, chosen_input_ids, chosen_mask, temperature
    )
    rejected_logp, rejected_len = _compute_logprobs_single(
        model, rejected_input_ids, rejected_mask, temperature
    )

    if debug:
        logger.info(f"  [chosen] sum logprob: {chosen_logp:.4f}, len: {chosen_len}")
        logger.info(f"  [rejected] sum logprob: {rejected_logp:.4f}, len: {rejected_len}")
        logger.info("=" * 60)

    new_row = copy.deepcopy(row)
    new_row["ref_chosen_logprob"] = chosen_logp
    new_row["chosen_length"] = chosen_len
    new_row["ref_rejected_logprob"] = rejected_logp
    new_row["rejected_length"] = rejected_len

    return new_row


def compute_logprobs_batch(model, batch, tokenizer, max_seq_len, batch_size, temperature=1.0, debug=False):
    """Process a slice of the dataset row by row, return a Dataset with logprob columns."""
    rows_result = []
    for i, row in enumerate(tqdm(batch, desc="Processing batch")):
        new_row = compute_logprobs_for_row(
            model, row, tokenizer, max_seq_len, batch_size, temperature,
            debug=(debug and i == 0),
        )
        rows_result.append(new_row)
    return datasets.Dataset.from_list(rows_result)


def compute_subpartition_start_end_indices(
    partition_start_idx, partition_end_idx, subpartition_number, num_subpartitions
):
    """Divide a partition range into subpartitions (one per GPU group)."""
    subpartition_size = math.ceil(
        (partition_end_idx - partition_start_idx) / num_subpartitions
    )
    start_idx = partition_start_idx + subpartition_number * subpartition_size
    end_idx = partition_start_idx + (subpartition_number + 1) * subpartition_size
    end_idx = min(end_idx, partition_end_idx)
    return start_idx, end_idx


def main(args):
    # GPU assignment for this subpartition
    tp_size = args.tensor_parallel_size
    cuda_devices = [args.subpartition_number * tp_size + i for i in range(tp_size)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_devices))
    logger.info(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Import after setting CUDA_VISIBLE_DEVICES to ensure correct GPU binding
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Must be right-padding for completion mask subtraction

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    ).to("cuda")
    model.eval()

    # Compute subpartition range within the partition
    num_subpartitions = args.num_gpus_per_node // tp_size
    subpartition_start_idx, subpartition_end_idx = (
        compute_subpartition_start_end_indices(
            args.partition_start,
            args.partition_end,
            args.subpartition_number,
            num_subpartitions,
        )
    )

    if subpartition_start_idx >= subpartition_end_idx:
        logger.info("Subpartition is empty. Exiting.")
        return

    logger.info(
        f"Subpartition {args.subpartition_number}: "
        f"processing rows [{subpartition_start_idx}, {subpartition_end_idx})"
    )

    full_dataset = datasets.load_from_disk(args.dataset_path)
    if args.split:
        full_dataset = full_dataset[args.split]
    subpartition_data = full_dataset.select(
        range(subpartition_start_idx, subpartition_end_idx)
    )

    # Output directory for this subpartition's checkpoints
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: find the latest checkpoint
    already_processed_samples = max(
        (
            int(item.name.split("-")[-1])
            for item in output_dir.iterdir()
            if item.is_dir() and item.name.startswith("checkpoint-")
        ),
        default=0,
    )
    if already_processed_samples == len(subpartition_data):
        logger.info("All samples already processed. Exiting.")
        return

    local_start_idx = already_processed_samples
    if local_start_idx > 0:
        logger.info(
            f"Resuming from checkpoint-{local_start_idx}. "
            f"Processing from sample {local_start_idx}."
        )

    pbar = tqdm(total=len(subpartition_data), desc="Computing logprobs")
    pbar.update(local_start_idx)

    while local_start_idx < len(subpartition_data):
        current_slice = (
            local_start_idx,
            min(local_start_idx + args.save_interval, len(subpartition_data)),
        )
        current_slice_data = subpartition_data.select(range(*current_slice))
        local_end_idx = local_start_idx + len(current_slice_data)

        current_slice_data = compute_logprobs_batch(
            model, current_slice_data, tokenizer, args.max_seq_len, args.batch_size,
            args.temperature, debug=(args.debug and local_start_idx == already_processed_samples),
        )

        save_path = output_dir / f"checkpoint-{local_end_idx}"
        current_slice_data.save_to_disk(str(save_path))
        logger.info(f"Saved checkpoint-{local_end_idx}")

        pbar.update(len(current_slice_data))
        local_start_idx = local_end_idx

    logger.info("Logprobs computed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute reference logprobs for preference dataset"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path to input HF dataset (output of stage 2)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to save checkpointed output (per subpartition)",
    )
    parser.add_argument(
        "--model-name-or-path", type=str, required=True,
        help="Reference model path",
    )
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument(
        "--batch-size", type=int, default=2,
        help="Inner batch size for completions within a row (2 = chosen + rejected)",
    )
    parser.add_argument("--partition-start", type=int, required=True)
    parser.add_argument("--partition-end", type=int, required=True)
    parser.add_argument(
        "--subpartition-number", type=int,
        default=int(os.environ.get("SLURM_PROCID", "0")),
        help="Subpartition index within the node (defaults to SLURM_PROCID)",
    )
    parser.add_argument("--num-gpus-per-node", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--save-interval", type=int, default=2048,
        help="Save a checkpoint every N rows",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true", help="Log debug info for the first row")
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    main(args)
