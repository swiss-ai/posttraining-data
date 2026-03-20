"""
Stage 3: Compute reference log-probabilities for chosen and rejected.

Adapted from swiss_alignment.data_alignment.compute_ref_logprobs_swissaiformat.

Features from swiss_alignment:
  - Batched logprob computation (chosen + rejected padded together)
  - Partition/subpartition parallelism (across nodes and GPUs within a node)
  - Checkpoint/resume with save_interval
  - GPU assignment via CUDA_VISIBLE_DEVICES per subpartition

Input:  HuggingFace dataset with 'chosen', 'rejected', and 'prompt_messages'
        columns (output of stage 2).
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


def compute_logprobs_for_row(model, row, tokenizer, max_seq_len, batch_size):
    """
    Compute logprobs for chosen and rejected completions in a single row.

    Adapted from swiss_alignment's compute_logprobs_for_row:
    replaces conversation_branches / linearise_sample_for_sft with the
    simpler (prompt_messages, chosen, rejected) format. Core logprob math
    is identical.
    """
    # Prompt is the conversation up to (excluding) the last assistant message.
    # chosen and rejected share the same prompt; last message is the completion.
    prompt_messages = row["chosen"][:-1]
    chosen_text = row["chosen"][-1]["content"]
    rejected_text = row["rejected"][-1]["content"]

    chats = [
        prompt_messages + [{"role": "assistant", "content": chosen_text}],
        prompt_messages + [{"role": "assistant", "content": rejected_text}],
    ]

    # Tokenize full chats (prompt + completion) with padding
    tokenized_chats = tokenizer.apply_chat_template(
        chats,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    )

    if tokenized_chats["input_ids"].shape[1] > max_seq_len:
        logger.warning(
            f"Sequence length {tokenized_chats['input_ids'].shape[1]} exceeds "
            f"max_seq_len {max_seq_len}, skipping row."
        )
        new_row = copy.deepcopy(row)
        new_row["ref_chosen_logprob"] = None
        new_row["chosen_length"] = None
        new_row["ref_rejected_logprob"] = None
        new_row["rejected_length"] = None
        return new_row

    # Tokenize prompts only (with generation prompt to get assistant header tokens)
    tokenized_prompts = tokenizer.apply_chat_template(
        [prompt_messages, prompt_messages],
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    )

    # Extra padding to ensure prompts and chats have the same sequence length
    max_len = tokenized_chats["input_ids"].shape[1]
    if tokenized_prompts["input_ids"].shape[1] < max_len:
        padding_length = max_len - tokenized_prompts["input_ids"].shape[1]
        tokenized_prompts["input_ids"] = torch.nn.functional.pad(
            tokenized_prompts["input_ids"],
            (0, padding_length),
            value=tokenizer.pad_token_id,
        )
        tokenized_prompts["attention_mask"] = torch.nn.functional.pad(
            tokenized_prompts["attention_mask"], (0, padding_length), value=0
        )

    chat_input_ids = tokenized_chats["input_ids"]
    chat_attention_mask = tokenized_chats["attention_mask"]
    prompt_attention_mask = tokenized_prompts["attention_mask"]
    completion_mask = chat_attention_mask - prompt_attention_mask

    all_logps = []
    all_lens = []

    num_iters = math.ceil(len(chats) / batch_size)
    for i in range(num_iters):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(chats))
        input_ids = chat_input_ids[start_idx:end_idx].to(model.device)
        attention_mask = chat_attention_mask[start_idx:end_idx].to(model.device)
        loss_mask = completion_mask[start_idx:end_idx].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:].clone()
            loss_mask = loss_mask[:, 1:].bool()

            labels[~loss_mask] = 0  # Dummy token
            per_token_logps = torch.gather(
                logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)
            per_token_logps[~loss_mask] = 0
            logps = per_token_logps.sum(-1).cpu()
            lens = loss_mask.sum(-1).int().cpu()
            all_logps.extend(logps.tolist())
            all_lens.extend(lens.tolist())

    new_row = copy.deepcopy(row)
    new_row["ref_chosen_logprob"] = all_logps[0]
    new_row["chosen_length"] = all_lens[0]
    new_row["ref_rejected_logprob"] = all_logps[1]
    new_row["rejected_length"] = all_lens[1]

    return new_row


def compute_logprobs_batch(model, batch, tokenizer, max_seq_len, batch_size):
    """Process a slice of the dataset row by row, return a Dataset with logprob columns."""
    rows_result = []
    for row in tqdm(batch, desc="Processing batch"):
        new_row = compute_logprobs_for_row(
            model, row, tokenizer, max_seq_len, batch_size
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
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
            model, current_slice_data, tokenizer, args.max_seq_len, args.batch_size
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
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()
    main(args)
