import json
import asyncio
import argparse
from typing import Any
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict, load_from_disk


def extract_original_data(sample: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract original prompt and original response from metadata."""
    metadata = sample.get("original_metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None, None

    orig_msgs_str = metadata.get("original_messages")
    if not orig_msgs_str:
        return None, None

    try:
        orig_msgs = json.loads(orig_msgs_str)
        
        # Extract original prompt
        initial_prompt = orig_msgs.get("initial_prompt", {})
        original_prompt = None
        if isinstance(initial_prompt, str):
            original_prompt = initial_prompt
        else:
            original_prompt = initial_prompt.get("content")

        # Extract original response (reference)
        original_response = None
        branches = orig_msgs.get("conversation_branches", [])
        if isinstance(branches, str):
            try:
                branches = json.loads(branches)
            except json.JSONDecodeError:
                branches = []
        
        if branches:
            first_branch = branches[0]
            messages = first_branch.get("messages", [])
            for msg in messages:
                if msg.get("role") == "assistant":
                    parts = msg.get("parts", [])
                    for part in parts:
                        if part.get("type") == "response":
                            original_response = part.get("content")
                            break
                    if original_response:
                        break
        
        return original_prompt, original_response
    except (json.JSONDecodeError, KeyError, IndexError):
        return None, None


def build_filter_prompt(
    tokenizer: AutoTokenizer,
    initial_prompt: str,
    response: str,
    instruction_descriptions: list[str],
    original_prompt: str | None = None,
    original_response: str | None = None,
) -> str:
    """Build a prompt to ask the model to evaluate the sample quality."""
    instructions_list = "\n".join(f"- {desc}" for desc in instruction_descriptions)

    # Use original prompt if available to make it clear what was added
    prompt_to_show = original_prompt if original_prompt else initial_prompt

    user_message = f"""You are a quality evaluator for instruction-following data samples. Your task is to determine if a data sample is high-quality and "useful" for training a model to follow constraints.

## User Request
{prompt_to_show}

## Added Constraints
{instructions_list}

## Assistant Response
{response}"""

    if original_response:
        user_message += f"""

## Original Reference Response (For comparison)
{original_response}"""

    user_message += f"""

## Evaluation Criteria
1. **Instruction Adherence**: Does the response strictly follow all the **Added Constraints**? (Already verified programmatically, but double-check for natural integration).
2. **Utility & Non-Redundancy**: Are the constraints "useful" in this context?
   - **Useful**: The constraints add specific requirements that were NOT already inherently required by the original request or already fully met by the original reference response.
   - **Redundant/Useless**: The original request already implied these constraints, or the reference response already satisfied them naturally without being told. We want to avoid samples where the model doesn't have to "effort" to follow the constraints because they were already met.
3. **Quality**: Is the response well-written, helpful, and coherent?

## Decision
- Output {{"keep": true}} if the sample is high quality, the instructions are followed, and at least some of the constraints were **meaningful** (not entirely redundant/pre-satisfied). Be reasonably lenient: if the constraints added some specific formatting or style that wasn't there before, keep it.
- Output {{"keep": false}} ONLY if:
    - The response violates the instructions.
    - The response is poor quality or incoherent.
    - **ALL** the added constraints were already perfectly satisfied in the original reference response (making the sample useless for learning to follow instructions).

Your response (JSON only):"""

    messages = [{"role": "user", "content": user_message}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return formatted_prompt


def extract_response_from_sample(sample: dict[str, Any]) -> str | None:
    """Extract the assistant response from a sample."""
    branches = sample.get("conversation_branches", [])
    if not branches:
        return None

    # Handle JSON-encoded branches
    if isinstance(branches, str):
        try:
            branches = json.loads(branches)
        except json.JSONDecodeError:
            return None

    first_branch = branches[0] if branches else {}
    messages = first_branch.get("messages", [])

    # For multi-turn, get the last assistant response
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            parts = msg.get("parts", [])
            for part in parts:
                if part.get("type") == "response":
                    return part.get("content", "")
    return None


def extract_instruction_descriptions(sample: dict[str, Any], original_prompt: str | None = None) -> list[str]:
    """Extract instruction descriptions. Tries to diff the prompt first, falls back to IDs."""
    initial_prompt = sample.get("initial_prompt", {})
    augmented_content = None
    if isinstance(initial_prompt, str):
        augmented_content = initial_prompt
    else:
        augmented_content = initial_prompt.get("content", "")

    # If we have the original prompt, we can try to find the added instructions by diffing
    if original_prompt and augmented_content and augmented_content.startswith(original_prompt):
        added_text = augmented_content[len(original_prompt):].strip()
        if added_text:
            # Try to split by common separators if multiple instructions
            # For now, just return as one or split by sentences/newlines if appropriate
            # In generate-ifbench.py, they are joined by spaces: " ".join(descriptions)
            # This is hard to split perfectly without the original descriptions.
            # But showing the whole added text is better than IDs.
            return [added_text]

    # Fallback to instruction IDs from ground truth
    ground_truth = sample.get("ground_truth", "{}")
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return []

    # For multi-turn, ground_truth is a list of dicts
    if isinstance(ground_truth, list):
        ids = []
        for gt in ground_truth:
            inst_ids = gt.get("instruction_id", [])
            if isinstance(inst_ids, list):
                ids.extend(inst_ids)
            else:
                ids.append(inst_ids)
        return ids

    instruction_ids = ground_truth.get("instruction_id", [])
    return instruction_ids if isinstance(instruction_ids, list) else [instruction_ids]


async def process_single_sample(
    session: aiohttp.ClientSession,
    sample_idx: int,
    prompt: str,
    sglang_url: str,
    max_new_tokens: int,
    temperature: float,
    semaphore: asyncio.Semaphore,
) -> tuple[int, bool]:
    """Process a single sample asynchronously."""
    async with semaphore:
        try:
            async with session.post(
                f"{sglang_url.rstrip('/')}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "json_schema": json.dumps({
                            "type": "object",
                            "properties": {
                                "keep": {"type": "boolean"}
                            },
                            "required": ["keep"]
                        }),
                    },
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if sample_idx == 0:  # Only print full error for first sample
                        print(f"Request failed for sample {sample_idx}: {response.status} - {error_text[:500]}")
                    return (sample_idx, False)

                result_data = await response.json()
                text = result_data.get("text", "").strip()

                try:
                    parsed = json.loads(text)
                    keep = parsed.get("keep", False)
                    return (sample_idx, bool(keep))
                except json.JSONDecodeError:
                    lower_text = text.lower()
                    return (sample_idx, "true" in lower_text)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if sample_idx < 5:  # Only print first few errors
                print(f"Request failed for sample {sample_idx}: {e}")
            return (sample_idx, False)


async def filter_samples_async(
    samples: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    sglang_url: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    concurrency: int = 64,
    timeout: int = 300,
) -> list[bool]:
    """Filter samples using sglang with concurrent requests."""
    # Prepare all prompts
    prompts_with_idx: list[tuple[int, str]] = []

    for i, sample in enumerate(samples):
        initial_prompt = sample.get("initial_prompt", {})
        if isinstance(initial_prompt, str):
            prompt_content = initial_prompt
        else:
            prompt_content = initial_prompt.get("content", "")

        response = extract_response_from_sample(sample)
        
        # Try to get original prompt and response for better filtering
        original_prompt, original_response = extract_original_data(sample)
        
        # Extract instructions - pass original_prompt to help with diffing
        instruction_descs = extract_instruction_descriptions(sample, original_prompt=original_prompt)

        if not prompt_content or not response:
            continue

        filter_prompt = build_filter_prompt(
            tokenizer, 
            prompt_content, 
            response, 
            instruction_descs,
            original_prompt=original_prompt,
            original_response=original_response
        )
        prompts_with_idx.append((i, filter_prompt))

    # Track which samples have valid prompts
    valid_sample_indices = {idx for idx, _ in prompts_with_idx}

    # Process samples concurrently
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout_config = aiohttp.ClientTimeout(total=timeout)

    results_map: dict[int, bool] = {}

    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        tasks = [
            asyncio.create_task(
                process_single_sample(
                    session=session,
                    sample_idx=sample_idx,
                    prompt=prompt,
                    sglang_url=sglang_url,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    semaphore=semaphore,
                )
            )
            for sample_idx, prompt in prompts_with_idx
        ]

        # Gather results with progress bar
        with tqdm_asyncio(total=len(tasks), desc="Filtering samples") as pbar:
            for coro in asyncio.as_completed(tasks):
                sample_idx, keep = await coro
                results_map[sample_idx] = keep
                pbar.update(1)

    # Reconstruct results in order
    all_results = []
    for i in range(len(samples)):
        if i in valid_sample_indices:
            all_results.append(results_map.get(i, False))
        else:
            all_results.append(False)

    return all_results


async def filter_dataset_async(
    input_path: str,
    output_path: str,
    sglang_url: str,
    model_path: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    concurrency: int = 64,
    timeout: int = 300,
    only_verified: bool = True,
) -> None:
    """Filter the dataset using a model for quality evaluation."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)

    if hasattr(dataset, "keys"):
        split_name = list(dataset.keys())[0]
        print(f"Using split: {split_name}")
        dataset = dataset[split_name]

    print(f"Loaded {len(dataset)} samples")

    # Convert to list for processing
    samples = [dict(sample) for sample in dataset]

    # Pre-filter: only keep samples that passed verification (if enabled)
    if only_verified:
        pre_filter_count = len(samples)
        samples = [s for s in samples if s.get("verification_passed", False)]
        print(
            f"Pre-filtered to {len(samples)} verified samples (removed {pre_filter_count - len(samples)})"
        )

    if not samples:
        print("No samples to filter after pre-filtering.")
        return

    print(f"Filtering {len(samples)} samples with concurrency={concurrency}...")

    # Filter using the model
    keep_flags = await filter_samples_async(
        samples=samples,
        tokenizer=tokenizer,
        sglang_url=sglang_url,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        concurrency=concurrency,
        timeout=timeout,
    )

    # Apply filter
    filtered_samples = [s for s, keep in zip(samples, keep_flags) if keep]

    print(f"\nFiltering complete:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Kept samples: {len(filtered_samples)}")
    print(f"  Removed samples: {len(samples) - len(filtered_samples)}")
    print(f"  Keep rate: {len(filtered_samples) / len(samples) * 100:.1f}%")

    # Save filtered dataset
    if filtered_samples:
        filtered_dataset = Dataset.from_list(filtered_samples)
        filtered_dataset_dict = DatasetDict({"train": filtered_dataset})
        filtered_dataset_dict.save_to_disk(output_path)
        print(f"\nSaved filtered dataset with {len(filtered_samples)} samples to {output_path}")
    else:
        print("\nNo samples passed filtering. Dataset not saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter IFBench augmented dataset using model-based quality evaluation"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input augmented dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for filtered dataset",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="SGLang server URL",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model (for tokenizer)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum new tokens for filter response (default: 64)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Number of concurrent batch requests (default: 64)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each batch request (default: 300)",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        help="Include samples that failed programmatic verification (default: only verified)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asyncio.run(
        filter_dataset_async(
            input_path=args.input,
            output_path=args.output,
            sglang_url=args.url,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            timeout=args.timeout,
            only_verified=not args.include_unverified,
        )
    )


if __name__ == "__main__":
    main()
