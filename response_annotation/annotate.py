import os
import json
import math
import asyncio
import argparse
import httpx
import uvloop
from openai import AsyncOpenAI
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

# Import your system prompts from the external file
from prompts import (
    PREFERENCE_ANNOTATION_SYSTEM_PROMPT,
    INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT,
    HONESTY_ANNOTATION_PROMPT,
    TRUTHFULNESS_ANNOTATION_PROMPT,
    HELPFULNESS_ANNOTATION_PROMPT,
    CHARTER_ANNOTATION_PROMPT,
)

# ==============================================================================
# PROMPT MAPPING
# ==============================================================================

ASPECT2ANNOTATION_PROMPT = {
    "instruction_following": INSTRUCTION_FOLLOWING_ANNOTATION_PROMPT,
    "honesty": HONESTY_ANNOTATION_PROMPT,
    "truthfulness": TRUTHFULNESS_ANNOTATION_PROMPT,
    "helpfulness": HELPFULNESS_ANNOTATION_PROMPT,
    # "charter": CHARTER_ANNOTATION_PROMPT,
}

# ==============================================================================
# PIPELINE LOGIC
# ==============================================================================

def format_prompt_input(prompt_data):
    """Parses single-turn and multi-turn conversations cleanly for the Judge."""
    if isinstance(prompt_data, str):
        return prompt_data.strip()
    
    if isinstance(prompt_data, list):
        if len(prompt_data) == 1:
            return prompt_data[0].get('content', '').strip()
            
        formatted_text = "### CONVERSATION HISTORY ###\n"
        for turn in prompt_data[:-1]:
            role = turn.get('role', '').upper()
            content = turn.get('content', '').strip()
            formatted_text += f"[{role}]: {content}\n\n"
            
        formatted_text += "### FINAL INSTRUCTION ###\n"
        formatted_text += prompt_data[-1].get('content', '').strip()
        return formatted_text

    return str(prompt_data)

def extract_probabilities(res, target_words=["1", "2", "3", "4", "5"]):
    """Extracts normalized logprobs from the OpenAI API response."""
    try:
        first_token_logprobs = res.choices[0].logprobs.content[0].top_logprobs
        token_logprobs = {lp.token: lp.logprob for lp in first_token_logprobs}
        
        target_logprobs = {w: token_logprobs.get(w, -float("inf")) for w in target_words}
        exp_values = [math.exp(lp) for lp in target_logprobs.values()]
        total = sum(exp_values)
        
        if total == 0:
            return {w: 0.0 for w in target_words}
            
        return {k: float(v) / total for k, v in zip(target_logprobs.keys(), exp_values)}
    except Exception as e:
        # Fallback if no probabilities are returned
        print(f"⚠️ Warning: Failed to extract probabilities, returning uniform distribution. Error: {e}")
        return {w: 0.0 for w in target_words}

async def writer_task(queue, filepath):
    """Listens to the queue and writes outputs to the JSONL file immediately."""
    with open(filepath, "a", encoding="utf-8") as f:
        while True:
            item = await queue.get()
            if item is None:  # Poison pill
                break
            f.write(json.dumps(item) + "\n")
            f.flush()
            queue.task_done()

async def get_aspect_annotation(client, model, messages, temperature, semaphore):
    """Fetches the judge's score for a single aspect."""
    async with semaphore:
        try:
            res = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1,
                temperature=temperature,
                logprobs=True,
                top_logprobs=20,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return extract_probabilities(res)
        except Exception as e:
            print(f"Error fetching annotation: {e}")
            return {w: 0.0 for w in ["1", "2", "3", "4", "5"]}

async def annotate_sample(idx, prompt_data, response_text, client, model, temperature, semaphore, queue):
    """Evaluates all 4 aspects of a single completion concurrently."""
    formatted_input = format_prompt_input(prompt_data)
    
    tasks = []
    aspects = list(ASPECT2ANNOTATION_PROMPT.keys())
    
    # Spawn a task for each aspect
    for aspect in aspects:
        user_prompt = ASPECT2ANNOTATION_PROMPT[aspect].format(
            prompt=formatted_input, 
            completion=response_text
        )
        messages = [
            {"role": "system", "content": PREFERENCE_ANNOTATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        tasks.append(get_aspect_annotation(client, model, messages, temperature, semaphore))
        
    # Wait for all 4 aspects to finish for this specific row
    results = await asyncio.gather(*tasks)
    
    # Bundle them into a single dictionary
    annotation_dict = {aspect: result for aspect, result in zip(aspects, results)}
    
    await queue.put({
        "index": idx,
        "annotation": annotation_dict
    })

async def main(args):
    custom_limits = httpx.Limits(
        max_connections=args.concurrent, 
        max_keepalive_connections=args.concurrent
    )
    custom_timeout = httpx.Timeout(7200.0) 
    http_client = httpx.AsyncClient(limits=custom_limits, timeout=custom_timeout)

    client = AsyncOpenAI(
        base_url=args.base_url, 
        api_key="EMPTY",
        http_client=http_client
    )

    dataset = load_from_disk(args.dataset_path)

    dataset_basename = os.path.basename(os.path.normpath(args.dataset_path))
    if not dataset_basename:
        dataset_basename = "processed_dataset"
        
    target_output_dir = os.path.join(args.output_dir, dataset_basename)
    os.makedirs(target_output_dir, exist_ok=True)
    
    output_jsonl = os.path.join(target_output_dir, "annotations.jsonl")

    processed_indices = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed_indices.add(data["index"])
        print(f"✅ Resuming: Found {len(processed_indices)} already annotated items in {target_output_dir}.")

    if len(processed_indices) >= len(dataset):
        print("Already finished annotating the entire dataset.")
        await http_client.aclose()
        return

    print(f"Loading {len(dataset)} samples for annotation...")
    
    # Identify items that still need processing
    valid_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    if not valid_indices:
        print("No valid completions left to annotate.")
        await http_client.aclose()
        return

    queue = asyncio.Queue()
    effective_concurrency = max(1, args.concurrent // 4) 
    semaphore = asyncio.Semaphore(effective_concurrency)
    
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    prompts = dataset[args.prompt_column_name]
    if isinstance(prompts[0], str):
        prompts = [[{"role": "user", "content": p}] for p in prompts]
    if args.remove_last_message:
        print("⚠️ Removing last message from each prompt as per --remove-last-message flag.")
        prompts = [p[:-1] if len(p) > 1 else p for p in prompts]
    print(f"🚀 Annotating {len(valid_indices)} responses...")
    
    tasks = [
        annotate_sample(
            idx=idx, 
            prompt_data=prompts[idx], 
            response_text=dataset[idx]["response"], 
            client=client, 
            model=args.model, 
            temperature=args.temperature, 
            semaphore=semaphore, 
            queue=queue
        ) 
        for idx in valid_indices
    ]
    
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

    await queue.put(None)
    await writer
    await http_client.aclose()

    print("✅ All annotations completed and written to JSONL.")

    print(f"💾 Reconstructing and saving final dataset to {target_output_dir}")
    
    final_annotations = [{}] * len(dataset)
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_annotations[data["index"]] = data["annotation"]

    dataset = dataset.add_column("annotation", final_annotations)
    dataset.save_to_disk(target_output_dir)

    with open(os.path.join(target_output_dir, "judge_model_used.txt"), "w") as f:
        f.write(args.model)

if __name__ == "__main__":
    print("🚀 Starting annotation process...")
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--prompt-column-name", type=str, default="chosen", help="Name of the column containing the prompt/messages")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history, e.g. if you take it from a 'chosen' column")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="The judge model (e.g., Llama-3.3-70B-Instruct)")
    parser.add_argument("--concurrent", type=int, default=1000, help="Total concurrent API connections")
    parser.add_argument("--base-url", type=str, default="https://serving.swissai.cscs.ch/")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    asyncio.run(main(args))