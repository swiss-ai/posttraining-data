import os
import json
import time
import asyncio
import argparse
import subprocess
import urllib.request
import httpx
import uvloop
from openai import AsyncOpenAI
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

own_answer_system_prompt = "You are a knowledgeable and helpful assistant. Answer the following user prompt."


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


def wait_for_server_url(server_job_id, workers):
    log_file = f"./logs/{server_job_id}/log.out"
    target_prefix = "Router URL: " if workers > 1 else "All worker URLs: "

    print(f"⏳ Waiting for URL in log file: {log_file}")
    wait_attempts = 0
    server_url = None
    while not server_url:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
            if target_prefix in content:
                for line in content.splitlines():
                    if line.startswith(target_prefix):
                        raw = line.split(target_prefix, 1)[1].strip()
                        server_url = (
                            f"{raw}/v1" if workers > 1
                            else f"{raw.rsplit(':', 1)[0]}:8080/v1"
                        )
                        print(f"✅ Found Server URL: {server_url}")
                        break
        else:
            if wait_attempts % 6 == 0:
                print(f"⚠️ Still waiting... log file {log_file} does not exist yet.")

        wait_attempts += 1
        time.sleep(5)

    health_url = server_url.replace("/v1", "/health")
    print(f"🏥 Pinging health check endpoint: {health_url}")

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health_attempts = 0
    while True:
        try:
            with opener.open(urllib.request.Request(health_url), timeout=5) as resp:
                if resp.getcode() == 200:
                    print("✅ Server is healthy!")
                    break
        except Exception as e:
            if health_attempts % 3 == 0:
                print(f"⚠️ Health check failed (Attempt {health_attempts}): {e}")

        health_attempts += 1
        time.sleep(10)

    return server_url


async def writer_task(queue, filepath):
    with open(filepath, "a", encoding="utf-8") as f:
        while True:
            item = await queue.get()
            if item is None:
                break
            f.write(json.dumps(item) + "\n")
            f.flush()
            queue.task_done()


async def get_own_response(client, model, messages, max_tokens, semaphore):
    async with semaphore:
        try:
            res = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Error fetching response: {e}")
            return ""


async def generate_response_for_sample(idx, prompt_data, client, model, max_tokens, semaphore, queue):
    formatted_input = format_prompt_input(prompt_data)
    messages = [
        {"role": "system", "content": own_answer_system_prompt},
        {"role": "user", "content": formatted_input},
    ]
    response_text = await get_own_response(client, model, messages, max_tokens, semaphore)
    await queue.put({"index": idx, "messages": messages, "response": response_text})


async def main(args):
    custom_limits = httpx.Limits(
        max_connections=args.concurrent,
        max_keepalive_connections=args.concurrent,
    )
    http_client = httpx.AsyncClient(limits=custom_limits, timeout=httpx.Timeout(7200.0))
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", http_client=http_client)

    dataset = load_from_disk(args.dataset_path)
    if args.debug:
        dataset = dataset.select(range(min(1000, len(dataset))))
        print(f"🐛 Debug mode: dataset limited to {len(dataset)} samples.")

    dataset_basename = os.path.basename(os.path.normpath(args.dataset_path))
    if not dataset_basename:
        dataset_basename = "processed_dataset"

    target_output_dir = os.path.join(args.output_dir, dataset_basename)
    os.makedirs(target_output_dir, exist_ok=True)

    output_jsonl = os.path.join(target_output_dir, "judge_own_responses.jsonl")

    processed_indices = set()
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    processed_indices.add(data["index"])
        print(f"✅ Resuming: Found {len(processed_indices)} already generated items in {target_output_dir}.")

    if len(processed_indices) >= len(dataset):
        print("Already finished generating responses for the entire dataset.")
        await http_client.aclose()
        return

    print(f"Loading {len(dataset)} samples for response generation...")

    valid_indices = [i for i in range(len(dataset)) if i not in processed_indices]

    if not valid_indices:
        print("No valid prompts left to process.")
        await http_client.aclose()
        return

    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(args.concurrent)
    writer = asyncio.create_task(writer_task(queue, output_jsonl))

    prompts = dataset[args.prompt_column_name]
    if isinstance(prompts[0], str):
        prompts = [[{"role": "user", "content": p}] for p in prompts]

    print(f"🚀 Generating responses for {len(valid_indices)} prompts...")

    tasks = [
        generate_response_for_sample(
            idx=idx,
            prompt_data=prompts[idx],
            client=client,
            model=args.model,
            max_tokens=args.max_tokens,
            semaphore=semaphore,
            queue=queue,
        )
        for idx in valid_indices
    ]

    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f

    await queue.put(None)
    await writer
    await http_client.aclose()

    print("✅ All responses generated and written to JSONL.")
    print(f"💾 Reconstructing and saving final dataset to {target_output_dir}")

    final_responses = [""] * len(dataset)
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                final_responses[data["index"]] = data["response"]

    with open(os.path.join(target_output_dir, "judge_model_used.txt"), "w") as f:
        f.write(args.model)


if __name__ == "__main__":
    print("🚀 Starting preannotation (asking judge for its own response)...")
    uvloop.install()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--prompt-column-name", type=str, default="prompt")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--concurrent", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    # provide either --base-url directly or --server-job-id to poll for it
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--server-job-id", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.server_job_id:
        args.base_url = wait_for_server_url(args.server_job_id, args.workers)
    elif not args.base_url:
        raise ValueError("Either --base-url or --server-job-id must be provided.")

    asyncio.run(main(args))

    if args.server_job_id:
        print(f"🚀 Cancelling server job: {args.server_job_id}")
        subprocess.run(["scancel", args.server_job_id])
