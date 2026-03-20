"""
Stage 2: Generate reference completions using a vLLM inference server.

Launches a vLLM serving job via submit_job.py, auto-discovers the server URL
from logs, waits for readiness, then generates N completions per prompt using
the OpenAI-compatible API with high-concurrency async requests.

Given a preference dataset, extracts the prompt (all messages except the
final assistant turn from 'chosen'), then generates N completions per prompt
from a specified reference model.

Input:  HuggingFace dataset with 'chosen' column (list of message dicts).
Output: Dataset with added 'reference_completions' column (list of N strings)
        and 'prompt' column (message list without final assistant turn).

Usage:
    python generate_completions.py \
        --dataset-path /path/to/filtered_dataset \
        --output-dir /path/to/output \
        --model-name-or-path /path/to/model \
        --served-model-name my-model \
        --n-completions 10 \
        --max-new-tokens 4096 \
        --concurrent 500
"""

import argparse
import asyncio
import getpass
import json
import os
import subprocess
import sys
import time
import urllib.request

import httpx
import uvloop
from datasets import load_from_disk, DatasetDict
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

SCRATCH = os.environ.get("SCRATCH", "/iopsstor/scratch/cscs/dmelikidze")
SUBMIT_JOB_PATH = f"{SCRATCH}/model-launch/legacy/serving/submit_job.py"
VLLM_ENV_PATH = f"{SCRATCH}/model-launch/legacy/serving/envs/vllm.toml"
WORKER_PORT = 8080


def extract_prompt(chosen_messages):
    """Extract the prompt by removing the final assistant message from 'chosen'."""
    if chosen_messages[-1]["role"] == "assistant":
        return chosen_messages[:-1]
    return chosen_messages


def sanitize_messages(messages):
    """Strip non-standard fields from messages to avoid vLLM validation errors."""
    ALLOWED_KEYS = {
        "system": {"role", "content", "name"},
        "user": {"role", "content", "name"},
        "assistant": {"role", "content", "name", "tool_calls", "refusal"},
        "tool": {"role", "content", "tool_call_id"},
    }
    DEFAULT_ALLOWED = {"role", "content", "name"}

    sanitized = []
    for msg in messages:
        role = msg.get("role", "user")
        allowed = ALLOWED_KEYS.get(role, DEFAULT_ALLOWED)
        clean_msg = {k: v for k, v in msg.items() if k in allowed and v is not None}
        clean_msg["role"] = role
        if "content" not in clean_msg:
            clean_msg["content"] = msg.get("content", "")
        sanitized.append(clean_msg)
    return sanitized


def launch_inference_server(args, logs_dir):
    """Launch a vLLM inference server via submit_job.py and return the SLURM job ID."""
    served_model_name = args.served_model_name or f"swissai-ref-model-{getpass.getuser()}"

    framework_args = (
        f"--model {args.model_name_or_path} "
        f"--host 0.0.0.0 "
        f"--port {WORKER_PORT} "
        f"--served-model-name {served_model_name} "
        f"--tensor-parallel-size {args.tensor_parallel_size} "
        f"--data-parallel-size {args.data_parallel_size} "
        f"--dtype {args.dtype} "
        f"--max-model-len {args.max_model_len}"
    )

    cmd = [
        sys.executable, SUBMIT_JOB_PATH,
        "--slurm-nodes", str(args.slurm_nodes),
        "--serving-framework", "vllm",
        "--worker-port", str(WORKER_PORT),
        "--slurm-environment", VLLM_ENV_PATH,
        "--workers", str(args.workers),
        "--use-router",
        "--framework-args", framework_args,
    ]

    if args.nodes_per_worker is not None:
        cmd.extend(["--nodes-per-worker", str(args.nodes_per_worker)])

    if args.router_environment:
        cmd.extend(["--router-environment", args.router_environment])

    if args.slurm_time:
        cmd.extend(["--slurm-time", args.slurm_time])

    print(f"Launching inference server:")
    print(f"  {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=logs_dir, capture_output=True, text=True)
    combined_output = result.stdout + "\n" + result.stderr
    print(combined_output)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to launch inference server (exit code {result.returncode})")

    job_id = None
    for line in combined_output.splitlines():
        if "Job submitted successfully with ID:" in line:
            job_id = line.split()[-1].strip()
            break

    if not job_id:
        raise RuntimeError("Could not parse SLURM job ID from submit_job.py output")

    print(f"Server SLURM job ID: {job_id}")
    return job_id


def discover_base_url(job_id, logs_dir, workers, timeout=600):
    """Parse the server URL from the SLURM job log file."""
    log_file = os.path.join(logs_dir, "logs", job_id, "log.out")
    target_prefix = "Router URL: " if workers > 1 else "All worker URLs: "

    print(f"Waiting for server URL in {log_file}...")
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if target_prefix in content:
                    for line in content.splitlines():
                        if line.startswith(target_prefix):
                            raw = line.split(target_prefix)[1].strip()
                            if workers > 1:
                                base_url = f"{raw}/v1"
                            else:
                                base_url = f"{raw.rsplit(':', 1)[0]}:8080/v1"
                            print(f"Discovered base URL: {base_url}")
                            return base_url
        time.sleep(5)

    raise TimeoutError(f"Could not discover server URL within {timeout}s")


def wait_for_health(base_url, timeout=600):
    """Wait for the server health endpoint to return 200."""
    health_url = base_url.replace("/v1", "/health")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    print(f"Waiting for server health at {health_url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with opener.open(urllib.request.Request(health_url), timeout=5) as resp:
                if resp.getcode() == 200:
                    print("Server is healthy.")
                    return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError(f"Server health check failed within {timeout}s")


async def writer_task(queue, filepath):
    """Background task that writes results to JSONL as they arrive."""
    with open(filepath, "a", encoding="utf-8") as f:
        while True:
            item = await queue.get()
            if item is None:
                break
            f.write(json.dumps(item) + "\n")
            f.flush()
            queue.task_done()


async def get_completions(idx, messages, client, model, n_completions, max_tokens,
                          temperature, top_p, semaphore, queue, max_n_per_request=10):
    """Fetch N completions for a single prompt and push to write queue."""
    async with semaphore:
        try:
            messages = sanitize_messages(messages)
            completions = []
            remaining = n_completions
            while remaining > 0:
                batch_n = min(remaining, max_n_per_request)
                res = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    n=batch_n,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                completions.extend([choice.message.content for choice in res.choices])
                remaining -= batch_n
        except Exception as e:
            print(f"Error for index {idx}: {e}")
            completions = [""] * n_completions

        await queue.put({
            "index": idx,
            "completions": completions,
        })


async def wait_for_server(base_url, model_name, timeout=600, poll_interval=15):
    """Poll the server until the model is available."""
    print(f"Waiting for server at {base_url} (timeout={timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            async with httpx.AsyncClient(timeout=10.0) as hc:
                resp = await hc.get(f"{base_url}/models")
                if resp.status_code == 200:
                    models = resp.json()
                    available = [m["id"] for m in models.get("data", [])]
                    if model_name in available or available:
                        print(f"Server ready! Available models: {available}")
                        return
                    print(f"Server responded but no models available yet")
        except Exception:
            elapsed = int(time.time() - start)
            print(f"  Server not ready yet ({elapsed}s elapsed)...")
        await asyncio.sleep(poll_interval)

    raise TimeoutError(f"Server did not become ready within {timeout}s")


async def main(args):
    served_model_name = args.served_model_name or f"swissai-ref-model-{getpass.getuser()}"
    server_job_id = None
    logs_dir = os.path.join(args.output_dir, "server-logs")

    # Launch server and discover URL, or use provided base-url
    if args.base_url:
        base_url = args.base_url
    else:
        os.makedirs(logs_dir, exist_ok=True)
        server_job_id = launch_inference_server(args, logs_dir)
        base_url = discover_base_url(server_job_id, logs_dir, args.workers, timeout=args.server_timeout)
        wait_for_health(base_url, timeout=args.server_timeout)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    if hasattr(dataset, "keys"):
        split = "train_split" if "train_split" in dataset else list(dataset.keys())[0]
        print(f"DatasetDict detected, using '{split}' split")
        dataset = dataset[split]

    if args.partition_start is not None and args.partition_end is not None:
        total = len(dataset)
        start = args.partition_start
        end = min(args.partition_end, total)
        print(f"Processing partition [{start}, {end}) out of {total}")
        dataset = dataset.select(range(start, end))
    
    print(f"Dataset size: {len(dataset)}")

    # Extract prompts
    prompts_messages = [extract_prompt(row["chosen"]) for row in dataset]

    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "completions.jsonl")

    # Resume from checkpoint
    existing = {}
    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    idx = data["index"]
                    completions = data.get("completions", [])
                    # Only count as done if we got the expected number of non-empty completions
                    if len(completions) == args.n_completions and all(c.strip() for c in completions):
                        existing[idx] = completions
        print(f"Resuming from checkpoint: {len(existing)} completed items found.")

    processed_indices = set(existing)
    valid_indices = [i for i in range(len(prompts_messages)) if i not in processed_indices]

    if not valid_indices:
        print("All prompts already processed.")
    else:
        # Wait for model to be loaded
        await wait_for_server(base_url, served_model_name, timeout=args.server_timeout)

        # Setup async client with matching connection pool
        custom_limits = httpx.Limits(
            max_connections=args.concurrent,
            max_keepalive_connections=args.concurrent,
        )
        custom_timeout = httpx.Timeout(7200.0)
        http_client = httpx.AsyncClient(limits=custom_limits, timeout=custom_timeout)

        client = AsyncOpenAI(
            base_url=base_url,
            api_key="EMPTY",
            http_client=http_client,
        )

        # Setup queue, writer, and semaphore
        queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(args.concurrent)
        writer = asyncio.create_task(writer_task(queue, output_jsonl))

        print(f"Processing {len(valid_indices)} prompts with concurrency={args.concurrent}...")
        tasks = [
            get_completions(
                idx, prompts_messages[idx], client, served_model_name,
                args.n_completions, args.max_new_tokens,
                args.temperature, args.top_p,
                semaphore, queue,
            )
            for idx in valid_indices
        ]

        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            await f

        # Shutdown writer
        await queue.put(None)
        await writer
        await http_client.aclose()

        print("All requests completed and written to JSONL.")

    # Reconstruct final dataset from JSONL
    print(f"Reconstructing and saving final dataset to {args.output_dir}")
    all_completions = [[""] * args.n_completions] * len(dataset)

    # Load from existing dict first, then overlay with JSONL
    for idx, comps in existing.items():
        all_completions[idx] = comps

    if os.path.exists(output_jsonl):
        with open(output_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    all_completions[data["index"]] = data["completions"]

    prompts_json = [json.dumps(p) for p in prompts_messages]
    dataset = dataset.add_column("prompt_messages", prompts_json)
    dataset = dataset.add_column("reference_completions", all_completions)

    DatasetDict({"train_split": dataset}).save_to_disk(args.output_dir)

    # Cancel the server job if we launched it
    if server_job_id:
        print(f"Cancelling server job {server_job_id}...")
        subprocess.run(["scancel", server_job_id])

    print("Done.")

"""
    python 2-generate-ref-completions/generate_completions.py --dataset-path /iopsstor/scratch/cscs/dmelikidze/posttraining-data/preference_acquisition/datasets/MaxMin \
        --output-dir ./datasets/MaxMin-Filtered-Completions \
        --model-name-or-path /iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364 \
        --n-completions 30 \
        --concurrent 500 \
        --slurm-nodes 4 \
        --workers 4 \
        --data-parallel-size 4 \
        --router-environment $SCRATCH/model-launch/legacy/serving/envs/sglang.toml
"""
if __name__ == "__main__":
    uvloop.install()

    parser = argparse.ArgumentParser(description="Generate reference completions via vLLM server")
    # Dataset args
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to input HF dataset")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save output dataset")
    parser.add_argument("--model-name-or-path", type=str, required=True, help="Reference model path")
    parser.add_argument("--partition-start", type=int, default=None, help="Start index for dataset partition")
    parser.add_argument("--partition-end", type=int, default=None, help="End index for dataset partition")

    # Generation args
    parser.add_argument("--n-completions", type=int, default=30, help="Number of completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max tokens to generate per completion")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    # Server connection args
    parser.add_argument("--served-model-name", type=str, default=None, help="Model name served by vLLM (default: swissai-ref-model-$USER)")
    parser.add_argument("--base-url", type=str, default=None, help="OpenAI-compatible API base URL; if not provided, launches a server and auto-discovers")
    parser.add_argument("--concurrent", type=int, default=2000, help="Max concurrent requests")
    parser.add_argument("--server-timeout", type=int, default=600, help="Seconds to wait for server readiness")

    # Server launch args (used when --base-url is not provided)
    parser.add_argument("--slurm-nodes", type=int, default=1, help="Number of SLURM nodes for the server")
    parser.add_argument("--slurm-time", type=str, default="12:00:00", help="SLURM job time limit")
    parser.add_argument("--workers", type=int, default=1, help="Number of independent vLLM workers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size per worker")
    parser.add_argument("--data-parallel-size", type=int, default=4, help="Data parallel size per worker")
    parser.add_argument("--nodes-per-worker", type=int, default=None, help="Nodes per worker (default: slurm-nodes / workers)")
    parser.add_argument("--router-environment", type=str, default=None, help="SLURM environment for router (default: same as worker)")
    parser.add_argument("--max-model-len", type=int, default=8192*2, help="Max model context length for vLLM")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype")

    args = parser.parse_args()
    asyncio.run(main(args))
