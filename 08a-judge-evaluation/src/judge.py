import os
import json
import asyncio
import argparse
import uvloop
from datasets import load_from_disk
from tqdm.asyncio import tqdm_asyncio

from src.utils import create_async_openai_client, load_module, stringify_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Judge reformatted benchmark rows via an OpenAI-compatible server (used by run_judge.py)"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset directory (HuggingFace disk format)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write judged dataset + score JSONL")
    parser.add_argument("--judge-cfg-path", type=str, required=True)
    parser.add_argument("--server-url", type=str, required=True)
    return parser.parse_args()


async def write(queue, filepath):
    with open(filepath, "a") as f_out:
        while True:
            item = await queue.get()
            if item is None:
                break
            print(json.dumps(item), file=f_out)
            f_out.flush()
            queue.task_done()
    
async def main(args):
    # Import judge args
    judge_cfg = load_module(args.judge_cfg_path)

    # Load input dataset, prepare output path and buffer score_distributions path
    dataset = load_from_disk(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    judge_responses_path = os.path.join(args.output_dir, "judge_responses.jsonl")

    # Create API client
    client, http_client = create_async_openai_client(
        args.server_url,
        judge_cfg.concurrent,
    )

    # Create queue and writer
    queue = asyncio.Queue()
    writer = asyncio.create_task(write(queue, judge_responses_path))

    # Init judging tasks
    semaphore = asyncio.Semaphore(max(1, judge_cfg.concurrent))
    async def judge_one_sample(sample_idx, sample):
        prompt = stringify_prompt(sample["prompt"])
        filled_user_prompt_for_judge = judge_cfg.user_prompt_for_judge.format(
            prompt=prompt,
            response=sample["response"],
        )
        messages = [
            {"role": "system", "content": judge_cfg.system_prompt_for_judge},
            {"role": "user", "content": filled_user_prompt_for_judge},
        ]
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=judge_cfg.model,
                    messages=messages,
                    max_tokens=judge_cfg.max_tokens,
                    temperature=judge_cfg.temperature,
                    logprobs=judge_cfg.logprobs,
                    top_logprobs=judge_cfg.top_logprobs,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                judge_response_text = response.choices[0].message.content
                score_distribution = judge_cfg.extract_score_distribution(
                    response, judge_cfg.scoring_range
                )
            except Exception as e:
                print(f"Error judging index {sample_idx}: {e}")
                judge_response_text = None
                score_distribution = {score: 0.0 for score in judge_cfg.scoring_range}
        await queue.put({
            "sample_idx": sample_idx,
            "score_distribution": score_distribution,
            "judge_response_text": judge_response_text,
        })
    tasks = [
        asyncio.create_task(judge_one_sample(sample_idx, sample))
        for sample_idx, sample in enumerate(dataset)
    ]

    # Wait for all tasks to be done
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        await f
    await queue.put(None)
    await writer
    await http_client.aclose()

    # Load judge responses back in, append to dataset, then export
    judge_response_texts = [{} for _ in dataset]
    score_distributions = [{} for _ in dataset]
    with open(judge_responses_path, "r") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                score_distributions[x["sample_idx"]] = x["score_distribution"]
                judge_response_texts[x["sample_idx"]] = x["judge_response_text"]
    dataset = dataset.add_column("score_distribution", score_distributions)
    dataset = dataset.add_column("judge_response_texts", judge_response_texts)
    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    uvloop.install()
    args = parse_args()
    asyncio.run(main(args))
