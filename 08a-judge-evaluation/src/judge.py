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
    parser.add_argument("--judge-args-path", type=str, required=True)
    parser.add_argument("--concurrent", type=int, default=32)
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
    judge_args = load_module(args.judge_args_path)

    # Load input dataset, prepare output path and buffer score_distributions path
    dataset = load_from_disk(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    score_distributions_path = os.path.join(args.output_dir, "score_distributions.jsonl")

    # Create API client
    client, http_client = create_async_openai_client(
        args.server_url,
        args.concurrent,
    )

    # Create queue and writer
    queue = asyncio.Queue()
    writer = asyncio.create_task(write(queue, score_distributions_path))

    # Init judging tasks
    semaphore = asyncio.Semaphore(max(1, args.concurrent))
    async def judge_one_sample(sample_idx, sample):
        prompt = stringify_prompt(sample["prompt"])
        filled_user_prompt_for_judge = judge_args.USER_PROMPT_FOR_JUDGE.format(
            prompt=prompt,
            response=sample["response"],
        )
        messages = [
            {"role": "system", "content": judge_args.SYSTEM_PROMPT_FOR_JUDGE},
            {"role": "user", "content": filled_user_prompt_for_judge},
        ]
        async with semaphore:
            try:
                res = await client.chat.completions.create(
                    model=judge_args.MODEL,
                    messages=messages,
                    max_tokens=judge_args.MAX_TOKENS,
                    temperature=judge_args.TEMPERATURE,
                    logprobs=judge_args.LOGPROBS,
                    top_logprobs=judge_args.TOP_LOGPROBS,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                score_distribution = judge_args.extract_score_distribution(
                    res, judge_args.SCORING_RANGE
                )
            except Exception as e:
                print(f"Error judging index {sample_idx}: {e}")
                score_distribution = {score: 0.0 for score in judge_args.SCORING_RANGE}
        await queue.put({
            "sample_idx": sample_idx,
            "score_distribution": score_distribution,
            "messages": messages,
            "judge_response": res.choices[0].message.content,
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

    # Load score_distributions back in, append to dataset, then export
    score_distributions = [{} for _ in dataset]
    with open(score_distributions_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                score_distributions[data["sample_idx"]] = data["score_distribution"]
    dataset = dataset.add_column("score_distribution", score_distributions)
    dataset.save_to_disk(args.output_dir)

if __name__ == "__main__":
    uvloop.install()
    args = parse_args()
    asyncio.run(main(args))
