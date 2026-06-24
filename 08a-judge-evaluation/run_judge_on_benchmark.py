import os
import time
import shutil
import argparse
import subprocess
import urllib.request

from src.utils import load_module

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run judge evaluation against a pre-submitted LLM server job."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--judge-cfg-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--job-time", type=str, default="01:00:00")
    parser.add_argument("--server-job-id", type=str, required=True,
                        help="Slurm job ID of the already-submitted LLM server job.")
    return parser.parse_args()


def wait_for_server_url(server_job_id: str, workers: int) -> str:
    """Poll submit_job log for the worker/router URL, then wait until /health returns 200."""
    log_file = f"./logs/{server_job_id}/log.out" # this is based on how model-launch/serving/submit_job.py is configured
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
                            f"{raw}/v1"
                            if workers > 1
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


def main():
    args = parse_args()
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving output to {args.output_dir}")
    judge_cfg = load_module(args.judge_cfg_path)

    server_url = wait_for_server_url(args.server_job_id, judge_cfg.workers)

    judge_cmd = [
        "python", "-m", "src.judge",
        "--input-dir", args.input_dir,
        "--output-dir", args.output_dir,
        "--judge-cfg-path", args.judge_cfg_path,
        "--server-url", server_url,
    ]

    print(f"🚀 Running: {' '.join(judge_cmd)}")
    subprocess.run(judge_cmd, check=True)

    # Cleanup: cancel server
    subprocess.run(["scancel", args.server_job_id])

if __name__ == "__main__":
    main()
