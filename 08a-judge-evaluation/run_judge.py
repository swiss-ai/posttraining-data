import os
import sys
import time
import argparse
import subprocess
import urllib.request

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate SGLang/vLLM server and judge.py evaluation"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--judge-args-path", type=str, required=True)
    parser.add_argument("--base-output-dir", type=str, required=True)
    parser.add_argument("--logs-dir", type=str, required=True)

    parser.add_argument("--job-time", type=str, default="01:00:00")
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--nodes-per-worker", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-ocf", action="store_true", help="Disable OCF optimization")
    parser.add_argument("--framework", type=str, default="sglang", help="Serving framework (e.g., sglang, vllm)")
    parser.add_argument("--concurrent", type=int, default=32, help="Max concurrent judge API calls (passed to judge.py)")
    return parser.parse_args()


def build_server_cmd(args) -> list[str]:
    scratch = os.environ.get("SCRATCH", "/tmp")
    input_name = os.path.basename(os.path.dirname(args.input_dir))
    judge_name = os.path.basename(args.judge_args_path).rstrip(".py")

    server_cmd = [
        "python", f"{scratch}/model-launch/serving/submit_job.py",
        "--slurm-job-name", f"{input_name}_{judge_name}",
        "--slurm-nodes", str(args.slurm_nodes),
        "--slurm-time", args.job_time,
        "--serving-framework", args.framework,
        "--worker-port", "8080",
        "--slurm-environment", f"{scratch}/model-launch/serving/envs/{args.framework}.toml",
    ]

    if args.workers > 1:
        server_cmd.extend([
            "--workers", str(args.workers),
            "--nodes-per-worker", str(args.nodes_per_worker),
            "--use-router"
        ])

    if args.disable_ocf:
        server_cmd.append("--disable-ocf")

    if args.framework == "sglang":
        fw_args = (
            f"--model-path {args.model} --host 0.0.0.0 --port 8080 "
            f"--served-model-name {args.model} --dp-size {args.dp_size} "
            f"--tp-size {args.tp_size} --trust-remote-code"
        )
    elif args.framework == "vllm":
        fw_args = (
            f"--model {args.model} --host 0.0.0.0 --port 8080 "
            f"--served-model-name {args.model} --data-parallel-size {args.dp_size} "
            f"--tensor-parallel-size {args.tp_size} --trust-remote-code"
        )
        if "mistral" in args.model.lower():
            fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
    else:
        raise ValueError(f"Invalid framework: {args.framework}")
    server_cmd.extend(["--framework-args", fw_args])

    return server_cmd


def get_server_job_id(server_cmd: list[str]) -> str:
    """Run submit_job.py and return the Slurm job ID printed in stdout/stderr."""
    print(f"🚀 Submitting: {' '.join(server_cmd)}")
    result = subprocess.run(
        server_cmd, capture_output=True, text=True, check=True
    )
    time.sleep(5)
    
    combined = result.stdout + "\n" + result.stderr
    for line in combined.splitlines():
        if "Job submitted successfully with ID:" in line:
            server_job_id = line.split()[-1].strip()
            print(f"✅ Found Job ID: {server_job_id}")
            return server_job_id
    print("❌ Failed to parse Job ID from output.")
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    sys.exit(1)


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
    os.makedirs(args.logs_dir, exist_ok=True)
    output_dir = os.path.join(args.base_output_dir, os.environ["SLURM_JOB_ID"])
    os.makedirs(output_dir, exist_ok=True)

    server_cmd = build_server_cmd(args)
    server_job_id = get_server_job_id(server_cmd)
    server_url = wait_for_server_url(server_job_id, args.workers)

    judge_cmd = [
        "python -m src.judge",
        "--input-dir", args.input_dir,
        "--output-dir", output_dir,
        "--judge-args-path", args.judge_args_path,
        "--server-url", server_url,
        "--concurrent", str(args.concurrent),
    ]

    print(f"🚀 Running: {' '.join(judge_cmd)}")
    subprocess.run(judge_cmd, check=True)

    # Cleanup: cancel server
    subprocess.run(["scancel", server_job_id])

if __name__ == "__main__":
    main()
