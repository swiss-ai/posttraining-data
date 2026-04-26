import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.request

from src.utils import load_module

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate SGLang/vLLM server and judge.py evaluation"
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--judge-cfg-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--job-time", type=str, default="01:00:00")
    return parser.parse_args()


def build_server_cmd(args, judge_cfg) -> list[str]:
    scratch = os.environ.get("SCRATCH", "/tmp")
    input_name = os.path.basename(os.path.dirname(args.input_dir))
    ml_python = os.path.join(scratch, "model-launch", ".venv", "bin", "python")

    server_cmd = [
        ml_python, f"{scratch}/model-launch/legacy/serving/submit_job.py",
        "--slurm-job-name", f"{input_name}_{judge_cfg.name}",
        "--slurm-nodes", str(judge_cfg.slurm_nodes),
        "--slurm-time", args.job_time,
        "--worker-port", "8080",
        "--serving-framework", judge_cfg.framework,
        "--slurm-environment", f"{scratch}/model-launch/legacy/serving/envs/{judge_cfg.framework}.toml",
    ]

    if judge_cfg.workers > 1:
        server_cmd.extend([
            "--workers", str(judge_cfg.workers),
            "--nodes-per-worker", str(judge_cfg.nodes_per_worker),
            "--use-router"
        ])

    if judge_cfg.disable_ocf:
        server_cmd.append("--disable-ocf")

    if judge_cfg.framework == "sglang":
        fw_args = (
            f"--model-path {judge_cfg.model} --host 0.0.0.0 --port 8080 "
            f"--served-model-name {judge_cfg.model} --dp-size {judge_cfg.dp_size} "
            f"--tp-size {judge_cfg.tp_size} --trust-remote-code"
        )
    elif judge_cfg.framework == "vllm":
        fw_args = (
            f"--model {judge_cfg.model} --host 0.0.0.0 --port 8080 "
            f"--served-model-name {judge_cfg.model} --data-parallel-size {judge_cfg.dp_size} "
            f"--tensor-parallel-size {judge_cfg.tp_size} --trust-remote-code"
        )
        if "mistral" in judge_cfg.model.lower():
            fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
    else:
        raise ValueError(f"Invalid framework: {judge_cfg.framework}")
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
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    judge_cfg = load_module(args.judge_cfg_path)

    server_cmd = build_server_cmd(args, judge_cfg)
    server_job_id = get_server_job_id(server_cmd)
    server_url = wait_for_server_url(server_job_id, judge_cfg.workers)

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
    subprocess.run(["scancel", server_job_id])

if __name__ == "__main__":
    main()
