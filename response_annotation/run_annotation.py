import os
import sys
import time
import argparse
import subprocess
import urllib.request

def main():
    parser = argparse.ArgumentParser(description="Orchestrate SGLang server and Generation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the column in the dataset that contains the prompts")
    parser.add_argument("--remove-last-message", action="store_true", help="Whether to remove the last message from the conversation history for the prompt")
    parser.add_argument("--base-output-dir", type=str, default="./output")
    parser.add_argument("--job-time", type=str, default="12:00:00")
    parser.add_argument("--logs-dir", type=str, default="./logs/annotation")
    parser.add_argument("--slurm-nodes", type=int, default=1)
    parser.add_argument("--slurm-exclude", type=str, default="", help="Comma-separated list of nodes to exclude")
    parser.add_argument("--workers", type=int, default=1, help="Number of sglang workers")
    parser.add_argument("--nodes-per-worker", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--disable-ocf", action="store_true", help="Disable OCF optimization")
    parser.add_argument("--framework", type=str, default="sglang", help="Serving framework (e.g., sglang, vllm)")
    
    args = parser.parse_args()
    scratch = os.environ.get("SCRATCH", "/tmp")
    os.makedirs(args.logs_dir, exist_ok=True)
    
    submit_cmd = [
        "python", f"{scratch}/model-launch/legacy/serving/submit_job.py",
        "--slurm-nodes", str(args.slurm_nodes),
        "--slurm-exclude", args.slurm_exclude,
        "--slurm-time", args.job_time,
        "--serving-framework", args.framework,
        "--worker-port", "8080",
        "--slurm-environment", f"{scratch}/model-launch/legacy/serving/envs/{args.framework}.toml",
    ]
    
    if args.workers > 1:
        submit_cmd.extend([
            "--workers", str(args.workers),
            "--nodes-per-worker", str(args.nodes_per_worker),
            "--use-router"
        ])

    if args.disable_ocf:
        submit_cmd.append("--disable-ocf")
    
    if args.framework == "sglang":
        fw_args = f"--model-path {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --dp-size {args.dp_size} --tp-size {args.tp_size} --trust-remote-code"
    elif args.framework == "vllm":
        fw_args = f"--model {args.model} --host 0.0.0.0 --port 8080 --served-model-name {args.model} --data-parallel-size {args.dp_size} --tensor-parallel-size {args.tp_size} --trust-remote-code"
        if "mistral" in args.model.lower():
            fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
    else:
        raise ValueError(f"Invalid framework: {args.framework}")
    
    submit_cmd.extend(["--framework-args", fw_args])

    print(f"🚀 Submitting: {' '.join(submit_cmd)}")
    result = subprocess.run(submit_cmd, cwd=args.logs_dir, capture_output=True, text=True, check=True)
    
    combined_output = result.stdout + "\n" + result.stderr
    job_id = None
    
    for line in combined_output.splitlines():
        if "Job submitted successfully with ID:" in line:
            job_id = line.split()[-1].strip()
            break
            
    if not job_id:
        print("❌ Failed to parse Job ID from output.")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        sys.exit(1)

    print(f"✅ Found Job ID: {job_id}")
        
    log_file = f"{args.logs_dir}/logs/{job_id}/log.out"
    base_url = None
    target_prefix = "Router URL: " if args.workers > 1 else "All worker URLs: "

    print(f"⏳ Waiting for URL in log file: {log_file}")
    wait_attempts = 0
    while not base_url:
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if target_prefix in content:
                    for line in content.splitlines():
                        if line.startswith(target_prefix):
                            raw = line.split(target_prefix, 1)[1].strip()
                            base_url = f"{raw}/v1" if args.workers > 1 else f"{raw.rsplit(':', 1)[0]}:8080/v1"
                            print(f"✅ Found Base URL: {base_url}")
                            break
        else:
            if wait_attempts % 6 == 0: # Print every 30 seconds
                print(f"⚠️ Still waiting... log file {log_file} does not exist yet.")
                
        wait_attempts += 1
        time.sleep(5)

    health_url = base_url.replace("/v1", "/health")
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
            if health_attempts % 3 == 0: # Print every 30 seconds
                print(f"⚠️ Health check failed (Attempt {health_attempts}): {e}")
        
        health_attempts += 1
        time.sleep(10)

    cmd = [
        "python", "response_annotation/annotate.py", 
        "--dataset-path", args.dataset, 
        "--output-dir", args.base_output_dir, 
        "--model", args.model, 
        "--base-url", base_url,
        "--prompt-column-name", args.prompt_column_name,
    ]
    if args.remove_last_message:
        cmd.append("--remove-last-message")

    print(f"🚀 Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print(f"🚀 Cancelling server job: {job_id}")
    subprocess.run(["scancel", job_id])

if __name__ == "__main__":
    main()