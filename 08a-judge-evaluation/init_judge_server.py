"""
Submit the LLM server job from the login node.
Prints the server job ID to stdout; all other output goes to stderr.
Requires only stdlib — safe to run with any Python 3.6+ environment.
"""
import os
import sys
import argparse
import subprocess
import importlib.util
from pathlib import Path


def load_module(module_path: str):
    path = Path(module_path).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_server_cmd(args, judge_cfg):
    scratch = os.environ.get("SCRATCH", "/tmp")
    input_name = os.path.basename(os.path.dirname(args.input_dir))
    ml_python = os.path.join(scratch, "model-launch", ".venv", "bin", "python")

    server_cmd = [
        ml_python, "{}/model-launch/legacy/serving/submit_job.py".format(scratch),
        "--slurm-job-name", "{}_{}".format(input_name, judge_cfg.name),
        "--slurm-nodes", str(judge_cfg.slurm_nodes),
        "--slurm-time", args.job_time,
        "--worker-port", "8080",
        "--serving-framework", judge_cfg.framework,
        "--slurm-environment", "{}/model-launch/legacy/serving/envs/{}.toml".format(scratch, judge_cfg.framework),
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
            "--model-path {} --host 0.0.0.0 --port 8080 "
            "--served-model-name {} --dp-size {} "
            "--tp-size {} --trust-remote-code"
        ).format(judge_cfg.model, judge_cfg.model, judge_cfg.dp_size, judge_cfg.tp_size)
    elif judge_cfg.framework == "vllm":
        fw_args = (
            "--model {} --host 0.0.0.0 --port 8080 "
            "--served-model-name {} --data-parallel-size {} "
            "--tensor-parallel-size {} --trust-remote-code"
        ).format(judge_cfg.model, judge_cfg.model, judge_cfg.dp_size, judge_cfg.tp_size)
        if "mistral" in judge_cfg.model.lower():
            fw_args += " --tokenizer_mode mistral --load_format mistral --config_format mistral"
    else:
        raise ValueError("Invalid framework: {}".format(judge_cfg.framework))

    server_cmd.extend(["--framework-args", fw_args])
    return server_cmd


def submit_server_and_get_job_id(server_cmd):
    print("Submitting: {}".format(" ".join(server_cmd)), file=sys.stderr)
    try:
        result = subprocess.run(server_cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("submit_job.py failed.", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        raise

    combined = result.stdout + "\n" + result.stderr
    print(combined, file=sys.stderr)

    for line in combined.splitlines():
        if "Job submitted successfully with ID:" in line:
            return line.split()[-1].strip()

    print("Failed to parse job ID from output.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--judge-cfg-path", type=str, required=True)
    parser.add_argument("--job-time", type=str, default="01:00:00")
    args = parser.parse_args()

    judge_cfg = load_module(args.judge_cfg_path)
    server_cmd = build_server_cmd(args, judge_cfg)
    server_job_id = submit_server_and_get_job_id(server_cmd)
    print(server_job_id)


if __name__ == "__main__":
    main()
