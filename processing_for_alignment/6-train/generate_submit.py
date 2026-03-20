from datetime import datetime
from pathlib import Path

"""
Generate sbatch commands for preference training on the MaxMin dataset.

Adapted from alignment-apertus-swissaiformat-template/6-train/generate_submit.py.

This script must be run from the project root:
    cd /iopsstor/scratch/cscs/dmelikidze/dmelikidze/projects/posttraining/run
    python reproducibility-scripts/alignment-apertus-swissaiformat-template/6-train/generate_submit.py
    # or:
    python /iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/6-train/generate_submit.py
"""

stdout_prefix = "init"
stdout_root = (
    Path(__file__).parent.resolve().relative_to(Path.cwd())
    / f"{stdout_prefix}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
)

job_name = "maxmin-preference-sweep"

# Dataset config name (matches the YAML in configs/dataset/)
dataset_config = "maxmin-preference-Nref30"

# Direct path to the training dataset (overrides the YAML default)
train_dataset_path = "/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered-Ref-Completions-Annotated-Combined-Final"

models = ["apertus-8b-sft"]
model_hps = {
    "apertus-8b-sft": {
        "ids_paths": [
            (
                "10T-mixture-7-7fea1f8c44336360",
                "\${artifacts_dir}/shared/outputs/train_sft/final-run/Apertus8B-tokens10.2T-it2059810-newcooldown-apertus-sft-mixture-7-ln-v2-ademamix/checkpoints/7fea1f8c44336360/checkpoint-8925",
            ),
        ],
        "batch_size": 512,
        "num_nodes_per_job": 4,
        "per_device_train_batch_size": 2,
        "accelerate_config": "src/swiss_alignment/configs/accelerate/ds-zero2.yaml",
    },
}

# Reward / ref settings
ref_logprobs_from_dataset = True
train_num_ref_rewards = -1  # Use precomputed quantile rewards from the dataset

# Hyperparameter sweep (minimal grid for testing)
losses = ["dpo"]
normalize_beta_by_length = False
betas = {
    "qrpo": [0.01],
    "dpo": [0.1],
}
learning_rates = [1e-6]
optimizers = ["adamw_torch"]
max_grad_norm = 20

# Seed sweep: each seed produces a different config hash → separate checkpoint dir
seeds = [42]

# Run index: use this to re-run exact same hyperparameters (changes the config hash).
# e.g. run_indices = [0, 1] to run the same config twice.
run_indices = [0]

num_devices_per_node = 4


commands = []
total_nodes_needed = 0
for model in models:
    sftids = model_hps[model]["ids_paths"]
    batch_size = model_hps[model]["batch_size"]
    num_nodes_per_job = model_hps[model]["num_nodes_per_job"]
    per_device_train_batch_size = model_hps[model]["per_device_train_batch_size"]
    accelerate_config = model_hps[model]["accelerate_config"]
    accumulation_steps = batch_size // (
        num_nodes_per_job * num_devices_per_node * per_device_train_batch_size
    )
    for sftid, sftid_path in sftids:
        for seed in seeds:
            for run_index in run_indices:
                for loss in losses:
                    for optimizer in optimizers:
                        for lr in learning_rates:
                            for beta in betas[loss]:
                                jobid = (
                                    f"{dataset_config}-{model}-{sftid}"
                                    f"-{loss}-{optimizer}-lr{lr}-beta{beta}"
                                    f"-lengthnorm{normalize_beta_by_length}"
                                    f"-seed{seed}-run{run_index}"
                                )
                                run_name = f"{job_name}/{jobid}"
                                commands.append(
                                    (
                                        "sbatch "
                                        f"-p normal "
                                        f"-t 7:00:00 "
                                        f"-N {num_nodes_per_job} "
                                        f"-o {stdout_root}/out/{jobid}.out "
                                        f"-e {stdout_root}/out/{jobid}.err "
                                        "./cscs-shared-submit-scripts/recursive-unattended-accelerate.sh "
                                        f"-m swiss_alignment.train_preference "
                                        f"accelerate_config={accelerate_config} "
                                        f"dataset={dataset_config} "
                                        f"dataset_args.dataset_name='{train_dataset_path}' "
                                        f"model={model} "
                                        f"model_args.model_name_or_path='{sftid_path}' "
                                        f"training_args.max_grad_norm={max_grad_norm} "
                                        f"training_args.gradient_accumulation_steps={accumulation_steps} "
                                        f"training_args.per_device_train_batch_size={per_device_train_batch_size} "
                                        f"training_args.optim={optimizer} "
                                        f"training_args.learning_rate={lr} "
                                        f"training_args.loss_type={loss} "
                                        f"training_args.normalize_beta_by_length={normalize_beta_by_length} "
                                        f"training_args.num_ref_rewards={train_num_ref_rewards} "
                                        f"training_args.ref_logprobs_from_dataset={ref_logprobs_from_dataset} "
                                        f"training_args.beta={beta} "
                                        f"seed={seed} "
                                        f"+run_index={run_index} "
                                        f"global_batch_size={batch_size} "
                                        f"num_nodes={num_nodes_per_job} "
                                        f"job_subdir={run_name} "
                                        f"wandb.run_name={run_name} "
                                        f"'wandb.tags=[prod,{job_name}]' "
                                        "artifacts_subdir=private "
                                        "resuming.resume=True "
                                    )
                                )
                                total_nodes_needed += num_nodes_per_job

# Write the submit commands
submit_dir = Path.cwd() / str(stdout_root)
submit_dir.mkdir(parents=True, exist_ok=True)
submit_file = submit_dir / "submit.sh"
print(f"Writing {len(commands)} commands to {submit_file}")
with open(submit_file, "w") as f:
    for command in commands:
        f.write(command + "\n")
print("Total nodes needed:", total_nodes_needed)
