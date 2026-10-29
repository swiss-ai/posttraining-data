#!/usr/bin/env python3
"""
gather-decontamination-prompts.py
─────────────────────────────────
Gather evaluation benchmark prompts for decontamination analysis.

This script collects prompts and questions from various evaluation benchmarks
(AGIEval, MMLU, etc.) to create a reference dataset for decontamination.
The gathered prompts are used to identify overlapping content in training data.

Key features:
- Benchmark list read from benchmarks.yaml
- Configurable prompt extraction from different field formats
- Handles various dataset configurations and splits
- Outputs unified prompt format for decontamination pipeline
"""
import argparse
import json
import os

import yaml
from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
from pyarrow.lib import ArrowTypeError
from huggingface_hub import hf_hub_download
from typing import Iterable

DEFAULT_BENCHMARKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks.yaml")


def names_clash(a, b):
    """Whether two prompt-set names can be the same ("<name>__*" stands for all configs of a dataset)."""
    if a.endswith("*"):
        return b.startswith(a[:-1])
    if b.endswith("*"):
        return a.startswith(b[:-1])
    return a == b


def load_benchmark_datasets(path):
    """Read the benchmark list (see benchmarks.yaml for the fields of an entry)."""
    with open(path) as f:
        benchmarks = yaml.safe_load(f)["benchmarks"]
    required = {"name_or_path", "config_name", "split_name", "prompt_col_name"}
    names = []  # the names in the prompt set; "<name>__*" for all configs of an "iterate" entry
    for entry in benchmarks:
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"Benchmark entry {entry} is missing {sorted(missing)}")
        name = entry.get("name", entry["name_or_path"].replace("/", "__"))
        config = entry["config_name"]
        if config == "iterate":
            new = [f"{name}__*"]
        elif isinstance(config, list):
            new = [f"{name}__{c.replace('/', '_')}" for c in config]
        else:
            new = [name]
        for n in new:
            if any(names_clash(n, m) for m in names):
                raise ValueError(f"Duplicate benchmark name {n}; give one of the entries its own 'name'")
            names.append(n)
    return benchmarks


def load_dataset_split(
    dataset_name,
    config_name,
    split_name,
    prompt_col_name,
    num_processes=1,
):
    print(
        f"Loading dataset: {dataset_name}, subset: {config_name}, split: {split_name}"
    )
    try:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split_name,
            num_proc=num_processes,
            trust_remote_code=True,
        )
        prompts = dataset[prompt_col_name]
    except (
        ArrowTypeError
    ) as e:  # Error with Qwen/P-MMEval - mlogiqa, ID column shards have different dtypes
        print("Error while loading: ", e)
        print("Try streaming instead")
        dataset = load_dataset(
            dataset_name,
            config_name,
            split=split_name,
            streaming=True,
            trust_remote_code=True,
        )
        prompts = [x[prompt_col_name] for x in dataset]
    return prompts


def get_prompts(
    dataset_name, config_name, split_name, prompt_col_name, num_processes=1
):
    if isinstance(split_name, str):
        prompts = load_dataset_split(
            dataset_name, config_name, split_name, prompt_col_name, num_processes
        )
        split_names = [split_name] * len(prompts)
    elif isinstance(split_name, Iterable):
        prompts, split_names = [], []
        for s_name in split_name:
            split_prompts = load_dataset_split(
                dataset_name,
                config_name,
                s_name,
                prompt_col_name,
                num_processes=num_processes,
            )
            prompts.extend(split_prompts)
            split_names.extend([s_name] * len(split_prompts))
    else:
        raise ValueError("Invalid split_name type")
    return prompts, split_names


def get_dataset(prompts, split_names):
    prompts_cleaned = []
    for p in prompts:
        if isinstance(p, list):  # list of turns (e.g. mt_bench_prompts): the first user turn
            p = p[0]
        if isinstance(p, str):
            prompts_cleaned.append(p)
        elif isinstance(p, dict):
            if "content" in p:
                prompts_cleaned.append(p["content"])
            elif "text" in p:
                prompts_cleaned.append(p["text"])
            else:
                raise ValueError("Unknown prompt label {}".format(p))
        else:
            raise ValueError("Unknown prompt type {}".format(p))
    return Dataset.from_dict({"prompt": prompts_cleaned, "split_name": split_names})


def main(args):
    decontamination_prompts = {}
    for dataset_args in load_benchmark_datasets(args.benchmarks):
        dataset_name = dataset_args["name_or_path"]
        dataset_name_save = dataset_args.get("name", dataset_name.replace("/", "__"))
        print("\n Processing dataset {}".format(dataset_name))
        if dataset_name == "lmarena-ai/arena-hard-auto":
            # Needs to be processed manually
            path = hf_hub_download(
                repo_id=dataset_name,
                filename="data/arena-hard-v2.0/question.jsonl",
                repo_type="dataset",
            )
            with open(path, "r") as f:
                data = [json.loads(x) for x in f]
            prompts = [x["prompt"] for x in data]
            split_names = ["default" for _ in range(len(prompts))]
            decontamination_prompts[dataset_name_save] = get_dataset(prompts, split_names)
        elif dataset_name == "DAMO-NLP-SG/MultiJail":
            # Needs custom processing as languages are formatted in columns not subsets or splits
            for prompt_col_name in dataset_args["prompt_col_name"]:
                prompts, split_names = get_prompts(
                    dataset_name,
                    dataset_args["config_name"],
                    dataset_args["split_name"],
                    prompt_col_name,
                    args.num_proc,
                )
                decontamination_prompts[dataset_name_save + f"__{prompt_col_name}"] = (
                    get_dataset(prompts, split_names)
                )
        elif (dataset_args["config_name"] != "iterate") and (
            isinstance(dataset_args["config_name"], str)
            or dataset_args["config_name"] is None
        ):
            prompts, split_names = get_prompts(
                dataset_name,
                dataset_args["config_name"],
                dataset_args["split_name"],
                dataset_args["prompt_col_name"],
                args.num_proc,
            )
            decontamination_prompts[dataset_name_save] = get_dataset(prompts, split_names)
        else:
            config_name_list = (
                get_dataset_config_names(dataset_name)
                if (dataset_args["config_name"] == "iterate")
                else dataset_args["config_name"]
            )
            print(
                f"Iterating over subsets in the dataset. Config names in the list: {config_name_list}"
            )
            for config_name in config_name_list:
                try:
                    prompts, split_names = get_prompts(
                        dataset_name,
                        config_name,
                        dataset_args["split_name"],
                        dataset_args["prompt_col_name"],
                        args.num_proc,
                    )
                except Exception as e:
                    # With "iterate", some subsets cannot be loaded (INCLUDE has empty ones, m_hellaswag fails
                    # for zh); configs listed explicitly must load
                    if dataset_args["config_name"] != "iterate":
                        raise
                    print(f"Skipping config name {config_name} due to exception {e}")
                    continue
                decontamination_prompts[
                    dataset_name_save + f"__{config_name.replace('/', '_')}"
                ] = get_dataset(prompts, split_names)
    decontamination_prompts = DatasetDict(decontamination_prompts)
    print("Final dataset")
    print(decontamination_prompts)

    # Save the final datasetdict to the give path
    decontamination_prompts.save_to_disk(args.output)


if __name__ == "__main__":
    """
    python gather_decontamination_prompts.py --output="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"
    """
    parser = argparse.ArgumentParser(description="Gather prompts for decontamination")
    parser.add_argument(
        "--output",
        type=str,
        help="Output path to save the prompts to use for decontamination.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=DEFAULT_BENCHMARKS_FILE,
        help="YAML file listing the benchmarks (default: benchmarks.yaml next to this script).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for loading and mapping the benchmarks. More than 1 can deadlock in "
             "load_dataset (seen with 16 on the cluster).",
    )
    args = parser.parse_args()
    main(args)
