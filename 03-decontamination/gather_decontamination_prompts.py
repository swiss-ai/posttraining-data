import argparse
import json
import os

from datasets import load_dataset, Dataset, DatasetDict, get_dataset_config_names
from pyarrow.lib import ArrowTypeError
from huggingface_hub import hf_hub_download
from typing import Iterable

agieval_datasets = [
    {
        "name_or_path": dataset_name,
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "query",
    }
    for dataset_name in [
        "hails/agieval-aqua-rat",
        "hails/agieval-gaokao-biology",
        "hails/agieval-gaokao-chemistry",
        "hails/agieval-gaokao-chinese",
        "hails/agieval-gaokao-english",
        "hails/agieval-gaokao-geography",
        "hails/agieval-gaokao-history",
        "hails/agieval-gaokao-mathqa",
        "hails/agieval-gaokao-physics",
        "hails/agieval-logiqa-en",
        "hails/agieval-logiqa-zh",
        "hails/agieval-sat-math",
        "hails/agieval-lsat-ar",
        "hails/agieval-lsat-lr",
        "hails/agieval-lsat-rc",
        "hails/agieval-sat-en",
        "hails/agieval-sat-en-without-passage",
        "hails/agieval-math",
        "hails/agieval-gaokao-mathcloze",
        "hails/agieval-jec-qa-kd",
        "hails/agieval-jec-qa-ca",
    ]
]

BENCHMARK_DATASETS = [
    {
        "name_or_path": "cais/mmlu",
        "config_name": "all",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "TIGER-Lab/MMLU-Pro",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/Global-MMLU",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "li-lab/MMLU-ProX",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "truthfulqa/truthful_qa",
        "config_name": "iterate",
        "split_name": "validation",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "alexandrainst/m_truthfulqa",
        "config_name": "iterate",
        "split_name": "val",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/include-base-44",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "include-base_v2",
        "config_name": "iterate",
        "split_name": None,
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "lukaemon/bbh",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "input",
    },
    {
        "name_or_path": "EleutherAI/drop",
        "config_name": None,
        "split_name": "validation",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "ibm-research/acp_bench",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "allenai/ai2_arc",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    # {
    #     "name_or_path": "LumiOpen/arc_challenge_mt",
    #     "config_name": "iterate",
    #     "split_name": "test",
    #     "prompt_col_name": "question",
    # },
    {
        "name_or_path": "alexandrainst/m_arc",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "instruction",
    },
    {
        "name_or_path": "Idavidrein/gpqa",
        "config_name": "gpqa_main",
        "split_name": "train",
        "prompt_col_name": "Pre-Revision Question",
    },
    {
        "name_or_path": "Qwen/P-MMEval",
        "config_name": ["mlogiqa"],
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "juletxara/mgsm",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "DigitalLearningGmbH/MATH-lighteval",
        "config_name": "default",
        "split_name": "test",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "openai/gsm8k",
        "config_name": "main",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "madrylab/gsm8k-platinum",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "allenai/math_qa",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "Problem",
    },
    {
        "name_or_path": "EleutherAI/hendrycks_math",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "AI-MO/aimo-validation-aime",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "problem",
    },
    {
        "name_or_path": "Qwen/PolyMath",
        "config_name": "iterate",
        "split_name": ["top", "high", "medium", "low"],
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "openai/openai_humaneval",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "THUDM/LongBench",
        "config_name": ["hotpotqa"],
        "split_name": "test",
        "prompt_col_name": "input",
    },
    {
        "name_or_path": "google-research-datasets/mbpp",
        "config_name": "full",
        "split_name": "test",
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "bigcode/bigcodebench",
        "config_name": None,
        "split_name": "v0.1.0_hf",
        "prompt_col_name": "instruct_prompt",
    },
    {
        "name_or_path": "google/IFEval",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "Rowan/hellaswag",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "ctx",
    },
    {
        "name_or_path": "alexandrainst/m_hellaswag",
        "config_name": "iterate",
        "split_name": "val",
        "prompt_col_name": "ctx",
    },
    {
        "name_or_path": "facebook/Multi-IF",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "turn_1_prompt",
    },
    {
        "name_or_path": "tatsu-lab/alpaca_eval",
        "config_name": None,
        "split_name": "eval",
        "prompt_col_name": "instruction",
    },
    {
        "name_or_path": "CohereLabs/m-ArenaHard-v2.0",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "lmarena-ai/arena-hard-auto",
        "config_name": None,
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "tatsu-lab/alpaca_eval",
        "config_name": None,
        "split_name": "eval",
        "prompt_col_name": "instruction",
    },
    {
        "name_or_path": "toxigen/toxigen-data",
        "config_name": "prompts",
        "split_name": [
            "hate_trans_1k",
            "neutral_black_1k",
            "hate_native_american_1k",
            "neutral_immigrant_1k",
            "hate_middle_east_1k",
            "neutral_lgbtq_1k",
            "neutral_women_1k",
            "neutral_chinese_1k",
            "hate_latino_1k",
            "hate_bisexual_1k",
            "hate_mexican_1k",
            "hate_asian_1k",
            "neutral_mental_disability_1k",
            "neutral_mexican_1k",
            "hate_mental_disability_1k",
            "neutral_bisexual_1k",
            "neutral_latino_1k",
            "hate_chinese_1k",
            "neutral_jewish_1k",
            "hate_muslim_1k",
            "neutral_asian_1k",
            "hate_physical_disability_1k",
            "hate_jewish_1k",
            "neutral_muslim_1k",
            "hate_immigrant_1k",
            "hate_black_1k",
            "hate_lgbtq_1k",
            "hate_women_1k",
            "neutral_middle_east_1k",
            "neutral_native_american_1k",
            "neutral_physical_disability_1k",
        ],
        "prompt_col_name": "text",
    },
    {
        "name_or_path": "allenai/real-toxicity-prompts",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "oskarvanderwal/bbq",
        "config_name": "All",
        "split_name": "test",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "CohereLabs/aya_redteaming",
        "config_name": None,
        "split_name": [
            "arabic",
            "english",
            "filipino",
            "french",
            "hindi",
            "russian",
            "serbian",
            "spanish",
        ],
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "ToxicityPrompts/PolygloToxicityPrompts",
        "config_name": "iterate",
        "split_name": "full",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "nayeon212/BLEnD",
        "config_name": ["multiple-choice-questions"],
        "split_name": "test",
        "prompt_col_name": "prompt",
    },
    {
        "name_or_path": "shanearora/CaLMQA",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": "question",
    },
    {
        "name_or_path": "kellycyy/CulturalBench",
        "config_name": "iterate",
        "split_name": "test",
        "prompt_col_name": "prompt_question",
    },
    {
        "name_or_path": "swissai/harmbench",
        "config_name": ["DirectRequest", "HumanJailbreaks"],
        "split_name": "test",
        "prompt_col_name": "Behavior",
    },
    {
        "name_or_path": "DAMO-NLP-SG/MultiJail",
        "config_name": None,
        "split_name": "train",
        "prompt_col_name": ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"],
    },
] + agieval_datasets


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
    for dataset_args in BENCHMARK_DATASETS:
        dataset_name = dataset_args["name_or_path"]
        dataset_name_save = dataset_name.replace("/", "__")
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
        elif dataset_name == "include-base_v2":
            data_dir = "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/include_v2_prompts"
            filenames = [x for x in os.listdir(data_dir) if x.endswith(".json")]
            for filename in filenames:
                with open(os.path.join(data_dir, filename)) as f:
                    prompts = [x["question"] for x in json.load(f)]
                language = filename.split(".json")[0]
                split_names = ["default" for _ in range(len(prompts))]
                decontamination_prompts[f"include-base_v2__{language}"] = get_dataset(prompts, split_names)
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
                except (
                    Exception
                ) as e:  # Error with INCLUDE (some subsets are empty) or m_hellaswag (errors with the subset ZH)
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
        "--num_proc",
        type=int,
        default=16,
        help="Number of processes to use for map operations.",
    )
    args = parser.parse_args()
    main(args)
