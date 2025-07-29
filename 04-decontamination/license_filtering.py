import argparse
import os.path

import datasets

def main(args):
    assert "02_standardised" in args.dataset_path, "Dataset must be in the 02_standardised directory!"
    dataset = datasets.load_from_disk(args.dataset_path)
    print("Dataset loaded from: ", args.dataset_path)
    print(dataset)
    dataset_name = args.dataset_path.split("/")[-1]
    output_path = args.dataset_path.replace("02_standardised", "03_license_filtered")

    if dataset_name == "tulu-3-sft-mixture":
        tmp_path = "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter_tulu_3_sft_mixture.arrow"
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in ["ai2-adapt-dev/tulu_hard_coded_repeated_10", "ai2-adapt-dev/no_robots_converted"],
            cache_file_name=tmp_path,
        )
    elif dataset_name == "EuroBlocks-SFT-Synthetic-1124":
        dataset_filtered = dataset
    elif dataset_name == "smoltalk":
        tmp_path = "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter_smoltalk.arrow"
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in [
                "openhermes-100k",
                "longalign",
                "explore-instruct-rewriting",
            ],
            cache_file_name=tmp_path,
        )
    elif dataset_name == "smoltalk2":
        for datasplit_name in [
            "LongAlign_64k_Qwen3_32B_yarn_131k_think",
            "LongAlign_64k_context_lang_annotated_lang_6_no_think",
            "OpenHermes_2.5_no_think",
            "smoltalk_smollm3_explore_instruct_rewriting_no_think",
            "Mixture_of_Thoughts_science_no_think",
            "hermes_function_calling_v1_no_think",
            "xlam_traces_no_think",  # Removing due to non-compatibility of schemas
            "smolagents_toolcalling_traces_think",  # Removing due to non-compatibility of schemas
        ]:
            del dataset[datasplit_name]
        augmented_splits = []
        for split_name, dataset in dataset.items():
            dataset = dataset.map(lambda x: {"original_metadata": x["original_metadata"] | {"source": split_name}})
            augmented_splits.append(dataset)
        dataset_filtered = datasets.concatenate_datasets(augmented_splits)
    elif dataset_name == "The-Tome":
        tmp_path = "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter_the_tome.arrow"
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["dataset"] not in [
                "infini-instruct-top-500k",
                "ultrainteract_trajectories_sharegpt",
                "qwen2-72b-magpie-en",
            ],
            cache_file_name=tmp_path,
        )
    elif dataset_name == "AceReason-1.1-SFT":
        tmp_path = "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter_acereason_sft.arrow"
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["source"] not in ["leetcode"],
            cache_file_name=tmp_path,
        )
    elif dataset_name in ["Llama-Nemotron-Post-Training-Dataset", "Llama-Nemotron-Post-Training-Dataset_wo_math_code"]:
        print("Processing Llama-Nemotron-Post-Training-Dataset")
        tmp_path = "/capstor/store/cscs/swissai/infra01/posttrain_data/02_standardised/tmp-filter_nemotron.arrow"
        dataset_filtered = dataset.filter(
            lambda x: x["original_metadata"]["license"] in ["cc-by-4.0", "odc-by"],
            cache_file_name=tmp_path,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print("Number of samples removed", len(dataset) - len(dataset_filtered))
    print("Saving filtered dataset to: ", output_path)
    dataset_filtered.save_to_disk(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='The dataset to process')
    args = parser.parse_args()
    main(args)