import yaml
import datasets
from datasets import concatenate_datasets
import argparse
from tqdm import tqdm

def get_nested_value(d, key_path):
    keys = key_path.split('.')
    for key in keys:
        d = d[key]
    return d

def main(args):
    # Load the data mixture configs
    with open(args.config_path) as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            quit()

    # Get the config for the given mixture
    try:
        config = configs[args.mixture_name]
    except KeyError as exc:
        print("Update the data-mixtures.yaml file with the configuration for the data mix.")
        print(exc)
        quit()

    dataset_mixture = []
    for dataset_config in config:
        print("Processing dataset {}".format(dataset_config["dataset_path"].split("/")[-1]))
        data = datasets.load_from_disk(dataset_config["dataset_path"])
        
        # Handle DatasetDict vs single Dataset
        if hasattr(data, 'keys'):
            # DatasetDict - concatenate all splits
            print(f"  Found DatasetDict with splits: {list(data.keys())}")
            splits_data = []
            for split_name in data.keys():
                split_data = data[split_name]
                print(f"  Processing split '{split_name}' with {len(split_data)} samples")
                splits_data.append(split_data)
            # Concatenate all splits into a single dataset
            data = concatenate_datasets(splits_data)
            print(f"  Concatenated all splits: {len(data)} total samples")
        
        print(f"  Applying {len(dataset_config['filters'])} filters...")
        for filter_config in dataset_config["filters"]:
            initial_size = len(data)
            data = data.filter(
                lambda sample: get_nested_value(sample, filter_config["field"]) in filter_config["values"],
                desc=f"Filtering by {filter_config['field']}"
            )
            print(f"    Filter '{filter_config['field']}' kept {len(data)}/{initial_size} samples")
        
        print(f"  Adding metadata...")
        data = data.map(
            lambda x: {"aggregation_input_data_path": dataset_config["dataset_path"]},
            desc="Adding aggregation metadata"
        )
        data = data.remove_columns("original_metadata")
        dataset_mixture.append(data)
    print(f"\nConcatenating {len(dataset_mixture)} datasets...")
    dataset_mixture = concatenate_datasets(dataset_mixture)
    print(f"Saving data with {len(dataset_mixture)} samples to: ", args.output)
    dataset_mixture.save_to_disk(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate multiple datasets into a single training mixture."
    )
    parser.add_argument(
        "--mixture_name",
        type=str,
        required=True,
        help="Name of the dataset mixture to create in data-mixtures.yaml",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the data-mixtures.yaml configuration file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for filtered dataset",
    )
    args=parser.parse_args()
    main(args)