import yaml
import datasets
import argparse

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
        for filter_config in dataset_config["filters"]:
            data = data.filter(
                lambda sample: get_nested_value(sample, filter_config["field"]) in filter_config["values"]
            )
        data = data.map(lambda x: {"aggregation_input_data_path": dataset_config["dataset_path"]})
        dataset_mixture.append(data)
    dataset_mixture = datasets.concatenate_datasets(dataset_mixture)
    print(f"Saving data with {len(dataset_mixture)} samples to: ", args.output)
    dataset_mixture.save_to_disk(args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a decontamination report for a dataset."
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