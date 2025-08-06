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
            lambda x: {**x, "aggregation_input_data_path": dataset_config["dataset_path"]},
            desc="Adding aggregation metadata"
        )
        # Keep original_metadata for new format datasets
        dataset_mixture.append(data)
    print(f"\nConcatenating {len(dataset_mixture)} datasets...")
    
    # Add missing fields to align all datasets
    print("  Aligning dataset schemas...")
    aligned_datasets = []
    
    for i, ds in tqdm(enumerate(dataset_mixture), total=len(dataset_mixture), desc="Aligning schemas"):
        dataset_name = ds[0]['dataset_source'] if len(ds) > 0 else f"dataset_{i+1}"
        
        # Check if dataset has available_functions field
        if 'available_functions' not in ds.column_names:
            print(f"    Adding available_functions to {dataset_name}")
            ds = ds.map(lambda x: {**x, 'available_functions': []}, desc="Adding available_functions")
            
            # Force consistent typing by adding a dummy function and removing it
            dummy_func = {
                'name': 'dummy',
                'description': 'dummy', 
                'parameters': {'type': 'object', 'properties': '{}', 'required': []}
            }
            ds = ds.map(lambda x: {**x, 'available_functions': [dummy_func]}, desc="Adding dummy function")
            ds = ds.map(lambda x: {**x, 'available_functions': []}, desc="Removing dummy function")
        
        # Ensure system_prompt exists (some datasets have it as null)
        def ensure_system_prompt(sample):
            if sample.get('system_prompt') is None:
                sample['system_prompt'] = {'content': '', 'metadata': {}}
            return sample
        
        ds = ds.map(ensure_system_prompt, desc="Ensuring system_prompt")
        
        # Harmonize parts structure to include all fields with consistent typing
        def harmonize_parts(sample):
            for branch in sample.get('conversation_branches', []):
                for message in branch.get('messages', []):
                    if 'parts' in message:
                        harmonized_parts = []
                        for part in message['parts']:
                            # Ensure all parts have the same fields with consistent string typing
                            # Use empty string instead of None to ensure consistent Arrow typing
                            harmonized_part = {
                                'type': part.get('type') or '',
                                'content': part.get('content') or '',
                                'metadata': part.get('metadata', {}),
                                'name': part.get('name') or '',
                                'args': part.get('args') or ''
                            }
                            harmonized_parts.append(harmonized_part)
                        message['parts'] = harmonized_parts
            return sample
        
        ds = ds.map(harmonize_parts, desc="Harmonizing parts structure")
        aligned_datasets.append(ds)
    
    # Convert all datasets to unified format and concatenate
    print("  Converting to unified format...")
    all_samples = []
    
    for ds in tqdm(aligned_datasets, desc="Collecting samples"):
        for sample in ds:
            all_samples.append(sample)
    
    print(f"  Creating unified dataset with {len(all_samples):,} samples...")
    dataset_mixture = datasets.Dataset.from_list(all_samples)
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