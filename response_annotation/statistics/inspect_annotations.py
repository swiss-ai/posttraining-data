"""Inspect the 'annotations' column of the combined annotated dataset."""

import json
from pathlib import Path
from datasets import load_from_disk

DS_DIR = Path("/iopsstor/scratch/cscs/dmelikidze/posttraining-data/response_annotation/datasets/combined_annotated")

ds = load_from_disk(str(DS_DIR))
if "train" in ds:
    ds = ds["train"]

print(f"Dataset: {len(ds)} rows, columns: {ds.column_names}\n")
ss = ds["annotations"][0]
# print(ss)
print(type(ss))
ss = json.loads(ss)
print(ss[0].keys())  # Print the raw JSON string of the first row's annotations

exit()
# Inspect structure of the 'annotations' column using row 0
annotations = json.loads(ds[0]["annotations"])


def describe_type(value, indent=0):
    """Recursively describe the type structure of a value."""
    prefix = "  " * indent
    if isinstance(value, dict):
        print(f"{prefix}dict with {len(value)} keys:")
        for k, v in value.items():
            print(f"{prefix}  '{k}': ", end="")
            if isinstance(v, (dict, list)):
                print()
                describe_type(v, indent + 2)
            else:
                print(f"{type(v).__name__} = {repr(v)}")
    elif isinstance(value, list):
        print(f"{prefix}list[{len(value)}]")
        if value:
            print(f"{prefix}  [0]:")
            describe_type(value[0], indent + 2)
    else:
        print(f"{prefix}{type(value).__name__} = {repr(value)}")


print("=== 'annotations' column structure (row 0) ===")
print(f"Top level: JSON string -> ", end="")
describe_type(annotations)
