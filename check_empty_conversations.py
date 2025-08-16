#!/usr/bin/env python3
from datasets import load_from_disk
import json
import hashlib
from tqdm import tqdm

def strip_values(obj):
    if isinstance(obj, dict):
        return {k: strip_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [strip_values(obj[0])] if obj else []
    else:
        return type(obj).__name__

print('Loading dataset...')
dataset = load_from_disk('data/apertus-sft-mixture-7-final-uniform')
print(f'Dataset size: {len(dataset):,} samples')

# Find samples that match each schema pattern
schema_samples = {}
empty_conversation_count = 0

# Check EVERY sample to find schema variations
print(f'Checking ALL {len(dataset):,} samples...')

for i in tqdm(range(len(dataset)), desc="Finding schema patterns"):
    sample = dataset[i]
    
    # Count empty conversations
    conv_branches = sample.get('conversation_branches', [])
    if conv_branches and len(conv_branches) > 0:
        messages = conv_branches[0].get('messages', [])
        if len(messages) == 0:
            empty_conversation_count += 1
    
    # Create schema for this sample
    schema = strip_values(sample)
    schema_str = json.dumps(schema, sort_keys=True)
    hash_val = hashlib.sha256(schema_str.encode()).hexdigest()
    
    if hash_val not in schema_samples:
        schema_samples[hash_val] = {
            'index': i,
            'schema': schema,
            'full_sample': sample,
            'sample_preview': {
                'conversation_id': sample.get('conversation_id'),
                'dataset_source': sample.get('dataset_source'),
                'conversation_branches_count': len(conv_branches),
                'messages_count': len(messages) if conv_branches and conv_branches else 0,
                'has_empty_messages': len(messages) == 0 if conv_branches and conv_branches else True
            }
        }

print(f'\nResults:')
print(f'Total samples checked: {len(dataset):,}')
print(f'Empty conversations found: {empty_conversation_count:,}')
print(f'Unique schema patterns: {len(schema_samples)}')

def find_schema_differences(schema1, schema2, path=''):
    """Find specific differences between two schemas"""
    diffs = []
    
    if isinstance(schema1, dict) and isinstance(schema2, dict):
        all_keys = set(schema1.keys()) | set(schema2.keys())
        for key in all_keys:
            new_path = f'{path}.{key}' if path else key
            if key not in schema1:
                diffs.append(f'{new_path}: Missing in schema1')
            elif key not in schema2:
                diffs.append(f'{new_path}: Missing in schema2')
            else:
                diffs.extend(find_schema_differences(schema1[key], schema2[key], new_path))
    elif isinstance(schema1, list) and isinstance(schema2, list):
        if len(schema1) != len(schema2):
            diffs.append(f'{path}: Different list lengths ({len(schema1)} vs {len(schema2)})')
        else:
            for i, (item1, item2) in enumerate(zip(schema1, schema2)):
                diffs.extend(find_schema_differences(item1, item2, f'{path}[{i}]'))
    elif schema1 != schema2:
        diffs.append(f'{path}: "{schema1}" != "{schema2}"')
    
    return diffs

print(f'\nSchema patterns found:')
schema_list = list(schema_samples.items())

for i, (hash_val, info) in enumerate(schema_list):
    print(f'\n{"="*80}')
    print(f'Schema {i+1} (hash: {hash_val[:8]}...):')
    print(f'  Sample index: {info["index"]}')
    print(f'  Preview: {json.dumps(info["sample_preview"], indent=4)}')
    
    # Show conversation_branches structure
    cb_schema = info["schema"]["conversation_branches"]
    print(f'  Conversation branches schema: {cb_schema}')
    
    print(f'\n  FULL SAMPLE JSON:')
    print(json.dumps(info["full_sample"], indent=2))
    print(f'{"="*80}')

# Show differences between schemas if multiple found
if len(schema_list) > 1:
    print(f'\n{"="*80}')
    print("SCHEMA DIFFERENCES ANALYSIS:")
    print(f'{"="*80}')
    
    for i in range(len(schema_list)):
        for j in range(i+1, len(schema_list)):
            schema1_hash = schema_list[i][0]
            schema2_hash = schema_list[j][0]
            schema1 = schema_list[i][1]["schema"]
            schema2 = schema_list[j][1]["schema"]
            
            print(f'\nDifferences between Schema {i+1} ({schema1_hash[:8]}...) and Schema {j+1} ({schema2_hash[:8]}...):')
            
            differences = find_schema_differences(schema1, schema2)
            
            if differences:
                for diff in differences:
                    print(f'  ❌ {diff}')
            else:
                print(f'  ✅ No differences found (identical schemas)')

if empty_conversation_count > 0:
    print(f'\n{empty_conversation_count} samples have empty conversation messages!')
else:
    print(f'\nNo empty conversations found in sampled data.')