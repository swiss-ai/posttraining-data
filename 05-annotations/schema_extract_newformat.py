#!/usr/bin/env python3
"""
Schema Extraction Script for New Chat Format Datasets

This script analyzes HuggingFace datasets to extract all schema variants,
save them to files, and generate a unified schema that encompasses all variants.
It creates the foundation for schema normalization preprocessing.

Designed specifically for new chat format datasets with parts structure.
"""

import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm


def extract_schema_structure(sample: Dict[str, Any], path: str = "", max_depth: int = 4) -> Dict[str, str]:
    """
    Extract the schema structure from a sample, mapping field paths to types.
    Optimized version with depth limiting to prevent exponential slowdown.
    
    Args:
        sample: Sample data to analyze
        path: Current path in the nested structure
        max_depth: Maximum recursion depth to prevent performance issues
        
    Returns:
        Dictionary mapping field paths to their types
    """
    schema = {}
    
    if sample is None:
        return {path: "null"}
    
    # Stop recursion if we've gone too deep
    depth = path.count('.')
    if depth >= max_depth:
        return {path: "deep_nested"}
    
    if isinstance(sample, dict):
        if not sample:  # Empty dict
            schema[path] = "empty_dict"
        else:
            schema[path] = "dict"
            # Only analyze key fields that matter for schema consistency
            key_fields = ['metadata', 'content', 'role', 'type', 'parts']
            for key, value in sample.items():
                if key in key_fields or depth < 2:  # Always analyze shallow fields
                    new_path = f"{path}.{key}" if path else key
                    schema.update(extract_schema_structure(value, new_path, max_depth))
    
    elif isinstance(sample, list):
        if not sample:  # Empty list
            schema[path] = "empty_list"
        else:
            schema[path] = "list"
            # Only analyze first item to understand list element types
            if len(sample) > 0:
                item_path = f"{path}[0]"
                schema.update(extract_schema_structure(sample[0], item_path, max_depth))
    
    elif isinstance(sample, str):
        schema[path] = "string"
    elif isinstance(sample, (int, float)):
        schema[path] = "number"
    elif isinstance(sample, bool):
        schema[path] = "boolean"
    else:
        schema[path] = f"other({type(sample).__name__})"
    
    return schema


def compare_schemas(schema1: Dict[str, str], schema2: Dict[str, str]) -> Dict[str, Any]:
    """
    Compare two schema structures and identify differences.
    
    Args:
        schema1: First schema to compare
        schema2: Second schema to compare
        
    Returns:
        Dictionary with difference analysis
    """
    all_fields = set(schema1.keys()) | set(schema2.keys())
    
    differences = {
        "missing_in_first": [],
        "missing_in_second": [],
        "type_mismatches": [],
        "total_fields_first": len(schema1),
        "total_fields_second": len(schema2),
        "common_fields": len(set(schema1.keys()) & set(schema2.keys()))
    }
    
    for field in all_fields:
        if field not in schema1:
            differences["missing_in_first"].append({
                "field": field,
                "type_in_second": schema2[field]
            })
        elif field not in schema2:
            differences["missing_in_second"].append({
                "field": field,
                "type_in_first": schema1[field]
            })
        elif schema1[field] != schema2[field]:
            differences["type_mismatches"].append({
                "field": field,
                "type_in_first": schema1[field],
                "type_in_second": schema2[field]
            })
    
    return differences


def normalize_schema_key(schema: Dict[str, str]) -> str:
    """
    Create a normalized key for grouping similar schemas.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        String key for grouping
    """
    # Sort fields and create a signature
    sorted_fields = sorted(schema.items())
    return json.dumps(sorted_fields, sort_keys=True)


def process_sample_schema(args):
    """
    Process a single sample for schema extraction (for parallel processing).
    
    Args:
        args: Tuple of (sample_idx, sample_data)
        
    Returns:
        Tuple of (sample_idx, schema_dict, schema_key)
    """
    sample_idx, sample = args
    schema = extract_schema_structure(sample)
    schema_key = normalize_schema_key(schema)
    return (sample_idx, schema, schema_key)


def analyze_dataset_schemas(dataset: Dataset, max_samples: int = 2000, 
                          random_seed: int = 42, num_proc: int = None) -> Dict[str, Any]:
    """
    Analyze schema consistency across dataset samples.
    
    Args:
        dataset: HuggingFace Dataset to analyze
        max_samples: Maximum number of samples to check
        random_seed: Random seed for sampling
        num_proc: Number of processes for parallel processing
        
    Returns:
        Dictionary with analysis results
    """
    total_samples = len(dataset)
    samples_to_check = min(max_samples, total_samples)
    
    # Set default number of processes
    if num_proc is None:
        num_proc = min(cpu_count(), 16)  # Cap at 16 to avoid memory issues
    else:
        # Cap user-specified processes at 32 for very large datasets
        num_proc = min(num_proc, 32)
    
    # Smart sampling strategy: spread across dataset for maximum schema diversity
    if total_samples <= max_samples:
        indices = list(range(total_samples))
    else:
        # Take samples spread across the entire dataset
        step = total_samples // samples_to_check
        indices = []
        
        # Evenly distributed samples
        for i in range(0, total_samples, step):
            if len(indices) < samples_to_check:
                indices.append(i)
        
        # Add some random samples to fill remaining slots
        if len(indices) < samples_to_check:
            random.seed(random_seed)
            remaining_slots = samples_to_check - len(indices)
            excluded_set = set(indices)
            available_indices = [i for i in range(total_samples) if i not in excluded_set]
            additional_indices = random.sample(available_indices, 
                                             min(remaining_slots, len(available_indices)))
            indices.extend(additional_indices)
    
    print(f"Analyzing {len(indices):,} samples out of {total_samples:,} total samples using {num_proc} processes...")
    
    schema_groups = defaultdict(list)  # Schema key -> list of (sample_idx, schema)
    sample_schemas = {}  # sample_idx -> schema
    
    # Process in chunks to avoid memory issues with large datasets
    chunk_size = min(1000, len(indices) // num_proc) if num_proc > 1 else len(indices)
    
    if num_proc > 1:
        # Parallel processing with chunking
        with Pool(num_proc) as pool:
            for i in tqdm(range(0, len(indices), chunk_size), desc="Processing chunks"):
                chunk_indices = indices[i:i + chunk_size]
                
                # Prepare chunk data (load samples just in time)
                chunk_data = []
                for idx in chunk_indices:
                    try:
                        sample = dataset[idx]
                        chunk_data.append((idx, sample))
                    except Exception as e:
                        print(f"Warning: Could not load sample {idx}: {e}")
                        continue
                
                # Process chunk in parallel
                chunk_results = pool.map(process_sample_schema, chunk_data)
                
                # Collect results
                for sample_idx, schema, schema_key in chunk_results:
                    schema_groups[schema_key].append((sample_idx, schema))
                    sample_schemas[sample_idx] = schema
    else:
        # Sequential processing
        for sample_idx in tqdm(indices, desc="Extracting schemas"):
            try:
                sample = dataset[sample_idx]
                sample_idx, schema, schema_key = process_sample_schema((sample_idx, sample))
                schema_groups[schema_key].append((sample_idx, schema))
                sample_schemas[sample_idx] = schema
            except Exception as e:
                print(f"Warning: Could not process sample {sample_idx}: {e}")
                continue
    
    # Find the most common schema (primary)
    schema_counts = {key: len(samples) for key, samples in schema_groups.items()}
    primary_schema_key = max(schema_counts, key=schema_counts.get)
    primary_schema = schema_groups[primary_schema_key][0][1]  # Get schema from first sample
    
    # Calculate actual samples processed
    actual_samples_processed = len(indices)
    
    # Analyze differences from primary schema
    schema_analysis = {
        "total_samples_checked": actual_samples_processed,
        "total_samples_in_dataset": total_samples,
        "unique_schemas_found": len(schema_groups),
        "primary_schema_key": primary_schema_key,
        "primary_schema": primary_schema,
        "schema_distribution": schema_counts,
        "primary_schema_samples": len(schema_groups[primary_schema_key]),
        "consistency_percentage": (schema_counts[primary_schema_key] / actual_samples_processed) * 100,
        "variant_schemas": {},
        "sample_indices_by_schema": {key: [idx for idx, _ in samples] 
                                   for key, samples in schema_groups.items()}
    }
    
    # Analyze each variant schema against primary
    for schema_key, samples in schema_groups.items():
        if schema_key != primary_schema_key:
            variant_schema = samples[0][1]  # Get schema from first sample
            differences = compare_schemas(primary_schema, variant_schema)
            
            schema_analysis["variant_schemas"][schema_key] = {
                "sample_count": len(samples),
                "percentage": (len(samples) / actual_samples_processed) * 100,
                "differences": differences,
                "example_sample_indices": [idx for idx, _ in samples[:5]]  # First 5 examples
            }
    
    return schema_analysis


def create_unified_schema(schema_analysis: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a unified schema that encompasses all field variations.
    
    Args:
        schema_analysis: Analysis results from analyze_dataset_schemas
        
    Returns:
        Unified schema with all possible fields and their types
    """
    unified_schema = schema_analysis['primary_schema'].copy()
    
    # Add fields from all variants
    for variant_info in schema_analysis['variant_schemas'].values():
        differences = variant_info['differences']
        
        # Add missing fields from variants
        for field_info in differences['missing_in_first']:
            field_path = field_info['field']
            field_type = field_info['type_in_second']
            unified_schema[field_path] = field_type
        
        # For type mismatches, prefer non-null types
        for field_info in differences['type_mismatches']:
            field_path = field_info['field']
            type_in_first = field_info['type_in_first']
            type_in_second = field_info['type_in_second']
            
            # Prefer non-null types
            if type_in_first == 'null' and type_in_second != 'null':
                unified_schema[field_path] = type_in_second
            elif type_in_second == 'null' and type_in_first != 'null':
                unified_schema[field_path] = type_in_first
            # For other conflicts, keep the primary schema type
    
    return unified_schema


def save_schema_analysis(analysis: Dict[str, Any], output_dir: Path):
    """
    Save schema analysis results to files.
    
    Args:
        analysis: Analysis results from analyze_dataset_schemas
        output_dir: Directory to save schema files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete analysis
    analysis_file = output_dir / "schema_analysis_complete.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Complete analysis saved to: {analysis_file}")
    
    # Save primary schema
    primary_file = output_dir / "schema_primary.json"
    with open(primary_file, 'w') as f:
        json.dump(analysis['primary_schema'], f, indent=2, sort_keys=True)
    print(f"Primary schema saved to: {primary_file}")
    
    # Save each variant schema
    variants_dir = output_dir / "schema_variants"
    variants_dir.mkdir(exist_ok=True)
    
    for i, (schema_key, variant_info) in enumerate(analysis['variant_schemas'].items(), 1):
        # Get the first sample with this schema to extract its structure
        sample_indices = analysis['sample_indices_by_schema'][schema_key]
        
        variant_file = variants_dir / f"variant_{i:02d}.json"
        variant_data = {
            "sample_count": variant_info['sample_count'],
            "percentage": variant_info['percentage'],
            "example_sample_indices": variant_info['example_sample_indices'],
            "differences_from_primary": variant_info['differences']
        }
        
        with open(variant_file, 'w') as f:
            json.dump(variant_data, f, indent=2)
        print(f"Variant {i} saved to: {variant_file}")
    
    # Create and save unified schema
    unified_schema = create_unified_schema(analysis)
    unified_file = output_dir / "schema_unified.json"
    with open(unified_file, 'w') as f:
        json.dump(unified_schema, f, indent=2, sort_keys=True)
    print(f"Unified schema saved to: {unified_file}")
    
    # Save normalization template
    norm_template = generate_normalization_template(unified_schema)
    template_file = output_dir / "normalization_template.py"
    with open(template_file, 'w') as f:
        f.write(norm_template)
    print(f"Normalization template saved to: {template_file}")
    
    return unified_schema


def generate_normalization_template(unified_schema: Dict[str, str]) -> str:
    """
    Generate Python code template for schema normalization.
    
    Args:
        unified_schema: Unified schema with all possible fields
        
    Returns:
        Python code string for normalization function
    """
    template = '''#!/usr/bin/env python3
"""
Auto-generated schema normalization template.
Customize default values as needed for your dataset.
"""

def get_default_value_for_type(field_type: str):
    """Get appropriate default value for a field type."""
    if field_type == "null":
        return None
    elif field_type == "string":
        return ""
    elif field_type == "number":
        return 0
    elif field_type == "boolean":
        return False
    elif field_type == "list" or field_type == "empty_list":
        return []
    elif field_type == "dict" or field_type == "empty_dict":
        return {}
    else:
        return None

def normalize_sample_schema(sample):
    """Normalize a single sample to match unified schema."""
    import copy
    normalized = copy.deepcopy(sample)
    
    # Add all missing fields with appropriate defaults
'''
    
    # Group fields by their parent paths for easier processing
    field_groups = defaultdict(list)
    for field_path, field_type in sorted(unified_schema.items()):
        if '.' in field_path:
            parent_path = '.'.join(field_path.split('.')[:-1])
            field_name = field_path.split('.')[-1]
            field_groups[parent_path].append((field_name, field_type))
        else:
            field_groups['ROOT'].append((field_path, field_type))
    
    # Generate normalization code for each group
    for parent_path, fields in field_groups.items():
        if parent_path == 'ROOT':
            for field_name, field_type in fields:
                template += f'    if "{field_name}" not in normalized:\n'
                template += f'        normalized["{field_name}"] = get_default_value_for_type("{field_type}")\n'
        else:
            template += f'    \n    # Normalize fields in {parent_path}\n'
            # This is simplified - real implementation would need recursive path handling
            template += f'    # TODO: Add normalization for {parent_path} fields\n'
    
    template += '''
    
    return normalized

def normalize_batch_schema(examples):
    """Normalize a batch of samples for dataset.map()."""
    import copy
    result = copy.deepcopy(examples)
    
    batch_size = len(examples[next(iter(examples))])
    
    for idx in range(batch_size):
        sample = {key: result[key][idx] for key in result}
        normalized_sample = normalize_sample_schema(sample)
        
        # Put back into batch
        for key in result:
            result[key][idx] = normalized_sample[key]
    
    return result
'''
    
    return template


def generate_schema_report(analysis: Dict[str, Any], verbose: bool = False) -> str:
    """
    Generate a detailed schema analysis report.
    
    Args:
        analysis: Analysis results from analyze_dataset_schemas
        verbose: Whether to include detailed field listings
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("DATASET SCHEMA ANALYSIS REPORT")
    report.append("=" * 80)
    
    # Summary statistics
    report.append(f"\nSUMMARY:")
    report.append(f"  Total samples in dataset: {analysis['total_samples_in_dataset']:,}")
    report.append(f"  Samples analyzed: {analysis['total_samples_checked']:,}")
    report.append(f"  Unique schemas found: {analysis['unique_schemas_found']}")
    report.append(f"  Schema consistency: {analysis['consistency_percentage']:.1f}%")
    
    # Primary schema info
    primary_count = analysis['primary_schema_samples']
    report.append(f"\nPRIMARY SCHEMA (most common):")
    report.append(f"  Samples using primary schema: {primary_count:,} ({analysis['consistency_percentage']:.1f}%)")
    report.append(f"  Total fields in primary schema: {len(analysis['primary_schema'])}")
    
    if verbose:
        report.append(f"\n  Primary schema fields:")
        for field, field_type in sorted(analysis['primary_schema'].items()):
            report.append(f"    {field}: {field_type}")
    
    # Schema variants
    if analysis['variant_schemas']:
        report.append(f"\nSCHEMA VARIANTS:")
        
        for i, (schema_key, variant_info) in enumerate(analysis['variant_schemas'].items(), 1):
            count = variant_info['sample_count']
            percentage = variant_info['percentage']
            differences = variant_info['differences']
            examples = variant_info['example_sample_indices']
            
            report.append(f"\n  Variant {i}: {count:,} samples ({percentage:.1f}%)")
            report.append(f"    Example sample indices: {examples}")
            
            # Show differences
            if differences['missing_in_first']:
                report.append(f"    Fields only in variant (not in primary):")
                for field_info in differences['missing_in_first'][:10]:  # Limit to 10
                    report.append(f"      + {field_info['field']}: {field_info['type_in_second']}")
                if len(differences['missing_in_first']) > 10:
                    report.append(f"      ... and {len(differences['missing_in_first']) - 10} more")
            
            if differences['missing_in_second']:
                report.append(f"    Fields only in primary (not in variant):")
                for field_info in differences['missing_in_second'][:10]:  # Limit to 10
                    report.append(f"      - {field_info['field']}: {field_info['type_in_first']}")
                if len(differences['missing_in_second']) > 10:
                    report.append(f"      ... and {len(differences['missing_in_second']) - 10} more")
            
            if differences['type_mismatches']:
                report.append(f"    Type mismatches:")
                for field_info in differences['type_mismatches'][:10]:  # Limit to 10
                    report.append(f"      ~ {field_info['field']}: {field_info['type_in_first']} → {field_info['type_in_second']}")
                if len(differences['type_mismatches']) > 10:
                    report.append(f"      ... and {len(differences['type_mismatches']) - 10} more")
    
    # Recommendations
    report.append(f"\nRECOMMENDATIONS:")
    
    if analysis['unique_schemas_found'] == 1:
        report.append("  ✅ Dataset has consistent schema - safe for dataset.map() with parallel processing")
    else:
        inconsistent_percentage = 100 - analysis['consistency_percentage']
        report.append(f"  ⚠️  Dataset has schema inconsistencies ({inconsistent_percentage:.1f}% of samples)")
        report.append("  📋 Consider these approaches:")
        report.append("     1. Use sequential processing (num_proc=1) with dataset.map()")
        report.append("     2. Pre-process dataset to normalize schema before annotation")
        report.append("     3. Use manual iteration instead of dataset.map()")
        
        if any('metadata' in str(variant) for variant in analysis['variant_schemas'].values()):
            report.append("  🔍 Detected metadata field inconsistencies - likely cause of Arrow errors")
    
    report.append("\n" + "=" * 80)
    
    return "\n".join(report)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract schema variants and create unified schema for new chat format datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract schemas and save to files
  venv/bin/python 05-annotations/schema_extract_newformat.py data/path/dataset --save-schemas schemas/
  
  # Check more samples for thorough analysis
  venv/bin/python 05-annotations/schema_extract_newformat.py data/path/dataset --max-samples 5000 --save-schemas schemas/
  
  # Just generate report without saving
  venv/bin/python 05-annotations/schema_extract_newformat.py data/path/dataset --verbose
  
  # Save both report and schemas
  venv/bin/python 05-annotations/schema_extract_newformat.py data/path/dataset --save-schemas schemas/ --output report.txt
        """
    )
    
    parser.add_argument(
        "dataset_path",
        help="Path to dataset directory to analyze"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples to analyze (default: 2000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include detailed field listings in report"
    )
    parser.add_argument(
        "--output",
        help="Save report to file instead of printing to console"
    )
    parser.add_argument(
        "--save-schemas",
        help="Directory to save schema files (variants, unified schema, normalization template)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42)"
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for parallel schema extraction (default: auto, max 16)"
    )
    
    return parser.parse_args()


def main():
    """Main extraction function."""
    args = parse_arguments()
    
    # Validate input path
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    try:
        dataset = load_from_disk(str(dataset_path))
        
        # Handle DatasetDict vs single Dataset
        if isinstance(dataset, DatasetDict):
            available_splits = list(dataset.keys())
            print(f"Found DatasetDict with splits: {available_splits}")
            
            if 'train' in available_splits:
                dataset = dataset['train']
                print(f"Using 'train' split")
            else:
                first_split = available_splits[0]
                dataset = dataset[first_split]
                print(f"Using '{first_split}' split")
        
        print(f"Dataset size: {len(dataset):,} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Analyze schema consistency
    print(f"\nStarting schema analysis...")
    print(f"Max samples to check: {args.max_samples:,}")
    print(f"Random seed: {args.random_seed}")
    
    analysis = analyze_dataset_schemas(
        dataset, 
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        num_proc=args.num_proc
    )
    
    # Save schemas if requested
    unified_schema = None
    if args.save_schemas:
        output_dir = Path(args.save_schemas)
        print(f"\nSaving schema analysis to: {output_dir}")
        unified_schema = save_schema_analysis(analysis, output_dir)
        print(f"\n📁 Schema files saved to: {output_dir}")
        print(f"   - schema_analysis_complete.json: Full analysis")
        print(f"   - schema_primary.json: Most common schema")
        print(f"   - schema_unified.json: Unified schema with all fields")
        print(f"   - schema_variants/: Individual variant details")
        print(f"   - normalization_template.py: Code template for normalization")
    
    # Generate report
    report = generate_schema_report(analysis, verbose=args.verbose)
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    else:
        print("\n" + report)
    
    # Summary with next steps
    if analysis['unique_schemas_found'] == 1:
        print(f"\n✅ Schema extraction COMPLETE - dataset is consistent")
        print(f"   No normalization needed for parallel processing.")
    else:
        inconsistent_count = analysis['total_samples_checked'] - analysis['primary_schema_samples']
        print(f"\n📊 Schema extraction COMPLETE - found {analysis['unique_schemas_found']} variants")
        print(f"   Inconsistent samples: {inconsistent_count:,} ({100 - analysis['consistency_percentage']:.1f}%)")
        
        if unified_schema:
            print(f"   Unified schema created with {len(unified_schema)} total fields")
            print(f"\n🔧 Next steps:")
            print(f"   1. Review normalization_template.py and customize defaults")
            print(f"   2. Create schema_normalize_newformat.py using the template")
            print(f"   3. Run normalization on dataset before keyword annotation")
        else:
            print(f"\n💡 To enable parallel processing:")
            print(f"   Run with --save-schemas to create normalization tools")


if __name__ == "__main__":
    main()