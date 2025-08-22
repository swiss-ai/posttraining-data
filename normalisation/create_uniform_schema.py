#!/usr/bin/env python3
import json
from collections import defaultdict

def merge_schemas(schemas):
    """Merge multiple schemas into one uniform schema, replacing NoneType appropriately"""
    
    def merge_value(existing, new_val):
        """Merge two schema values, preferring non-None types"""
        if existing == "NoneType" and new_val != "NoneType":
            return new_val
        elif existing != "NoneType" and new_val == "NoneType":
            return existing
        elif existing == new_val:
            return existing
        else:
            # Both are non-None but different - keep first seen
            return existing
    
    def merge_dict_schemas(existing, new_dict):
        """Recursively merge dictionary schemas"""
        if not isinstance(existing, dict):
            existing = {}
        
        for key, value in new_dict.items():
            if key not in existing:
                existing[key] = value
            elif isinstance(value, dict) and isinstance(existing[key], dict):
                existing[key] = merge_dict_schemas(existing[key], value)
            elif isinstance(value, list) and isinstance(existing[key], list):
                # Merge list schemas (take first element as template)
                if value and not existing[key]:
                    existing[key] = value
                elif value and existing[key]:
                    if isinstance(value[0], dict) and isinstance(existing[key][0], dict):
                        existing[key] = [merge_dict_schemas(existing[key][0], value[0])]
                    else:
                        existing[key] = [merge_value(existing[key][0], value[0])]
            else:
                existing[key] = merge_value(existing[key], value)
        
        return existing
    
    # Start with empty schema
    unified = {}
    
    # Merge all schemas
    for schema_str in schemas:
        schema = json.loads(schema_str)
        unified = merge_dict_schemas(unified, schema)
    
    return unified

def replace_none_types(schema):
    """Replace ALL NoneType values with appropriate empty values - NO None values should remain"""
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if value == "NoneType" or value is None:
                # Determine appropriate empty value based on key context
                if 'metadata' in key.lower() or key in ['original_metadata']:
                    result[key] = {}
                elif key in ['answers', 'available_functions'] or 'list' in str(key).lower():
                    result[key] = []
                else:
                    result[key] = ""
            elif isinstance(value, dict):
                result[key] = replace_none_types(value)
            elif isinstance(value, list):
                result[key] = [replace_none_types(item) for item in value]
            else:
                result[key] = value
        return result
    elif isinstance(schema, list):
        return [replace_none_types(item) for item in schema]
    elif schema == "NoneType" or schema is None:
        return ""  # Default to empty string for any remaining None
    else:
        return schema

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create uniform schema from collection of unique schemas")
    parser.add_argument("input_schemas", help="Path to unique schemas JSON file")
    parser.add_argument("output_uniform_schema", help="Path to save uniform schema JSON file")
    args = parser.parse_args()
    
    from pathlib import Path
    input_path = Path(args.input_schemas)
    output_path = Path(args.output_uniform_schema)
    
    # Load unique schemas
    print(f"Loading unique schemas from: {input_path}")
    with open(str(input_path), "r") as f:
        schema_data = json.load(f)
    
    print(f"Processing {len(schema_data)} unique schemas...")
    
    # Extract schema strings
    schemas = list(schema_data.values())
    
    # Merge schemas
    unified_schema = merge_schemas(schemas)
    
    # Replace NoneType with appropriate empty values
    final_schema = replace_none_types(unified_schema)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save result
    with open(str(output_path), "w") as f:
        json.dump(final_schema, f, indent=2)
    
    # Verify no None values remain
    schema_str = json.dumps(final_schema)
    none_count = schema_str.count('"NoneType"') + schema_str.count('null')
    
    print(f"Uniform schema created: {output_path}")
    print(f"Schema has {len(final_schema)} top-level fields")
    print(f"Verification: {none_count} None values remaining (should be 0)")
    print("\nUniform Schema:")
    print(json.dumps(final_schema, indent=2))

if __name__ == "__main__":
    main()