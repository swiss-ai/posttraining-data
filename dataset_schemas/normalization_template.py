#!/usr/bin/env python3
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
    if "" not in normalized:
        normalized[""] = get_default_value_for_type("dict")
    if "aggregation_input_data_path" not in normalized:
        normalized["aggregation_input_data_path"] = get_default_value_for_type("string")
    if "available_functions" not in normalized:
        normalized["available_functions"] = get_default_value_for_type("empty_list")
    if "conversation_branches" not in normalized:
        normalized["conversation_branches"] = get_default_value_for_type("list")
    if "conversation_branches[0]" not in normalized:
        normalized["conversation_branches[0]"] = get_default_value_for_type("dict")
    if "conversation_id" not in normalized:
        normalized["conversation_id"] = get_default_value_for_type("string")
    if "created_timestamp" not in normalized:
        normalized["created_timestamp"] = get_default_value_for_type("string")
    if "dataset_source" not in normalized:
        normalized["dataset_source"] = get_default_value_for_type("string")
    if "initial_prompt" not in normalized:
        normalized["initial_prompt"] = get_default_value_for_type("dict")
    if "original_metadata" not in normalized:
        normalized["original_metadata"] = get_default_value_for_type("dict")
    if "system_prompt" not in normalized:
        normalized["system_prompt"] = get_default_value_for_type("dict")
    
    # Normalize fields in conversation_branches[0]
    # TODO: Add normalization for conversation_branches[0] fields
    
    # Normalize fields in conversation_branches[0].messages[0]
    # TODO: Add normalization for conversation_branches[0].messages[0] fields
    
    # Normalize fields in conversation_branches[0].messages[0].parts[0]
    # TODO: Add normalization for conversation_branches[0].messages[0].parts[0] fields
    
    # Normalize fields in initial_prompt
    # TODO: Add normalization for initial_prompt fields
    
    # Normalize fields in initial_prompt.metadata
    # TODO: Add normalization for initial_prompt.metadata fields
    
    # Normalize fields in original_metadata
    # TODO: Add normalization for original_metadata fields
    
    # Normalize fields in system_prompt
    # TODO: Add normalization for system_prompt fields

    
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
