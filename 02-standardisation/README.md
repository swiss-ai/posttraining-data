# Dataset Standardisation

Converts datasets to the unified chat format for downstream processing. The project is transitioning to the new chat format with parts structure as the standard.

## Chat Format Schema

### Legacy Format (Being Phased Out)

The original chat format uses string content for messages:

```json
{
  "conversation_id": "unique_identifier",
  "dataset_source": "source_dataset_name",
  "original_metadata": {},
  "system_prompt": {
    "content": "system message",
    "metadata": {}
  },
  "initial_prompt": {
    "role": "user",
    "content": "user message",
    "metadata": {}
  },
  "conversation_branches": [
    {
      "messages": [
        {
          "role": "assistant",
          "content": "response",
          "metadata": {}
        }
      ]
    }
  ],
  "created_timestamp": "ISO datetime"
}
```

### New Chat Format (Standard)

The new standard format uses parts structure for all messages, enabling support for tool use, thinking, and verifiable responses: 

```json
{
  "conversation_id": "unique_identifier",
  "dataset_source": "source_dataset_name",
  "original_metadata": {},
  "system_prompt": {
    "content": "You are a friendly assistant with access to some functions.",
    "metadata": {}
  },
  "initial_prompt": {
    "role": "user",
    "content": "Hey, can you tell me the weather in Bern?",
    "metadata": {}
  },
  "available_functions": [
    {
      "name": "get_weather_data",
      "description": "Retrieves weather information for a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
          }
        },
        "required": ["location"]
      }
    }
  ],
  "conversation_branches": [
    {
      "messages": [
        {
          "role": "assistant",
          "parts": [
            {
              "type": "thought",
              "content": "The user wants to know the weather. I have a tool ... Let me ... ",
            },
            {
              "type": "function-call",
              "name": "get_weather_data",
              "args": {
                "location": "Bern, Switzerland",
                "unit": "celsius"
              }
            },
            {
              "type": "function-output",
              "content": "{\"location\": \"Bern, Switzerland\", \"temperature\": 24, \"weather\": \"sunny\", \"unit\": \"celsius\"}"
            },
            {
              "type": "response",
              "content": "Hey sure. The weather will be sunny.",
              "metadata": {}
            },
            {
              "type": "verifiable-responses",
              "answers": ["sunny"],
            }
          ],
        },
        {
          "role": "user",
          "parts": [
            {
              "type": "response",
              "content": "thanks! Is this warmer than last year? ",
              "metadata": {}
            }
          ],
        }
      ]
    }
  ],
  "created_timestamp": "ISO datetime"
}
```

### Parts Schema and Compatibility

Each part in the parts array has a unified schema to ensure Arrow compatibility across all datasets:

```json
{
  "type": "response|thought|function-call|function-output|verifiable-responses",
  "content": "text content (empty string if not used)",
  "metadata": {},
  "name": "function_name (empty string if not used)", 
  "args": "function_args (empty string if not used)"
}
```

**Schema Harmonization**: All parts include all fields with consistent string typing to prevent Arrow schema incompatibility when merging datasets. Different part types use different field combinations:

- **response**: Uses `type`, `content`, `metadata`
- **thought**: Uses `type`, `content` 
- **function-call**: Uses `type`, `name`, `args`
- **function-output**: Uses `type`, `content`
- **verifiable-responses**: Uses `type`, special format with answers array

**Migration Tools**: Use `07-dataset-aggregation/convert_old_to_new_format.py` to convert legacy format datasets to the new standard. The aggregation script `dataset-aggregation-newformat.py` automatically harmonizes schemas when merging datasets.

### OpenAI Function Format

The `available_functions` list follows the OpenAI API function calling specification:

```json
{
  "name": "function_name",
  "description": "Function description",
  "parameters": {
    "type": "object",
    "properties": {
      "parameter_name": {
        "type": "string|integer|number|boolean|array|object",
        "description": "Parameter description",
        "enum": ["option1", "option2"],
        "default": "default_value"
      }
    },
    "required": ["required_parameter_names"]
  }
}
```

**Note**: Some existing datasets may have `parameters.properties` as a JSON string instead of an object due to Arrow schema constraints. This should be parsed as JSON when used.

## Available Converters

### Standard Chat Format Converter
`convert_to_chat_format.py` - Handles most conversational datasets including chat messages, ShareGPT, instruction-response, and preference formats.

### XLAM Function Calling Converter  
`convert_xlam_function_calling.py` - Specialized converter for xlam-function-calling-60k dataset that:
- Converts XLAM's custom parameter format to OpenAI-compatible JSON Schema
- Transforms single-turn function calling examples to the new chat format
- Handles complex parameter types (`List[...]`, `Dict[...]`) properly
- Preserves all tool definitions and function call arguments
- Generates 60,000 examples of function calling training data

### APIGen Function Calling Converter
`convert_apigen_function_calling.py` - Specialized converter for APIGen-MT-5k dataset that:
- Converts multi-turn conversations with embedded function calling workflows
- Preserves conversation structure with proper user/assistant message alternation
- Maps function_call/observation pairs to function-call/function-output parts
- Handles think function calls as thought parts for reasoning steps
- Processes 5,000 examples of multi-turn function calling conversations
- Tools already in OpenAI-compatible format (no parameter conversion needed)

### OLMO Prompts-Only Converter (New Format)
`convert_olmo_prompts_only_newformat.py` - Extracts prompts from OLMO-2-0325-32b-preference-mix dataset:
- Extracts only the user prompts from preference data (discards chosen/rejected responses)
- Outputs in new chat format with empty conversation branches (prompts only)
- Selectively preserves metadata (keeps `id` and `source`, excludes model names)
- Generates 377,807 prompt-only examples for prompt diversity
- Useful for prompt augmentation and diversity in training mixtures

## Usage

### Conversion Commands
```bash
# Standard datasets (run from project root)
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/tulu-3-sft-mixture data/02-standardised/
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/smoltalk data/02-standardised/
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/smoltalk2 data/02-standardised/
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/The-Tome data/02-standardised/
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/EuroBlocks-SFT-Synthetic-1124 data/02-standardised/
python 02-standardisation/convert_to_chat_format.py data/01-hf-data/Llama-Nemotron-Post-Training-Dataset data/02-standardised/

# Function calling datasets (new format with OpenAI-compatible functions)
venv/bin/python 02-standardisation/convert_xlam_function_calling.py \
  /users/schlag/store/posttrain_data/01_raw_hf_data/xlam-function-calling-60k \
  --output data/02-standardised/

# APIGen multi-turn function calling dataset
venv/bin/python 02-standardisation/convert_apigen_function_calling.py \
  /capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/APIGen-MT-5k \
  --output data/02-standardised/apigen-mt-5k

# Or specify exact output name
venv/bin/python 02-standardisation/convert_xlam_function_calling.py \
  /users/schlag/store/posttrain_data/01_raw_hf_data/xlam-function-calling-60k \
  --output data/02-standardised/xlam-function-calling-60k

# OLMO prompts-only extraction (new format)
venv/bin/python 02-standardisation/convert_olmo_prompts_only_newformat.py \
  /capstor/store/cscs/swissai/infra01/posttrain_data/01_raw_hf_data/olmo-2-0325-32b-preference-mix \
  --output data/02-standardisation_newformat/
```

### Browse Converted Datasets

Use the interactive browser to inspect converted datasets:

```bash
# Old format datasets (string content)
python 02-standardisation/browse_sample.py data/02-standardised/tulu-3-sft-mixture

# New format datasets (parts structure)
python 02-standardisation/browse_sample_new.py data/converted/dataset

# Start at specific sample
python 02-standardisation/browse_sample.py data/02-standardised/smoltalk --start-idx 100

# Show multiple samples at once
python 02-standardisation/browse_sample.py data/02-standardised/The-Tome --num-samples 5

# Raw JSON output
python 02-standardisation/browse_sample.py data/02-standardised/tulu-3-sft-mixture --raw-json
```
