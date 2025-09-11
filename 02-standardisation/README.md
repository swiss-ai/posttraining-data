# Dataset Standardisation

Converts datasets to the unified schema for further processing.

## Data Schema

The new format uses parts structure for all messages, enabling support for tool use, thinking, and verifiable responses.

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

## Usage

Every conversion script takes an input source creates an new copy. 

### Conversion Commands
```bash
python 02-standardisation/convert_to_chat.py data/01-hf-data/tulu-3-sft-mixture data/02-standardised/
python 02-standardisation/convert_to_chat.py data/01-hf-data/smoltalk data/02-standardised/
python 02-standardisation/convert_to_chat.py data/01-hf-data/The-Tome data/02-standardised/
python 02-standardisation/convert_to_chat.py data/01-hf-data/Llama-Nemotron-Post-Training-Dataset data/02-standardised/
```

### Browse Converted Datasets

Use the interactive browser to inspect converted datasets:

```bash
# Start at specific sample
python 02-standardisation/browse_sample.py data/02-standardised/smoltalk --start-idx 100

# Show multiple samples at once
python 02-standardisation/browse_sample.py data/02-standardised/The-Tome --num-samples 5

# Raw JSON output
python 02-standardisation/browse_sample.py data/02-standardised/tulu-3-sft-mixture --raw-json
```
