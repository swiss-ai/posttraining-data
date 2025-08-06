# Dataset Standardisation

Converts datasets to the unified chat format for downstream processing.

## Old Chat Format

All datasets are standardized to this unified chat format:

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

## New Chat Format

If a dataset has augmentations to the assistant, the assistant content is not a str but a list. 

```json
{
  "conversation_id": "unique_identifier",
  "dataset_source": "source_dataset_name",
  "original_metadata": {},
  "system_prompt": {
    "content": "you are a friendly assistant. Here is a list of tools you have available ... ",
    "metadata": {}
  },
  "initial_prompt": {
    "role": "user",
    "content": "hey this my example prompt. Can you tell me the weather?",
    "metadata": {}
  },
  "available_functions": [
    {"name": "get_weather_data", "description": "Retrieves weather information and takes two args ..."}
    {"name": "another_tool", "description": "this tool ..."}
    {"name": "python_exec", "description": "this tool takes as an argument python code, executes it, and returns the output"}
  ],
  "conversation_branches": [
    {
      "messages": [
        {
          "role": "user",
          "parts": [
            {
              "type": "response",
              "content": "Hey, can you tell me the weather in Bern?",
              "metadata": {}
            }
          ],
        },
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
                "argname1": "34FR343FLQMP4I946K",
                "argname2": "Bern"
              }
            },
            {
              "type": "function-output",
              "content": "{location: Bern, Temperature: 24, Weather: sunny}",
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

# Or specify exact output name
venv/bin/python 02-standardisation/convert_xlam_function_calling.py \
  /users/schlag/store/posttrain_data/01_raw_hf_data/xlam-function-calling-60k \
  --output data/02-standardised/xlam-function-calling-60k
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