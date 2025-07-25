# Dataset Standardisation

Converts datasets to the unified chat format for downstream processing.

## Chat Format Schema

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
```

### Browse Converted Datasets

Use the interactive browser to inspect converted datasets:

```bash
# Interactive browsing (default, run from project root)
python 02-standardisation/browse_sample.py data/02-standardised/tulu-3-sft-mixture

# Start at specific sample
python 02-standardisation/browse_sample.py data/02-standardised/smoltalk --start-idx 100

# Show multiple samples at once
python 02-standardisation/browse_sample.py data/02-standardised/The-Tome --num-samples 5

# Raw JSON output
python 02-standardisation/browse_sample.py data/02-standardised/tulu-3-sft-mixture --raw-json
```