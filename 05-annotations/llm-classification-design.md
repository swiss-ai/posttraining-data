# LLM-Based Classification Implementation Plan

## Overview
This document outlines the implementation plan for LLM-based classification features in the 04-annotations folder, starting with refusal detection. The implementation will be simplified compared to the old codebase while maintaining core functionality and integrating with the existing processing pipeline.

## Key Differences from Old Codebase

### 1. Simplified API Access
- **Old**: Complex multi-node discovery and load balancing across Swiss AI compute nodes
- **New**: Direct API calls to `https://api.swissai.cscs.ch/v1` using standard OpenAI client
- **Rationale**: Reduces complexity, Swiss AI API already handles load balancing

### 2. Modular Architecture  
- **Old**: Monolithic `chat_format_classifier.py` base class with complex inheritance
- **New**: Lightweight shared library for common LLM operations, simple classifier scripts
- **Rationale**: Easier to understand, maintain, and extend

### 3. Processing Pipeline Integration
- **Follows existing pattern**: Load source dataset → Apply processing → Save to new output directory
- **Metadata continuity**: Append to existing `processing_log` in `dataset_metadata.json`
- **Consistent structure**: Match the patterns used in 02-standardisation and 04-annotations

## Implementation Structure

### Directory Layout
```
04-annotations/
├── prompts/                    # Prompt templates
│   ├── refusal.txt
│   ├── assistant.txt
│   ├── ideological.txt
│   └── quality.txt
├── lib/                       # Shared libraries
│   └── llm_classifier.py      # Core LLM classification logic
├── classify_refusal.py        # Refusal classification script
├── classify_assistant.py      # Assistant classification (future)
├── classify_ideological.py    # Ideological classification (future)
└── classify_quality.py        # Quality classification (future)
```

## Core Components

### 1. LLM Classifier Library (`lib/llm_classifier.py`)

#### Key Features:
- Simple OpenAI client wrapper for Swiss AI API
- Batch processing with configurable concurrency
- Token usage tracking
- Error handling and retries
- JSON response parsing with validation

#### Main Classes:
```python
class LLMClassifier:
    """Base class for LLM-based classification"""
    
    def __init__(self, api_key: str, model: str = "meta-llama/Llama-3.3-70B-Instruct"):
        self.client = openai.Client(
            api_key=api_key,
            base_url="https://api.swissai.cscs.ch/v1"
        )
        self.model = model
    
    async def classify_single(self, prompt: str, valid_categories: List[str]) -> Dict:
        """Classify a single item"""
        
    async def classify_batch(self, items: List[Dict], prompt_template: str, 
                           valid_categories: List[str], max_concurrent: int = 10) -> List[Dict]:
        """Classify multiple items concurrently"""
```

### 2. Refusal Classifier (`classify_refusal.py`)

#### Command Line Interface:
```bash
python classify_refusal.py \
    data/02-standardised/tulu-3-sft-mixture \
    --output data/05-annotations/tulu-3-sft-mixture-refusal \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --max-concurrent 20 \
    --chunk-size 10000
```

#### Key Design Decisions:
1. **Message Selection**: Only classify assistant messages (refusals only occur in responses)
2. **Context Handling**: Include previous user message as context for better accuracy
3. **Metadata Structure**: Store under `message.metadata.refusal_classification`
4. **Error Handling**: Mark failed classifications as "inconclusive" with error details

### 3. Processing Log Integration

#### Metadata Structure Following Existing Pattern:
```json
{
  "dataset_name": "source_dataset_name",
  "processing_log": [
    {
      "operation": "refusal_classification",
      "script": "classify_refusal.py", 
      "timestamp": "2025-07-26T...",
      "input_path": "data/02-standardised/tulu-3-sft-mixture",
      "output_path": "data/05-annotations/tulu-3-sft-mixture-refusal",
      "model": "meta-llama/Llama-3.3-70B-Instruct",
      "api_provider": "swiss_ai",
      "messages_classified": 15234,
      "classification_success": true,
      "messages_failed": 12,
      "total_tokens_used": 2456789
    }
  ]
}
```

## Implementation Details

### API Integration
```python
# Swiss AI API setup
client = openai.Client(
    api_key=os.getenv("SWISSAI_API"),
    base_url="https://api.swissai.cscs.ch/v1"
)

# Make classification request
response = await client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": formatted_prompt}
    ],
    temperature=0.1,  # Low temperature for consistency
    response_format={"type": "json_object"}  # Ensure JSON response
)
```

### Processing Workflow Following Existing Pattern

1. **Load Source Dataset**
   ```python
   dataset = load_from_disk(input_path)
   ```

2. **Load Existing Metadata**
   ```python
   metadata_file = Path(input_path) / "dataset_metadata.json"
   if metadata_file.exists():
       with open(metadata_file, 'r') as f:
           original_metadata = json.load(f)
   ```

3. **Process in Chunks**
   ```python
   annotated_samples = []
   for chunk in chunked_data:
       classified_chunk = await classify_chunk(chunk, model, prompt_template)
       annotated_samples.extend(classified_chunk)
   ```

4. **Create Output Dataset**
   ```python
   output_dataset = Dataset.from_list(annotated_samples)
   output_dataset.save_to_disk(output_path)
   ```

5. **Update Metadata with Processing Log**
   ```python
   updated_metadata = {
       **original_metadata,
       "processing_log": original_metadata.get("processing_log", []) + [{
           "operation": "refusal_classification",
           "script": "classify_refusal.py",
           "timestamp": datetime.now().isoformat(),
           # ... other fields
       }]
   }
   
   with open(output_path / "dataset_metadata.json", 'w') as f:
       json.dump(updated_metadata, f, indent=2)
   ```

## Classification Types

### 1. Refusal Classification
- **Target**: Assistant messages only
- **Categories**: refusal, no_refusal  
- **Context**: Previous user message
- **Metadata Path**: `message.metadata.refusal_classification`

### 2. Assistant Classification (Future)
- **Target**: Assistant messages only
- **Categories**: ai_assistant_related, non_ai_assistant
- **Context**: Full conversation
- **Metadata Path**: `message.metadata.assistant_classification`

### 3. Ideological Sensitivity (Future) 
- **Target**: User and system messages
- **Categories**: 0 (none), 1 (mild), 2 (moderate), 3 (high)
- **Context**: Message content only
- **Metadata Path**: `message.metadata.ideological_classification`

### 4. Quality Classification (Future)
- **Target**: User and system messages
- **Dimensions**: well_formedness, clarity_of_intent, answerable_scope, completeness
- **Scoring**: 1-3 per dimension, overall average
- **Metadata Path**: `message.metadata.quality_classification`

## Performance Considerations

### Concurrency Tuning
- Start with conservative limit (10-20 concurrent requests)
- Monitor rate limit responses
- Adjust based on API feedback

### Memory Management
- Process dataset in chunks (default: 10,000 samples)
- Stream data rather than loading all at once  
- Clear processed data from memory
- Use async/await for efficient I/O

## Error Handling Strategy

### API Errors
- Exponential backoff with max retries (3 attempts)
- Reduce concurrency on rate limit errors
- Mark failed messages with error details in metadata

### Classification Failures
```python
# Failed classification metadata structure
{
    "classification": "inconclusive",
    "reasoning": "API error: Rate limit exceeded",
    "success": false,
    "error": "RateLimitError: ...",
    "timestamp": "2025-07-26T..."
}
```

## Migration from Old Codebase

### What to Reuse:
1. **Prompt templates**: Can be adapted with minor formatting changes
2. **Classification categories**: Keep the same category names
3. **Message selection logic**: Same role-based filtering rules

### What to Simplify:
1. **No node discovery**: Use direct API calls
2. **No adaptive batching**: Simple concurrent processing
3. **No complex progress tracking**: Basic progress bars
4. **No processing conflicts**: Simple overwrite behavior

## Future Enhancements

1. **Multi-model Support**
   - Allow different models for different classification types
   - Support for future model updates

2. **Resume Capability**
   - Track partially processed datasets
   - Resume from interruption points

3. **Batch Optimization**
   - Group similar-length prompts
   - Optimize for token efficiency

## Summary

This implementation provides a clean, maintainable approach to LLM-based classification that:

- **Integrates seamlessly** with existing processing pipeline (02-standardisation → 03-filtering → 04-annotations)
- **Follows established patterns** for metadata and processing logs
- **Uses only Swiss AI API** backend with simple OpenAI client
- **Creates new dataset versions** for each classification run  
- **Maintains processing history** in consistent metadata format
- **Simplifies complexity** while preserving core functionality

The key insight is to leverage the existing data pipeline patterns while simplifying the LLM integration to focus on core classification functionality.