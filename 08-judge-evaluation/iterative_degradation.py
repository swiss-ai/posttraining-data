#!/usr/bin/env python3
"""
Iterative completion degradation using Qwen via Swiss AI API.

Takes samples from a dataset and iteratively makes completions worse by:
1. Identifying a dimension to degrade
2. Generating a worse completion
3. Repeating for multiple iterations
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from datasets import load_from_disk
import openai


class SwissAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.swissai.cscs.ch/v1"):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
    
    async def chat_completion(self, messages, model="Qwen/Qwen3-32B", max_tokens=1000):
        """Make a chat completion request."""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                stream=False,
                timeout=120.0
            )
            return response.choices[0].message.content, response.usage
        except Exception as e:
            raise Exception(f"API request failed: {e}")


def extract_original_completion(sample, target_model="Qwen/Qwen3-32B"):
    """Extract the original completion from the target model."""
    for branch in sample.get('conversation_branches', []):
        for message in branch.get('messages', []):
            if message.get('role') == 'assistant' and 'parts' in message:
                for part in message.get('parts', []):
                    if isinstance(part, dict) and 'metadata' in part:
                        model = part['metadata'].get('model')
                        if model == target_model and part.get('type') == 'response':
                            return part.get('content', '')
    return None


async def degrade_completion(client, user_prompt, completions_history, reasonings_history, iteration):
    """Degrade a completion by identifying a dimension and making it worse."""
    
    # Build the full history for context (truncate completions to 500 chars, keep full reasonings)
    history_text = ""
    for i, (completion, reasoning) in enumerate(zip(completions_history, reasonings_history + [None])):
        # Truncate completion to first 500 chars
        truncated_completion = completion[:500] + "..." if len(completion) > 500 else completion
        
        if i == 0:
            history_text += f"ORIGINAL COMPLETION:\n{truncated_completion}\n\n"
        else:
            prev_reasoning = reasonings_history[i-1] if i-1 < len(reasonings_history) else ""
            history_text += f"ITERATION {i} (Previous Reasoning: {prev_reasoning}):\n{truncated_completion}\n\n"
    
    # Ask Qwen to reason about what to change and then generate the worse completion
    degradation_prompt = f"""You are helping create training data by generating an alternative version of an AI response. 

User Prompt: {user_prompt}

Full Degradation History:
{history_text}

Current Latest Completion: {completions_history[-1]}

IMPORTANT: Make the completion objectively worse in quality, not just different in content. Focus on degrading the AI's response quality, not changing the narrative content.

Please respond using EXACTLY this format:

REASONING:
Look at the full degradation history above and identify ONE NEW dimension that hasn't been degraded yet to make the response objectively worse in quality. Choose from these possible modifications: lower factual accuracy (add wrong facts, incorrect dates/numbers), reduce logical coherence (make arguments contradictory or illogical), make it incomplete (remove key parts, leave things unfinished), worsen organization/structure (poor flow, confusing order, bad formatting), make it unfocused on the task (add irrelevant information, go off-topic), reduce language quality (introduce typos, grammatical errors, unclear phrasing), use inappropriate certainty levels (be overconfident about uncertain things or uncertain about facts), ignore format instructions (if specific format was requested), skip/ignore parts of the instructions, add faulty reasoning (use incorrect logic, make wrong assumptions, draw invalid conclusions), or provide wrong/no answers (give incorrect final answers, fail to answer the question, or provide no conclusion at all). Select a NEW dimension that hasn't been used in previous iterations. Explain specifically what NEW dimension you will change. IMPORTANT: The degradation should be SIGNIFICANT and HARD TO MISS, not subtle - make sure the quality drop is obvious and noticeable.

COMPLETION:
CRITICAL: You must preserve ALL previous degradations from the latest completion while adding the new degradation. Do not fix, remove, or undo any of the existing problems - keep all previous typos, errors, inconsistencies, missing parts, etc. from the current latest completion. Only ADD the new degradation on top of the existing issues. The new degradation should be SIGNIFICANT and HARD TO MISS - not a subtle change but an obvious quality problem that clearly makes the response worse. Start with the current latest completion and make it noticeably worse in the new dimension while keeping all existing degradations intact. Generate a completely natural response without any brackets, notes, or annotations indicating what was changed. Make the degradation seamless and natural - do not add parenthetical comments or explanatory notes about the modifications. DO NOT warn the user about any errors, problems, or issues in your response - act as if the degraded response is normal and complete."""
    
    messages = [{"role": "user", "content": degradation_prompt}]
    response, usage = await client.chat_completion(messages, max_tokens=1500)
    
    # Display actual token counts from API
    print(f"  📊 Prompt tokens: {usage.prompt_tokens:,}")
    print(f"  📊 Response tokens: {usage.completion_tokens:,}")
    print(f"  📊 Total tokens: {usage.total_tokens:,}")
    
    # Parse the delimited response
    try:
        if "REASONING:" in response and "COMPLETION:" in response:
            parts = response.split("COMPLETION:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            worse_completion = parts[1].strip()
        else:
            # Fallback if format not followed
            reasoning = "Format not followed correctly"
            worse_completion = response
    except Exception as e:
        # Fallback if parsing fails
        reasoning = f"Parsing failed: {e}"
        worse_completion = response
    
    return worse_completion, reasoning, usage


async def process_sample(client, sample, iterations=10):
    """Process a single sample through iterative degradation."""
    
    # Extract user prompt and original completion
    user_prompt = sample['initial_prompt']['content']
    original_completion = extract_original_completion(sample)
    
    if not original_completion:
        print(f"No original completion found for sample {sample['conversation_id']}")
        return None
    
    # Store the progression
    completions = [original_completion]
    reasonings = []
    
    current_completion = original_completion
    
    print(f"\nProcessing sample: {sample['conversation_id']}")
    print("=" * 80)
    
    # Print user prompt and original completion immediately
    print(f"\n🔵 USER PROMPT:")
    print(f"{'─'*80}")
    print(f"{user_prompt}")
    
    print(f"\n🟢 ORIGINAL COMPLETION:")
    print(f"{'─'*80}")
    print(f"{original_completion}")
    print(f"\n{'='*80}")
    print(f"Starting {iterations} degradation iterations...")
    print(f"{'='*80}")
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}...")
        try:
            worse_completion, reasoning, usage = await degrade_completion(
                client, user_prompt, completions, reasonings, i+1
            )
            completions.append(worse_completion)
            reasonings.append(reasoning)
            current_completion = worse_completion
            
            # Warning for context limits based on actual prompt tokens (32k limit)
            if usage.prompt_tokens > 24000:  # Warning at 24k tokens (75% of 32k limit)
                print(f"  ⚠️  WARNING: {usage.prompt_tokens:,} prompt tokens - approaching 32k token limit!")
            elif usage.prompt_tokens > 16000:  # Caution at 16k tokens (50% of limit)
                print(f"  ⚡ CAUTION: {usage.prompt_tokens:,} prompt tokens - halfway to 32k token limit")
            elif usage.prompt_tokens > 8000:  # Info at 8k tokens (25% of limit)
                print(f"  📈 INFO: {usage.prompt_tokens:,} prompt tokens - 25% of 32k token limit used")
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {e}")
            break
    
    return {
        'conversation_id': sample['conversation_id'],
        'user_prompt': user_prompt,
        'completions': completions,
        'reasonings': reasonings
    }


async def main():
    parser = argparse.ArgumentParser(description="Iteratively degrade completions using Qwen")
    parser.add_argument("dataset_path", help="Path to the dataset directory")
    parser.add_argument("--iterations", "-i", type=int, default=10, 
                       help="Number of degradation iterations (default: 10)")
    parser.add_argument("--samples", "-s", type=int, default=1,
                       help="Number of samples to process (default: 1)")
    parser.add_argument("--index", "-idx", type=int, default=0,
                       help="Index of sample to start from (default: 0)")
    parser.add_argument("--output", "-o", help="Output JSON file to save results")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("SWISSAI_API_KEY")
    if not api_key:
        print("Error: SWISSAI_API_KEY environment variable required")
        return 1
    
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    if hasattr(dataset, 'keys'):
        data = dataset[list(dataset.keys())[0]]
    else:
        data = dataset
    
    print(f"Dataset contains {len(data)} samples")
    print(f"Processing {args.samples} samples starting from index {args.index} with {args.iterations} iterations each")
    
    # Process samples
    results = []
    
    async with SwissAIClient(api_key) as client:
        for i in range(args.samples):
            idx = args.index + i
            if idx >= len(data):
                print(f"Index {idx} exceeds dataset size {len(data)}")
                break
            sample = data[idx]
            result = await process_sample(client, sample, args.iterations)
            if result:
                results.append(result)
                
                # Print the progression
                print(f"\n{'='*100}")
                print(f"RESULTS FOR SAMPLE {i+1}: {result['conversation_id']}")
                print(f"{'='*100}")
                print(f"USER PROMPT:\n{result['user_prompt']}")
                print(f"\n{'='*100}")
                
                for j, completion in enumerate(result['completions']):
                    if j == 0:
                        print(f"\n🟢 ORIGINAL COMPLETION:")
                        print(f"{'─'*80}")
                        print(f"{completion}")
                        print(f"\n{'#'*100}")
                    else:
                        reasoning = result['reasonings'][j-1] if j-1 < len(result['reasonings']) else "No reasoning provided"
                        print(f"\n🔴 ITERATION {j}:")
                        print(f"{'─'*80}")
                        print(f"📝 REASONING:")
                        print(f"   {reasoning}")
                        print(f"\n📄 MODIFIED COMPLETION:")
                        print(f"{'─'*80}")
                        print(f"{completion}")
                        print(f"\n{'#'*100}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    
    print(f"\nProcessed {len(results)} samples successfully")


if __name__ == "__main__":
    asyncio.run(main())