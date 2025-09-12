#!/usr/bin/env python3
"""
Async version of minimal_thinking_logprobs.py that processes 10 requests in parallel.

Shows how to extract answer logprobs when model returns thinking tokens first,
demonstrating model consistency and performance across multiple parallel requests.
"""

import asyncio
import openai
import os
import numpy as np
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional

# curl -s 148.187.108.172:8092/v1/dnt/table | python3 -m json.tool

async def process_single_request(client: openai.AsyncOpenAI, messages: List[Dict], request_id: int) -> Dict:
    """Process a single async request and extract answer token data."""
    try:
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-32B",  # Qwen/Qwen3-32B,  meta-llama/Llama-3.3-70B-Instruct
            messages=messages,
            logprobs=True,
            top_logprobs=20,
            temperature=0.1,
            max_tokens=10,
        )
        
        # Extract tokens from response
        tokens = response.choices[0].logprobs.content
        
        # Find answer token (skip thinking phase)
        in_thinking = False
        for i, token_data in enumerate(tokens):
            token = token_data.token.strip()
            # Track thinking phase
            if token == '<think>':
                in_thinking = True
            elif token == '</think>':
                in_thinking = False
            # Find answer after thinking ends
            elif not in_thinking and token in ['A', 'B', 'C', 'D']:
                # Get probabilities for all choices at this position
                choices = {'A': None, 'B': None, 'C': None, 'D': None}
                choices[token] = token_data.logprob
                
                if token_data.top_logprobs:
                    for alt in token_data.top_logprobs:
                        if alt.token in choices:
                            choices[alt.token] = alt.logprob
                
                return {
                    'request_id': request_id,
                    'answer': token,
                    'confidence': np.exp(token_data.logprob),
                    'logprob': token_data.logprob,
                    'all_choices': choices,
                    'response_content': response.choices[0].message.content,
                    'success': True
                }
        
        # No answer found
        return {
            'request_id': request_id,
            'answer': None,
            'confidence': 0.0,
            'logprob': None,
            'all_choices': {},
            'response_content': response.choices[0].message.content,
            'success': False,
            'error': 'No answer token found'
        }
        
    except Exception as e:
        return {
            'request_id': request_id,
            'answer': None,
            'confidence': 0.0,
            'logprob': None,
            'all_choices': {},
            'response_content': None,
            'success': False,
            'error': str(e)
        }

def analyze_aggregated_results(results: List[Dict]) -> Dict:
    """Analyze results across all parallel requests."""
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        return {'error': 'No successful requests'}
    
    # Count answer frequencies
    answer_counts = Counter(r['answer'] for r in successful_results)
    
    # Calculate average confidence per choice
    choice_confidences = defaultdict(list)
    choice_logprobs = defaultdict(list)
    
    for result in successful_results:
        for choice, logprob in result['all_choices'].items():
            if logprob is not None:
                choice_confidences[choice].append(np.exp(logprob))
                choice_logprobs[choice].append(logprob)
    
    avg_confidences = {
        choice: np.mean(confs) if confs else 0.0 
        for choice, confs in choice_confidences.items()
    }
    
    avg_logprobs = {
        choice: np.mean(logprobs) if logprobs else None
        for choice, logprobs in choice_logprobs.items()
    }
    
    return {
        'total_requests': len(results),
        'successful_requests': len(successful_results),
        'failed_requests': len(results) - len(successful_results),
        'answer_distribution': dict(answer_counts),
        'avg_confidences': avg_confidences,
        'avg_logprobs': avg_logprobs,
        'consensus_answer': answer_counts.most_common(1)[0][0] if answer_counts else None,
        'consensus_strength': answer_counts.most_common(1)[0][1] / len(successful_results) if answer_counts else 0.0
    }

async def main():
    """Main async function to process 10 parallel requests."""
    # Initialize async client
    client = openai.AsyncOpenAI(
        api_key=os.getenv("SWISSAI_API_KEY"),
        #base_url="https://api.swissai.cscs.ch/v1"
        base_url="http://148.187.108.172:8092/v1/service/llm/v1/"
    )
    
    # Define the messages
    messages = [{
        "role": "user",
        "content": "What is the capital of Switzerland?\nA) Geneva\nB) Bern\nC) Zurich\nD) Basel\n\nDon't think or explain. Answer with only the letter."
    }]
    
    print("=" * 80)
    print("ASYNC PARALLEL THINKING LOGPROBS ANALYSIS")
    print("=" * 80)
    print("Processing 10 parallel requests...")
    print()
    
    # Start timing
    start_time = time.time()
    
    # Launch 10 parallel requests
    tasks = [
        process_single_request(client, messages, i) 
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    print(f"Completed 10 requests in {total_time:.2f} seconds")
    print()
    
    # Analyze aggregated results
    analysis = analyze_aggregated_results(results)
    
    if 'error' in analysis:
        print(f"Analysis failed: {analysis['error']}")
        return
    
    # Print individual results
    print("=" * 80)
    print("INDIVIDUAL RESULTS:")
    print("=" * 80)
    for result in results:
        if result['success']:
            print(f"Request {result['request_id']:2d}: Answer {result['answer']} "
                  f"(confidence: {result['confidence']*100:5.1f}%, "
                  f"logprob: {result['logprob']:7.4f})")
        else:
            print(f"Request {result['request_id']:2d}: FAILED - {result['error']}")
    
    print()
    
    # Print aggregated analysis
    print("=" * 80)
    print("AGGREGATED ANALYSIS:")
    print("=" * 80)
    print(f"Successful requests: {analysis['successful_requests']}/{analysis['total_requests']}")
    print(f"Consensus answer: {analysis['consensus_answer']} "
          f"({analysis['consensus_strength']*100:.1f}% agreement)")
    print()
    
    print("ANSWER DISTRIBUTION:")
    print("-" * 30)
    for choice in ['A', 'B', 'C', 'D']:
        count = analysis['answer_distribution'].get(choice, 0)
        percentage = (count / analysis['successful_requests'] * 100) if analysis['successful_requests'] > 0 else 0
        print(f"  {choice}: {count:2d} votes ({percentage:5.1f}%)")
    
    print()
    print("AVERAGE CONFIDENCE PER CHOICE:")
    print("-" * 35)
    for choice in ['A', 'B', 'C', 'D']:
        avg_conf = analysis['avg_confidences'].get(choice, 0.0)
        avg_logprob = analysis['avg_logprobs'].get(choice)
        if avg_logprob is not None:
            print(f"  {choice}: {avg_conf*100:6.2f}% (avg logprob: {avg_logprob:7.4f})")
        else:
            print(f"  {choice}: No data available")
    
    print()
    print("PERFORMANCE STATS:")
    print("-" * 20)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Theoretical sequential time: ~{total_time * 10:.1f} seconds")
    print(f"Speedup: ~{10:.0f}x faster than sequential")

if __name__ == "__main__":
    asyncio.run(main())