#!/usr/bin/env python3
"""
Minimal example showing how to extract answer logprobs when model returns thinking tokens first.

The issue: Models that think before answering return <think>...</think> tokens before the actual answer,
which means you need to skip past the thinking phase to find the real answer and its logprobs.
"""

import openai
import os
import numpy as np

# Initialize client
client = openai.Client(
    api_key=os.getenv("SWISSAI_API_KEY"),
    #base_url="https://api.swissai.cscs.ch/v1"
    base_url="http://148.187.108.173:8092/v1/service/llm/v1/"
)

# Make API call with logprobs enabled
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{
        "role": "user",
        "content": "What is the capital of Switzerland?\nA) Geneva\nB) Bern\nC) Zurich\nD) Basel\n\nAnswer with only the letter:"
    }],
    logprobs=True,
    top_logprobs=10,
    max_tokens=1500  # Must be large enough to get past thinking phase
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
        print(f"Answer: {token}")
        print(f"Confidence: {np.exp(token_data.logprob)*100:.1f}%")
        
        # Get probabilities for all choices at this position
        choices = {'A': None, 'B': None, 'C': None, 'D': None}
        choices[token] = token_data.logprob
        
        if token_data.top_logprobs:
            for alt in token_data.top_logprobs:
                if alt.token in choices:
                    choices[alt.token] = alt.logprob
        
        print("\nAll choice probabilities:")
        for letter, logprob in sorted(choices.items()):
            if logprob is not None:
                print(f"  {letter}: {np.exp(logprob)*100:6.2f}%")
        break