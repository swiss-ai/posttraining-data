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
    base_url="http://148.187.108.172:8092/v1/service/llm/v1/"
)

# Make API call with logprobs enabled
messages = [{
    "role": "user",
    "content": "What is the capital of Switzerland?\nA) Geneva\nB) Bern\nC) Zurich\nD) Basel\n\nDon't think or explain. Answer with only the letter."
}]

response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=messages,
    logprobs=True,
    top_logprobs=20,
    temperature=0.1,
    max_tokens=10,
)

# Print entire prompt and response
print("=" * 80)
print("FULL PROMPT:")
print("=" * 80)
for msg in messages:
    print(f"Role: {msg['role']}")
    print(f"Content: {msg['content']}")
print("x" * 80)

print("=" * 80)
print("FULL RESPONSE:")
print("=" * 80)
print(response.choices[0].message.content)
print("x" * 80)

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
        print("=" * 80)
        print("ANSWER TOKEN ANALYSIS:")
        print("=" * 80)
        print(f"Answer: {token}")
        print(f"Primary Token Confidence: {np.exp(token_data.logprob)*100:.1f}%")
        print(f"Primary Token Logprob: {token_data.logprob:.6f}")
        print()
        
        print("ALL TOKENS AND LOGPROBS AT THIS POSITION:")
        print("-" * 50)
        print(f"Selected Token: '{token_data.token}' (logprob: {token_data.logprob:.6f}, prob: {np.exp(token_data.logprob)*100:.2f}%)")
        
        if token_data.top_logprobs:
            print("\nTop Alternative Tokens:")
            for i, alt in enumerate(token_data.top_logprobs, 1):
                print(f"  {i:2d}. '{alt.token}' (logprob: {alt.logprob:.6f}, prob: {np.exp(alt.logprob)*100:.2f}%)")
        
        print()
        print("MULTIPLE CHOICE ANALYSIS:")
        print("-" * 30)
        # Get probabilities for all choices at this position
        choices = {'A': None, 'B': None, 'C': None, 'D': None}
        choices[token] = token_data.logprob
        
        if token_data.top_logprobs:
            for alt in token_data.top_logprobs:
                if alt.token in choices:
                    choices[alt.token] = alt.logprob
        
        for letter, logprob in sorted(choices.items()):
            if logprob is not None:
                print(f"  {letter}: {np.exp(logprob)*100:6.2f}% (logprob: {logprob:.6f})")
            else:
                print(f"  {letter}: Not in top logprobs")
        break