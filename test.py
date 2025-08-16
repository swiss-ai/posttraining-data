import openai
import numpy as np
import os

client = openai.Client(
    api_key=os.getenv("SWISSAI_API_KEY"),
    base_url="https://api.swissai.cscs.ch/v1"
)

# First try without logprobs to see raw response
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-32B",
        messages=[
            {
                "content": """What is the capital of Switzerland?
A) Geneva
B) Bern
C) Zurich
D) Basel

Answer with only the letter (A, B, C, or D):""",
                "role": "user",
            }
        ],
        stream=False,
        logprobs=False,  # Try without logprobs first
        max_tokens=1500  # Need more to get past full thinking phase
    )
    print("RAW RESPONSE:")
    print(f"Content: {response.choices[0].message.content}")
    if hasattr(response.choices[0].message, 'reasoning_content'):
        print(f"Reasoning: {response.choices[0].message.reasoning_content}")
    print("="*50)
except Exception as e:
    print(f"Error without logprobs: {e}")
    
# Now try with logprobs
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[
        {
            "content": """What is the capital of Switzerland?
A) Geneva
B) Bern
C) Zurich
D) Basel

Answer with only the letter (A, B, C, or D):""",
            "role": "user",
        }
    ],
    stream=False,
    logprobs=True,
    top_logprobs=10,
    max_tokens=1500  # Need more to get past full thinking phase
)

print("=" * 50)
print("MULTIPLE CHOICE LOGPROBS")
print("=" * 50)

if response.choices[0].logprobs and response.choices[0].logprobs.content:
    tokens = response.choices[0].logprobs.content
    
    # Find the actual answer token (skip thinking tokens)
    answer_token_idx = None
    in_thinking = False
    
    for i, token_data in enumerate(tokens):
        token = token_data.token.strip()
        
        # Track if we're in thinking mode
        if token == '<think>':
            in_thinking = True
            continue
        elif token == '</think>':
            in_thinking = False
            continue
            
        # Look for A, B, C, or D after thinking phase ends
        if not in_thinking and token in ['A', 'B', 'C', 'D']:
            answer_token_idx = i
            break
    
    if answer_token_idx is not None:
        token_data = tokens[answer_token_idx]
        
        print(f"Selected Answer: '{token_data.token}'")
        print(f"Logprob: {token_data.logprob:.4f}")
        print(f"Probability: {np.exp(token_data.logprob)*100:.2f}%")
        
        print("\n" + "-" * 50)
        print("ALL CHOICE PROBABILITIES AT DECISION POINT:")
        print("-" * 50)
        
        # Check this position's alternatives
        choices = {'A': None, 'B': None, 'C': None, 'D': None}
        
        # Add the selected token
        if token_data.token in choices:
            choices[token_data.token] = token_data.logprob
        
        # Add alternatives at this position
        if token_data.top_logprobs:
            for alt in token_data.top_logprobs:
                if alt.token in choices:
                    choices[alt.token] = alt.logprob
        
        # Display all choices
        for letter, logprob in sorted(choices.items()):
            if logprob is not None:
                prob = np.exp(logprob) * 100
                print(f"Choice {letter}: {logprob:8.4f} ({prob:6.2f}%)")
            else:
                print(f"Choice {letter}: Not in top alternatives")
    else:
        # Show what was actually generated
        print("No A/B/C/D answer found. Generated text:")
        full_text = ''.join([t.token for t in tokens])
        print(full_text[:200])
        
    # Debug: show tokens to understand structure
    print("\n" + "-" * 50)
    print(f"TOTAL TOKENS: {len(tokens)}")
    print("TOKEN SEQUENCE (all tokens):")
    print("-" * 50)
    in_thinking = False
    thinking_end_idx = None
    
    for i, t in enumerate(tokens):
        display = t.token.replace('\n', '\\n')
        if t.token.strip() == '<think>':
            in_thinking = True
            print(f"{i:3}: '{display}' <-- THINKING STARTS")
        elif t.token.strip() == '</think>':
            in_thinking = False
            thinking_end_idx = i
            print(f"{i:3}: '{display}' <-- THINKING ENDS")
        elif not in_thinking and t.token.strip() in ['A', 'B', 'C', 'D']:
            print(f"{i:3}: '{display}' <-- ANSWER FOUND!")
            # Show a few more tokens after answer
            for j in range(i+1, min(i+5, len(tokens))):
                print(f"{j:3}: '{tokens[j].token.replace(chr(10), '\\n')}'")
            break
        else:
            # Show limited tokens during thinking, all tokens after
            if i < 10 or (thinking_end_idx and i <= thinking_end_idx + 10):
                print(f"{i:3}: '{display}'")
