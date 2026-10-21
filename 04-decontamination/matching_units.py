"""The units the decontamination check compares: tokens of the normalised text, numbers as one unit.

normalize(text):
  - Unicode NFKC (full-width characters, ligatures, special spaces -> their plain form)
  - a run of one repeated character that is neither a letter, digit nor whitespace is collapsed to one
    (`━━━━━` -> `━`, `-----` -> `-`, `!!!` -> `!`)
  - whitespace runs are collapsed to one space, and leading and trailing whitespace is removed

Units are the tokenizer's tokens of the normalised text, without special tokens (no `<s>`), except that a
number (`183`, `1,023`, `16.482`, `2'832`) and the space before it form one unit. The tokenizer splits
numbers into single digits, so without this one number would count as up to ~10 tokens. A number unit is
a negative id derived from the number's text: equal numbers match, and it never equals a token id.

Benchmark prompts shorter than the n-gram length have no n-gram, so the n-gram check never sees them.
They are checked by short_prompt_match instead:
  - 0 units: skipped (counted as empty)
  - 1 to EXACT_MAX_UNITS units: the training prompt must be identical to the benchmark prompt
  - EXACT_MAX_UNITS + 1 to n - 1 units: the whole benchmark prompt must appear as one run in the training
    prompt and make up at least SHORT_MIN_COVERAGE of the training prompt's units
"""
import hashlib
import re
import unicodedata
from collections import defaultdict

# Part of the benchmark n-gram cache key: bump when the units change.
# v2: units as above; also get_ngrams no longer drops the last n-gram.
UNITS_VERSION = "v2"
# Recorded in the processing log: bump when the units or the matching rules change.
# v3: benchmark prompts shorter than the n-gram length are checked by short_prompt_match.
MATCHING_VERSION = "v3"

EXACT_MAX_UNITS = 3
SHORT_MIN_COVERAGE = 0.5

NUMBER = re.compile(r"\d+(?:[.,'’]\d+)*")
REPEATED = re.compile(r"([^\w\s]|_)\1+")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return WHITESPACE.sub(" ", REPEATED.sub(r"\1", text)).strip()


def normalize_with_positions(text: str):
    """(NFKC form of text, normalize(text), position of every normalised character in the NFKC form).

    For showing matches in readable text: the NFKC form keeps line breaks and repeated characters."""
    nfkc = unicodedata.normalize("NFKC", text)
    out, pos = [], []
    for i, c in enumerate(nfkc):
        if c.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                pos.append(i)
        elif not (i and c == nfkc[i - 1] and (c == "_" or not (c.isalnum() or c.isspace()))):
            out.append(c)
            pos.append(i)
    if out and out[-1] == " ":
        out.pop()
        pos.pop()
    return nfkc, "".join(out), pos


def number_id(number: str) -> int:
    return -1 - int(hashlib.md5(number.encode()).hexdigest()[:15], 16)


def merge_numbers(token_ids, offsets, text):
    """(units, character span of every unit in text) from the tokens of the normalised text."""
    numbers = [m.span() for m in NUMBER.finditer(text)]
    units, spans, j, last = [], [], 0, None
    for token, (a, b) in zip(token_ids, offsets):
        while j < len(numbers) and numbers[j][1] <= a:
            j += 1
        if j < len(numbers):
            start, end = numbers[j]
            lead = start - 1 if start and text[start - 1] == " " else start
            if lead <= a and b <= end:  # a digit or separator of the number, or the space before it
                if last != j:
                    units.append(number_id(text[start:end]))
                    spans.append((start, end))
                    last = j
                continue
        units.append(token)
        spans.append((a, b))
    return units, spans


def to_units(tokenizer, texts):
    """Units of every text in the list texts."""
    texts = [normalize(t) for t in texts]
    enc = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True)
    return [merge_numbers(ids, offs, t)[0] for ids, offs, t in zip(enc["input_ids"], enc["offset_mapping"], texts)]


def short_prompt_index(eval_units, ngram_length):
    """({units of a benchmark prompt shorter than ngram_length: [its indices]}, counts per length class)."""
    index = defaultdict(list)
    counts = {"empty": 0, "exact_only": 0, "contained": 0}
    for idx, units in enumerate(eval_units):
        if len(units) >= ngram_length:  # covered by the n-gram check
            continue
        if not units:
            counts["empty"] += 1
            continue
        counts["exact_only" if len(units) <= EXACT_MAX_UNITS else "contained"] += 1
        index[tuple(units)].append(idx)
    return dict(index), counts


def short_prompt_match(train_units, index, lengths):
    """Index of the first short benchmark prompt that train_units match (see module docstring), else None.

    lengths: the sorted lengths of the keys of index."""
    t = len(train_units)
    for n in lengths:
        if n <= EXACT_MAX_UNITS:
            if t == n and tuple(train_units) in index:
                return index[tuple(train_units)][0]
        elif n <= t and n >= SHORT_MIN_COVERAGE * t:
            # a window of n units equal to a key is the whole benchmark prompt as one run
            for i in range(t - n + 1):
                hit = index.get(tuple(train_units[i:i + n]))
                if hit:
                    return hit[0]
    return None


def max_short_train_units(ngram_length):
    """Training prompts longer than this can never match a short benchmark prompt (7 / 0.5 = 14 for 8-grams)."""
    return max(EXACT_MAX_UNITS, int((ngram_length - 1) / SHORT_MIN_COVERAGE))


def units_with_spans(tokenizer, text):
    """(normalised text, its units, character span of every unit in the normalised text)."""
    text = normalize(text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return (text, *merge_numbers(enc["input_ids"], enc["offset_mapping"], text))
