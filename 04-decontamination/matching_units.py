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
"""
import hashlib
import re
import unicodedata

# Part of the benchmark n-gram cache key and recorded in the processing log: bump when the units change.
# v2: units as above; also get_ngrams no longer drops the last n-gram.
MATCHING_VERSION = "v2"

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


def units_with_spans(tokenizer, text):
    """(normalised text, its units, character span of every unit in the normalised text)."""
    text = normalize(text)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return (text, *merge_numbers(enc["input_ids"], enc["offset_mapping"], text))
