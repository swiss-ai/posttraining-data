"""Dataset-specific preprocessing for nvidia/Nemotron-Competitive-Programming-v1.

Its competitive-coding files refer to each problem by `dataset`, `split` and `index`. The rows
from CodeContests and open-r1/codeforces carry the prompt, which is always NVIDIA's instruction
for the file's language followed by the problem statement. The rows from TACO and APPS leave
the user turn as "-". Those are restored from the lookup table of ocr2_questions.py, with the
same instruction, so every row of a file is prompted the same way.

Rows whose problem is from an evaluation split (APPS test, TACO test, CodeContests valid/test)
are rejected: those are benchmark problems. The infinibyte files have no such fields and are
kept as released.
"""

import os

from .base import DatasetPreprocessor, prepare_fields
from .nvidia_opencodereasoning_2 import DEFAULT_LOOKUP, questions

# The instruction in front of every released prompt of these files, per language
INSTRUCTION = (
    "You are a helpful and harmless assistant. You should think step-by-step before responding "
    "to the instruction below.\n\nPlease use {name} programming language only.\n\nYou must use "
    "```{fence} for just the final solution code block with the following format:\n```{fence}\n"
    "{comment} Your code here\n```\n\n"
)
LANGUAGES = {
    "competitive_coding_python": dict(name="python", fence="python", comment="#"),
    "competitive_coding_cpp": dict(name="c++", fence="cpp", comment="//"),
}


class NvidiaNemotronCompetitiveProgrammingV1Preprocessor(DatasetPreprocessor):
    repo = "nvidia/Nemotron-Competitive-Programming-v1"
    policy = "code_source"
    description = "Restore left-out TACO/APPS questions with the release's instruction; reject benchmark-split problems."

    def prepare(self, row, entry, file):
        from ocr2_questions import EVALUATION_SPLITS

        key = (row.get("dataset"), row.get("split"))
        if key[0] is None:
            return prepare_fields(row)
        if key in EVALUATION_SPLITS:
            raise ValueError(
                f"benchmark_split_question: the problem is from the {key[1]} split of {key[0]}, an evaluation set"
            )
        messages = row.get("messages") or []
        if messages and isinstance(messages[0], dict) and messages[0].get("content") == "-":
            language = next((v for k, v in LANGUAGES.items() if k in os.path.basename(file.get("path", ""))), None)
            if language is None:
                raise ValueError(f"question_not_restored: unknown language of {file.get('path')}")
            lookup = questions(os.environ.get("OCR2_QUESTIONS", DEFAULT_LOOKUP))
            question = lookup.get((key[0], key[1], int(row.get("index", -1))))
            if question is None:
                raise ValueError(f"question_not_restored: no question for {key[0]} {key[1]} row {row.get('index')}")
            first = dict(messages[0], content=INSTRUCTION.format(**language) + question)
            row = dict(row, messages=[first] + list(messages[1:]))
        return prepare_fields(row)
