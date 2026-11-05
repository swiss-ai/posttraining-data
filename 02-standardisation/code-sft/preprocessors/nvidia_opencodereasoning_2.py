"""Dataset-specific preprocessing for nvidia/OpenCodeReasoning-2.

The release ships every row with `question` set to "-": the problem statements are left out
and have to be fetched from the datasets they came from (see ocr2_questions.py). This
preprocessor fills them in from the lookup table that script writes, so each row becomes the
question as the user turn and R1's generation as the reply.

Rows whose question comes from an evaluation split (APPS test, TACO test, CodeContests
valid/test) are rejected: those are benchmark problems.

The lookup table is found at $OCR2_QUESTIONS, or by default next to the downloaded datasets.
"""

import os
from functools import lru_cache

from .base import DatasetPreprocessor, prepare_fields

DEFAULT_LOOKUP = (
    "/capstor/store/cscs/swissai/infra01/datasets/swe_terminal_sft_apertus_v2/"
    ".question_sources/ocr2_questions.parquet"
)


@lru_cache(maxsize=1)
def questions(path):
    import pyarrow.parquet as pq

    if not os.path.exists(path):
        raise ValueError(
            f"OpenCodeReasoning-2 question lookup not found at {path}; build it with "
            "02-standardisation/code-sft/ocr2_questions.py or set OCR2_QUESTIONS"
        )
    table = pq.read_table(path, columns=["dataset", "split", "index", "question"])
    return {
        (r["dataset"], r["split"], r["index"]): r["question"] for r in table.to_pylist()
    }


class NvidiaOpencodereasoning2Preprocessor(DatasetPreprocessor):
    repo = "nvidia/OpenCodeReasoning-2"
    policy = "code_source"
    description = "Restore the left-out questions from their source datasets; reject benchmark-split questions."

    def prepare(self, row, entry, file):
        from ocr2_questions import EVALUATION_SPLITS

        key = (row.get("dataset"), row.get("split"))
        if key in EVALUATION_SPLITS:
            raise ValueError(
                f"benchmark_split_question: the question is from the {key[1]} split of "
                f"{key[0]}, an evaluation set"
            )
        if row.get("question") not in (None, "", "-"):
            return prepare_fields(row)
        lookup = questions(os.environ.get("OCR2_QUESTIONS", DEFAULT_LOOKUP))
        question = lookup.get((key[0], key[1], int(row.get("index", -1))))
        if question is None:
            raise ValueError(
                f"question_not_restored: no question for {key[0]} {key[1]} row {row.get('index')}"
            )
        row = dict(row, question=question)
        return prepare_fields(row)
