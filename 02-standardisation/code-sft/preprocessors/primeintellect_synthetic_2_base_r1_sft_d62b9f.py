"""Dataset-specific preprocessing for PrimeIntellect/SYNTHETIC-2-Base-R1-SFT."""

from .base import DatasetPreprocessor, prepare_fields, benchmark_fields


class PrimeintellectSynthetic2BaseR1SftPreprocessor(DatasetPreprocessor):
    repo = "PrimeIntellect/SYNTHETIC-2-Base-R1-SFT"
    policy = "labels"
    description = "Preserve released conversation fields, tools and source records; normalize role aliases and map available prompt/response fields. Keep only explicit coding domain/source/task labels; exclude known non-code and unclassified records."

    def prepare(self, row, entry, file):
        return prepare_fields(row)
