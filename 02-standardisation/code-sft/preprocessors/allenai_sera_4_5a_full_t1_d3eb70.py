"""Dataset-specific preprocessing for allenai/Sera-4.5A-Full-T1."""

from .base import DatasetPreprocessor, prepare_fields, benchmark_fields


class AllenaiSera45AFullT1Preprocessor(DatasetPreprocessor):
    repo = "allenai/Sera-4.5A-Full-T1"
    policy = "code_source"
    description = "Preserve released conversation fields, tools and source records; normalize role aliases and map available prompt/response fields."

    def prepare(self, row, entry, file):
        return prepare_fields(row)
