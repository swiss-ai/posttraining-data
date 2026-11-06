"""Dataset-specific preprocessing for m-a-p/Code-Feedback."""

from .base import DatasetPreprocessor, prepare_fields, benchmark_fields


class MAPCodeFeedbackPreprocessor(DatasetPreprocessor):
    repo = "m-a-p/Code-Feedback"
    policy = "code_source"
    description = "Preserve released conversation fields, tools and source records; normalize role aliases and map available prompt/response fields."

    def prepare(self, row, entry, file):
        return prepare_fields(row)
