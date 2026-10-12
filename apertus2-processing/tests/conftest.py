import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait, load_encoding
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from apertus2_processing import _indexed
from apertus2_processing.cli import main


def write_parquet(path, rows, row_group_size=None):
    table = pa.table({"conversation_json": pa.array(rows, pa.string())})
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def run_all(mode, source, run, *options, workers=1):
    """Prepare, encode, and merge a run through the CLI; return merge's exit code."""
    assert main(["prepare", mode, str(source), str(run), *map(str, options)]) == 0
    assert main(["encode", str(run), "--workers", str(workers)]) == 0
    return main(["merge", str(run)])


def documents(prefix, dtype):
    """Every document of an indexed dataset as a list of Python numbers."""
    lengths = _indexed.read_lengths(prefix, dtype)
    return [_indexed.read_document(prefix, dtype, lengths, i).tolist() for i in range(len(lengths))]


def encoded(artifact, records):
    """Direct encodings as (token ids, float32 loss weights) per record."""
    encoding = load_encoding(artifact)
    results = [
        encoding.encode_conversation(Conversation.from_json(r), return_loss_weights=True)
        for r in records
    ]
    return [r.token_ids for r in results], [np.float32(r.loss_weights).tolist() for r in results]


@pytest.fixture(autouse=True)
def isolate_slurm_step(monkeypatch):
    """Unit tests invoke encode locally, even when pytest runs inside an srun step."""
    for name in ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_PROCID", "SLURM_NTASKS"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def artifact(tmp_path):
    """A tiny byte-level tokenizer with the Apertus 2 control-token roles."""
    path = tmp_path / "tokenizer"
    path.mkdir()
    controls = ["<|in|>", "<|/in|>", "<|out|>", "<|/out|>", "<|hdr|>", "<|wait|>", "<|pad|>"]
    specials = [*controls, "<s>"]
    vocab = {
        token: index
        for index, token in enumerate([*specials, *sorted(pre_tokenizers.ByteLevel.alphabet())])
    }
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in specials])
    roles = dict(
        zip(
            (
                "input_start_token",
                "input_end_token",
                "output_start_token",
                "output_end_token",
                "header_end_token",
                "wait_token",
            ),
            controls[:-1],
            strict=True,
        )
    )
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        extra_special_tokens=roles,
        pad_token=controls[-1],
        bos_token="<s>",
        clean_up_tokenization_spaces=False,
    )
    hf.save_pretrained(path)
    return path


@pytest.fixture
def records():
    """Seven valid native conversations of different lengths, as JSON strings."""
    return [
        Conversation(
            system=SystemPrompt.build(),
            items=[
                User(payload=f"question {i} 🙂"),
                Reply(payload="answer " * (i % 3 + 1)),
                Wait(),
            ],
        ).to_json()
        for i in range(7)
    ]
