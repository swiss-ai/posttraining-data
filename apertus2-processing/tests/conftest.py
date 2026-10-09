import pytest
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


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
