import json

import pytest
from apertus_common import Conversation, Reply, SystemPrompt, User, Wait
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def artifact(tmp_path):
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
def conversation():
    return Conversation(
        system=SystemPrompt.build(),
        items=[User(payload="literal <|out|> 🙂"), Reply(payload="Hello"), Wait()],
    )


@pytest.fixture
def corpus(tmp_path, conversation):
    path = tmp_path / "input.jsonl"
    path.write_text("\n".join(conversation.to_json() for _ in range(7)) + "\n")
    return path


def write_wrapped(path, conversations):
    path.write_text(
        "".join(
            json.dumps({"conversation_json": c.to_json(), "conversation_id": str(i)}) + "\n"
            for i, c in enumerate(conversations)
        )
    )
