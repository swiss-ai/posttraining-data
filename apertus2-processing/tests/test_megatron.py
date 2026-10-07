import argparse
import os
import struct

import numpy as np
import pytest
from apertus_common import Conversation, Reply, SystemPrompt, Wait, load_encoding

from apertus2_processing._megatron import export, validate_row
from apertus2_processing._util import read_json
from apertus2_processing.cli import main


def test_reject_invalid_export_rows():
    for ids, count, weights in [
        ([2**31], 1, [1.0]),
        ([1], 2, [1.0]),
        ([1], 1, []),
        ([1], 1, [float("nan")]),
    ]:
        with pytest.raises(ValueError):
            validate_row({"token_ids": ids, "token_count": count, "loss_weights": weights}, True)


@pytest.mark.skipif(
    not os.environ.get("MEGATRON_PATH"),
    reason="set MEGATRON_PATH in a consumer-compatible environment",
)
def test_real_megatron_reader_and_paired_weights(tmp_path, artifact):
    conversation = Conversation(
        system=SystemPrompt.build(), items=[Reply(payload="hello", loss_weight=0.125), Wait()]
    )
    source = tmp_path / "input.jsonl"
    source.write_text((conversation.to_json() + "\n") * 3)
    tokens = tmp_path / "tokens"
    assert (
        main(
            [
                "tokenize",
                str(source),
                "--output",
                str(tokens),
                "--tokenizer",
                str(artifact),
                "--emit-loss-weights",
            ]
        )
        == 0
    )
    output = tmp_path / "indexed"
    args = argparse.Namespace(
        token_output=str(tokens),
        output=str(output),
        megatron_path=os.environ["MEGATRON_PATH"],
        documents_per_shard=2,
        tokens_per_shard=100000,
        resume=False,
    )
    export(args)
    from megatron.core.datasets.indexed_dataset import IndexedDataset

    manifest = read_json(output / "manifest.json")
    assert [part["documents"] for part in manifest["shards"]] == [2, 1]
    expected = load_encoding(artifact).encode_conversation(conversation, return_loss_weights=True)
    for part in manifest["shards"]:
        ids = IndexedDataset(str(output / part["token_prefix"]))
        weights = IndexedDataset(str(output / part["loss_weight_prefix"]))
        assert len(ids) == len(weights) == part["documents"]
        assert (
            ids.document_indices.tolist()
            == weights.document_indices.tolist()
            == list(range(len(ids) + 1))
        )
        assert (
            ids.sequence_lengths.tolist()
            == weights.sequence_lengths.tolist()
            == [len(expected.token_ids)] * len(ids)
        )
        for i in range(len(ids)):
            np.testing.assert_array_equal(ids[i], expected.token_ids)
            np.testing.assert_array_equal(weights[i], expected.loss_weights)
        # The consumer reader's dtype agrees with the written index headers.
        with (output / (part["token_prefix"] + ".idx")).open("rb") as stream:
            assert stream.read(9) == b"MMIDIDX\x00\x00"
            assert struct.unpack("<Q", stream.read(8))[0] == 1
            assert stream.read(1) == bytes([4])
        with (output / (part["loss_weight_prefix"] + ".idx")).open("rb") as stream:
            stream.seek(17)
            assert stream.read(1) == bytes([6])
    args.resume = True
    export(args)
    assert read_json(output / "manifest.json") == manifest
