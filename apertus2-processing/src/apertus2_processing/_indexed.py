"""Megatron-LM indexed datasets (MMIDIDX v1) with one sequence per document."""

import os
import shutil
import struct

import numpy as np

HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODES = {np.dtype(np.int32): 4, np.dtype(np.float32): 7}  # Megatron DType codes
_PREAMBLE = len(HEADER) + 8 + 1 + 8 + 8
_CHUNK = 1 << 20


class Writer:
    """Stream documents into `<prefix>.bin`; a clean exit writes `<prefix>.idx`."""

    def __init__(self, prefix, dtype):
        self.prefix = str(prefix)
        self.dtype = np.dtype(dtype)
        self.lengths = []

    def __enter__(self):
        self.data = open(f"{self.prefix}.bin", "wb")
        return self

    def __exit__(self, error, *_):
        self.data.close()
        if error is None:
            lengths = [np.asarray(self.lengths, dtype=np.int64)]
            write_index(f"{self.prefix}.idx", self.dtype, lengths)

    def add(self, values):
        array = np.asarray(values, dtype=self.dtype)
        self.data.write(array.tobytes())
        self.lengths.append(len(array))


def write_index(path, dtype, lengths):
    """Write an index for consecutive documents whose lengths come in chunks."""
    dtype = np.dtype(dtype)
    count = sum(len(chunk) for chunk in lengths)
    with open(path, "wb") as index:
        index.write(HEADER + struct.pack("<QBQQ", 1, DTYPE_CODES[dtype], count, count + 1))
        index.writelines(np.asarray(chunk, dtype=np.int32).tobytes() for chunk in lengths)
        offset = 0
        for chunk in lengths:
            sizes = np.asarray(chunk, dtype=np.int64) * dtype.itemsize
            index.write((offset + np.cumsum(sizes) - sizes).tobytes())
            offset += int(sizes.sum())
        index.writelines(
            np.arange(start, min(start + _CHUNK, count + 1), dtype=np.int64).tobytes()
            for start in range(0, count + 1, _CHUNK)
        )


def read_lengths(prefix, dtype):
    """Return document lengths after validating the index and its data file."""
    dtype = np.dtype(dtype)
    with open(f"{prefix}.idx", "rb") as index:
        preamble = index.read(_PREAMBLE)
    if len(preamble) != _PREAMBLE or not preamble.startswith(HEADER):
        raise ValueError(f"not an indexed dataset: {prefix}.idx")
    version, code, sequences, documents = struct.unpack("<QBQQ", preamble[len(HEADER) :])
    if version != 1 or code != DTYPE_CODES[dtype] or documents != sequences + 1:
        raise ValueError(f"unexpected index layout: {prefix}.idx")
    lengths = np.fromfile(f"{prefix}.idx", dtype=np.int32, count=sequences, offset=_PREAMBLE)
    if os.path.getsize(f"{prefix}.bin") != int(lengths.sum(dtype=np.int64)) * dtype.itemsize:
        raise ValueError(f"data size disagrees with index: {prefix}.bin")
    return lengths


def merge(prefixes, target, dtype):
    """Concatenate indexed datasets in order and return the document count."""
    lengths = [read_lengths(prefix, dtype) for prefix in prefixes]
    with open(f"{target}.bin", "wb") as data:
        for prefix in prefixes:
            with open(f"{prefix}.bin", "rb") as source:
                shutil.copyfileobj(source, data, 16 * _CHUNK)
    write_index(f"{target}.idx", dtype, lengths)
    return sum(len(chunk) for chunk in lengths)
