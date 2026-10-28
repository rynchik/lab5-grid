"""Потоковая генерация файлов NumPy без хранения всей матрицы в RAM."""

import json
from pathlib import Path

import numpy as np


def generate(directory, M, N, seed, chunk_bytes=16 * 1024 * 1024):
    if M < N or N <= 0:
        raise ValueError("Требуются M >= N > 0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    truth = np.random.default_rng(seed + 1).normal(size=N)
    matrix = np.lib.format.open_memmap(
        directory / "A.npy", mode="w+", dtype=np.float64, shape=(M, N)
    )
    b = np.lib.format.open_memmap(
        directory / "b.npy", mode="w+", dtype=np.float64, shape=(M,)
    )
    gradient = np.zeros(N)
    rng = np.random.default_rng(seed)
    block_rows = max(1, chunk_bytes // (8 * N))
    for first in range(0, M, block_rows):
        last = min(first + block_rows, M)
        block = rng.normal(size=(last - first, N))
        block /= np.sqrt(M)
        value = block @ truth
        matrix[first:last] = block
        b[first:last] = value
        gradient += block.T @ value
    matrix.flush()
    b.flush()
    np.save(directory / "x.npy", truth)
    np.save(directory / "Atb.npy", gradient)
    metadata = {
        "M": M,
        "N": N,
        "seed": seed,
        "dtype": "float64",
        "matrix_bytes": 8 * M * N,
        "chunk_bytes": chunk_bytes,
    }
    (directory / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    del matrix, b
    return metadata
