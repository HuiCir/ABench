"""Token shards in the llm.c / modded-nanogpt format: 256 int32 header (magic 20240520, version 1, token count),
then uint16 GPT-2 token ids. Shared by B-OPT-001 and PRO-SPD-004."""
from __future__ import annotations

from pathlib import Path

import numpy as np

MAGIC, VERSION = 20240520, 1


def read_shard(path: Path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    if header[0] != MAGIC or header[1] != VERSION:
        raise ValueError(f"{path}: not a token shard")
    n = int(header[2])
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=256 * 4, shape=(n,))
    return toks


def write_shard(path: Path, tokens: np.ndarray) -> None:
    header = np.zeros(256, np.int32)
    header[:3] = (MAGIC, VERSION, len(tokens))
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.asarray(tokens, np.uint16).tobytes())
