"""U-LOOP-001: the data of B-OPT-001 (FineWeb-10B GPT-2 token shards, kjj0/fineweb10B-gpt2, MIT): train shards 1-2
verbatim in the sandbox; held out, the first 4,194,305 tokens of fineweb_val_000000.bin."""
from __future__ import annotations

from prepare.b_opt_001 import FILES, N_HELD, REV, SOURCES, build, synthetic  # noqa: F401
