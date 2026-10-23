"""A vLLM sampler pinned to one GPU (student on GPU 0 with in-place weight reload; teacher on GPU 1)."""
from __future__ import annotations

import os


def worker(conn, model_dir: str, max_len: int, gpu: int = 0, mem: float = 0.85):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("VLLM_USE_V1", "0")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm import LLM
    import math_tool as MT
    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=max_len)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        if msg[0] == "weights":
            model.load_weights(load_file(msg[1]).items())
            conn.send("ok")
        else:
            _, problems, n, seed, max_tokens = msg
            conn.send(MT.generate(llm, tok, problems, n, tools=False, seed=seed, max_tokens=max_tokens))
