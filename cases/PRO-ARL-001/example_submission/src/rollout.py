"""Rollout worker on GPU 0: a vLLM engine that reloads the learner's weights from shared memory before each batch
and samples tool-using answers with the same loop as the evaluator (math_tool.generate)."""
from __future__ import annotations

import os


def worker(conn, model_dir: str, max_len: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("VLLM_USE_V1", "0")            # the V0 engine exposes the model for in-place weight loading
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm import LLM
    import math_tool as MT
    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=max_len,
              enable_prefix_caching=True)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            return
        if msg[0] == "weights":
            model.load_weights(load_file(msg[1]).items())
            llm.llm_engine.reset_prefix_cache()
            conn.send("ok")
        elif msg[0] == "generate":
            _, problems, n, seed = msg
            conn.send(MT.generate(llm, tok, problems, n, tools=True, seed=seed))
