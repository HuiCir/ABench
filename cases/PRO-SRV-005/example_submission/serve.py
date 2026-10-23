"""Reference submission for PRO-SRV-005: vLLM (PagedAttention, continuous batching, chunked prefill, prefix caching)
serving Qwen3-8B in bfloat16 with tensor parallelism over the 4 GPUs, behind the case's streaming protocol.

  python3 serve.py --model MODEL_DIR --port PORT
"""
from __future__ import annotations

import argparse
import json
import uuid

TP, MAX_SEQS, MAX_BATCHED, MAX_LEN = 4, 512, 16384, 16384


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.inputs import TokensPrompt
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=a.model, dtype="bfloat16", tensor_parallel_size=TP, max_num_seqs=MAX_SEQS, max_model_len=MAX_LEN,
        max_num_batched_tokens=MAX_BATCHED, enable_chunked_prefill=True, enable_prefix_caching=True,
        gpu_memory_utilization=0.92))
    app = FastAPI()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        sp = SamplingParams(temperature=0.0, max_tokens=int(body["max_new_tokens"]),
                            ignore_eos=bool(body.get("ignore_eos", True)))

        async def stream():
            sent = 0
            async for out in engine.generate(TokensPrompt(prompt_token_ids=body["input_ids"]), sp, uuid.uuid4().hex):
                ids = out.outputs[0].token_ids
                if len(ids) > sent:
                    yield json.dumps({"ids": list(ids[sent:])}) + "\n"
                    sent = len(ids)
            yield json.dumps({"done": True}) + "\n"
        return StreamingResponse(stream(), media_type="application/x-ndjson")

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
