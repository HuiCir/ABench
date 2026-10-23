"""Reference submission for PRO-SPEC-006: vLLM's EAGLE-3 speculative decoding (Li et al. 2025) with the draft trained
by train.py (ABENCH_DRAFT_DIR), 5 speculative tokens per step, exact greedy verification, one GPU."""
from __future__ import annotations

import argparse
import json
import os
import uuid


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
    spec = {"method": "eagle3", "model": os.environ["ABENCH_DRAFT_DIR"], "num_speculative_tokens": 5}
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(model=a.model, dtype="bfloat16", tensor_parallel_size=1,
                                                             max_model_len=8192, gpu_memory_utilization=0.9,
                                                             speculative_config=spec))
    app = FastAPI()

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        sp = SamplingParams(temperature=0.0, max_tokens=int(body["max_new_tokens"]), ignore_eos=bool(body.get("ignore_eos", True)))

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
