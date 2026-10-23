"""Reference recipe for PRO-ARL-001: DAPO (Yu et al. 2025) with the Python tool, from Qwen3-4B-Base.

Launch (registered with the supervisor, from this directory):
  python3 pro_audit.py run --name dapo --code . -- torchrun --standalone --nproc_per_node=3 train.py

Ranks 0-2 train on GPUs 1-3 with FSDP; rank 0 also drives a vLLM rollout worker on GPU 0 (src/rollout.py).
Each step: 64 problems x 16 answers with tools; reward 1 for a correct boxed integer, 0 otherwise, minus a linear
overlong penalty over the last 2,048 tokens of the 16,384 budget; groups whose answers are all right or all wrong are
dropped and refilled (dynamic sampling); advantages are group-normalised; the loss is the token-mean clipped
surrogate with clip-higher (eps_low 0.2, eps_high 0.28) and no KL term; tool outputs are masked out of the loss.
A checkpoint (model/ in Hugging Face format, optimiser and sampler state, DONE) is written every 20 steps.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import math_tool as MT  # noqa: E402

STEPS, PROBLEMS, N, LR = 400, 64, 16, 1e-6
EPS_LOW, EPS_HIGH, OVERLONG = 0.2, 0.28, 2048
MICRO, SAVE_EVERY = 2, 20


def items(data: Path) -> list:
    return [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]


def rewards(resps: list, gold: int, tok) -> list:
    out = []
    for r in resps:
        n = len(tok(r)["input_ids"])
        pen = max(0.0, (n - (MT.MAX_TOKENS - OVERLONG)) / OVERLONG)
        out.append(float(MT.correct_int(r, gold)) - pen)
    return out


def masks(tok, prompt: str, resp: str):
    """Token ids of prompt + response and a loss mask that is 0 on the prompt and on tool outputs."""
    ids_p = tok(prompt)["input_ids"]
    pieces, mask = [], []
    rest = resp
    while "<output>" in rest:
        a, b = rest.split("<output>", 1)
        b2 = b.split("</output>", 1)
        pieces.append((a, 1))
        pieces.append(("<output>" + b2[0] + "</output>", 0))
        rest = b2[1] if len(b2) > 1 else ""
    pieces.append((rest, 1))
    ids = list(ids_p)
    mask = [0] * len(ids_p)
    for text, m in pieces:
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        mask += [m] * len(t)
    return ids, mask


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank + 1)
    dev = torch.device("cuda", rank + 1)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data, models = Path(os.environ["ABENCH_DATA_DIR"]), Path(os.environ["ABENCH_MODELS_DIR"])
    ck_root, resume = Path(os.environ["ABENCH_CKPT_DIR"]), os.environ.get("ABENCH_RESUME_FROM")
    base = models / "Qwen3-4B-Base"
    tok = AutoTokenizer.from_pretrained(base)
    src = Path(resume) / "model" if resume else base
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model = FSDP(model, device_id=dev, mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                 reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16), use_orig_params=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
    step0, rng = 0, random.Random(0)
    if resume:
        st = torch.load(Path(resume) / f"state_rank{rank}.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        step0, rng = st["step"], random.Random()
        rng.setstate(st["rng"])
    pool = items(data)
    conn = None
    if rank == 0:
        import multiprocessing as mp
        from rollout import worker
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        ctx.Process(target=worker, args=(child, str(src), 4096 + MT.MAX_TOKENS), daemon=True).start()
        conn.recv()
    shm = Path("/dev/shm/abench_arl_weights.safetensors")
    for step in range(step0, STEPS):
        t0 = time.time()
        batch = []
        if rank == 0:
            while len(batch) < PROBLEMS:
                probs = rng.sample(pool, PROBLEMS)
                conn.send(("generate", [p["problem"] for p in probs], N, step * 1000 + len(batch)))
                outs = conn.recv()
                for p, rs in zip(probs, outs):
                    r = rewards(rs, p["answer"], tok)
                    if max(r) > min(r):                       # dynamic sampling: keep informative groups
                        batch.append((p, rs, r))
            batch = batch[:PROBLEMS]
        obj = [batch]
        dist.broadcast_object_list(obj, src=0)
        batch = obj[0]
        samples = []
        for p, rs, r in batch:
            mu, sd = sum(r) / len(r), (sum((x - sum(r) / len(r)) ** 2 for x in r) / len(r)) ** 0.5 + 1e-6
            prompt = MT.chat_prompt(tok, p["problem"], True)
            for resp, x in zip(rs, r):
                samples.append((prompt, resp, (x - mu) / sd))
        mine = samples[rank::world]
        # old log-probabilities (one on-policy update per batch, so old = current before the step)
        opt.zero_grad(set_to_none=True)
        n_tok = torch.zeros((), device=dev)
        for prompt, resp, adv in mine:
            ids, m = masks(tok, prompt, resp)
            n_tok += sum(m)
        dist.all_reduce(n_tok)
        for i in range(0, len(mine), MICRO):
            loss = torch.zeros((), device=dev)
            for prompt, resp, adv in mine[i:i + MICRO]:
                ids, m = masks(tok, prompt, resp)
                x = torch.as_tensor([ids], device=dev)
                logp = torch.log_softmax(model(x).logits[0, :-1].float(), -1).gather(1, x[0, 1:, None])[:, 0]
                mask = torch.as_tensor(m[1:], device=dev, dtype=torch.float32)
                ratio = torch.exp(logp - logp.detach())
                a = torch.full_like(logp, adv)
                surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - EPS_LOW, 1 + EPS_HIGH) * a)
                loss = loss - (surr * mask).sum() / n_tok * world
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        mean_r = sum(x for _, _, r in batch for x in r) / max(1, sum(len(r) for _, _, r in batch))
        if rank == 0:
            print(f"ABENCH step={step} reward={mean_r:.5f} seconds={time.time() - t0:.1f}", flush=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full = model.state_dict()
        if rank == 0:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in full.items()}, str(shm))
            conn.send(("weights", str(shm)))
            conn.recv()
        if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
            d = ck_root / f"step_{step + 1:06d}"
            if rank == 0:
                (d / "model").mkdir(parents=True, exist_ok=True)
                model.module.save_pretrained(d / "model", state_dict=full, safe_serialization=True)
                tok.save_pretrained(d / "model")
            dist.barrier()
            torch.save({"opt": opt.state_dict(), "step": step + 1, "rng": rng.getstate()}, d / f"state_rank{rank}.pt")
            dist.barrier()
            if rank == 0:
                (d / "DONE").write_text("")
    if rank == 0:
        conn.send(("stop",))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
