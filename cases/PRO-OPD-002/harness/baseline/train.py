"""Public training baseline for PRO-OPD-002: on-policy distillation of Qwen3-8B into Qwen3-1.7B-Base (after GKD, Agarwal et
al. 2024, with the reverse KL estimated on the student's own samples).

Launch (registered with the supervisor, from this directory):
  python3 pro_audit.py run --name opd --code . -- torchrun --standalone --nproc_per_node=2 train.py

GPU 0: vLLM sampler of the current student (src/sampler.py); GPU 1: the frozen teacher, as a vLLM sampler (60% of
the memory) and as a transformers model that computes the log-probabilities of sampled tokens (bfloat16); GPUs 2-3: the student trained with FSDP (ranks 0
and 1). Phase 1 (first WARM steps): supervised fine-tuning on teacher answers to
the training prompts, so that the base student learns the chat format and the boxed answer. Phase 2: each step the
student samples 4 answers to each of 64 prompts (at most 8,192 tokens); the teacher gives every sampled token its log-probability;
the per-token advantage is log p_teacher - log p_student (the negative per-token reverse KL), and the student
maximises sum_t A_t log p_student(y_t) with the advantage held constant. Checkpoints every 25 steps.
"""
from __future__ import annotations

import json
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

STEPS, WARM, PROMPTS, N, LR, MAXTOK, SAVE_EVERY = 600, 100, 64, 4, 1e-6, 8192, 25


def logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids).logits[0, :-1].float()
    return torch.log_softmax(out, -1).gather(1, ids[0, 1:, None])[:, 0]


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", rank + 2)
    torch.cuda.set_device(dev)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    data, models = Path(os.environ["ABENCH_DATA_DIR"]), Path(os.environ["ABENCH_MODELS_DIR"])
    ck_root, resume = Path(os.environ["ABENCH_CKPT_DIR"]), os.environ.get("ABENCH_RESUME_FROM")
    student_dir = Path(resume) / "model" if resume else models / "Qwen3-1.7B-Base"
    tok = AutoTokenizer.from_pretrained(models / "Qwen3-8B")          # the chat template of the teacher family
    student = AutoModelForCausalLM.from_pretrained(student_dir, torch_dtype=torch.bfloat16)
    student.gradient_checkpointing_enable()
    student = FSDP(student, device_id=dev, use_orig_params=True, mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16))
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    step0, rng = 0, random.Random(0)
    if resume:
        st = torch.load(Path(resume) / f"state_rank{rank}.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
        rng.setstate(st["rng"])
    pool = [json.loads(x) for x in (data / "train.jsonl").read_text().splitlines()]
    teacher = conn = tconn = None
    if rank == 0:
        import multiprocessing as mp
        from sampler import worker
        ctx = mp.get_context("spawn")
        conn, child = ctx.Pipe()
        ctx.Process(target=worker, args=(child, str(student_dir), 4096 + MAXTOK), daemon=True).start()
        conn.recv()
        tconn, tchild = ctx.Pipe()
        ctx.Process(target=worker, args=(tchild, str(models / "Qwen3-8B"), 4096 + MAXTOK, 1, 0.6), daemon=True).start()
        tconn.recv()
    teacher = AutoModelForCausalLM.from_pretrained(models / "Qwen3-8B", torch_dtype=torch.bfloat16).to("cuda:1").eval() \
        if rank == 0 else None
    shm = Path("/dev/shm/abench_opd_student.safetensors")
    for step in range(step0, STEPS):
        t0 = time.time()
        probs = rng.sample(pool, PROMPTS)
        obj = [None]
        if rank == 0:
            if step < WARM:
                tconn.send(("generate", [p["problem"] for p in probs], 1, step, MAXTOK))
                outs = tconn.recv()
                samples = [(MT.chat_prompt(tok, p["problem"], False), o[0], None) for p, o in zip(probs, outs)
                           if MT.correct_int(o[0], p["answer"])]
            else:
                conn.send(("generate", [p["problem"] for p in probs], N, step, MAXTOK))
                outs = conn.recv()
                samples = []
                with torch.no_grad():
                    for p, os_ in zip(probs, outs):
                        prompt = MT.chat_prompt(tok, p["problem"], False)
                        for o in os_:
                            ids_p = tok(prompt)["input_ids"]
                            ids = ids_p + tok(o, add_special_tokens=False)["input_ids"]
                            lt = logprobs(teacher, torch.as_tensor([ids], device="cuda:1"))[len(ids_p) - 1:]
                            samples.append((prompt, o, lt.cpu()))
            obj = [samples]
        dist.broadcast_object_list(obj, src=0)
        mine = obj[0][rank::world]
        opt.zero_grad(set_to_none=True)
        tot, n_tok = 0.0, 0
        for prompt, o, lt in mine:
            ids_p = tok(prompt)["input_ids"]
            ids = torch.as_tensor([ids_p + tok(o, add_special_tokens=False)["input_ids"]], device=dev)
            ls = logprobs(student, ids)[len(ids_p) - 1:]
            if lt is None:                                     # phase 1: SFT on correct teacher answers
                loss = -ls.mean()
            else:                                              # phase 2: per-token reverse-KL policy gradient
                adv = (lt.to(dev) - ls.detach())
                loss = -(adv * ls).mean()
            (loss / max(len(mine), 1)).backward()
            tot += float(loss)
            n_tok += ls.numel()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        t = torch.tensor([tot, max(len(mine), 1)], device=dev)
        dist.all_reduce(t)
        if rank == 0:
            print(f"ABENCH step={step} loss={float(t[0] / t[1]):.5f} seconds={time.time() - t0:.1f}", flush=True)
        with FSDP.state_dict_type(student, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            full = student.state_dict()
        if rank == 0:
            from safetensors.torch import save_file
            save_file({k: v.contiguous() for k, v in full.items()}, str(shm))
            conn.send(("weights", str(shm)))
            conn.recv()
        if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
            d = ck_root / f"step_{step + 1:06d}"
            if rank == 0:
                (d / "model").mkdir(parents=True, exist_ok=True)
                student.module.save_pretrained(d / "model", state_dict=full, safe_serialization=True)
                tok.save_pretrained(d / "model")
            dist.barrier()
            torch.save({"opt": opt.state_dict(), "step": step + 1, "rng": rng.getstate()}, d / f"state_rank{rank}.pt")
            dist.barrier()
            if rank == 0:
                (d / "DONE").write_text("")
    if rank == 0:
        conn.send(("stop",))
        tconn.send(("stop",))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
