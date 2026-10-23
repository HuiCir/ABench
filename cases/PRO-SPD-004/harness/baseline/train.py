"""PRO-SPD-004 reference submission, training program: the modded-nanogpt GPT-2 Medium track record of 2025-11-12
(record 17, "Remove Redundant Mask Op"; https://github.com/KellerJordan/modded-nanogpt, commit 4ea6b937, file
records/track_2_medium/2025-11-12_BlockMaskRedundantOp/train_gpt_medium.py, MIT licence, Copyright (c) 2024 Keller
Jordan), adapted from 8 to 4 GPUs and to the ABench launch interface.

  torchrun --standalone --nproc_per_node=4 train.py --data DIR --out DIR --seed S [--quick] [--steps N]

What the record logged (8 x H100, PyTorch 2.7.0+cu126): val_loss 2.918512 at step 5,960, train_time 1,496,063 ms.
Changes, all made so that the recipe is the record's own on half the GPUs:
  - Same global batch (8 x 65,536 tokens per step): with W GPUs every rank runs 8 / W micro-steps per step and the
    gradients are averaged over micro-steps and ranks before the optimizer step (the record: one micro-step on 8 GPUs).
  - Steps 6,100 instead of 5,960 (+2.3%): the record ended 0.0015 below the 2.92 target, and the ABench evaluation
    differs slightly (32,768-token windows instead of 262,144) and runs two secret seeds; not calibrated yet.
  - No validation inside the run (the validation shard is not available), no logging of the code or nvidia-smi.
  - torch.manual_seed(seed) before model construction; rank 0 writes the state dict to OUT/model.pt.
  - Data from --data instead of data/fineweb10B/; the shards are read in order, as in the record.
Unchanged: model (model.py), Muon (with bf16 weights + uint16 mantissa), Adam groups and learning rates, the
stable-then-linear-decay schedule (cooldown 0.7), the cubic attention-window schedule, Muon momentum warm-up, and the
10-step kernel warm-up with state reset. Expected wall time on 4 x H100: about twice the record's, plus compilation.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch import Tensor, nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TRAIN_SEQ_LEN, build_model, next_multiple_of_n  # noqa: E402

torch._inductor.config.coordinate_descent_tuning = True  # allowed in the record's (Medium) track
torch._dynamo.config.compiled_autograd = True

GLOBAL_MICRO = 8                 # the record's 8 GPUs x one 65,536-token sequence per step

# -----------------------------------------------------------------------------
# Muon optimizer (verbatim from the record)

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ∈ [1 - l, 1 + r], which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def update(acc_bf16_view_u16: Tensor, mantissa: Tensor, momentum_buffer: Tensor, grad: Tensor, momentum: Tensor, eff_lr: Tensor, eff_weight_decay: Tensor):
    assert acc_bf16_view_u16.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    momentum_buffer.copy_(momentum * momentum_buffer + (1 - momentum) * grad)
    v = zeropower_via_newtonschulz5(momentum * momentum_buffer + (1 - momentum) * grad)

    acc_m_u32 = (acc_bf16_view_u16.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    acc_m_u32.view(torch.float32).mul_(1 - eff_weight_decay)
    acc_m_u32.view(torch.float32).add_(other=v, alpha=-eff_lr)
    acc_bf16_view_u16.copy_((acc_m_u32 >> 16).to(torch.uint16))
    mantissa.copy_(acc_m_u32.to(torch.uint16))

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        assert all(p.dtype == torch.bfloat16 for group in self.param_groups for p in group["params"])

    @torch.no_grad()
    def step(self):
        futures: list[torch.Future] = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * self.world_size
            momentum = torch._as_tensor_fullprec(group["momentum"])
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["mantissa"] = torch.zeros_like(p, dtype=torch.uint16)
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                    update(
                        p.view(torch.uint16), state["mantissa"], state["momentum_buffer"],
                        p.grad, momentum,
                        eff_lr=torch._as_tensor_fullprec(group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5),
                        eff_weight_decay=torch._as_tensor_fullprec(group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)),
                    )
                futures.append(dist.all_gather(params_pad[base_i:base_i + self.world_size], params_pad[base_i + self.rank], async_op=True).get_future())
        torch.futures.collect_all(futures).wait()

# -----------------------------------------------------------------------------
# data (the record's loader, reading DIR/fineweb_train_*.bin)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(files: list[Path], batch_size: int, rank: int, world_size: int):
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs, targets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6100)
    ap.add_argument("--quick", action="store_true", help="smoke run: 20 steps, no kernel warm-up")
    a = ap.parse_args()
    num_iterations = 20 if a.quick else a.steps
    cooldown_frac = 0.7
    seq_len = TRAIN_SEQ_LEN

    rank, world_size = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert GLOBAL_MICRO % world_size == 0, "the recipe needs 1, 2, 4 or 8 GPUs"
    accum = GLOBAL_MICRO // world_size
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.empty(1, device="cuda", requires_grad=True).backward()  # prevents a bug on some systems (record)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master = rank == 0

    torch.manual_seed(a.seed)
    model: nn.Module = build_model().cuda()
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[nn.Parameter] = [model.lm_head_w]
    collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
    assert {p for ps in collections for p in ps} == {*model.parameters()}
    adam_param_groups = [dict(params=head_params, lr=1 / 320), dict(params=embed_params, lr=0.3),
                         dict(params=scalar_params, lr=0.015)]
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
    opt2params = {opt: [p for g in opt.param_groups for p in g["params"]] for opt in optimizers}
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    def get_lr(step: int):
        x = step / num_iterations
        return 1.0 if x < 1 - cooldown_frac else (1 - x) / cooldown_frac

    @lru_cache(1)
    def window_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def get_window_size_blocks(step: int):
        x = step / num_iterations
        factor = 4 * x ** 3 - 6 * x ** 2 + 3 * x          # cubic schedule (record)
        return window_helper(next_multiple_of_n(3456 * factor, n=128))

    raw_model = model
    model = torch.compile(model, dynamic=False)

    def train_step(step: int, batches) -> None:
        for inputs, targets in batches:
            (model(inputs, targets, get_window_size_blocks(step)) / accum).backward()
        futures = {opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
                   for opt, params in opt2params.items()}
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1)
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        for opt in optimizers:
            torch.futures.collect_all(futures[opt]).wait()
            opt.step()
        model.zero_grad(set_to_none=True)

    if not a.quick:   # kernel warm-up, then restore the initial state (record)
        initial_state = copy.deepcopy(dict(model=raw_model.state_dict(), optimizers=[o.state_dict() for o in optimizers]))
        for _ in range(10):
            inputs = targets = torch.randint(0, 50257, size=(seq_len,), device="cuda")
            train_step(0, [(inputs.to(torch.int32), targets)] * accum)
        raw_model.load_state_dict(initial_state["model"])
        for opt, st in zip(optimizers, initial_state["optimizers"]):
            opt.load_state_dict(st)
        del initial_state

    files = sorted(Path(a.data).glob("fineweb_train_*.bin"))
    assert files, f"no fineweb_train_*.bin in {a.data}"
    loader = distributed_data_generator(files, world_size * seq_len, rank, world_size)
    model.train()
    for step in range(num_iterations):
        train_step(step, [next(loader) for _ in range(accum)])
        if master and (step % 250 == 0 or step == num_iterations - 1):
            print(f"step {step + 1}/{num_iterations}", flush=True)

    dist.barrier()
    if master:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in raw_model.state_dict().items()}, out / "model.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
