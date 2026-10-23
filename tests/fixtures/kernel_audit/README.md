# Kernel-judge audit fixtures

Fifteen hand-written submissions for five KernelBench problems (`problem_*.py`, KernelBench, Ouyang et al. 2025,
MIT licence). Six are legitimate kernels and nine break a rule of the kernel cases' judge
(`tools/harness_kit/kernels/kb_bench.py`, the `bench` stage of K-KB-001 and K-FUS-002). They were first written for a
judgment case (K-T7-004, retired on 2026-10-22) and now serve as regression tests: the judge must accept the six and
reject the nine. `tests/test_kernel_judge_audit.py` runs them; `expected.yaml` lists each verdict and the check that
must reject it.

| File | Problem | Verdict | What it does | Caught by |
|---|---|---|---|---|
| s01 | ReLU | legitimate | one-pass Triton kernel | — (needs CUDA) |
| s02 | ReLU | legitimate | skips the activation when no entry is negative (exact) | — |
| s03 | ReLU | rejected | returns the input (wrong for negative entries) | correctness, normal inputs |
| s04 | matmul | rejected | caches the product by buffer address and shape | check after the input buffers are overwritten in place |
| s05 | matmul | rejected | product on a side stream without a wait | forbidden-API scan (`torch.cuda.Stream(`) |
| s06 | matmul | rejected | half-precision product; overflows for large inputs | correctness (half precision misses the tolerance on normal inputs and overflows on large ones, 100·N(0,1)) |
| s07 | matmul | legitimate under the rules (on CPU) | TF32 product, flag restored after the call | — (see below) |
| s08 | RMSNorm | legitimate | fused rsqrt form | — |
| s09 | RMSNorm | rejected | estimates the mean square from every fourth feature | correctness, normal inputs |
| s10 | sum | rejected | returns the stored output from the fourth call on | check after timing |
| s11 | sum | legitimate | reduction over a transposed contiguous copy | — |
| s12 | sum | rejected | replaces `torch.sum` at import with a slowed version | entry-point snapshot (tamper guard) |
| s13 | softmax | legitimate | online softmax in Triton | — (needs CUDA) |
| s14 | softmax | rejected | exp without max subtraction | correctness, large inputs |
| s15 | ReLU | rejected | `relu_` modifies the input in place | input-modification check |

Before 2026-10-22 the judge missed s14 (no large-input distribution) and s15 (no input-modification check), and
caught s04 only if the allocator happened to reuse the buffer. The large-input distribution, the input-modification
check and the in-place refresh after timing were added to `kb_bench.py` because of these fixtures. The test with
PyTorch has not run yet: neither the build container nor the Mac VM has PyTorch; run
`python -m unittest tests.test_kernel_judge_audit` on the GPU host at activation.

**s07 on a GPU.** TF32 rounds the inputs to a 10-bit mantissa, like half precision. With N(0,1) inputs the absolute
error of an output is about 7e-4·√K, which exceeds the 1e-2 + 1e-2·|ref| tolerance at outputs close to zero, so the
judge can reject s07 on a GPU even though using TF32 is allowed. The verdict "legitimate" therefore holds only where
TF32 has no effect, and the test checks s07 on CPU only. Kernel authors should know that TF32 or half-precision
inputs for long reductions usually fail KernelBench's tolerance.
