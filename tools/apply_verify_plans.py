"""Write a reduced verification plan (verify_plan) for every case in index.yaml according to spec/compute_budget.md, and reassign compute tiers accordingly.

GPU-h are estimated on the reference card (one 24GB, RTX 4090 class):
  dev     agent development budget
  verify  verification per submission (including local-discovery ablations)
  ref     one-off precompute of reference methods (0 means reusing another case's reference results or needing no reference training)
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CAPS = {"low": (2, 0.5, 10), "medium": (4, 2, 30), "high": (8, 6, 80), "pro": (96, 16, 200)}   # pro is in H100-h

J = dict(proxy=["JUDGE"], tier="low", dev=1, verify=0.3, ref=0, risk=[], plan="Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training")

P = {
 # ---- A architecture
 "A-HYB-001": dict(proxy=["SYN", "MICRO"], tier="high", dev=8, verify=5, ref=70, risk=["scale_mismatch"], plan="Synthetic probes such as MQAR recall + μ1–μ3 language modeling, training context ≤ 4k; the throughput–context curve is inference only, measured up to 16k"),
 "A-LIN-002": dict(proxy=["SYN", "MICRO"], tier="high", dev=8, verify=5, ref=60, risk=["scale_mismatch"], plan="Shares probes and ladder with A-HYB-001; only the state-update rule of linear attention is replaced"),
 "A-MOE-003": dict(proxy=["MICRO"], tier="high", dev=8, verify=6, ref=60, risk=["variance", "scale_mismatch"], plan="Active parameters at μ2 scale, 8 experts, 3 seeds; load balance and expert utilization as constraint metrics"),
 "A-KV-004": dict(proxy=["SURG", "SYN"], tier="medium", dev=4, verify=2, ref=20, risk=[], plan="Replace attention in SmolLM2-135M with the candidate KV structure and continue training for 30M tokens; KV-byte and length-extrapolation probes"),
 "A-NORM-005": dict(proxy=["MICRO"], tier="medium", dev=3, verify=1.5, ref=15, risk=[], plan="μ1–μ2 ladder, count loss spikes"),
 "A-T7-006": J,
 # ---- B optimizers
 "B-OPT-001": dict(proxy=["MICRO", "STREAM"], tier="high", dev=8, verify=6, ref=74, risk=["scale_mismatch"], plan="μ1 tuning and long-training variants, μ2 public verification and large-batch variants, μ3 hidden extrapolation point; behavioral probes on a fixed gradient stream"),
 "B-OPT-002": dict(proxy=["STREAM", "MICRO"], tier="high", dev=8, verify=5.5, ref=0, risk=["scale_mismatch"], plan="Primitive search only on the fixed gradient stream and μ1; found methods reuse the ladder and reference pool of B-OPT-001 for final verification (originally estimated as medium, which underestimated the final-verification cost)"),
 "B-OPT-003": dict(proxy=["MICRO"], tier="high", dev=8, verify=5.5, ref=40, risk=[], plan="Reuse the B-OPT-001 ladder with optimizer-state memory capped at 1×; the reference pool recomputes only methods that satisfy the constraint"),
 "B-OPT-004": dict(proxy=["MICRO"], tier="medium", dev=3, verify=1.5, ref=15, risk=[], plan="Sweep 4 batch sizes at μ2, measure critical batch size and speedup"),
 "B-T7-005": {**J, "plan": "Judged against the reference results already precomputed for B-OPT-001; no new training"},
 # ---- C inference
 "C-SPEC-001": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=["memory"], plan="Target model ≤ 1.5B, draft ≤ 0.5B (HF weights); 5 workload types, including a low-acceptance workload"),
 "C-KV-002": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=[], plan="≤ 1.5B model, context up to 16k; compression-ratio–accuracy Pareto"),
 "C-QNT-003": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=[], plan="Weight quantization of 0.5B–1.5B models, 128 calibration samples"),
 "C-FA-004": dict(proxy=["STREAM"], tier="low", dev=1, verify=0.3, ref=2, risk=[], plan="Fixed-shape attention workloads, evaluation shared with K-FA-003"),
 "C-T7-005": J,
 # ---- V vision
 "V-GEN-001": dict(proxy=["SUBSET", "MICRO"], tier="medium", dev=4, verify=2, ref=20, risk=[], plan="Small diffusion model on CIFAR-10 32px, FID-10k"),
 "V-SMP-002": dict(proxy=["INFER"], tier="medium", dev=4, verify=1, ref=5, risk=[], plan="Fixed pretrained CIFAR / ImageNet-64 diffusion weights from HF; search only samplers and schedules; FID-5k during search, FID-50k at final verification"),
 "V-DIT-003": dict(proxy=["SUBSET", "MICRO"], tier="high", dev=8, verify=6, ref=60, risk=["scale_mismatch"], plan="100-class subset of ImageNet 64px, DiT-S class, pre-encoded latents; FID-10k"),
 "V-DST-004": dict(proxy=["INFER", "SUBSET"], tier="high", dev=8, verify=4, ref=20, risk=[], plan="Fixed pretrained HF teacher (CIFAR / ImageNet-64), train the student only; 1–4 step FID"),
 "V-3DG-005": dict(proxy=["SUBSET"], tier="medium", dev=3, verify=1, ref=10, risk=[], plan="6 small scenes, 3 input views, minutes of per-scene optimization, poses hidden"),
 "V-BB-006": dict(proxy=["MICRO"], tier="medium", dev=3, verify=1.5, ref=15, risk=[], plan="Imagenette 160px, 30 epochs"),
 "V-T7-007": J,
 # ---- E embodied
 "E-ACT-001": dict(proxy=["FROZEN", "SUBSET"], tier="high", dev=6, verify=4, ref=40, risk=["sim_throughput", "variance"], plan="Frozen ResNet-18 observation encoder, train only the action head; 2 LIBERO suites × 20 episodes per task, plus perturbed versions"),
 "E-RT-002": dict(proxy=["SUBSET"], tier="medium", dev=4, verify=2, ref=15, risk=["sim_throughput"], plan="Single LIBERO suite, evaluated under a 30 Hz latency cap"),
 "E-VLA-003": dict(proxy=["FROZEN"], tier="high", dev=8, verify=4, ref=30, risk=["memory", "sim_throughput"], plan="Switch to a small VLA of ≤ 0.5B (such as SmolVLA), freeze the VLM and only replace or fine-tune the action head; 7B OpenVLA fine-tuning is not in v1"),
 "E-T7-004": J,
 # ---- K kernels
 "K-KB-001": dict(proxy=["STREAM"], tier="medium", dev=3, verify=0.5, ref=5, risk=[], plan="30-problem subset of KernelBench Level 1, timed on the reference card, four hidden input distributions"),
 "K-FUS-002": dict(proxy=["STREAM"], tier="medium", dev=3, verify=0.5, ref=5, risk=[], plan="20-problem subset of Level 2 fusion problems, as above"),
 "K-FA-003": dict(proxy=["STREAM"], tier="low", dev=1, verify=0.3, ref=2, risk=[], plan="Fixed-shape attention workloads"),
 "K-T7-004": J,
 # ---- X training recipes / RL / PEFT
 "X-PPO-001": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=["variance"], plan="Small continuous-control environments, 5 seeds, fixed sample count"),
 "X-GRPO-002": dict(proxy=["SUBSET"], tier="high", dev=8, verify=6, ref=60, risk=["variance", "scale_mismatch"], plan="0.5B instruct model, verifiable tasks such as Countdown, rollouts ≤ 512 tokens, 200 steps, 3 seeds"),
 "X-MUP-003": dict(proxy=["MICRO"], tier="medium", dev=4, verify=2, ref=20, risk=[], plan="Tune at μ1, transfer to μ2 / μ3 and measure the gap"),
 "X-LORA-004": dict(proxy=["SURG"], tier="medium", dev=3, verify=1, ref=10, risk=[], plan="0.5B base, 3 downstream tasks, fixed trainable-parameter count"),
 # ---- T cross-domain
 "T-FM-001": dict(proxy=["FROZEN", "SUBSET"], tier="high", dev=6, verify=4, ref=30, risk=["sim_throughput"], plan="Same reduced setting as E-ACT-001"),
 "T-OPT-002": dict(proxy=["MICRO"], tier="medium", dev=4, verify=2, ref=20, risk=[], plan="Train a small ViT on Imagenette, reusing the reference implementations of track B"),
 "T-SPEC-003": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=[], plan="Fixed pretrained diffusion weights (same as V-SMP-002)"),
 # ---- U non-mainstream
 "U-LOOP-001": dict(proxy=["MICRO", "SYN"], tier="high", dev=8, verify=5, ref=50, risk=["scale_mismatch"], plan="Equal-parameter comparison at μ1–μ2 with ≤ 8 loops; only 1 hidden point at μ3"),
 "U-LOOP-002": dict(proxy=["SYN"], tier="medium", dev=4, verify=1, ref=10, risk=[], plan="Accuracy–test-time-compute Pareto on synthetic reasoning tasks such as arithmetic and graph reachability"),
 "U-LOOP-003": dict(proxy=["SYN"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="Synthetic algorithmic tasks, training length ≤ 20, tested up to 100"),
 "U-DLM-004": dict(proxy=["MICRO"], tier="high", dev=8, verify=6, ref=60, risk=["scale_mismatch"], plan="μ2–μ3 scale; compare how the gap to an equal-compute autoregressive baseline changes with scale"),
 "U-DLM-005": dict(proxy=["INFER"], tier="medium", dev=4, verify=2, ref=5, risk=["memory"], plan="Use ≤ 1B masked-diffusion LM weights from HF and change only sampling; LLaDA-8B serves only as literature background"),
 "U-DLM-006": dict(proxy=["MICRO"], tier="medium", dev=3, verify=1.5, ref=15, risk=[], plan="μ1–μ2 ladder"),
 "U-BYTE-007": dict(proxy=["SYN", "MICRO"], tier="medium", dev=4, verify=2, ref=20, risk=["scale_mismatch"], plan="Downgraded: measure only bits-per-byte and character-noise robustness at μ2; scaling claims are handled by a judgment question"),
 "U-TTT-008": dict(proxy=["SURG", "SYN"], tier="high", dev=8, verify=4, ref=40, risk=["scale_mismatch"], plan="Replace some layers of SmolLM2-135M with TTT layers and continue training for 50M tokens; long-context recall probes up to 8k"),
 "U-KAN-009": {**J, "tier": "low", "dev": 2, "verify": 0.5, "ref": 5, "proxy": ["JUDGE", "MICRO"], "plan": "Judgment question + small experiments on function fitting and MNIST-level tasks"},
 "U-T7-010": J,
 # ---- FN finance
 "FN-TS-001": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=["variance"], plan="Daily frequency, ≤ 500 assets, 5 walk-forward windows"),
 "FN-LOB-002": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=["variance"], plan="Public LOB data such as FI-2010, 3 walk-forward windows"),
 "FN-TAB-003": dict(proxy=["SUBSET"], tier="low", dev=0, verify=0, ref=0, cpu=True, risk=[], plan="CPU only, 5 credit-risk datasets"),
 "FN-QA-004": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=[], plan="≤ 1.5B base model; design only the retrieval and program-execution pipeline"),
 "FN-T7-005": J,
 # ---- LG law
 "LG-LONG-001": dict(proxy=["SUBSET"], tier="medium", dev=4, verify=1.5, ref=15, risk=[], plan="ECtHR subtasks, encoder ≤ 150M, maximum length 4k"),
 "LG-HOLD-002": dict(proxy=["SUBSET"], tier="low", dev=1, verify=0.3, ref=3, risk=[], plan="Small encoder fine-tuning"),
 "LG-RSN-003": dict(proxy=["INFER"], tier="medium", dev=3, verify=1, ref=5, risk=[], plan="≤ 1.5B base, 20 LegalBench subtasks"),
 # ---- PH physics
 "PH-NO-001": dict(proxy=["SUBSET"], tier="medium", dev=4, verify=1.5, ref=15, risk=[], plan="2 PDEBench equations each in 1D and 2D, 64² resolution"),
 "PH-WELL-002": dict(proxy=["SUBSET"], tier="high", dev=6, verify=4, ref=40, risk=["IO"], plan="Only 3 sub-datasets, downsampled to 64², total data < 200GB"),
 "PH-MLIP-003": dict(proxy=["SUBSET"], tier="medium", dev=4, verify=1.5, ref=15, risk=[], plan="Small datasets at the rMD17 / 3BPA level, MD stability measured over 10 ps"),
 "PH-T7-004": J,
 "PH-SYM-005": dict(proxy=["SUBSET"], tier="low", dev=0, verify=0, ref=0, cpu=True, risk=[], plan="CPU only, 10 simulated dynamical systems"),
 # ---- BI biology
 "BI-PROT-001": dict(proxy=["FROZEN"], tier="medium", dev=3, verify=1, ref=10, risk=[], plan="Frozen ESM-2 150M / 650M representations + lightweight head, no re-pretraining; 30 hidden DMS assays"),
 "BI-DNA-002": dict(proxy=["SURG", "SUBSET"], tier="medium", dev=4, verify=2, ref=20, risk=[], plan="Continue training or fine-tune a small HyenaDNA model from HF on a subset of genomic benchmarks"),
 "BI-T7-003": J,
 # ---- MD medicine
 "MD-SEG-001": dict(proxy=["SUBSET"], tier="high", dev=8, verify=4, ref=60, risk=["variance"], plan="3 small MSD tasks (Hippocampus, Prostate, Heart), 100 epochs, 1 fold, plus 1 hidden task; no 5-fold 1000-epoch runs"),
 "MD-T7-002": J,
 "MD-MNIST-003": dict(proxy=["SUBSET"], tier="low", dev=1, verify=0.3, ref=3, risk=[], plan="2 MedMNIST subsets each in 2D and 3D"),
 # ---- entry level
 "DS-GBM-001": dict(proxy=["SUBSET"], tier="low", dev=0, verify=0, ref=0, cpu=True, risk=[], plan="CPU only, 20 hidden OpenML datasets, each ≤ 50k rows"),
 "DS-PIPE-002": dict(proxy=["SUBSET"], tier="low", dev=0, verify=0, ref=0, cpu=True, risk=[], plan="CPU only, 1 hour per dataset"),
 "DS-T7-003": {**J, "dev": 0, "verify": 0, "cpu": True},
 "IC-SPD-001": dict(proxy=["MICRO"], tier="low", dev=1, verify=0.2, ref=2, risk=[], plan="airbench-level CIFAR-10 training, seconds to minutes per run, multiple seeds"),
 "IC-RES-002": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="Imagenette 128px, 20 epochs"),
 "IC-AUG-003": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="During search, candidate policies are evaluated with the CIFAR-10 speedrun recipe"),
 "TC-FT-001": dict(proxy=["SUBSET"], tier="low", dev=0, verify=0, ref=0, cpu=True, risk=[], plan="CPU only"),
 "TC-ENC-002": dict(proxy=["SUBSET"], tier="low", dev=1, verify=0.3, ref=3, risk=[], plan="≤ 110M encoder, 4 GLUE subtasks"),
 "TC-T7-003": {**J, "dev": 0, "verify": 0, "cpu": True},
 "GN-GCN-001": dict(proxy=["SUBSET"], tier="low", dev=0.5, verify=0.1, ref=1, risk=[], plan="Cora / CiteSeer / PubMed, 10 seeds"),
 "GN-OGB-002": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="ogbn-arxiv and ogbg-molhiv"),
 "GN-T7-003": J,
 "RS-SEQ-001": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="MovieLens-1M and similar, split by time, full ranking"),
 "RS-GRAPH-002": dict(proxy=["SUBSET"], tier="low", dev=2, verify=0.5, ref=5, risk=[], plan="Gowalla / Yelp-scale data, full ranking"),
 "RS-T7-003": J,
 # ---- Pro Set (in H100-h)
 "PRO-ARL-001": dict(proxy=["EVAL", "AUDIT"], tier="pro", unit="H100-h", dev=96, verify=12, ref=192, risk=["variance"], plan="4 × H100 × 24 h; hidden tool-task evaluation about 6 H100-h + 1-hour sampled resume (4 H100-h); reference recipes DAPO/GRPO run with 2 seeds"),
 "PRO-OPD-002": dict(proxy=["EVAL", "AUDIT"], tier="pro", unit="H100-h", dev=96, verify=10, ref=192, risk=["variance"], plan="Teacher inference and student training share the 4 GPUs; evaluation about 4 H100-h + sampled resume; references are one off-policy KD and one SFT+RL run"),
 "PRO-MOE-003": dict(proxy=["EVAL", "AUDIT"], tier="pro", unit="H100-h", dev=96, verify=8, ref=192, risk=["scale_mismatch"], plan="Evaluate validation loss and downstream tasks for about 2 H100-h + sampled resume; references are a DeepSeekMoE-style fine-grained recipe and an auxiliary-loss-free balancing recipe"),
 "PRO-SPD-004": dict(proxy=["RERUN"], tier="pro", unit="H100-h", dev=96, verify=16, ref=32, risk=[], plan="The benchmark reruns the final recipe in full twice (each ≤ 2 hours × 4 GPUs); the reference recipe takes about 4 hours"),
 "PRO-SRV-005": dict(proxy=["RERUN", "EVAL"], tier="pro", unit="H100-h", dev=96, verify=8, ref=16, risk=[], plan="The benchmark reruns twice on hidden traffic traces, about 1 hour × 4 GPUs each; references are tuned vLLM and SGLang configurations"),
 "PRO-SPEC-006": dict(proxy=["EVAL", "AUDIT"], tier="pro", unit="H100-h", dev=96, verify=6, ref=96, risk=[], plan="Evaluate end-to-end speedup for about 2 H100-h + sampled resume; the reference is the official EAGLE-3 training recipe"),
}


def flow(d: dict) -> str:
    items = [f"proxy: [{', '.join(d['proxy'])}]", f"dev: {d['dev']}", f"verify: {d['verify']}", f"ref: {d['ref']}"]
    if d.get("cpu"):
        items.append("cpu: true")
    if d.get("unit"):
        items.append(f"unit: {d['unit']}")
    items.append(f"risk: [{', '.join(d['risk'])}]")
    items.append(f'plan: "{d["plan"]}"')
    return "{" + ", ".join(items) + "}"


def difficulty(c: dict, group: str) -> str:
    """Difficulty: low for entry level and calibration, medium for main evaluation on mature anchors and judgment tasks, high for frontier, primitive search, stress, cross-domain and Pro."""
    if group == "pro":
        return "high"
    if group == "entry":
        return "medium" if c["subset"] == "primitive" else "low"
    if c["subset"] in ("primitive", "stress") or c["type"] == "TX":
        return "high"
    if c["type"] == "T7":
        return "medium"
    if c["subset"] == "calibration":
        return "low"
    if group == "frontier" or c["age"] == "frontier":
        return "high"
    return "medium"


def main():
    path = ROOT / "index.yaml"
    s = path.read_text(encoding="utf-8")
    data = yaml.safe_load(s)
    ids = [c["id"] for t in data["tracks"] for c in t["cases"]]
    case_of = {c["id"]: c for t in data["tracks"] for c in t["cases"]}
    group_of = {c["id"]: t["group"] for t in data["tracks"] for c in t["cases"]}
    missing = [i for i in ids if i not in P]
    assert not missing, missing
    for cid in ids:
        d = P[cid]
        cap = CAPS[d["tier"]]
        assert d["dev"] <= cap[0] and d["verify"] <= cap[1] and d["ref"] <= cap[2], (cid, d)
        start = s.index("{id: " + cid + ",")
        seg_end = s.index("\n      - {id:", start) if "\n      - {id:" in s[start:] else len(s)
        seg = s[start:seg_end]
        seg = re.sub(r"budget: (XS|S|M|PRO|low|medium|high|pro)\b", "budget: " + d["tier"], seg, count=1)
        seg = re.sub(r",?\s*difficulty: (low|medium|high)", "", seg)            # idempotent
        seg = seg.replace("{id: " + cid + ",", "{id: " + cid + ", difficulty: " + difficulty(case_of[cid], group_of[cid]) + ",", 1)
        seg = re.sub(r",?\s*verify_plan: \{[^{}]*\}", "", seg)            # idempotent
        seg = seg.replace("{id: " + cid + ",", "{id: " + cid + ", verify_plan: " + flow(d) + ",", 1)
        s = s[:start] + seg + s[seg_end:]
    path.write_text(s, encoding="utf-8")
    data = yaml.safe_load(s)
    cs = [c for t in data["tracks"] for c in t["cases"]]
    import collections
    print(collections.Counter(c["difficulty"] for c in cs), collections.Counter(c["budget"] for c in cs))
    print(len(cs), "cases;", "dev", sum(c["verify_plan"]["dev"] for c in cs),
          "verify", sum(c["verify_plan"]["verify"] for c in cs), "ref", sum(c["verify_plan"]["ref"] for c in cs))


if __name__ == "__main__":
    main()
