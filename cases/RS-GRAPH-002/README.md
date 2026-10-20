# ABench Case RS-GRAPH-002: Collaborative-filtering model design

This is a bench scaffold for the RS track, generated from the task card in index.yaml. Only the interface, resource contract and evaluation protocol are frozen so far; no training curves, reference scores or simulated results are provided.

- Anchor: LightGCN (arXiv: 2002.02126)
- Subset: main; difficulty: low; resource tier: low
- Task: Collaborative-filtering model design
- Metrics: Full-ranking Recall@20 and NDCG@20
- Verification plan: Gowalla / Yelp-scale data, full ranking

## Directory

- case/: anchor card, task prompt, resource list, evaluation protocol, scoring configuration and anti-gaming rules.
- harness/: notes on the submission interface and the server run contract; no fake runner is included.
- scoring/: input/output conventions for real run artifacts; no scores are generated.
- example_submission/: describes only the submission file format and contains no simulated results.

## Must be completed before formal runs

1. Freeze and hash data, model/vocabulary, software environment and hardware.
2. Produce the benchmark operator's real reference runs and the hidden evaluation set.
3. Map Full-ranking Recall@20 and NDCG@20 to reproducible raw metrics with statistical confidence intervals.
4. Complete the leak-prevention, sandbox, timeout, memory and network policies.
5. Calibrate the fidelity of the proxy plan; if fidelity is insufficient, downgrade to calibration or postpone launch.

Complex or not-yet-verified resources stay in case/resources.yaml; this scaffold does not assume they exist.
