# ABench Case DS-T7-003: Judge whether the gains of a given tabular neural network exceed a tuned GBDT

This is a bench scaffold for the DS track, generated from the task card in index.yaml. Only the interface, resource contract and evaluation protocol are frozen so far; no training curves, reference scores or simulated results are provided.

- Anchor: Why do tree-based models still outperform deep learning on tabular data? (arXiv: 2207.08815)
- Subset: calibration; difficulty: low; resource tier: low
- Task: Judge whether the gains of a given tabular neural network exceed a tuned GBDT
- Metrics: Judgment accuracy
- Verification plan: Judgment question + one minimal verification experiment at μ1 or equivalent scale; no new large-scale training

## Directory

- case/: anchor card, task prompt, resource list, evaluation protocol, scoring configuration and anti-gaming rules.
- harness/: notes on the submission interface and the server run contract; no fake runner is included.
- scoring/: input/output conventions for real run artifacts; no scores are generated.
- example_submission/: describes only the submission file format and contains no simulated results.

## Must be completed before formal runs

1. Freeze and hash data, model/vocabulary, software environment and hardware.
2. Produce the benchmark operator's real reference runs and the hidden evaluation set.
3. Map Judgment accuracy to reproducible raw metrics with statistical confidence intervals.
4. Complete the leak-prevention, sandbox, timeout, memory and network policies.
5. Calibrate the fidelity of the proxy plan; if fidelity is insufficient, downgrade to calibration or postpone launch.

Complex or not-yet-verified resources stay in case/resources.yaml; this scaffold does not assume they exist.
