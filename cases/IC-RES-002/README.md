# ABench Case IC-RES-002: Image classification backbone for small data (Imagenette subset, calibration)

This is a bench scaffold for the IC track, generated from the task card in index.yaml. Only the interface, resource contract and evaluation protocol are frozen so far; no training curves, reference scores or simulated results are provided.

- Anchor: ResNet (arXiv: 1512.03385)
- Subset: calibration; difficulty: low; resource tier: low
- Task: Image classification backbone for small data (Imagenette subset, calibration)
- Metrics: Accuracy–FLOPs Pareto
- Verification plan: Imagenette 128px, 20 epochs

## Directory

- case/: anchor card, task prompt, resource list, evaluation protocol, scoring configuration and anti-gaming rules.
- harness/: notes on the submission interface and the server run contract; no fake runner is included.
- scoring/: input/output conventions for real run artifacts; no scores are generated.
- example_submission/: describes only the submission file format and contains no simulated results.

## Must be completed before formal runs

1. Freeze and hash data, model/vocabulary, software environment and hardware.
2. Produce the benchmark operator's real reference runs and the hidden evaluation set.
3. Map Accuracy–FLOPs Pareto to reproducible raw metrics with statistical confidence intervals.
4. Complete the leak-prevention, sandbox, timeout, memory and network policies.
5. Calibrate the fidelity of the proxy plan; if fidelity is insufficient, downgrade to calibration or postpone launch.

Complex or not-yet-verified resources stay in case/resources.yaml; this scaffold does not assume they exist.
