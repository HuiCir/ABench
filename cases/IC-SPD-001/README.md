# ABench Case IC-SPD-001: Design a network and training recipe that reaches 94% CIFAR-10 accuracy in the shortest time

This is a bench scaffold for the IC track, generated from the task card in index.yaml. Only the interface, resource contract and evaluation protocol are frozen so far; no training curves, reference scores or simulated results are provided.

- Anchor: 94% on CIFAR-10 in 3.29 seconds (airbench) (arXiv: 2404.00498)
- Subset: main; difficulty: low; resource tier: low
- Task: Design a network and training recipe that reaches 94% CIFAR-10 accuracy in the shortest time
- Metrics: Wall-clock time to reach 94%; CIFAR-10.1 accuracy
- Verification plan: airbench-level CIFAR-10 training, seconds to minutes per run, multiple seeds

## Directory

- case/: anchor card, task prompt, resource list, evaluation protocol, scoring configuration and anti-gaming rules.
- harness/: notes on the submission interface and the server run contract; no fake runner is included.
- scoring/: input/output conventions for real run artifacts; no scores are generated.
- example_submission/: describes only the submission file format and contains no simulated results.

## Must be completed before formal runs

1. Freeze and hash data, model/vocabulary, software environment and hardware.
2. Produce the benchmark operator's real reference runs and the hidden evaluation set.
3. Map Wall-clock time to reach 94%; CIFAR-10.1 accuracy to reproducible raw metrics with statistical confidence intervals.
4. Complete the leak-prevention, sandbox, timeout, memory and network policies.
5. Calibrate the fidelity of the proxy plan; if fidelity is insufficient, downgrade to calibration or postpone launch.

Complex or not-yet-verified resources stay in case/resources.yaml; this scaffold does not assume they exist.
