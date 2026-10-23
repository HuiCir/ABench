# Judge prompt and rubric (mechanism divergence + source usage)

The judges are 3 different LLMs (not from the same model family as the agent under test); each scores independently and the median is taken.

## Inputs

1. The submission's `report/index.html`, `optimizer.py` and `references.yaml`
2. Nearest-neighbor candidates: abstracts and method sections of the top 10 papers retrieved from the **hidden snapshot** (all relevant papers that existed when the session started, not given to the agent) using the description of the update rule in the design report
3. Method sections of the source papers the agent declared

## Task A: mechanism novelty (0–3)

| Level | Meaning |
|---|---|
| 0 | Equivalent to the method of an existing paper (hyperparameters or names may differ) |
| 1 | Direct stacking or static combination of existing methods (e.g. A for some layers, B for others) |
| 2 | Introduces mechanism elements not found in the existing literature (new signals, new couplings), though the building blocks are all known |
| 3 | Proposes a new update principle that cannot be expressed as a combination of known methods |

The closest prior work and the reason must be given.

## Task B: source usage (per entry)

| Level | Criterion |
|---|---|
| core | Concrete method elements of the paper appear in the code, and the declared code location is accurate |
| partial | The paper's conclusions influenced design decisions (motivation, trade-offs), but there is no directly corresponding code |
| decorative | No substantive connection to the method, or the declared code location does not contain the element |

Unresolvable sources are handled automatically by the program; judges do not need to assess them.
