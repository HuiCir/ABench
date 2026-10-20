# Definition and scoring of discovery in ABench (v2026.07)

## 1. Definition

Every design task in ABench requires some degree of architectural innovation. "Discovery" means not only searching out a new algorithm from scratch, but also **local algorithmic discovery**:

> Diagnose a concrete problem of the current model, find a known result in another field or earlier research that solves it, build it into a local module of the model, and show through ablation that the module actually brings a gain.

The building blocks of such a discovery need not be original, but "applying this result to this problem" must not have appeared in the ML literature before the cutoff, and the source must be correctly traced. It is scored the same as an original discovery.

### Reference case: DeepSeek mHC (arXiv 2512.24880)

| Step | Content |
|---|---|
| Diagnosed problem | The residual mixing matrix of Hyper-Connections is unconstrained and breaks the identity mapping; the peak signal gain approaches 3000 and training becomes unstable |
| Known result brought in | The Sinkhorn–Knopp algorithm (1967): projects a matrix entropically onto the Birkhoff polytope, giving a doubly stochastic matrix |
| Why it works | A doubly stochastic matrix forms convex combinations of the inputs, preserving the mean and bounding the norm; doubly stochastic matrices are closed under multiplication, so the property holds when composed across layers |
| Module built | A manifold-constrained residual mixing layer |

## 2. Distinction from existing concepts

| Situation | Classification | Scoring |
|---|---|---|
| The result brought in was already used in the ML literature for the same problem before the cutoff | retrieval reuse | counts only toward reference quality Q, no discovery score |
| Already used in ML, but for a different problem | transfer discovery | discovery score × 0.5 |
| This use not found in the ML literature before the cutoff | local discovery | full discovery score |
| A source is declared but the module contributes nothing to the result (no difference in ablation) | decorative import | not scored |
| The declared source does not contain the transferred idea | incorrect provenance | not scored, and Q is penalized as for an unresolvable source |

Method divergence D measures "difference from the reference methods", not whether building blocks are original, so a local discovery is not penalized in D because its building blocks are known.

## 3. Submission

Declare each claim in `submission/discoveries.yaml` (at most 3 are scored):

```yaml
discoveries:
  - module: module name
    problem: the diagnosed problem and the evidence supporting the diagnosis (L1 experiments, literature conclusions)
    source: {id: paper identifier, year: year, field: source field}
    transferred: the specific result or algorithm transferred
    code: corresponding code location
    ablation_fallback: what replaces the module in the ablation (must be a runnable trivial substitute)
```

## 4. Scoring

For claim i:

- **Effectiveness a_i**: the benchmark operator retrains in the L2 setting with the module replaced by `ablation_fallback`. e_i = ln(S_full / S_ablated); e_i must exceed z·σ (seed noise), otherwise it is judged a decorative import. a_i = clip(e_i / ln 1.05, 0, 1), i.e. a module that contributes 5% effective speedup on its own earns full marks.
- **Application novelty n_i**: judges search the hidden snapshot for existing uses of the source in ML; the value is 1 / 0.5 / 0 (see Section 2).
- **Provenance distance r_i**: distance between the source field and the task field. Same subfield 0, other ML directions 0.3, other computer-science directions 0.6, other disciplines such as mathematics, physics, biology and economics 1.0.
- **Provenance correctness c_i**: 1 if the source actually contains the transferred result, otherwise 0.

Per-claim score x_i = a_i · n_i · c_i · (0.5 + 0.5·r_i); discovery score X = 1 − Π(1 − x_i).

Overall score:

```
Score = 100 · 1[thresholds] · M^0.5 · D^0.3 · Q^0.2 · (1 + 0.25·X)
```

The discovery score is a multiplicative bonus rather than a fourth geometric-mean dimension: a method without local discoveries is not zeroed, and effective local discoveries add at most 25%.

## 5. Weights of cross-field sources

Local discoveries often cite old mathematics or literature from other disciplines, which would be undervalued on ML's venue-tier and citation scales. Therefore:

- venue tier is assessed in the source's own field (for example Pacific J. Math, J. ACM and SIAM journals in mathematics)
- the citation percentile is computed within a "same field × same era" cohort, not ranked together with ML papers

## 6. Anti-gaming

| Gaming method | Protection |
|---|---|
| Dressing an idea from a recent ML paper in an old mathematical source to pass it off as a discovery | Judges search the hidden snapshot for ML precedents of the use; any precedent downgrades it to retrieval reuse |
| Declaring many sources when only one or two matter | Each claim is ablated separately; at most 3 are counted |
| Choosing a deliberately poor substitute in ablation | `ablation_fallback` must be confirmed by the judges as a reasonable trivial substitute (constant, identity, existing standard component) |
| Spreading the same idea over several modules to count it repeatedly | Judges merge claims from the same source |
