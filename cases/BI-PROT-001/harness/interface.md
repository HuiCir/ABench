# BI-PROT-001 interface

## Entry point: `model.py`

```python
def load(checkpoint_dir: str, models_dir: str, device: str):
    """models_dir holds esm2_t30_150M_UR50D/ and esm2_t33_650M_UR50D/ (Hugging Face format, frozen)."""

def predict(model, assay: dict) -> numpy.ndarray:
    """assay = {"wild_type": str, "mutants": [str], "few_shot_mutants": [str] (64), "few_shot_scores": float (64,),
    "selection_type": str}. Return one score per mutant, higher = fitter."""
```

- Mutants are single substitutions "A23G": wild-type residue, 1-based position, new residue.
- `model.py` may import modules under `src/`; the submission directory is on `sys.path`.

## Evaluator code you can read

`stages.py` (the stage and the dev setting), `metrics.py` (Spearman per assay and by selection type),
`abench_metrics.py`, `abench_stage.py`, `selfcheck.py`, and the reference submission's code under `baseline/`.

## Available software

Python 3, PyTorch (CUDA, Triton), transformers, NumPy, SciPy, scikit-learn. Nothing can be installed.
