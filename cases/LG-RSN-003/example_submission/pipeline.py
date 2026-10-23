"""LG-RSN-003 reference submission: LegalBench base prompt plus constrained label choice by likelihood.

For each row the task's base prompt is filled with the row (instructions, the task's few-shot examples, the query)
and every label is scored as a continuation " <label>" by the frozen model's mean log-probability per token; the
best-scoring label is the answer. No chat template is used: the base prompts are written for completion.
"""
from __future__ import annotations

import re
from pathlib import Path

MODEL = "Qwen2.5-1.5B-Instruct"
SLOT = re.compile(r"\{\{(.*?)\}\}")
MAX_PROMPT = 3000                           # tokens kept from the end of a long prompt


def fill(template: str, row: dict) -> str:
    return SLOT.sub(lambda m: str(row.get(m.group(1), "")), template)


def load(checkpoint_dir: str, models_dir: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = str(Path(models_dir) / MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    return {"tok": tok, "model": model, "device": device}


def label_scores(state, prompt: str, labels: list) -> list:
    """Mean log-probability per token of each ' <label>' after the prompt (one forward pass per label)."""
    import torch
    tok, model, dev = state["tok"], state["model"], state["device"]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"][-MAX_PROMPT:]
    with torch.no_grad():
        out = model(torch.tensor([p_ids], device=dev), use_cache=True)
        past, last = out.past_key_values, out.logits[0, -1].float().log_softmax(-1)
        scores = []
        for lab in labels:
            l_ids = tok(" " + lab, add_special_tokens=False)["input_ids"]
            lp = [last[l_ids[0]].item()]
            if len(l_ids) > 1:
                import copy
                o = model(torch.tensor([l_ids[:-1]], device=dev), past_key_values=copy.deepcopy(past), use_cache=True)
                lps = o.logits[0].float().log_softmax(-1)
                lp += [lps[i, l_ids[i + 1]].item() for i in range(len(l_ids) - 1)]
            scores.append(sum(lp) / len(lp))
    return scores


def predict(state, task: dict, rows: list) -> list:
    out = []
    for r in rows:
        prompt = fill(task["prompt_template"], r)
        s = state.get("score", label_scores)(state, prompt, task["labels"])
        out.append(task["labels"][max(range(len(s)), key=s.__getitem__)])
    return out
