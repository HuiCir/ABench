"""FN-QA-004 reference submission: retrieve evidence, prompt the frozen model with nearest-neighbour exemplars to write a
FinQA program, execute it.

For each question: the 8 facts (table rows and sentences) most similar to the question by TF-IDF; the 4 training
questions most similar to it, each with its gold evidence and program, as few-shot examples; greedy decoding of one
program line (at most 96 new tokens); execution with a local copy of the case's executor. An invalid program gives no
answer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import finqa_exec as F  # noqa: E402
from src import retrieve as R  # noqa: E402

MODEL = "Qwen2.5-1.5B-Instruct"
INSTR = ("Write a program that answers the question from the facts. Use only these operations: add, subtract, "
         "multiply, divide, exp, greater, table_max, table_min, table_sum, table_average. Arguments are numbers from "
         "the facts, constants such as const_100, or #i for the result of step i; separate steps with commas. "
         "Answer with the program only, on one line.")


def build_prompt(item: dict, idf: dict, bank: list) -> str:
    shots = R.nearest(item["question"], bank, idf, 4)
    parts = [INSTR, ""]
    for e in shots:
        parts += ["Facts:"] + [f"- {f}" for f in e["facts"]] + [f"Question: {e['question']}", f"Program: {e['program']}", ""]
    parts += ["Facts:"] + [f"- {s}" for _, s in R.top_facts(item, idf, 8)] + [f"Question: {item['question']}", "Program:"]
    return "\n".join(parts)


def first_program(text: str) -> str:
    line = text.strip().splitlines()[0] if text.strip() else ""
    return line.replace("Program:", "").strip()


def load(checkpoint_dir: str, models_dir: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = Path(checkpoint_dir)
    path = str(Path(models_dir) / MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device).eval()
    return {"tok": tok, "model": model, "device": device,
            "idf": json.loads((ck / "idf.json").read_text()), "bank": json.loads((ck / "exemplars.json").read_text())}


def generate(state, prompts: list) -> list:
    import torch
    tok, model = state["tok"], state["model"]
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(chats, return_tensors="pt", padding=True).to(state["device"])
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def answer(state, items: list) -> list:
    prompts = [build_prompt(it, state["idf"], state["bank"]) for it in items]
    texts = state.get("generate", generate)(state, prompts)
    out = []
    for it, t in zip(items, texts):
        prog = first_program(t)
        val = F.execute(prog, it["table"]) if prog else None
        out.append({"program": prog, "answer": val})
    return out
