"""Math answers with a Python tool (evaluator code, agent-visible; shared by PRO-ARL-001 and PRO-OPD-002).

Conversation: the system prompt SYSTEM and the problem as the user turn, in the model's chat template (with
thinking enabled for Qwen3 templates). The model may run Python by writing <python>...</python> (only when the case
allows tools); the evaluator stops generation at </python>, runs the code (run_python: a fresh isolated interpreter,
10 s of CPU, 2 GB of memory, no network, stdout and stderr truncated to 2,000 characters) and appends
"<output>...</output>" before generation continues; at most MAX_CALLS calls and MAX_TOKENS generated tokens per
answer, tool outputs included.

Answer: the content of the last \\boxed{...} in the response after the last tool output; for integer-answer
problems (AIME) it is correct if it parses to the gold integer (commas, spaces, $, \\text{} and a trailing period are
ignored; "042" equals 42).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

SYSTEM_TOOLS = ("Solve the problem. You may run Python code by writing it between <python> and </python>; its output "
                "will be shown between <output> and </output>. Give the final answer as \\boxed{...}.")
SYSTEM_PLAIN = "Solve the problem. Give the final answer as \\boxed{...}."
MAX_CALLS, MAX_TOKENS, TOOL_SECONDS, TOOL_CHARS = 8, 16384, 10, 2000
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def to_int(ans: str | None):
    if ans is None:
        return None
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", ans)
    s = s.replace("$", "").replace(",", "").replace("\\!", "").replace(" ", "").strip().rstrip(".")
    if re.fullmatch(r"[-+]?\d+", s):
        return int(s)
    if re.fullmatch(r"[-+]?\d+\.0*", s):
        return int(float(s))
    return None


def final_text(response: str) -> str:
    """The part of a response after its last tool output."""
    k = response.rfind("</output>")
    return response[k + len("</output>"):] if k >= 0 else response


def correct_int(response: str, gold: int) -> bool:
    return to_int(extract_boxed(final_text(response))) == int(gold)


def run_python(code: str) -> str:
    """Run code in a fresh isolated interpreter with CPU, memory and output limits."""
    with tempfile.TemporaryDirectory(prefix="abench-tool-") as d:
        p = os.path.join(d, "t.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(code)

        def limits():
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (TOOL_SECONDS, TOOL_SECONDS))
            resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
            os.setsid()
        try:
            r = subprocess.run([sys.executable, "-I", p], cwd=d, capture_output=True, text=True,
                               timeout=TOOL_SECONDS + 5, preexec_fn=limits,
                               env={"PATH": "/usr/bin:/bin", "HOME": d, "OMP_NUM_THREADS": "1"})
            out = (r.stdout + r.stderr) or (f"killed by the CPU or memory limit (exit {r.returncode})" if r.returncode < 0 else "")
        except subprocess.TimeoutExpired:
            out = "TimeoutError: the code ran for more than 10 seconds"
    return out[:TOOL_CHARS]


def chat_prompt(tokenizer, problem: str, tools: bool) -> str:
    msgs = [{"role": "system", "content": SYSTEM_TOOLS if tools else SYSTEM_PLAIN}, {"role": "user", "content": problem}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(llm, tokenizer, problems: list, n: int, tools: bool, seed: int, max_tokens: int = MAX_TOKENS) -> list:
    """n sampled responses per problem with vLLM (llm: vllm.LLM); returns a list of lists of response strings."""
    from vllm import SamplingParams
    convs = [{"pid": i, "k": k, "prompt": chat_prompt(tokenizer, p, tools), "resp": "", "used": 0, "calls": 0}
             for i, p in enumerate(problems) for k in range(n)]
    active = list(range(len(convs)))
    while active:
        params = []
        for j in active:
            c = convs[j]
            params.append(SamplingParams(max_tokens=max(1, max_tokens - c["used"]), seed=seed * 1_000_003 + j * 7919 + c["calls"],
                                         stop=["</python>"] if tools and c["calls"] < MAX_CALLS else None,
                                         include_stop_str_in_output=True, **SAMPLING))
        outs = llm.generate([convs[j]["prompt"] + convs[j]["resp"] for j in active], params, use_tqdm=False)
        nxt = []
        for j, o in zip(active, outs):
            c, g = convs[j], o.outputs[0]
            c["resp"] += g.text
            c["used"] += len(g.token_ids)
            if tools and g.text.endswith("</python>") and c["calls"] < MAX_CALLS and c["used"] < max_tokens:
                code = g.text[g.text.rfind("<python>") + len("<python>"):-len("</python>")] if "<python>" in g.text else ""
                res = run_python(code)
                c["resp"] += f"\n<output>\n{res}\n</output>\n"
                c["used"] += len(tokenizer(f"\n<output>\n{res}\n</output>\n")["input_ids"])
                c["calls"] += 1
                if c["used"] < max_tokens:
                    nxt.append(j)
        active = nxt
    res = [[""] * n for _ in problems]
    for c in convs:
        res[c["pid"]][c["k"]] = c["resp"]
    return res
