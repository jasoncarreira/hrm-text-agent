#!/usr/bin/env python3
"""Code benchmarks for HRM-Text: HumanEval and MBPP, pass@1 (greedy).

Why these and NOT SWE-bench: SWE-bench is repo-scale agentic (whole codebases, multi-file
patches, a Docker test harness) needing 10k-100k-token context — a 1B model with a 4k
window scores ~0 and tells you nothing about whether training helped. HumanEval/MBPP are
self-contained function-level tasks with unit tests: cheap, and the pass@1 number actually
MOVES with training, so it measures the experiment. (Rigor upgrade: EvalPlus's HumanEval+/
MBPP+ add many more tests.)

Generates in the synth,cot lane (how the code expert is trained), extracts code, and runs
it against the tests in a subprocess with a timeout. WARNING: executes model-generated
code — use a disposable pod.

  python eval_code.py --bench humaneval --model models/hrm-code --out he_code.json
  python eval_code.py --bench mbpp      --model models/hrm-code --out mbpp_code.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import BOX_END, IM_END, IM_START

SYNTH_COT = "<|quad_end|><|object_ref_end|>"
HEADER = ("from typing import *\nimport math, re, collections, itertools, functools, "
          "heapq, bisect, string, operator\n")


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(model, tok, instruction, device, max_new):
    prompt = f"{IM_START}{SYNTH_COT}{instruction}{IM_END}"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    tti = torch.ones_like(ids["input_ids"])
    eos = tok.convert_tokens_to_ids(BOX_END)
    with torch.no_grad():
        out = model.generate(**ids, token_type_ids=tti, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=eos,
                             pad_token_id=(tok.pad_token_id if tok.pad_token_id is not None else eos))
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def extract(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return m.group(1) if m else text


def run_prog(prog, timeout=12):
    f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    f.write(prog)
    f.close()
    try:
        return subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout).returncode == 0
    except Exception:  # noqa: BLE001
        return False
    finally:
        os.unlink(f.name)


def humaneval_items(limit):
    ds = load_dataset("openai/openai_humaneval", split="test")
    n = len(ds) if limit < 0 else min(limit, len(ds))
    items = []
    for i in range(n):
        r = ds[i]
        instr = ("Complete the following Python function. Reply with the full function only.\n\n"
                 + r["prompt"])

        def build(code, r=r):
            cand = code if f"def {r['entry_point']}" in code else r["prompt"] + code
            return HEADER + "\n" + cand + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
        items.append((instr, build))
    return items


def mbpp_items(limit):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    n = len(ds) if limit < 0 else min(limit, len(ds))
    items = []
    for i in range(n):
        r = ds[i]
        tests = r["test_list"]
        setup = "\n".join(r.get("test_imports", []) or [])
        instr = f"{r['prompt']}\nWrite a Python function that satisfies:\n" + "\n".join(tests)

        def build(code, tests=tests, setup=setup):
            return HEADER + setup + "\n" + code + "\n\n" + "\n".join(tests) + "\n"
        items.append((instr, build))
    return items


BENCHES = {"humaneval": humaneval_items, "mbpp": mbpp_items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=list(BENCHES), default="humaneval")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    items = BENCHES[args.bench](args.limit)
    n, npass = len(items), 0
    for i, (instr, build) in enumerate(items):
        out = generate(model, tok, instr, device, args.max_new_tokens)
        npass += int(run_prog(build(extract(out))))
        if (i + 1) % 20 == 0:
            print(f"  {args.bench} {i+1}/{n} pass@1~{npass/(i+1):.3f}", flush=True)
    acc = npass / max(1, n)
    print(f"[{args.bench}] model={args.model} pass@1={acc:.3f} ({npass}/{n})")
    if args.out:
        json.dump({"bench": args.bench, "model": args.model, "pass@1": acc, "passed": npass, "n": n},
                  open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
