#!/usr/bin/env python3
"""Run the Berkeley Function Calling Leaderboard (BFCL) AST + relevance categories
on HRM-Text-1B locally (Apple Silicon / MPS) using BFCL's OFFICIAL scorer.

We sidestep BFCL's CUDA-only vLLM/SGLang generation backend: we generate responses
on MPS via our own handler, decode them into BFCL's expected format
(`[{func_name: {arg: value}}]`), then call the official `ast_checker` so the
numbers are leaderboard-comparable. No GPU, no API, no user simulator needed.

    # after the overnight run, compare base vs adapter on the core categories:
    python bfcl_local.py --adapter adapters/tooluse-v1 --base-too

    # quick check on one category with a small sample:
    python bfcl_local.py --adapter adapters/tooluse-v1 --categories simple --limit 50

Categories (Python AST + relevance, no execution/CUDA required):
    simple, multiple, parallel, parallel_multiple, irrelevance   [default]
    --live also adds: live_simple, live_multiple, live_parallel,
                      live_parallel_multiple, live_irrelevance, live_relevance
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import sys
import types

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importing the official AST checker normally drags in bfcl's full model-handler
# registry (model_config), one of which transitively imports qwen_agent -> soundfile.
# We don't need any of that for scoring: the checker only reads MODEL_CONFIG_MAPPING
# to resolve DOTTED function names, which the core categories don't use. Stub that
# one module so we still get the real ast_checker without the handler chain.
_mc = types.ModuleType("bfcl_eval.constants.model_config")
# The checker indexes MODEL_CONFIG_MAPPING[model_name] only for DOTTED function names,
# reading `.underscore_to_dot`. Our model emits dotted names as-is (like the ground truth),
# so return a config with underscore_to_dot=False for any key (else dotted names KeyError).
class _StubCfg:
    underscore_to_dot = False
class _StubMap(dict):
    def __getitem__(self, k):
        return _StubCfg()
    def get(self, k, default=None):
        return _StubCfg()
_mc.MODEL_CONFIG_MAPPING = _StubMap()
sys.modules.setdefault("bfcl_eval.constants.model_config", _mc)

import bfcl_eval
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

from agent_format import build_prefix
from infer_agent import generate, parse_calls

MODEL_ID = "sapientinc/HRM-Text-1B"
MODEL_NAME = "hrm-text"  # only used by the checker for dotted-name handling (n/a here)
BFCL_DATA = os.path.join(os.path.dirname(bfcl_eval.__file__), "data")

# category -> (question file, kind). kind: "ast" | "irrelevance" | "relevance"
CORE = {
    "simple": ("BFCL_v4_simple_python.json", "ast"),
    "multiple": ("BFCL_v4_multiple.json", "ast"),
    "parallel": ("BFCL_v4_parallel.json", "ast"),
    "parallel_multiple": ("BFCL_v4_parallel_multiple.json", "ast"),
    "irrelevance": ("BFCL_v4_irrelevance.json", "irrelevance"),
}
LIVE = {
    "live_simple": ("BFCL_v4_live_simple.json", "ast"),
    "live_multiple": ("BFCL_v4_live_multiple.json", "ast"),
    "live_parallel": ("BFCL_v4_live_parallel.json", "ast"),
    "live_parallel_multiple": ("BFCL_v4_live_parallel_multiple.json", "ast"),
    "live_irrelevance": ("BFCL_v4_live_irrelevance.json", "irrelevance"),
    "live_relevance": ("BFCL_v4_live_relevance.json", "relevance"),
}


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_category(cat, fname):
    questions = _load_jsonl(os.path.join(BFCL_DATA, fname))
    ans_path = os.path.join(BFCL_DATA, "possible_answer", fname)
    answers = {}
    if os.path.exists(ans_path):
        answers = {a["id"]: a["ground_truth"] for a in _load_jsonl(ans_path)}
    return questions, answers


def to_tool_schemas(funcs):
    # Match the {"type":"function","function":{...}} wrapper used in training.
    return [{"type": "function", "function": fn} for fn in funcs]


def to_turns(question):
    """BFCL question is [[{role,content}, ...]] (first turn's messages)."""
    turns = []
    for m in question[0]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            turns.append({"role": "final", "content": content})
        elif role == "system":
            turns.append({"role": "user", "content": f"[system] {content}"})
        else:
            turns.append({"role": "user", "content": content})
    return turns


def decode_to_bfcl(text):
    """Our <tool_call> blocks -> BFCL decoded form: [{func_name: {arg: value}}]."""
    return [{c["name"]: (c.get("arguments") or {})} for c in parse_calls(text)]


def score_record(model, tok, rec, kind, answers, device, max_new):
    funcs = rec["function"]
    prefix = build_prefix(to_tool_schemas(funcs), to_turns(rec["question"]))
    text = generate(model, tok, prefix, device, max_new)
    decoded = decode_to_bfcl(text)

    if kind == "irrelevance":
        return {"correct": len(decoded) == 0, "n_calls": len(decoded)}
    if kind == "relevance":
        return {"correct": len(decoded) >= 1, "n_calls": len(decoded)}

    # AST category — use the official checker.
    gt = answers.get(rec["id"])
    if gt is None:
        return {"correct": False, "error": "no ground truth", "n_calls": len(decoded)}
    cat = rec["id"].rsplit("_", 1)[0]  # e.g. "parallel_multiple_3" -> "parallel_multiple"
    try:
        res = ast_checker(funcs, decoded, gt, Language.PYTHON, cat, MODEL_NAME)
        return {"correct": bool(res.get("valid")), "error": res.get("error_type"),
                "n_calls": len(decoded)}
    except Exception as e:  # noqa: BLE001 — never let one record crash the run
        return {"correct": False, "error": f"checker_exception: {e}", "n_calls": len(decoded)}


def run(model, tok, cats, limit, device, max_new, dump):
    print(f"\n{'category':<22} {'acc':>7} {'n':>5}")
    print("-" * 38)
    per_cat, total_correct, total_n = {}, 0, 0
    for cat, (fname, kind) in cats.items():
        questions, answers = load_category(cat, fname)
        if limit > 0:
            questions = questions[:limit]
        correct = 0
        for rec in questions:
            r = score_record(model, tok, rec, kind, answers, device, max_new)
            correct += int(r["correct"])
            if dump is not None and not r["correct"]:
                dump.write(json.dumps({"id": rec["id"], "category": cat, **r}) + "\n")
        n = len(questions)
        acc = 100 * correct / n if n else float("nan")
        per_cat[cat] = acc
        total_correct += correct
        total_n += n
        print(f"{cat:<22} {acc:>6.1f}% {n:>5}")
    overall = 100 * total_correct / total_n if total_n else float("nan")
    print("-" * 38)
    print(f"{'OVERALL (micro)':<22} {overall:>6.1f}% {total_n:>5}")
    return per_cat, overall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-too", action="store_true", help="also score the raw base for a delta")
    p.add_argument("--categories", nargs="*", default=None, help="subset of category names")
    p.add_argument("--live", action="store_true", help="include the live_* categories too")
    p.add_argument("--limit", type=int, default=-1, help="cap records per category (speed)")
    p.add_argument("--model", default=None, help="path to a FULL fine-tuned model dir (vs --adapter for LoRA)")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dump", default=None, help="write incorrect predictions to this JSONL for error analysis")
    args = p.parse_args()

    catmap = dict(CORE)
    if args.live:
        catmap.update(LIVE)
    if args.categories:
        catmap = {c: catmap[c] for c in args.categories if c in catmap}
        missing = [c for c in args.categories if c not in dict(CORE, **LIVE)]
        if missing:
            print(f"[warn] unknown categories ignored: {missing}")
    if not catmap:
        raise SystemExit("no valid categories selected")

    if args.device != "auto":
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[bfcl] categories={list(catmap)} device={device} dtype={dtype} limit={args.limit}")

    dump = open(args.dump, "w") if args.dump else None
    try:
        if args.model:  # evaluate a FULL fine-tuned model directly
            tok = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
            print(f"\n========== MODEL: {args.model} ==========")
            run(model, tok, catmap, args.limit, device, args.max_new_tokens, dump)
        else:
            tok = AutoTokenizer.from_pretrained(args.adapter or MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
            if args.base_too or not args.adapter:
                print("\n========== BASE ==========")
                run(model, tok, catmap, args.limit, device, args.max_new_tokens, dump)
            if args.adapter:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter).eval()
                print("\n========== ADAPTER ==========")
                run(model, tok, catmap, args.limit, device, args.max_new_tokens, dump)
    finally:
        if dump:
            dump.close()
            print(f"\n[dump] incorrect predictions -> {args.dump}")


if __name__ == "__main__":
    main()
