#!/usr/bin/env python3
"""Score a (LoRA-adapted or base) HRM-Text-1B on held-out tool-use prompts.

Evaluates the model's FIRST action given the user query — the cleanest signal:
  * gold wants a tool call  -> did it emit valid JSON? right tool? required args? exact match?
  * gold wants no tool       -> did it correctly answer directly (abstain)?

    # after the overnight run, compare base vs adapter on the held-out set:
    python eval_agent.py --adapter adapters/tooluse-v1 --base-too

    # adapter only, more examples:
    python eval_agent.py --adapter adapters/tooluse-v1 --limit 200

Metrics are reported separately for held-out Hermes scenarios and for a handful
of local-tool prompts (which the adapter never saw — tests format transfer).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import build_prefix, TOOL_OPEN
from infer_agent import generate, parse_calls
from tools import TOOLS

MODEL_ID = "sapientinc/HRM-Text-1B"

# Local-tool prompts (unseen tools). gold_tool=None means "no tool needed".
LOCAL_TOOLS = list(TOOLS.keys())
LOCAL_EVAL = [
    {"user": "What's the weather in London?", "tool": "get_weather", "args": {"city": "London"}},
    {"user": "How's the weather in Tokyo right now?", "tool": "get_weather", "args": {"city": "Tokyo"}},
    {"user": "What is 19 times 23?", "tool": "calculator", "args": {"expression": "19 * 23"}},
    {"user": "Compute (45 + 15) / 4.", "tool": "calculator", "args": {"expression": "(45 + 15) / 4"}},
    {"user": "Look up the speed of light.", "tool": "web_search", "args": {"query": "speed of light"}},
    {"user": "What time is it?", "tool": "get_current_time", "args": {}},
    {"user": "Hi there, what can you do?", "tool": None, "args": None},
    {"user": "Thanks for your help!", "tool": None, "args": None},
]


def required_map(tool_schemas):
    out = {}
    for s in tool_schemas:
        fn = s.get("function", s) if isinstance(s, dict) else {}
        name = fn.get("name")
        if name:
            out[name] = set((fn.get("parameters") or {}).get("required", []) or [])
    return out


def _norm(d):
    return json.dumps(d or {}, sort_keys=True)


def score_one(model, tok, tool_schemas, context, gold, device, max_new):
    prefix = build_prefix(tool_schemas, context)
    text = generate(model, tok, prefix, device, max_new)
    pred = parse_calls(text)
    reqs = required_map(tool_schemas)
    gold_is_call = gold["role"] == "calls"

    if gold_is_call:
        gold_calls = gold["calls"]
        gold_names = [c["name"] for c in gold_calls]
        r = {
            "kind": "tool",
            "valid_call": len(pred) >= 1,
            "first_tool_match": bool(pred) and pred[0]["name"] == gold_names[0],
            "tool_set_match": {c["name"] for c in pred} == set(gold_names),
            "required_args_ok": bool(pred) and reqs.get(pred[0]["name"], set())
                                <= set((pred[0].get("arguments") or {}).keys()),
            "exact_first_call": bool(pred) and pred[0]["name"] == gold_calls[0]["name"]
                                and _norm(pred[0].get("arguments")) == _norm(gold_calls[0].get("arguments")),
        }
    else:
        r = {"kind": "final", "correct_abstain": (len(pred) == 0 and TOOL_OPEN not in text)}
    return r


def run_eval(model, tok, examples, device, max_new):
    rows = [score_one(model, tok, ex["tools"], ex["context"], ex["gold"], device, max_new) for ex in examples]
    tool = [r for r in rows if r["kind"] == "tool"]
    final = [r for r in rows if r["kind"] == "final"]
    def rate(rs, k): return 100 * sum(r[k] for r in rs) / len(rs) if rs else float("nan")
    return {
        "n_tool": len(tool), "n_final": len(final),
        "valid_call": rate(tool, "valid_call"),
        "first_tool_match": rate(tool, "first_tool_match"),
        "tool_set_match": rate(tool, "tool_set_match"),
        "required_args_ok": rate(tool, "required_args_ok"),
        "exact_first_call": rate(tool, "exact_first_call"),
        "abstain": rate(final, "correct_abstain"),
    }


def print_metrics(label, m):
    print(f"\n=== {label} ===")
    print(f"  tool-call prompts (n={m['n_tool']}):")
    print(f"     valid JSON call ........ {m['valid_call']:.0f}%")
    print(f"     correct first tool ..... {m['first_tool_match']:.0f}%")
    print(f"     exact tool set ......... {m['tool_set_match']:.0f}%")
    print(f"     required args present .. {m['required_args_ok']:.0f}%")
    print(f"     exact first call ....... {m['exact_first_call']:.0f}%")
    print(f"  no-tool prompts (n={m['n_final']}):")
    print(f"     correctly abstained .... {m['abstain']:.0f}%")


def first_assistant_idx(turns):
    for i, t in enumerate(turns):
        if t["role"] in ("calls", "final"):
            return i
    return None


def load_examples(test_path, limit):
    examples = []
    if test_path and os.path.exists(test_path):
        for line in open(test_path):
            if not line.strip():
                continue
            c = json.loads(line)
            idx = first_assistant_idx(c["turns"])
            if idx is None:
                continue
            examples.append({"tools": c["tools"], "context": c["turns"][:idx],
                             "gold": c["turns"][idx], "set": "hermes"})
    if limit > 0:
        examples = examples[:limit]
    local_schemas = [TOOLS[n]["schema"] for n in LOCAL_TOOLS]
    for e in LOCAL_EVAL:
        gold = ({"role": "calls", "calls": [{"name": e["tool"], "arguments": e["args"]}]}
                if e["tool"] else {"role": "final", "content": ""})
        examples.append({"tools": local_schemas, "context": [{"role": "user", "content": e["user"]}],
                         "gold": gold, "set": "local"})
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None)
    p.add_argument("--base-too", action="store_true", help="also score the raw base for a delta")
    p.add_argument("--test", default=os.path.join(os.path.dirname(__file__), "data", "tool_sft_hermes_test.jsonl"))
    p.add_argument("--limit", type=int, default=120, help="cap held-out Hermes examples (speed)")
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--max-new-tokens", type=int, default=96)
    args = p.parse_args()

    device = "mps" if (args.device == "auto" and torch.backends.mps.is_available()) else \
             (args.device if args.device != "auto" else "cpu")

    examples = load_examples(args.test, args.limit)
    hermes = [e for e in examples if e["set"] == "hermes"]
    local = [e for e in examples if e["set"] == "local"]
    print(f"[eval] {len(hermes)} held-out Hermes + {len(local)} local-tool prompts | device={device}")

    tok = AutoTokenizer.from_pretrained(args.adapter or MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()

    if args.base_too or not args.adapter:
        print_metrics("BASE — Hermes held-out", run_eval(model, tok, hermes, device, args.max_new_tokens))
        print_metrics("BASE — local tools", run_eval(model, tok, local, device, args.max_new_tokens))

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
        print_metrics("ADAPTER — Hermes held-out", run_eval(model, tok, hermes, device, args.max_new_tokens))
        print_metrics("ADAPTER — local tools", run_eval(model, tok, local, device, args.max_new_tokens))


if __name__ == "__main__":
    main()
