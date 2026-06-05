#!/usr/bin/env python3
"""Convert Salesforce/xlam-function-calling-60k -> our {tools, turns} schema.

xLAM is the data behind xLAM-2-1b-fc-r (the strongest ~1B on BFCL). Single-turn, with
PARALLEL/MULTIPLE calls in the `answers` array — exactly the structure our weak BFCL
categories (parallel / parallel_multiple) need. Tools are normalized to the same
OpenAI-style schema our Hermes/glaive data uses, so the model sees ONE tool format.

GATED: accept terms at huggingface.co/datasets/Salesforce/xlam-function-calling-60k and
have HF_TOKEN set (login or env).

  python convert_xlam.py --out data/xlam.jsonl --bias-multicall
"""
import argparse
import json
import os

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def _loads(x):
    return x if isinstance(x, (list, dict)) else json.loads(x)


def _norm_tool(t):
    """xLAM {name, description, parameters:{p:{type,description,required}}} ->
    OpenAI {type:function, function:{name, description, parameters:{type:object, properties, required}}}."""
    props, required = {}, []
    for pname, spec in (t.get("parameters", {}) or {}).items():
        spec = spec or {}
        props[pname] = {k: spec[k] for k in ("type", "description") if k in spec}
        if spec.get("required"):
            required.append(pname)
    return {"type": "function", "function": {
        "name": t.get("name", ""), "description": t.get("description", ""),
        "parameters": {"type": "object", "properties": props, "required": required}}}


def convert(rec):
    try:
        query = rec["query"]
        tools = [_norm_tool(t) for t in _loads(rec["tools"])]
        answers = [{"name": a["name"], "arguments": a.get("arguments", {})} for a in _loads(rec["answers"])]
    except Exception:  # noqa: BLE001
        return None
    if not query or not tools or not answers:
        return None
    return {"tools": tools, "turns": [{"role": "user", "content": query},
                                      {"role": "calls", "calls": answers}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "xlam.jsonl"))
    ap.add_argument("--bias-multicall", action="store_true",
                    help="sort multi-call examples first, so a downstream `--extra xlam.jsonl:N` "
                         "cap keeps more of the parallel/multiple cases we're weak on")
    args = ap.parse_args()
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    convos = [c for c in (convert(r) for r in ds) if c]
    if args.bias_multicall:
        convos.sort(key=lambda c: -len(c["turns"][1]["calls"]))
    multi = sum(1 for c in convos if len(c["turns"][1]["calls"]) > 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in convos:
            f.write(json.dumps(c) + "\n")
    print(f"[xlam] {len(convos)} convos ({multi} multi-call) -> {args.out}")


if __name__ == "__main__":
    main()
