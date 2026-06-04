#!/usr/bin/env python3
"""Run the tool-use agent loop with a LoRA-adapted (or base) HRM-Text-1B.

    python infer_agent.py --adapter adapters/tooluse-v1 "What's the weather in Paris?"
    python infer_agent.py "What is 23 * 47?"          # base model, for comparison

The loop: build the prefix -> generate -> if the model emits one or more
<tool_call> blocks, execute each via tools.py and append the observations ->
regenerate -> ... until the model returns a plain answer or --max-iters is hit.

The local tools here (get_weather, calculator, ...) need NOT match the tools the
adapter was trained on (e.g. Hermes) — the point is that the model generalizes
the *format* to whatever tool schemas appear in the prompt.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import build_prefix, BOX_END, TOOL_OPEN, TOOL_CLOSE
from tools import TOOLS, execute

MODEL_ID = "sapientinc/HRM-Text-1B"
CALL_RE = re.compile(re.escape(TOOL_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TOOL_CLOSE), re.DOTALL)


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(model, tok, prefix, device, max_new_tokens):
    inputs = tok(prefix, return_tensors="pt", add_special_tokens=False).to(device)
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])  # bidirectional prefix
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=False)
    for t in (BOX_END, "<|im_end|>", "<|endoftext|>"):
        text = text.replace(t, "")
    return text.strip()


def parse_calls(text):
    calls = []
    for m in CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if obj.get("name"):
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except Exception:  # noqa: BLE001
            pass
    return calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default="What's the weather in Paris?")
    p.add_argument("--model", default=None, help="path/repo of a FULL fine-tuned model (vs --adapter for LoRA)")
    p.add_argument("--adapter", default=None, help="path to a trained LoRA adapter dir")
    p.add_argument("--tools", nargs="*", default=list(TOOLS.keys()))
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--max-iters", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if args.model:
        print(f"[load] full model={args.model} device={device}")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    else:
        print(f"[load] base={MODEL_ID} adapter={args.adapter or '(none — raw base)'} device={device}")
        tok = AutoTokenizer.from_pretrained(args.adapter or MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).eval()

    tool_schemas = [TOOLS[n]["schema"] for n in args.tools if n in TOOLS]
    turns = [{"role": "user", "content": args.prompt}]
    for it in range(args.max_iters):
        prefix = build_prefix(tool_schemas, turns)
        text = generate(model, tok, prefix, device, args.max_new_tokens)
        calls = parse_calls(text)
        if not calls:
            print(f"\n=== answer ===\n{text}")
            return
        observations = []
        for c in calls:
            obs = execute(c["name"], c["arguments"])
            observations.append(obs)
            print(f"[iter {it}] tool_call: {c['name']}({json.dumps(c['arguments'])}) -> {obs}")
        turns.append({"role": "calls", "calls": calls, "observations": observations})
    print("\n[stop] hit max-iters without a final answer.")


if __name__ == "__main__":
    main()
