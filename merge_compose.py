#!/usr/bin/env python3
"""Compose distinct HRM-Text experts by task arithmetic:
       merged = base + sum_i coeff_i * (expert_i - base)
The real composition test (e.g. tool + code): do skills ADD, or interfere / collapse?
Runs a collapse check (next-token entropy; near log(vocab) => collapsed) + a BFCL subset,
and SAVES the merged model so you can run eval_code.py on it afterward.

  python merge_compose.py --base sapientinc/HRM-Text-1B \
      --experts jasoncarreira/hrm-text-agent-v2,models/hrm-code --coeffs 1.0,1.0 \
      --merge-out models/hrm-merge-toolcode --categories simple,multiple,parallel,irrelevance --limit 200
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import DIRECT, IM_END, IM_START

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH_COT = "<|quad_end|><|object_ref_end|>"
PROMPTS = [
    f"{IM_START}{DIRECT}You are a helpful assistant. User: What is the capital of France?{IM_END}",
    f"{IM_START}{SYNTH_COT}Write a Python function that returns the nth Fibonacci number.{IM_END}",
]


def collapse_check(model, tok, device):
    model.eval()
    eos = tok.convert_tokens_to_ids("<|box_end|>")
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    ents, texts = [], []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
        tti = torch.ones_like(ids["input_ids"])
        with torch.no_grad():
            logits = model(**ids, token_type_ids=tti).logits[0, -1].float()
            probs = torch.softmax(logits, -1)
            ents.append(-(probs * torch.log(probs + 1e-12)).sum().item())
            out = model.generate(**ids, token_type_ids=tti, max_new_tokens=24, do_sample=False,
                                 eos_token_id=eos, pad_token_id=pad)
            texts.append(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)[:60])
    return sum(ents) / len(ents), texts


def run_bfcl(model_dir, cats, limit, device):
    cmd = [sys.executable, os.path.join(HERE, "bfcl_local.py"), "--model", model_dir,
           "--categories", *cats, "--limit", str(limit), "--device", device]
    r = subprocess.run(cmd, capture_output=True, text=True)
    scores = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s*(\w+)\s+([\d.]+)%\s+(\d+)", line)
        if m and m.group(1) in cats:
            scores[m.group(1)] = float(m.group(2))
    if not scores:
        scores["_error"] = (r.stderr.strip().splitlines() or ["?"])[-1][:80]
    return scores


def sd_of(repo):
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sapientinc/HRM-Text-1B")
    ap.add_argument("--experts", required=True, help="comma list of expert repos/dirs")
    ap.add_argument("--coeffs", default="", help="comma list (matches --experts); default 1.0 each")
    ap.add_argument("--merge-out", default="models/hrm-merge")
    ap.add_argument("--categories", default="simple,multiple,parallel,irrelevance")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(HERE, "merge_compose.json"))
    args = ap.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    experts = [e.strip() for e in args.experts.split(",") if e.strip()]
    coeffs = [float(c) for c in args.coeffs.split(",")] if args.coeffs else [1.0] * len(experts)
    assert len(experts) == len(coeffs), "experts/coeffs length mismatch"
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    print(f"[compose] base={args.base} experts={list(zip(experts, coeffs))}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    carrier = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in carrier.state_dict().items()}
    merged = {k: base_sd[k].clone() for k in base_sd}
    for e, c in zip(experts, coeffs):
        esd = sd_of(e)
        assert set(esd) == set(base_sd), f"key mismatch vs {e}"
        for k in merged:
            merged[k] += c * (esd[k] - base_sd[k])
        print(f"[compose] + {c} * delta({e})", flush=True)
    carrier.load_state_dict(merged)

    vlog = math.log(carrier.config.vocab_size)
    carrier.to(device)
    ent, texts = collapse_check(carrier, tok, device)
    carrier.to("cpu")
    os.makedirs(args.merge_out, exist_ok=True)
    carrier.save_pretrained(args.merge_out)
    tok.save_pretrained(args.merge_out)
    scores = run_bfcl(args.merge_out, cats, args.limit, device)
    res = {"base": args.base, "experts": experts, "coeffs": coeffs, "merge_out": args.merge_out,
           "entropy_nats": round(ent, 2), "collapsed": ent > 0.8 * vlog, "bfcl": scores, "sample": texts}
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[compose] entropy={ent:.2f} (uniform~{vlog:.2f}) collapsed={res['collapsed']}")
    print(f"[compose] BFCL={scores}")
    print(f"[compose] sample={texts}")
    print(f"[compose] merged -> {args.merge_out}  (run eval_code.py on it for code skill)\n-> {args.out}")


if __name__ == "__main__":
    main()
