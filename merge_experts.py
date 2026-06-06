#!/usr/bin/env python3
"""Model-soup merge experiment for HRM-Text tool-use experts.

We only have tool-use deltas so far (v1 and v2), so this serves two purposes at once:
  (a) MECHANICS GATE — does HRM's weight-shared recurrence tolerate a merged task vector,
      or does it collapse the way LoRA did? (LoRA destabilized because a delta in the
      weight-shared layers is amplified across every recurrence cycle.) If a soup of two
      *real* deltas stays coherent, merging is viable and a complementary (e.g. code)
      expert is worth training. If it collapses, drop merging -> model-routing instead.
  (b) USEFUL EXPERIMENT — v1 has good irrelevance / weaker calls; v2 has great calls /
      poor irrelevance (80.8 -> 60.8). A LERP soup may land on a better balance for free.

For each alpha: merged = (1-alpha)*v1 + alpha*v2   (alpha=0 -> v1, alpha=1 -> v2).
  - COLLAPSE CHECK: mean next-token entropy (nats). Healthy LMs are peaked (~1-5 nats);
    near-uniform (~log(vocab) = 11.1) means the distribution collapsed. Plus a sample
    generation to eyeball coherence.
  - BFCL subset: default `simple` + `irrelevance` (the tradeoff axis), capped by --limit.

  python merge_experts.py --alphas 0,0.25,0.5,0.75,1.0 --limit 100
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import DIRECT, IM_END, IM_START

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH_COT = "<|quad_end|><|object_ref_end|>"
PROMPTS = [
    f"{IM_START}{DIRECT}You are a helpful assistant. User: What is the capital of France?{IM_END}",
    f"{IM_START}{SYNTH_COT}A train travels 60 miles in 1.5 hours. What is its average speed? Reason step by step.{IM_END}",
]


def collapse_check(model, tok, device):
    """(mean next-token entropy in nats, [sample generations]). Near log(V) => collapsed."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="jasoncarreira/hrm-text-agent")
    ap.add_argument("--v2", default="jasoncarreira/hrm-text-agent-v2")
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0", help="0 -> pure v1, 1 -> pure v2")
    ap.add_argument("--categories", default="simple,irrelevance", help="BFCL cats (the tradeoff axis)")
    ap.add_argument("--limit", type=int, default=100, help="records per category (Mac speed)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default=os.path.join(HERE, "merge_results.json"))
    ap.add_argument("--keep", action="store_true", help="keep the merged model dirs")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]

    print(f"[merge] v1={args.v1} v2={args.v2} alphas={alphas} cats={cats} limit={args.limit}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.v1)
    print("[merge] loading v1 ...", flush=True)
    carrier = AutoModelForCausalLM.from_pretrained(args.v1, dtype=torch.float32)
    sd1 = {k: v.clone() for k, v in carrier.state_dict().items()}
    print("[merge] loading v2 ...", flush=True)
    m2 = AutoModelForCausalLM.from_pretrained(args.v2, dtype=torch.float32)
    sd2 = {k: v.clone() for k, v in m2.state_dict().items()}
    del m2
    assert set(sd1.keys()) == set(sd2.keys()), "v1/v2 state_dict key mismatch"

    vlog = math.log(carrier.config.vocab_size)
    results = []
    for a in alphas:
        carrier.load_state_dict({k: (1 - a) * sd1[k] + a * sd2[k] for k in sd1})
        carrier.to(args.device)
        ent, texts = collapse_check(carrier, tok, args.device)
        collapsed = ent > 0.8 * vlog
        carrier.to("cpu")
        d = tempfile.mkdtemp(prefix=f"merge_a{a}_")
        carrier.save_pretrained(d)
        tok.save_pretrained(d)
        scores = run_bfcl(d, cats, args.limit, args.device)
        if not args.keep:
            shutil.rmtree(d, ignore_errors=True)
        results.append({"alpha": a, "entropy_nats": round(ent, 2), "collapsed": collapsed,
                        "bfcl": scores, "sample": texts})
        print(f"[merge] alpha={a}: entropy={ent:.2f} (uniform~{vlog:.2f}) collapsed={collapsed} bfcl={scores}", flush=True)
        print(f"         sample: {texts}", flush=True)

    json.dump({"v1": args.v1, "v2": args.v2, "results": results}, open(args.out, "w"), indent=2)
    print("\n=== soup sweep (alpha: 0=v1 .. 1=v2) ===")
    print("alpha  entropy  collapse  " + "  ".join(f"{c:>11}" for c in cats))
    for r in results:
        cells = []
        for c in cats:
            v = r["bfcl"].get(c)
            cells.append(f"{v:>10.1f}%" if isinstance(v, float) else f"{str(v):>11}")
        print(f"{r['alpha']:<6} {r['entropy_nats']:<8} {str(r['collapsed']):<9} " + "  ".join(cells))
    print(f"\nRead: if intermediate blends stay coherent (entropy not near {vlog:.1f}) and BFCL is sane,")
    print("HRM tolerates merging -> a complementary (code) expert is worth training. If they collapse,")
    print(f"merging is dead on this recurrence -> use model-routing instead.\n-> {args.out}")


if __name__ == "__main__":
    main()
