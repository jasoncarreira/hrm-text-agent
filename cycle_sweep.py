#!/usr/bin/env python3
"""Cycle-scaling experiment: is HRM's *recurrence* what drives its reasoning?

Hypothesis: HRM punches above its parameter count because iterating the H/L modules
gives compute-depth >> param-count, and for reasoning that compute substitutes for
parameters. If true, REDUCING the recurrence cycles should hurt a reasoning task more
than a non-reasoning control. (Raising cycles ABOVE the trained setting would be the
other half, but see the KV-cache note below.)

Design: sweep H_cycles/L_cycles on a reasoning task (GSM8k) + a control (MMLU, where
extra compute shouldn't help), at a fixed seeded subset, and tabulate accuracy.
  - GSM8k drops as cycles fall but MMLU stays flat  -> recurrence is doing reasoning work ✅
  - both flat                                        -> recurrence adds no inference-time reasoning here
Runs the *base* model by default (the architecture question; our fine-tune would confound).

### Findings baked in from a Step-0 probe on the base model
  - Cycles are read from `model.config.H_cycles/L_cycles` at runtime (config IS the knob).
  - DOWN-ablation works with the KV cache on (default here) and is the robust test:
    e.g. H=1,L=1 already produced visibly degraded/looping output vs the default 2x3.
  - UP-scaling (cycles > trained 2x3) IndexErrors with the cache on (the cache is sized
    for the trained unroll). `--up` adds `--no-cache` to attempt it, but it's slow and the
    model wasn't trained for more compute, so it may still degrade or fail.

Usage:
  python cycle_sweep.py --model sapientinc/HRM-Text-1B --limit 200            # down-ablation (default)
  python cycle_sweep.py --model sapientinc/HRM-Text-1B --limit 100 --up       # try up-scaling (no-cache, slow)
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DOWN = [(2, 3), (2, 1), (1, 3), (1, 1)]      # default + down-ablations (KV-cache safe)
UP = [(2, 3), (3, 4), (4, 6), (6, 9)]        # default + up (needs --no-cache; may fail/be slow)


def run_one(model, bench, limit, h, l, up):
    out = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    cmd = [sys.executable, os.path.join(HERE, "eval_academic.py"), "--model", model,
           "--benchmarks", bench, "--limit", str(limit),
           "--h-cycles", str(h), "--l-cycles", str(l), "--out", out]
    if up:
        cmd.append("--no-cache")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or ["<no stderr>"])[-1]
        return None, tail
    data = json.load(open(out))
    os.unlink(out)
    m = data["results"][bench]
    return m.get("acc", m.get("f1")), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sapientinc/HRM-Text-1B", help="base model = the clean arch test")
    ap.add_argument("--limit", type=int, default=200, help="examples per benchmark per setting")
    ap.add_argument("--benchmarks", default="GSM8k,MMLU", help="reasoning,control (first should benefit from compute)")
    ap.add_argument("--up", action="store_true", help="sweep cycles UP instead of down (adds --no-cache; slow, may fail)")
    args = ap.parse_args()

    settings = UP if args.up else DOWN
    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(f"[cycle-sweep] model={args.model} limit={args.limit} benchmarks={benches} "
          f"mode={'UP (no-cache)' if args.up else 'DOWN'} settings={settings}")

    rows = {}
    for h, l in settings:
        rows[(h, l)] = {}
        for b in benches:
            acc, err = run_one(args.model, b, args.limit, h, l, args.up)
            rows[(h, l)][b] = acc if acc is not None else f"FAIL:{err[:48]}"
            print(f"  H={h} L={l}  {b:>10}: "
                  + (f"{acc:.3f}" if isinstance(acc, float) else rows[(h, l)][b]), flush=True)

    print("\n=== accuracy vs recurrence cycles ===")
    print("H×L".ljust(8) + "".join(f"{b:>12}" for b in benches))
    for (h, l) in settings:
        cells = "".join(
            (f"{rows[(h, l)][b]:>12.3f}" if isinstance(rows[(h, l)][b], float) else f"{str(rows[(h, l)][b]):>12}")
            for b in benches)
        print(f"{f'{h}x{l}':<8}" + cells + ("  <- default" if (h, l) == (2, 3) else ""))
    print("\nRead: reasoning (GSM8k) falling as cycles drop while the control (MMLU) stays flat = the")
    print("recurrence is doing reasoning-specific compute (supports the 'punches above its weight' thesis).")
    print("Both flat = no inference-time reasoning gain from the recurrence on this checkpoint.")


if __name__ == "__main__":
    main()
