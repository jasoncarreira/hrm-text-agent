#!/usr/bin/env python3
"""Build a format-discipline slice to recover the v1 regression (single-letter MCQ +
\\boxed{} math output discipline). Emits "raw" {raw_prompt, raw_target} convos in the
SAME envelope/condition eval_academic uses (direct for MCQ, synth,cot for math), so the
model relearns to answer crisply in that format.

Leakage-safe — TRAIN/aux splits ONLY (never the eval test sets):
  - MCQ-letter: allenai/ai2_arc ARC-Challenge TRAIN + cais/mmlu auxiliary_train
  - \\boxed{} math: EleutherAI/hendrycks_math TRAIN (solutions already contain \\boxed{})

  python make_format_slice.py --out data/format_slice.jsonl --n-mcq 2000 --n-math 1000
"""
import argparse
import json
import os
import random

from datasets import get_dataset_config_names, load_dataset

from agent_format import BOX_END, IM_END, IM_START

DIRECT = "<|object_ref_start|>"            # MCQ condition (matches eval_academic)
SYNTH_COT = "<|quad_end|><|object_ref_end|>"  # math condition (matches eval_academic)
HERE = os.path.dirname(os.path.abspath(__file__))


def _mcq_prompt(q, choices):
    t = q.strip() + "\n"
    for j, c in enumerate(choices):
        t += f"{chr(65 + j)}. {str(c).strip()}\n"
    return f"{IM_START}{DIRECT}{t}Answer:{IM_END}"


def build_mcq(n, rng):
    out = []
    try:
        for r in load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train"):
            if r["answerKey"] in r["choices"]["label"]:
                gi = r["choices"]["label"].index(r["answerKey"])
                out.append({"raw_prompt": _mcq_prompt(r["question"], r["choices"]["text"]),
                            "raw_target": f"{chr(65 + gi)}{BOX_END}"})
    except Exception as e:  # noqa: BLE001
        print(f"[format-slice] ARC train load failed: {e}")
    try:  # bounded slice — auxiliary_train is ~100k; we only need a few thousand
        for r in load_dataset("cais/mmlu", "all", split=f"auxiliary_train[:{max(3000, n * 3)}]"):
            out.append({"raw_prompt": _mcq_prompt(r["question"], r["choices"]),
                        "raw_target": f"{chr(65 + int(r['answer']))}{BOX_END}"})
    except Exception as e:  # noqa: BLE001
        print(f"[format-slice] MMLU aux load failed: {e}")
    rng.shuffle(out)
    return out[:n]


def build_math(n, rng):
    out = []
    try:
        for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
            for r in load_dataset("EleutherAI/hendrycks_math", subset, split="train"):
                if "\\boxed" in r["solution"]:
                    out.append({"raw_prompt": f"{IM_START}{SYNTH_COT}{r['problem']}{IM_END}",
                                "raw_target": f"{r['solution']}{BOX_END}"})
    except Exception as e:  # noqa: BLE001
        print(f"[format-slice] MATH train load failed: {e}")
    rng.shuffle(out)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "format_slice.jsonl"))
    ap.add_argument("--n-mcq", type=int, default=2000)
    ap.add_argument("--n-math", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    mcq, math = build_mcq(args.n_mcq, rng), build_math(args.n_math, rng)
    allc = mcq + math
    rng.shuffle(allc)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in allc:
            f.write(json.dumps(c) + "\n")
    print(f"[format-slice] mcq={len(mcq)} math={len(math)} total={len(allc)} -> {args.out}")


if __name__ == "__main__":
    main()
