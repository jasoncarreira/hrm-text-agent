#!/usr/bin/env python3
"""Build a CODE-expert SFT set for HRM-Text: instruction -> code.

Trained in the **synth,cot** condition (the reasoning lane) — deliberately a DIFFERENT
lane than the tool expert's `direct` — so the two deltas occupy different input regimes
and (hopefully) interfere less when merged (HRM's condition token acts as a soft router).

Emits "raw" {raw_prompt, raw_target} convos (bypass tool-use build_prefix). Pulls from
short/medium function-level instruct-code datasets and char-filters to fit the 2048
window (train_full also drops anything still > max_len at tokenization).

  python make_code_data.py --out data/code_sft.jsonl --n 25000
"""
import argparse
import json
import os
import random

from datasets import load_dataset

from agent_format import BOX_END, IM_END, IM_START

SYNTH_COT = "<|quad_end|><|object_ref_end|>"
HERE = os.path.dirname(os.path.abspath(__file__))

# (hf_id, split, instruction_fields, answer_field) — verified schemas
SOURCES = [
    ("m-a-p/CodeFeedback-Filtered-Instruction", "train", ("query",), "answer"),
    ("sahil2801/CodeAlpaca-20k", "train", ("instruction", "input"), "output"),
]


def _instr(rec, fields):
    parts = [str(rec.get(f, "")).strip() for f in fields]
    return "\n".join(p for p in parts if p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data", "code_sft.jsonl"))
    ap.add_argument("--n", type=int, default=25000, help="total convos (split across sources)")
    ap.add_argument("--max-chars", type=int, default=6000, help="skip instr+answer longer than this (~2048 tok)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    convos = []
    per = max(1, args.n // len(SOURCES))
    for hf_id, split, ifields, afield in SOURCES:
        try:
            ds = load_dataset(hf_id, split=split)
        except Exception as e:  # noqa: BLE001
            print(f"[code] {hf_id} load failed: {e}")
            continue
        order = list(range(len(ds)))
        rng.shuffle(order)
        kept = 0
        for idx in order:
            if kept >= per:
                break
            rec = ds[idx]
            instr, ans = _instr(rec, ifields), str(rec.get(afield, "")).strip()
            if not instr or not ans or len(instr) + len(ans) > args.max_chars:
                continue
            convos.append({"raw_prompt": f"{IM_START}{SYNTH_COT}{instr}{IM_END}",
                           "raw_target": f"{ans}{BOX_END}"})
            kept += 1
        print(f"[code] {hf_id}: kept {kept}")
    rng.shuffle(convos)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in convos:
            f.write(json.dumps(c) + "\n")
    print(f"[code] total {len(convos)} convos -> {args.out}")


if __name__ == "__main__":
    main()
