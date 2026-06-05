#!/usr/bin/env python3
"""Build the mixed SFT dataset: tool-use + instruction-following + irrelevance.

  1. tool-use convos   -> from data/tool_sft_hermes.jsonl (run convert_hermes.py first)
  2. instruction convos -> HuggingFaceH4/no_robots (no tools -> answer directly)
  3. irrelevance convos -> an instruction + random IRRELEVANT tools -> answer directly
                           (teaches "tools present but none fit -> don't call")

Writes data/sft_mixed.jsonl in the {tools, turns} schema for train_full.py.
"""
import argparse
import json
import os
import random

from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool-file", default=os.path.join(HERE, "data", "tool_sft_hermes.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "data", "sft_mixed.jsonl"))
    ap.add_argument("--n-tool", type=int, default=12000)
    ap.add_argument("--n-instr", type=int, default=5000)
    ap.add_argument("--n-irrel", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra pre-built convo files to mix in, as PATH or PATH:COUNT (each line a "
                         "{tools,turns} or {raw_prompt,raw_target} convo). "
                         "e.g. --extra data/xlam.jsonl:14000 data/format_slice.jsonl:3000")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) tool-use convos (subsample) + a pool of their tool schemas for irrelevance
    tool = [json.loads(l) for l in open(args.tool_file) if l.strip()]
    rng.shuffle(tool)
    tool = tool[:args.n_tool]
    pool = [s for c in tool for s in c.get("tools", [])]

    # 2) instruction convos from no_robots (no tools -> answer directly)
    nr = load_dataset("HuggingFaceH4/no_robots", split="train")
    instr = []
    for ex in nr:
        msgs = ex["messages"]
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        asst = next((m["content"] for m in msgs if m["role"] == "assistant"), None)
        if user and asst and len(asst) < 4000:
            instr.append({"tools": [], "turns": [{"role": "user", "content": user},
                                                 {"role": "final", "content": asst}]})
    rng.shuffle(instr)
    pure = instr[:args.n_instr]
    irrel_src = instr[args.n_instr:args.n_instr + args.n_irrel]

    # 3) irrelevance: same instruction, but with random irrelevant tools attached
    irrel = []
    for ex in irrel_src:
        k = rng.randint(1, 3)
        tools = rng.sample(pool, min(k, len(pool))) if pool else []
        irrel.append({"tools": tools, "turns": ex["turns"]})

    # 4) extra pre-built convo sources (xLAM tool-calls, format-discipline slice, ...)
    extra = []
    extra_counts = {}
    for spec in args.extra:
        path, _, cnt = spec.partition(":")
        convos = [json.loads(l) for l in open(path) if l.strip()]
        rng.shuffle(convos)
        if cnt:
            convos = convos[:int(cnt)]
        extra += convos
        extra_counts[os.path.basename(path)] = len(convos)

    mixed = tool + pure + irrel + extra
    rng.shuffle(mixed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for c in mixed:
            f.write(json.dumps(c) + "\n")
    print(f"[mixed] tool={len(tool)} instruction={len(pure)} irrelevance={len(irrel)} "
          f"extra={extra_counts} total={len(mixed)} -> {args.out}")


if __name__ == "__main__":
    main()
