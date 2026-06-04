#!/usr/bin/env python3
"""Convert NousResearch/hermes-function-calling-v1 into our conversation schema,
with a leak-safe held-out test split.

Hermes already uses the <tools>/<tool_call>/<tool_response> ChatML convention, so
conversion is mostly: parse the top-level `tools` JSON, pull the user message, and
turn each gpt/tool turn into a calls-step or final-step.

The singleturn and multiturn files share the SAME scenarios, so we split by user
message (every variant of a scenario lands on the same side) to avoid train/test
leakage. The test set keeps one convo per held-out scenario (preferring the
variant that has a final answer).

    python convert_hermes.py                       # train + 150-scenario test split, + seed
    python convert_hermes.py --holdout 0           # no test split
    python convert_hermes.py --limit 2000 --no-seed

Outputs:
    data/tool_sft_hermes.jsonl        (train)
    data/tool_sft_hermes_test.jsonl   (held-out test, disjoint from train)
"""
import argparse
import ast
import json
import os
import random
import re

from huggingface_hub import hf_hub_download

REPO = "NousResearch/hermes-function-calling-v1"
DEFAULT_FILES = ["func-calling.json", "func-calling-singleturn.json", "glaive-function-calling-5k.json"]
HERE = os.path.dirname(os.path.abspath(__file__))

CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _loads(s):
    """Parse JSON, tolerating Python-dict-ish single quotes."""
    try:
        return json.loads(s)
    except Exception:  # noqa: BLE001
        return ast.literal_eval(s)


def parse_calls(text):
    calls = []
    for m in CALL_RE.finditer(text):
        try:
            obj = _loads(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(obj, dict) and obj.get("name"):
            calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
    return calls


def parse_responses(text):
    return [m.group(1).strip() for m in RESP_RE.finditer(text)]


def convert_record(r):
    """Convert one Hermes/glaive record into the multi-turn {tools, turns} schema.
    Handles multiple human turns (glaive is heavily multi-turn)."""
    try:
        tools = r["tools"]
        tools = _loads(tools) if isinstance(tools, str) else tools
    except Exception:  # noqa: BLE001
        return None
    if not tools:
        return None

    turns = []
    for t in r["conversations"]:
        frm, val = t["from"], t["value"]
        if frm == "human":
            if val.strip():
                turns.append({"role": "user", "content": val.strip()})
        elif frm == "gpt":
            calls = parse_calls(val)
            if calls:
                turns.append({"role": "calls", "calls": calls})
            elif val.strip():
                turns.append({"role": "final", "content": val.strip()})
        elif frm == "tool":
            resps = parse_responses(val)
            if resps and turns and turns[-1]["role"] == "calls":
                turns[-1]["observations"] = resps
        # system turn is skipped — we render our own system text + tools block
    if not any(t["role"] == "user" for t in turns):
        return None
    if not any(t["role"] in ("calls", "final") for t in turns):
        return None
    return {"tools": tools, "turns": turns}


def _first_user(c):
    return next((t["content"] for t in c["turns"] if t["role"] == "user"), "")


def _has_final(c):
    return any(t["role"] == "final" for t in c["turns"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    p.add_argument("--limit", type=int, default=-1, help="cap total converted conversations")
    p.add_argument("--holdout", type=int, default=0,
                   help="held-out test scenarios (default 0: keep all for training; we evaluate on BFCL)")
    p.add_argument("--out", default=os.path.join(HERE, "data", "tool_sft_hermes.jsonl"))
    p.add_argument("--test-out", default=os.path.join(HERE, "data", "tool_sft_hermes_test.jsonl"))
    p.add_argument("--no-seed", action="store_true", help="do NOT append the local seed convos to train")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    all_convos = []
    for fname in args.files:
        path = hf_hub_download(REPO, fname, repo_type="dataset")
        data = json.load(open(path))
        kept = 0
        for r in data:
            convo = convert_record(r)
            if convo is None:
                continue
            all_convos.append(convo)
            kept += 1
            if args.limit > 0 and len(all_convos) >= args.limit:
                break
        print(f"  {fname}: {kept}/{len(data)} converted")
        if args.limit > 0 and len(all_convos) >= args.limit:
            break

    # Leak-safe split: choose held-out *scenarios* by first user message.
    keys = sorted({_first_user(c) for c in all_convos})
    random.Random(args.seed).shuffle(keys)
    test_keys = set(keys[:args.holdout]) if args.holdout > 0 else set()

    train = [c for c in all_convos if _first_user(c) not in test_keys]
    test_by_key = {}
    for c in all_convos:
        k = _first_user(c)
        if k in test_keys:
            cur = test_by_key.get(k)
            if cur is None or (_has_final(c) and not _has_final(cur)):
                test_by_key[k] = c  # prefer the variant with a final answer
    test = list(test_by_key.values())

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_train = 0
    with open(args.out, "w") as f:
        for c in train:
            f.write(json.dumps(c) + "\n")
            n_train += 1
        if not args.no_seed:
            seed = os.path.join(HERE, "data", "tool_sft.jsonl")
            if os.path.exists(seed):
                added = 0
                for line in open(seed):
                    if line.strip():
                        f.write(line if line.endswith("\n") else line + "\n")
                        added += 1
                print(f"  seed: appended {added} local-tool convos to train")
                n_train += added
            else:
                print("  seed: data/tool_sft.jsonl not found (run make_dataset.py); skipping")

    with open(args.test_out, "w") as f:
        for c in test:
            f.write(json.dumps(c) + "\n")

    n_test_final = sum(1 for c in test if _has_final(c))
    print(f"\n[done] train: {n_train} convos -> {args.out}")
    print(f"       test:  {len(test)} held-out scenarios -> {args.test_out}"
          f"  ({n_test_final} have a final answer, {len(test) - n_test_final} are call-only)")
    print(f"       split is disjoint by user message (no scenario leakage).")


if __name__ == "__main__":
    main()
