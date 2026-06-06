# Code-expert experiment — runbook

**Goal:** train a **code expert** for HRM-Text-1B, then **compose-merge** it with the existing
tool expert (v2) and test whether HRM experts **compose** (gain both skills) or **interfere/
collapse**. This is the follow-up to the merge-viability result (the v1+v2 weight-soup stayed
coherent — see main `README.md` §4 and `merge_experts.py`), now with two *different-skill*
deltas instead of two tool deltas.

Three questions this answers:
1. **Did the code expert learn to code?** (code-expert vs base on HumanEval/MBPP)
2. **Does the composition stay stable on HRM?** (collapse check — the same per-cycle
   amplification that killed LoRA could wreck a summed task vector)
3. **Do the skills compose?** — does the merged model keep **tools** (BFCL) *and* **code**
   (HumanEval/MBPP) at once, or trade them?

---

## Prerequisites
- A CUDA GPU pod (A100 80GB; image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).
- `export HF_TOKEN=hf_...` — needed to **push the code expert** (base/v1/v2 are public, no gating).
- **Run on a disposable pod:** `eval_code.py` *executes model-generated code* (subprocess + 12s
  timeout). HumanEval/MBPP are toy algorithmic problems, but it is running generated code.

## One command
```bash
git pull
export HF_TOKEN=hf_...
bash run_code_expert.sh
```

That's the whole thing. It's **non-destructive**: the code expert trains to its own dir and
pushes to a **separate** repo (`jasoncarreira/hrm-text-code`, with a hard guard that refuses to
collide with base / v1 / v2); the merge saves to its own dir and is **not** pushed.

## What it does (7 steps)
| Step | Command (inside `run_code_expert.sh`) | Purpose |
|---|---|---|
| 1 | deps (torch≥2.7 cu126 + requirements) | same env fix as `run_v2.sh` |
| 2 | `make_code_data.py` → `data/code_sft.jsonl` | instruction→code SFT, **synth,cot lane** |
| 3 | `train_full.py --data … --out-dir models/hrm-code` | train the code expert (fresh from base) |
| 4 | `eval_code.py --bench {humaneval,mbpp}` on **base** and **code** | did training add code skill? |
| 5 | push `models/hrm-code` → `hrm-text-code` (separate repo) | save the expert |
| 6 | `merge_compose.py` tool(v2) ⊕ code → `models/hrm-merge-toolcode` | the composition merge + collapse + BFCL |
| 7 | `eval_code.py` on the **merge** | did code survive the merge? |

## Design choices (why)
- **synth,cot lane for code.** The code expert trains/evals in the `synth,cot` condition
  (`<|quad_end|><|object_ref_end|>`), *distinct* from the tool expert's `direct` condition. HRM's
  condition token acts as a soft router, so putting the two experts in different input lanes
  should reduce merge interference. (Both `make_code_data.py` and `eval_code.py` use this lane.)
- **HumanEval + MBPP, not SWE-bench.** SWE-bench is repo-scale/agentic (10k–100k-token context,
  Docker test harness) — a 1B/4k model scores ~0 and you learn nothing. HumanEval/MBPP are
  self-contained, unit-tested, pass@1, and the number *moves with training*. (Rigor upgrade
  later: EvalPlus HumanEval+/MBPP+.)
- **Task arithmetic for the merge:** `merged = base + Σ cᵢ·(expertᵢ − base)`, default `cᵢ = 1.0`.
- **Collapse check:** mean next-token entropy; near `log(vocab) ≈ 11.1` = collapsed. The v1+v2
  soup stayed ~0.8–4.0 (healthy), so we expect the same — but the *summed* task vector grows
  magnitude, which is exactly where the LoRA-style per-cycle amplification could bite. Watch it.

## Knobs (env vars / args)
- `TOOL_REPO` (default `jasoncarreira/hrm-text-agent-v2`) — the tool expert to merge with.
- `CODE_REPO` (default `jasoncarreira/hrm-text-code`), `CODE_DIR`, `MERGE_DIR`.
- Merge coefficients: edit the `--coeffs 1.0,1.0` in step 6. If the merge collapses or interferes,
  try `0.7,0.7` (scale both deltas down) — magnitude is the likely culprit.
- `make_code_data.py --n` (default 25000), `eval_code.py --limit` (default = full set).

## How to read the results (output files)
| File(s) | Question |
|---|---|
| `he_base.json` / `mbpp_base.json` vs `he_code.json` / `mbpp_code.json` | **Did the code expert learn to code?** (expect base low single digits → code-expert meaningfully higher) |
| `merge_compose.json` | `collapsed` flag (**stable?**), `bfcl` scores (**tools kept?**), sample generations |
| `he_merge.json` / `mbpp_merge.json` | **Did code survive the merge** alongside tools? (the composition verdict) |

**The win condition:** the merge is `collapsed: false`, BFCL stays near v2's call levels, *and*
HumanEval/MBPP stay near the code-expert's levels. That = skills composed. If code or tools crater
in the merge, it's interference → try lower coeffs, or fall back to model-routing.

## Handoff back to the laptop
When done, **commit the result JSONs** so the laptop Claude can pick them up and write it up
(don't edit the main `README.md` — the laptop owns it):
```bash
git pull
git add he_*.json mbpp_*.json merge_compose.json code_*.log
git commit -m "code-expert experiment results"
git push
```
The laptop is watching for that and will summarize + tear down the pod.

> Note: base HumanEval/MBPP are also being measured on the laptop (`he_base.json`/`mbpp_base.json`).
> The pod re-runs them in step 4 for a self-contained result — fine, or reuse if already committed.
