# HRM-Text-1B — tool-use fine-tuning + a recurrence-vs-reasoning probe

[`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B) is an unusual 1B
**base** model: a *Hierarchical Reasoning Model* with two weight-shared recurrent stacks (a
slow "H" planner and a fast "L" worker) that are iterated `H_cycles × L_cycles` times per
forward pass — so its effective compute depth is much larger than its parameter count. This
repo does three things with it:

1. **Fine-tunes it to call tools**, evaluated with the official **Berkeley Function Calling
   Leaderboard (BFCL v4)** AST checker.
2. **Checks what that fine-tuning did to its general capability** (an 8-benchmark base-vs-tuned
   forgetting check).
3. **Tests *why* it reasons** — a cycle-scaling experiment probing whether the recurrence is
   what lets a 1B punch above its weight.

Trained model: [`jasoncarreira/hrm-text-agent`](https://huggingface.co/jasoncarreira/hrm-text-agent).

## TL;DR
- **Tool-calling:** full-parameter SFT took BFCL `simple` from **0% (base) → 61.5%**, with every
  multi-call category well off zero. Comfortably above *generic* 1B instructs; below the best
  purpose-built 1B (xLAM-2-1b-fc-r) and 3B FC models.
- **Forgetting check:** the SFT was **benign** — reasoning and knowledge stayed intact (GSM8k
  even +1.1); the only real drops were single-letter-MCQ **format discipline** on 2 of 8 tasks,
  much of it recoverable.
- **Recurrence:** **cutting the recurrence cycles collapses reasoning** (GSM8k 87.5% → 0%) while
  a knowledge control only drifts to chance — direct evidence that the recurrent compute is what
  drives the reasoning. And test-time compute depth turns out to be **fixed at training**.

---

## 1. Tool-use fine-tuning (BFCL v4, official AST checker, full test sets)

| Category | n | Base | LoRA (`lm_head`) | **Full-param SFT** |
|---|---|---|---|---|
| simple | 400 | 0% | 8% | **61.5%** |
| multiple | 200 | 0% | 0% | **53.5%** |
| parallel | 200 | 0% | 0% | **37.5%** |
| parallel_multiple | 200 | 0% | 0% | **28.0%** |
| irrelevance | 240 | 100% | 89% | **80.8%** |

Overall micro-average (all 1,240): **54.7%**. Across the four *call* categories alone
(excluding the "don't-call" irrelevance set), count-weighted ≈ **48%**. `simple` 0 → 61.5% on
**held-out** tools is the headline — full-FT cured the base model's instinct to *answer* instead
of *call*, which LoRA never could. (Base/LoRA columns are earlier 100-sample reads.)

**Where that lands among small models** (BFCL non-live AST, approximate): above every *generic*
1B instruct (Llama-3.2-1B ~38, Gemma-3-1b ~20, Falcon3-1B ~9), below the best purpose-built 1B
(xLAM-2-1b-fc-r ~69) and the 3B FC models — strong for a 1B *base* + SFT, not yet 3B-level. The
weak spots are the **parallel/parallel_multiple** categories; lifting those is the v2 goal.

### How we got there (the useful findings)
1. **LoRA on the recurrent layers → unstable.** HRM reuses each physical layer across its H×L
   cycles, so a LoRA delta is injected *many times per forward*; the cumulative perturbation
   collapses the output distribution toward uniform. Training loss **hides** it (dominated by
   easy in-JSON tokens — loss hit 0.5 while generating garbage). Lowering the LoRA scale only
   *delays* the collapse.
2. **LoRA on `lm_head` only → stable but weak (8%).** Adapting just the output head (applied once,
   outside the recurrence) is stable but too weak to override the base's answer-instinct.
3. **Full-parameter SFT → 61.5%.** Matches sapientinc's official recipe (full-parameter, never
   LoRA). Three changes together did it:
   - **full-parameter SFT** (not LoRA) — moves the real weights gently, recurrence stays stable;
   - the model's **`direct` condition token** (`<|object_ref_start|>` opening the
     `<|im_start|>…<|im_end|>` prompt — the documented mode for structured output, which we'd
     been omitting);
   - a **mixed dataset** (tool calls + general instructions + irrelevance) — the base has no
     instruction-tuning, so this teaches *when to call vs. answer* and keeps `irrelevance` high.

---

## 2. Did the tool-use SFT erode general capability? (forgetting check)

Base vs. fine-tune on 8 academic benchmarks (`eval_academic.py`). **Verdict: benign.**

| Benchmark | Base | Fine-tune | Δ | Note |
|---|---|---|---|---|
| GSM8k | 84.5% | **85.6%** | **+1.1** | free-form reasoning — intact (↑) |
| BoolQ | 86.3% | 87.3% | +1.0 | MCQ, 0% invalid — intact |
| DROP (F1) | 84.8% | 83.3% | −1.4 | free-form — intact |
| HellaSwag | 63.3% | 61.9% | −1.4 | MCQ, 0% invalid — intact |
| Winogrande | 72.2% | 70.6% | −1.6 | MCQ, 0% invalid — intact |
| MATH | 49.3% | 45.4% | −3.9 | `\boxed{}`-format discipline |
| MMLU | 60.1% | 55.5% | −4.6 | MCQ, 11.9% invalid — format |
| ARC-C | 83.5% | 75.1% | −8.4 | MCQ, 9.9% invalid — format |

**The whole story in one line:** every real drop is an *output-format* cost, never a
reasoning/knowledge cost. The tasks that stayed format-clean (0% invalid) are flat-to-up,
including both free-form reasoning sets; the only drops are the tasks where the model started
answering in prose instead of emitting a bare letter / `\boxed{}`. The cleanest internal control
is **GSM8k (+1.1) vs MATH (−3.9)** — same math skill, but the format-strict one regressed and the
free-form one improved. A peek at ARC's invalid outputs confirmed it: the model often answers
correctly in prose ("The correct answer is C…"), so backing out the chance-credited invalids puts
the true knowledge slip at ≈ **−0.5 (MMLU)** / **−2.9 (ARC)**.

*Harness fidelity:* reproduces the repo's published base numbers within ~1.5 pts (e.g. GSM8k 84.5
vs 84.7). MATH's absolute is harness-limited (512-token budget → ~26% of solutions don't emit a
parseable `\boxed{}`), but the budget is identical for both models, so the base-vs-tuned **delta**
is clean.

The fix for v2 is cheap and targeted — a **format-discipline data slice** (single-letter MCQ +
`\boxed{}` math), not any capability work. See [v2 plan](#data-mix--v2-plan).

---

## 3. Does the recurrence drive the reasoning? (cycle-scaling experiment)

The hypothesis: HRM punches above its parameter count *because* iterating the H/L modules gives
compute-depth ≫ param-count, and for reasoning that compute substitutes for parameters. Test it
by ablating the recurrence cycles on the **base** model and watching a reasoning task vs. a
knowledge control (`cycle_sweep.py` → `eval_academic.py --h-cycles/--l-cycles`).

| H×L | GSM8k (reasoning) | MMLU (control) |
|---|---|---|
| **2×3** (trained default) | **87.5%** | **44.5%** |
| 2×1 | 20.0% | 35.0% |
| 1×3 | 0.0% | 25.0% (chance) |
| 1×1 | 0.5% | 25.0% (chance) |
| 3×4 / 4×6 / 6×9 (up) | — IndexError — | — IndexError — |

**Down-ablation is decisive.** The same intervention hits the two tasks completely differently:
reasoning falls off a **cliff** (the single L-ablation 2×3→2×1 costs −67.5), while the knowledge
control **slopes gently** to chance (−9.5) — reasoning is ~**7× more cycle-sensitive**. Step-by-step
math needs a threshold of recurrent compute or it collapses entirely; shallow pattern-match MCQ
degrades smoothly. That dissociation is direct support for *"the architecture is why a 1B punches
above its weight."*

**Up-scaling is structurally blocked — itself a finding.** You can't raise cycles above the trained
2×3: `config.L_bp_steps` is a list with one entry *per trained H-cycle* (length 2), so `H_cycles=3`
indexes off the end (and `--no-cache` only addresses KV-cache sizing, not this). **HRM's compute
depth is fixed at training time** — you can't buy more reasoning at inference via config; you'd
have to extend `L_bp_steps` and retrain.

*Scope:* this is an inference-time **down**-ablation on a model trained at 2×3, so the rigorous
claim is "the trained model's reasoning is load-bearing on its recurrent compute," not "more
cycles → more reasoning" (structurally untestable here). The MMLU absolute here (44.5%) is on a
seeded subset and differs from the full-run 60.1% — within-sweep **deltas** are the signal.

---

## Training recipe (matches sapientinc `cfg_sft`)
- full-parameter, **bf16** autocast + fp32 master weights
- **lr 3e-5**, cosine decay to 10%, no warmup; AdamW (0.9, 0.95), weight_decay 0.1
- 3 epochs, `max_len` 2048, effective batch ~32, NaN guard on
- ~25k mixed examples, ~3.5 h on an A100 80GB

## Data mix & v2 plan
**v1** (`make_mixed_data.py`):

| slice | teaches | source | count |
|---|---|---|---|
| tool calls | call the right tool + args | Hermes + glaive (`convert_hermes.py`) | ~8k convos |
| instructions | follow prompt / answer directly | HuggingFaceH4/no_robots | ~5k |
| irrelevance | tools present but none fit → don't call | synthesized | ~2k |

**v2** (`run_v2.sh`) — evidence-driven, all interleaved/shuffled:
- **+ xLAM** (`convert_xlam.py`, Salesforce/xlam-60k, multi-call-biased) → lift the weak
  `parallel`/`parallel_multiple` categories;
- **+ format-discipline slice** (`make_format_slice.py`: single-letter-MCQ + `\boxed{}`-math from
  *train/aux* splits — leakage-safe) → recover the §2 format regression;
- **+ more general instructions** → preserve everyday behavior.
- **Non-destructive:** trains to a separate dir and pushes to a **separate** HF repo
  (`hrm-text-agent-v2`), with a hard guard that refuses to overwrite v1.

## Files
| File | Purpose |
|---|---|
| `agent_format.py` | prompt format (PrefixLM + `direct` condition); supports tool convos and "raw" format-slice convos |
| `convert_hermes.py` | Hermes + glaive → `{tools, turns}` schema |
| `convert_xlam.py` | **xLAM-60k** → `{tools, turns}`, normalized tool schema, multi-call bias (v2) |
| `make_mixed_data.py` | builds the mixed SFT set (+ no_robots, + irrelevance, `--extra` sources) |
| `make_format_slice.py` | **format-discipline slice** (MCQ + `\boxed{}` math, leakage-safe) (v2) |
| `train_full.py` | **full-parameter SFT trainer** (CUDA) |
| `train_lora.py` | LoRA trainer (the earlier MPS experiments) |
| `bfcl_local.py` | **BFCL eval** with the official AST checker (CUDA or MPS) |
| `eval_academic.py` | 8-benchmark academic suite; `--h-cycles/--l-cycles/--no-cache` for the cycle probe |
| `cycle_sweep.py` | **cycle-scaling experiment** (§3): sweep recurrence vs reasoning/control |
| `infer_agent.py` | agent loop: generate → parse `<tool_call>` → execute → repeat |
| `tools.py` | toy tool registry for the agent loop |
| `run.sh` / `run_v2.sh` | turnkey GPU runners (v1 / v2) |
| `run_academic_eval.sh` | turnkey academic-suite runner |

## Run it

**Train (GPU pod):**
```bash
git clone https://github.com/jasoncarreira/hrm-text-agent && cd hrm-text-agent
export HF_TOKEN=hf_...     # optional: auto-push the trained model to your HF
bash run.sh                # v1: deps → data → full-param SFT → BFCL → push
# v2 (needs xLAM gating accepted on your HF account):
bash run_v2.sh             # +xLAM +format-slice → trains to a SEPARATE repo, v1 untouched
```

**Try / evaluate it (no GPU required — runs on Apple Silicon/MPS):**
```bash
pip install -r requirements.txt
python infer_agent.py --model jasoncarreira/hrm-text-agent "What's the weather in Paris?"
python bfcl_local.py --model jasoncarreira/hrm-text-agent --dump errs.jsonl   # --live for live_* cats
```

**Reproduce the cycle experiment (on the base model):**
```bash
python cycle_sweep.py --model sapientinc/HRM-Text-1B --limit 200          # down-ablation (the result)
python cycle_sweep.py --model sapientinc/HRM-Text-1B --limit 100 --up     # up-scaling (will IndexError — see §3)
```

## Gotchas (hard-won)
- **CUDA:** `transformers>=5.9` (native `hrm_text`) needs **torch≥2.7 + matching torchvision**
  (else `torchvision::nms` error). HRM full-FT **can't use gradient checkpointing on CUDA** (the
  recurrent recompute fails the determinism check) — train with `--no-grad-checkpoint` (fits on
  80 GB). `run.sh` handles the torch upgrade.
- **Apple MPS:** keep fp32 + gradient checkpointing; `--bf16` is ~5× *slower* (Metal CPU-fallback)
  and `--no-grad-checkpoint` ~27× slower (RAM swap). Inference runs ~4.4 s/example.
- **Recurrence cycles are fixed at training** (§3): `H_cycles`/`L_cycles` can be *reduced* at
  inference but not raised above the trained 2×3 (`L_bp_steps` length).

---
🤖 Built with Claude Code — including a second Claude running on the GPU pod driving training and eval.
