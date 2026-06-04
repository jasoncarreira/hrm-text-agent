# HRM-Text-1B — tool-use LoRA fine-tuning (Apple Silicon)

Teach [`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B)
(a 1B **base** model) to call tools, via a LoRA adapter, on an M-series Mac (MPS).

> **Reality check.** This is a 1B *base, pre-alignment* model. LoRA can teach the
> tool-call *format and behavior pattern*, but it cannot add capability the base
> lacks. Expect a fun proof-of-concept that emits well-formed tool calls on
> in-distribution prompts — not a robust, generalizing agent. The single biggest
> lever on quality is **dataset size/quality**, not the LoRA hyperparameters.

## Run log & findings (2026-06-03/04)

Trained on ~8k Hermes+glaive examples (1 epoch, fp32/MPS). Three runs:

| run | LoRA target | α (scale) | outcome |
|---|---|---|---|
| v1 | attn+MLP, both recurrent stacks | 32 (2.0) | **collapsed** — output distribution went ~uniform by step ~20; NaN earlier (pre-guard) |
| v2 | attn+MLP, both stacks | 8 (0.5) | **delayed collapse** — sharp at step 100, flattening by step 200 |
| v3 | **`lm_head` only** | 32 | **stable + learns to call** — final BFCL `simple` 8% |

**Root cause (the key finding):** HRM reuses each physical layer across H×L recurrence
cycles, so a LoRA delta on those layers is injected *many times per forward* and the
cumulative perturbation collapses the output logits toward uniform. **Training loss hides
it** (loss is dominated by easy in-JSON continuation tokens; the broken first-token
transition barely moves the average — v1 hit loss 0.5 while generating garbage). Lowering
α only delays the collapse; it doesn't prevent it.

**The fix:** adapt **`lm_head` only** — it's applied *once*, outside the recurrence, so
there's no per-cycle compounding. v3 stayed stable and `<tool_call>` climbed from rank
~39,500 (base) to rank 0 over training.

**Final v3 result (BFCL, official AST checker, 100/category):**
`simple 8%` · `multiple 0%` · `parallel 0%` · `parallel_multiple 0%` · `irrelevance 89%`.
Failure breakdown: **80% of failures = answered directly (no call)**; 20% = called but wrong
(mostly wrong call-count on multiple/parallel, some missing required args).

**Verdict:** a 1B *base* model + `lm_head` LoRA learns the tool-call *format* and gets a
small fraction of *simple* single-calls correct, but mostly defaults to answering directly
and can't do multiple/parallel calls. Partial success against the documented 1B-base ceiling.

**MPS gotchas measured here:** keep fp32 + gradient-checkpointing (defaults) — `--bf16` is
~5× *slower* on Metal (CPU fallback) and `--no-grad-checkpoint` ~27× slower (RAM swap). Rate
~4.4 s/example at normal priority; `taskpolicy -b` (low priority) ~3× slower.

### What I'd try next (to push past 8%)
- Higher-capacity output adapt: `lm_head` at r=32/64, or `lm_head` + `embed_tokens`.
- Add a tiny number of *out-of-recurrence* layers if any exist (most are in the stacks).
- Instruction-tune first (the base defaults to *answering*, not calling), or upweight the
  first/`<tool_call>` token in the loss so it learns to call rather than answer.
- More data / 2–3 epochs; or accept the 1B-base ceiling and try a larger base.

## Files

| File | Purpose |
|---|---|
| `tools.py` | Toy tool registry — JSON schemas (rendered into prompts) + Python impls (executed at inference). |
| `agent_format.py` | Prompt format + PrefixLM example construction (bidirectional prefix → causal target). |
| `make_dataset.py` | Writes a 16-conversation **seed** set to `data/tool_sft.jsonl`. Template to scale up. |
| `convert_hermes.py` | Converts NousResearch Hermes function-calling into our schema (thousands of examples). |
| `train_lora.py` | LoRA SFT training loop (fp32/MPS by default; `--bf16` opt-in). |
| `infer_agent.py` | Agent loop: generate → parse `<tool_call>` → execute → feed result back → repeat. |
| `bfcl_local.py` | **Benchmark.** Runs BFCL (Berkeley Function Calling Leaderboard) AST + relevance categories locally on MPS with BFCL's *official* scorer. |
| `eval_agent.py` | Quick local-tool sanity check (a handful of prompts). Secondary to `bfcl_local.py`. |
| `generate.py` | Plain text completion with the base model (no tools). |

## Workflow

```bash
source .venv/bin/activate

# 1. build the dataset (Hermes func-calling + glaive-5k + local seed)
python make_dataset.py                 # 16 local-tool seed convos
python convert_hermes.py               # ~8k convos -> data/tool_sft_hermes.jsonl (no held-out;
                                       #   we evaluate on BFCL). --limit N caps volume.
# ~18k training examples after the multi-turn split.

# 2a. smoke test (confirms forward+backward work on MPS — ~1.5 min)
python train_lora.py --data data/tool_sft_hermes.jsonl --max-steps 3 --grad-accum 2 \
    --max-len 2048 --log-every 1 --out-dir adapters/smoketest

# 2b. overnight run. caffeinate keeps the Mac awake; --save-every checkpoints; --max-examples caps time.
caffeinate -i python train_lora.py --data data/tool_sft_hermes.jsonl \
    --epochs 1 --max-len 2048 --max-examples 8000 --save-every 500 --out-dir adapters/tooluse-v1
# MEASURED rate: ~4.4 s/example (fp32 + grad-checkpointing — the optimal config on MPS).
#   time ~= (examples) x 4.4s.  Full data (~18.9k ex) = ~23 h.  --max-examples 8000 = ~10 h, 6000 = ~7 h.
# DO NOT add --bf16 (benchmarked ~5x SLOWER on MPS) or --no-grad-checkpoint (~27x slower, swaps RAM).
# --save-every 500 (steps) checkpoints often, so you can stop anytime and keep/benchmark the latest.
#
# RESUME after a kill: each checkpoint saves the adapter + optimizer + LR schedule + step count.
#   python train_lora.py --data data/tool_sft_hermes.jsonl --epochs 1 --max-len 2048 \
#       --max-examples 8000 --resume adapters/tooluse-v1-step1500 --out-dir adapters/tooluse-v1
#   Pass the SAME --data/--max-len/--max-examples/--epochs/--seed so the step accounting lines up.
#   (Optimizer momentum + LR continue exactly; data re-shuffles for the remaining steps — fine for SFT.)
#
# LOW-PRIORITY (yields CPU+GPU+I/O to your foreground work via macOS background QoS):
#   prefix the command with `taskpolicy -b`, e.g. fire-and-forget with a log:
#   nohup caffeinate -i taskpolicy -b python train_lora.py ...args... > train.log 2>&1 &
#   (demote a running job: `taskpolicy -b -p <pid>` and/or `renice +20 -p <pid>`)

# 3. BENCHMARK on BFCL (official scorer, local MPS, base-vs-adapter delta)
python bfcl_local.py --adapter adapters/tooluse-v1 --base-too
#   --dump errs.jsonl   to log every incorrect prediction for error analysis
#   --limit 50          for a quick per-category sample
#   --live              to also run the live_* categories

# 4. try it interactively
python infer_agent.py --adapter adapters/tooluse-v1 "What's the weather in Paris?"
python infer_agent.py "What is 23 * 47?"          # base model, for comparison
```

## Benchmark: BFCL (the bench of record)

`bfcl_local.py` runs the **Berkeley Function Calling Leaderboard** categories that
need no GPU/API/execution — `simple, multiple, parallel, parallel_multiple,
irrelevance` (add `--live` for the live_* set) — and scores them with BFCL's
**official `ast_checker`**, so the numbers are leaderboard-comparable. Generation
runs on your MPS; only the model registry (used for dotted names, which the core
categories lack) is stubbed out to avoid BFCL's CUDA/handler dependencies.

What each category measures: correct function name, all required args present, no
hallucinated params, and per-type value matching (with multiple acceptable values);
`irrelevance` checks the model correctly emits **no** call. Base-model baseline is
~0% on the AST categories and ~100% on irrelevance (it never calls anything) — so
after training, watch the AST categories climb while irrelevance stays high (that
balance is the real signal: calling correctly *and* knowing when not to).

## Benchmark: academic suite (Math, MMLU, …) — `eval_academic.py`

`eval_academic.py` runs the **same general-capability benchmarks sapientinc/HRM-Text
reports** — `GSM8k, MATH, MMLU, ARC, HellaSwag, Winogrande, BoolQ, DROP` — against
**any HF-format HRM checkpoint** (the base `sapientinc/HRM-Text-1B` *or* our local
fine-tune). Their official harness only evaluates their native-format checkpoints
(`SimpleEngine`) or standard-arch baselines via vLLM (which can't load HRM); this
reimplements their prompting + scoring on top of `transformers` so it runs on HF
models. Prompt/scoring logic is ported from their `evaluation/benchmarks.py`.

It applies the **HRM condition prefix** the model expects (per the HRM-Text-1B model
card): math/reasoning uses the composite `synth,cot`; NLP/MCQ uses `direct` + few-shot.
Every prompt is wrapped `<|im_start|>{condition}{task}<|im_end|>`, generated to
`<|box_end|>`, with `token_type_ids=1` over the prompt (PrefixLM prefill).

```bash
# base model (reference) vs our fine-tune — same harness, apples-to-apples
python eval_academic.py --model sapientinc/HRM-Text-1B --out base_acad.json
python eval_academic.py --model models/hrm-tooluse-full --out ours_acad.json
python eval_academic.py --model models/hrm-tooluse-full --benchmarks GSM8k,MATH --limit 50  # quick
```

Most useful as a **forgetting check**: our model is a *tool-use* SFT, so compare the
base-vs-fine-tune delta to see whether tool training eroded math/reasoning. Numbers
won't match the repo's headline figures (lm-eval-style prompts differ from their
native harness), but the base-vs-ours comparison is consistent. Needs a GPU for the
generative sets (GSM8k/MATH); the MCQ sets are single-token and cheap. Uses HF
`datasets` (downloaded on demand) + `math_verify` for MATH grading.

## The format (what the model is trained on)

Each conversation is split into single **(prefix → target)** examples so every
example stays a clean PrefixLM block (the documented HRM regime):

```
<|im_start|>
You are a function-calling assistant. ...
Available tools:
<tools>
{...json schema per tool...}
</tools>

User: What's the weather in Paris?<|im_end|>          ← prefix: token_type_ids=1, masked from loss
<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call><|box_end|>
                                                      ← target: token_type_ids=0, contributes to loss
```

`<tool_call>`, `</tool_call>`, `<|im_start|>`, `<|im_end|>`, `<|box_end|>` are all
**single dedicated tokens** in this tokenizer — the vocab was built with function
calling in mind, which is why this works as well as it does.

## Scaling to a real run (the important part)

16 seed conversations will **memorize, not generalize**. For a useful adapter,
grow `data/tool_sft.jsonl` to thousands of diverse conversations:

- Add more by hand in `make_dataset.py` (same schema), or
- Reformat a public function-calling set into the conversation schema:
  **Glaive function-calling v2**, **NousResearch Hermes function-calling**,
  **Salesforce xLAM / APIGen**, **ToolACE**.

Keep a healthy fraction of **no-tool** conversations (answer directly) so the
model learns restraint, and vary tool names/arg shapes so it doesn't latch onto
one pattern.

## Notes / gotchas

- **Keep the defaults: fp32 + gradient checkpointing.** Benchmarked on M2 Max,
  this is the *fastest* config. `--bf16` is ~5x SLOWER (bf16 ops fall back to CPU
  on Metal) and `--no-grad-checkpoint` is ~27x slower (the recurrent unroll's
  activations exceed RAM and the machine swaps). Both flags exist only for CUDA.
- **Control runtime with `--max-examples`**, not precision/checkpoint flags. The
  rate is fixed at ~4.4 s/example, so e.g. `--max-examples 8000` ≈ 10 h.
- **Low LR (1e-4).** HRM reuses each physical layer across recurrent cycles, so a
  LoRA delta is applied many times per forward — changes get amplified. If
  training is unstable, lower the LR or `--lora-alpha` before anything else.
- HRM trains with a **truncated gradient** (`L_bp_cycles` in the config), which
  bounds backward memory — a 1B LoRA fits comfortably in 32 GB.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so any op MPS lacks falls back to CPU
  instead of crashing (slower, but it won't die mid-run).
- Training is CPU/GPU-portable: `--device cuda` on a rented GPU runs the same
  code far faster; copy the adapter back and run inference locally.

Delete `adapters/smoketest` whenever — it was only a 4-step pipeline check.
