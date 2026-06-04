# HRM-Text-1B — tool-use fine-tuning

Fine-tune [`sapientinc/HRM-Text-1B`](https://huggingface.co/sapientinc/HRM-Text-1B) — a 1B
**base** (pre-alignment) model — to do **function / tool calling**, evaluated with the
official **Berkeley Function Calling Leaderboard (BFCL)** AST checker.

**Headline:** full-parameter SFT took held-out BFCL `simple` from **0% (base) → 61.5%** (full
test set), with the multi-call categories well off zero. Trained model:
[`jasoncarreira/hrm-text-agent`](https://huggingface.co/jasoncarreira/hrm-text-agent).

## Results (BFCL v4, official AST checker, full test sets)

| Category | n | Base | LoRA (`lm_head`) | **Full-param SFT** |
|---|---|---|---|---|
| simple | 400 | 0% | 8% | **61.5%** |
| multiple | 200 | 0% | 0% | **53.5%** |
| parallel | 200 | 0% | 0% | **37.5%** |
| parallel_multiple | 200 | 0% | 0% | **28.0%** |
| irrelevance | 240 | 100% | 89% | **80.8%** |

Non-live AST aggregate ≈ **48%** (count-weighted). `simple` 0→61.5% on **held-out** tools is the
win — full-FT cured the base model's instinct to *answer* instead of *call*, which LoRA never
could. (Full-SFT column = full sets; base/LoRA columns are earlier 100-sample reads.)

**Where that lands among small models** (BFCL non-live AST): comfortably above every *generic*
1B instruct (Llama-3.2-1B ~38, Gemma-3-1b ~20, Falcon3-1B ~9), but below the best purpose-built
1B (xLAM-2-1b-fc-r ~69) and the 3B FC models — i.e. strong for a 1B *base* + SFT, not yet
3B-level. Next lever: more parallel/multi-call data (xLAM/ToolACE) to lift the weak categories.

## How we got there (the useful findings)

1. **LoRA on the recurrent layers → unstable.** HRM reuses each physical layer across its
   H×L recurrence cycles, so a LoRA delta is injected *many times per forward*; the cumulative
   perturbation collapses the output distribution toward uniform. Training loss **hides** it
   (dominated by easy in-JSON tokens — v1 hit loss 0.5 while generating garbage). Lowering the
   LoRA scale only *delays* the collapse.
2. **LoRA on `lm_head` only → stable but weak (8%).** Adapting just the output head (applied
   once, outside the recurrence) is stable, but too weak to override the base's answer-instinct;
   it mostly answered in prose instead of calling.
3. **Full-parameter SFT → 70%.** Matches sapientinc's official recipe (their SFT is
   full-parameter, never LoRA): it moves the real weights gently (the recurrence stays stable)
   and is strong enough to rewrite the behavior. Three changes together did it:
   - **full-parameter SFT** (not LoRA),
   - the model's **`direct` condition token** (`<|object_ref_start|>` opening the
     `<|im_start|>…<|im_end|>` prompt — the documented mode for structured output; we had been
     omitting it),
   - a **mixed dataset** (tool calls + general instructions + irrelevance). The base has no
     instruction-tuning, so the instruction/irrelevance data teaches *when to call vs. answer*
     and keeps `irrelevance` from collapsing.

## Training recipe (matches sapientinc `cfg_sft`)
- full-parameter, **bf16** autocast + fp32 master weights
- **lr 3e-5**, cosine decay to 10%, no warmup; AdamW (0.9, 0.95), weight_decay 0.1
- 3 epochs, `max_len` 2048, effective batch ~32, NaN guard on
- ~25k mixed examples, ~3.5 h on an A100 80GB

## Data mix (`make_mixed_data.py`)
| slice | teaches | source | count |
|---|---|---|---|
| tool calls | call right tool + args | Hermes + glaive (`convert_hermes.py`) | ~8k convos |
| instructions | follow prompt / answer directly | HuggingFaceH4/no_robots | ~5k |
| irrelevance | tools present but none fit → don't call | synthesized | ~2k |

## Files
| File | Purpose |
|---|---|
| `convert_hermes.py` | Hermes + glaive → `{tools, turns}` schema |
| `make_mixed_data.py` | builds the mixed SFT set (+ no_robots, + irrelevance) |
| `agent_format.py` | prompt format (PrefixLM + `direct` condition) + example construction |
| `train_full.py` | **full-parameter SFT trainer** (CUDA) |
| `train_lora.py` | LoRA trainer (the earlier MPS experiments) |
| `bfcl_local.py` | **BFCL eval** with the official AST checker (CUDA or MPS) |
| `eval_academic.py` | academic suite: GSM8K, MATH, MMLU, ARC-C, HellaSwag, Winogrande, BoolQ, DROP |
| `infer_agent.py` | agent loop: generate → parse `<tool_call>` → execute → repeat |
| `tools.py` | toy tool registry for the agent loop |
| `run.sh`, `run_academic_eval.sh` | turnkey GPU runners |

## Train (GPU pod)
```bash
git clone https://github.com/jasoncarreira/hrm-text-agent && cd hrm-text-agent
export HF_TOKEN=hf_...            # optional: auto-push the trained model to your HF
bash run.sh                       # deps → data → full-param SFT → BFCL eval → push
```

## Try / evaluate it (no GPU required — runs on Apple Silicon/MPS)
```bash
pip install -r requirements.txt
# chat with the tool-calling model:
python infer_agent.py --model jasoncarreira/hrm-text-agent "What's the weather in Paris?"
# full BFCL eval (add --live for the live_* categories):
python bfcl_local.py --model jasoncarreira/hrm-text-agent --dump errs.jsonl
```

## Gotchas (hard-won)
- **CUDA:** `transformers>=5.9` (native `hrm_text`) needs **torch≥2.7 + matching torchvision**
  (else `torchvision::nms` error). HRM full-FT **can't use gradient checkpointing on CUDA**
  (the recurrent recompute fails the determinism check) — train without it (fits on 80 GB).
  `run.sh` handles the torch upgrade.
- **Apple MPS:** keep fp32 + gradient checkpointing; `--bf16` is ~5× *slower* (Metal CPU-fallback)
  and `--no-grad-checkpoint` ~27× slower (RAM swap). LoRA training / inference runs ~4.4 s/example.

---
🤖 Built with Claude Code — including a second Claude running on the GPU pod driving training.
