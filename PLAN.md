# Plan: full-parameter SFT for HRM-Text-1B tool-use

## Background
LoRA experiments hit a wall: LoRA on HRM's recurrent layers destabilizes the model
(the delta is amplified across the weight-shared recurrence cycles), and `lm_head`-only
LoRA is stable but too weak to override the base model's instinct to *answer* instead of
*call* (plateaued at BFCL `simple` 8%). The model's own repo confirms the fix: their
official SFT recipe is **full-parameter, never LoRA**. Two more findings from their docs:
we were missing the **`direct` condition token** (prompts must open
`<|im_start|><|object_ref_start|>...`), and the base has **no** instruction-tuning, so it
ignores the "use tools" instruction.

## Goal & success criteria
- **Primary:** BFCL `simple` AST well above 8% (target ~20–40%), `irrelevance` ≥ 80%.
- Baselines: base = 0% simple; lm_head-LoRA = 8% simple.
- Stretch: `multiple` off zero (`parallel` likely stays hard — 1B capacity ceiling).

## Approach (the three fixes, together)
1. **Full-parameter SFT** instead of LoRA — strong enough to rewrite the answer-instinct, and stable (gently moves the real weights vs. an amplified add-on).
2. **`direct` condition** (`<|object_ref_start|>`) on every prompt — on-distribution for structured output.
3. **Mixed data** — tool calls + general instructions + irrelevance — teaches *when to call* and *when not to*.

## Data mix (~19k examples)
| slice | teaches | source | count |
|---|---|---|---|
| tool-call | call right tool + args | Hermes + glaive (our `convert_hermes.py`) | 12,000 |
| instruction | follow prompt / answer directly | HuggingFaceH4/no_robots (human-written, CC-BY-NC; hobby use) | 5,000 |
| irrelevance | tools present but none fit → don't call | synthesized (instruction + random tools → plain answer) | 2,000 |

PrefixLM masking (loss on response only) + `token_type_ids` on the prefix.

## Training config (matched to sapientinc `cfg_sft.yaml`)
- **full-parameter** (all ~1.18B weights), **bf16** autocast + fp32 master weights
- **lr 3e-5**, cosine decay to 10% (`min_lr_ratio 0.1`), no warmup
- AdamW betas **(0.9, 0.95)**, weight_decay **0.1**
- **3 epochs** (their default is 5; 3 = cost/quality balance)
- effective batch ~32 (batch 4 × grad-accum 8), **max_len 2048**, gradient checkpointing on
- NaN guard (skip non-finite loss/grad)

## Hardware & cost
- **A100 40 GB** (full fp32-Adam, no memory tricks). ~$1.3–2/hr × ~2–2.5 h ≈ **$3–5**.

## Run (on a CUDA pod)
```bash
git clone <public repo> && cd hrm-text-tooluse
bash run.sh        # installs deps, builds data, trains, evals, prints BFCL numbers
# optional: export HF_TOKEN=...  to auto-push the model to jasoncarreira/hrm-text-agent
#           (override the target with HF_REPO=<user>/<repo>)
```
Files: `convert_hermes.py` → `make_mixed_data.py` → `train_full.py` → `bfcl_local.py`.

## Honest expectations
Still a 1B base — expect `simple` to climb markedly and `irrelevance` to stay healthy;
`multiple`/`parallel` may remain weak (capacity, not recipe). Mix ratio may need a tweak
after seeing results.
