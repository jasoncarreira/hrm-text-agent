#!/usr/bin/env python3
"""Run text generation with sapientinc/HRM-Text-1B on Apple Silicon (MPS).

HRM-Text-1B is a *base* (pre-alignment) model: a raw text-completion model,
not a chat assistant. Quality is best when the prompt is wrapped in the
control tokens the model saw during pretraining, and when the prompt span is
marked as a bidirectional prefix via token_type_ids (see the model card).

Usage:
    python generate.py "Explain why the sky is blue."
    python generate.py --max-new-tokens 128 --device cpu "Once upon a time"
"""
# Let any op that MPS doesn't implement fall back to CPU instead of crashing.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_ID = "sapientinc/HRM-Text-1B"

# Control tokens that condition the base model (from the model card example:
# the "synth,cot" composite that nudges it toward reasoning-style completions).
CONDITION = "<|quad_end|><|object_ref_end|>"


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text with HRM-Text-1B.")
    p.add_argument("prompt", nargs="?", default="Explain why the sky is blue.",
                   help="The text prompt to complete.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                   help="bf16 matches the training precision; fp32 is the safe fallback.")
    p.add_argument("--sample", action="store_true",
                   help="Use sampling (temperature) instead of greedy decoding.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--raw", action="store_true",
                   help="Send the prompt verbatim, without the control-token wrapper.")
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    # CPU bf16 is slow/unsupported for many ops -> use fp32 there.
    if device == "cpu":
        dtype = torch.float32

    print(f"[load] model={MODEL_ID} device={device} dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model = model.to(device).eval()

    text = args.prompt if args.raw else f"<|im_start|>{CONDITION}{args.prompt}<|im_end|>"
    inputs = tok(text, return_tensors="pt").to(device)
    # Mark the whole prompt as a bidirectional prefix block (PrefixLM). Omitting
    # this falls back to pure causal attention and gives noticeably worse logits.
    inputs["token_type_ids"] = torch.ones_like(inputs["input_ids"])

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
    if args.sample:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)

    streamer = TextStreamer(tok, skip_prompt=False, skip_special_tokens=False)
    print("\n[generate]\n")
    with torch.no_grad():
        model.generate(**inputs, streamer=streamer, **gen_kwargs)
    print("\n[done]")


if __name__ == "__main__":
    main()
