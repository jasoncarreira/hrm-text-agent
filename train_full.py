#!/usr/bin/env python3
"""Full-parameter SFT for HRM-Text-1B tool-use (NVIDIA GPU / CUDA).

Trains ALL weights (no LoRA), matching sapientinc's official SFT recipe:
lr 3e-5, AdamW(0.9, 0.95) wd 0.1, cosine decay to 10%, bf16. PrefixLM masking
(loss on response only) + the `direct` condition. fp32 master weights + bf16
autocast — fits comfortably on a 40GB A100.

    python train_full.py --data data/sft_mixed.jsonl --epochs 3 --out-dir models/hrm-tooluse-full
"""
import argparse
import contextlib
import json
import math
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_format import encode_example, expand_conversation

MODEL_ID = "sapientinc/HRM-Text-1B"


def pick_device(req):
    if req != "auto":
        return req
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SFTDataset(Dataset):
    def __init__(self, path, tok, max_len):
        self.ex = []
        skipped = 0
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            for pre, tgt in expand_conversation(json.loads(line)):
                enc = encode_example(tok, pre, tgt, max_len)
                if enc is None:
                    skipped += 1
                    continue
                self.ex.append(enc)
        print(f"[data] {len(self.ex)} examples ({skipped} skipped: > max_len)")

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return self.ex[i]


def make_collate(pad_id):
    def collate(batch):
        m = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "labels": []}
        for b in batch:
            pad = m - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["token_type_ids"].append(b["token_type_ids"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
    return collate


def cosine_min(step, total, min_ratio, warmup):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="models/hrm-tooluse-full")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)              # sapientinc cfg_sft
    p.add_argument("--weight-decay", type=float, default=0.1)     # sapientinc cfg_sft
    p.add_argument("--min-lr-ratio", type=float, default=0.1)     # sapientinc cfg_sft
    p.add_argument("--warmup-ratio", type=float, default=0.0)     # sapientinc cfg_sft
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--max-examples", type=int, default=-1)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    use_bf16 = device == "cuda"
    print(f"[setup] device={device} bf16_autocast={use_bf16} lr={args.lr} "
          f"eff_batch={args.batch_size * args.grad_accum}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)  # fp32 master
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
        print("[setup] gradient checkpointing on")
    except Exception as e:  # noqa: BLE001
        print(f"[setup] gradient checkpointing unavailable: {e}")
    model.to(device).train()
    print(f"[setup] FULL fine-tune: "
          f"{sum(pp.numel() for pp in model.parameters() if pp.requires_grad)/1e9:.3f}B trainable")

    ds = SFTDataset(args.data, tok, args.max_len)
    if 0 < args.max_examples < len(ds):
        import random as _r
        _r.Random(args.seed).shuffle(ds.ex)
        ds.ex = ds.ex[:args.max_examples]
        print(f"[data] capped to {len(ds.ex)}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=make_collate(tok.pad_token_id))

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: cosine_min(s, total_steps, args.min_lr_ratio, warmup))
    print(f"[train] {total_steps} optimizer steps over {args.epochs} epoch(s)")

    def save(path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tok.save_pretrained(path)

    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext())
    step, nan_mb, nan_steps = 0, 0, 0
    for epoch in range(args.epochs):
        optim.zero_grad(set_to_none=True)
        running, running_n = 0.0, 0
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast:
                loss = model(**batch).loss
            if loss is not None and torch.isfinite(loss):
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
            else:
                nan_mb += 1
            if (i + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if torch.isfinite(gn):
                    optim.step()
                else:
                    nan_steps += 1
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    avg = running / max(1, running_n)
                    extra = f"  [skipped {nan_steps} st / {nan_mb} mb]" if nan_mb else ""
                    print(f"  epoch {epoch} step {step}/{total_steps} loss {avg:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}{extra}")
                    running, running_n = 0.0, 0
                if args.save_every and step % args.save_every == 0:
                    save(f"{args.out_dir}-step{step}")
                    print(f"  [ckpt] {args.out_dir}-step{step}")
    save(args.out_dir)
    print(f"[done] full model saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
