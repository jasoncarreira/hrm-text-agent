#!/usr/bin/env python3
"""LoRA SFT for tool-use on sapientinc/HRM-Text-1B (Apple Silicon / MPS friendly).

Trains a low-rank adapter over the attention + MLP projections of both the H and
L recurrent stacks, on (prefix -> target) tool-use examples. Defaults to fp32 for
MPS numerical stability (slower but robust) since this is meant for an unattended
overnight run; pass --bf16 to try half precision.

Smoke test (a handful of steps, confirms forward+backward survive on MPS):
    python train_lora.py --max-steps 6 --log-every 1

Full run:
    python make_dataset.py
    python train_lora.py --epochs 3 --out-dir adapters/tooluse-v1
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU-fallback for unimplemented MPS ops

import argparse
import json
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from agent_format import expand_conversation, encode_example

MODEL_ID = "sapientinc/HRM-Text-1B"
# Both stacks expose these via L_module.* and H_module.*; PEFT matches by suffix.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device(req: str) -> str:
    if req != "auto":
        return req
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ToolUseDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int):
        self.examples = []
        skipped = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                convo = json.loads(line)
                for prefix, target in expand_conversation(convo):
                    enc = encode_example(tokenizer, prefix, target, max_len)
                    if enc is None:
                        skipped += 1
                        continue
                    self.examples.append(enc)
        print(f"[data] {len(self.examples)} training examples"
              + (f" ({skipped} skipped: target too long)" if skipped else ""))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id: int):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "attention_mask": [], "token_type_ids": [], "labels": []}
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
            out["token_type_ids"].append(b["token_type_ids"] + [0] * pad)
            out["labels"].append(b["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data", "tool_sft.jsonl"))
    p.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "adapters", "tooluse-v1"))
    p.add_argument("--resume", default=None,
                   help="resume from a checkpoint dir (restores adapter + optimizer + LR schedule + step). "
                        "Re-run with the SAME --data/--max-len/--max-examples/--seed/--epochs.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke test)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", nargs="+", default=TARGET_MODULES,
                   help="LoRA target module suffixes. Default = attn+MLP of both recurrent stacks. "
                        "Use 'lm_head' to adapt ONLY the output head (applied once, outside the "
                        "recurrence) — avoids the per-cycle compounding that destabilizes the model.")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--max-examples", type=int, default=-1,
                   help="cap training examples (after build) to control runtime: time ~= N * 4.4s on M2 Max")
    p.add_argument("--bf16", action="store_true",
                   help="WARNING: ~5x SLOWER on Apple MPS (bf16 ops fall back to CPU). CUDA only.")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0, help="save a checkpoint every N steps (0=only at end)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="WARNING: ~27x SLOWER on MPS — the recurrent unroll's activations exceed RAM and swap.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    dtype = torch.bfloat16 if (args.bf16 and device != "cpu") else torch.float32
    print(f"[setup] device={device} dtype={dtype} r={args.lora_r} alpha={args.lora_alpha} "
          f"lr={args.lr} eff_batch={args.batch_size * args.grad_accum}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
    model.config.use_cache = False

    if args.resume:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        print(f"[resume] loaded adapter from {args.resume}")
    else:
        lora = LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
                          lora_dropout=args.lora_dropout, bias="none", target_modules=args.target_modules)
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if not args.no_grad_checkpoint:
        try:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            print("[setup] gradient checkpointing enabled")
        except Exception as e:  # noqa: BLE001
            print(f"[setup] gradient checkpointing unavailable ({e}); continuing "
                  "(HRM's truncated gradient already bounds backward memory)")

    model.to(device).train()

    ds = ToolUseDataset(args.data, tok, args.max_len)
    if len(ds) == 0:
        raise SystemExit("no training examples — did you run make_dataset.py?")
    if 0 < args.max_examples < len(ds):
        import random as _random
        _random.Random(args.seed).shuffle(ds.examples)
        ds.examples = ds.examples[:args.max_examples]
        print(f"[data] capped to {len(ds.examples)} examples (~{len(ds.examples) * 4.4 / 3600:.1f} h on M2 Max)")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=make_collate(tok.pad_token_id))

    steps_per_epoch = math.ceil(len(dl) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    # If resuming, read the ORIGINAL total_steps first so the LR-schedule shape (built
    # from total_steps) matches — LambdaLR doesn't serialize its lambda, so we must
    # reconstruct it with the same total_steps before loading the scheduler state.
    step, start_epoch, resume_state = 0, 0, None
    if args.resume:
        sp = os.path.join(args.resume, "trainer_state.pt")
        if os.path.exists(sp):
            resume_state = torch.load(sp, map_location="cpu")
            total_steps = resume_state.get("total_steps", total_steps)
            step = resume_state.get("step", 0)
            start_epoch = resume_state.get("epoch", 0)
        else:
            print(f"[resume] WARNING: no trainer_state.pt in {args.resume}; adapter loaded but "
                  "optimizer/LR/step restart from zero.")

    sched = get_linear_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    if resume_state is not None:
        try:
            optim.load_state_dict(resume_state["optimizer"])
            sched.load_state_dict(resume_state["scheduler"])
        except Exception as e:  # noqa: BLE001
            print(f"[resume] WARNING: couldn't load optimizer/scheduler ({e}); "
                  "continuing with fresh optimizer (LR may not be continuous).")
        for st in optim.state.values():  # loaded on CPU -> move to train device
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
        print(f"[resume] continuing from step {step}/{total_steps} (epoch {start_epoch})")
    print(f"[train] target {total_steps} optimizer steps over {args.epochs} epoch(s)")

    def save_checkpoint(path):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tok.save_pretrained(path)
        torch.save({"optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                    "step": step, "epoch": epoch, "total_steps": total_steps},
                   os.path.join(path, "trainer_state.pt"))

    epoch = start_epoch
    done = step >= total_steps
    nan_mb, nan_steps = 0, 0
    for epoch in range(start_epoch, args.epochs):
        if done:
            break
        optim.zero_grad(set_to_none=True)
        running, running_n = 0.0, 0
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if loss is None:  # fallback: manual masked CE if model doesn't return loss
                logits = out.logits[:, :-1, :].float()
                labels = batch["labels"][:, 1:]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            # NaN guard: HRM's deep recurrence can overflow on an occasional batch (fp32/MPS).
            # Never let a non-finite loss/grad reach the weights — skip it instead of poisoning the run.
            if torch.isfinite(loss):
                (loss / args.grad_accum).backward()
                running += loss.item()
                running_n += 1
            else:
                nan_mb += 1

            if (i + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                if torch.isfinite(gnorm):
                    optim.step()
                else:
                    nan_steps += 1  # bad grad slipped through; drop the whole step
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    avg = running / max(1, running_n)
                    extra = f"  [skipped {nan_steps} steps / {nan_mb} mb]" if nan_mb else ""
                    print(f"  epoch {epoch} step {step}/{total_steps} loss {avg:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e}{extra}")
                    running, running_n = 0.0, 0
                if args.save_every and step % args.save_every == 0:
                    ck = f"{args.out_dir}-step{step}"
                    save_checkpoint(ck)
                    print(f"  [ckpt] saved {ck}  (resume: --resume {ck})")
                if step >= total_steps:
                    done = True
                    break

    save_checkpoint(args.out_dir)
    with open(os.path.join(args.out_dir, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[done] adapter + trainer_state saved -> {args.out_dir}")


if __name__ == "__main__":
    main()
