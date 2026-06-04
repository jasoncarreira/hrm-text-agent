#!/usr/bin/env python3
"""Academic benchmark harness for HRM-Text (HuggingFace format), adapted to run on
ANY HF-format HRM checkpoint — the base `sapientinc/HRM-Text-1B` or our fine-tune.

Mirrors the benchmarks sapientinc/HRM-Text reports (GSM8K, MATH, MMLU, ARC-C,
HellaSwag, Winogrande, BoolQ, DROP). Their official harness uses a native-checkpoint
engine (`SimpleEngine`); this reimplements the *same prompting + scoring* on top of
`transformers` so it runs against HF models. Benchmark prompt/scoring logic is ported
from sapientinc/HRM-Text `evaluation/benchmarks.py` (Apache-2.0).

The HRM-specific prompting (from the HRM-Text-1B model card):
    prompt = <|im_start|>{condition_tokens}{task}<|im_end|>      then generate to <|box_end|>
    token_type_ids = 1 over the whole prompt (PrefixLM bidirectional prefill).
Condition tags -> tokenizer specials:
    direct -> <|object_ref_start|>   cot -> <|object_ref_end|>
    noisy  -> <|quad_start|>         synth -> <|quad_end|>
Math/reasoning uses the composite `synth,cot`; NLP/MCQ uses `direct` + few-shot.

  python eval_academic.py --model sapientinc/HRM-Text-1B            # base
  python eval_academic.py --model models/hrm-tooluse-full           # our fine-tune
  python eval_academic.py --model <m> --benchmarks GSM8k,MATH --limit 50
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict

import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- HRM prompt envelope + condition tokens (HRM-Text-1B model card) ----
BOQ, EOQ, EOA = "<|im_start|>", "<|im_end|>", "<|box_end|>"
CONDITION_MAP = {
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}

MODEL_ID = "sapientinc/HRM-Text-1B"


# ============================ generation engine ============================
class HFEngine:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
        self.model.config.use_cache = True
        self.device = device
        self.eos_id = self.tok.convert_tokens_to_ids(EOA)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _wrap(self, condition: str, prompt: str) -> str:
        cond = "".join(CONDITION_MAP[c] for c in condition.split(",")) if condition else ""
        return f"{BOQ}{cond}{prompt}{EOQ}"

    @torch.inference_mode()
    def generate(self, prompts: list[str], condition: str, max_new_tokens: int,
                 temperature: float = 0.0, batch_size: int = 16,
                 max_prompt_len: int = 3072) -> list[str]:
        """Batched generation with the PrefixLM prefill (token_type_ids=1 over prompt)."""
        texts = [self._wrap(condition, p) for p in prompts]
        out: list[str] = []
        self.tok.padding_side = "left"
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_prompt_len, add_special_tokens=False).to(self.device)
            # PrefixLM: entire prompt is one bidirectional block. Decode steps are causal
            # (the port only consults token_type_ids on the prefill / first iteration).
            token_type_ids = torch.ones_like(enc["input_ids"])
            gen = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=token_type_ids,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 1e-5,
                temperature=temperature if temperature > 1e-5 else None,
                eos_token_id=self.eos_id,
                pad_token_id=self.tok.pad_token_id,
            )
            new = gen[:, enc["input_ids"].shape[1]:]
            out.extend(self.tok.batch_decode(new, skip_special_tokens=True))
        return out


# ============================ scoring helpers ============================
def last_boxed_only_string(s: str):
    """Return the contents of the last \\boxed{...}/\\fbox{...} in s (ported from MATH)."""
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i, depth, lo, hi = idx, 0, None, None
    while i < len(s):
        if s[i] == "{":
            depth += 1
            if lo is None:
                lo = i
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                hi = i
                break
        i += 1
    return s[lo + 1:hi].strip() if lo is not None and hi is not None else None


try:
    from math_verify import parse as _mv_parse, verify as _mv_verify

    def math_equal(gold: str, pred: str) -> bool:
        try:
            return bool(_mv_verify(_mv_parse(gold), _mv_parse(pred)))
        except Exception:
            return gold.strip() == pred.strip()
    HAVE_MATH_VERIFY = True
except Exception:  # noqa: BLE001
    def math_equal(gold: str, pred: str) -> bool:
        return gold.strip() == pred.strip()
    HAVE_MATH_VERIFY = False


def _drop_normalize(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return s.split()


def drop_f1_em(pred: str, golds: list[str]) -> tuple[float, float]:
    """Standard DROP/SQuAD token-F1 + EM, max over gold answers."""
    best_f1 = best_em = 0.0
    p = _drop_normalize(pred)
    for g in golds:
        gt = _drop_normalize(g)
        best_em = max(best_em, float(p == gt))
        common = Counter(p) & Counter(gt)
        ncommon = sum(common.values())
        if ncommon == 0 or not p or not gt:
            f1 = float(p == gt)
        else:
            prec, rec = ncommon / len(p), ncommon / len(gt)
            f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1, best_em


def micro_macro(stats: dict) -> dict:
    res, tot_acc, tot_inv = {}, 0.0, 0.0
    for k, v in stats.items():
        acc = v["correct"] / max(1, v["n"])
        inv = v["invalid"] / max(1, v["n"])
        res[f"acc_{k}"], res[f"invalid_{k}"] = acc, inv
        tot_acc += acc
        tot_inv += inv
    res["acc"] = tot_acc / max(1, len(stats))
    res["invalid"] = tot_inv / max(1, len(stats))
    res["n"] = sum(v["n"] for v in stats.values())
    return res


# ============================ benchmarks ============================
# Each benchmark exposes: .prompts, .condition, .max_new_tokens, .compute_metrics(generations)
class GSM8k:
    condition, max_new_tokens = "synth,cot", 512

    def __init__(self, limit=None):
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        self.prompts = ds["question"]
        self.truth = [int(a.split("####")[-1].strip().replace(",", "")) for a in ds["answer"]]

    def _ans(self, t: str):
        b = last_boxed_only_string(t)
        t = b if b else t
        nums = re.findall(r"-?\d[\d,]*\.?\d*", t.replace("$", ""))
        if not nums:
            return None
        try:
            return int(float(nums[-1].replace(",", "")))
        except (ValueError, OverflowError):
            return None

    def compute_metrics(self, gens):
        correct = invalid = 0
        for i, t in enumerate(gens):
            a = self._ans(t)
            if a is None:
                invalid += 1
            elif a == self.truth[i]:
                correct += 1
        n = len(gens)
        return {"n": n, "acc": correct / max(1, n), "invalid": invalid / max(1, n)}


class MATH:
    condition, max_new_tokens = "synth,cot", 512

    def __init__(self, limit=None):
        self.prompts, self.truth = [], []
        for subset in get_dataset_config_names("EleutherAI/hendrycks_math"):
            ds = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
            for item in ds:
                lab = last_boxed_only_string(item["solution"])
                if lab is None:
                    continue
                self.prompts.append(item["problem"])
                self.truth.append(lab)
        if limit:
            self.prompts, self.truth = self.prompts[:limit], self.truth[:limit]

    def compute_metrics(self, gens):
        correct = invalid = 0
        for i, t in enumerate(gens):
            a = last_boxed_only_string(t)
            if a is None:
                invalid += 1
                a = t
            if math_equal(self.truth[i], a):
                correct += 1
        n = len(gens)
        return {"n": n, "acc": correct / max(1, n), "invalid": invalid / max(1, n)}


class DROP:
    condition, max_new_tokens = "direct", 48
    EXAMPLES = [
        "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007\nQ: How many points did the buccaneers need to tie in the first?\nA: 3",
        "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.\nQ: How many games had the Bills won before this game?\nA: 0",
        "The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.\nQ: How many years before Navarrase used the captivity of John II?\nA: 9",
    ]

    @staticmethod
    def _answer_strings(a: dict) -> list[str]:
        """All acceptable answer strings from one DROP answer dict (number/spans/date)."""
        if a.get("number"):
            return [str(a["number"])]
        spans = [s for s in (a.get("spans") or []) if s]
        if spans:
            # accept each span and the full multi-span answer (joined)
            return spans + ([" ".join(spans)] if len(spans) > 1 else [])
        d = a.get("date") or {}
        s = " ".join(p for p in (d.get("month", ""), d.get("day", ""), d.get("year", "")) if p).strip()
        return [s] if s else []

    def __init__(self, limit=None, num_shots=3):
        prefix = "".join(f"{s}\n\n" for s in self.EXAMPLES[:num_shots])
        ds = load_dataset("EleutherAI/drop", split="validation")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        self.golds, self.prompts = [], []
        for doc in ds:
            golds = list(self._answer_strings(doc.get("answer") or {}))
            va = doc.get("validated_answers") or {}
            for i in range(len(va.get("spans", []) or [])):
                golds += self._answer_strings({
                    "number": (va.get("number") or [""] * (i + 1))[i],
                    "spans": (va.get("spans") or [[]] * (i + 1))[i],
                    "date": (va.get("date") or [{}] * (i + 1))[i],
                })
            seen = [g for g in dict.fromkeys(golds) if g]
            self.golds.append(seen or [""])
            self.prompts.append(f"{prefix}{doc['passage']}\nQ: {doc['question']}\nA:")

    def compute_metrics(self, gens):
        em = f1 = 0.0
        for i, t in enumerate(gens):
            pred = t.strip().split("\n")[0]
            a, b = drop_f1_em(pred, self.golds[i])
            f1 += a
            em += b
        n = len(gens)
        return {"n": n, "em": em / max(1, n), "f1": f1 / max(1, n)}


def _fmt_mcq(query, choices, gold_idx=None):
    text = query.strip() + "\n"
    letters = []
    for j, c in enumerate(choices):
        L = chr(65 + j)
        text += f"{L}. {str(c).strip()}\n"
        letters.append(L)
    text += f"Answer: {chr(65 + gold_idx)}" if gold_idx is not None else "Answer:"
    return text, letters


class _MCQ:
    """Few-shot multiple-choice; generate 1 token, read the letter (matches HRM-Text)."""
    condition, max_new_tokens = "direct", 1

    def __init__(self, rows, shots, by_subject=False):
        self.prompts, self.truth = [], []
        self.by_subject = by_subject
        shot_by = defaultdict(list)
        if isinstance(shots, dict):
            for k, v in shots.items():
                shot_by[k] = v
        else:
            shot_by[None] = shots
        for subj, query, choices, gi in rows:
            shot_rows = shot_by.get(subj, shot_by.get(None, []))
            prefix = ""
            for sq, sc, sgi in shot_rows:
                st, _ = _fmt_mcq(sq, sc, sgi)
                prefix += st + "\n\n"
            tgt, letters = _fmt_mcq(query, choices)
            self.prompts.append(prefix + tgt)
            self.truth.append({"gold": chr(65 + gi), "valid": set(letters), "subject": subj})

    def compute_metrics(self, gens):
        stats = defaultdict(lambda: {"n": 0, "correct": 0.0, "invalid": 0})
        for pred, gt in zip(gens, self.truth):
            key = gt["subject"] if self.by_subject else "all"
            stats[key]["n"] += 1
            p = pred.strip().upper()[:1]
            if p not in gt["valid"]:
                stats[key]["invalid"] += 1
                stats[key]["correct"] += 1 / len(gt["valid"])
            elif p == gt["gold"]:
                stats[key]["correct"] += 1
        if self.by_subject:
            return micro_macro(stats)
        s = stats["all"]
        return {"n": s["n"], "acc": s["correct"] / max(1, s["n"]),
                "invalid": s["invalid"] / max(1, s["n"])}


def build_mmlu(limit=None, num_shots=5):
    shot_ds = load_dataset("cais/mmlu", "all", split="dev")
    shots = defaultdict(list)
    for r in shot_ds:
        shots[r["subject"]].append((r["question"], r["choices"], r["answer"]))
    shots = {k: v[:num_shots] for k, v in shots.items()}
    ds = load_dataset("cais/mmlu", "all", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [(r["subject"], r["question"], r["choices"], r["answer"]) for r in ds]
    return _MCQ(rows, shots, by_subject=True)


def build_arc(limit=None, num_shots=25):
    shot_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"validation[:{num_shots}]")
    shots = [(r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))
             for r in shot_ds if r["answerKey"] in r["choices"]["label"]]
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [(None, r["question"], r["choices"]["text"], r["choices"]["label"].index(r["answerKey"]))
            for r in ds if r["answerKey"] in r["choices"]["label"]]
    return _MCQ(rows, shots)


def build_hellaswag(limit=None, num_shots=10):
    def q(r):
        return f"{r['ctx_a']} {r['ctx_b'].capitalize()}\nQuestion: Which is the most logical continuation?"
    shot_ds = load_dataset("Rowan/hellaswag", split=f"train[:{num_shots}]")
    shots = [(q(r), r["endings"], int(r["label"])) for r in shot_ds]
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [(None, q(r), r["endings"], int(r["label"])) for r in ds]
    return _MCQ(rows, shots)


def build_winogrande(limit=None, num_shots=5):
    shot_ds = load_dataset("allenai/winogrande", "winogrande_debiased", split=f"train[:{num_shots}]")
    shots = [(r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1) for r in shot_ds]
    ds = load_dataset("allenai/winogrande", "winogrande_debiased", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [(None, r["sentence"], [r["option1"], r["option2"]], int(r["answer"]) - 1) for r in ds]
    return _MCQ(rows, shots)


def build_boolq(limit=None, num_shots=5):
    def q(r):
        return f"{r['passage']}\nQuestion: {r['question'].capitalize()}?"
    shot_ds = load_dataset("google/boolq", split=f"train[:{num_shots}]")
    shots = [(q(r), ["Yes", "No"], 1 - int(bool(r["answer"]))) for r in shot_ds]
    ds = load_dataset("google/boolq", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    rows = [(None, q(r), ["Yes", "No"], 1 - int(bool(r["answer"]))) for r in ds]
    return _MCQ(rows, shots)


BENCHMARKS = {
    "GSM8k": lambda limit, shots: GSM8k(limit),
    "MATH": lambda limit, shots: MATH(limit),
    "DROP": lambda limit, shots: DROP(limit),
    "MMLU": lambda limit, shots: build_mmlu(limit),
    "ARC": lambda limit, shots: build_arc(limit),
    "HellaSwag": lambda limit, shots: build_hellaswag(limit),
    "Winogrande": lambda limit, shots: build_winogrande(limit),
    "BoolQ": lambda limit, shots: build_boolq(limit),
}
DEFAULT_ORDER = ["GSM8k", "MATH", "MMLU", "ARC", "HellaSwag", "Winogrande", "BoolQ", "DROP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID, help="HF model id or local dir")
    ap.add_argument("--benchmarks", default=",".join(DEFAULT_ORDER),
                    help="comma list; subset of " + ",".join(DEFAULT_ORDER))
    ap.add_argument("--limit", type=int, default=None, help="cap examples per benchmark (quick runs)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    for n in names:
        if n not in BENCHMARKS:
            raise SystemExit(f"unknown benchmark {n!r}; choose from {list(BENCHMARKS)}")

    print(f"[eval] model={args.model} device={args.device} dtype={args.dtype} "
          f"limit={args.limit} math_verify={'on' if HAVE_MATH_VERIFY else 'OFF (fallback)'}")
    engine = HFEngine(args.model, args.device, getattr(torch, args.dtype))

    results = {}
    for name in names:
        print(f"\n===== {name} =====")
        bench = BENCHMARKS[name](args.limit, None)
        print(f"  prompts: {len(bench.prompts)}  condition={bench.condition}  "
              f"max_new_tokens={bench.max_new_tokens}")
        gens = engine.generate(bench.prompts, bench.condition, bench.max_new_tokens,
                               temperature=args.temperature, batch_size=args.batch_size)
        metrics = bench.compute_metrics(gens)
        results[name] = metrics
        print(f"  {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items() if k in ('n','acc','invalid','em','f1')})}")

    print("\n================ SUMMARY ================")
    for name in names:
        m = results[name]
        score = m.get("acc", m.get("f1"))
        extra = f"  (EM {m['em']:.3f})" if "em" in m else ""
        inv = f"  invalid {m['invalid']:.3f}" if "invalid" in m else ""
        print(f"  {name:12s} n={m['n']:<6d} {score:.4f}{extra}{inv}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "limit": args.limit, "results": results}, f, indent=2)
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
