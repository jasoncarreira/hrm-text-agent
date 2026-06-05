#!/usr/bin/env bash
# Turnkey v2 SFT for HRM-Text-1B tool-use. Run on a CUDA GPU pod.
#   git pull && bash run_v2.sh
#
# v2 = the v1 mix + xLAM (parallel/multi-call, the weak BFCL categories) + a
#      format-discipline slice (single-letter MCQ + \boxed{} math, to recover the
#      v1 MCQ-format regression).
#
# NON-DESTRUCTIVE: trains into a SEPARATE dir and pushes to a SEPARATE HF repo
# (default hrm-text-agent-v2). It REFUSES to write the v1 repo, so v1 is never
# overwritten. Override the target with `export HF_REPO_V2=<user>/<repo>`.
#
# Needs HF_TOKEN with xLAM gating accepted:
#   huggingface.co/datasets/Salesforce/xlam-function-calling-60k
set -euo pipefail
cd "$(dirname "$0")"

V1_REPO="jasoncarreira/hrm-text-agent"                       # protected: never overwrite
HF_REPO_V2="${HF_REPO_V2:-jasoncarreira/hrm-text-agent-v2}"  # v2 goes to its own repo
OUT_DIR="${OUT_DIR:-models/hrm-tooluse-v2}"
if [ "$HF_REPO_V2" = "$V1_REPO" ]; then
  echo "REFUSING: HF_REPO_V2 ($HF_REPO_V2) is the v1 repo — that would overwrite v1." >&2
  echo "Pick a different repo (e.g. ${V1_REPO}-v2) and re-run." >&2
  exit 1
fi

echo "===== [1/7] deps (torch>=2.7 for transformers 5.x; cu126 runs on any 12.x driver) ====="
python -c "import torch,sys; sys.exit(0 if hasattr(torch,'float8_e8m0fnu') else 1)" \
  || pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1"
pip install -q -r requirements.txt

echo "===== [2/7] build tool data (Hermes + glaive) ====="
python convert_hermes.py --holdout 0

echo "===== [3/7] build new sources (xLAM multi-call + format-discipline slice) ====="
python convert_xlam.py --bias-multicall          # GATED: needs HF_TOKEN + xLAM terms accepted
python make_format_slice.py                       # train/aux splits only (leakage-safe)

echo "===== [4/7] assemble v2 mix (shuffled, evenly interleaved) ====="
python make_mixed_data.py --out data/sft_mixed_v2.jsonl \
  --n-tool 8000 --n-instr 6000 --n-irrel 2500 \
  --extra data/xlam.jsonl:14000 data/format_slice.jsonl:3000

echo "===== [5/7] full-parameter SFT (3 epochs, bf16, lr 3e-5) -> ${OUT_DIR} ====="
python train_full.py --data data/sft_mixed_v2.jsonl --epochs 3 --max-len 2048 \
  --batch-size 4 --grad-accum 8 --save-every 0 --no-grad-checkpoint \
  --out-dir "$OUT_DIR"

echo "===== [6/7] BFCL eval (full set, official AST checker) ====="
python bfcl_local.py --model "$OUT_DIR" --dump bfcl_v2_errs.jsonl

echo "===== [7/7] academic regression suite on v2 (compare to the base numbers in the README) ====="
python eval_academic.py --model "$OUT_DIR" --out academic_v2.json

if [ -n "${HF_TOKEN:-}" ]; then
  echo "===== push -> ${HF_REPO_V2}  (LAST step, after ALL evals; v1 repo ${V1_REPO} untouched) ====="
  HF_REPO_V2="$HF_REPO_V2" OUT_DIR="$OUT_DIR" V1_REPO="$V1_REPO" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
repo, out, v1 = os.environ["HF_REPO_V2"], os.environ["OUT_DIR"], os.environ["V1_REPO"]
assert repo != v1, f"guard: refusing to overwrite v1 repo {v1}"
AutoModelForCausalLM.from_pretrained(out).push_to_hub(repo, private=True)
AutoTokenizer.from_pretrained(out).push_to_hub(repo, private=True)
print("pushed", repo)
PY
else
  echo "(set HF_TOKEN to auto-push -> ${HF_REPO_V2})"
fi
echo "===== v2 DONE ====="
