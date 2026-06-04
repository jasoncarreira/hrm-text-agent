#!/usr/bin/env bash
# Turnkey full-parameter SFT for HRM-Text-1B tool-use. Run on a CUDA GPU pod.
#   git clone <this repo> && cd <repo> && bash run.sh
# Optional: export HF_TOKEN=... HF_REPO=<user>/hrm-tooluse  to push the model to HF.
set -euo pipefail

echo "===== [1/5] install deps ====="
pip install -q -r requirements.txt

echo "===== [2/5] build tool data (Hermes + glaive) ====="
python convert_hermes.py --holdout 0

echo "===== [3/5] build mixed dataset (+ no_robots + irrelevance) ====="
python make_mixed_data.py

echo "===== [4/5] full-parameter SFT (3 epochs, bf16, lr 3e-5) ====="
python train_full.py --data data/sft_mixed.jsonl --epochs 3 --max-len 2048 \
  --batch-size 4 --grad-accum 8 --save-every 0 --out-dir models/hrm-tooluse-full

echo "===== [5/5] BFCL eval (official AST checker) ====="
python bfcl_local.py --model models/hrm-tooluse-full --limit 100 --dump bfcl_errs.jsonl

export HF_REPO="${HF_REPO:-jasoncarreira/hrm-text-agent}"
if [ -n "${HF_TOKEN:-}" ]; then
  echo "===== pushing model -> ${HF_REPO} ====="
  python - <<PY
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
r = os.environ["HF_REPO"]
AutoModelForCausalLM.from_pretrained("models/hrm-tooluse-full").push_to_hub(r, private=True)
AutoTokenizer.from_pretrained("models/hrm-tooluse-full").push_to_hub(r, private=True)
print("pushed", r)
PY
else
  echo "(set HF_TOKEN to auto-push the model to ${HF_REPO})"
fi
echo "===== ALL DONE ====="
