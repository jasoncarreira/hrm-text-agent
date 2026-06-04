#!/usr/bin/env bash
# Turnkey full-parameter SFT for HRM-Text-1B tool-use. Run on a CUDA GPU pod.
#   git clone <this repo> && cd <repo> && bash run.sh
# Optional: export HF_TOKEN=... HF_REPO=<user>/hrm-tooluse  to push the model to HF.
set -euo pipefail

echo "===== [1/5] install deps ====="
# transformers>=5.9 (native hrm_text) needs torch>=2.7 (float8_e8m0fnu), but RunPod
# images often ship torch 2.4. Install a torch>=2.7 cu118 wheel: the CUDA 11.8 runtime
# runs on any modern driver (incl. hosts whose driver is <12.8), and A100 is fully
# supported by cu118. Skipped automatically if torch>=2.7 is already present.
python -c "import torch,sys; sys.exit(0 if tuple(map(int,torch.__version__.split('.')[:2]))>=(2,7) else 1)" \
  || pip install -q --upgrade "torch>=2.7" --index-url https://download.pytorch.org/whl/cu118
pip install -q -r requirements.txt
# transformers>=5.9 (native HrmText) imports torch.float8_e8m0fnu, which needs torch>=2.7.
# The RunPod base image ships torch 2.4.1, so upgrade torch+torchvision to a matching pair.
# cu126 wheels are forward-compatible with CUDA 12.x drivers (tested on driver 555 / CUDA 12.5).
python - <<'PY'
import torch, sys
sys.exit(0 if hasattr(torch, "float8_e8m0fnu") else 1)
PY
if [ $? -ne 0 ]; then
  echo "  torch $(python -c 'import torch;print(torch.__version__)') too old; upgrading to 2.7.1 (cu126)"
  pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1"
fi

echo "===== [2/5] build tool data (Hermes + glaive) ====="
python convert_hermes.py --holdout 0

echo "===== [3/5] build mixed dataset (+ no_robots + irrelevance) ====="
python make_mixed_data.py

echo "===== [4/5] full-parameter SFT (3 epochs, bf16, lr 3e-5) ====="
python train_full.py --data data/sft_mixed.jsonl --epochs 3 --max-len 2048 \
  --batch-size 4 --grad-accum 8 --save-every 0 --no-grad-checkpoint \
  --out-dir models/hrm-tooluse-full

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
