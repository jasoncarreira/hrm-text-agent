#!/usr/bin/env bash
# Train a CODE expert, then compose-merge it with the tool expert (v2) and test whether
# HRM experts COMPOSE (gain both skills) or interfere/collapse.
#
# Non-destructive: code expert -> its own dir + its own HF repo (guarded against base/v1/
# tool); the merge -> its own dir (not pushed). Set HF_TOKEN to push the code expert.
# Run on a DISPOSABLE pod — eval_code.py executes model-generated code.
set -euo pipefail
cd "$(dirname "$0")"

BASE="sapientinc/HRM-Text-1B"
TOOL_REPO="${TOOL_REPO:-jasoncarreira/hrm-text-agent-v2}"   # the tool expert to merge with
CODE_DIR="${CODE_DIR:-models/hrm-code}"
MERGE_DIR="${MERGE_DIR:-models/hrm-merge-toolcode}"
CODE_REPO="${CODE_REPO:-jasoncarreira/hrm-text-code}"       # separate repo for the code expert
for prot in "$BASE" "jasoncarreira/hrm-text-agent" "$TOOL_REPO"; do
  [ "$CODE_REPO" = "$prot" ] && { echo "REFUSING: CODE_REPO collides with $prot" >&2; exit 1; }
done

echo "===== [1/7] deps (torch>=2.7 for transformers 5.x) ====="
python -c "import torch,sys; sys.exit(0 if hasattr(torch,'float8_e8m0fnu') else 1)" \
  || pip install -q --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.1" "torchvision==0.22.1"
pip install -q -r requirements.txt

echo "===== [2/7] build code data (instruction -> code, synth,cot lane) ====="
python make_code_data.py --out data/code_sft.jsonl --n 25000

echo "===== [3/7] train code expert -> ${CODE_DIR} ====="
python train_full.py --data data/code_sft.jsonl --out-dir "$CODE_DIR" \
  --epochs 3 --max-len 2048 --no-grad-checkpoint

echo "===== [4/7] code bench: base vs code expert (did training add code skill?) ====="
python eval_code.py --bench humaneval --model "$BASE"     --out he_base.json
python eval_code.py --bench mbpp      --model "$BASE"     --out mbpp_base.json
python eval_code.py --bench humaneval --model "$CODE_DIR" --out he_code.json
python eval_code.py --bench mbpp      --model "$CODE_DIR" --out mbpp_code.json

echo "===== [5/7] push code expert -> ${CODE_REPO} (separate repo) ====="
if [ -n "${HF_TOKEN:-}" ]; then
  CODE_DIR="$CODE_DIR" CODE_REPO="$CODE_REPO" python - <<'PY'
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
d, r = os.environ["CODE_DIR"], os.environ["CODE_REPO"]
AutoModelForCausalLM.from_pretrained(d).push_to_hub(r, private=True)
AutoTokenizer.from_pretrained(d).push_to_hub(r, private=True)
print("pushed", r)
PY
else
  echo "(set HF_TOKEN to push the code expert)"
fi

echo "===== [6/7] compose-merge tool(v2) + code -> ${MERGE_DIR} ====="
python merge_compose.py --base "$BASE" --experts "$TOOL_REPO,$CODE_DIR" --coeffs 1.0,1.0 \
  --merge-out "$MERGE_DIR" --categories simple,multiple,parallel,irrelevance --limit 200 \
  --out merge_compose.json

echo "===== [7/7] code bench on the merged model (did code survive the merge?) ====="
python eval_code.py --bench humaneval --model "$MERGE_DIR" --out he_merge.json
python eval_code.py --bench mbpp      --model "$MERGE_DIR" --out mbpp_merge.json

echo "===== DONE ====="
echo "Read the result:"
echo "  he_base/mbpp_base   vs  he_code/mbpp_code   -> did the code expert learn to code?"
echo "  merge_compose.json  (BFCL = tools kept? collapsed flag = stable?)"
echo "  he_merge/mbpp_merge -> did code SURVIVE the merge alongside tools? (composition)"
