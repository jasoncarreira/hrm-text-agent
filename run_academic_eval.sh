#!/usr/bin/env bash
# Option B: base vs fine-tune on the academic suite. Full sets except MATH (limit 1000).
# Fast (MCQ/short) benchmarks first for early signal, then GSM8k, then MATH.
set -uo pipefail
cd /workspace/hrm-text-agent
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation OOMs
# Per-benchmark batch sizes live in eval_academic.py (MCQ few-shot prompts are long and
# need a small batch; math is short and runs at 48), so we do NOT pass --batch-size here.
FAST="MMLU,ARC,HellaSwag,Winogrande,BoolQ,DROP,GSM8k"
run() {  # tag  model
  echo "######## MODEL=$1 ($2) — full suite (ex-MATH) ########"
  python eval_academic.py --model "$2" --benchmarks "$FAST" --out "$1_acad.json"
  echo "######## MODEL=$1 — MATH (limit 1000) ########"
  python eval_academic.py --model "$2" --benchmarks MATH --limit 1000 --out "$1_math.json"
}
run base sapientinc/HRM-Text-1B
run ours models/hrm-tooluse-full
echo "######## ACADEMIC EVAL ALL DONE ########"
