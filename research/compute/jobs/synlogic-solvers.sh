#!/usr/bin/env bash
# SynLogic: classical solvers for 13 task types, checked by the benchmark's own verifiers.
set -euo pipefail
pip install -q pandas numpy nltk sympy math-verify tqdm pydantic markdown inflect english-words
git clone -q --depth 1 https://github.com/MiniMax-AI/SynLogic "$DATA/synlogic-repo"
R=research/compute/results/synlogic-samples/easy-validation.jsonl
P=research/hfm-projects/post-training/synlogic-training/synlogic_solvers.py
python -u $P solve $R "$OUT/answers.jsonl"
python -u $P verify $R "$OUT/answers.jsonl" "$DATA/synlogic-repo"
