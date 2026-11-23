#!/usr/bin/env bash
# Full reproduction: ~1 minute on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
pip install -q tiktoken==0.14.0 tokenizers sentencepiece numpy scipy matplotlib
./fetch_assets.sh
python3 tokenize_udhr.py
python3 analysis.py > ../results/analysis_stdout.txt

python3 robustness.py
echo "outputs in ../results"
