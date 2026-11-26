#!/usr/bin/env bash
# EmbSpatial-Bench: monocular depth (Depth Anything V2 Small, CPU) for the nearest/farthest questions.
set -euo pipefail
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q transformers pillow pandas pyarrow requests numpy
cd "$DATA"
python -c "
import requests
p = requests.get('https://huggingface.co/api/datasets/FlagEval/EmbSpatial-Bench/parquet', timeout=60).json()
open('test.parquet', 'wb').write(requests.get(p['default']['test'][0], timeout=900).content)"
cd "$GITHUB_WORKSPACE"
timeout 300m python -u research/hfm-projects/post-training/embspatial-training/embspatial_depth.py "$DATA/test.parquet" "$OUT/depth-features.jsonl" || echo "stopped early; partial results kept"
