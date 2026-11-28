#!/usr/bin/env bash
# Re-download the public METR time-horizon run data (not committed; ~43 MB clone).
# Pinned to the commit analysed in the paper.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data_cache
if [ ! -d data_cache/metr ]; then
  git clone https://github.com/METR/eval-analysis-public.git data_cache/metr
fi
git -C data_cache/metr checkout -q 52cb829c7a2efb2d659285c4b1768d191d97f8d2 || true
ls -la data_cache/metr/reports/time-horizon-1-1/data/raw/runs.jsonl
