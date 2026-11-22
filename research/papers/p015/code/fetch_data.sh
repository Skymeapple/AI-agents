#!/usr/bin/env bash
# Fetch METR's public Time Horizon run data (TH1.1) at a pinned commit.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/data"
TMP="$(mktemp -d)"
git clone --quiet https://github.com/METR/eval-analysis-public.git "$TMP/metr"
git -C "$TMP/metr" checkout --quiet 52cb829c7a2efb2d659285c4b1768d191d97f8d2
cp "$TMP/metr/reports/time-horizon-1-1/data/raw/runs.jsonl" "$HERE/data/runs_th11.jsonl"
cp "$TMP/metr/reports/time-horizon-1-0/data/raw/runs.jsonl" "$HERE/data/runs_th10.jsonl"
cp "$TMP/metr/data/external/release_dates.yaml" "$HERE/data/release_dates.yaml"
rm -rf "$TMP"
echo "data fetched into $HERE/data"
