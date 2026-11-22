#!/usr/bin/env bash
# Fetch the public human + LLM Alternate Uses / Verbal Fluency data of Nath, Dayan & Stevenson (ICCC 2024).
# Only the two CSV files are used (no pickles are loaded).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
git clone https://github.com/surabhisnath/Creative_Process.git "$TMP/cp"
git -C "$TMP/cp" checkout bd885366fd76166d349c6aec301557fbdebecdbd
mkdir -p "$HERE/data"
cp "$TMP/cp/csvs/data_humans.csv" "$TMP/cp/csvs/data_LLMs.csv" "$HERE/data/"
sha256sum "$HERE/data/"*.csv
