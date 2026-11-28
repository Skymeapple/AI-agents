#!/usr/bin/env bash
# Long-only portfolios under trading costs on Kenneth French's 49 industry portfolios (daily).
set -euo pipefail
pip install -q numpy
mkdir -p "$DATA/ff" && cd "$DATA/ff"
curl -sSL -A "research-script" -o ind.zip "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
unzip -o -q ind.zip && ls -la
F=$(ls | grep -i -E '\.csv$' | head -1)
cd "$GITHUB_WORKSPACE"
for c in 5 10 20; do
  echo "=== cost $c basis points ==="
  python -W ignore research/hfm-projects/quant-finance/portfolio-with-frictions/portfolio_costs.py "$DATA/ff/$F" $c
done
