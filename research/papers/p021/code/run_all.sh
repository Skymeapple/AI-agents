#!/usr/bin/env bash
# Reproduce all results for p021. Total CPU time ~15-20 min on one core.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d ../data/eval-analysis-public ]; then
  git clone https://github.com/METR/eval-analysis-public ../data/eval-analysis-public
  (cd ../data/eval-analysis-public && git checkout 52cb829c7a2efb2d659285c4b1768d191d97f8d2)
fi
NBOOT=300 python3 a1_fits.py      # per-agent 2PL vs lapse fits, horizons, bootstrap  (~7 min)
python3 a2_decomp.py              # failure anatomy on short tasks, ICC, pass^k        (<1 min)
python3 a3_trends.py              # trends over release date, paired tests             (<1 min)
NSIM=200 python3 a4_sim.py        # simulation: detectability, 2PL extrapolation, CI coverage (~8 min)
python3 a5_cert.py                # certification sample sizes                         (<1 min)
