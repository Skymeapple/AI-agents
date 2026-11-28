#!/usr/bin/env bash
# Reproduce every number and figure in the paper (~12 min on 1 CPU core).
set -e
cd "$(dirname "$0")"
python3 exp1_validate.py      # E1  closed forms vs simulator   (~2.5 min)
python3 exp2_floor.py         # E2/E3 invariance, floor, power law (~1.5 min)
python3 exp2b_replicate.py    # replication of one E3 Monte-Carlo check
python3 exp4_age.py           # E4  context-age hazard           (~8.5 min)
python3 exp5_mutation.py      # E5  mutation-analysis detectability (~15 s)
python3 exp6_horizon.py       # E6  horizon multipliers (analytic)
