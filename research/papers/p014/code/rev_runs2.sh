#!/bin/bash
# Revision round 1 (second pass): remaining runs from rev_runs.sh. The first wave of
# rev_runs.sh (laplace s0, int32 s0, int128 s0, int128 s1) completed; obs_p05 s1 and
# obs_p09 s2 were dropped for CPU budget.
cd "$(dirname "$0")"
cat <<'LIST' | xargs -P 6 -I{} sh -c "python3 twins.py {} > /dev/null 2>>../results/rev_errors.log"
--name laplace_p0 --p_order 0.0 --family laplace --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 1
--name obs_p09 --p_order 0.9 --seed 1
--name obs_p1 --p_order 1.0 --seed 2
--name int32_p0 --p_order 0.0 --n_int 32 --seed 2
--name int128_p0 --p_order 0.0 --n_int 128 --seed 2
LIST
echo REV2_DONE
