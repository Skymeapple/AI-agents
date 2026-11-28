#!/bin/bash
# Revision round 1: retrain informative-evidence regimes (bit-identical to round 0 for
# all original fields; adds in-prior / direction-balanced / non-adjacent twin probes,
# kappa bootstrap CIs, saved weights) and add seeds for single-seed regimes.
cd "$(dirname "$0")"
cat <<'LIST' | xargs -P 4 -I{} sh -c "python3 twins.py {} > /dev/null 2>>../results/rev_errors.log"
--name laplace_p0 --p_order 0.0 --family laplace
--name int32_p0 --p_order 0.0 --n_int 32
--name int128_p0 --p_order 0.0 --n_int 128
--name int128_p0 --p_order 0.0 --n_int 128 --seed 1
--name laplace_p0 --p_order 0.0 --family laplace --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 1
--name obs_p09 --p_order 0.9 --seed 1
--name obs_p05 --p_order 0.5 --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 2
--name int128_p0 --p_order 0.0 --n_int 128 --seed 2
--name obs_p09 --p_order 0.9 --seed 2
--name obs_p1 --p_order 1.0 --seed 2
LIST
echo REV_DONE
