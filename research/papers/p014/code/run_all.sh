#!/bin/bash
# Reproduces every result in the paper (approx. 20 min per wave of 4 runs on 4 CPU cores).
# Wave 0 (observational regimes, seed 0) was run first; waves 1-2 follow.
set -e
cd "$(dirname "$0")"
python3 verify_theory.py > /dev/null
cat <<'LIST' | xargs -P 4 -I{} sh -c "python3 twins.py {} > /dev/null"
--name obs_p0 --p_order 0.0
--name obs_p05 --p_order 0.5
--name obs_p09 --p_order 0.9
--name obs_p1 --p_order 1.0
--name laplace_p0 --p_order 0.0 --family laplace
--name graph_p0 --p_order 0.0 --graph
--name int32_p0 --p_order 0.0 --n_int 32
--name int128_p0 --p_order 0.0 --n_int 128
--name obs_p0 --p_order 0.0 --seed 1
--name obs_p1 --p_order 1.0 --seed 1
--name laplace_p0 --p_order 0.0 --family laplace --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 1
LIST
python3 summarize.py
