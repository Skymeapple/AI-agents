#!/bin/bash
# Reproduces every result in the paper (approx. 20-25 min on 4 CPU cores).
cd "$(dirname "$0")"
cat <<'LIST' | xargs -P 4 -I{} sh -c "python3 twins.py {} > /dev/null"
--name obs_p0 --p_order 0.0
--name obs_p05 --p_order 0.5
--name obs_p09 --p_order 0.9
--name obs_p1 --p_order 1.0
--name laplace_p0 --p_order 0.0 --family laplace
--name graph_p0 --p_order 0.0 --graph
--name int8_p0 --p_order 0.0 --n_int 8
--name int32_p0 --p_order 0.0 --n_int 32
--name int128_p0 --p_order 0.0 --n_int 128
--name obs_p0 --p_order 0.0 --seed 1
--name obs_p09 --p_order 0.9 --seed 1
--name laplace_p0 --p_order 0.0 --family laplace --seed 1
--name int32_p0 --p_order 0.0 --n_int 32 --seed 1
--name obs_p1 --p_order 1.0 --seed 1
--name graph_p0 --p_order 0.0 --graph --seed 1
--name int128_p0 --p_order 0.0 --n_int 128 --seed 1
LIST
python3 summarize.py
