#!/bin/bash
# Revision round 1: pooled-moment amortized models (stronger test subject + feature ablation)
cd "$(dirname "$0")"
cat <<'LIST' | xargs -P 2 -L 1 sh -c 'python3 strong_model.py "$0" "$1" "$2" >> ../results/strong_log.txt 2>>../results/rev_errors.log'
obs_p0 m2 0
laplace_p0 m4 0
laplace_p0 m2 0
int32_p0 m2 0
int128_p0 m2 0
obs_p1 m2 0
obs_p09 m2 0
obs_p0 m2 1
laplace_p0 m4 1
laplace_p0 m2 1
int32_p0 m2 1
int128_p0 m2 1
obs_p1 m2 1
obs_p09 m2 1
LIST
echo STRONG_DONE >> ../results/strong_log.txt
