#!/bin/bash
# runs all MLP jobs, 4 in parallel
cd "$(dirname "$0")"
for s in 0 1 2 3; do for sc in perm split; do echo "$sc $s"; done; done | xargs -P 4 -n 2 python3 mlp_experiments.py
