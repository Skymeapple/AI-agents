#!/bin/bash
# E3 configurations actually run for the paper. The CPU was shared with other jobs, so the grid is reduced:
#   value objective   : B in {4, 6, inf} seed 0; B = 6 also seed 1
#   pointer objective : B in {4, 6, inf}, p_single = 0.125, seeds 0, 1, 2; B = 6, p_single = 0.5, seeds 0, 1
# Each run writes its own JSON under results/e3/. 3 workers, 1 thread each.
cd "$(dirname "$0")"; mkdir -p ../results/e3
jobs=()
for B in 4 6 inf; do jobs+=("--B $B --seed 0 --out e3/val_B${B}_s0.json"); done
for s in 0 1 2; do for B in 4 6 inf; do jobs+=("--pointer --B $B --p_single 0.125 --seed $s --out e3/ptr_B${B}_ps0.125_s${s}.json"); done; done
for s in 0 1; do jobs+=("--pointer --B 6 --p_single 0.5 --seed $s --out e3/ptr_B6_ps0.5_s${s}.json"); done
jobs+=("--B 6 --seed 1 --out e3/val_B6_s1.json")
# skip runs whose output already exists (lets the script resume)
for j in "${jobs[@]}"; do f=../results/${j##*--out }; [ -f "$f" ] || echo "$j"; done | \
  xargs -P "${P:-3}" -I{} sh -c 'python3 e3_single_head.py --steps 4000 {} > /dev/null 2>&1; echo done {}'
