#!/bin/bash
# E3 configurations actually run for the paper (CPU was shared with other jobs, so the grid is reduced).
# value objective: B in {4, 6, inf}, seed 0 (B=6, inf also seeds 1, 2); pointer objective: B in {4, 6, inf}
# with p_single = 0.125 (seeds 0-2) and B = 6 with p_single = 0.5 (seeds 0-2). Each run writes its own JSON.
cd "$(dirname "$0")"; mkdir -p ../results/e3
jobs=()
for s in 0 1 2; do for B in 4 6 inf; do jobs+=("--pointer --B $B --p_single 0.125 --seed $s --out e3/ptr_B${B}_ps0.125_s${s}.json"); done; done
for s in 0 1 2; do jobs+=("--pointer --B 6 --p_single 0.5 --seed $s --out e3/ptr_B6_ps0.5_s${s}.json"); done
for s in 1 2; do for B in 6 inf; do jobs+=("--B $B --seed $s --out e3/val_B${B}_s${s}.json"); done; done
if [ "$1" != "rest" ]; then for B in 4 6 inf; do jobs=("--B $B --seed 0 --out e3/val_B${B}_s0.json" "${jobs[@]}"); done; fi
printf '%s\n' "${jobs[@]}" | xargs -P "${P:-3}" -I{} sh -c 'python3 e3_single_head.py --steps 4000 {} > /dev/null 2>&1; echo done {}'
