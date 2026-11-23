#!/usr/bin/env bash
# Reproduce every number in paper.md.  Total CPU: ~6 minutes on 4 cores.
set -euo pipefail
cd "$(dirname "$0")"
# 1. geng (nauty 2.8.8) -- built from the nauty sources bundled in the pynauty 2.8.8.1 sdist:
if [ ! -x geng_bin ]; then
  pip download --no-binary :all: --no-deps pynauty==2.8.8.1 -d /tmp/pynauty_src
  tar xzf /tmp/pynauty_src/pynauty-2.8.8.1.tar.gz -C /tmp/pynauty_src
  (cd /tmp/pynauty_src/pynauty-2.8.8.1/src/nauty2_8_8 && ./configure >/dev/null && make geng >/dev/null)
  cp /tmp/pynauty_src/pynauty-2.8.8.1/src/nauty2_8_8/geng geng_bin
fi
gcc -O2 -o invariants invariants.c
python3 validate_invariants.py                      # brute-force cross-check of the C engine
mkdir -p ../data
for n in 2 3 4 5 6 7 8 9; do ./geng_bin -cq $n | ./invariants > ../data/inv_n$n.bin; done
for r in 0 1 2 3; do (./geng_bin -cq 10 $r/4 | ./invariants > ../data/inv_n10_part$r.bin) & done; wait
cat ../data/inv_n10_part*.bin > ../data/inv_n10.bin; rm ../data/inv_n10_part*.bin
for r in 0 1 2 3; do (./geng_bin -cq 11 $r/256 | ./invariants > ../data/inv_n11_slice$r.bin) & done; wait
cat ../data/inv_n11_slice[0-9]*.bin > ../data/inv_n11_slice.bin; rm ../data/inv_n11_slice[0-9]*.bin
python3 gen_large_sample.py
python3 analysis.py  > ../results/analysis_stdout.txt
python3 analysis2.py > ../results/analysis2_stdout.txt
python3 analysis3.py > ../results/analysis3_stdout.txt
