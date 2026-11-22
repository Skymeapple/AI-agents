#!/usr/bin/env bash
# Reproduce every number and figure in the paper (CPU only, ~3 minutes total).
# Step 0 (data): clone the public MMLU-Pro repository and unzip each model's outputs into its own folder:
#   git clone https://github.com/TIGER-AI-Lab/MMLU-Pro.git   (we used commit f418b116db00b065c2aea046518d8fcf74d39872)
#   mkdir mp_unz && for z in MMLU-Pro/eval_results/*.zip; do n=$(basename $z .zip); n=${n#model_outputs_}; mkdir -p mp_unz/$n; unzip -q $z -d mp_unz/$n; done
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then python3 build_matrix.py "$1" ../results/answers.npz > ../results/build_log.txt; fi
python3 exp1_correlation.py   > ../results/exp1_log.txt
python3 exp2_voting.py        > ../results/exp2_log.txt 2>&1
for P in all46 excl_top1 peer_0.60-0.80; do python3 exp3_division.py $P > ../results/exp3_log_$P.txt 2>&1; done
python3 exp4_debate.py        > ../results/exp4_log.txt
python3 exp5_predict.py       > ../results/exp5_log.txt
echo done
