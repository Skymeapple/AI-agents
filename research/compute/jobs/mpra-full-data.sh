# Regulatory DNA: the multi-task comparison on all training sequences, 3 seeds, 4 epochs.
set -e
pip -q install numpy scipy pandas torch --extra-index-url https://download.pytorch.org/whl/cpu
cd "$DATA"
curl -sS -L -m 1800 -o legnet.zip "https://zenodo.org/api/records/10558183/files/human_legnet-main.zip/content"
unzip -q -o legnet.zip 'human_legnet-main/datasets/original/*_averaged.tsv'
cd "$GITHUB_WORKSPACE"
MPRA_MAX_TRAIN=1000000 MPRA_EPOCHS=4 MPRA_SEEDS=3 python -u research/hfm-projects/biology/regulatory-dna-mpra/mpra_multitask.py "$DATA/human_legnet-main/datasets/original"
