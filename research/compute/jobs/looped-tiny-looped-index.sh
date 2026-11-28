# Toy-scale looped-transformer study, model "looped+index": 6,000 steps, 2 seeds, TinyStories bytes.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
cd "$DATA"
curl -sS -L -f -m 1800 -r 0-150000000 -o train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl -sS -L -f -m 600 -o valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
ls -la train.txt valid.txt
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/pretraining/looped-pretraining-fixed-compute/looped_tiny.py "$DATA/train.txt" "$DATA/valid.txt" "looped+index" 6000 2
