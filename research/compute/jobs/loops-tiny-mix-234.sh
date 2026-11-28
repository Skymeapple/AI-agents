# Toy-scale looped study: training arm mix-234, two seeds; scored at 1 to 8 loops, with simulated early exit.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
cd "$DATA"
curl -sS -L -f -m 1800 -r 0-150000000 -o train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl -sS -L -f -m 600 -o valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
cd "$GITHUB_WORKSPACE"
timeout 320m python -u research/hfm-projects/pretraining/looped-pretraining-v1/loops_tiny.py "$DATA/train.txt" "$DATA/valid.txt" mix-234 6000 0 1 || echo "stopped early; partial results kept"
