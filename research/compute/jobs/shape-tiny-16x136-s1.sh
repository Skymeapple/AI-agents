# Toy-scale fixed-inference-cost study: 16 layers, width 136, seed 1.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
cd "$DATA"
curl -sS -L -f -m 1800 -r 0-150000000 -o train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl -sS -L -f -m 600 -o valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
cd "$GITHUB_WORKSPACE"
timeout 320m python -u research/hfm-projects/pretraining/pretraining-fixed-inference-cost/shape_tiny.py "$DATA/train.txt" "$DATA/valid.txt" 16 136 6000 1 || echo "stopped early; partial results kept"
