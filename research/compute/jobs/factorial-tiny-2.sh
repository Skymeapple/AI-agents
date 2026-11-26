# Recipe factorial at toy scale: arms wsd and long, three seeds, 6000 steps.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
cd "$DATA"
curl -sS -L -f -m 1800 -r 0-150000000 -o train.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
curl -sS -L -f -m 600 -o valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
cd "$GITHUB_WORKSPACE"
for arm in wsd long; do timeout 150m python -u research/hfm-projects/pretraining/agentic-pretraining/factorial_tiny.py "$DATA/train.txt" "$DATA/valid.txt" $arm 6000 0 1 2 || echo "stopped early; partial results kept"; done
