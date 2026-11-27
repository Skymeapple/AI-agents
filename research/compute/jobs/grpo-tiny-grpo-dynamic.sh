# Toy-scale RL recipe study: recipe grpo-dynamic, 1500 updates, three seeds.
set -e
pip -q install torch --extra-index-url https://download.pytorch.org/whl/cpu
timeout 330m python -u research/hfm-projects/post-training/agentic-rl-recipes/grpo_tiny.py grpo-dynamic 1500 0 1 2 || echo "stopped early; partial results kept"
