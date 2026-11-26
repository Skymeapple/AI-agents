# Data-mix study at toy scale: 4-layer model, mixes web=0.6,code=0.2,math=0.2 and web=0.8,code=0.1,math=0.1, three seeds, scored on four domains.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
pip -q install pyarrow fsspec aiohttp zstandard
python -u research/hfm-projects/pretraining/pretraining-architecture-pool/corpora.py "$DATA" 30 web dclm code math
for mix in web=0.6,code=0.2,math=0.2 web=0.8,code=0.1,math=0.1; do timeout 160m python -u research/hfm-projects/pretraining/pretraining-architecture-pool/mix_tiny.py "$DATA" dense-4 6000 $mix 0 1 2 || echo "stopped early; partial results kept"; done
