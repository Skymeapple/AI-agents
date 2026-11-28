# Looped model at toy scale: FineWeb-Edu share 0.25 against DCLM share 0.75 in training, two seeds, scored on held-out DCLM.
set -e
pip -q install numpy torch --extra-index-url https://download.pytorch.org/whl/cpu
pip -q install pyarrow fsspec aiohttp zstandard
python -u research/hfm-projects/pretraining/pretraining-architecture-pool/corpora.py "$DATA" 30 web dclm
timeout 320m python -u research/hfm-projects/pretraining/pretraining-architecture-pool/mix_tiny.py "$DATA" looped 6000 web=0.25,dclm=0.75 0 1 || echo "stopped early; partial results kept"
