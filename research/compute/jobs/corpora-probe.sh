# Check that the four public pretraining corpora can be fetched and split (5 MB each).
set -e
pip -q install pyarrow fsspec aiohttp zstandard
for c in web dclm code math; do
  python -u research/hfm-projects/pretraining/pretraining-architecture-pool/corpora.py "$DATA" 5 $c || echo "FAILED $c"
done
ls -la "$DATA"
