# QASPER note floor (third run: adds paragraph-level learned builders)
set -e
pip install -q pandas pyarrow scikit-learn scipy numpy
cd "$DATA"
for s in train validation test; do
  curl -sSfL -o $s.parquet "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/$s/0000.parquet"
done
ls -l
cd "$GITHUB_WORKSPACE"
python -u research/hfm-projects/agent-harnesses/paper-memory-agent/note_floor.py "$DATA/train.parquet" "$DATA/validation.parquet" "$DATA/test.parquet"
