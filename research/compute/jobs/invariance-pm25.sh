# Mapping B test: shared invariance methods (ERM, balanced, group DRO, per-environment standardisation, V-REx, IRM) on PM2.5 calibration, by city and by site.
set -e
pip -q install numpy pandas lightgbm scikit-learn
pip -q install torch --index-url https://download.pytorch.org/whl/cpu
curl -sS -L -f -m 3000 -o "$DATA/sens.zip" "https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content"
python -u research/hfm-projects/_framework/run_invariance.py pm25 "$DATA/sens.zip" 2>&1 | grep -v "hours with reference$"
