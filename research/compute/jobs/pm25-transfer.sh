# Air-quality sensors: PM2.5 calibration transferred across cities and sites (SensEURCity).
set -e
pip -q install numpy pandas lightgbm scikit-learn
curl -sS -L -f -m 3000 -o "$DATA/sens.zip" "https://zenodo.org/api/records/17858205/files/senseurcity_data_v03.zip/content"
python -u research/hfm-projects/physics/pm25-calibration/pm25_transfer.py "$DATA/sens.zip" 2>&1 | grep -v "hours with reference$" 
