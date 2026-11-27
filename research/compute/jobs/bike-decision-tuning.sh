# Layer-4 test: bike forecast policy settings chosen by forecast accuracy against chosen by simulated failures (Helsinki, June 2021).
set -e
pip -q install numpy pandas
curl -sS -L -m 900 -o "$DATA/2021-06.csv" "https://dev.hsl.fi/citybikes/od-trips-2021/2021-06.csv"
timeout 330m python -u research/hfm-projects/optimization/bike-rebalancing/decision_tuning.py "$DATA/2021-06.csv"
