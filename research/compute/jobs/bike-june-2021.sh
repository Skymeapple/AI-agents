# Bike rebalancing: all test weekdays of June 2021 (Helsinki), five policies.
set -e
pip -q install numpy pandas
curl -sS -L -m 900 -o "$DATA/2021-06.csv" "https://dev.hsl.fi/citybikes/od-trips-2021/2021-06.csv"
python -u research/hfm-projects/optimization/bike-rebalancing/bike_sim.py "$DATA/2021-06.csv"
# rerun with a look-ahead that averages over four past days
