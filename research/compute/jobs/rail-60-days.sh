# Rail dispatching: 60 real weekdays, all policies, hindsight bound with a 60 s limit per day.
set -e
pip -q install numpy scipy
P=research/hfm-projects/optimization/rail-dispatch
python $P/fetch_days.py "$DATA/rail" 2024-01-08 60
python -u $P/rail_sim.py "$DATA/rail"
