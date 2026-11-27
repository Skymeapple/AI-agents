# Rail dispatching: does a look-ahead that samples realistic entry delays for unseen
# trains beat one that assumes they are on time? 60 weekdays; first five are used to
# learn the delay distribution and left out of the test. No hindsight bound (see the
# rail-60-days job for that).
set -e
pip -q install numpy scipy
P=research/hfm-projects/optimization/rail-dispatch
python $P/fetch_days.py "$DATA/rail" 2024-01-08 60
RAIL_NO_BOUND=1 python -u $P/rail_sim.py "$DATA/rail"
