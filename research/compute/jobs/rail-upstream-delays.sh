# Rail dispatching, 60 weekdays: look-ahead that uses each approaching train's delay as
# recorded at stations before the corridor, against plain look-ahead and against a
# look-ahead that knows the real entry times.
set -e
pip -q install numpy scipy
P=research/hfm-projects/optimization/rail-dispatch
python $P/fetch_days.py "$DATA/rail" 2024-01-08 60
RAIL_NO_BOUND=1 RAIL_FAST=1 python -u $P/rail_sim.py "$DATA/rail"
