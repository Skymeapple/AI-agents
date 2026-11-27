# Wireless MILP stand-in (cell sleeping + association, 400 cells, 6000 users): greedy vs HiGHS vs LNS at 60 s and 300 s, four seeds.
set -e
pip -q install numpy "scipy==1.16.1"
for T in 60 300; do for s in 0 1 2 3; do
  timeout 30m python3 -u research/hfm-projects/optimization/wireless-milp/sleep_milp.py $s $T 400 6000 || echo "stopped seed $s"
done; done
