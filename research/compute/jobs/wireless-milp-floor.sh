# Wireless MILP stand-in, round 2: adds a HiGHS-then-LNS hybrid; 300 s budget, four seeds.
set -e
pip -q install numpy "scipy==1.16.1"
for T in 300; do for s in 0 1 2 3; do
  timeout 40m python3 -u research/hfm-projects/optimization/wireless-milp/sleep_milp.py $s $T 400 6000 || echo "stopped seed $s"
done; done
