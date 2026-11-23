"""Discretisation check for the Theorem-3 relaxation (fact binning). Compares the default binning
(exact top 1000 facts, 60 log-bins/decade) with a 4x finer one (exact top 10000, 240 bins/decade).
Output: results/relax_resolution.txt"""
import itertools
from theory_relax import relax_opt, fact_bins
lines = ["s    H  lam C     | loss(default) loss(fine) rel.diff | eps*(default) eps*(fine) | h(default) h(fine)"]
for s, H, lam, C in itertools.product([1.2, 1.5], [4, 10], [9], [1e5, 1e8]):
    a = relax_opt(s, H, lam, C, bins=fact_bins(s))
    b = relax_opt(s, H, lam, C, bins=fact_bins(s, I0=10000, per_dec=240))
    la, lb = 1 - a['score'], 1 - b['score']
    lines.append(f"{s:<4} {H:<2} {lam:<3} {C:.0e} | {la:.6e} {lb:.6e} {abs(la-lb)/lb:.1e} | {a['eps']:.3e} {b['eps']:.3e} | {a['halluc']:.3e} {b['halluc']:.3e}")
print("\n".join(lines)); open('../results/relax_resolution.txt', 'w').write("\n".join(lines) + "\n")
