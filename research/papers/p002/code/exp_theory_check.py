"""Check Theorem 2 numerically: E_n((1-p)^k) on [0,1] lies in [rho/(4e), rho], rho = P(|H-T| > 2n)
for 2k fair coin tosses (Newman-Rivlin via the z^2 substitution), and compare with erfc(n/sqrt(k))/2.
Also check Theorem 3 (closed form for the verifier ceiling).  Reads results/theory_En.csv."""
import os, numpy as np, pandas as pd
from scipy.stats import binom
from scipy.special import erfc
RES = os.path.join(os.path.dirname(__file__), "..", "results")
df = pd.read_csv(os.path.join(RES, "theory_En.csv"))
d = df[df.family == "passk"].copy()
def rho(k, n):
    N = 2 * k  # tosses; |H-T| > 2n  <=>  H > k+n or H < k-n
    return binom.sf(k + n, N, 0.5) + binom.cdf(k - n - 1, N, 0.5)
d["rho"] = [rho(int(k), int(n)) for k, n in zip(d.k, d.n)]
d = d[d.k > d.n]
d["ratio"] = d.E / d.rho
d["half_erfc"] = 0.5 * erfc(d.n / np.sqrt(d.k))
d["sv_bound"] = 2 * np.exp(-d.n ** 2 / d.k)
big = d[d.E > 1e-5]  # below ~1e-6 the LP solver's tolerance dominates
out = dict(n_rows=len(big), ratio_min=float(big.ratio.min()), ratio_max=float(big.ratio.max()),
           lower_const=1 / (4 * np.e), sandwich_holds=bool((big.ratio >= 1 / (4 * np.e) - 1e-6).all() and (big.ratio <= 1 + 1e-6).all()),
           sv_bound_holds=bool((big.E <= big.sv_bound + 1e-9).all()),
           max_abs_dev_from_half_erfc_for_n_ge_16=float(np.abs(big[big.n >= 16].E - big[big.n >= 16].half_erfc).max()),
           max_rel_dev_from_half_erfc_n_ge_16_x_ge_0p25=float((np.abs(big.E - big.half_erfc) / big.E)[(big.n >= 16) & (big.x >= 0.25)].max()))
c = df[df.family == "ceiling"].copy(); c = c[c.E > 1e-5]
out["ceiling_max_rel_err_vs_closed_form"] = float((np.abs(c.E - c.closed) / c.closed).max())
print(out)
import json; json.dump(out, open(os.path.join(RES, "theory_checks.json"), "w"), indent=1)
d.to_csv(os.path.join(RES, "theory_passk_with_rho.csv"), index=False, float_format="%.6g")
# horizon table: largest k (on the computed grid) with worst-case width 2E <= delta
from scipy.special import erfcinv
rows = []
for n in [8, 16, 32, 64, 128, 256]:
    for delta in [0.01, 0.05]:
        rows.append(dict(n=n, delta=delta, k_rigorous=int(n * n / np.log(4 / delta)),
                         k_erfc_rule=int(n * n / erfcinv(delta) ** 2)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "theory_horizon.csv"), index=False)
print(pd.DataFrame(rows))
rows = []
for n, g in d.groupby("n"):
    for delta in [0.01, 0.05]:
        ok = g[2 * g.E <= delta]
        rows.append(dict(n=n, delta=delta, kmax_grid=int(ok.k.max()) if len(ok) else None,
                         kmax_over_n2=(ok.k.max() / n ** 2) if len(ok) else None))
pd.DataFrame(rows).to_csv(os.path.join(RES, "theory_horizon_grid.csv"), index=False)
print(pd.DataFrame(rows))
