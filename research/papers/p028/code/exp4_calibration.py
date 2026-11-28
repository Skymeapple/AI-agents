"""Exp 4: back-of-envelope calibration of the latent-defect model to published
figures on false-positive acceptance of agent patches, with sensitivity analysis.

SOURCED inputs (see paper Sec. 7 for citations):
  q = 0.078 : share of test-passing ("plausible") SWE-bench Verified patches that fail the
              full developer test suite (Wang, Pradel & Liu, ICSE 2026, abstract)
  q = 0.296 : share of plausible patches behaviourally divergent from the reference patch
              under differential testing (same paper) -- an upper proxy
  q = 0.50  : approx. share of test-passing SWE-bench Verified PRs that maintainers would not
              merge (METR note, 10 March 2026) -- a broader notion of "defect"
ASSUMED inputs: acceptance rate a (swept), defect tolerance m (swept).
Everything else is computed from the model."""
import json, os
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SCEN = {"fail_dev_tests_0.078": 0.078, "behaviour_divergent_0.296": 0.296, "not_mergeable_0.50": 0.50}
out = {"scenarios": {}}

def n50(q, level=0.5):
    return float(np.log(level) / np.log(1 - q))

def n_tol(q, m, level=0.5):
    """largest N with P(Bin(N,q) <= m) >= level"""
    N = m
    while binom.cdf(m, N + 1, q) >= level:
        N += 1
    return N

for name, q in SCEN.items():
    row = dict(q=q, N50=n50(q), N80=n50(q, 0.8), success_at_N=dict(), invert=dict(), tolerance=dict(),
               escape_reduction_needed=dict())
    for N in [10, 30, 100, 300, 1000]:
        row["success_at_N"][str(N)] = float((1 - q) ** N)
    for a in [0.5, 0.7, 0.9]:  # assumed per-attempt acceptance (test-pass) rate
        p = a * (1 - q); s = a * q / (1 - p)
        row["invert"][f"a={a}"] = dict(p=p, escape_s=s)
        for Nt in [100, 1000]:
            qt = 1 - 0.5 ** (1 / Nt)          # latent-defect rate needed for N50 = Nt
            st = qt * p / ((1 - qt) * (1 - p))  # escape rate needed at unchanged p
            row["escape_reduction_needed"][f"a={a},N50={Nt}"] = dict(q_target=qt, s_target=st, factor=s / st)
    for m in [0, 1, 3, 10]:
        row["tolerance"][f"m={m}"] = n_tol(q, m)
    out["scenarios"][name] = row

# generator-only route: with escape s fixed, what p is needed for N50 = Nt ?
gen = {}
for s in [0.05, 0.15, 0.4]:
    for Nt in [100, 1000]:
        qt = 1 - 0.5 ** (1 / Nt)
        # q = (1-p)s/(p+(1-p)s)  ->  p = s(1-q)/(q + s(1-q) - q s) ... solve numerically
        p = brentq(lambda p: (1 - p) * s / (p + (1 - p) * s) - qt, 1e-9, 1 - 1e-12)
        gen[f"s={s},N50={Nt}"] = dict(p_needed=p, error_rate_needed=1 - p)
out["generator_only_route"] = gen
json.dump(out, open(os.path.join(OUT, "exp4_calibration.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
