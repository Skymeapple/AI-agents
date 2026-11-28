"""Exp 2: Theorem 1 (evaluator-budget lower bound) vs. the direct-probe design.

Lower bound: to distinguish C(B) = tau0 from C(B) >= tau0 + Delta with both error
probabilities <= delta, ANY evaluation protocol (any allocation of attempts,
adaptive or not, any number of tasks) needs total attempts
    N >= ln(1/(2 delta)) / (-ln(1 - c/B)),   c = -ln(1 - Delta/W),
where W = 1 - tau0 is the null's unsolved mass.  ~ (B W / Delta) ln(1/2delta).
Achievability: sample m tasks, run each until first success or B attempts,
exact one-sided binomial test.  Cost <= m (W B + (1-W) E[attempts | solvable]).
We also verify the total-variation bound by Monte Carlo.
"""
import json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def n_lower(B, tau0, Delta, delta):
    W = 1 - tau0
    c = -np.log(1 - Delta / W)
    return np.log(1 / (2 * delta)) / (-np.log1p(-c / B))


def m_probe(tau0, Delta, delta):
    """Smallest m s.t. an exact one-sided binomial test of p=tau0+Delta vs p=tau0
    has both error probabilities <= delta."""
    for m in range(5, 200000):
        k = stats.binom.ppf(delta, m, tau0 + Delta)  # declare 'safe' if S <= k-1 ...
        # choose largest k with P_{alt}(S<=k) <= delta
        k = int(k)
        while k >= 0 and stats.binom.cdf(k, m, tau0 + Delta) > delta:
            k -= 1
        if k < 0:
            continue
        if stats.binom.sf(k, m, tau0) <= delta:  # P_null(S > k) = miss of 'safe'
            return m, k
    return None, None


def mc_tv_check(B, tau0, Delta, N, T=2000, reps=4000, seed=0):
    """Construction: null: W*T tasks p=0, rest p=1; alt: those tasks p=c/B.
    Spread N attempts uniformly over tasks; best test = 'any success on a task
    that also has failures'.  Estimate its power (= TV) and compare to bound."""
    rng = np.random.default_rng(seed)
    W = 1 - tau0
    c = -np.log(1 - Delta / W)
    hard = int(W * T)
    per = max(1, N // T)
    # under alt, number of successes on hard tasks ~ Bin(hard*per, c/B)
    s = rng.binomial(hard * per, c / B, size=reps)
    power = float(np.mean(s > 0))
    bound = float(1 - (1 - c / B) ** (hard * per))
    return power, bound


if __name__ == "__main__":
    tau0, delta = 0.5, 0.05
    rows = []
    for Delta in [0.05, 0.1, 0.2]:
        m, k = m_probe(tau0, Delta, delta)
        for B in [10, 30, 100, 300, 1000, 3000, 10000, 100000]:
            NL = float(n_lower(B, tau0, Delta, delta))
            W = 1 - tau0
            Nprobe = float(m * (W * B + (1 - W) * 1.0))
            rows.append(dict(Delta=Delta, B=B, N_lower=NL, m_probe=m, N_probe=Nprobe,
                             ratio=Nprobe / NL))
    checks = []
    for (B, N) in [(1000, 2000), (1000, 20000), (1000, 200000), (10000, 200000)]:
        pw, bd = mc_tv_check(B, tau0, 0.1, N)
        checks.append(dict(B=B, N=N, mc_power=pw, tv_bound=bd))
    json.dump(dict(tau0=tau0, delta=delta, rows=rows, mc_checks=checks),
              open(os.path.join(OUT, "exp2_budget.json"), "w"), indent=1)
    for r in rows:
        print(r)
    print(checks)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for Delta, col in zip([0.05, 0.1, 0.2], ["C0", "C1", "C2"]):
        rr = [r for r in rows if r["Delta"] == Delta]
        ax.loglog([r["B"] for r in rr], [r["N_lower"] for r in rr], "-", color=col,
                  label=f"lower bound, Δ={Delta}")
        ax.loglog([r["B"] for r in rr], [r["N_probe"] for r in rr], "--", color=col,
                  label=f"direct probe, Δ={Delta}")
    ax.set_xlabel("adversary attempts per task B")
    ax.set_ylabel("evaluator total attempts N")
    ax.set_title(r"Certifying $C(B)<\tau_0+\Delta$ ($\tau_0$=0.5, δ=0.05)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_budget_bound.png"), dpi=150)
