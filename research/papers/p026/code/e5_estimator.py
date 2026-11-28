"""E5: Estimating long-horizon identity persistence from short probe chains.

Ground truth: M facts, each with its own corruption rate p_i (heterogeneous: a fast
and a slow subpopulation) and attractor mass pi_i = pi(x_i) (default-matching or
idiosyncratic). Population fidelity F(t) = mean_i[pi_i + (1-pi_i)(1-p_i)^t].
Observed data: per fact, n_chains replicate chains of K consolidation steps (probe
correctness after each step) plus m cold-start probes of the memoryless model
(each returns x_i w.p. pi_i).
Estimators of F(t) at long horizons:
  A  naive: least-squares fit of c + (1-c) exp(-b t) to the aggregate curve t=0..K
  B  aggregate exponential with floor fixed to the cold-start estimate
  C  per-fact plug-in MLE (p_hat_i from correct->wrong transitions)
  D  per-fact empirical Bayes (nonparametric MLE prior over p on a grid, EM)  [proposed]
"""
import json, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
OUT = Path(__file__).resolve().parents[1] / "results"
rng = np.random.default_rng(5)

M, K, NCH, MCOLD = 300, 5, 20, 50
HORIZONS = [5, 20, 50, 200, 1000]
GRID = np.concatenate([[0.0], np.logspace(-5, 0, 400)])

SCEN = "bimodal"
def make_facts():
    if SCEN == "bimodal":       # 50% fast (mean p~0.09), 50% slow (mean p~0.0033)
        fast = rng.random(M) < 0.5
        p = np.where(fast, rng.beta(2, 20, M), rng.beta(1, 300, M))
    elif SCEN == "lognormal":   # continuous heterogeneity, median p = 0.01
        p = np.clip(np.exp(rng.normal(np.log(0.01), 1.5, M)), 0, 1)
    elif SCEN == "subresolution":  # 40% fast, 60% with p=2e-4, far below 1/(n_chains*K)=0.01
        p = np.where(rng.random(M) < 0.4, rng.beta(2, 20, M), 2e-4)
    pi = np.where(rng.random(M) < 0.3, 0.5, 0.5 / 9)
    return p, pi

def truth(p, pi, t):
    return float(np.mean(pi + (1 - pi) * (1 - p) ** t))

def observe(p, pi):
    # correctness chains: state 1 = correct
    st = np.ones((M, NCH), dtype=int); traj = [st.copy()]
    n_c = np.zeros(M); k_cw = np.zeros(M)
    for _ in range(K):
        u = rng.random((M, NCH))
        to_wrong = (st == 1) & (u < (p * (1 - pi))[:, None])
        to_right = (st == 0) & (u < (p * pi)[:, None])
        n_c += (st == 1).sum(1); k_cw += to_wrong.sum(1)
        st = np.where(to_wrong, 0, np.where(to_right, 1, st)); traj.append(st.copy())
    agg = np.array([tr.mean() for tr in traj])          # aggregate curve t=0..K
    pi_hat = rng.binomial(MCOLD, pi) / MCOLD
    return agg, n_c, k_cw, pi_hat

def est_A(agg):
    t = np.arange(K + 1)
    f = lambda t, c, b: c + (1 - c) * np.exp(-b * t)
    (c, b), _ = curve_fit(f, t, agg, p0=[0.3, 0.05], bounds=([0, 0], [1, 10]))
    return lambda T: c + (1 - c) * np.exp(-b * T)

def est_B(agg, pi_hat):
    c = pi_hat.mean(); t = np.arange(K + 1)
    f = lambda t, b: c + (1 - c) * np.exp(-b * t)
    (b,), _ = curve_fit(f, t, agg, p0=[0.05], bounds=([0], [10]))
    return lambda T: c + (1 - c) * np.exp(-b * T)

def est_C(n_c, k_cw, pi_hat):
    ph = np.clip((k_cw / np.maximum(n_c, 1)) / np.maximum(1 - pi_hat, 1e-3), 0, 1)
    return lambda T: float(np.mean(pi_hat + (1 - pi_hat) * (1 - ph) ** T))

def est_D(n_c, k_cw, pi_hat, iters=500):
    # likelihood of k_cw | p on grid: Binomial(n_c, p*(1-pi_hat)) (constant terms dropped)
    th = np.clip(GRID[None, :] * (1 - pi_hat)[:, None], 1e-12, 1 - 1e-12)
    ll = k_cw[:, None] * np.log(th) + (n_c - k_cw)[:, None] * np.log1p(-th)
    L = np.exp(ll - ll.max(1, keepdims=True))
    w = np.full(len(GRID), 1 / len(GRID))
    for _ in range(iters):                     # NPMLE via EM (Kiefer-Wolfowitz)
        post = L * w; post /= post.sum(1, keepdims=True)
        w = post.mean(0)
    post = L * w; post /= post.sum(1, keepdims=True)
    return lambda T: float(np.mean(pi_hat + (1 - pi_hat) * (post * (1 - GRID[None, :]) ** T).sum(1)))

REPS = 40
ALL = {}
for SCEN in ["bimodal", "lognormal", "subresolution"]:
    print("=== scenario", SCEN)
    errs = {k: {h: [] for h in HORIZONS} for k in "ABCD"}
    truths = {h: [] for h in HORIZONS}
    for rep in range(REPS):
        p, pi = make_facts()
        agg, n_c, k_cw, pi_hat = observe(p, pi)
        ests = dict(A=est_A(agg), B=est_B(agg, pi_hat), C=est_C(n_c, k_cw, pi_hat), D=est_D(n_c, k_cw, pi_hat))
        for h in HORIZONS:
            tr = truth(p, pi, h); truths[h].append(tr)
            for k, f in ests.items():
                errs[k][h].append(float(f(h)) - tr)
    summary = {}
    for k in "ABCD":
        summary[k] = {h: dict(bias=float(np.mean(errs[k][h])), rmse=float(np.sqrt(np.mean(np.square(errs[k][h])))))
                      for h in HORIZONS}
    summary["truth_mean"] = {h: float(np.mean(truths[h])) for h in HORIZONS}
    names = dict(A="naive aggregate exp fit", B="aggregate exp, floor=cold-start", C="per-fact plug-in MLE", D="per-fact empirical Bayes (NPMLE)")
    print("true mean F(t):", {h: round(v, 3) for h, v in summary["truth_mean"].items()})
    for k in "ABCD":
        print(f"{k} {names[k]:34}", "  ".join(f"t={h}: bias={summary[k][h]['bias']:+.3f} rmse={summary[k][h]['rmse']:.3f}" for h in HORIZONS))
    ALL[SCEN] = summary

json.dump(dict(scenarios=ALL, names=names, config=dict(M=M, K=K, n_chains=NCH, m_cold=MCOLD, reps=REPS)),
          open(OUT / "e5_estimator.json", "w"), indent=1)
SCEN = "bimodal"
# figure: one representative replicate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p, pi = make_facts(); agg, n_c, k_cw, pi_hat = observe(p, pi)
ests = dict(A=est_A(agg), B=est_B(agg, pi_hat), C=est_C(n_c, k_cw, pi_hat), D=est_D(n_c, k_cw, pi_hat))
T = np.unique(np.logspace(0, 3, 60).astype(int))
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(T, [truth(p, pi, t) for t in T], "k-", lw=2.5, label="ground truth")
for k, c in zip("ABCD", ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]):
    ax.plot(T, [ests[k](t) for t in T], "--", color=c, label=f"{k}: {names[k]}")
ax.plot(np.arange(K + 1)[1:], agg[1:], "ko", ms=4, label="observed (t≤5)")
ax.set_xscale("log"); ax.set_xlabel("consolidation steps t"); ax.set_ylabel("population identity fidelity F(t)")
ax.legend(fontsize=6.5); fig.tight_layout(); fig.savefig(OUT / "fig5_estimator.png", dpi=150)
