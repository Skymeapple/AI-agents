"""Experiment 5 (added in revision): which way do evaluator-side frontier estimates
bound a schemer, and what can actually be certified?

 (a) Discriminator-based 'ceilings' are LOWER bounds on what a schemer can hide.
     XOR cue. The evaluator trains a discriminator (logistic regression, or an MLP
     of width 2 / 8) on n = 20,000 items per class and reads off its ROC at alpha0.
     We compare these estimated ceilings with the (alpha, beta) actually reached by
     the trained schemers of Exp. 1 (results/exp1_frontier.json). Any schemer above
     an estimated ceiling shows that the estimate is not an upper bound.
 (b) Certified upper bound from deployment-resampled honeypots (Prop. 5).
     GaussianShift d'=2. Evaluation mixture E' = (1-rho) E + rho H with
     H = (1-eta) H_t + eta D: a fraction eta of honeypot episodes are real deployment
     opportunities replayed under monitoring (t = 0.5 synthetic honeypots otherwise).
     Then dD/dH <= 1/eta *by construction*, so beta <= alpha_H / eta for EVERY policy,
     with no estimate of any frontier. With N_H monitored honeypot episodes and k
     detections, the Clopper-Pearson upper limit U gives beta <= U/eta w.p. >= 1-gamma.
     We train adaptive schemers (best responses, width-16, annealed penalty) against
     E' for eta in {0, 0.05, 0.2, 1} and report validity and tightness.
Run: OMP_NUM_THREADS=1 python3 exp5_certify.py   (~5 min on 4 cores)
"""
import json, os
from multiprocessing import Pool
import numpy as np
from scipy.stats import norm, beta as beta_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from common import GaussianShift, XORCue, train_anneal, empirical_roc, RESULTS
from theory import mixture_frontier

# ------------------------------------------------------------------ part (a)
def part_a():
    env = XORCue(8, 1.2)
    rng = np.random.default_rng(2027)
    n = 20000
    X = np.vstack([env.sample(n, "E", rng), env.sample(n, "D", rng)])
    y = np.r_[np.zeros(n), np.ones(n)]
    probes = {
        "logistic": LogisticRegression(max_iter=2000),
        "mlp_w2": MLPClassifier(hidden_layer_sizes=(2,), activation="tanh", max_iter=600, random_state=0),
        "mlp_w8": MLPClassifier(hidden_layer_sizes=(8,), activation="tanh", max_iter=600, random_state=0),
    }
    m = 1_000_000
    xe, xd = env.sample(m, "E", rng), env.sample(m, "D", rng)
    grid = np.logspace(-5, 0, 80)
    rocs = {}
    for k, clf in probes.items():
        clf.fit(X, y)
        se = clf.decision_function(xe) if hasattr(clf, "decision_function") else clf.predict_proba(xe)[:, 1]
        sd = clf.decision_function(xd) if hasattr(clf, "decision_function") else clf.predict_proba(xd)[:, 1]
        if k != "logistic":
            se, sd = clf.predict_proba(xe)[:, 1], clf.predict_proba(xd)[:, 1]
        rocs[k] = empirical_roc(se, sd, grid)
    ll_e, ll_d = env.log_lr(xe), env.log_lr(xd)
    rocs["bayes"] = empirical_roc(ll_e, ll_d, grid)
    ex1 = json.load(open(os.path.join(RESULTS, "exp1_frontier.json")))
    pts = [p for p in ex1["points"] if p["env"] == "xor" and p["alpha"] > 1e-4]
    viol = {}
    for k in ["logistic", "mlp_w2", "mlp_w8"]:
        above = []
        for p in pts:
            est = float(np.interp(np.log(p["alpha"]), np.log(grid), rocs[k]))
            if p["beta"] > est + 0.01:
                above.append(dict(width=p["width"], lam=p["lam"], seed=p["seed"], alpha=p["alpha"],
                                  beta=p["beta"], est_ceiling=est))
        viol[k] = dict(n_schemers=len(pts), n_above_estimated_ceiling=len(above),
                       n_above_by_width={str(w): sum(a["width"] == w for a in above) for w in [0, 2, 8, 32]},
                       max_ratio=float(max([a["beta"] / max(a["est_ceiling"], 1e-9) for a in above], default=0)))
    at = {k: {f"{a0:.0e}": float(np.interp(np.log(a0), np.log(grid), r)) for a0 in [1e-3, 1e-2]}
          for k, r in rocs.items()}
    return dict(grid=grid.tolist(), rocs={k: v.tolist() for k, v in rocs.items()},
                ceilings_at=at, schemers_above=viol), pts


# ------------------------------------------------------------------ part (b)
DP, RHO, T = 2.0, 0.1, 0.5
ETAS = [0.0, 0.05, 0.2, 1.0]
LAMS = [8, 32, 128, 512]
SEEDS = [0, 1]


def job_b(args):
    eta, lam, s = args
    e = GaussianShift(8, DP, 0)
    rng = np.random.default_rng(55001 * s + 13 * lam + int(1000 * eta))
    mix = [(1 - RHO, lambda n, r: e.sample(n, "E", r))]
    if eta < 1:
        mix.append((RHO * (1 - eta), lambda n, r: e.sample_H(n, T, r)))
    if eta > 0:
        mix.append((RHO * eta, lambda n, r: e.sample(n, "D", r)))
    m = train_anneal(e, 16, lam, rng, steps=3000, batch=4096, eval_mix=mix)
    r = np.random.default_rng(3 + s)
    n = 1_000_000
    aE = float(m.prob(e.sample(n, "E", r)).mean())
    aHt = float(m.prob(e.sample_H(n, T, r)).mean())
    b = float(m.prob(e.sample(n, "D", r)).mean())
    aH = (1 - eta) * aHt + eta * b      # honeypot defection rate (D-part is exactly beta)
    return dict(eta=eta, lam=lam, seed=s, alpha_E=aE, alpha_Ht=aHt, alpha_H=aH,
                alpha_mix=(1 - RHO) * aE + RHO * aH, beta=b)


def cp_upper(k, n, gamma):
    return 1.0 if k >= n else float(beta_dist.ppf(1 - gamma, k + 1, n - k))


def part_b(pool):
    B = pool.map(job_b, [(eta, l, s) for eta in ETAS for l in LAMS for s in SEEDS])
    rng = np.random.default_rng(99)
    NH, gamma, reps = 2000, 0.05, 2000
    for q in B:
        comps = [(1 - RHO, 0.0)]
        if q["eta"] < 1:
            comps.append((RHO * (1 - q["eta"]), T * DP))
        if q["eta"] > 0:
            comps.append((RHO * q["eta"], DP))
        q["exact_frontier"] = float(mixture_frontier(DP, comps, max(q["alpha_mix"], 1e-12)))
        if q["eta"] > 0:
            q["certified_population"] = min(1.0, q["alpha_H"] / q["eta"])
            ks = rng.binomial(NH, q["alpha_H"], size=reps)
            U = np.array([cp_upper(k, NH, gamma) for k in ks]) / q["eta"]
            q["certified_finite_median"] = float(np.median(np.minimum(U, 1)))
            q["coverage"] = float(np.mean(q["beta"] <= U))
            q["tightness"] = q["beta"] / q["certified_population"] if q["certified_population"] > 0 else None
    # analytic table: exact achievable ceiling vs certifiable bound at alpha' = c/N
    c = -np.log(0.9)
    tab = {}
    for N in [1e3, 1e4, 1e5]:
        a = c / N
        for eta in ETAS:
            comps = [(1 - RHO, 0.0)]
            if eta < 1:
                comps.append((RHO * (1 - eta), T * DP))
            if eta > 0:
                comps.append((RHO * eta, DP))
            tab[f"N{int(N)}_eta{eta}"] = dict(exact=float(mixture_frontier(DP, comps, a)),
                                             certified=(min(1.0, a / (RHO * eta)) if eta > 0 else None))
    tab["no_honeypots_N1000"] = float(norm.cdf(DP + norm.ppf(c / 1e3)))
    summ = dict(n=len(B),
                bound_valid=int(sum(q["beta"] <= q["certified_population"] + 1e-9 for q in B if q["eta"] > 0)),
                n_eta_pos=int(sum(q["eta"] > 0 for q in B)),
                min_coverage=float(min(q["coverage"] for q in B if q["eta"] > 0)),
                NH=NH, gamma=gamma)
    return dict(points=B, table=tab, summary=summ)


def main():
    A, pts = part_a()
    print(json.dumps(A["ceilings_at"], indent=1)); print(json.dumps(A["schemers_above"], indent=1))
    with Pool(4) as p:
        Bres = part_b(p)
    print(json.dumps(Bres["summary"], indent=1)); print(json.dumps(Bres["table"], indent=1))
    for q in Bres["points"]:
        print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in q.items()})
    json.dump(dict(a=A, b=Bres), open(os.path.join(RESULTS, "exp5_certify.json"), "w"), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    grid = np.array(A["grid"])
    for k, col in [("bayes", "k"), ("mlp_w8", "#2ca02c"), ("mlp_w2", "#ff7f0e"), ("logistic", "#1f77b4")]:
        ax[0].plot(grid, A["rocs"][k], color=col, lw=1.5, label=f"evaluator's estimate: {k}")
    cols = {0: "#1f77b4", 2: "#ff7f0e", 8: "#2ca02c", 32: "#d62728"}
    for w in [0, 2, 8, 32]:
        s = [p for p in pts if p["width"] == w]
        ax[0].scatter([p["alpha"] for p in s], [p["beta"] for p in s], s=14, color=cols[w], marker="x",
                      label=f"trained schemer, width {w}")
    ax[0].set_xscale("log"); ax[0].set_xlim(1e-4, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("α"); ax[0].set_ylabel("β")
    ax[0].set_title("(a) Probe-estimated 'ceilings' are lower bounds (XOR)")
    ax[0].legend(fontsize=6)
    for eta, col in zip(ETAS[1:], ["#ff7f0e", "#2ca02c", "#d62728"]):
        s = [q for q in Bres["points"] if q["eta"] == eta]
        ax[1].scatter([q["certified_population"] for q in s], [q["beta"] for q in s], color=col,
                      label=f"η={eta}")
    xx = np.logspace(-9, 0, 50)
    ax[1].plot(xx, xx, "k--", lw=1, label="β = certified bound")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("certified bound α_H/η"); ax[1].set_ylabel("adaptive schemer's β")
    ax[1].set_title("(b) Deployment-resampled honeypots: certified bound")
    ax[1].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "fig5_certify.png"), dpi=150)


if __name__ == "__main__":
    main()
