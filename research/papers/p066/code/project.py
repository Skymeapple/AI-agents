"""
Long-run (age 12 -> 18, then 18 -> 20 without AI) projections of the two-state model
for every parameter vector in the calibrated identified set (results/abc_accepted.npz).

Extra long-run assumptions (all swept; none are estimated):
  zeta : ratio of weekly change in general competence to within-session change in topic mastery
  tau  : plasticity decay constant, eta(age) = eta * exp(-(age-12)/tau)
  beta(age): 'flat' (calibrated beta) or 'maturing' (gap to 1 closes linearly, 60% by 18)
  qA_LR: in-the-moment quality of the unguarded assistant (calibrated qA, or 0.95)
  mu   : persistence mean-reversion toward each student's own baseline

Run: python3 project.py  -> results/projection_*.json, results/fig_projection_*.png
"""
import json, os, time, itertools
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model2 import A, THETA, W, transitions

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
rng = np.random.default_rng(7)
KEYS = ["m_s", "m_g", "eta", "lam", "kappa", "rho", "sigma", "qA", "qT", "c", "Omega", "beta"]
WEEKS, YEARS, POST = 36, 6, 2
NST, SD = 200, 0.8
Z1 = rng.standard_normal(NST); Z2 = rng.standard_normal(NST)


def sig(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def best_action(s, g, P, q, sigma):
    a = A[None, None, :]
    S, G = s[..., None], g[..., None]
    Pe = {k: (v[..., None] if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    Pe["sigma"] = sigma[..., None] if isinstance(sigma, np.ndarray) else sigma
    s2, g2 = transitions(S, G, a, Pe)
    qq = q[..., None] if isinstance(q, np.ndarray) else q
    U = W * ((1 - a) * G * S + a * qq) - Pe["c"] * (1 - a) * (1 - THETA * S) + Pe["beta"] * Pe["Omega"] * g2 * s2
    i = np.argmax(U, -1)
    return A[i], np.take_along_axis(U, i[..., None], -1)[..., 0]


def run(th, regime, zeta=0.25, tau=8.0, beta_mode="flat", qA_LR=None, mu=0.01,
        window=None, record=False):
    """th: dict of arrays (n_draws,). Returns dict of (n_draws, NST) arrays."""
    nd = len(th["eta"])
    s = sig(logit(th["m_s"])[:, None] + SD * Z1[None, :])
    g0 = sig(logit(np.minimum(th["m_g"], 0.98))[:, None] + SD * Z2[None, :])
    g = g0.copy()
    qA = (th["qA"] if qA_LR is None else np.full(nd, qA_LR))[:, None]
    qT = th["qT"][:, None]; sT = th["sigma"][:, None]
    T = WEEKS * (YEARS + POST)
    rec = []; out = {}
    for t in range(T):
        age = 12 + t / WEEKS
        phi = np.exp(-(age - 12) / tau) if np.isfinite(tau) else 1.0
        if beta_mode == "flat":
            be = th["beta"][:, None]
        else:
            frac = min((age - 12) / 10.0, 1.0)
            be = (th["beta"] + (1 - th["beta"]) * frac)[:, None]
        P = dict(eta=th["eta"][:, None] * zeta, phi=phi, lam=th["lam"][:, None] * zeta,
                 kappa=th["kappa"][:, None] * zeta, rho=th["rho"][:, None], c=th["c"][:, None],
                 Omega=th["Omega"][:, None], beta=be, mu=mu, gbar=g0)
        avail = age < 18 and (window is None or (window[0] <= age < window[1]))
        reg = regime if avail else "none"
        if reg == "none":
            a = np.zeros_like(s); sg = 0.0
        elif reg == "answer":
            a, _ = best_action(s, g, P, qA, 0.0); sg = 0.0
        elif reg == "tutor":
            a, _ = best_action(s, g, P, qT, sT); sg = sT
        elif reg == "both":
            aA, uA = best_action(s, g, P, qA, 0.0)
            aT, uT = best_action(s, g, P, qT, sT)
            useT = uT >= uA
            a = np.where(useT, aT, aA); sg = np.where(useT, sT, 0.0)
            if record:
                rec.append(useT.mean())
        P["sigma"] = sg
        s, g = transitions(s, g, a, P)
        if abs(age + 1 / WEEKS - 18) < 1e-9:
            out["E18"], out["s18"], out["g18"] = g * s, s.copy(), g.copy()
    out["E20"], out["s20"], out["g20"] = g * s, s.copy(), g.copy()
    if record:
        out["tutor_share"] = np.array(rec)
    return out


def summarize(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))


def main(max_draws=150):
    t0 = time.time()
    Z = np.load(os.path.join(RES, "abc_accepted.npz"))
    n = len(Z["eta"])
    idx = rng.choice(n, size=min(n, max_draws), replace=False)
    th = {k: Z[k][idx] for k in KEYS}
    res = {"n_draws_used": int(len(idx)), "n_identified_set": int(n)}

    # ---- 1. Regime comparison under baseline long-run assumptions and scenarios
    scen = []
    for zeta, tau, bm, qa in itertools.product([0.1, 0.25, 0.5], [4.0, 8.0, np.inf], ["flat", "maturing"], [None, 0.95]):
        base = run(th, "none", zeta, tau, bm, qa)
        row = dict(zeta=zeta, tau=str(tau), beta_mode=bm, qA_LR="calibrated" if qa is None else qa)
        for reg in ["answer", "tutor", "both"]:
            o = run(th, reg, zeta, tau, bm, qa)
            rel18 = o["E18"].mean(1) / base["E18"].mean(1) - 1
            rel20 = o["E20"].mean(1) / base["E20"].mean(1) - 1
            trapped = (o["E18"] < 0.5 * base["E18"]).mean(1)
            row[reg] = dict(rel_E18=summarize(rel18), rel_E20=summarize(rel20),
                            trapped_share=summarize(trapped),
                            rel_s18=summarize(o["s18"].mean(1) / base["s18"].mean(1) - 1),
                            rel_g18=summarize(o["g18"].mean(1) / base["g18"].mean(1) - 1))
        row["none_E18"] = summarize(base["E18"].mean(1))
        scen.append(row)
    res["scenarios"] = scen

    # ---- 2. Baseline scenario detail + heterogeneity by initial skill
    BASEKW = dict(zeta=0.25, tau=8.0, beta_mode="maturing", qA_LR=None)
    base = run(th, "none", **BASEKW)
    ans = run(th, "answer", **BASEKW)
    tut = run(th, "tutor", **BASEKW)
    both = run(th, "both", record=True, **BASEKW)
    both95 = run(th, "both", record=True, **dict(BASEKW, qA_LR=0.95))
    order = np.argsort(Z1); terc = np.array_split(order, 3)
    het = {}
    for name, o in [("answer", ans), ("tutor", tut), ("both", both)]:
        het[name] = [summarize(o["E18"][:, ix].mean(1) / base["E18"][:, ix].mean(1) - 1) for ix in terc]
    res["baseline_kw"] = {k: str(v) for k, v in BASEKW.items()}
    res["heterogeneity_by_initial_skill_tercile"] = het
    res["both_tutor_choice_share"] = dict(calibrated_qA=summarize(both["tutor_share"]),
                                          qA_0p95=summarize(both95["tutor_share"]))

    # ---- 3. Exposure-window experiment (answer AI available only in a 2-year window)
    win = {}
    for bm in ["flat", "maturing"]:
        for tau in [4.0, 8.0, np.inf]:
            kw = dict(zeta=0.25, tau=tau, beta_mode=bm, qA_LR=None)
            b = run(th, "none", **kw)
            key = f"{bm}_tau{tau}"
            win[key] = {}
            for w0 in [12, 14, 16]:
                o = run(th, "answer", window=(w0, w0 + 2), **kw)
                win[key][f"{w0}-{w0+2}"] = dict(
                    rel_E18=summarize(o["E18"].mean(1) / b["E18"].mean(1) - 1),
                    rel_E20=summarize(o["E20"].mean(1) / b["E20"].mean(1) - 1))
    res["window_experiment"] = win

    # ---- 4. Global sensitivity: PRCC of rel. loss (answer regime, baseline kw) on parameters
    rel = ans["E18"].mean(1) / base["E18"].mean(1) - 1
    X = np.stack([th[k] for k in KEYS], 1)
    R = np.column_stack([stats.rankdata(X[:, j]) for j in range(X.shape[1])])
    ry = stats.rankdata(rel)
    prcc = {}
    for j, k in enumerate(KEYS):
        others = np.delete(R, j, 1); Xo = np.column_stack([np.ones(len(ry)), others])
        rx = R[:, j] - Xo @ np.linalg.lstsq(Xo, R[:, j], rcond=None)[0]
        rr = ry - Xo @ np.linalg.lstsq(Xo, ry, rcond=None)[0]
        prcc[k] = float(np.corrcoef(rx, rr)[0, 1])
    res["prcc_answer_relE18"] = prcc
    res["runtime_sec"] = time.time() - t0
    with open(os.path.join(RES, "projection_results.json"), "w") as f:
        json.dump(res, f, indent=2)

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    labs = ["answer", "tutor", "both"]
    cols = ["#c0392b", "#2c7fb8", "#7b3294"]
    for i, (name, o) in enumerate([("answer", ans), ("tutor", tut), ("both", both)]):
        r = o["E18"].mean(1) / base["E18"].mean(1) - 1
        ax[0].hist(r, bins=30, alpha=0.55, color=cols[i], label=name)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set_xlabel("relative change in unassisted performance at 18 vs no-AI")
    ax[0].set_ylabel("# parameter vectors"); ax[0].legend(frameon=False)
    ax[0].set_title("A. Long-run effect across identified set")
    ks = list(prcc); vals = [prcc[k] for k in ks]
    ax[1].barh(ks, vals, color=["#c0392b" if v < 0 else "#2c7fb8" for v in vals])
    ax[1].axvline(0, color="k", lw=0.8); ax[1].set_title("B. PRCC with long-run effect (answer-AI)")
    for j, (key, lab) in enumerate([("maturing_tau8.0", "maturing beta, tau=8"), ("flat_tau8.0", "flat beta, tau=8"),
                                    ("maturing_tauinf", "maturing beta, no decay")]):
        m = [win[key][w]["rel_E20"]["median"] for w in ["12-14", "14-16", "16-18"]]
        ax[2].plot([13, 15, 17], m, "o-", label=lab)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xlabel("age window of answer-AI exposure (midpoint)")
    ax[2].set_ylabel("median rel. change at age 20")
    ax[2].legend(frameon=False, fontsize=8); ax[2].set_title("C. Timing of exposure")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_projection.png"), dpi=150); plt.close()
    print(json.dumps({k: res[k] for k in ["n_draws_used", "heterogeneity_by_initial_skill_tercile",
                                          "both_tutor_choice_share", "prcc_answer_relE18", "runtime_sec"]}, indent=1))


if __name__ == "__main__":
    main()
