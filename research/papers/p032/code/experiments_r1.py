"""Revision-1 experiments (R0-R7). Usage:  python3 experiments_r1.py
Outputs go to ../results/r1_*.json and ../results/fig_r1_*.png.

R0  validation of the extended DP (copies r, copy correlation rho_s, threshold t,
    correction prob c, full topology) against direct Monte Carlo.
R1  design study with games CALIBRATED to the four NSO success rates published by
    Engels et al. (2025) at a 400-Elo gap, over families of slope pairs consistent
    with their qualitative slope description; matched-cost families
    chain / chain-r (same-generation copies) / graph / hybrid; full topology included.
R2  sensitivity sweep over (m_G, m_H, beta).
R3  robustness of the design conclusions to imperfect correction c, threshold
    aggregation t, and a false-alarm cost / false-alarm constraint.
R4  distance-dependent correlation and one-factor misspecification (Monte Carlo).
R5  Theorem 3 bounds and the leading-order conjecture (quasi-stationary analysis).
R6  regularity audit of the original E6 table (anchor choice, houdini-steeper gains).
"""
import json
import os
import time
import itertools
from multiprocessing import Pool

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, multivariate_normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nso import (Game, RDesign, dp_fast, monte_carlo_r, false_alarms, flag_prob,
                 window_transition, R_k)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "lines.linewidth": 2})
LOG = {}


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


# ------------------------------------------------------------------ designs
def design_space(B=32, kmax=5, nmax_graph=20, rs=(1, 2, 3, 4), hyb_r=(2,), hyb_k=(2, 3)):
    out = []
    for n in range(1, B + 1):
        for r in rs:                                        # chain and chain-r
            if r * n <= B:
                out.append(dict(n=n, k=1, anchor=False, r=r, full=False))
        if n > nmax_graph or n < 2:
            continue
        out.append(dict(n=n, k=1, anchor=True, r=1, full=False))   # anchor + chain
        for k in range(2, min(kmax, n - 1) + 1):
            for anchor in (False, True):
                if anchor and k >= n - 1:
                    continue
                out.append(dict(n=n, k=k, anchor=anchor, r=1, full=False))
                if k in hyb_k and not anchor:
                    for r in hyb_r:
                        out.append(dict(n=n, k=k, anchor=False, r=r, full=False))
        out.append(dict(n=n, k=n, anchor=False, r=1, full=True))    # full topology
    keep = []
    for d in out:
        dd = RDesign(**d)
        if dd.cost() <= B:
            d["cost"] = dd.cost()
            d["family"] = dd.family()
            keep.append(d)
    return keep


def eval_setting(args):
    """Evaluate every design for one (game, Delta, rho, rho_s mode, c, t)."""
    gp, Delta, rho, rs_mode, c, t, dspace = args
    g = Game(*gp)
    rows = []
    for d in dspace:
        if isinstance(rs_mode, float):
            rho_s = rs_mode
        else:
            rho_s = rho if rs_mode == "eq" else (1 + rho) / 2 if rs_mode == "mid" else 1.0
        dd = RDesign(n=d["n"], k=d["k"], anchor=d["anchor"], r=d["r"], full=d["full"],
                     rho=rho, rho_s=rho_s, gamma=0.0, t=t, c=c)
        tgt, st = dp_fast(g, dd, Delta)
        fa = {}
        for phi in (0.01, 0.05):
            e, pany = false_alarms(dd, phi)
            fa[f"efa_{phi}"] = e
            fa[f"pfa_{phi}"] = pany
        rows.append(dict(d, target=float(tgt), strict=float(st), **fa))
    return rows


def best_by_family(rows, B, key="target", kappa=0.0, phi=0.05, pfa_max=None):
    res = {}
    for fam in ("chain", "chain-r", "graph", "hybrid"):
        cand = [r for r in rows if r["family"] == fam and r["cost"] <= B
                and (pfa_max is None or r[f"pfa_{phi}"] <= pfa_max)]
        if not cand:
            continue
        val = lambda r: r[key] - kappa * r[f"efa_{phi}"]
        b = max(cand, key=val)
        res[fam] = dict(n=b["n"], k=b["k"], anchor=b["anchor"], r=b["r"], full=b["full"],
                        cost=b["cost"], value=val(b), target=b["target"], strict=b["strict"],
                        efa=b[f"efa_{phi}"], pfa=b[f"pfa_{phi}"])
    return res


# ----------------------------------------------------------- R0 validation
def r0_validation():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(24):
        g = Game(float(rng.choice([0.4, 1.0])), float(rng.choice([0.4, 1.0])),
                 float(rng.choice([-400, -300, -100])))
        full = bool(i % 6 == 5)
        n = int(rng.integers(3, 9 if full else 13))
        rho = float(rng.choice([0, .3, .6]))
        d = RDesign(n=n, k=int(rng.integers(1, 4)), anchor=bool(rng.integers(0, 2)) and not full,
                    r=int(rng.integers(1, 4)), rho=rho, rho_s=float(min(1, rho + rng.choice([0, .2, .4]))),
                    gamma=float(rng.choice([0, .25])), t=int(rng.integers(1, 3)),
                    c=float(rng.choice([1, .8, .6])), full=full)
        Delta = float(rng.choice([400, 800, 1600]))
        tg, st = dp_fast(g, d, Delta)
        mc = monte_carlo_r(g, d, Delta, n_runs=200_000, seed=100 + i)
        rows.append(dict(game=[g.mG, g.mH, g.beta], n=d.n, k=d.k, anchor=d.anchor, r=d.r,
                         rho=d.rho, rho_s=d.rho_s, gamma=d.gamma, t=d.t, c=d.c, full=full,
                         Delta=Delta, dp_target=float(tg), mc_target=float(mc["target"]),
                         dp_strict=float(st), mc_strict=float(mc["strict"]),
                         z=float((tg - mc["target"]) /
                                 max(np.sqrt(tg * (1 - tg) / 200_000), mc["target_se"], 1e-6))))
    save("r1_r0_validation.json", rows)
    LOG["r0"] = dict(n=len(rows), max_abs_z=max(abs(r["z"]) for r in rows),
                     max_err_target=max(abs(r["dp_target"] - r["mc_target"]) for r in rows),
                     max_err_strict=max(abs(r["dp_strict"] - r["mc_strict"]) for r in rows))


# ------------------------------------------------- R1 calibrated design study
# Published NSO success at a 400-Elo general gap (Engels et al. 2025, abstract).
PUBLISHED = {"Debate": 0.517, "Mafia": 0.135, "Backdoor Code": 0.100, "Wargames": 0.094}
# Slope pairs (m_G, m_H). Debate: Guard slope positive, Houdini slope near zero;
# Backdoor Code: Houdini slope steeper than Guard slope (qualitative description
# in Engels et al.). For Mafia and Wargames we do not use slope information and
# span symmetric and both asymmetric cases.
SLOPES = {"Debate": [(0.5, 0.0), (1.0, 0.0), (1.0, 0.25)],
          "Backdoor Code": [(0.1, 1.0), (0.25, 1.0), (0.4, 0.8)],
          "Mafia": [(1.0, 1.0), (0.75, 1.0), (1.0, 0.75)],
          "Wargames": [(1.0, 1.0), (0.75, 1.0), (1.0, 0.75)]}


def chain_opt(g, Delta, nmax=20):
    best = (0, None)
    for n in range(1, nmax + 1):
        v = dp_fast(g, RDesign(n=n), Delta)[0]
        if v > best[0]:
            best = (float(v), n)
    return best


def calibrate(mG, mH, s_star, Delta=400.0):
    f = lambda b: chain_opt(Game(mG, mH, b), Delta)[0] - s_star
    return brentq(f, -3000, 3000, xtol=1e-6)


def r1_calibrated(pool):
    games = []
    for name, s in PUBLISHED.items():
        for (mG, mH) in SLOPES[name]:
            b = calibrate(mG, mH, s)
            v, n = chain_opt(Game(mG, mH, b), 400.0)
            games.append(dict(game=name, mG=mG, mH=mH, beta=b, s_star=s, chain400=v, n400=n))
    dsp = design_space(32)
    jobs, meta = [], []
    for gi, gm in enumerate(games):
        for Delta in (400.0, 800.0, 1600.0):
            for rho in (0.0, 0.5, 0.9):
                for mode in ("eq", "mid"):
                    jobs.append(((gm["mG"], gm["mH"], gm["beta"]), Delta, rho, mode, 1.0, 1, dsp))
                    meta.append((gi, Delta, rho, mode))
    res = pool.map(eval_setting, jobs, chunksize=1)
    table = []
    for (gi, Delta, rho, mode), rows in zip(meta, res):
        for B in (16, 32):
            table.append(dict(**games[gi], Delta=Delta, rho=rho, rho_s_mode=mode, B=B,
                              target=best_by_family(rows, B, "target"),
                              strict=best_by_family(rows, B, "strict")))
    save("r1_r1_calibrated.json", dict(games=games, n_designs=len(dsp), table=table))
    LOG["r1_games"] = games
    LOG["r1_n_designs"] = len(dsp)
    LOG["r1_n_settings"] = len(table)
    return games, table


# ----------------------------------------------------------- R2 sensitivity
def r2_sensitivity(pool):
    dsp = [d for d in design_space(32, kmax=4, nmax_graph=16, rs=(1, 2, 3, 4), hyb_r=())
           if not d["full"]]
    grid = list(itertools.product((0.25, 0.5, 0.75, 1.0), (0.0, 0.5, 1.0, 1.25),
                                  (-500.0, -300.0, -100.0)))
    jobs, meta = [], []
    for (mG, mH, b) in grid:
        for Delta in (800.0, 1600.0):
            for rho in (0.0, 0.5):
                jobs.append(((mG, mH, b), Delta, rho, "eq", 1.0, 1, dsp))
                meta.append((mG, mH, b, Delta, rho))
    res = pool.map(eval_setting, jobs, chunksize=2)
    out = []
    for (mG, mH, b, Delta, rho), rows in zip(meta, res):
        bt = best_by_family(rows, 32, "target")
        bs = best_by_family(rows, 32, "strict")
        p1 = Game(mG, mH, b).p(0, Delta)     # direct (one-step) oversight prob
        out.append(dict(mG=mG, mH=mH, beta=b, Delta=Delta, rho=rho, p_direct=p1,
                        chain=bt["chain"]["value"], chain_r=bt["chain-r"]["value"],
                        graph=bt["graph"]["value"],
                        graph_design=[bt["graph"][x] for x in ("n", "k", "anchor")],
                        chain_r_design=[bt["chain-r"][x] for x in ("n", "r")],
                        strict_chain=bs["chain"]["value"], strict_chain_r=bs["chain-r"]["value"],
                        strict_graph=bs["graph"]["value"]))
    save("r1_r2_sensitivity.json", out)
    # summary statistics
    def frac(cond, rows):
        rows = list(rows)
        return sum(cond(r) for r in rows) / max(len(rows), 1)
    summ = {}
    for rho in (0.0, 0.5):
        R = [r for r in out if r["rho"] == rho]
        summ[f"rho={rho}"] = dict(
            n=len(R),
            frac_graph_ge_chain_r=frac(lambda r: r["graph"] >= r["chain_r"] - 1e-9, R),
            frac_graph_gain_gt_0p1_over_chain=frac(lambda r: r["graph"] - r["chain"] > 0.1, R),
            frac_graph_gain_gt_0p1_over_chain_r=frac(lambda r: r["graph"] - r["chain_r"] > 0.1, R),
            frac_strict_graph_ge_chain_r=frac(lambda r: r["strict_graph"] >= r["strict_chain_r"] - 1e-9, R),
            houdini_steeper=dict(
                n=len([r for r in R if r["mH"] > r["mG"]]),
                frac_gain_gt_0p1=frac(lambda r: r["graph"] - r["chain"] > 0.1,
                                      [r for r in R if r["mH"] > r["mG"]])),
            guard_steeper_or_equal=dict(
                n=len([r for r in R if r["mH"] <= r["mG"]]),
                frac_gain_gt_0p1=frac(lambda r: r["graph"] - r["chain"] > 0.1,
                                      [r for r in R if r["mH"] <= r["mG"]])))
        # gain as function of direct-oversight probability bins
        bins = [0, 1e-3, 1e-2, 0.05, 0.2, 1.0]
        summ[f"rho={rho}"]["gain_by_p_direct"] = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            S = [r for r in R if lo <= r["p_direct"] < hi]
            if S:
                summ[f"rho={rho}"]["gain_by_p_direct"].append(dict(
                    p_direct=[lo, hi], n=len(S),
                    median_gain_vs_chain=float(np.median([r["graph"] - r["chain"] for r in S])),
                    max_gain_vs_chain=float(np.max([r["graph"] - r["chain"] for r in S])),
                    median_gain_vs_chain_r=float(np.median([r["graph"] - r["chain_r"] for r in S])),
                    max_graph=float(np.max([r["graph"] for r in S]))))
    save("r1_r2_summary.json", summ)
    LOG["r2"] = summ
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, rho in zip(axes, (0.0, 0.5)):
        R = [r for r in out if r["rho"] == rho]
        hs = np.array([r["mH"] > r["mG"] for r in R])
        x = np.array([r["p_direct"] for r in R])
        ax.scatter(x[~hs], [r["graph"] - r["chain"] for r, h in zip(R, hs) if not h], s=12,
                   color=C[0], label="graph − chain (m_H ≤ m_G)")
        ax.scatter(x[hs], [r["graph"] - r["chain"] for r, h in zip(R, hs) if h], s=12,
                   color=C[1], marker="^", label="graph − chain (m_H > m_G)")
        ax.scatter(x, [r["graph"] - r["chain_r"] for r in R], s=8, color="#777777",
                   marker="x", label="graph − chain-r (matched cost)")
        ax.set_xscale("log"); ax.axhline(0, color="#333333", lw=0.8)
        ax.set_xlabel("direct-oversight probability p(0 → target)")
        ax.set_title(f"target-success gain, B=32, ρ={rho}", fontsize=9)
    axes[0].set_ylabel("gain in target success"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_r1_sensitivity.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------- R3 robustness
def r3_robustness(pool, games):
    pick = {"symmetric (orig.)": (1.0, 1.0, -300.0)}
    for gm in games:
        if (gm["game"], gm["mG"], gm["mH"]) in (("Debate", 1.0, 0.0), ("Backdoor Code", 0.25, 1.0),
                                                ("Mafia", 1.0, 1.0)):
            pick[f'{gm["game"]} (m_G={gm["mG"]}, m_H={gm["mH"]})'] = (gm["mG"], gm["mH"], gm["beta"])
    dsp = design_space(32, kmax=5, nmax_graph=16, rs=(1, 2, 3, 4), hyb_r=(2,), hyb_k=(2, 3))
    jobs, meta = [], []
    for name, gp in pick.items():
        for Delta in (800.0, 1600.0):
            for rho in (0.0, 0.5):
                for c in (1.0, 0.9, 0.7, 0.5):
                    for t in (1, 2):
                        jobs.append((gp, Delta, rho, "eq", c, t, dsp))
                        meta.append((name, Delta, rho, c, t))
    res = pool.map(eval_setting, jobs, chunksize=1)
    out = []
    for (name, Delta, rho, c, t), rows in zip(meta, res):
        entry = dict(game=name, Delta=Delta, rho=rho, c=c, t=t, objectives={})
        for kappa in (0.0, 0.02, 0.1):
            for phi in (0.01, 0.05):
                if kappa == 0.0 and phi == 0.01:
                    continue
                entry["objectives"][f"kappa={kappa},phi={phi}"] = best_by_family(
                    rows, 32, "target", kappa=kappa, phi=phi)
        entry["objectives"]["pfa<=0.25,phi=0.05"] = best_by_family(
            rows, 32, "target", phi=0.05, pfa_max=0.25)
        entry["strict"] = best_by_family(rows, 32, "strict")
        out.append(entry)
    save("r1_r3_robustness.json", out)
    LOG["r3_n"] = len(out)
    return out


# --------------------------------------------- R4 distance-dependent correlation
def r4_distance_corr():
    g = Game(1.0, 1.0, -300.0)
    Delta, n = 1600.0, 16
    out = []
    for rmax, eta in [(0.5, 0.6), (0.5, 0.9), (0.8, 0.85), (0.9, 0.95)]:
        def corr_fn(O, rmax=rmax, eta=eta):
            O = np.asarray(O)
            D = np.abs(O[:, None] - O[None, :])
            R = rmax * eta ** (D - 1.0)   # PSD because rmax <= eta (KMS)
            np.fill_diagonal(R, 1.0)
            return R
        for k in (1, 2, 3, 4):
            d = RDesign(n=n, k=k)
            mc = monte_carlo_r(g, d, Delta, n_runs=200_000, seed=21 + k, corr_fn=corr_fn)
            # one-factor fit: average pairwise correlation in a full window
            if k >= 2:
                O = list(range(k))
                R = corr_fn(O); rbar = float(R[np.triu_indices(k, 1)].mean())
            else:
                rbar = 0.0
            tg, st = dp_fast(g, RDesign(n=n, k=k, rho=rbar), Delta)
            # bracket: one-factor at the smallest / largest pairwise correlation
            rlo = float(rmax * eta ** (k - 2)) if k >= 2 else 0.0
            tlo, slo = dp_fast(g, RDesign(n=n, k=k, rho=rmax), Delta)
            thi, shi = dp_fast(g, RDesign(n=n, k=k, rho=rlo), Delta)
            out.append(dict(rho_adjacent=rmax, eta=eta, k=k, mc_target=float(mc["target"]),
                            mc_strict=float(mc["strict"]), se=float(mc["target_se"]),
                            onefactor_rho_bar=rbar, onefactor_target=float(tg),
                            onefactor_strict=float(st),
                            onefactor_target_at_rho_adjacent=float(tlo),
                            onefactor_target_at_rho_farthest=float(thi)))
    save("r1_r4_distance_corr.json", out)
    LOG["r4"] = out


# ---------------------------------------------------- R5 Theorem 3 numerics
def qsd(M):
    w, V = np.linalg.eig(M.T)
    i = int(np.argmax(w.real))
    v = np.abs(V[:, i].real)
    return float(w[i].real), v / v.sum()


def theorem3_check(q, rho):
    """Return exact hazard (via h = q_k nu(T C^{k-1})), R_k, and the bounds of Thm 3."""
    q = np.asarray(q, float); k = len(q)
    qfun = lambda m: q[m - 1]
    M = window_transition(qfun, k, rho)
    lam, nu = qsd(M)
    keep = [s for s in range(1 << k) if s != 0]
    idx = {s: i for i, s in enumerate(keep)}
    s_TC = 1                       # bit 0 = oldest trusted, others compromised
    s_TT = (1 << k) - 1
    h_id = q[-1] * nu[idx[s_TC]]   # exact identity
    h_eig = 1 - lam
    Rk = R_k(q, rho)
    prodq = float(np.prod(q))
    if k == 2:
        a1 = q[0] / (lam + q[0])
        Pb = Rk / q[-1]
        upper = Rk / (lam - q[0] * (1 - q[1]) / lam)
        nuTC_ub = Pb / (lam - q[0] * (1 - q[1]) / lam)
        lower = Rk * max(0.0, 1 - nuTC_ub * (1 + 1 / lam))
    else:
        a1 = q[0] / (lam + q[0] - q[k - 2]) if k >= 3 else q[0] / (lam + q[0])
        lower = Rk * max(0.0, 1 - k * a1 * lam ** (-(k - 2)))
        upper = prodq / lam ** (k - 1)
    return dict(q=q.tolist(), rho=rho, k=k, h=float(h_id), h_eig=float(h_eig), R_k=Rk,
                prod_q=prodq, lower=float(lower), upper=float(upper), nu_TT=float(nu[idx[s_TT]]),
                ok=bool(lower <= h_id * (1 + 1e-9) and h_id <= upper * (1 + 1e-9)))


def mp_hazard(q, rho, dps=60):
    import mpmath as mp
    mp.mp.dps = dps
    q = [mp.mpf(float(x)) for x in q]; k = len(q)
    keep = [s for s in range(1, 1 << k)]
    idx = {s: i for i, s in enumerate(keep)}
    M = mp.zeros(len(keep), len(keep))
    for s in keep:
        tr = [q[k - pos - 1] for pos in range(k) if (s >> pos) & 1]   # distance k-pos
        miss = (mp.fprod(tr) if rho == 0 else min(tr)) if tr else mp.mpf(1)
        for nb, w in ((1, 1 - miss), (0, miss)):
            ns = (s >> 1) | (nb << (k - 1))
            if ns != 0:
                M[idx[s], idx[ns]] += w
    E, ER = mp.eig(M.T)
    i = max(range(len(E)), key=lambda j: mp.re(E[j]))
    v = [abs(mp.re(ER[r, i])) for r in range(len(keep))]
    tot = mp.fsum(v)
    return float(q[-1] * v[idx[1]] / tot)


def r5_theorem3():
    rng = np.random.default_rng(5)
    rand = []
    for i in range(300):
        k = int(rng.integers(1, 6)) if i % 3 else int(rng.integers(2, 6))
        k = max(k, 1)
        q = np.sort(rng.uniform(0.001, 0.45, size=k))
        rho = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]))
        if k == 1:
            continue
        rand.append(theorem3_check(q, rho))
    # scaling test: q_m = eps * a_m. For rho in {0,1} the transition matrix has
    # closed-form entries and is solved in 60-digit arithmetic (mpmath); for
    # rho = 0.5 double precision is used and points with R_k < 1e-11 are dropped.
    a = np.array([1.0, 1.4, 1.8, 2.3, 2.8])
    scal = []
    for k in (2, 3, 4, 5):
        for rho in (0.0, 0.5, 1.0):
            for eps in (0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
                if rho in (0.0, 1.0):
                    q = eps * a[:k]
                    h = mp_hazard(q, rho)
                    Rk = float(np.prod(q ** np.arange(1, k + 1))) if rho == 0 else float(np.prod(q))
                    scal.append(dict(k=k, rho=rho, eps=eps, ratio=h / Rk, precision="mp60"))
                    continue
                r = theorem3_check(eps * a[:k], rho)
                if r["R_k"] < 1e-11:
                    continue
                scal.append(dict(k=k, rho=rho, eps=eps, ratio=r["h"] / r["R_k"],
                                 lower_ratio=r["lower"] / r["R_k"], upper_ratio=r["upper"] / r["R_k"],
                                 precision="double"))
    # identity check vs eigenvalue
    # (1 - lambda) loses precision when h is tiny, so compare where h > 1e-8
    id_err = max(abs(r["h"] - r["h_eig"]) / r["h"] for r in rand if r["h"] > 1e-8)
    save("r1_r5_theorem3.json", dict(random=rand, scaling=scal))
    LOG["r5"] = dict(n_random=len(rand), all_bounds_hold=all(r["ok"] for r in rand),
                     max_rel_identity_err=float(id_err),
                     scaling={f"k={s['k']},rho={s['rho']},eps={s['eps']}": s["ratio"] for s in scal})
    # figure
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for ci, k in enumerate((2, 3, 4, 5)):
        for rho, ls in ((0.0, "-"), (0.5, "--"), (1.0, ":")):
            S = [s for s in scal if s["k"] == k and s["rho"] == rho]
            ax.plot([s["eps"] for s in S], [s["ratio"] - 1 for s in S], color=C[ci], ls=ls,
                    label=f"k={k}" if rho == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("miss-probability scale ε (q_m = ε a_m)")
    ax.set_ylabel("h / R_k − 1")
    ax.set_title("Thm 3 / Conj. 3′: exact hazard vs R_k (solid ρ=0, dashed 0.5, dotted 1)", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_r1_theorem3.png"), dpi=160)
    plt.close(fig)


# ------------------------------- R7 same-generation correlation crossover (M3)
def r7_rho_s(pool):
    dsp = design_space(32)
    jobs, meta = [], []
    for gname, gp in (("symmetric", (1.0, 1.0, -300.0)), ("guard-steeper", (1.0, 0.4, -100.0)),
                      ("houdini-steeper", (0.4, 1.0, -400.0))):
        for Delta in (800.0, 1600.0):
            for rho in (0.0, 0.5):
                for rho_s in np.round(np.linspace(rho, 1.0, 6), 3):
                    jobs.append((gp, Delta, rho, float(rho_s), 1.0, 1, dsp))
                    meta.append((gname, Delta, rho, float(rho_s)))
    res = pool.map(eval_setting, jobs, chunksize=1)
    out = []
    for (gname, Delta, rho, rho_s), rows in zip(meta, res):
        for B in (16, 32):
            out.append(dict(regime=gname, Delta=Delta, rho=rho, rho_s=rho_s, B=B,
                            target=best_by_family(rows, B, "target"),
                            strict=best_by_family(rows, B, "strict")))
    save("r1_r7_rho_s.json", out)
    LOG["r7_n"] = len(out)


# ----------------------------------------------- R6 audit of original E6 claims
def r6_audit():
    E6 = json.load(open(os.path.join(OUT, "e6_budget.json")))
    out = {}
    for reg in ("symmetric", "guard-steeper", "houdini-steeper"):
        R = [r for r in E6 if r["regime"] == reg]
        out[reg] = dict(
            n=len(R),
            anchor_chosen_target=sum(r["best_target"]["anchor"] for r in R),
            anchor_by_Delta={str(D): sum(r["best_target"]["anchor"] for r in R if r["Delta"] == D)
                             for D in (400, 800, 1600)},
            max_target_gain={str(D): max(r["best_target"]["value"] - r["chain_target"]["value"]
                                         for r in R if r["Delta"] == D) for D in (400, 800, 1600)},
            max_strict_gain_rho09=max(r["best_strict"]["value"] - r["chain_strict"]["value"]
                                      for r in R if r["rho"] == 0.9))
    save("r1_r6_audit.json", out)
    LOG["r6"] = out


if __name__ == "__main__":
    t0 = time.time()
    r0_validation(); print("r0", time.time() - t0, flush=True)
    r5_theorem3(); print("r5", time.time() - t0, flush=True)
    r6_audit()
    r4_distance_corr(); print("r4", time.time() - t0, flush=True)
    with Pool(4) as pool:
        games, _ = r1_calibrated(pool); print("r1", time.time() - t0, flush=True)
        r3_robustness(pool, games); print("r3", time.time() - t0, flush=True)
        r7_rho_s(pool); print("r7", time.time() - t0, flush=True)
        r2_sensitivity(pool); print("r2", time.time() - t0, flush=True)
    LOG["runtime_s"] = time.time() - t0
    save("r1_summary.json", LOG)
    print("done", LOG["runtime_s"])
