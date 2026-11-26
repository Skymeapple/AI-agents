"""Q10 前沿实验：物理先验错设下“物理 / 数据 / 混合”三者的偏差-方差理论与交叉样本量。

设定（线性-高斯可解模型）：t~U[0,10]，正交基（在均匀测度下）由
  物理特征 φ1=e^{-0.15t}cos2t, φ2=e^{-0.15t}sin2t（p=2）与 q=20 个三角（傅里叶）残差函数经 Gram-Schmidt 得到，
  残差基 v_k 与物理子空间正交。真函数 f = 物理部分 + δ，δ=√B·Σ c_k v_k（c_k∝1/k，‖c‖=1），即 ‖δ‖²=B（错设强度）。
模型：物理 OLS（p 个参数）；数据 OLS（p+q 个参数，含真函数）；混合 = 物理不惩罚 + 残差基岭回归，
  λ 取理论最优（oracle）或 GCV。理论预测：
  R_phys = B + σ²p/(n-p-1)；R_data = σ²(p+q)/(n-p-q-1)；
  R_hyb(λ) ≈ σ²p/n + Σ_k[(λ/(n+λ))²δ_k² + nσ²/(n+λ)²]，交叉样本量 n*≈qσ²/B。
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, mean_ci, plt, results_dir

import json
import numpy as np
from scipy import stats, optimize

T0 = time.time()
OUT = results_dir(__file__)
P, Q, SIG = 2, 20, 0.1
BS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
NS = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048]
SEEDS = 40
LAMS = np.logspace(-3, 7, 81)

# ---------- orthonormal basis under U[0,10] ----------
G = np.linspace(0, 10, 4001)


def raw(t):
    phys = np.c_[np.exp(-0.15 * t) * np.cos(2 * t), np.exp(-0.15 * t) * np.sin(2 * t)]
    k = np.arange(1, Q // 2 + 1)
    trig = [np.ones_like(t)[:, None], np.cos(2 * np.pi * np.outer(t, k) / 10.0), np.sin(2 * np.pi * np.outer(t, k[:-1]) / 10.0)]
    return np.hstack([phys] + trig)          # 1 + 10 cos + 9 sin = Q=20 residual functions


Rg = raw(G) / np.sqrt(len(G))
_, Rmat = np.linalg.qr(Rg)
Rinv = np.linalg.inv(Rmat)


def basis(t):  # columns: u1,u2 (physics span), v1..vq (orthogonal residual)
    return raw(t) @ Rinv


chk = basis(G); gram = chk.T @ chk / len(G)
ortho_err = float(np.abs(gram - np.eye(P + Q)).max())
c = 1.0 / np.arange(1, Q + 1); c /= np.linalg.norm(c)
theta_phys = np.array([1.0, 0.3])          # coefficients on u1,u2


def truth_coef(B):
    return np.r_[theta_phys, np.sqrt(B) * c]


# ---------- theory ----------
def th_phys(n, B): return B + SIG ** 2 * P / (n - P - 1) if n > P + 1 else np.nan
def th_data(n): return SIG ** 2 * (P + Q) / (n - P - Q - 1) if n > P + Q + 1 else np.nan


def th_hyb(n, B, lam):
    d2 = B * c ** 2
    return SIG ** 2 * P / n + np.sum((lam / (n + lam)) ** 2 * d2 + n * SIG ** 2 / (n + lam) ** 2)


def th_hyb_opt(n, B):
    if B == 0:
        return SIG ** 2 * P / n
    r = optimize.minimize_scalar(lambda ll: th_hyb(n, B, np.exp(ll)), bounds=(-10, 25), method="bounded")
    return float(r.fun)


def lam_oracle(B):
    return Q * SIG ** 2 / B if B > 0 else 1e12


# ---------- estimators ----------
def fit_hybrid(Phi, y, lams):
    """physics columns unpenalized, residual columns ridge; returns coef for each lam and GCV scores."""
    U, V = Phi[:, :P], Phi[:, P:]
    Qu, _ = np.linalg.qr(U)
    Vp = V - Qu @ (Qu.T @ V); yp = y - Qu @ (Qu.T @ y)       # profile out physics
    Us, s, Wt = np.linalg.svd(Vp, full_matrices=False)
    uy = Us.T @ yp
    n = len(y); out = []
    for lam in lams:
        f = s ** 2 / (s ** 2 + lam)
        beta = Wt.T @ (s / (s ** 2 + lam) * uy)
        resid = yp - Vp @ beta
        df = P + f.sum()
        gcv = n * (resid @ resid) / max(n - df, 1e-9) ** 2
        theta = np.linalg.lstsq(U, y - V @ beta, rcond=None)[0]
        r2 = uy ** 2
        s2hat = (np.sum(r2 / (1 + s ** 2 / lam)) + yp @ yp - r2.sum()) / (n - P)
        nll = 0.5 * (np.sum(np.log1p(s ** 2 / lam)) + (n - P) * np.log(s2hat))   # restricted marginal likelihood
        out.append((np.r_[theta, beta], gcv, nll))
    return out


raw_res = {B: {m: {n: [] for n in NS} for m in ["物理", "数据", "混合-oracle λ", "混合-GCV", "混合-EB"]} for B in BS}
lam_gcv_rec = {B: {n: [] for n in NS} for B in BS}
for B in BS:
    ct = truth_coef(B)
    for n in NS:
        for s in range(SEEDS):
            g = rng(10000 * BS.index(B) + 100 * NS.index(n) + s)
            t = g.uniform(0, 10, n); Phi = basis(t)
            y = Phi @ ct + SIG * g.standard_normal(n)
            th = np.linalg.lstsq(Phi[:, :P], y, rcond=None)[0]
            raw_res[B]["物理"][n].append(np.sum((np.r_[th, np.zeros(Q)] - ct) ** 2))
            if n > P + Q + 1:
                cd = np.linalg.lstsq(Phi, y, rcond=None)[0]
                raw_res[B]["数据"][n].append(np.sum((cd - ct) ** 2))
            co = fit_hybrid(Phi, y, [lam_oracle(B)])[0][0]
            raw_res[B]["混合-oracle λ"][n].append(np.sum((co - ct) ** 2))
            fits = fit_hybrid(Phi, y, LAMS)
            k = int(np.argmin([f[1] for f in fits]))
            raw_res[B]["混合-GCV"][n].append(np.sum((fits[k][0] - ct) ** 2))
            lam_gcv_rec[B][n].append(float(LAMS[k]))
            k2 = int(np.argmin([f[2] for f in fits]))
            raw_res[B]["混合-EB"][n].append(np.sum((fits[k2][0] - ct) ** 2))
    print("B", B, round(time.time() - T0, 1), flush=True)

# ---------- summarise ----------
summary = {}
for B in BS:
    summary[B] = {}
    for m in raw_res[B]:
        summary[B][m] = {}
        for n in NS:
            v = raw_res[B][m][n]
            if not v:
                continue
            mu, ci = mean_ci(v)
            summary[B][m][n] = {"mean": mu, "ci95": ci, "median": float(np.median(v))}
    summary[B]["theory"] = {n: {"物理": th_phys(n, B), "数据": th_data(n), "混合-oracle λ": th_hyb(n, B, lam_oracle(B)),
                                "混合-最优λ": th_hyb_opt(n, B)} for n in NS}
    summary[B]["gcv_lambda_median"] = {n: float(np.median(lam_gcv_rec[B][n])) for n in NS}

# theory-vs-measurement deviation (n >= 2(p+q) for data; n>=16 for physics/hybrid)
dev = {}
for m, nmin in [("物理", 16), ("数据", 2 * (P + Q)), ("混合-oracle λ", 16)]:
    rel = []
    for B in BS:
        for n in NS:
            if n >= nmin and n in summary[B][m]:
                th = summary[B]["theory"][n][m]
                rel.append(abs(summary[B][m][n]["mean"] - th) / th)
    dev[m] = {"median_rel_dev": float(np.median(rel)), "max_rel_dev": float(np.max(rel)), "count": len(rel)}


# crossover: physics vs data
def crossover(xs, a, b):
    """first n where b < a, log-interpolated; a,b arrays"""
    d = np.log(np.array(b)) - np.log(np.array(a))
    for i in range(1, len(xs)):
        if d[i - 1] > 0 and d[i] <= 0:
            x0, x1 = np.log(xs[i - 1]), np.log(xs[i])
            return float(np.exp(x0 + (x1 - x0) * d[i - 1] / (d[i - 1] - d[i])))
    return None


cross = {}
nfine = np.unique(np.round(np.logspace(np.log10(24), np.log10(20000), 400)).astype(int))
for B in BS[1:]:
    ns_d = [n for n in NS if n > P + Q + 1]
    emp = crossover(ns_d, [summary[B]["物理"][n]["mean"] for n in ns_d], [summary[B]["数据"][n]["mean"] for n in ns_d])
    thc = crossover(list(nfine), [th_phys(n, B) for n in nfine], [th_data(n) for n in nfine])
    emp_med = crossover(ns_d, [summary[B]["物理"][n]["median"] for n in ns_d], [summary[B]["数据"][n]["median"] for n in ns_d])
    cross[B] = {"empirical": emp, "empirical_median_based": emp_med, "theory_exact": thc, "theory_approx_q_sigma2_over_B": Q * SIG ** 2 / B}
    if emp and thc:
        cross[B]["ratio_emp_over_theory"] = emp / thc

# hybrid gain at (nearest) crossover & dominance check
gain = {}
for B in BS[1:]:
    rows = []
    for n in NS:
        if n in summary[B]["数据"]:
            mn = min(summary[B]["物理"][n]["mean"], summary[B]["数据"][n]["mean"])
            rows.append((n, summary[B]["混合-oracle λ"][n]["mean"] / mn, summary[B]["混合-GCV"][n]["mean"] / mn,
                         summary[B]["混合-EB"][n]["mean"] / mn))
    gain[B] = {"per_n_ratio_hybrid_oracle_over_best_single": {r[0]: r[1] for r in rows},
               "per_n_ratio_hybrid_gcv_over_best_single": {r[0]: r[2] for r in rows},
               "max_ratio_oracle": max(r[1] for r in rows), "min_ratio_oracle": min(r[1] for r in rows),
               "max_ratio_gcv": max(r[2] for r in rows), "min_ratio_gcv": min(r[2] for r in rows),
               "per_n_ratio_hybrid_eb_over_best_single": {r[0]: r[3] for r in rows},
               "max_ratio_eb": max(r[3] for r in rows), "min_ratio_eb": min(r[3] for r in rows)}
# B=0: hybrid vs physics
b0 = {m: {n: summary[0.0][m][n]["mean"] / summary[0.0]["物理"][n]["mean"] for n in NS} for m in ["混合-GCV", "混合-EB"]}
eb_vs_oracle = {B: {n: summary[B]["混合-EB"][n]["mean"] / summary[B]["混合-oracle λ"][n]["mean"] for n in NS if n >= 2 * Q} for B in BS[1:]}
gcv_vs_oracle = {B: {n: summary[B]["混合-GCV"][n]["mean"] / summary[B]["混合-oracle λ"][n]["mean"] for n in NS if n >= 2 * Q} for B in BS[1:]}
# paired test at predicted crossover-nearest n for B=1e-3: hybrid GCV vs best single
pt = {}
for B in BS[1:]:
    ns_d = [n for n in NS if n > P + Q + 1]
    n0 = min(ns_d, key=lambda n: abs(np.log(n) - np.log(cross[B]["theory_exact"] or 1e9)))
    h = np.array(raw_res[B]["混合-EB"][n0]); ph = np.array(raw_res[B]["物理"][n0]); da = np.array(raw_res[B]["数据"][n0])
    best = ph if ph.mean() < da.mean() else da
    pt[B] = {"n": n0, "hyb_eb_mean": h.mean(), "best_single_mean": best.mean(),
             "p_wilcoxon": float(stats.wilcoxon(h, best).pvalue)}

# ---------- figures ----------
Pl = plt()
fig, axes = Pl.subplots(1, 4, figsize=(21, 4.6), sharey=False)
cols = {"物理": "C1", "数据": "C0", "混合-oracle λ": "C2", "混合-GCV": "C3", "混合-EB": "C4"}
for ax, B in zip(axes, BS[1:]):
    for m in ["物理", "数据", "混合-oracle λ", "混合-EB"]:
        ns = [n for n in NS if n in summary[B][m]]
        md = [summary[B][m][n]["median"] for n in ns]
        ax.plot(ns, md, "o", ms=3.5, color=cols[m], label=f"{m}（实测中位数）")
    ax.plot(NS, [th_phys(n, B) for n in NS], "-", color="C1", lw=1, label="物理（理论）")
    ax.plot([n for n in NS if n > 23], [th_data(n) for n in NS if n > 23], "-", color="C0", lw=1, label="数据（理论）")
    ax.plot(NS, [th_hyb_opt(n, B) for n in NS], "--", color="C2", lw=1, label="混合最优λ（理论）")
    if cross[B]["theory_exact"]:
        ax.axvline(cross[B]["theory_exact"], color="grey", ls=":", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("样本数 n"); ax.set_ylabel("L2 风险")
    lo = min(summary[B]["混合-EB"][n]["median"] for n in NS); hi = max(summary[B]["物理"][n]["median"] for n in NS)
    ax.set_ylim(lo * 0.3, hi * 30)
    ax.set_title(f"错设强度 B={B:g}（灰线=理论交叉点；数据模型小 n 时重尾，图示中位数）", fontsize=8.5); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "frontier_risk_curves.png"); Pl.close(fig)

fig, ax = Pl.subplots(figsize=(6, 4.4))
for B in BS[1:]:
    ns = [n for n in NS if n in summary[B]["数据"]]
    ax.plot(ns, [gain[B]["per_n_ratio_hybrid_eb_over_best_single"][n] for n in ns], "o-", ms=3, label=f"EB, B={B:g}")
    ax.plot(ns, [gain[B]["per_n_ratio_hybrid_oracle_over_best_single"][n] for n in ns], ":", lw=1, color=ax.lines[-1].get_color())
ax.axhline(1, color="k", lw=.8); ax.axhline(0.5, color="grey", ls=":", lw=.8)
ax.set_xscale("log"); ax.set_xlabel("样本数 n"); ax.set_ylabel("混合风险 / min(物理, 数据)"); ax.set_title("混合（EB 实线 / oracle 点线）相对最佳单一模型")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "frontier_hybrid_gain.png"); Pl.close(fig)

runtime = time.time() - T0
res = {"config": {"p": P, "q": Q, "sigma": SIG, "B": BS, "n": NS, "seeds": SEEDS, "lambda_grid": [float(LAMS[0]), float(LAMS[-1]), len(LAMS)],
                  "delta_profile": "c_k ∝ 1/k normalized", "theta_phys": theta_phys, "basis_orthonormality_max_err": ortho_err},
       "summary": summary, "theory_vs_measured": dev, "crossover": cross, "hybrid_gain": gain,
       "B0_ratio_hybrid_gcv_over_physics": b0, "gcv_over_oracle": gcv_vs_oracle, "eb_over_oracle": eb_vs_oracle, "paired_test_near_crossover": pt,
       "runtime_sec": runtime}
(OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
print("runtime", runtime)
