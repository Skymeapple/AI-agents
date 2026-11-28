"""Q08 前沿实验：多环境线性因果表示学习——需要多少个环境、什么样的干预才能识别潜变量？

模型：x = A z（A 随机可逆，d×d），潜变量 z 在每个环境 e 中独立高斯，方差 v_{e,i}；
      环境 0 为观测环境（v=1），其余为干预环境。单一环境内高斯潜变量只能识别到旋转。
估计：用环境 0 白化，再对其余环境的白化协方差做 Jacobi 联合对角化（Cardoso & Souloumiac 1996）。
理论（见 paper.md §4）：潜变量 i 可识别 ⇔ 其“方差比轮廓” (v_{e,i}/v_{0,i})_{e≥1} 与其他潜变量都不相同。
  设计 S（单节点干预，每个环境干预一个不同潜变量）：可识别数 = k（k≤d-2），k≥d-1 时为 d；
  设计 M（通用多节点干预，所有潜变量方差随机变化）：K=1 个干预环境即全部可识别；
  设计 B（稀疏二值多节点干预，每个潜变量以 1/2 概率被同幅度干预）：E[可识别数]=d(1-2^{-K})^{d-1}。
  有限样本：两环境时 E[1-|corr_i|] ≈ (1/2n) Σ_{j≠i} r_i(r_i+r_j)/(r_i-r_j)²。
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

T0 = time.time()
OUT = results_dir(__file__)
THR = 0.99          # pre-registered threshold for "identified"
AMP = 4.0           # variance factor of an intervention


def jacobi_jd(Ms, tol=1e-12, max_sweeps=100):
    Ms = [M.copy() for M in Ms]; d = Ms[0].shape[0]; V = np.eye(d)
    for _ in range(max_sweeps):
        rot = False
        for p in range(d - 1):
            for q in range(p + 1, d):
                g = np.array([[M[p, p] - M[q, q], M[p, q] + M[q, p]] for M in Ms])
                G = g.T @ g
                ton, toff = G[0, 0] - G[1, 1], G[0, 1] + G[1, 0]
                th = 0.5 * np.arctan2(toff, ton + np.hypot(ton, toff))
                c, s = np.cos(th), np.sin(th)
                if abs(s) > tol:
                    rot = True
                    for M in Ms:
                        Mp, Mq = M[:, p].copy(), M[:, q].copy()
                        M[:, p], M[:, q] = c * Mp + s * Mq, -s * Mp + c * Mq
                        Mp, Mq = M[p, :].copy(), M[q, :].copy()
                        M[p, :], M[q, :] = c * Mp + s * Mq, -s * Mp + c * Mq
                    Vp, Vq = V[:, p].copy(), V[:, q].copy()
                    V[:, p], V[:, q] = c * Vp + s * Vq, -s * Vp + c * Vq
        if not rot:
            break
    return V


def inv_sqrt(S):
    w, U = np.linalg.eigh(S)
    return U @ np.diag(w ** -0.5) @ U.T


def estimate_unmixing(covs):
    W = inv_sqrt(covs[0])
    if len(covs) == 1:
        return W
    V = jacobi_jd([W @ C @ W for C in covs[1:]])
    return V.T @ W


def sample_cov(A, v, n, g):
    z = g.standard_normal((n, len(v))) * np.sqrt(v)
    x = z @ A.T
    return x.T @ x / n


def evaluate(Bhat, A, d, g, ntest=4000):
    z = g.standard_normal((ntest, d)); x = z @ A.T; s = x @ Bhat.T
    C = np.abs(np.corrcoef(z.T, s.T)[:d, d:])
    r, cidx = linear_sum_assignment(-C)
    per = np.zeros(d); per[r] = C[r, cidx]
    return per


def design_profiles(kind, d, K, g):
    """returns (K+1) x d variance matrix; row 0 = observational (all ones)."""
    Vv = np.ones((K + 1, d))
    if kind == "S":
        tg = g.permutation(d)
        for e in range(1, K + 1):
            Vv[e, tg[e - 1]] = AMP
    elif kind == "M":
        Vv[1:] = np.exp(g.uniform(-1, 1, (K, d)))
    elif kind == "B":
        Vv[1:] = np.where(g.uniform(size=(K, d)) < 0.5, AMP, 1.0)
    return Vv


def theory_identified(Vv):
    """latent identifiable iff its ratio profile is unique."""
    prof = [tuple(np.round(Vv[1:, i] / Vv[0, i], 9)) for i in range(Vv.shape[1])]
    return sum(prof.count(p) == 1 for p in prof) if Vv.shape[0] > 1 else 0


# ---------------- Exp 1: identified latents vs number of environments ----------------
N1, SEEDS1 = 20000, 10
exp1 = {}
for d in [8, 16]:
    Ks = {"S": list(range(0, d + 1)), "M": [0, 1, 2, 3], "B": list(range(0, 11))}
    exp1[d] = {}
    for kind, klist in Ks.items():
        exp1[d][kind] = {}
        for K in klist:
            if d == 16 and kind == "S" and K not in (0, 4, 8, 12, 14, 15, 16):
                continue
            emp, th, mcc, thf = [], [], [], []
            for s in range(SEEDS1):
                g = rng(1000 * d + 100 * "SMB".index(kind) + 1000000 * K + s)
                A = g.standard_normal((d, d))
                Vv = design_profiles(kind, d, K, g)
                covs = [sample_cov(A, Vv[e], N1, g) for e in range(K + 1)]
                per = evaluate(estimate_unmixing(covs), A, d, g)
                emp.append(int((per > THR).sum())); th.append(theory_identified(Vv)); mcc.append(per.mean())
                if kind == "M" and K == 1:
                    r = Vv[1]
                    pe = [sum(r[i] * (r[i] + r[j]) / (r[i] - r[j]) ** 2 for j in range(d) if j != i) / (2 * N1) for i in range(d)]
                    thf.append(int(np.sum(np.array(pe) < 1 - THR)))
            exp1[d][kind][K] = {"identified_mean": float(np.mean(emp)), "identified_ci95": mean_ci(emp)[1],
                                "theory_mean": float(np.mean(th)), "mcc_mean": float(np.mean(mcc)),
                                "exact_agreement_rate": float(np.mean(np.array(emp) == np.array(th))),
                                "full_identification_rate": float(np.mean(np.array(emp) == d))}
            if thf:
                exp1[d][kind][K]["theory_finite_n_mean"] = float(np.mean(thf))
                exp1[d][kind][K]["finite_n_exact_agreement"] = float(np.mean(np.array(thf) == np.array(emp)))
            if kind == "B" and K >= 1:
                exp1[d][kind][K]["theory_expected_formula"] = d * (1 - 2.0 ** -K) ** (d - 1)
                pf = 1.0
                for i in range(1, d):
                    pf *= max(0.0, 1 - i / 2 ** K)
                exp1[d][kind][K]["theory_full_prob"] = pf
        print("exp1", d, kind, round(time.time() - T0, 1), flush=True)

# ---------------- Exp 2: finite-sample law for two environments ----------------
D2, SEEDS2 = 5, 30
NS2 = [500, 2000, 8000, 32000]
pred_all, meas_all, exp2 = [], [], {}
for n in NS2:
    pr, me = [], []
    for s in range(SEEDS2):
        g = rng(77000 + 100 * NS2.index(n) + s)
        A = g.standard_normal((D2, D2))
        Vv = design_profiles("M", D2, 1, g); r = Vv[1]
        covs = [sample_cov(A, Vv[e], n, g) for e in range(2)]
        per = evaluate(estimate_unmixing(covs), A, D2, g, ntest=200000)
        for i in range(D2):
            p_i = sum(r[i] * (r[i] + r[j]) / (r[i] - r[j]) ** 2 for j in range(D2) if j != i) / (2 * n)
            pr.append(p_i); me.append(max(1 - per[i], 1e-12))
    pr, me = np.array(pr), np.array(me)
    ok = pr < 0.05            # perturbation regime
    exp2[n] = {"mean_measured": float(me[ok].mean()), "mean_predicted": float(pr[ok].mean()),
               "ratio_mean": float(me[ok].mean() / pr[ok].mean()), "frac_in_regime": float(ok.mean()),
               "spearman": float(stats.spearmanr(pr[ok], me[ok]).correlation)}
    pred_all += list(pr[ok]); meas_all += list(me[ok])
pa, ma = np.log(pred_all), np.log(meas_all)
sl = stats.linregress(pa, ma)
exp2["pooled_loglog_fit"] = {"slope": sl.slope, "slope_se": sl.stderr, "intercept": sl.intercept, "r": sl.rvalue,
                             "n_points": len(pa), "median_ratio": float(np.median(np.exp(ma - pa)))}
print("exp2", round(time.time() - T0, 1), flush=True)

# ---------------- figures ----------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(13, 4.6))
for ax, d in zip(axes, [8, 16]):
    for kind, col, lab in [("S", "C0", "单节点干预"), ("M", "C2", "通用多节点"), ("B", "C1", "稀疏二值多节点")]:
        Ks = sorted(exp1[d][kind]); e = exp1[d][kind]
        ax.errorbar(Ks, [e[k]["identified_mean"] for k in Ks], yerr=[e[k]["identified_ci95"] for k in Ks], fmt="o", color=col, capsize=2, label=f"{lab}（实测）")
        ax.plot(Ks, [e[k]["theory_mean"] for k in Ks], "-", color=col, lw=1, label=f"{lab}（理论，按轮廓唯一性）")
        if kind == "B":
            kk = [k for k in Ks if k >= 1]
            ax.plot(kk, [e[k]["theory_expected_formula"] for k in kk], ":", color=col, lw=1, label="稀疏二值：d(1-2^{-K})^{d-1}")
    ax.axhline(d, color="k", lw=.6)
    ax.set_xlabel("干预环境数 K（另有 1 个观测环境）"); ax.set_ylabel(f"可识别潜变量数（|corr|>{THR}）")
    ax.set_title(f"d={d}，每环境 n={N1}"); ax.grid(alpha=.3)
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "frontier_identifiability.png"); P.close(fig)

fig, ax = P.subplots(figsize=(5.6, 4.4))
ax.loglog(pred_all, meas_all, ".", ms=3, alpha=.5)
lim = [min(pred_all), max(pred_all)]; ax.plot(lim, lim, "k--", lw=1, label="y=x（理论）")
ax.set_xlabel("理论预测 (1/2n)Σ r_i(r_i+r_j)/(r_i-r_j)²"); ax.set_ylabel("实测 1-|corr_i|")
ax.set_title(f"两环境有限样本误差：斜率 {sl.slope:.2f}"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "frontier_finite_sample.png"); P.close(fig)

runtime = time.time() - T0
res = {"config": {"threshold": THR, "intervention_factor": AMP, "exp1": {"n_per_env": N1, "seeds": SEEDS1, "d": [8, 16]},
                  "exp2": {"d": D2, "n": NS2, "seeds": SEEDS2, "design": "M, log v ~ U[-1,1]", "regime_filter": "predicted < 0.05"}},
       "exp1": exp1, "exp2": exp2, "runtime_sec": runtime}
(OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print("runtime", runtime)
