"""Q14 公平性与价值对齐.

(a) 公平性不可能定理的实证（Kleinberg et al. 2016; Chouldechova 2017）与 Hardt et al. 2016
    均等化几率后处理的代价边界。两组人群，组 A 基率固定 0.3，组 B 基率 0.3..0.6；
    个体真实风险 p ~ Beta（均值=基率，集中度 KAPPA），分数 s=p（组内完美校准），y~Bernoulli(p)。
(b) 最小化的对齐实验：Bradley–Terry 代理奖励（有限、含噪偏好数据）+ best-of-n 优化压力，
    度量真实奖励与代理奖励的背离（Gao et al. 2022 式过优化曲线）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit, linprog
from sklearn.linear_model import LogisticRegression

from common.util import mean_ci, plt, results_dir, rng, save_results

T0 = time.time()
SEEDS = list(range(5))

# ================================ (a) fairness ================================
BASE_A, BASE_B = 0.3, [0.3, 0.35, 0.4, 0.5, 0.6]
KAPPA, N_GROUP = 4.0, 20000
TGRID = np.linspace(0, 1.0001, 101)
GAMMAS = [0.0, 0.02, 0.05, 0.1, 0.2, 1.0]


def population(g, base):
    p = g.beta(base * KAPPA, (1 - base) * KAPPA, N_GROUP)
    y = (g.random(N_GROUP) < p).astype(int)
    return p, y


def ece(s, y, bins=10):
    idx = np.minimum((s * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return e


def rates(s, y, t):
    pred = s >= t
    tpr = (pred & (y == 1)).sum() / max((y == 1).sum(), 1)
    fpr = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
    ppv = (pred & (y == 1)).sum() / max(pred.sum(), 1)
    return tpr, fpr, ppv


def eo_postprocess(sA, yA, sB, yB, gamma):
    """Hardt et al. 2016: LP over randomized mixtures of group-specific thresholds.
    maximise overall accuracy s.t. |TPR_A-TPR_B|<=gamma and |FPR_A-FPR_B|<=gamma."""
    roc = {}
    for gname, s, y in [("A", sA, yA), ("B", sB, yB)]:
        roc[gname] = np.array([rates(s, y, t)[:2] for t in TGRID])  # (tpr, fpr)
    nA, nB = len(sA), len(sB)
    piA, piB = nA / (nA + nB), nB / (nA + nB)
    bA, bB = yA.mean(), yB.mean()
    T = len(TGRID)
    # accuracy_g = b*TPR + (1-b)(1-FPR)
    cA = piA * (bA * roc["A"][:, 0] - (1 - bA) * roc["A"][:, 1])
    cB = piB * (bB * roc["B"][:, 0] - (1 - bB) * roc["B"][:, 1])
    c = -np.r_[cA, cB]
    dT = np.r_[roc["A"][:, 0], -roc["B"][:, 0]]
    dF = np.r_[roc["A"][:, 1], -roc["B"][:, 1]]
    A_ub = np.vstack([dT, -dT, dF, -dF])
    b_ub = np.full(4, gamma)
    A_eq = np.vstack([np.r_[np.ones(T), np.zeros(T)], np.r_[np.zeros(T), np.ones(T)]])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1, 1], bounds=(0, 1), method="highs")
    lam = res.x
    tA, fA = lam[:T] @ roc["A"][:, 0], lam[:T] @ roc["A"][:, 1]
    tB, fB = lam[T:] @ roc["B"][:, 0], lam[T:] @ roc["B"][:, 1]
    acc = piA * (bA * tA + (1 - bA) * (1 - fA)) + piB * (bB * tB + (1 - bB) * (1 - fB))
    ppvA = bA * tA / max(bA * tA + (1 - bA) * fA, 1e-12)
    ppvB = bB * tB / max(bB * tB + (1 - bB) * fB, 1e-12)
    return dict(acc=acc, tpr_gap=abs(tA - tB), fpr_gap=abs(fA - fB), ppv_gap=abs(ppvA - ppvB))


fair = []
score_frontier = []
for seed in SEEDS:
    g = rng(seed)
    for bB in BASE_B:
        sA, yA = population(g, BASE_A)
        sB, yB = population(g, bB)
        row = dict(seed=seed, baseB=bB, gap=bB - BASE_A)
        # score-level criteria (Kleinberg): calibration within groups + balance for pos/neg class
        row["ece_A"], row["ece_B"] = ece(sA, yA), ece(sB, yB)
        row["bal_pos_gap"] = abs(sA[yA == 1].mean() - sB[yB == 1].mean())
        row["bal_neg_gap"] = abs(sA[yA == 0].mean() - sB[yB == 0].mean())
        # single threshold 0.5 classifier (Chouldechova)
        tA_, fA_, pA_ = rates(sA, yA, 0.5)
        tB_, fB_, pB_ = rates(sB, yB, 0.5)
        row.update(fpr_gap_t05=abs(fA_ - fB_), fnr_gap_t05=abs(tA_ - tB_), ppv_gap_t05=abs(pA_ - pB_))
        # Chouldechova identity check: FPR = p/(1-p) * (1-PPV)/PPV * (1-FNR)
        pB_hat = yB.mean()
        row["chould_identity_abs_err"] = abs(fB_ - pB_hat / (1 - pB_hat) * (1 - pB_) / pB_ * tB_)
        acc_unc = (np.r_[sA >= 0.5, sB >= 0.5] == np.r_[yA, yB]).mean()
        row["acc_unconstrained"] = acc_unc
        for gm in GAMMAS:
            r = eo_postprocess(sA, yA, sB, yB, gm)
            for k, v in r.items():
                row[f"eo{gm}_{k}"] = v
        fair.append(row)
        if bB == 0.5:
            # score transformation frontier: shift group-B logits by c, s' = sigmoid(logit(s)+c)
            for cshift in np.linspace(-2.0, 0.5, 26):
                lB = np.log(np.clip(sB, 1e-6, 1 - 1e-6) / np.clip(1 - sB, 1e-6, 1))
                sB2 = 1 / (1 + np.exp(-(lB + cshift)))
                score_frontier.append(dict(seed=seed, c=float(cshift), ece_B=ece(sB2, yB),
                                           bal_pos_gap=abs(sA[yA == 1].mean() - sB2[yB == 1].mean()),
                                           bal_neg_gap=abs(sA[yA == 0].mean() - sB2[yB == 0].mean())))
print(f"(a) done {time.time()-T0:.0f}s", flush=True)

# ================================ (b) Goodhart ================================
DIM, CURV, BETA_BT = 10, 0.5, 1.0
M_PREFS = [100, 300, 1000, 3000, 10000]
N_MAX, TRIALS = 4096, 300
NS = 2 ** np.arange(13)


def true_reward(X, w):
    return X @ w - 0.5 * CURV * (X**2).sum(-1)


def feats(X, kind):
    return X if kind == "linear" else np.concatenate([X, X**2], -1)


def fit_bt(g, w, m, kind):
    A = g.standard_normal((m, DIM))
    B = g.standard_normal((m, DIM))
    pr = 1 / (1 + np.exp(-BETA_BT * (true_reward(A, w) - true_reward(B, w))))
    lab = (g.random(m) < pr).astype(int)
    D = feats(A, kind) - feats(B, kind)
    D = np.vstack([D, -D])  # symmetrise so no intercept is needed
    lab = np.r_[lab, 1 - lab]
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(D, lab)
    return clf.coef_[0]


bon = []
for seed in SEEDS:
    g = rng(1000 + seed)
    w = g.standard_normal(DIM)
    w /= np.linalg.norm(w)
    pool = g.standard_normal((TRIALS, N_MAX, DIM))   # base-policy samples
    R_true = true_reward(pool, w)
    base_mean = R_true.mean()
    # oracle best-of-n (selection by true reward)
    for n in NS:
        bon.append(dict(seed=seed, kind="oracle", m=0, n=int(n),
                        true=float(R_true[:, :n].max(1).mean() - base_mean), proxy=np.nan))
    for kind in ["linear", "quadratic"]:
        for m in M_PREFS:
            theta = fit_bt(g, w, m, kind)
            R_prox = feats(pool, kind) @ theta / BETA_BT
            corr = float(np.corrcoef(R_prox.ravel()[:200000], R_true.ravel()[:200000])[0, 1])
            pbase = R_prox.mean()
            for n in NS:
                sel = R_prox[:, :n].argmax(1)
                t = R_true[np.arange(TRIALS), sel].mean() - base_mean
                p = R_prox[np.arange(TRIALS), sel].mean() - pbase
                bon.append(dict(seed=seed, kind=kind, m=m, n=int(n), true=float(t), proxy=float(p), corr=corr))
print(f"(b) done {time.time()-T0:.0f}s", flush=True)


def kl_bon(n):
    return np.log(n) - (n - 1) / n


def gao(dd, a, b):
    return dd * (a - b * dd)


# ================================ aggregation =================================
def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


out = {"config": dict(seeds=SEEDS, fairness=dict(base_A=BASE_A, base_B=BASE_B, kappa=KAPPA, n_per_group=N_GROUP,
                                                  gammas=GAMMAS, threshold_grid=len(TGRID)),
                      goodhart=dict(dim=DIM, curvature=CURV, bt_beta=BETA_BT, m_prefs=M_PREFS, n_max=N_MAX,
                                    trials=TRIALS, true_reward="w·x - 0.5*c*||x||^2, ||w||=1"))}
keys = ["ece_A", "ece_B", "bal_pos_gap", "bal_neg_gap", "fpr_gap_t05", "fnr_gap_t05", "ppv_gap_t05",
        "chould_identity_abs_err", "acc_unconstrained"] + [f"eo{gm}_{k}" for gm in GAMMAS for k in ["acc", "tpr_gap", "fpr_gap", "ppv_gap"]]
out["fairness"] = {f"baseB={b}": {k: ag(fair, k, baseB=b) for k in keys} for b in BASE_B}
for b in BASE_B:
    x = [r["eo1.0_acc"] - r["eo0.0_acc"] for r in fair if r["baseB"] == b]  # 无约束最优（γ=1）减 均等化几率
    out["fairness"][f"baseB={b}"]["eo_cost_acc"] = dict(zip(["mean", "ci95"], mean_ci(x)))
cs = sorted(set(r["c"] for r in score_frontier))
out["score_frontier_baseB=0.5"] = [{"c": c, **{k: ag(score_frontier, k, c=c)["mean"] for k in ["ece_B", "bal_pos_gap", "bal_neg_gap"]}} for c in cs]
# Goodhart
out["goodhart"] = {}
peaks = {}
for kind in ["linear", "quadratic"]:
    for m in M_PREFS:
        e = {}
        for n in NS:
            e[f"n={n}"] = {"true": ag(bon, "true", kind=kind, m=m, n=int(n)), "proxy": ag(bon, "proxy", kind=kind, m=m, n=int(n))}
        e["corr_proxy_true_base"] = ag(bon, "corr", kind=kind, m=m, n=1)
        # per-seed peak n and drop from peak to n_max
        pk, drop, ab = [], [], []
        for s in SEEDS:
            tv = np.array([[r["true"] for r in bon if r["seed"] == s and r["kind"] == kind and r["m"] == m and r["n"] == n][0] for n in NS])
            pk.append(np.log2(NS[int(np.argmax(tv))]))
            drop.append(tv.max() - tv[-1])
            dd = np.sqrt(kl_bon(NS.astype(float)))
            try:
                (a_, b_), _ = curve_fit(gao, dd, tv, p0=[1.0, 0.1])
            except RuntimeError:
                a_, b_ = np.nan, np.nan
            ab.append((a_, b_))
        e["log2_peak_n"] = dict(zip(["mean", "ci95"], mean_ci(pk)))
        e["peak_minus_final_true"] = dict(zip(["mean", "ci95"], mean_ci(drop)))
        e["gao_alpha"] = dict(zip(["mean", "ci95"], mean_ci([x[0] for x in ab])))
        e["gao_beta"] = dict(zip(["mean", "ci95"], mean_ci([x[1] for x in ab])))
        e["drop_one_sample_t_p"] = float(stats.ttest_1samp(drop, 0).pvalue) if np.std(drop) > 0 else 1.0
        peaks[(kind, m)] = pk
        out["goodhart"][f"{kind}|m={m}"] = e
out["goodhart"]["oracle"] = {f"n={n}": ag(bon, "true", kind="oracle", n=int(n)) for n in NS}
out["goodhart"]["kl_bon"] = {f"n={n}": float(kl_bon(n)) for n in NS}
# gap to oracle at n_max vs m
gaps = {}
for kind in ["linear", "quadratic"]:
    for m in M_PREFS:
        x = [o["true"] - r["true"] for o in bon if o["kind"] == "oracle" and o["n"] == N_MAX
             for r in bon if r["kind"] == kind and r["m"] == m and r["n"] == N_MAX and r["seed"] == o["seed"]]
        gaps[f"{kind}|m={m}"] = dict(zip(["mean", "ci95"], mean_ci(x)))
out["goodhart"]["oracle_gap_at_nmax"] = gaps
out["tests"] = {
    "quadratic_peak_log2n_m10000_vs_m100": {
        "diff": float(np.mean(np.array(peaks[("quadratic", 10000)]) - peaks[("quadratic", 100)])),
        "paired_t_p": float(stats.ttest_rel(peaks[("quadratic", 10000)], peaks[("quadratic", 100)]).pvalue)
        if np.std(np.array(peaks[("quadratic", 10000)]) - peaks[("quadratic", 100)]) > 0 else 1.0},
}
out["runtime_sec"] = time.time() - T0

# ================================== figures ===================================
P = plt()
R = results_dir(__file__)
fig, axes = P.subplots(1, 3, figsize=(14, 4))
gaps_x = [b - BASE_A for b in BASE_B]
for k, lab in [("bal_pos_gap", "正类平衡差 |E[s|y=1]|"), ("bal_neg_gap", "负类平衡差 |E[s|y=0]|"),
               ("fpr_gap_t05", "FPR 差 (t=0.5)"), ("fnr_gap_t05", "FNR 差 (t=0.5)"), ("ece_B", "组 B 校准误差 ECE")]:
    axes[0].errorbar(gaps_x, [out["fairness"][f"baseB={b}"][k]["mean"] for b in BASE_B],
                     yerr=[out["fairness"][f"baseB={b}"][k]["ci95"] for b in BASE_B], marker="o", capsize=3, label=lab)
axes[0].set_xlabel("基率差 p_B - p_A")
axes[0].set_title("校准分数下的公平性差距")
axes[0].legend(fontsize=7)
for b in BASE_B:
    xs = [out["fairness"][f"baseB={b}"][f"eo{gm}_ppv_gap"]["mean"] for gm in GAMMAS]
    ys = [out["fairness"][f"baseB={b}"][f"eo{gm}_acc"]["mean"] for gm in GAMMAS]
    axes[1].plot(GAMMAS[:-1], ys[:-1], "o-", label=f"p_B={b}")
axes[1].set_xlabel("允许的 TPR/FPR 差 γ")
axes[1].set_ylabel("后处理后总准确率")
axes[1].set_title("均等化几率后处理：准确率-公平性边界")
axes[1].legend(fontsize=7)
sf = out["score_frontier_baseB=0.5"]
axes[2].plot([x["bal_pos_gap"] + x["bal_neg_gap"] for x in sf], [x["ece_B"] for x in sf], "o-", ms=3)
axes[2].set_xlabel("正类+负类平衡差（组 B 分数平移后）")
axes[2].set_ylabel("组 B 校准误差 ECE")
axes[2].set_title("分数层面：平衡 vs 校准（p_B=0.5）")
fig.tight_layout()
fig.savefig(R / "fairness.png")

fig, axes = P.subplots(1, 2, figsize=(12, 4.2), sharey=True)
dd = np.sqrt(kl_bon(NS.astype(float)))
for ax, kind in zip(axes, ["linear", "quadratic"]):
    ax.plot(dd, [out["goodhart"]["oracle"][f"n={n}"]["mean"] for n in NS], "k-", lw=2, label="按真实奖励选择（上限）")
    for i, m in enumerate(M_PREFS):
        e = out["goodhart"][f"{kind}|m={m}"]
        ax.plot(dd, [e[f"n={n}"]["true"]["mean"] for n in NS], "-o", ms=3, color=f"C{i}", label=f"真实奖励 m={m}")
        ax.plot(dd, [e[f"n={n}"]["proxy"]["mean"] for n in NS], ":", color=f"C{i}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("√KL(best-of-n ‖ 基础策略)")
    ax.set_title(f"{'线性（错设）' if kind == 'linear' else '二次（正确设定）'}代理奖励；点线=代理奖励")
axes[0].set_ylabel("相对基础策略的奖励增益")
axes[0].legend(fontsize=7)
axes[0].set_ylim(-3, 6)
fig.tight_layout()
fig.savefig(R / "goodhart.png")

save_results(__file__, out)
print(f"done {time.time()-T0:.0f}s")
