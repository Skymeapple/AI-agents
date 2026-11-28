"""Q14 前沿实验：奖励模型集成与悲观/谨慎策略何时能缓解奖励过优化？——相关误差的极限.

Part 1（解析可控）：真实奖励 r ~ N(0,1)；K 个代理 r̂_k = r + e_shared + e_k，
    Var(e_shared)=ρσ_e²，Var(e_k)=(1-ρ)σ_e²；误差分布为高斯（轻尾）或 Student-t(3)（重尾，方差归一）。
    best-of-n 选择，聚合方式：单模型 / 均值 / 最小值 / 均值-标准差（UWO）。
    理论（§4）：高斯情形下 E[r|选中] = Var(r)/sqrt(Var(r)+σ_eff²)·E[max_n Z]，σ_eff² = ρσ_e² + (1-ρ)σ_e²/K，
    对 n 单调递增——轻尾误差下不会出现"先升后降"；重尾的共享误差才导致灾难性 Goodhart（Kwa et al. 2024）。
Part 2（学习得到的 Bradley–Terry 奖励模型，延续第一阶段玩具）：r*(x)=w·x-0.25||x||²，x~N(0,I_10)；
    线性（错设）与二次（正确设定）RM，K=5 个 bootstrap 集成；缓解方法：均值/最小值/UWO 集成、
    软 best-of-n（温度 β）、"谨慎"惩罚（对偏离基础分布的 ||x||² 加罚，近似 KL/密度正则，参照 Yu et al. 2026 的 caution）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

from common.util import SEED, mean_ci, plt, results_dir, rng

T0 = time.time()
SEEDS = list(range(5))
NS = 2 ** np.arange(0, 15)          # 1 .. 16384
TRIALS = 150
K = 5
SIG_E = 1.0
RHOS = [0.0, 0.5, 0.9, 1.0]


# ================================ Part 1 ================================
def errs(g, shape, tail):
    if tail == "gauss":
        return g.standard_normal(shape)
    return g.standard_t(3, shape) / np.sqrt(3.0)          # unit variance


P1 = []
for seed in SEEDS:
    for tail in ["gauss", "t3"]:
        for rho in RHOS:
            g = rng(100 * seed + int(10 * rho) + (0 if tail == "gauss" else 50))
            r = g.standard_normal((TRIALS, NS[-1]))
            es = np.sqrt(rho) * SIG_E * errs(g, (TRIALS, NS[-1]), tail)
            ek = np.sqrt(1 - rho) * SIG_E * errs(g, (K, TRIALS, NS[-1]), tail)
            prox = r[None] + es[None] + ek                    # K x TRIALS x N
            aggs = {"single": prox[0], "mean": prox.mean(0), "min": prox.min(0), "uwo": prox.mean(0) - prox.std(0)}
            for an, A in aggs.items():
                for n in NS:
                    sel = A[:, :n].argmax(1)
                    P1.append(dict(seed=seed, tail=tail, rho=rho, agg=an, n=int(n),
                                   true=float(r[np.arange(TRIALS), sel].mean())))
    print(f"P1 seed {seed} {time.time()-T0:.0f}s", flush=True)

# theory for Gaussian single / mean ensemble
gth = rng(999)
EMAX = {int(n): float(np.mean([gth.standard_normal(int(n)).max() for _ in range(400)])) if n > 1 else 0.0 for n in NS}


def theory_true(n, rho, agg):
    s2 = SIG_E**2 if agg == "single" else rho * SIG_E**2 + (1 - rho) * SIG_E**2 / K
    return 1.0 / np.sqrt(1.0 + s2) * EMAX[int(n)]


# ================================ Part 2 ================================
DIM, CURV = 10, 0.5
M_PREFS = [100, 1000]
NS2 = 2 ** np.arange(0, 13)
TR2 = 200
LAM_CAUTION, BETA_SOFT = 0.1, 0.5


def true_reward(X, w):
    return X @ w - 0.5 * CURV * (X**2).sum(-1)


def feats(X, kind):
    return X if kind == "linear" else np.concatenate([X, X**2], -1)


def fit_bt(A, B, lab, kind):
    D = feats(A, kind) - feats(B, kind)
    clf = LogisticRegression(fit_intercept=False, C=10.0, max_iter=2000).fit(np.vstack([D, -D]), np.r_[lab, 1 - lab])
    return clf.coef_[0]


P2 = []
rho_hat = {}
for seed in SEEDS:
    g = rng(5000 + seed)
    w = g.standard_normal(DIM)
    w /= np.linalg.norm(w)
    pool = g.standard_normal((TR2, NS2[-1], DIM))
    Rt = true_reward(pool, w)
    base = Rt.mean()
    sq = (pool**2).sum(-1)
    for kind in ["linear", "quadratic"]:
        for m in M_PREFS:
            A = g.standard_normal((m, DIM))
            B = g.standard_normal((m, DIM))
            lab = (g.random(m) < 1 / (1 + np.exp(-(true_reward(A, w) - true_reward(B, w))))).astype(int)
            thetas = []
            for k in range(K):
                bi = g.integers(0, m, m)                         # bootstrap 集成（同一份偏好数据）
                thetas.append(fit_bt(A[bi], B[bi], lab[bi], kind))
            Fp = feats(pool, kind)
            Rk = np.stack([Fp @ th for th in thetas])           # K x TR2 x N
            # 成员误差相关性（在基础分布上，先把每个代理线性回归到真实奖励尺度上再取残差）
            resid = []
            for k in range(K):
                x, y = Rk[k].ravel()[:100000], Rt.ravel()[:100000]
                a_, b_ = np.polyfit(y, x, 1)
                resid.append((x - (a_ * y + b_)) / a_)
            C = np.corrcoef(np.array(resid))
            rho_hat[f"{kind}|m={m}|seed={seed}"] = float(C[np.triu_indices(K, 1)].mean())
            aggs = {"single": Rk[0], "mean": Rk.mean(0), "min": Rk.min(0), "uwo": Rk.mean(0) - Rk.std(0),
                    "caution": Rk.mean(0) - LAM_CAUTION * Rk.mean(0).std() * (sq - DIM)}   # 谨慎：按代理尺度惩罚偏离基础分布的 ||x||²
            for an, Agg in aggs.items():
                for n in NS2:
                    sel = Agg[:, :n].argmax(1)
                    P2.append(dict(seed=seed, kind=kind, m=m, method=an, n=int(n), true=float(Rt[np.arange(TR2), sel].mean() - base)))
            # soft best-of-n on the mean ensemble: sample one of n with prob ∝ exp(r̂/β)
            Mn = Rk.mean(0)
            gs = rng(7000 + seed)
            for n in NS2:
                z = Mn[:, :n] / (BETA_SOFT * Mn.std())
                pz = np.exp(z - z.max(1, keepdims=True))
                pz /= pz.sum(1, keepdims=True)
                cum = pz.cumsum(1)
                sel = (cum < gs.random((TR2, 1))).sum(1).clip(0, n - 1)
                P2.append(dict(seed=seed, kind=kind, m=m, method="softbon", n=int(n), true=float(Rt[np.arange(TR2), sel].mean() - base)))
            # oracle
    for n in NS2:
        P2.append(dict(seed=seed, kind="oracle", m=0, method="oracle", n=int(n), true=float(Rt[:, :n].max(1).mean() - base)))
    print(f"P2 seed {seed} {time.time()-T0:.0f}s", flush=True)


# ================================ aggregation ================================
def ag(rows, **flt):
    v = [r["true"] for r in rows if all(r[a] == b for a, b in flt.items())]
    m_, h = mean_ci(v)
    return {"mean": m_, "ci95": h}


out = {"config": dict(seeds=SEEDS, K=K, sigma_e=SIG_E, rhos=RHOS, ns=NS.tolist(), trials=TRIALS, tails=["gauss", "t3"],
                      part2=dict(dim=DIM, curvature=CURV, m_prefs=M_PREFS, ns=NS2.tolist(), trials=TR2,
                                 lam_caution=LAM_CAUTION, beta_soft=BETA_SOFT, ensemble="K=5 bootstrap of the same m pairs"))}
p1 = {}
for tail in ["gauss", "t3"]:
    for rho in RHOS:
        for an in ["single", "mean", "min", "uwo"]:
            e = {f"n={n}": ag(P1, tail=tail, rho=rho, agg=an, n=int(n)) for n in NS}
            # peak & drop per seed
            pk, dr = [], []
            for s in SEEDS:
                tv = np.array([[r["true"] for r in P1 if r["seed"] == s and r["tail"] == tail and r["rho"] == rho and r["agg"] == an and r["n"] == n][0] for n in NS])
                pk.append(np.log2(NS[int(np.argmax(tv))]))
                dr.append(tv.max() - tv[-1])
            e["log2_peak_n"] = dict(zip(["mean", "ci95"], mean_ci(pk)))
            e["drop_peak_to_final"] = dict(zip(["mean", "ci95"], mean_ci(dr)))
            e["drop_t_p"] = float(stats.ttest_1samp(dr, 0).pvalue) if np.std(dr) > 0 else 1.0
            if tail == "gauss" and an in ("single", "mean"):
                e["theory"] = {f"n={n}": float(theory_true(n, rho, an)) for n in NS}
            p1[f"{tail}|rho={rho}|{an}"] = e
out["part1"] = p1
# theory vs measured (gauss, n=1024)
pairs = [(p1[f"gauss|rho={rho}|{an}"]["theory"]["n=1024"], p1[f"gauss|rho={rho}|{an}"]["n=1024"]["mean"]) for rho in RHOS for an in ["single", "mean"]]
out["part1_theory_fit_n1024"] = {"pairs_theory_measured": pairs, "max_rel_error": float(max(abs(m_ - t) / t for t, m_ in pairs))}
p2 = {}
for kind in ["linear", "quadratic"]:
    for m in M_PREFS:
        for an in ["single", "mean", "min", "uwo", "caution", "softbon"]:
            e = {f"n={n}": ag(P2, kind=kind, m=m, method=an, n=int(n)) for n in NS2}
            pk, dr, fin = [], [], []
            for s in SEEDS:
                tv = np.array([[r["true"] for r in P2 if r["seed"] == s and r["kind"] == kind and r["m"] == m and r["method"] == an and r["n"] == n][0] for n in NS2])
                pk.append(np.log2(NS2[int(np.argmax(tv))]))
                dr.append(tv.max() - tv[-1])
                fin.append(tv[-1])
            e["log2_peak_n"] = dict(zip(["mean", "ci95"], mean_ci(pk)))
            e["drop_peak_to_final"] = dict(zip(["mean", "ci95"], mean_ci(dr)))
            e["drop_t_p"] = float(stats.ttest_1samp(dr, 0).pvalue) if np.std(dr) > 0 else 1.0
            e["peak_true"] = dict(zip(["mean", "ci95"], mean_ci([max([[r["true"] for r in P2 if r["seed"] == s and r["kind"] == kind and r["m"] == m and r["method"] == an and r["n"] == n][0] for n in NS2]) for s in SEEDS])))
            e["_final_per_seed"] = fin
            p2[f"{kind}|m={m}|{an}"] = e
        p2[f"{kind}|m={m}|rho_hat_member_errors"] = dict(zip(["mean", "ci95"], mean_ci([rho_hat[f"{kind}|m={m}|seed={s}"] for s in SEEDS])))
p2["oracle"] = {f"n={n}": ag(P2, kind="oracle", n=int(n)) for n in NS2}
T = {}
for kind in ["linear", "quadratic"]:
    for m in M_PREFS:
        base_ = np.array(p2[f"{kind}|m={m}|single"]["_final_per_seed"])
        for an in ["mean", "min", "uwo", "caution", "softbon"]:
            x = np.array(p2[f"{kind}|m={m}|{an}"]["_final_per_seed"])
            T[f"{kind}|m={m}|{an}_minus_single_at_nmax"] = {"diff": float(np.mean(x - base_)), "paired_t_p": float(stats.ttest_rel(x, base_).pvalue)}
out["part2"] = p2
out["tests"] = T
out["runtime_sec"] = time.time() - T0
out["_meta"] = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
R = results_dir(__file__)
(R / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

# ---------------------------------- figures ----------------------------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, tail in zip(axes, ["gauss", "t3"]):
    for i, rho in enumerate(RHOS):
        for an, ls in [("single", ":"), ("mean", "-")]:
            e = p1[f"{tail}|rho={rho}|{an}"]
            ax.plot(np.log2(NS), [e[f"n={n}"]["mean"] for n in NS], ls, color=f"C{i}", label=f"ρ={rho} {'单模型' if an == 'single' else 'K=5 均值集成'}")
            if tail == "gauss" and an == "mean":
                ax.plot(np.log2(NS), [e["theory"][f"n={n}"] for n in NS], "x", color=f"C{i}", ms=4)
    ax.set_xlabel("log2 n（best-of-n）")
    ax.set_title(f"Part 1：{'高斯（轻尾）误差；×=理论' if tail == 'gauss' else 'Student-t(3)（重尾）误差'}")
axes[0].set_ylabel("被选样本的真实奖励")
axes[0].legend(fontsize=6, ncol=2)
fig.tight_layout()
fig.savefig(R / "frontier_part1.png")

fig, axes = P.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, kind in zip(axes, ["linear", "quadratic"]):
    ax.plot(np.log2(NS2), [p2["oracle"][f"n={n}"]["mean"] for n in NS2], "k-", lw=2, label="按真实奖励选择")
    for i, an in enumerate(["single", "mean", "min", "uwo", "caution", "softbon"]):
        e = p2[f"{kind}|m=100|{an}"]
        ax.plot(np.log2(NS2), [e[f"n={n}"]["mean"] for n in NS2], "-o", ms=3, color=f"C{i}", label=an)
    ax.set_xlabel("log2 n")
    ax.set_title(f"Part 2：{'线性（错设）' if kind == 'linear' else '二次（正确设定）'} RM，m=100")
axes[0].set_ylabel("真实奖励增益")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "frontier_part2.png")
print(f"done {time.time()-T0:.0f}s")
