"""Q18 前沿实验：多智能体（LLM）投票与辩论——何时“更多智能体”有用？

问题（2025–2026 争论）：多智能体辩论(MAD)的收益主要来自多数投票 (arXiv:2508.17536)；
辩论是鞅、无法提升期望正确率 (2508.17536, 2603.06801)；同质辩论导致从众与错误共识
(2605.00914, 2509.05396, 2608.02827)；多样性是否是关键 (2601.19921, 2609.35875)。

模型（参数在运行前固定）：
- 题目 i 有共同难度 z_i~N(0,1)，模型家族因子 u_{f,i}~N(0,1)。智能体 j（家族 f）的“正确性得分”
  s_ij = √ρ_g z_i + √ρ_f u_{f,i} + √(1-ρ_g-ρ_f) ε_ij；答对 ⇔ s_ij > Φ^{-1}(1-p_j)。
  4 选 1；答错时以 κ=0.5 选“共同干扰项”，否则在 3 个错误项中均匀选（错误答案相关）。
- E1 投票规模律：单家族，p=0.6，ρ_g∈{0,0.1,0.3,0.5}，n∈{1,3,5,9,15,25,51}，多数（相对多数）投票。
  理论：n→∞ 上限 = P_z[p(z) > (1-p(z))·(κ+(1-κ)/3)]，有闭式；有限 n 用 Gauss–Hermite 数值积分 × 多项分布蒙特卡洛。
- E2 多样性–精度权衡（n=9）：同质强家族（p=0.65, ρ_g=0.1, ρ_f=0.4）vs 3 家族×3（ρ_f=0.4）vs 9 家族；
  多样池个体准确率 p_div 扫描 {0.50,…,0.65}。
- E3 辩论动力学（n=9, p=0.6, ρ_g=0.3）：每轮每个智能体随机选一名同伴，以
  w(同伴)/(w(自己)+w(同伴)) 的概率采纳同伴答案，w(正确)=r、w(错误)=1（r=1 为“香草”辩论，
  r>1 表示正确论证更有说服力=验证能力，r<1 表示错误答案更有说服力=谄媚/幻觉自信），r∈{0.8,1,1.1,…,5}；另设“从众”变体：以 0.5 概率采纳当前多数答案。
  R=0..10 轮。理论：异步极限下正确答案的固定概率（Moran）φ(k)=(1-r^{-k})/(1-r^{-n})，r=1 时 φ=k/n。
- E4 等预算比较：辩论（n=9 智能体 × (R+1) 次调用）vs 独立投票 n·(R+1) 个样本（同家族同相关）。
- 10 个种子（每种子 3000 题）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng  # noqa: E402

from scipy import stats  # noqa: E402

T0 = time.time()
SEEDS = list(range(10))
NI, K, KAPPA = 3000, 4, 0.5
Q_DIS = KAPPA + (1 - KAPPA) / 3  # share of wrong answers on the common distractor


def answers(g, p, fam, rho_g, rho_f, z=None, u=None):
    """p: array of agent accuracies; fam: family index per agent. returns (NI, n) answers"""
    n = len(p)
    z = g.standard_normal(NI) if z is None else z
    nf = int(max(fam)) + 1
    u = g.standard_normal((NI, nf)) if u is None else u
    eps = g.standard_normal((NI, n))
    s = np.sqrt(rho_g) * z[:, None] + np.sqrt(rho_f) * u[:, fam] + np.sqrt(1 - rho_g - rho_f) * eps
    corr = s > stats.norm.ppf(1 - np.asarray(p))[None, :]
    wrong = np.where(g.random((NI, n)) < KAPPA, 1, g.integers(1, K, (NI, n)))
    return np.where(corr, 0, wrong)


def plurality(A, g):
    cnt = np.stack([(A == k).sum(1) for k in range(K)], 1).astype(float)
    cnt += 1e-3 * g.random(cnt.shape)  # random tie-break
    return cnt.argmax(1)


def theory_ceiling(p, rho):
    if rho == 0:
        return float(p > (1 - p) * Q_DIS)
    tau = stats.norm.ppf(1 - p)
    thr = Q_DIS / (1 + Q_DIS)  # p(z) > thr
    return float(1 - stats.norm.cdf((tau + np.sqrt(1 - rho) * stats.norm.ppf(thr)) / np.sqrt(rho)))


def theory_finite(p, rho, n, g, mc=4000):
    xs, ws = np.polynomial.hermite_e.hermegauss(40)
    ws = ws / ws.sum()
    tau = stats.norm.ppf(1 - p)
    acc = 0.0
    for x, w in zip(xs, ws):
        pz = stats.norm.cdf((np.sqrt(rho) * x - tau) / np.sqrt(1 - rho)) if rho > 0 else p
        pv = np.array([pz, (1 - pz) * Q_DIS, (1 - pz) * (1 - Q_DIS) / 2, (1 - pz) * (1 - Q_DIS) / 2])
        C = g.multinomial(n, pv, size=mc).astype(float) + 1e-3 * g.random((mc, K))
        acc += w * (C.argmax(1) == 0).mean()
    return float(acc)


out = {"config": dict(seeds=SEEDS, n_items=NI, K=K, kappa=KAPPA)}

# ---------------------------------------------------------------- E1
NS = [1, 3, 5, 9, 15, 25, 51]
RHOS = [0.0, 0.1, 0.3, 0.5]
e1 = {}
for rho in RHOS:
    sim = {n: [] for n in NS}
    for sd in SEEDS:
        g = rng(100 + sd + int(100 * rho))
        A = answers(g, np.full(max(NS), 0.6), np.zeros(max(NS), int), rho, 0.0)
        for n in NS:
            sim[n].append(float((plurality(A[:, :n], g) == 0).mean()))
    e1[str(rho)] = {"sim": {str(n): mean_ci(sim[n]) for n in NS},
                    "theory_finite": {str(n): theory_finite(0.6, rho, n, rng(7)) for n in NS},
                    "theory_ceiling": theory_ceiling(0.6, rho)}
out["E1_vote_scaling"] = e1
print(f"E1 {time.time()-T0:.1f}s", flush=True)

# ---------------------------------------------------------------- E2
PDIV = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
e2 = {"homog_p0.65": [], "w_homog": []}
for pd in PDIV:
    e2[f"fam3_p{pd}"] = []; e2[f"fam9_p{pd}"] = []
for sd in SEEDS:
    g = rng(300 + sd)
    z = g.standard_normal(NI); u = g.standard_normal((NI, 9))
    A = answers(g, np.full(9, 0.65), np.zeros(9, int), 0.1, 0.4, z, u)
    e2["homog_p0.65"].append(float((plurality(A, g) == 0).mean()))
    for pd in PDIV:
        A3 = answers(g, np.full(9, pd), np.repeat(np.arange(3), 3), 0.1, 0.4, z, u)
        A9 = answers(g, np.full(9, pd), np.arange(9), 0.1, 0.4, z, u)
        e2[f"fam3_p{pd}"].append(float((plurality(A3, g) == 0).mean()))
        e2[f"fam9_p{pd}"].append(float((plurality(A9, g) == 0).mean()))
out["E2_diversity"] = {k: mean_ci(v) for k, v in e2.items() if v}
# design-effect theory: n_eff = n / (1 + (n-1) * rho_bar), rho_bar = mean pairwise latent correlation
def n_eff(n_per_fam, nfam, rho_g, rho_f):
    n = n_per_fam * nfam
    pairs_same = nfam * n_per_fam * (n_per_fam - 1); pairs_all = n * (n - 1)
    rbar = rho_g + rho_f * pairs_same / pairs_all
    return n / (1 + (n - 1) * rbar), rbar
out["E2_theory_neff"] = {"homog": n_eff(9, 1, 0.1, 0.4), "fam3": n_eff(3, 3, 0.1, 0.4), "fam9": n_eff(1, 9, 0.1, 0.4)}
# break-even p_div by linear interpolation of mean curves
hm = np.mean(e2["homog_p0.65"])
for tag in ("fam3", "fam9"):
    ys = [np.mean(e2[f"{tag}_p{pd}"]) for pd in PDIV]
    be = None
    for (x0, y0), (x1, y1) in zip(zip(PDIV, ys), zip(PDIV[1:], ys[1:])):
        if (y0 - hm) * (y1 - hm) <= 0 and y1 != y0:
            be = x0 + (hm - y0) * (x1 - x0) / (y1 - y0); break
    out["E2_diversity"][f"breakeven_pdiv_{tag}"] = be
out["E2_tests"] = {f"fam3_p0.6_vs_homog": {"diff_pp": mean_ci(100 * (np.array(e2["fam3_p0.6"]) - np.array(e2["homog_p0.65"]))),
                                           "p": float(stats.ttest_rel(e2["fam3_p0.6"], e2["homog_p0.65"]).pvalue)},
                   f"fam9_p0.6_vs_homog": {"diff_pp": mean_ci(100 * (np.array(e2["fam9_p0.6"]) - np.array(e2["homog_p0.65"]))),
                                           "p": float(stats.ttest_rel(e2["fam9_p0.6"], e2["homog_p0.65"]).pvalue)}}
print(f"E2 {time.time()-T0:.1f}s", flush=True)

# ---------------------------------------------------------------- E3 / E4 debate
N, P0, RHO, RMAX = 9, 0.6, 0.3, 10
RS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 3.0, 5.0]


def debate(A, r, g, herd=False):
    A = A.copy(); hist_maj, hist_ind = [], []
    idx = np.arange(NI)
    for t in range(RMAX + 1):
        hist_maj.append(float((plurality(A, g) == 0).mean())); hist_ind.append(float((A == 0).mean()))
        if t == RMAX:
            break
        newA = A.copy()
        maj = plurality(A, g)
        for j in range(N):
            if herd:
                adopt = g.random(NI) < 0.5
                newA[:, j] = np.where(adopt, maj, A[:, j])
                continue
            peer = (j + g.integers(1, N, NI)) % N
            pa = A[idx, peer]; own = A[:, j]
            w_own = np.where(own == 0, r, 1.0); w_peer = np.where(pa == 0, r, 1.0)
            adopt = g.random(NI) < w_peer / (w_own + w_peer)
            newA[:, j] = np.where(adopt, pa, own)
        A = newA
    return hist_maj, hist_ind


def fixation_theory(A, r):
    k = (A == 0).sum(1)
    if r == 1.0:
        return float((k / N).mean())
    return float(((1 - r ** (-k.astype(float))) / (1 - r ** (-float(N)))).mean())


e3 = {f"r{r}": {"maj": [], "ind": [], "fix_theory": []} for r in RS}
e3["herd"] = {"maj": [], "ind": []}
vote_budget = {R: [] for R in range(RMAX + 1)}
headroom, init_vote = [], []
for sd in SEEDS:
    g = rng(500 + sd)
    z = g.standard_normal(NI)
    Abig = answers(g, np.full(N * (RMAX + 1), P0), np.zeros(N * (RMAX + 1), int), RHO, 0.0, z)
    A = Abig[:, :N]
    v0 = plurality(A, g) == 0
    init_vote.append(float(v0.mean()))
    headroom.append(float(((A == 0).any(1) & ~v0).mean()))
    for R in range(RMAX + 1):
        vote_budget[R].append(float((plurality(Abig[:, :N * (R + 1)], g) == 0).mean()))
    for r in RS:
        m, i = debate(A, r, rng(600 + 37 * sd + int(100 * r)))
        e3[f"r{r}"]["maj"].append(m); e3[f"r{r}"]["ind"].append(i); e3[f"r{r}"]["fix_theory"].append(fixation_theory(A, r))
    m, i = debate(A, 1.0, rng(700 + sd), herd=True)
    e3["herd"]["maj"].append(m); e3["herd"]["ind"].append(i)
res3 = {"initial_vote": mean_ci(init_vote), "headroom": mean_ci(headroom),
        "vote_matched_budget": {str(R): mean_ci(v) for R, v in vote_budget.items()}}
for k, d in e3.items():
    M = np.array(d["maj"]); I = np.array(d["ind"])
    res3[k] = {"maj_by_round": [mean_ci(M[:, t]) for t in range(RMAX + 1)],
               "ind_by_round": [mean_ci(I[:, t]) for t in range(RMAX + 1)],
               "final_maj_minus_initial_vote_pp": mean_ci(100 * (M[:, -1] - M[:, 0])),
               "p_final_vs_initial": float(stats.ttest_rel(M[:, -1], M[:, 0]).pvalue),
               "final_vs_matched_vote_pp": mean_ci(100 * (M[:, -1] - np.array(vote_budget[RMAX]))),
               "p_final_vs_matched_vote": float(stats.ttest_rel(M[:, -1], vote_budget[RMAX]).pvalue)}
    if "fix_theory" in d:
        res3[k]["fixation_theory"] = mean_ci(d["fix_theory"])
out["E3_debate"] = res3
# empirical r* where final debate majority (R=10) equals matched-budget vote; theory r* where fixation equals matched vote
emp = [np.mean(np.array(e3[f"r{r}"]["maj"])[:, -1]) for r in RS]
thy = [np.mean(e3[f"r{r}"]["fix_theory"]) for r in RS]
target = np.mean(vote_budget[RMAX])


def cross(xs, ys, t):
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if (y0 - t) * (y1 - t) <= 0 and y1 != y0:
            return x0 + (t - y0) * (x1 - x0) / (y1 - y0)
    return None


out["E4_rstar"] = {"target_matched_vote": float(target), "r_star_empirical": cross(RS, emp, target),
                   "r_star_fixation_theory": cross(RS, thy, target),
                   "r_star_vs_initial_vote_empirical": cross(RS, emp, np.mean(init_vote)),
                   "r_star_vs_initial_vote_theory": cross(RS, thy, np.mean(init_vote))}
out["runtime_s"] = time.time() - T0
od = results_dir(__file__)
(od / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- figures
P = plt()
fig, ax = P.subplots(1, 3, figsize=(16, 4.3))
for rho, c in zip(RHOS, ["C0", "C1", "C2", "C3"]):
    ax[0].errorbar(NS, [e1[str(rho)]["sim"][str(n)][0] for n in NS], yerr=[e1[str(rho)]["sim"][str(n)][1] for n in NS], color=c, marker="o", capsize=2, label=f"仿真 ρ={rho}")
    ax[0].plot(NS, [e1[str(rho)]["theory_finite"][str(n)] for n in NS], color=c, ls="--")
    ax[0].axhline(e1[str(rho)]["theory_ceiling"], color=c, ls=":")
ax[0].set_xscale("log"); ax[0].set_xlabel("智能体数 n"); ax[0].set_ylabel("多数投票准确率"); ax[0].set_title("相关误差下的 Condorcet：虚线=理论，点线=上限")
ax[0].legend(fontsize=7)
for tag, c in (("fam3", "C1"), ("fam9", "C2")):
    ax[1].errorbar(PDIV, [out["E2_diversity"][f"{tag}_p{pd}"][0] for pd in PDIV], yerr=[out["E2_diversity"][f"{tag}_p{pd}"][1] for pd in PDIV], color=c, marker="o", capsize=2,
                   label={"fam3": "3 家族×3", "fam9": "9 个不同家族"}[tag])
ax[1].axhline(out["E2_diversity"]["homog_p0.65"][0], color="k", ls="--", label="同质家族 p=0.65")
ax[1].set_xlabel("多样池个体准确率"); ax[1].set_ylabel("9 智能体投票准确率"); ax[1].legend(fontsize=7); ax[1].set_title("多样性–精度权衡")
rr = np.arange(RMAX + 1)
for k, c in [("r1.0", "C0"), ("r1.5", "C1"), ("r2.0", "C2"), ("r3.0", "C3"), ("r5.0", "C4"), ("herd", "C5")]:
    ax[2].plot(rr, [x[0] for x in res3[k]["maj_by_round"]], color=c, marker="o", ms=3, label=f"辩论 {k}")
ax[2].plot(rr, [res3["vote_matched_budget"][str(R)][0] for R in rr], "k--", label="等预算独立投票")
ax[2].axhline(res3["r1.0"]["ind_by_round"][0][0], color="gray", ls=":", label="个体平均准确率")
ax[2].set_xlabel("辩论轮数 R"); ax[2].set_ylabel("多数答案准确率"); ax[2].legend(fontsize=7); ax[2].set_title("辩论：鞅、从众与“真相偏置”")
fig.tight_layout(); fig.savefig(od / "frontier_vote_debate.png"); P.close(fig)
print(f"done {time.time()-T0:.1f}s")
