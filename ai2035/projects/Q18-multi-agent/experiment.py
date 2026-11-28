"""Q18 多智能体与群体智能：合作涌现、独立学习的协调失败、群体相变。

(a) 迭代囚徒困境（R,S,T,P = 3,0,5,1）。8 种 Axelrod 式策略，200 轮，执行噪声 ε
    （意图动作以概率 ε 翻转）。每个种子以 20 次重复模拟估计支付矩阵，再在 40 个
    Dirichlet(1) 随机初态上运行带突变（μ=1e-3）的离散复制子动力学 3000 代。
(b) 学习：
    - 爬山博弈（Claus & Boutilier 1998）：独立 Q 学习 IL、迟滞 Q 学习 HYS（Matignon 2007）、
      联合动作学习 JAL（对手经验频率模型）、集中式学习 CEN（单一体选联合动作）。
      Boltzmann 探索，温度 T_t = max(0.05, 10·0.996^t)，α=0.1，3000 步；每种子 40 次独立运行。
    - 重复囚徒困境：无状态 IL（γ=0）、一阶记忆 IL（状态=上一联合动作, γ=0.95）、集中式（最大化
      双方收益和）。ε-greedy 从 1 线性衰减到 0.02（前 80% 步），α=0.1，20000 步。
(c) Vicsek 模型：密度 ρ=4，v=0.03，r=1，噪声 η∈[0,5]，N∈{100,300}，600 步，取后 200 步序参量。
5 个种子；均值 ± 95% CI。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng, save_results  # noqa: E402

from scipy import stats  # noqa: E402

T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
res = {"config": {"seeds": SEEDS}}

# ============================================================ (a) IPD tournament
STRATS = ["ALLC", "ALLD", "TFT", "GTFT", "WSLS", "GRIM", "TF2T", "RAND"]
PAY = np.array([[3.0, 0.0], [5.0, 1.0]])  # PAY[my, opp], 0=C, 1=D
ROUNDS, REPS = 200, 20
NOISES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
N_INIT, GENS, MU = 40, 3000, 1e-3


def intend(k, t, my1, op1, op2, grim, g):
    """Intended action (1 = defect) for strategy k; arrays over matches."""
    if t == 0:
        base = np.zeros_like(my1)
        if STRATS[k] == "ALLD":
            base = np.ones_like(my1)
        if STRATS[k] == "RAND":
            base = (g.random(my1.shape) < 0.5).astype(int)
        return base
    s = STRATS[k]
    if s == "ALLC":
        return np.zeros_like(my1)
    if s == "ALLD":
        return np.ones_like(my1)
    if s == "TFT":
        return op1.copy()
    if s == "GTFT":
        return ((op1 == 1) & (g.random(my1.shape) > 1 / 3)).astype(int)
    if s == "WSLS":  # repeat if payoff R or T (opp cooperated), else switch
        return np.where(op1 == 0, my1, 1 - my1)
    if s == "GRIM":
        return grim.copy()
    if s == "TF2T":
        return ((op1 == 1) & (op2 == 1)).astype(int)
    if s == "RAND":
        return (g.random(my1.shape) < 0.5).astype(int)


def payoff_matrix(eps, g):
    n = len(STRATS)
    A = np.zeros((n, n)); C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a1 = np.zeros(REPS, int); a2 = np.zeros(REPS, int)
            b1 = np.zeros(REPS, int); b2 = np.zeros(REPS, int)
            ga = np.zeros(REPS, int); gb = np.zeros(REPS, int)
            pay, coop = 0.0, 0.0
            for t in range(ROUNDS):
                x = intend(i, t, a1, b1, b2, ga, g)
                y = intend(j, t, b1, a1, a2, gb, g)
                x = np.where(g.random(REPS) < eps, 1 - x, x)
                y = np.where(g.random(REPS) < eps, 1 - y, y)
                pay += PAY[x, y].mean(); coop += (x == 0).mean()
                a2, a1 = a1, x; b2, b1 = b1, y
                ga = np.maximum(ga, y); gb = np.maximum(gb, x)
            A[i, j] = pay / ROUNDS; C[i, j] = coop / ROUNDS
    return A, C


def replicator(A, x0):
    x = x0.copy()
    n = len(x)
    for _ in range(GENS):
        f = A @ x
        x = x * f / (x @ f)
        x = (1 - MU) * x + MU / n
    return x


ipd = {}
for eps in NOISES:
    coop_runs, dom_counts, mean_freq = [], np.zeros(len(STRATS)), []
    seed_coop = []
    for sd in SEEDS:
        g = rng(int(1000 * eps) + 17 * sd)
        A, C = payoff_matrix(eps, g)
        X0 = g.dirichlet(np.ones(len(STRATS)), N_INIT)
        cs = []
        for x0 in X0:
            x = replicator(A, x0)
            cs.append(float(x @ C @ x))
            dom_counts[np.argmax(x)] += 1
            mean_freq.append(x)
        seed_coop.append(np.mean(cs)); coop_runs += cs
    coop_runs = np.array(coop_runs)
    ipd[str(eps)] = {
        "pop_coop": mean_ci(seed_coop),
        "frac_inits_coop_gt_0.5": float((coop_runs > 0.5).mean()),
        "dominant_share": {s: float(dom_counts[k] / dom_counts.sum()) for k, s in enumerate(STRATS)},
        "mean_freq": {s: float(np.mean(mean_freq, 0)[k]) for k, s in enumerate(STRATS)},
        "payoff_matrix_last_seed": A.tolist(),
    }
    print(f"IPD eps={eps} coop={np.mean(seed_coop):.3f} {time.time()-T0:.1f}s", flush=True)
res["ipd"] = {"strategies": STRATS, "noises": NOISES, "by_noise": ipd}

# ============================================================ (b1) climbing game
CLIMB = np.array([[11.0, -30.0, 0.0], [-30.0, 7.0, 6.0], [0.0, 0.0, 5.0]])
RUNS, STEPS, ALPHA = 40, 3000, 0.1


def temp(t):
    return max(0.05, 10.0 * 0.996 ** t)


def boltz(Q, T, g):
    z = (Q - Q.max(-1, keepdims=True)) / T
    p = np.exp(z); p /= p.sum(-1, keepdims=True)
    u = g.random((Q.shape[0], 1))
    return (p.cumsum(-1) < u).sum(-1).clip(0, Q.shape[-1] - 1)


def run_climb(method, g):
    R = RUNS; idx = np.arange(R)
    greedy_hist = np.zeros((STEPS, R), int)
    if method in ("IL", "HYS"):
        Q1 = np.zeros((R, 3)); Q2 = np.zeros((R, 3))
        for t in range(STEPS):
            T = temp(t)
            a = boltz(Q1, T, g); b = boltz(Q2, T, g)
            r = CLIMB[a, b]
            for Q, act in ((Q1, a), (Q2, b)):
                d = r - Q[idx, act]
                lr = ALPHA if method == "IL" else np.where(d >= 0, 0.1, 0.01)
                Q[idx, act] += lr * d
            greedy_hist[t] = Q1.argmax(1) * 3 + Q2.argmax(1)
    elif method == "JAL":
        Q1 = np.zeros((R, 3, 3)); Q2 = np.zeros((R, 3, 3))  # Q_i[own, other]
        N1 = np.ones((R, 3)); N2 = np.ones((R, 3))  # counts of other's actions
        for t in range(STEPS):
            T = temp(t)
            ev1 = np.einsum("rij,rj->ri", Q1, N2 / N2.sum(1, keepdims=True))
            ev2 = np.einsum("rij,rj->ri", Q2, N1 / N1.sum(1, keepdims=True))
            a = boltz(ev1, T, g); b = boltz(ev2, T, g)
            r = CLIMB[a, b]
            Q1[idx, a, b] += ALPHA * (r - Q1[idx, a, b]); Q2[idx, b, a] += ALPHA * (r - Q2[idx, b, a])
            N1[idx, a] += 1; N2[idx, b] += 1
            greedy_hist[t] = ev1.argmax(1) * 3 + ev2.argmax(1)
    elif method == "CEN":
        Q = np.zeros((R, 9))
        for t in range(STEPS):
            j = boltz(Q, temp(t), g)
            r = CLIMB.ravel()[j]
            Q[idx, j] += ALPHA * (r - Q[idx, j])
            greedy_hist[t] = Q.argmax(1)
    final = greedy_hist[-1]
    # convergence time: last change of greedy joint action
    changed = greedy_hist[1:] != greedy_hist[:-1]
    conv_t = np.array([(np.nonzero(changed[:, r])[0].max() + 1) if changed[:, r].any() else 0 for r in range(R)])
    return final, conv_t


climb = {}
for method in ["IL", "HYS", "JAL", "CEN"]:
    opt_rates, outcome = [], np.zeros(9)
    conv_ts = []
    for sd in SEEDS:
        final, conv_t = run_climb(method, rng(500 + sd))
        opt_rates.append(float((final == 0).mean()))
        outcome += np.bincount(final, minlength=9)
        conv_ts += conv_t.tolist()
    k, n = int(outcome[0]), int(outcome.sum())
    wil = stats.binomtest(k, n).proportion_ci(method="wilson")
    climb[method] = {"p_optimal": mean_ci(opt_rates), "p_optimal_pooled": k / n,
                     "wilson95": [wil.low, wil.high],
                     "outcome_share": {f"({i // 3},{i % 3})": float(outcome[i] / n) for i in range(9)},
                     "conv_step": mean_ci(conv_ts),
                     "expected_payoff_final": float((outcome * CLIMB.ravel()).sum() / n)}
    print(f"climb {method} p_opt={k/n:.3f} {time.time()-T0:.1f}s", flush=True)
res["climbing"] = climb
res["climbing_tests"] = {
    "IL_vs_CEN_fisher_p": float(stats.fisher_exact([[climb["IL"]["p_optimal_pooled"] * 200, 200 - climb["IL"]["p_optimal_pooled"] * 200],
                                                     [climb["CEN"]["p_optimal_pooled"] * 200, 200 - climb["CEN"]["p_optimal_pooled"] * 200]])[1]),
}

# ============================================================ (b2) repeated PD learners
PD_STEPS, PD_RUNS = 20000, 40
PD = np.array([[3.0, 0.0], [5.0, 1.0]])


def run_pd(method, g):
    R = PD_RUNS; idx = np.arange(R)
    eps_sched = lambda t: max(0.02, 1 - t / (0.8 * PD_STEPS) * 0.98)  # noqa: E731
    coop_tail = np.zeros(R); tail = int(0.1 * PD_STEPS)
    if method == "CEN":
        Q = np.zeros((R, 4))
        for t in range(PD_STEPS):
            e = eps_sched(t)
            j = np.where(g.random(R) < e, g.integers(0, 4, R), Q.argmax(1))
            a, b = j // 2, j % 2
            r = PD[a, b] + PD[b, a]
            Q[idx, j] += 0.1 * (r - Q[idx, j])
            if t >= PD_STEPS - tail:
                coop_tail += ((a == 0) & (b == 0))
        return coop_tail / tail
    mem = method == "IL_mem1"
    gamma = 0.95 if mem else 0.0
    nS = 4 if mem else 1
    Q1 = np.zeros((R, nS, 2)); Q2 = np.zeros((R, nS, 2))
    s = np.zeros(R, int)
    for t in range(PD_STEPS):
        e = eps_sched(t)
        a = np.where(g.random(R) < e, g.integers(0, 2, R), Q1[idx, s].argmax(1))
        b = np.where(g.random(R) < e, g.integers(0, 2, R), Q2[idx, s].argmax(1))
        r1, r2 = PD[a, b], PD[b, a]
        s2 = (a * 2 + b) if mem else s
        s2b = (b * 2 + a) if mem else s  # agent 2 sees (own, other)
        sb = s if not mem else ((s % 2) * 2 + s // 2)
        Q1[idx, s, a] += 0.1 * (r1 + gamma * Q1[idx, s2].max(1) - Q1[idx, s, a])
        Q2[idx, sb, b] += 0.1 * (r2 + gamma * Q2[idx, s2b].max(1) - Q2[idx, sb, b])
        s = s2
        if t >= PD_STEPS - tail:
            coop_tail += ((a == 0) & (b == 0))
    return coop_tail / tail


pdres = {}
pd_seed = {}
for method in ["IL_stateless", "IL_mem1", "CEN"]:
    per_seed, allr = [], []
    for sd in SEEDS:
        c = run_pd(method, rng(900 + sd))
        per_seed.append(float(c.mean())); allr += c.tolist()
    allr = np.array(allr)
    pd_seed[method] = per_seed
    pdres[method] = {"mutual_coop_rate": mean_ci(per_seed),
                     "frac_runs_mostly_coop": float((allr > 0.5).mean()),
                     "frac_runs_mostly_defect": float((allr < 0.1).mean())}
    print(f"PD {method} coop={np.mean(per_seed):.3f} {time.time()-T0:.1f}s", flush=True)
res["pd_learning"] = pdres
res["pd_tests"] = {
    "mem1_vs_stateless": {"t": float(stats.ttest_rel(pd_seed["IL_mem1"], pd_seed["IL_stateless"]).statistic),
                          "p": float(stats.ttest_rel(pd_seed["IL_mem1"], pd_seed["IL_stateless"]).pvalue)},
}

# ============================================================ (c) Vicsek
ETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
VN = [100, 300]
V_STEPS, V_TAIL = 600, 200


def vicsek(N, eta, g):
    L = np.sqrt(N / 4.0)
    pos = g.random((N, 2)) * L
    th = g.uniform(-np.pi, np.pi, N)
    va = []
    for t in range(V_STEPS):
        dx = pos[:, None, 0] - pos[None, :, 0]; dx -= L * np.round(dx / L)
        dy = pos[:, None, 1] - pos[None, :, 1]; dy -= L * np.round(dy / L)
        nb = (dx * dx + dy * dy <= 1.0).astype(np.float64)
        s = nb @ np.sin(th); c = nb @ np.cos(th)
        th = np.arctan2(s, c) + g.uniform(-eta / 2, eta / 2, N)
        pos = (pos + 0.03 * np.stack([np.cos(th), np.sin(th)], 1)) % L
        if t >= V_STEPS - V_TAIL:
            va.append(np.hypot(np.cos(th).mean(), np.sin(th).mean()))
    va = np.array(va)
    return float(va.mean()), float(va.var() * N)  # order parameter, susceptibility


vic = {}
for N in VN:
    d = {}
    for eta in ETAS:
        vs, chis = [], []
        for sd in SEEDS:
            v, chi = vicsek(N, eta, rng(3000 + 10 * sd + N))
            vs.append(v); chis.append(chi)
        d[str(eta)] = {"va": mean_ci(vs), "chi": mean_ci(chis)}
    vic[str(N)] = d
    print(f"Vicsek N={N} {time.time()-T0:.1f}s", flush=True)
res["vicsek"] = {"etas": ETAS, "by_N": vic}
res["runtime_s"] = time.time() - T0

# ============================================================ figures
P = plt()
od = results_dir(__file__)
fig, ax = P.subplots(1, 2, figsize=(12, 4))
m = [ipd[str(e)]["pop_coop"][0] for e in NOISES]; c = [ipd[str(e)]["pop_coop"][1] for e in NOISES]
ax[0].errorbar(NOISES, m, yerr=c, marker="o", capsize=3, color="k")
ax[0].set_xlabel("执行噪声 ε"); ax[0].set_ylabel("演化稳态合作率"); ax[0].set_title("IPD 复制子动力学：合作率 vs 噪声")
ax[0].set_ylim(0, 1.02)
bottom = np.zeros(len(NOISES))
for s in STRATS:
    v = np.array([ipd[str(e)]["mean_freq"][s] for e in NOISES])
    ax[1].bar(range(len(NOISES)), v, bottom=bottom, label=s); bottom += v
ax[1].set_xticks(range(len(NOISES))); ax[1].set_xticklabels([str(e) for e in NOISES])
ax[1].set_xlabel("ε"); ax[1].set_ylabel("平均稳态频率"); ax[1].legend(fontsize=7, ncol=2); ax[1].set_title("策略组成")
fig.tight_layout(); fig.savefig(od / "ipd_noise.png"); P.close(fig)

fig, ax = P.subplots(1, 2, figsize=(12, 4))
meths = ["IL", "HYS", "JAL", "CEN"]
ax[0].bar(meths, [climb[k]["p_optimal_pooled"] for k in meths],
          yerr=[[climb[k]["p_optimal_pooled"] - climb[k]["wilson95"][0] for k in meths],
                [climb[k]["wilson95"][1] - climb[k]["p_optimal_pooled"] for k in meths]], capsize=4)
ax[0].set_ylabel("收敛到最优 (A,A) 的比例"); ax[0].set_title("爬山博弈")
pm = ["IL_stateless", "IL_mem1", "CEN"]
ax[1].bar(pm, [pdres[k]["mutual_coop_rate"][0] for k in pm], yerr=[pdres[k]["mutual_coop_rate"][1] for k in pm], capsize=4)
ax[1].set_ylabel("末 10% 步互相合作率"); ax[1].set_title("重复囚徒困境中的 Q 学习者")
fig.tight_layout(); fig.savefig(od / "learning.png"); P.close(fig)

fig, ax = P.subplots(figsize=(6, 4))
for N in VN:
    ax.errorbar(ETAS, [vic[str(N)][str(e)]["va"][0] for e in ETAS], yerr=[vic[str(N)][str(e)]["va"][1] for e in ETAS],
                marker="o", capsize=3, label=f"N={N}")
ax.set_xlabel("噪声 η"); ax.set_ylabel("序参量 v_a"); ax.set_title("Vicsek 模型有序–无序转变"); ax.legend()
fig.tight_layout(); fig.savefig(od / "vicsek.png"); P.close(fig)

save_results(__file__, res)
print(f"done in {time.time()-T0:.1f}s")
