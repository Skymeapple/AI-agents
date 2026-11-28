"""Q05 强化学习的理论基础 —— 探索-利用与样本复杂度。

实验 A（多臂老虎机）：K=10 伯努利臂，μ = [0.7, 0.6, 0.55, 0.5, 0.5, 0.45, 0.4, 0.4, 0.35, 0.3]。
  算法：ε-greedy（ε=0.1、ε=0.01 固定）；衰减 ε_t=min(1, cK/(Δ_min² t))，c=1（Auer et al. 2002 形式，需预知 Δ_min）；
        UCB1；Thompson 采样（Beta(1,1) 先验）。T=1e5，每算法 100 次独立运行（向量化）。
  指标：伪遗憾 R(T)=Σ Δ_{a_t}；在 T∈[1e3,1e5] 上拟合 R = a + b ln T 与 log R = c + γ log T；
        与 Lai–Robbins 常数 C = Σ Δ_i / kl(μ_i, μ*) 比较。
实验 B（链式 MDP 的深度探索）：长度 L 的确定性链，“右”前进，在末端再选“右”获奖励 1 并回到起点；“左”回到起点、奖励 0。
  Q 学习（α=0.5, γ=0.99）三种探索：ε-greedy(ε=0.1, Q0=0)；乐观初始化 Q0=1/(1-γ) + 贪心；乐观 + ε-greedy。
  指标：首次到达终点步数；贪心策略首次在所有状态都为“右”（最优）的步数；预算 1e5 步（截尾）。

运行：python ai2035/projects/Q05-rl-theory/experiment.py
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt as get_plt, results_dir, rng, save_results  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)

MU = np.array([0.7, 0.6, 0.55, 0.5, 0.5, 0.45, 0.4, 0.4, 0.35, 0.3])
K = len(MU)
GAP = MU.max() - MU
TMAX = 100_000
RUNS = 100
CHECK = np.unique(np.round(np.logspace(1, 5, 41)).astype(int))


def kl_bern(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


LR_C = sum(GAP[i] / kl_bern(MU[i], MU.max()) for i in range(K) if GAP[i] > 0)


def bandit(alg, g):
    """返回 (RUNS, len(CHECK)) 的累计伪遗憾。"""
    R = RUNS
    cnt = np.zeros((R, K)); sm = np.zeros((R, K))
    reg = np.zeros(R)
    out = np.zeros((R, len(CHECK)))
    ci = 0
    rows = np.arange(R)
    dmin = np.min(GAP[GAP > 0])
    for t in range(1, TMAX + 1):
        if alg == "ts":
            a = np.argmax(g.beta(sm + 1, cnt - sm + 1), 1)
        elif t <= K:
            a = np.full(R, t - 1)
        else:
            mean = sm / cnt
            if alg == "ucb1":
                a = np.argmax(mean + np.sqrt(2 * math.log(t) / cnt), 1)
            else:
                if alg == "eps0.1":
                    eps = 0.1
                elif alg == "eps0.01":
                    eps = 0.01
                else:  # 衰减
                    eps = min(1.0, 1.0 * K / (dmin ** 2 * t))
                a = np.argmax(mean, 1)
                ex = g.random(R) < eps
                if ex.any():
                    a[ex] = g.integers(0, K, ex.sum())
        r = (g.random(R) < MU[a]).astype(float)
        cnt[rows, a] += 1; sm[rows, a] += r
        reg += GAP[a]
        if ci < len(CHECK) and t == CHECK[ci]:
            out[:, ci] = reg; ci += 1
    return out


def fit_regret(curve):
    m = (CHECK >= 1000)
    b, a = np.polyfit(np.log(CHECK[m]), curve[m], 1)
    gam = np.polyfit(np.log(CHECK[m]), np.log(np.maximum(curve[m], 1e-9)), 1)[0]
    return b, gam


# ---------------------------------------------------------------- 链式 MDP
def chain_run(L, mode, seed, budget=100_000, alpha=0.5, gamma=0.99):
    rnd = random.Random(seed)
    q0 = 1 / (1 - gamma) if mode in ("opt", "opt_eps") else 0.0
    eps = 0.1 if mode in ("eps", "opt_eps") else 0.0
    Q = [[q0, q0] for _ in range(L)]  # 动作 0=左，1=右
    s = 0
    n_right_greedy = 0  # 贪心严格选“右”的状态数
    first_goal = None
    for t in range(1, budget + 1):
        qs = Q[s]
        if eps > 0 and rnd.random() < eps:
            a = rnd.randrange(2)
        elif qs[0] == qs[1]:
            a = rnd.randrange(2)
        else:
            a = 0 if qs[0] > qs[1] else 1
        if a == 1:
            if s == L - 1:
                s2, r = 0, 1.0  # 到达终点：奖励 1 并回到起点（回合式重置）
                if first_goal is None:
                    first_goal = t
            else:
                s2, r = s + 1, 0.0
        else:
            s2, r = 0, 0.0
        before = qs[1] > qs[0]
        q2 = Q[s2]
        qs[a] += alpha * (r + gamma * (q2[0] if q2[0] > q2[1] else q2[1]) - qs[a])
        after = qs[1] > qs[0]
        n_right_greedy += int(after) - int(before)
        if n_right_greedy == L and first_goal is not None:
            return first_goal, t
        s = s2
    return first_goal, None


if __name__ == "__main__":
    plt = get_plt()
    # ---------------- A
    algs = ["eps0.1", "eps0.01", "eps_decay", "ucb1", "ts"]
    names = {"eps0.1": "ε-greedy ε=0.1", "eps0.01": "ε-greedy ε=0.01", "eps_decay": "衰减 ε（预知 Δmin）",
             "ucb1": "UCB1", "ts": "Thompson 采样"}
    curves = {}
    resA = {"forced_explore_rate": {"eps0.1": 0.1 * GAP.sum() / K, "eps0.01": 0.01 * GAP.sum() / K}, "mu": MU, "T": TMAX, "runs": RUNS, "lai_robbins_C": LR_C, "checkpoints": CHECK, "algs": {}}
    tA = time.time()
    for k, alg in enumerate(algs):
        c = bandit(alg, rng(10 + k))
        curves[alg] = c
        fits = np.array([fit_regret(c[i]) for i in range(RUNS)])
        mean_curve = c.mean(0)
        b_mean, g_mean = fit_regret(mean_curve)
        resA["algs"][alg] = {
            "name": names[alg],
            "regret_mean": mean_curve, "regret_ci": [mean_ci(c[:, j])[1] for j in range(len(CHECK))],
            "R_T": mean_ci(c[:, -1]), "R_1e4": mean_ci(c[:, list(CHECK).index(10000)]),
            "slope_b_vs_lnT_meancurve": b_mean, "slope_b_vs_lnT_runs": mean_ci(fits[:, 0]),
            "b_over_LR_C": b_mean / LR_C,
            "loglog_gamma_meancurve": g_mean, "loglog_gamma_runs": mean_ci(fits[:, 1]),
            "R_T_over_lnT": float(mean_curve[-1] / math.log(TMAX)),
            "late_rate_per_step_1e4_1e5": mean_ci((c[:, -1] - c[:, list(CHECK).index(10000)]) / (TMAX - 10000)),
            "late_loglog_gamma_1e4_1e5": float(np.log(mean_curve[-1] / mean_curve[list(CHECK).index(10000)]) / np.log(10)),
            "frac_runs_R_T_gt_1000": float(np.mean(c[:, -1] > 1000)),
            "R_T_over_C_lnT": float(mean_curve[-1] / (LR_C * math.log(TMAX))),
        }
    # 配对比较：TS vs UCB1 的 R(T)（独立运行，用 Welch t 检验 + Cohen d）
    a_, b_ = curves["ts"][:, -1], curves["ucb1"][:, -1]
    tt = stats.ttest_ind(a_, b_, equal_var=False)
    resA["ts_vs_ucb1"] = {"t": float(tt.statistic), "p": float(tt.pvalue),
                          "cohen_d": float((a_.mean() - b_.mean()) / np.sqrt((a_.var(ddof=1) + b_.var(ddof=1)) / 2))}
    resA["runtime_sec"] = time.time() - tA

    # ---------------- B
    tB = time.time()
    Ls = [4, 6, 8, 10, 12, 14, 16, 20, 24, 32]
    modes = {"eps": "ε-greedy, Q0=0", "opt": "乐观初始化 + 贪心", "opt_eps": "乐观初始化 + ε-greedy"}
    SEEDS = 10
    BUDGET = 100_000
    resB = {"Ls": Ls, "seeds": SEEDS, "budget": BUDGET, "alpha": 0.5, "gamma": 0.99, "modes": {}}
    for mode, mname in modes.items():
        goal = np.full((SEEDS, len(Ls)), np.nan); opt = np.full((SEEDS, len(Ls)), np.nan)
        all_cens_streak = 0
        for j, L in enumerate(Ls):
            if all_cens_streak >= 2:
                continue  # 连续两个长度全部截尾，后续必然截尾，跳过以节省时间
            for s in range(SEEDS):
                fg, to = chain_run(L, mode, seed=1000 * L + s + {"eps": 0, "opt": 100, "opt_eps": 200}[mode], budget=BUDGET)
                goal[s, j] = fg if fg is not None else np.nan
                opt[s, j] = to if to is not None else np.nan
            all_cens_streak = all_cens_streak + 1 if np.all(np.isnan(opt[:, j])) else 0
        med_opt = np.array([np.median(np.where(np.isnan(opt[:, j]), np.inf, opt[:, j])) for j in range(len(Ls))])
        med_goal = np.array([np.median(np.where(np.isnan(goal[:, j]), np.inf, goal[:, j])) for j in range(len(Ls))])
        ok = np.isfinite(med_opt)
        Larr = np.array(Ls, float)
        fit = {}
        if ok.sum() >= 3:
            sl_exp, ic_exp, r_exp, *_ = stats.linregress(Larr[ok], np.log(med_opt[ok]))
            sl_pow, ic_pow, r_pow, *_ = stats.linregress(np.log(Larr[ok]), np.log(med_opt[ok]))
            fit = {"exp_rate_per_step_of_L": sl_exp, "exp_base": math.exp(sl_exp), "R2_loglinear": r_exp ** 2,
                   "poly_degree": sl_pow, "R2_loglog": r_pow ** 2, "n_points": int(ok.sum())}
        okg = np.isfinite(med_goal)
        if okg.sum() >= 3:
            sl, _, r, *_ = stats.linregress(Larr[okg], np.log(med_goal[okg]))
            slp, _, rp, *_ = stats.linregress(np.log(Larr[okg]), np.log(med_goal[okg]))
            fit.update({"goal_exp_base": math.exp(sl), "goal_R2_loglinear": r ** 2, "goal_poly_degree": slp, "goal_R2_loglog": rp ** 2})
        resB["modes"][mode] = {"name": mname, "median_steps_to_optimal": med_opt, "median_steps_to_goal": med_goal,
                               "censored_frac_optimal": np.mean(np.isnan(opt), 0),
                               "steps_to_optimal_per_seed": opt, "fit": fit}
    resB["runtime_sec"] = time.time() - tB

    # ---------------- 图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for alg in algs:
        ax[0].plot(CHECK, curves[alg].mean(0), label=names[alg])
    ax[0].plot(CHECK, LR_C * np.log(CHECK), "k--", label=f"Lai–Robbins C·lnT (C={LR_C:.1f})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("T"); ax[0].set_ylabel("累计伪遗憾（100 次均值）")
    ax[0].legend(fontsize=8); ax[0].set_title("K=10 伯努利老虎机")
    for alg in ["eps_decay", "ucb1", "ts"]:
        ax[1].plot(CHECK, curves[alg].mean(0), label=names[alg])
    ax[1].plot(CHECK, LR_C * np.log(CHECK), "k--", label="C·lnT")
    ax[1].set_xscale("log"); ax[1].set_xlabel("T（对数轴）"); ax[1].set_ylabel("累计伪遗憾（线性轴）"); ax[1].legend(fontsize=8)
    ax[1].set_title("对数增长：斜率即 ln T 系数")
    fig.tight_layout(); fig.savefig(OUT / "A_bandit_regret.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.3))
    for mode, d in resB["modes"].items():
        m = np.array(d["median_steps_to_optimal"], float)
        okm = np.isfinite(m)
        ax.plot(np.array(Ls)[okm], m[okm], "o-", label=d["name"])
    ax.axhline(BUDGET, color="gray", ls=":", label="预算（截尾）")
    ax.set_yscale("log"); ax.set_xlabel("链长 L"); ax.set_ylabel("贪心策略达到最优所需步数（中位数）")
    ax.set_title("链式 MDP：抖动探索 vs 乐观探索"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "B_chain_exploration.png"); plt.close(fig)

    p = save_results(__file__, {"A_bandit": resA, "B_chain": resB, "runtime_sec_total": time.time() - T0})
    print("saved", p, time.time() - T0)
    for alg in algs:
        d = resA["algs"][alg]
        print(alg, d["R_T"], d["slope_b_vs_lnT_meancurve"], d["b_over_LR_C"], d["loglog_gamma_meancurve"])
    print("C", LR_C)
    for mode, d in resB["modes"].items():
        print(mode, d["median_steps_to_optimal"], d["median_steps_to_goal"], d["fit"])
