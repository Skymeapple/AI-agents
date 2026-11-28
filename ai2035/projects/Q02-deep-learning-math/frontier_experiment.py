"""Q02 前沿实验：grokking 的“权重衰减时钟”——延迟定律的指数、初始化范数与临界范数。

检验 2025–2026 年的三种竞争说法（见 paper.md）：
  (a) 范数分离延迟定律：T_grok − T_mem = Θ((ηλ)^-1 · log(‖θ_mem‖²/‖θ_post‖²))（预测 λ 指数 = −1）；
  (b) 经验标度拟合：T_grok ∝ λ^-0.64（预测指数 ≈ −0.64）；
  (c) 临界范数：grokking 发生在几乎恒定的权重范数 W_c 处。
模型：
  E0 线性岭回归（过参数化，GD + 权重衰减，大初始化）——延迟定律的精确可解对照；
  E1–E3 二层二次激活 MLP（Gromov 2023 式）在模加法 (a+b) mod 31 上，全批量 AdamW。
  E1 λ 扫描（α=1）；E2 初始化尺度 α 扫描（λ=0.3, 1）；E3 训练数据比例扫描（λ=1）。
运行：python ai2035/projects/Q02-deep-learning-math/frontier_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import _jsonable, mean_ci, plt as get_plt, results_dir, rng  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)
P_MOD, H, LR = 31, 128, 3e-3
SEEDS = 5
MAXSTEP = 15000
EVAL = 50


# ---------------------------------------------------------------- E0 线性模型（精确理论）
def linear_grok(alpha, lam, seed, d=400, n=100, eta=0.2, eps=0.05, maxstep=400000):
    """GD + 权重衰减的过参数化线性回归（n<d）：返回 (T_mem, T_grok, T_theory)。

    GD 不动点为岭解 w_λ；T_grok 定义为 ‖w_t − w*‖² 首次 < ‖w_λ − w*‖² + eps。
    零空间分量 a=(I−P)w_0 按 r=(1−ηλ)^t 衰减（行空间分量以 O(1) 速率收敛），故
    ‖w_t − w*‖² − ‖w_λ − w*‖² ≈ r²‖a‖² + 2r·a·b，b=(I−P)(w_λ − w*)；令其 = eps 解出 r，T = ln r / ln(1−ηλ)。
    """
    g = rng(50000 + seed)
    X = g.standard_normal((n, d)) / np.sqrt(d)
    wstar = g.standard_normal(d); wstar /= np.linalg.norm(wstar)
    y = X @ wstar
    w = alpha * 3 * g.standard_normal(d) / np.sqrt(d)  # 初始化范数 ≈ 3α
    Pr = X.T @ np.linalg.solve(X @ X.T, X)
    w_lam = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
    e_inf = np.sum((w_lam - wstar) ** 2)
    a = w - Pr @ w
    b = (w_lam - wstar) - Pr @ (w_lam - wstar)
    A2, AB = np.sum(a * a), np.sum(a * b)
    r = (-2 * AB + np.sqrt(4 * AB ** 2 + 4 * A2 * eps)) / (2 * A2)
    t_th = np.log(r) / np.log(1 - eta * lam) if 0 < r < 1 else 0.0
    tm = tg = None
    for t in range(1, maxstep + 1):
        w = w - eta * (X.T @ (X @ w - y) + lam * w)
        if t % 5 == 0:
            if tm is None and np.mean((X @ w - y) ** 2) < 1e-3:
                tm = t
            if tg is None and np.sum((w - wstar) ** 2) < e_inf + eps:
                tg = t
                break
    return tm, tg, t_th, e_inf


# ---------------------------------------------------------------- 模加法数据与二次 MLP
def mod_data(frac, seed):
    p = P_MOD
    A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    A, B = A.ravel(), B.ravel()
    Y = (A + B) % p
    X = np.zeros((p * p, 2 * p)); X[np.arange(p * p), A] = 1; X[np.arange(p * p), p + B] = 1
    g = rng(7000 + seed)
    perm = g.permutation(p * p)
    ntr = int(round(frac * p * p))
    return X[perm[:ntr]], Y[perm[:ntr]], X[perm[ntr:]], Y[perm[ntr:]]


def train(frac, lam, alpha, seed, maxstep=MAXSTEP):
    Xtr, Ytr, Xte, Yte = mod_data(frac, seed)
    g = rng(9000 + seed)
    p = P_MOD
    W1 = g.standard_normal((2 * p, H)) * alpha / np.sqrt(2 * p) * np.sqrt(2)
    W2 = g.standard_normal((H, p)) * alpha / np.sqrt(H)
    Ps = [W1, W2]
    M = [np.zeros_like(w) for w in Ps]; V = [np.zeros_like(w) for w in Ps]
    b1, b2 = 0.9, 0.98
    n = len(Ytr)
    norm0 = np.sqrt(sum((w * w).sum() for w in Ps))
    tm = tg = None; nm = ng = None
    traj = []
    for t in range(1, maxstep + 1):
        Z = Xtr @ Ps[0]; Hh = Z * Z; O = Hh @ Ps[1]
        O -= O.max(1, keepdims=True); E = np.exp(O); S = E / E.sum(1, keepdims=True)
        S[np.arange(n), Ytr] -= 1; S /= n
        g2 = Hh.T @ S; g1 = Xtr.T @ ((S @ Ps[1].T) * 2 * Z)
        for k, gr in enumerate([g1, g2]):
            M[k] = b1 * M[k] + (1 - b1) * gr; V[k] = b2 * V[k] + (1 - b2) * gr * gr
            Ps[k] *= (1 - LR * lam)  # 解耦权重衰减（AdamW）
            Ps[k] -= LR * (M[k] / (1 - b1 ** t)) / (np.sqrt(V[k] / (1 - b2 ** t)) + 1e-8)
        if t % EVAL == 0:
            atr = np.mean(np.argmax(((Xtr @ Ps[0]) ** 2) @ Ps[1], 1) == Ytr)
            ate = np.mean(np.argmax(((Xte @ Ps[0]) ** 2) @ Ps[1], 1) == Yte)
            nrm = np.sqrt(sum((w * w).sum() for w in Ps))
            traj.append((t, atr, ate, nrm))
            if tm is None and atr >= 0.99:
                tm, nm = t, nrm
            if tg is None and ate >= 0.95:
                tg, ng = t, nrm
                break
    return {"T_mem": tm, "T_grok": tg, "norm0": norm0, "norm_mem": nm, "norm_grok": ng, "traj": traj}


def summarize(runs):
    tg = np.array([r["T_grok"] if r["T_grok"] else np.nan for r in runs], float)
    tm = np.array([r["T_mem"] if r["T_mem"] else np.nan for r in runs], float)
    ng = np.array([r["norm_grok"] if r["norm_grok"] else np.nan for r in runs], float)
    nm = np.array([r["norm_mem"] if r["norm_mem"] else np.nan for r in runs], float)
    ok = ~np.isnan(tg)
    return {"T_grok": mean_ci(tg[ok]) if ok.sum() >= 2 else None, "T_mem": mean_ci(tm[~np.isnan(tm)]),
            "delay": mean_ci((tg - tm)[ok]) if ok.sum() >= 2 else None,
            "n_grokked": int(ok.sum()), "norm_grok": mean_ci(ng[ok]) if ok.sum() >= 2 else None,
            "norm_mem": mean_ci(nm[~np.isnan(nm)]), "T_grok_per_seed": tg, "norm_grok_per_seed": ng}


if __name__ == "__main__":
    plt = get_plt()
    res = {"setup": {"p": P_MOD, "hidden": H, "lr": LR, "seeds": SEEDS, "max_steps": MAXSTEP, "eval_every": EVAL,
                     "T_mem": "训练准确率 ≥0.99 的首个评估步", "T_grok": "测试准确率 ≥0.95 的首个评估步",
                     "model": "x=[onehot(a),onehot(b)] → (xW1)^2 → W2，交叉熵，AdamW(β=0.9,0.98)"}}

    # ---------------- E0 线性模型
    e0 = {"alphas": [1, 2, 4, 8], "lams": [0.005, 0.01, 0.02], "eta": 0.2, "eps": 0.05, "rows": []}
    for lam in e0["lams"]:
        for a in e0["alphas"]:
            r = [linear_grok(a, lam, s) for s in range(SEEDS)]
            tg = np.array([x[1] for x in r], float); th = np.array([x[2] for x in r], float)
            e0["rows"].append({"lam": lam, "alpha": a, "T_grok": mean_ci(tg), "T_theory": mean_ci(th),
                               "rel_err": mean_ci(np.abs(tg - th) / th), "T_mem": mean_ci(np.array([x[0] for x in r], float))})
    allrel = np.array([r["rel_err"][0] for r in e0["rows"]])
    e0["max_rel_err"] = float(allrel.max())
    # λ 指数（固定 α=4）
    sub = [r for r in e0["rows"] if r["alpha"] == 4]
    e0["lam_exponent_alpha4"] = float(np.polyfit(np.log([r["lam"] for r in sub]), np.log([r["T_grok"][0] for r in sub]), 1)[0])
    res["E0_linear"] = e0
    t_e0 = time.time() - T0

    # ---------------- E1 λ 扫描
    lams = [0.1, 0.17, 0.3, 0.5, 1.0, 1.7, 3.0]
    e1 = {"lams": lams, "alpha": 1.0, "frac": 0.5, "rows": []}
    E1runs = {}
    for lam in lams:
        runs = [train(0.5, lam, 1.0, s) for s in range(SEEDS)]
        E1runs[lam] = runs
        row = summarize(runs); row["lam"] = lam
        e1["rows"].append(row)
    # 全局与分段的 λ 指数（逐种子在 log–log 上拟合延迟 T_grok − T_mem）
    def exponent(lsub):
        out = []
        for s in range(SEEDS):
            x, y = [], []
            for lam in lsub:
                r = E1runs[lam][s]
                if r["T_grok"]:
                    x.append(np.log(lam)); y.append(np.log(r["T_grok"] - r["T_mem"]))
            if len(x) >= 3:
                out.append(np.polyfit(x, y, 1)[0])
        return mean_ci(out), out
    e1["exponent_all"], _ = exponent(lams)
    e1["exponent_small_lam(0.1-0.5)"], _ = exponent([0.1, 0.17, 0.3, 0.5])
    e1["exponent_large_lam(0.5-3)"], _ = exponent([0.5, 1.0, 1.7, 3.0])
    sm = e1["exponent_small_lam(0.1-0.5)"]; lg = e1["exponent_large_lam(0.5-3)"]
    a_ = np.array(exponent([0.1, 0.17, 0.3, 0.5])[1]); b_ = np.array(exponent([0.5, 1.0, 1.7, 3.0])[1])
    if len(a_) == len(b_) and len(a_) >= 2:
        tt = stats.ttest_rel(a_, b_)
        e1["small_vs_large_exponent_paired_t"] = {"t": float(tt.statistic), "p": float(tt.pvalue)}
    # ηλ·(T_grok−T_mem)：若延迟定律成立且 log 项不变，则应近似常数
    e1["eta_lam_times_delay"] = [mean_ci(np.array([LR * lam * (r["T_grok"] - r["T_mem"]) for r in E1runs[lam] if r["T_grok"]]))
                                 for lam in lams]
    res["E1_lambda"] = e1
    t_e1 = time.time() - T0

    # ---------------- E2 α 扫描
    alphas = [0.5, 1.0, 2.0, 4.0]
    e2 = {"alphas": alphas, "lams": [0.3, 1.0], "rows": [], "slopes": {}}
    E2runs = {}
    for lam in e2["lams"]:
        for a in alphas:
            # α=1 且 λ 在 E1 网格内时复用 E1 的运行（同种子、同配置，结果相同）
            runs = E1runs[lam] if (a == 1.0 and lam in E1runs) else [train(0.5, lam, a, s) for s in range(SEEDS)]
            E2runs[(lam, a)] = runs
            row = summarize(runs); row.update({"lam": lam, "alpha": a})
            e2["rows"].append(row)
        # T_grok 对 ln α 的斜率（逐种子），以及 ηλ·斜率（延迟定律预测 ≈ 常数 k，ln‖θ‖ 每增 1，延迟增 k/(ηλ)）
        sl, r2 = [], []
        for s in range(SEEDS):
            x = [np.log(a) for a in alphas if E2runs[(lam, a)][s]["T_grok"]]
            y = [E2runs[(lam, a)][s]["T_grok"] for a in alphas if E2runs[(lam, a)][s]["T_grok"]]
            if len(x) >= 3:
                fit = stats.linregress(x, y); sl.append(fit.slope); r2.append(fit.rvalue ** 2)
        e2["slopes"][str(lam)] = {"slope_per_ln_alpha": mean_ci(sl), "R2": mean_ci(r2),
                                  "eta_lam_x_slope": mean_ci(np.array(sl) * LR * lam),
                                  "theory_eta_lam_x_slope_if_norm_clock": 1.0}
    res["E2_alpha"] = e2
    t_e2 = time.time() - T0

    # ---------------- 临界范数检验（合并 E1、E2 全部 grok 的运行）
    allng, groups = [], {}
    for lam in lams:
        v = [r["norm_grok"] for r in E1runs[lam] if r["norm_grok"]]
        allng += v; groups[f"E1_lam{lam}"] = float(np.mean(v)) if v else None
    for (lam, a), runs in E2runs.items():
        v = [r["norm_grok"] for r in runs if r["norm_grok"]]
        allng += v; groups[f"E2_lam{lam}_a{a}"] = float(np.mean(v)) if v else None
    allng = np.array(allng)
    e1_ng = np.array([r["norm_grok"] for lam in lams for r in E1runs[lam] if r["norm_grok"]])
    e2_lam1 = np.array([r["norm_grok"] for a in alphas for r in E2runs[(1.0, a)] if r["norm_grok"]])
    res["critical_norm"] = {"CV_all_runs": float(allng.std() / allng.mean()), "mean_all": float(allng.mean()),
                            "CV_E1_across_lambda": float(e1_ng.std() / e1_ng.mean()),
                            "CV_E2_lam1_across_alpha": float(e2_lam1.std() / e2_lam1.mean()),
                            "group_means": groups,
                            "spearman_normgrok_vs_lambda_E1": float(stats.spearmanr(
                                [lam for lam in lams for r in E1runs[lam] if r["norm_grok"]], e1_ng)[0])}

    # ---------------- E3 数据比例
    fracs = [0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
    e3 = {"fracs": fracs, "lam": 1.0, "max_steps": 12000, "rows": []}
    for fr in fracs:
        runs = E1runs[1.0] if fr == 0.5 else [train(fr, 1.0, 1.0, s, maxstep=12000) for s in range(SEEDS)]
        row = summarize(runs); row["frac"] = fr
        e3["rows"].append(row)
    okf = [(r["frac"], r["T_grok"][0]) for r in e3["rows"] if r["T_grok"] and r["n_grokked"] == SEEDS]
    if len(okf) >= 3:
        e3["frac_exponent_all_grokked"] = float(np.polyfit(np.log([x[0] for x in okf]), np.log([x[1] for x in okf]), 1)[0])
    res["E3_fraction"] = e3
    res["runtime_sec"] = {"E0": t_e0, "E1": t_e1 - t_e0, "E2": t_e2 - t_e1, "total": time.time() - T0, "cpu_process_time": time.process_time()}

    # ---------------- 图
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.array(lams)
    y = [r["delay"][0] if r["delay"] else np.nan for r in e1["rows"]]
    ye = [r["delay"][1] if r["delay"] else 0 for r in e1["rows"]]
    ax[0].errorbar(x, y, yerr=ye, fmt="o-", label="实测 T_grok − T_mem")
    ax[0].plot(x, y[2] * (x / x[2]) ** -1.0, "k--", lw=1, label="斜率 −1（范数分离定律）")
    ax[0].plot(x, y[2] * (x / x[2]) ** -0.64, "k:", lw=1, label="斜率 −0.64（经验拟合）")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("权重衰减 λ"); ax[0].set_ylabel("grokking 延迟（步）")
    ax[0].legend(fontsize=8); ax[0].set_title("E1：延迟 vs 权重衰减")
    for lam in e2["lams"]:
        rows = [r for r in e2["rows"] if r["lam"] == lam]
        ax[1].errorbar(alphas, [r["T_grok"][0] for r in rows], yerr=[r["T_grok"][1] for r in rows], fmt="o-", label=f"λ={lam}")
    ax[1].set_xscale("log"); ax[1].set_xlabel("初始化尺度 α"); ax[1].set_ylabel("T_grok（步）"); ax[1].legend(); ax[1].set_title("E2：初始化范数")
    for lam in [0.1, 0.3, 1.0, 3.0]:
        tr = np.array(E1runs[lam][0]["traj"])
        ax[2].plot(tr[:, 0], tr[:, 3], label=f"λ={lam}")
        if E1runs[lam][0]["T_grok"]:
            ax[2].plot(E1runs[lam][0]["T_grok"], E1runs[lam][0]["norm_grok"], "k*", ms=9)
    ax[2].set_xscale("log"); ax[2].set_xlabel("步数"); ax[2].set_ylabel("权重范数 ‖θ‖"); ax[2].legend(fontsize=8)
    ax[2].set_title("范数轨迹（★=grokking 时刻，种子 0）")
    fig.tight_layout(); fig.savefig(OUT / "frontier_grokking_laws.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fracs, [r["n_grokked"] / SEEDS for r in e3["rows"]], "o-", label="grokking 比例")
    ax2 = ax.twinx()
    ax2.plot(fracs, [r["T_grok"][0] if r["T_grok"] else np.nan for r in e3["rows"]], "s--", color="C1", label="T_grok")
    ax.set_xlabel("训练数据比例"); ax.set_ylabel("grokking 的种子比例"); ax2.set_ylabel("T_grok（步）"); ax.set_title("E3：数据比例")
    fig.tight_layout(); fig.savefig(OUT / "frontier_data_fraction.png"); plt.close(fig)

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "traj"}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    pth = OUT / "frontier.json"
    pth.write_text(json.dumps(strip(res), ensure_ascii=False, indent=2, default=_jsonable))
    print("saved", pth, res["runtime_sec"])
    print(json.dumps(strip({k: v for k, v in res.items() if k != "setup"}), default=_jsonable, ensure_ascii=False)[:6000])
