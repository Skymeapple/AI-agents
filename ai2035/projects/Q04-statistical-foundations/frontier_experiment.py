"""Q04 前沿实验：上下文学习（ICL）的统计学——任务多样性阈值、预训练样本双下降与贝叶斯基线。

模型：单层线性注意力在上下文线性回归上的等价形式 ŷ_q = x_qᵀ Γ h，h = (1/n) Σ_i y_i x_i（von Oswald 等 2022；
      Zhang–Frei–Bartlett 2023；Lu 等 2024 的约化）。Γ 为 d×d 矩阵，由预训练序列做最小二乘拟合（对 Γ 线性）。
任务：x ~ N(0, I_d)，y = wᵀx + ε，ε ~ N(0, σ²)；预训练任务池为 k 个固定向量 w_j ~ N(0, I/d)（k=∞ 表示每条序列新抽）。
理论（paper.md §4）：
  无限预训练序列时 Γ* = S C⁻¹，S = (1/k) Σ w_j w_jᵀ，C = (1+1/n) S + ((tr S + σ²)/n) I；
  新任务误差 E_new(Γ) = tr(I/d) − 2 tr(Γ/d) + tr(Γ C_new Γᵀ) + σ²，C_new = (1+1/n) I/d + ((1+σ²)/n) I。
  k<d 时 Γ* 只作用在任务子空间上 ⇒ E_new ≥ (1 − k/d)·1 + …（“记忆任务子空间”）；k→∞ 时收敛到标量最优。
  有限预训练序列 B 时，最小范数拟合 Γ（d² 个参数）在 B ≈ d² 处出现插值奇点（双下降）。
实验：E1 任务多样性 k 扫描（理论 vs 大 B 拟合 vs 贝叶斯岭 / dMMSE 基线）；E2 预训练样本数 B 扫描（最小范数 vs 验证集岭）。
运行：python ai2035/projects/Q04-statistical-foundations/frontier_experiment.py
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
D, NCTX, SIG = 8, 16, 0.25
SEEDS = 10
NTEST = 20000
KS = [1, 2, 4, 6, 8, 12, 16, 32, 64, 256, 1024, 0]  # 0 表示 k=∞


# ---------------------------------------------------------------- 数据
def sample_seqs(g, B, tasks=None):
    """返回 (H (B,d), Xq (B,d), yq (B,), W (B,d), X (B,n,d), Y (B,n))。tasks=None 表示每条序列新抽任务。"""
    if tasks is None:
        W = g.standard_normal((B, D)) / np.sqrt(D)
    else:
        W = tasks[g.integers(0, len(tasks), B)]
    X = g.standard_normal((B, NCTX, D))
    Y = np.einsum("bnd,bd->bn", X, W) + SIG * g.standard_normal((B, NCTX))
    Hh = np.einsum("bn,bnd->bd", Y, X) / NCTX
    Xq = g.standard_normal((B, D))
    yq = (Xq * W).sum(1) + SIG * g.standard_normal(B)
    return Hh, Xq, yq, W, X, Y


def features(Hh, Xq):
    return np.einsum("bi,bj->bij", Xq, Hh).reshape(len(Hh), -1)  # vec(x_q hᵀ)，ŷ = <Γ, x_q hᵀ>


def fit_gamma(g, B, tasks, lam=0.0):
    Hh, Xq, yq, *_ = sample_seqs(g, B, tasks)
    F = features(Hh, Xq)
    if lam == 0.0:
        coef = np.linalg.pinv(F) @ yq
    else:
        coef = np.linalg.solve(F.T @ F + lam * B * np.eye(D * D), F.T @ yq)
    return coef.reshape(D, D)


# ---------------------------------------------------------------- 理论
def gamma_star(S):
    C = (1 + 1 / NCTX) * S + ((np.trace(S) + SIG ** 2) / NCTX) * np.eye(D)
    return S @ np.linalg.inv(C)


def err_new_theory(G):
    Cn = (1 + 1 / NCTX) * np.eye(D) / D + ((1 + SIG ** 2) / NCTX) * np.eye(D)
    return np.trace(np.eye(D) / D) - 2 * np.trace(G) / D + np.trace(G @ Cn @ G.T) + SIG ** 2


def err_tasks_theory(G, tasks):
    """在预训练任务（均匀混合）上的误差。"""
    S = tasks.T @ tasks / len(tasks)
    C = (1 + 1 / NCTX) * S + ((np.trace(S) + SIG ** 2) / NCTX) * np.eye(D)
    return np.trace(S) - 2 * np.trace(G @ S) + np.trace(G @ C @ G.T) + SIG ** 2


# ---------------------------------------------------------------- 评估（蒙特卡洛）
def eval_mc(g, predictor, tasks=None, ntest=NTEST):
    Hh, Xq, yq, W, X, Y = sample_seqs(g, ntest, tasks)
    return float(np.mean((predictor(Hh, Xq, X, Y) - yq) ** 2))


def pred_gamma(G):
    return lambda Hh, Xq, X, Y: np.einsum("bi,ij,bj->b", Xq, G, Hh)


def pred_ridge_bayes():
    # 先验 w ~ N(0, I/d)，噪声 σ²：后验均值 (XᵀX + σ² d I)⁻¹ Xᵀy —— 新任务上的贝叶斯最优
    def f(Hh, Xq, X, Y):
        A = np.einsum("bni,bnj->bij", X, X) + SIG ** 2 * D * np.eye(D)
        b = np.einsum("bni,bn->bi", X, Y)
        w = np.linalg.solve(A, b[..., None])[..., 0]
        return (Xq * w).sum(1)
    return f


def pred_dmmse(tasks):
    # 离散先验（均匀落在 k 个预训练任务上）的后验均值 —— “记忆型”贝叶斯预测器
    def f(Hh, Xq, X, Y):
        R = Y[:, None, :] - np.einsum("bnd,kd->bkn", X, tasks)
        ll = -0.5 * (R ** 2).sum(2) / SIG ** 2
        ll -= ll.max(1, keepdims=True)
        P = np.exp(ll); P /= P.sum(1, keepdims=True)
        w = P @ tasks
        return (Xq * w).sum(1)
    return f


if __name__ == "__main__":
    plt = get_plt()
    res = {"setup": {"d": D, "n_context": NCTX, "sigma": SIG, "seeds": SEEDS, "n_test": NTEST, "ks(0=inf)": KS,
                     "task_prior": "N(0, I/d)", "null_error": 1 + SIG ** 2}}

    # ---------------- E1 任务多样性
    BIG = 40_000
    rows = []
    for k in KS:
        th_new, fit_new, mc_new, th_in, mc_in, ridge_new, dm_new, dm_in = ([] for _ in range(8))
        for s in range(SEEDS):
            g = rng(1000 + 37 * s + k)
            tasks = None if k == 0 else g.standard_normal((k, D)) / np.sqrt(D)
            S = np.eye(D) / D if k == 0 else tasks.T @ tasks / k
            Gs = gamma_star(S)
            th_new.append(err_new_theory(Gs))
            G = fit_gamma(g, BIG, tasks)
            fit_new.append(err_new_theory(G))
            mc_new.append(eval_mc(g, pred_gamma(G), None))
            if k > 0:
                th_in.append(err_tasks_theory(Gs, tasks))
                mc_in.append(eval_mc(g, pred_gamma(G), tasks))
                if k <= 64:
                    dm_new.append(eval_mc(g, pred_dmmse(tasks), None, ntest=4000))
                    dm_in.append(eval_mc(g, pred_dmmse(tasks), tasks, ntest=4000))
            if s < 3:
                ridge_new.append(eval_mc(g, pred_ridge_bayes(), None))
        rows.append({"k": "inf" if k == 0 else k, "new_theory": mean_ci(th_new), "new_fitted_formula": mean_ci(fit_new),
                     "new_mc": mean_ci(mc_new), "in_theory": mean_ci(th_in) if th_in else None,
                     "in_mc": mean_ci(mc_in) if mc_in else None,
                     "gap_new_minus_in_mc": mean_ci(np.array(mc_new) - np.array(mc_in)) if mc_in else None,
                     "dmmse_new": mean_ci(dm_new) if dm_new else None, "dmmse_in": mean_ci(dm_in) if dm_in else None,
                     "ridge_bayes_new": mean_ci(ridge_new),
                     "abs_err_theory_vs_mc": float(abs(np.mean(th_new) - np.mean(mc_new)))})
    lin_inf = rows[-1]["new_theory"][0]
    e1 = {"B_pretrain": BIG, "rows": rows, "lin_attn_new_error_k_inf_theory": lin_inf,
          "ridge_bayes_new_error": rows[-1]["ridge_bayes_new"],
          "max_abs_err_theory_vs_mc": float(max(r["abs_err_theory_vs_mc"] for r in rows))}
    # k>d 区域：超额误差 (E_new(k) − E_new(∞)) 对 k 的双对数斜率（理论预测 ≈ −1，Marchenko–Pastur 涨落）
    kk = np.array([r["k"] for r in rows if r["k"] != "inf" and r["k"] >= 2 * D], float)
    ex = np.array([r["new_theory"][0] - lin_inf for r in rows if r["k"] != "inf" and r["k"] >= 2 * D])
    exm = np.array([r["new_mc"][0] - rows[-1]["new_mc"][0] for r in rows if r["k"] != "inf" and r["k"] >= 2 * D])
    e1["excess_loglog_slope_theory_k>=2d"] = float(np.polyfit(np.log(kk), np.log(ex), 1)[0])
    e1["excess_loglog_slope_mc_k>=2d"] = float(np.polyfit(np.log(kk[exm > 0]), np.log(exm[exm > 0]), 1)[0]) if np.sum(exm > 0) >= 2 else None
    # k<d：新任务误差下界 (1 − k/d)·Var(wᵀx) + σ²（任务子空间外的方差无法被 Γ 触及）
    e1["subspace_lower_bound"] = {str(r["k"]): (1 - r["k"] / D) * 1.0 + SIG ** 2 for r in rows if r["k"] != "inf" and r["k"] < D}
    # 任务多样性阈值：新任务误差首次低于 ridge 线性注意力极限 × 1.1 的 k（逐种子用理论曲线）
    res["E1_task_diversity"] = e1
    t1 = time.time() - T0

    # ---------------- E2 预训练样本双下降（k=∞ 与 k=d）
    Bs = [16, 32, 48, 56, 60, 64, 68, 72, 80, 96, 128, 256, 512, 2048]
    lam_grid = np.logspace(-6, 0, 13)
    e2 = {"Bs": Bs, "n_params": D * D, "lambda_grid": lam_grid, "curves": {}}
    for kname in ["inf", str(D)]:
        mn, tu = np.zeros((SEEDS, len(Bs))), np.zeros((SEEDS, len(Bs)))
        for s in range(SEEDS):
            g = rng(5000 + 11 * s + (0 if kname == "inf" else 1))
            tasks = None if kname == "inf" else g.standard_normal((D, D)) / np.sqrt(D)
            for j, B in enumerate(Bs):
                Hh, Xq, yq, *_ = sample_seqs(g, B, tasks)
                F = features(Hh, Xq)
                G0 = (np.linalg.pinv(F) @ yq).reshape(D, D)
                mn[s, j] = err_new_theory(G0)
                # 验证集（同分布的 B 条新序列）选 λ
                Hv, Xv, yv, *_ = sample_seqs(g, max(B, 32), tasks)
                Fv = features(Hv, Xv)
                best, bestv = None, np.inf
                for lam in lam_grid:
                    c = np.linalg.solve(F.T @ F + lam * B * np.eye(D * D), F.T @ yq)
                    v = np.mean((Fv @ c - yv) ** 2)
                    if v < bestv:
                        bestv, best = v, c
                tu[s, j] = err_new_theory(best.reshape(D, D))
        med = np.median(mn, 0)
        jpk = int(np.argmax(med))
        dif = np.diff(np.median(tu, 0))
        e2["curves"][kname] = {"minnorm_median": med, "minnorm_mean_ci": [mean_ci(mn[:, j]) for j in range(len(Bs))],
                               "tuned_median": np.median(tu, 0), "tuned_mean_ci": [mean_ci(tu[:, j]) for j in range(len(Bs))],
                               "peak_B_median_curve": Bs[jpk],
                               "peak_B_per_seed": [Bs[int(np.argmax(mn[s]))] for s in range(SEEDS)],
                               "tuned_max_increase_between_B": float(dif.max()),
                               "minnorm_at_64_over_at_2048": float(med[Bs.index(64)] / med[-1]),
                               "wilcoxon_minnorm_vs_tuned_at_64": float(stats.wilcoxon(mn[:, Bs.index(64)], tu[:, Bs.index(64)]).pvalue)}
    res["E2_pretrain_samples"] = e2
    res["runtime_sec"] = {"E1": t1, "total": time.time() - T0, "cpu_process_time": time.process_time()}

    # ---------------- 图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    kx = np.array([r["k"] if r["k"] != "inf" else 4096 for r in rows], float)
    ax[0].plot(kx, [r["new_theory"][0] for r in rows], "k-", label="理论 E_new（线性注意力，B→∞）")
    ax[0].errorbar(kx, [r["new_mc"][0] for r in rows], yerr=[r["new_mc"][1] for r in rows], fmt="o", label="拟合 Γ（B=4×10⁴）新任务")
    ax[0].plot(kx[:-1], [r["in_mc"][0] for r in rows[:-1]], "s--", label="拟合 Γ 在预训练任务上")
    ax[0].plot([r["k"] for r in rows if r["dmmse_new"]], [r["dmmse_new"][0] for r in rows if r["dmmse_new"]], "^-", label="dMMSE（记忆型贝叶斯）新任务")
    ax[0].axhline(rows[-1]["ridge_bayes_new"][0], color="C3", ls=":", label="贝叶斯岭（新任务最优）")
    ax[0].axvline(D, color="gray", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("预训练任务数 k（最右点=∞）"); ax[0].set_ylabel("ICL 误差")
    ax[0].legend(fontsize=7); ax[0].set_title(f"任务多样性（d={D}, n={NCTX}）")
    for kname, lab in [("inf", "k=∞"), (str(D), f"k=d={D}")]:
        c = e2["curves"][kname]
        ax[1].plot(Bs, c["minnorm_median"], "o-", label=f"最小范数拟合 {lab}")
        ax[1].plot(Bs, c["tuned_median"], "s--", label=f"验证集岭 {lab}")
    ax[1].axvline(D * D, color="gray", ls=":", label="B = d² = 参数数")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("预训练序列数 B"); ax[1].set_ylabel("新任务 ICL 误差（中位数）")
    ax[1].legend(fontsize=7); ax[1].set_title("预训练样本的双下降")
    fig.tight_layout(); fig.savefig(OUT / "frontier_icl.png"); plt.close(fig)

    pth = OUT / "frontier.json"
    pth.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_jsonable))
    print("saved", pth, res["runtime_sec"])
    for r in rows:
        print(r["k"], r["new_theory"][0], r["new_mc"], r["in_mc"], r["dmmse_new"], r["dmmse_in"], r["ridge_bayes_new"][0])
    print({k: v for k, v in e1.items() if k != "rows"})
    for kname in e2["curves"]:
        c = e2["curves"][kname]; print(kname, c["peak_B_median_curve"], c["peak_B_per_seed"], c["tuned_max_increase_between_B"], c["minnorm_at_64_over_at_2048"], np.round(c["minnorm_median"], 3), np.round(c["tuned_median"], 3))
