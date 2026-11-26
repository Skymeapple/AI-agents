"""Q04 机器学习的统计学基础 —— 过参数化下的泛化：双下降、岭正则与良性过拟合。

设置：x ~ N(0, I_20)，n=200 训练样本；教师 f(x) = u·x + 0.5 tanh(2 v·x)（标准化为单位方差），
      标签 y = f(x) + σ ε，σ ∈ {0, 0.25, 0.5, 1.0}。
模型：随机 ReLU 特征 φ(x)=ReLU(Wx/√d)/√p，特征数 p 从 0.1n 到 20n（嵌套子集）。
估计器：最小范数最小二乘（p<n 时即普通最小二乘，p≥n 时为最小范数插值）；
        固定岭 λ ∈ {1e-3, 1e-2, 1e-1}；验证集（n_val=500）上从 13 个 λ 中选优的“最优岭”。
指标：超额风险 E[(f̂(x)-f(x))²]（对无噪声 f，在 2000 个测试点上估计）；训练误差。
种子：20（每个种子重抽数据、特征与噪声）。

运行：python ai2035/projects/Q04-statistical-foundations/experiment.py
"""
from __future__ import annotations

import os
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
n, n_val, n_te = 200, 500, 2000
DIMS = [20, 100]  # d=20 为主实验；d=100（n/d=2）为高维对照（探索性，见报告）
RATIOS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 2, 3, 5, 10, 20])
PS = np.round(RATIOS * n).astype(int)
SIGMAS = [0.0, 0.25, 0.5, 1.0]
FIXED_L = [1e-3, 1e-2, 1e-1]
LGRID = np.logspace(-6, 2, 13)
SEEDS = 20

def make_teacher(dim):
    gT = rng(dim)
    u = gT.standard_normal(dim); u /= np.linalg.norm(u)
    v = gT.standard_normal(dim); v /= np.linalg.norm(v)
    xr = gT.standard_normal((200000, dim))
    fr = xr @ u + 0.5 * np.tanh(2 * xr @ v)
    mu, sd = fr.mean(), fr.std()
    return lambda X: (X @ u + 0.5 * np.tanh(2 * X @ v) - mu) / sd


def run_seed(s, d):
    f = make_teacher(d)
    g = rng(100 + s + 1000 * d)
    Xtr, Xva, Xte = g.standard_normal((n, d)), g.standard_normal((n_val, d)), g.standard_normal((n_te, d))
    ftr, fva, fte = f(Xtr), f(Xva), f(Xte)
    eps_tr, eps_va = g.standard_normal(n), g.standard_normal(n_val)
    W = g.standard_normal((d, PS.max()))
    Ftr_all = np.maximum(Xtr @ W / np.sqrt(d), 0)
    Fva_all = np.maximum(Xva @ W / np.sqrt(d), 0)
    Fte_all = np.maximum(Xte @ W / np.sqrt(d), 0)
    ns = len(SIGMAS)
    out = {"mn": np.zeros((ns, len(PS))), "mn_train": np.zeros((ns, len(PS))),
           "fixed": np.zeros((len(FIXED_L), ns, len(PS))), "tuned": np.zeros((ns, len(PS))),
           "tuned_lambda": np.zeros((ns, len(PS))), "mn_norm": np.zeros((ns, len(PS)))}
    for j, p in enumerate(PS):
        sc = 1 / np.sqrt(p)
        Ftr, Fva, Fte = Ftr_all[:, :p] * sc, Fva_all[:, :p] * sc, Fte_all[:, :p] * sc
        U, S, Vt = np.linalg.svd(Ftr, full_matrices=False)
        tol = S.max() * max(Ftr.shape) * np.finfo(float).eps
        for k, sig in enumerate(SIGMAS):
            ytr = ftr + sig * eps_tr
            yva = fva + sig * eps_va
            Uy = U.T @ ytr
            # 最小范数解
            inv = np.where(S > tol, 1 / np.where(S > tol, S, 1), 0)
            a = Vt.T @ (inv * Uy)
            out["mn"][k, j] = np.mean((Fte @ a - fte) ** 2)
            out["mn_train"][k, j] = np.mean((Ftr @ a - ytr) ** 2)
            out["mn_norm"][k, j] = np.linalg.norm(a)

            def ridge(lam):
                return Vt.T @ (S / (S ** 2 + n * lam) * Uy)
            for q, lam in enumerate(FIXED_L):
                out["fixed"][q, k, j] = np.mean((Fte @ ridge(lam) - fte) ** 2)
            vals = [np.mean((Fva @ ridge(lam) - yva) ** 2) for lam in LGRID]
            best = LGRID[int(np.argmin(vals))]
            out["tuned"][k, j] = np.mean((Fte @ ridge(best) - fte) ** 2)
            out["tuned_lambda"][k, j] = best
    return out


def analyse(d, plt):
    t0 = time.time()
    runs = [run_seed(s, d) for s in range(SEEDS)]
    t_run = time.time() - t0
    MN = np.array([r["mn"] for r in runs])            # (S, sig, p)
    MNtr = np.array([r["mn_train"] for r in runs])
    FX = np.array([r["fixed"] for r in runs])         # (S, lam, sig, p)
    TU = np.array([r["tuned"] for r in runs])
    TL = np.array([r["tuned_lambda"] for r in runs])
    NORM = np.array([r["mn_norm"] for r in runs])
    j1 = int(np.where(RATIOS == 1.0)[0][0])
    j20 = int(np.where(RATIOS == 20)[0][0])
    under = RATIOS < 0.9
    over = RATIOS >= 2

    res = {"setup": {"d": d, "n": n, "n_val": n_val, "n_test": n_te, "ratios": RATIOS, "ps": PS, "sigmas": SIGMAS,
                     "fixed_lambdas": FIXED_L, "lambda_grid": LGRID, "seeds": SEEDS,
                     "metric": "超额风险 E[(f_hat - f)^2]（无噪声目标，Var f = 1）"},
           "by_sigma": {}}
    peak_heights = []
    for k, sig in enumerate(SIGMAS):
        mn = MN[:, k]
        med = np.median(mn, 0)
        # 峰位置：中位数曲线的最大值（排除 p/n=0.1 端点——欠参数端误差也大）
        jpk = int(np.argmax(med[1:])) + 1
        per_seed_peak = RATIOS[1 + np.argmax(mn[:, 1:], 1)]
        best_under = mn[:, under].min(1)
        best_over = mn[:, over].min(1)
        w = stats.wilcoxon(best_over, best_under) if sig > 0 or np.any(best_over != best_under) else None
        tuned = TU[:, k]
        # 最优岭单调性：相邻 p 之间均值增幅的最大值，及其是否超过配对差的 95% CI
        dif = np.diff(tuned, axis=1)
        worst = int(np.argmax(dif.mean(0)))
        worst_ci = mean_ci(dif[:, worst])
        res["by_sigma"][str(sig)] = {
            "minnorm_mean": mn.mean(0), "minnorm_ci": [mean_ci(mn[:, j])[1] for j in range(len(PS))],
            "minnorm_median": med, "minnorm_train_mean": MNtr[:, k].mean(0),
            "minnorm_coef_norm_median": np.median(NORM[:, k], 0),
            "peak_ratio_median_curve": float(RATIOS[jpk]),
            "peak_ratio_per_seed": per_seed_peak, "frac_seeds_peak_in_0.9_1.1": float(np.mean((per_seed_peak >= 0.9) & (per_seed_peak <= 1.1))),
            "err_at_interp_mean_ci": mean_ci(mn[:, j1]), "err_at_interp_median": float(np.median(mn[:, j1])),
            "best_underparam_mean_ci": mean_ci(best_under), "best_overparam_mean_ci": mean_ci(best_over),
            "over_vs_under_wilcoxon_p": float(w.pvalue) if w is not None else None,
            "frac_seeds_over_better": float(np.mean(best_over < best_under)),
            "fixed_ridge_mean": {str(l): FX[:, q, k].mean(0) for q, l in enumerate(FIXED_L)},
            "fixed_ridge_at_interp": {str(l): mean_ci(FX[:, q, k, j1]) for q, l in enumerate(FIXED_L)},
            "tuned_mean": tuned.mean(0), "tuned_ci": [mean_ci(tuned[:, j])[1] for j in range(len(PS))],
            "tuned_at_interp": mean_ci(tuned[:, j1]),
            "tuned_max_step_increase": {"between_ratios": [float(RATIOS[worst]), float(RATIOS[worst + 1])],
                                        "mean_ci": worst_ci},
            "tuned_lambda_median": np.median(TL[:, k], 0),
            "benign_at_20n": {"minnorm": mean_ci(mn[:, j20]), "tuned": mean_ci(tuned[:, j20]),
                              "ratio_minnorm_over_tuned": mean_ci(mn[:, j20] / tuned[:, j20]),
                              "minnorm_train_mse": mean_ci(MNtr[:, k, j20]),
                              "excess_over_noise_var": (mean_ci(mn[:, j20] / sig ** 2) if sig > 0 else None),
                              "null_predictor_risk": 1.0},
        }
        peak_heights.append(mn[:, j1])
    # 峰高随噪声单调增：Spearman（逐种子堆叠）
    ph = np.concatenate(peak_heights); sg = np.repeat(SIGMAS, SEEDS)
    rho = stats.spearmanr(sg, ph)
    res["peak_height_vs_sigma_spearman"] = {"rho": float(rho[0]), "p": float(rho[1])}
    res["runtime_sec"] = {"run": t_run}

    # ---------------------------------------------------------------- 图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for k, sig in enumerate(SIGMAS):
        ax[0].plot(RATIOS, np.median(MN[:, k], 0), "o-", ms=3, label=f"最小范数 σ={sig}")
    ax[0].axvline(1, color="k", ls=":", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("p / n"); ax[0].set_ylabel("超额风险（20 种子中位数）")
    ax[0].set_title("双下降：插值阈值 p=n 处的峰"); ax[0].legend(fontsize=8)
    k = SIGMAS.index(0.5)
    ax[1].plot(RATIOS, np.median(MN[:, k], 0), "o-", ms=3, label="最小范数（λ→0）")
    for q, l in enumerate(FIXED_L):
        ax[1].plot(RATIOS, np.median(FX[:, q, k], 0), "-", label=f"岭 λ={l:g}")
    ax[1].plot(RATIOS, np.median(TU[:, k], 0), "k--", lw=2, label="验证集最优岭")
    ax[1].axhline(1.0, color="gray", lw=0.8, ls=":")
    ax[1].axvline(1, color="k", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("p / n"); ax[1].set_title("σ=0.5：正则化消除峰")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"double_descent_d{d}.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, sig in enumerate(SIGMAS):
        ax.plot(RATIOS, np.median(NORM[:, k], 0), "o-", ms=3, label=f"σ={sig}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.axvline(1, color="k", ls=":", lw=1)
    ax.set_xlabel("p / n"); ax.set_ylabel("最小范数解 ‖a‖（中位数）"); ax.set_title("插值阈值处系数范数爆炸"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / f"coef_norm_d{d}.png"); plt.close(fig)

    return res, MN, TU


if __name__ == "__main__":
    plt = get_plt()
    allres, curves = {}, {}
    for d in DIMS:
        r, MN, TU = analyse(d, plt)
        allres[f"d{d}"] = r
        curves[d] = (MN, TU)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d in DIMS:
        MN, TU = curves[d]
        for k, sig in enumerate(SIGMAS):
            if sig == 0:
                continue
            ax.plot(RATIOS[RATIOS >= 1.25], np.median(MN[:, k] / TU[:, k], 0)[RATIOS >= 1.25], "o-" if d == 20 else "s--", ms=3, label=f"d={d}, σ={sig}")
    ax.axhline(1, color="k", lw=0.8); ax.axhline(1.25, color="gray", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("p / n"); ax.set_ylabel("最小范数插值 / 最优岭 超额风险比（中位数）")
    ax.set_title("良性 vs 温和过拟合"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "benign_ratio.png"); plt.close(fig)
    allres["runtime_sec_total"] = time.time() - T0
    p = save_results(__file__, allres)
    print("saved", p, time.time() - T0)
    for d in DIMS:
      for sig in SIGMAS:
        b = allres[f"d{d}"]["by_sigma"][str(sig)]
        print(d, sig, b["peak_ratio_median_curve"], b["frac_seeds_peak_in_0.9_1.1"], b["err_at_interp_median"], b["best_underparam_mean_ci"], b["best_overparam_mean_ci"], b["benign_at_20n"]["ratio_minnorm_over_tuned"], b["tuned_max_step_increase"])
