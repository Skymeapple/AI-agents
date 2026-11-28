"""Q10 知识与数据融合：物理先验 / 符号约束带来的样本效率，以及先验错误时的代价。

(a)+(c) 物理先验：真实系统 x'' + 2γx' + ω0²x + βx³ = 0（γ=0.15, ω0=2, x(0)=1, x'(0)=0），
    在 t∈[0,6] 取 n∈{6,10,20,40,80} 个带噪观测（σ=0.05）。β=0 时“线性阻尼振子”先验正确；β∈{0.5,2} 时先验错误；
    另加一种错设：未建模外力 +0.6cos(0.8t)（持续性误差，不随阻尼消失）。
    模型：纯数据 GP(RBF)；物理参数拟合（线性阻尼振子闭式解 A e^{-γt}cos(ωt+φ)，多起点最小二乘）；
          混合残差模型（物理拟合 + GP 拟合残差）。指标：内插 [0,6] 与外推 (6,12] 的 RMSE；偏差²/方差分解。
(b)+(c) 符号约束：单调性先验。logit = 2tanh(1.5x1)+0.5x1 + cos(2x2)，x3..x5 无关；
    HistGradientBoosting 无约束 vs 正确约束(x1↑) vs 错误约束(x1↑ 且 x2↑；x2 实为非单调)。
    n∈{50,100,200,500,2000}；指标：测试 log-loss、相对 Bayes 的超额损失、单调性违背率。
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, save_results, mean_ci, plt, results_dir

import numpy as np
from scipy import stats, optimize
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
OUT = results_dir(__file__)

# ======================= (a) physics prior =======================
SEEDS = 10
NS = [6, 10, 20, 40, 80]
CASES = {"β=0（先验正确）": (0.0, 0.0), "β=0.5（弱错设）": (0.5, 0.0), "β=2（强错设）": (2.0, 0.0),
         "未建模外力 F=0.6": (0.0, 0.6)}
BETAS = list(CASES)
GAM, W0, SIG, TOBS, TEXT = 0.15, 2.0, 0.05, 6.0, 12.0
tg = np.linspace(0, TEXT, 1201); i_in = tg <= TOBS; i_ex = tg > TOBS


def truth(case):
    beta, F = CASES[case]
    sol = solve_ivp(lambda t, z: [z[1], -2 * GAM * z[1] - W0 ** 2 * z[0] - beta * z[0] ** 3 + F * np.cos(0.8 * t)],
                    (0, TEXT), [1.0, 0.0], t_eval=tg, rtol=1e-10, atol=1e-12)
    return sol.y[0]


TRUE = {b: truth(b) for b in BETAS}


def osc(p, t):
    A, gam, w, phi = p
    return A * np.exp(-gam * t) * np.cos(w * t + phi)


def fit_physics(t, y):
    best = None
    for w0 in np.linspace(0.5, 5.0, 10):
        for phi0 in [0.0, np.pi / 2]:
            r = optimize.least_squares(lambda p: osc(p, t) - y, [1.0, 0.1, w0, phi0],
                                       bounds=([0, 0, 0.1, -np.pi * 2], [5, 3, 8, np.pi * 2]))
            if best is None or r.cost < best.cost:
                best = r
    return best.x


def gp():
    return GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-2, (1e-6, 1)),
                                    normalize_y=True, n_restarts_optimizer=2, random_state=0)


MODELS = ["纯数据 GP", "物理拟合(线性振子)", "混合：物理+GP残差"]
preds = {b: {n: {m: [] for m in MODELS} for n in NS} for b in BETAS}
for b in BETAS:
    f = TRUE[b]
    for n in NS:
        for s in range(SEEDS):
            g = rng(100 * n + s + 1000 * BETAS.index(b))
            t = np.sort(g.uniform(0, TOBS, n)); y = np.interp(t, tg, f) + SIG * g.standard_normal(n)
            pg = gp().fit(t[:, None], y).predict(tg[:, None])
            th = fit_physics(t, y); pp = osc(th, tg)
            res = y - osc(th, t)
            ph = pp + gp().fit(t[:, None], res).predict(tg[:, None])
            for m, p in zip(MODELS, [pg, pp, ph]):
                preds[b][n][m].append(p)
    print("beta", b, time.time() - T0, flush=True)

A = {}
for b in BETAS:
    A[b] = {}
    f = TRUE[b]
    for n in NS:
        A[b][n] = {}
        for m in MODELS:
            P_ = np.array(preds[b][n][m])
            ri = np.sqrt(((P_[:, i_in] - f[i_in]) ** 2).mean(1)); re = np.sqrt(((P_[:, i_ex] - f[i_ex]) ** 2).mean(1))
            mean_pred = P_.mean(0)
            A[b][n][m] = {"rmse_interp": dict(zip(["mean", "ci95"], mean_ci(ri))),
                          "rmse_extrap": dict(zip(["mean", "ci95"], mean_ci(re))),
                          "bias2_interp": float(((mean_pred[i_in] - f[i_in]) ** 2).mean()),
                          "var_interp": float(P_[:, i_in].var(0, ddof=1).mean()),
                          "bias2_extrap": float(((mean_pred[i_ex] - f[i_ex]) ** 2).mean()),
                          "var_extrap": float(P_[:, i_ex].var(0, ddof=1).mean()),
                          "_ri": ri, "_re": re}
        # paired tests (log RMSE) of interest
        for (m1, m2) in [("物理拟合(线性振子)", "纯数据 GP"), ("混合：物理+GP残差", "物理拟合(线性振子)"),
                         ("混合：物理+GP残差", "纯数据 GP")]:
            for key in ["_ri", "_re"]:
                d = np.log10(A[b][n][m1][key]) - np.log10(A[b][n][m2][key])
                A[b][n][f"{m1} vs {m2} [{'内插' if key == '_ri' else '外推'}] Δlog10RMSE"] = \
                    dict(zip(["mean", "ci95"], mean_ci(d))) | {"p_paired": float(stats.ttest_rel(np.log10(A[b][n][m1][key]), np.log10(A[b][n][m2][key])).pvalue)}
    for n in NS:
        for m in MODELS:
            A[b][n][m].pop("_ri"); A[b][n][m].pop("_re")
# physics bias floor: ratio of rmse at n=80 vs n=20
floor = {b: {m: A[b][80][m]["rmse_interp"]["mean"] / A[b][20][m]["rmse_interp"]["mean"] for m in MODELS} for b in BETAS}

# ======================= (b) monotonic constraint =======================
SEEDS_B = 20
NB = [50, 100, 200, 500, 2000]


def logit_true(X):
    return 2 * np.tanh(1.5 * X[:, 0]) + 0.5 * X[:, 0] + np.cos(2 * X[:, 1])


def gen_b(n, g):
    X = g.uniform(-2, 2, (n, 5)); p = 1 / (1 + np.exp(-logit_true(X)))
    return X, (g.uniform(size=n) < p).astype(int), p


CST = {"无约束": [0, 0, 0, 0, 0], "正确约束 (x1↑)": [1, 0, 0, 0, 0], "错误约束 (x1↑, x2↑)": [1, 1, 0, 0, 0]}
rawB = {n: {c: {"logloss": [], "viol": [], "excess": []} for c in CST} for n in NB}
for n in NB:
    for s in range(SEEDS_B):
        g = rng(5000 + 7 * n + s)
        X, y, _ = gen_b(n, g); Xt, yt, pt = gen_b(5000, g)
        bayes = log_loss(yt, pt)
        Xp = Xt.copy(); Xp[:, 0] = np.minimum(2, Xp[:, 0] + 0.5)
        for c, cst in CST.items():
            m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=10,
                                               monotonic_cst=cst, early_stopping=False, random_state=s).fit(X, y)
            pr = np.clip(m.predict_proba(Xt)[:, 1], 1e-6, 1 - 1e-6)
            ll = log_loss(yt, pr)
            viol = np.mean(m.predict_proba(Xp)[:, 1] < pr - 1e-9)
            rawB[n][c]["logloss"].append(ll); rawB[n][c]["excess"].append(ll - bayes); rawB[n][c]["viol"].append(viol)
B = {}
for n in NB:
    B[n] = {}
    base = np.array(rawB[n]["无约束"]["logloss"])
    for c in CST:
        e = {k: dict(zip(["mean", "ci95"], mean_ci(v))) for k, v in rawB[n][c].items()}
        if c != "无约束":
            a = np.array(rawB[n][c]["logloss"])
            e["logloss_diff_vs_unconstrained"] = dict(zip(["mean", "ci95"], mean_ci(a - base))) | {
                "p_paired": float(stats.ttest_rel(a, base).pvalue)}
        B[n][c] = e
print("b", time.time() - T0, flush=True)

# ======================= figures =======================
P = plt()
fig, axes = P.subplots(2, 4, figsize=(21, 8.5))
for col, b in enumerate(BETAS):
    for row, key, tt in [(0, "rmse_interp", "内插 t∈[0,6]"), (1, "rmse_extrap", "外推 t∈(6,12]")]:
        ax = axes[row, col]
        for m in MODELS:
            mu = [A[b][n][m][key]["mean"] for n in NS]; ci = [A[b][n][m][key]["ci95"] for n in NS]
            ax.errorbar(NS, mu, yerr=ci, marker="o", capsize=3, label=m)
        ax.axhline(SIG, color="k", ls=":", lw=1, label="观测噪声 σ")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("观测点数 n"); ax.set_ylabel("RMSE")
        ax.set_title(f"{b}：{tt}", fontsize=9); ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "physics_prior.png"); P.close(fig)

fig, axes = P.subplots(1, 4, figsize=(21, 3.8))
for ax, b in zip(axes, BETAS):
    ax.plot(tg, TRUE[b], "k", lw=2, label="真实轨迹")
    for m in MODELS:
        ax.plot(tg, preds[b][10][m][0], lw=1.2, label=m)
    ax.axvline(TOBS, color="grey", ls="--"); ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"{b}，n=10，种子 0", fontsize=9); ax.set_xlabel("t")
axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "trajectories.png"); P.close(fig)

fig, axes = P.subplots(1, 2, figsize=(12, 4.2))
for c in CST:
    mu = [B[n][c]["excess"]["mean"] for n in NB]; ci = [B[n][c]["excess"]["ci95"] for n in NB]
    axes[0].errorbar(NB, mu, yerr=ci, marker="o", capsize=3, label=c)
    mu = [B[n][c]["viol"]["mean"] for n in NB]; ci = [B[n][c]["viol"]["ci95"] for n in NB]
    axes[1].errorbar(NB, mu, yerr=ci, marker="o", capsize=3, label=c)
for ax, yl in zip(axes, ["相对 Bayes 的超额 log-loss", "单调性违背率（x1+0.5 后概率下降）"]):
    ax.set_xscale("log"); ax.set_xlabel("训练样本数 n"); ax.set_ylabel(yl); ax.grid(alpha=.3)
axes[0].set_yscale("log"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "monotonic.png"); P.close(fig)

runtime = time.time() - T0
save_results(__file__, {"config": {"a": {"seeds": SEEDS, "n": NS, "cases(beta,F)": CASES, "forcing": "F cos(0.8 t)", "gamma": GAM, "omega0": W0, "noise": SIG,
                                         "t_obs": [0, TOBS], "t_extrap": [TOBS, TEXT],
                                         "gp": "Const*RBF+White, normalize_y, 2 restarts",
                                         "physics": "A exp(-γt) cos(ωt+φ), least squares, 20 starts"},
                                   "b": {"seeds": SEEDS_B, "n": NB, "hgb": "max_iter100 lr0.1 leaves15 min_leaf10",
                                         "constraints": CST, "violation_probe": "x1 += 0.5 (clipped at 2)"}},
                        "a_physics": A, "a_rmse_ratio_n80_over_n20": floor, "b_monotonic": B, "runtime_sec": runtime})
print("runtime", runtime)
