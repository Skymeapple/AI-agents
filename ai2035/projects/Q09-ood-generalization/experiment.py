"""Q09 分布外泛化与外推：三类分布偏移的机理实验。

(a) 外推：x~U[-1,1] 训练，在 [-5,5] 评估；目标 f=x^2 与 f=sin(2πx)。
    模型：ReLU-MLP、tanh-MLP、GP(RBF)、高次多项式(9 次，错误先验)、"正确归纳偏置"模型（x^2: 2 次多项式；sin: 傅里叶基 k=1..4）。
(b) 协变量偏移（Shimodaira 2000 设定）：f=-x+x^3, ε~N(0,0.3^2)，训练 x~N(0.5,0.5^2)，测试 x~N(0,0.3^2)。
    线性模型（错设）vs 三次模型（正确设定）；权重 w^λ, λ∈{0,0.5,1}（真密度比），以及用逻辑回归估计的密度比 (λ=1)。
(c) 组合泛化：属性 a,b∈{0..5}，x = U_a + V_b + s·W_ab + 噪声（d=30），留出 12 个组合；交互强度 s∈{0,0.5,1,2}。
    模型：联合 36 类 LR；因子化 LR（分别预测 a、b）；因子化 MLP；因子化 kNN。
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, save_results, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
OUT = results_dir(__file__)

# ======================= (a) extrapolation =======================
SEEDS_A = 8
TARGETS = {"x^2": lambda x: x ** 2, "sin(2πx)": lambda x: np.sin(2 * np.pi * x)}
BINS = [(0, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4)]   # distance outside [-1,1]
BIN_NAMES = ["域内", "(0,0.5]", "(0.5,1]", "(1,2]", "(2,4]"]
xg = np.linspace(-5, 5, 2001); dist = np.maximum(0, np.abs(xg) - 1)


def poly_feats(x, deg):
    return np.vstack([x ** k for k in range(1, deg + 1)]).T


def fourier_feats(x, K=4):
    return np.hstack([np.c_[np.sin(k * np.pi * x), np.cos(k * np.pi * x)] for k in range(1, K + 1)])


def fit_predict(model, x, y, xq, target):
    if model == "ReLU-MLP" or model == "tanh-MLP":
        m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu" if model == "ReLU-MLP" else "tanh", solver="lbfgs",
                         max_iter=3000, alpha=1e-4, random_state=int(x[0] * 1e6) % 10000)
        return m.fit(x[:, None], y).predict(xq[:, None])
    if model == "GP-RBF":
        k = ConstantKernel(1.0) * RBF(0.5) + WhiteKernel(0.01)
        return GaussianProcessRegressor(k, normalize_y=True).fit(x[:, None], y).predict(xq[:, None])
    if model == "9次多项式":
        return Ridge(1e-6).fit(poly_feats(x, 9), y).predict(poly_feats(xq, 9))
    if model == "正确归纳偏置":
        F = (lambda z: poly_feats(z, 2)) if target == "x^2" else fourier_feats
        return Ridge(1e-6).fit(F(x), y).predict(F(xq))


MODELS_A = ["ReLU-MLP", "tanh-MLP", "GP-RBF", "9次多项式", "正确归纳偏置"]
rawA = {t: {m: [] for m in MODELS_A} for t in TARGETS}
linR2 = {t: [] for t in TARGETS}; tanh_slope = {t: [] for t in TARGETS}
example_pred = {}
for t, f in TARGETS.items():
    for s in range(SEEDS_A):
        g = rng(s)
        x = g.uniform(-1, 1, 200); y = f(x) + 0.05 * g.standard_normal(200)
        for m in MODELS_A:
            p = fit_predict(m, x, y, xg, t)
            err = (p - f(xg)) ** 2
            rawA[t][m].append([np.sqrt(err[(dist == 0)].mean()) if lo == 0 and hi == 0 else
                               np.sqrt(err[(dist > lo) & (dist <= hi)].mean()) for lo, hi in BINS])
            if m == "ReLU-MLP":
                for side in [xg >= 2, xg <= -2]:
                    lr_ = LinearRegression().fit(xg[side][:, None], p[side])
                    linR2[t].append(lr_.score(xg[side][:, None], p[side]))
            if m == "tanh-MLP":
                for side in [xg >= 4, xg <= -4]:
                    tanh_slope[t].append(abs(np.polyfit(xg[side], p[side], 1)[0]))
            if s == 0:
                example_pred[(t, m)] = p
A = {}
for t in TARGETS:
    A[t] = {}
    for m in MODELS_A:
        R = np.array(rawA[t][m])
        A[t][m] = {BIN_NAMES[j]: dict(zip(["mean", "ci95"], mean_ci(np.log10(R[:, j])))) for j in range(len(BINS))}
        A[t][m]["rmse_mean"] = {BIN_NAMES[j]: float(R[:, j].mean()) for j in range(len(BINS))}
    A[t]["ReLU_linear_fit_R2_|x|>=2"] = dict(zip(["mean", "ci95"], mean_ci(linR2[t]))) | {"min": float(np.min(linR2[t]))}
    A[t]["tanh_abs_slope_|x|>=4"] = dict(zip(["mean", "ci95"], mean_ci(tanh_slope[t])))
    # paired test: ReLU vs correct bias at far bin
    a1 = np.log10(np.array(rawA[t]["ReLU-MLP"])[:, -1]); a2 = np.log10(np.array(rawA[t]["正确归纳偏置"])[:, -1])
    A[t]["far_bin_log10_ReLU_minus_correct"] = dict(zip(["mean", "ci95"], mean_ci(a1 - a2))) | {"p_paired": float(stats.ttest_rel(a1, a2).pvalue)}
print("a", time.time() - T0, flush=True)

# ======================= (b) covariate shift =======================
SEEDS_B = 100
NB = [30, 100, 300, 1000]
fB = lambda x: -x + x ** 3
mu0, s0, mu1, s1 = 0.5, 0.5, 0.0, 0.3
xt_test = None


def dens(x, mu, s):
    return stats.norm.pdf(x, mu, s)


VARIANTS = ["λ=0（不加权）", "λ=0.5", "λ=1（真密度比）", "λ=1（估计密度比）"]
rawB = {spec: {n: {v: [] for v in VARIANTS} for n in NB} for spec in ["线性(错设)", "三次(正确设定)"]}
ess = {n: [] for n in NB}
for n in NB:
    for s in range(SEEDS_B):
        g = rng(7000 + 13 * n + s)
        x = mu0 + s0 * g.standard_normal(n); y = fB(x) + 0.3 * g.standard_normal(n)
        xte = mu1 + s1 * g.standard_normal(5000); yte = fB(xte) + 0.3 * g.standard_normal(5000)
        xu = mu1 + s1 * g.standard_normal(n)     # unlabeled test-domain sample for ratio estimation
        w_true = dens(x, mu1, s1) / dens(x, mu0, s0)
        clf = LogisticRegression(C=1.0).fit(np.r_[x, xu][:, None] ** [1, 2], np.r_[np.zeros(n), np.ones(n)])
        pr = clf.predict_proba((x[:, None]) ** [1, 2])[:, 1]; w_est = pr / (1 - pr)
        ess[n].append(w_true.sum() ** 2 / (w_true ** 2).sum() / n)
        for spec, deg in [("线性(错设)", 1), ("三次(正确设定)", 3)]:
            Ftr, Fte = poly_feats(x, deg), poly_feats(xte, deg)
            for v, w in zip(VARIANTS, [np.ones(n), w_true ** 0.5, w_true, w_est]):
                m = LinearRegression().fit(Ftr, y, sample_weight=w)
                rawB[spec][n][v].append(np.mean((m.predict(Fte) - yte) ** 2))
B = {}
for spec in rawB:
    B[spec] = {}
    for n in NB:
        B[spec][n] = {}
        base = np.array(rawB[spec][n]["λ=0（不加权）"])
        for v in VARIANTS:
            a = np.array(rawB[spec][n][v]); e = dict(zip(["mean", "ci95"], mean_ci(a)))
            e["median"] = float(np.median(a))
            if v != "λ=0（不加权）":
                e["log_ratio_vs_unweighted"] = dict(zip(["mean", "ci95"], mean_ci(np.log(a / base))))
                e["p_wilcoxon"] = float(stats.wilcoxon(a, base).pvalue)
            B[spec][n][v] = e
B["relative_ESS_true_weights"] = {n: dict(zip(["mean", "ci95"], mean_ci(ess[n]))) for n in NB}
B["noise_floor_mse"] = 0.09
print("b", time.time() - T0, flush=True)

# ======================= (c) compositional generalization =======================
SEEDS_C = 10
K, D, NPER, SIG = 6, 30, 30, 0.3
S_LIST = [0.0, 0.5, 1.0, 2.0]
held = {(a, b) for a in range(K) for b in range(K) if (b - a) % K in (0, 3)}
seen = [(a, b) for a in range(K) for b in range(K) if (a, b) not in held]
MODELS_C = ["联合36类LR", "因子化LR", "因子化MLP", "因子化kNN"]
rawC = {sv: {m: {"seen": [], "unseen": []} for m in MODELS_C} for sv in S_LIST}


def gen(combos, nper, U, V, W, sv, g):
    X, A_, B_ = [], [], []
    for a, b in combos:
        X.append(U[a] + V[b] + sv * W[a, b] + SIG * g.standard_normal((nper, D)))
        A_ += [a] * nper; B_ += [b] * nper
    return np.vstack(X), np.array(A_), np.array(B_)


for sv in S_LIST:
    for s in range(SEEDS_C):
        g = rng(800 + s)
        U = g.standard_normal((K, D)) / np.sqrt(D) * 2; V = g.standard_normal((K, D)) / np.sqrt(D) * 2
        W = g.standard_normal((K, K, D)) / np.sqrt(D) * 2
        Xtr, atr, btr = gen(seen, NPER, U, V, W, sv, g)
        Xs, as_, bs_ = gen(seen, 20, U, V, W, sv, g)
        Xu, au, bu = gen(sorted(held), 20, U, V, W, sv, g)
        for m in MODELS_C:
            if m == "联合36类LR":
                c = LogisticRegression(max_iter=3000).fit(Xtr, atr * K + btr)
                ps, pu = c.predict(Xs), c.predict(Xu)
                accs = np.mean(ps == as_ * K + bs_); accu = np.mean(pu == au * K + bu)
            else:
                mk = {"因子化LR": lambda: LogisticRegression(max_iter=3000),
                      "因子化MLP": lambda: MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=s),
                      "因子化kNN": lambda: KNeighborsClassifier(5)}[m]
                ca, cb = mk().fit(Xtr, atr), mk().fit(Xtr, btr)
                accs = np.mean((ca.predict(Xs) == as_) & (cb.predict(Xs) == bs_))
                accu = np.mean((ca.predict(Xu) == au) & (cb.predict(Xu) == bu))
            rawC[sv][m]["seen"].append(accs); rawC[sv][m]["unseen"].append(accu)
C = {}
for sv in S_LIST:
    C[sv] = {}
    for m in MODELS_C:
        se, un = np.array(rawC[sv][m]["seen"]), np.array(rawC[sv][m]["unseen"])
        C[sv][m] = {"seen": dict(zip(["mean", "ci95"], mean_ci(se))), "unseen": dict(zip(["mean", "ci95"], mean_ci(un))),
                    "gap": dict(zip(["mean", "ci95"], mean_ci(se - un)))}
print("c", time.time() - T0, flush=True)

# ======================= figures =======================
P = plt()
fig, axes = P.subplots(2, 2, figsize=(13, 8))
for col, t in enumerate(TARGETS):
    ax = axes[0, col]
    ax.plot(xg, TARGETS[t](xg), "k", lw=2, label="真实函数")
    for m in MODELS_A:
        ax.plot(xg, example_pred[(t, m)], lw=1.2, label=m)
    ax.axvspan(-1, 1, color="grey", alpha=.15)
    ax.set_ylim(-3, 26) if t == "x^2" else ax.set_ylim(-4, 4)
    ax.set_title(f"f={t}：训练区间 [-1,1]（灰），种子 0"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax = axes[1, col]
    for m in MODELS_A:
        mu = [A[t][m][b]["mean"] for b in BIN_NAMES]; ci = [A[t][m][b]["ci95"] for b in BIN_NAMES]
        ax.errorbar(range(len(BIN_NAMES)), mu, yerr=ci, marker="o", capsize=3, label=m)
    ax.set_xticks(range(len(BIN_NAMES))); ax.set_xticklabels(BIN_NAMES)
    ax.set_xlabel("距训练区间的距离"); ax.set_ylabel("log10 RMSE"); ax.grid(alpha=.3); ax.set_title(f"f={t}：误差随外推距离")
fig.tight_layout(); fig.savefig(OUT / "extrapolation.png"); P.close(fig)

fig, axes = P.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, spec in zip(axes, ["线性(错设)", "三次(正确设定)"]):
    for v in VARIANTS:
        mu = [B[spec][n][v]["mean"] for n in NB]; ci = [B[spec][n][v]["ci95"] for n in NB]
        ax.errorbar(NB, mu, yerr=ci, marker="o", capsize=3, label=v)
    ax.axhline(0.09, color="k", ls=":", label="噪声下界 0.09")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("训练样本数 n"); ax.set_ylabel("测试域 MSE")
    ax.set_title(f"协变量偏移：{spec}"); ax.grid(alpha=.3)
axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "covariate_shift.png"); P.close(fig)

fig, axes = P.subplots(1, 2, figsize=(12, 4.2), sharey=True)
for ax, key, tt in [(axes[0], "seen", "已见组合"), (axes[1], "unseen", "留出（未见）组合")]:
    for m in MODELS_C:
        mu = [C[sv][m][key]["mean"] for sv in S_LIST]; ci = [C[sv][m][key]["ci95"] for sv in S_LIST]
        ax.errorbar(S_LIST, mu, yerr=ci, marker="o", capsize=3, label=m)
    ax.set_xlabel("交互强度 s（0=完全可组合生成）"); ax.set_ylabel("两个属性都正确的比例"); ax.set_title(tt); ax.grid(alpha=.3)
axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "compositional.png"); P.close(fig)

runtime = time.time() - T0
save_results(__file__, {"config": {"a": {"seeds": SEEDS_A, "n_train": 200, "noise": 0.05, "mlp": "(64,64) lbfgs alpha=1e-4",
                                         "bins_distance_outside": BIN_NAMES, "metric": "log10 RMSE per distance bin"},
                                   "b": {"seeds": SEEDS_B, "n": NB, "train": "N(0.5,0.5^2)", "test": "N(0,0.3^2)", "noise_sd": 0.3,
                                         "estimated_ratio": "logistic regression on (x,x^2), train vs unlabeled test sample of size n"},
                                   "c": {"seeds": SEEDS_C, "K": K, "d": D, "per_seen_combo_train": NPER, "noise": SIG,
                                         "held_out": sorted(held), "s": S_LIST}},
                        "a_extrapolation": A, "b_covariate_shift": B, "c_compositional": C, "runtime_sec": runtime})
print("runtime", runtime)
