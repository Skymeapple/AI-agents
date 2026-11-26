"""Q20 科学智能：用 SINDy 从数据中发现控制方程，并与黑箱回归对比。

设计（参数在运行前固定）：
- 系统 1：Lorenz（σ=10, ρ=28, β=8/3），solve_ivp(rtol=atol=1e-10)，采样 dt=0.005，丢弃前 1 个时间单位。
  观测噪声：x_obs = x + η·std(x)·N(0,1)，η ∈ {0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1}。
  数据长度 T ∈ {0.5, 1, 2, 5, 10, 20}（时间单位；主实验 T=10）。
- 导数估计：fd = 二阶中心差分（np.gradient）；sg = Savitzky–Golay（窗口 21，3 阶）平滑状态并求导；
  oracle = 真实导数（在真实状态上计算），但库仍用带噪状态（误差在变量中）。
- SINDy：三阶多项式库（20 项），STLSQ 阈值 λ=0.1，10 次迭代；另报告 λ 敏感性（不用于选参）。
  对照：Dense-LS（同一库、不稀疏化的最小二乘）。
- 指标：项支撑的精确率/召回率、精确恢复（支撑完全一致）、系数相对误差 ||Ξ−Ξ*||_F/||Ξ*||_F。
- 预测对比（η=0.01, T=10, sg 导数）：SINDy vs Dense-LS vs MLP(64,64,tanh) vs GBDT(HistGB)，
  均学习 x→ẋ，再用 RK4（dt=0.01）积分；20 个吸引子上的测试初值，有效预测时间 VPT
  （归一化误差首次 > 0.5），以 Lyapunov 时间（λ1≈0.906）计；外推：20 个远离吸引子的初值
  （|x|,|y|≤40, z∈[-20,80]）上前 1 个时间单位的归一化 RMSE。
- 系统 2：阻尼单摆 θ̇=ω, ω̇=−4 sinθ − 0.2ω。训练：5 条小振幅轨迹（|θ0|≤1，各 10 s，dt=0.01，η=0.01）。
  库 A：(θ,ω) 三阶多项式（不含真实结构）；库 B：二阶多项式 ∪ {sinθ, cosθ}（含真实结构）。
  测试：分布内（θ0=0.8）与外推（θ0=2.8，接近分界线）10 s 轨迹的 θ RMSE；与 MLP 对比。
  事后补充（首轮结果后加入，非预注册）：训练振幅放宽到 |θ0|≤2.5，检验“数据信息量”假说。
- 5 个种子。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from itertools import combinations_with_replacement  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng, save_results  # noqa: E402

from scipy import stats  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

warnings.filterwarnings("ignore")
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
DT = 0.005
NOISES = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
LENGTHS = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
LAMBDAS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
LAM = 0.1
SG_WIN, SG_ORD = 21, 3
LYAP = 0.906


def lorenz(t, s, sig=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s
    return [sig * (y - x), x * (rho - z) - y, x * y - beta * z]


def lorenz_np(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    return np.stack([10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z], -1)


# polynomial library
EXPS = [()]
for d in (1, 2, 3):
    EXPS += list(combinations_with_replacement(range(3), d))
NAMES = ["1"] + ["".join("xyz"[i] for i in e) for e in EXPS[1:]]


def poly_lib(S):
    cols = [np.ones(S.shape[:-1])]
    for e in EXPS[1:]:
        c = np.ones(S.shape[:-1])
        for i in e:
            c = c * S[..., i]
        cols.append(c)
    return np.stack(cols, -1)


XI_TRUE = np.zeros((len(EXPS), 3))
XI_TRUE[NAMES.index("x"), 0], XI_TRUE[NAMES.index("y"), 0] = -10, 10
XI_TRUE[NAMES.index("x"), 1], XI_TRUE[NAMES.index("y"), 1], XI_TRUE[NAMES.index("xz"), 1] = 28, -1, -1
XI_TRUE[NAMES.index("xy"), 2], XI_TRUE[NAMES.index("z"), 2] = 1, -8 / 3


def stlsq(Theta, dX, lam, iters=10):
    Xi = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    for _ in range(iters):
        small = np.abs(Xi) < lam
        Xi[small] = 0
        for k in range(dX.shape[1]):
            big = ~small[:, k]
            if big.any():
                Xi[big, k] = np.linalg.lstsq(Theta[:, big], dX[:, k], rcond=None)[0]
    return Xi


def support_metrics(Xi, Xt):
    s, st = Xi != 0, Xt != 0
    tp = (s & st).sum()
    prec = tp / s.sum() if s.sum() else 0.0
    rec = tp / st.sum()
    return dict(precision=float(prec), recall=float(rec), exact=float((s == st).all()),
                coef_err=float(np.linalg.norm(Xi - Xt) / np.linalg.norm(Xt)))


def simulate_lorenz(x0, T, dt=DT):
    burn = 1.0
    t = np.arange(0, burn + T + 1e-9, dt)
    sol = solve_ivp(lorenz, (0, t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-10, method="LSODA")
    S = sol.y.T[t >= burn - 1e-12]
    return S


def derivs(Sobs, S, method, dt=DT):
    if method == "fd":
        return Sobs, np.gradient(Sobs, dt, axis=0)
    if method == "sg":
        return (savgol_filter(Sobs, SG_WIN, SG_ORD, axis=0),
                savgol_filter(Sobs, SG_WIN, SG_ORD, deriv=1, delta=dt, axis=0))
    return Sobs, lorenz_np(S)  # oracle derivative


def make_data(sd, T, eta, g):
    x0 = np.array([-8.0, 8.0, 27.0]) + g.standard_normal(3)
    S = simulate_lorenz(x0, T)
    Sobs = S + eta * S.std(0) * g.standard_normal(S.shape)
    return S, Sobs


res = {"config": dict(seeds=SEEDS, dt=DT, noises=NOISES, lengths=LENGTHS, lam=LAM, lambdas=LAMBDAS,
                      sg=[SG_WIN, SG_ORD], library=NAMES)}

# ---------------- (1) noise sweep at T=10 ; (2) length sweep at eta=0.01 ; (3) lambda sensitivity
METHODS = ["fd", "sg", "oracle"]
noise_tab = {m: {e: [] for e in NOISES} for m in METHODS}
dense_tab = {e: [] for e in NOISES}
len_tab = {m: {L: [] for L in LENGTHS} for m in ["fd", "sg"]}
lam_tab = {e: {lam: [] for lam in LAMBDAS} for e in [0.01, 0.03]}
for sd in SEEDS:
    g = rng(sd)
    S_long = simulate_lorenz(np.array([-8.0, 8.0, 27.0]) + g.standard_normal(3), max(LENGTHS))
    for eta in NOISES:
        gg = rng(1000 + sd * 31 + int(eta * 1e4))
        S = S_long[: int(10 / DT) + 1]
        Sobs = S + eta * S.std(0) * gg.standard_normal(S.shape)
        for m in METHODS:
            Xs, dX = derivs(Sobs, S, m)
            Xi = stlsq(poly_lib(Xs), dX, LAM)
            noise_tab[m][eta].append(support_metrics(Xi, XI_TRUE))
            if m == "sg":
                Xd = np.linalg.lstsq(poly_lib(Xs), dX, rcond=None)[0]
                dense_tab[eta].append(float(np.linalg.norm(Xd - XI_TRUE) / np.linalg.norm(XI_TRUE)))
            if m == "sg" and eta in lam_tab:
                for lam in LAMBDAS:
                    lam_tab[eta][lam].append(support_metrics(stlsq(poly_lib(Xs), dX, lam), XI_TRUE))
    for L in LENGTHS:
        gg = rng(5000 + sd * 31 + int(L * 10))
        S = S_long[: int(L / DT) + 1]
        Sobs = S + 0.01 * S_long.std(0) * gg.standard_normal(S.shape)
        for m in ["fd", "sg"]:
            Xs, dX = derivs(Sobs, S, m)
            len_tab[m][L].append(support_metrics(stlsq(poly_lib(Xs), dX, LAM), XI_TRUE))
print(f"sweeps done {time.time()-T0:.1f}s", flush=True)


def agg(lst):
    return {k: mean_ci([d[k] for d in lst]) for k in lst[0]}


res["noise_sweep_T10"] = {m: {str(e): agg(noise_tab[m][e]) for e in NOISES} for m in METHODS}
res["dense_ls_coef_err_sg"] = {str(e): mean_ci(dense_tab[e]) for e in NOISES}
res["length_sweep_eta0.01"] = {m: {str(L): agg(len_tab[m][L]) for L in LENGTHS} for m in len_tab}
res["lambda_sensitivity_sg"] = {str(e): {str(l): agg(lam_tab[e][l]) for l in LAMBDAS} for e in lam_tab}
# paired test: sg vs fd exact recovery at eta=0.01 and 0.03
res["tests"] = {}
for e in [0.003, 0.01, 0.03]:
    a = np.array([d["coef_err"] for d in noise_tab["sg"][e]]); b = np.array([d["coef_err"] for d in noise_tab["fd"][e]])
    st = stats.ttest_rel(np.log(a), np.log(b))
    res["tests"][f"sg_vs_fd_logcoeferr_eta{e}"] = {"ratio_geo_mean": float(np.exp(np.mean(np.log(a / b)))),
                                                    "t": float(st.statistic), "p": float(st.pvalue)}


# ---------------- (4) forecasting & extrapolation on Lorenz
def rk4(f, X0, dt, n):
    X = X0.copy(); out = [X.copy()]
    for _ in range(n):
        k1 = f(X); k2 = f(X + dt / 2 * k1); k3 = f(X + dt / 2 * k2); k4 = f(X + dt * k3)
        X = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X = np.clip(np.nan_to_num(X, nan=1e3, posinf=1e3, neginf=-1e3), -1e3, 1e3)
        out.append(X.copy())
    return np.stack(out, 1)  # (n_ic, n+1, 3)


class MLPf:
    def __init__(self, X, Y, seed):
        self.mx, self.sx, self.my, self.sy = X.mean(0), X.std(0), Y.mean(0), Y.std(0)
        net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=300, learning_rate_init=3e-3,
                           random_state=seed, tol=1e-7, n_iter_no_change=1000, batch_size=128)
        net.fit((X - self.mx) / self.sx, (Y - self.my) / self.sy)
        self.W, self.b = net.coefs_, net.intercepts_

    def __call__(self, X):
        a = (X - self.mx) / self.sx
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            a = a @ W + b
            if i < len(self.W) - 1:
                a = np.tanh(a)
        return a * self.sy + self.my


class HGBf:
    def __init__(self, X, Y, seed):
        self.m = [HistGradientBoostingRegressor(max_iter=150, random_state=seed).fit(X, Y[:, k]) for k in range(Y.shape[1])]

    def __call__(self, X):
        return np.stack([m.predict(X) for m in self.m], -1)


FDT, FH = 0.01, 10.0
NF = int(FH / FDT)
vpt = {k: [] for k in ["SINDy", "Dense-LS", "MLP", "GBDT"]}
extrap = {k: [] for k in vpt}
attr_std = None
for sd in SEEDS:
    g = rng(200 + sd)
    S, Sobs = make_data(sd, 10.0, 0.01, g)
    Xs, dX = derivs(Sobs, S, "sg")
    attr_std = float(np.linalg.norm(S.std(0)))
    Xi = stlsq(poly_lib(Xs), dX, LAM)
    Xd = np.linalg.lstsq(poly_lib(Xs), dX, rcond=None)[0]
    models = {"SINDy": lambda X, Xi=Xi: poly_lib(X) @ Xi, "Dense-LS": lambda X, Xd=Xd: poly_lib(X) @ Xd,
              "MLP": MLPf(Xs, dX, sd), "GBDT": HGBf(Xs, dX, sd)}
    # test ICs on attractor
    Sref = simulate_lorenz(np.array([5.0, -5.0, 20.0]) + g.standard_normal(3), 40.0)
    ic = Sref[g.choice(len(Sref) - 10, 20, replace=False)]
    truth = rk4(lorenz_np, ic, FDT, NF)
    ic_far = np.stack([g.uniform(-40, 40, 20), g.uniform(-40, 40, 20), g.uniform(-20, 80, 20)], 1)
    truth_far = rk4(lorenz_np, ic_far, FDT, int(1.0 / FDT))
    for k, f in models.items():
        pred = rk4(f, ic, FDT, NF)
        err = np.linalg.norm(pred - truth, axis=-1) / attr_std
        first = np.array([np.argmax(e > 0.5) if (e > 0.5).any() else NF for e in err])
        vpt[k].append(float(np.mean(first * FDT * LYAP)))
        pf = rk4(f, ic_far, FDT, int(1.0 / FDT))
        extrap[k].append(float(np.sqrt(np.mean(np.sum((pf - truth_far) ** 2, -1))) / attr_std))
    print(f"forecast seed {sd} {time.time()-T0:.1f}s", flush=True)
res["forecast"] = {"vpt_lyapunov_times": {k: mean_ci(v) for k, v in vpt.items()},
                   "extrap_nrmse_1tu": {k: mean_ci(v) for k, v in extrap.items()},
                   "vpt_per_seed": vpt, "extrap_per_seed": extrap,
                   "horizon_lyap": FH * LYAP}
res["tests"]["vpt_SINDy_vs_MLP"] = {"p": float(stats.ttest_rel(vpt["SINDy"], vpt["MLP"]).pvalue)}
res["tests"]["extrap_SINDy_vs_MLP"] = {"p": float(stats.ttest_rel(np.log(extrap["SINDy"]), np.log(extrap["MLP"])).pvalue)}


# ---------------- (5) pendulum: library = prior knowledge
PDT = 0.01


def pend(t, s):
    return [s[1], -4 * np.sin(s[0]) - 0.2 * s[1]]


def pend_np(S):
    return np.stack([S[..., 1], -4 * np.sin(S[..., 0]) - 0.2 * S[..., 1]], -1)


def lib_A(S):  # cubic polynomials in (θ, ω)
    th, om = S[..., 0], S[..., 1]
    one = np.ones_like(th)
    return np.stack([one, th, om, th ** 2, th * om, om ** 2, th ** 3, th ** 2 * om, th * om ** 2, om ** 3], -1)


NAMES_A = ["1", "θ", "ω", "θ²", "θω", "ω²", "θ³", "θ²ω", "θω²", "ω³"]


def lib_B(S):  # quadratic polynomials + sin, cos
    th, om = S[..., 0], S[..., 1]
    one = np.ones_like(th)
    return np.stack([one, th, om, th ** 2, th * om, om ** 2, np.sin(th), np.cos(th)], -1)


NAMES_B = ["1", "θ", "ω", "θ²", "θω", "ω²", "sinθ", "cosθ"]
XI_B = np.zeros((8, 2)); XI_B[2, 0] = 1; XI_B[6, 1] = -4; XI_B[2, 1] = -0.2


def pend_traj(th0, T=10.0):
    t = np.arange(0, T + 1e-9, PDT)
    return solve_ivp(pend, (0, T), [th0, 0.0], t_eval=t, rtol=1e-10, atol=1e-10).y.T


pend_res = {k: {"in": [], "out": []} for k in ["SINDy-A(多项式)", "SINDy-B(含sin)", "MLP"]}
pend_support = []
pend_coefA = []
# post-hoc (added after first run): same pipeline but training amplitudes |θ0|≤2.5 (informative data)
wide_res = {k: {"in": [], "out": []} for k in ["SINDy-A(多项式)", "SINDy-B(含sin)"]}
wide_support = []


def pend_data(g, amp):
    Xs_all, dX_all = [], []
    for th0 in g.uniform(-amp, amp, 5):
        S = pend_traj(th0)
        Sobs = S + 0.01 * S.std(0) * g.standard_normal(S.shape)
        Xs_all.append(savgol_filter(Sobs, SG_WIN, SG_ORD, axis=0))
        dX_all.append(savgol_filter(Sobs, SG_WIN, SG_ORD, deriv=1, delta=PDT, axis=0))
    return np.concatenate(Xs_all), np.concatenate(dX_all)


def pend_eval(fs, store):
    for tag, th0 in (("in", 0.8), ("out", 2.8)):
        truth = rk4(pend_np, np.array([[th0, 0.0]]), PDT, 1000)[0]
        for k, f in fs.items():
            p = rk4(f, np.array([[th0, 0.0]]), PDT, 1000)[0]
            store[k][tag].append(float(np.sqrt(np.mean((p[:, 0] - truth[:, 0]) ** 2))))


for sd in SEEDS:
    g = rng(700 + sd)
    Xs, dX = pend_data(g, 1.0)
    XiA = stlsq(lib_A(Xs), dX, LAM); XiB = stlsq(lib_B(Xs), dX, LAM)
    pend_support.append(support_metrics(XiB, XI_B))
    pend_coefA.append(XiA[:, 1])
    mlp = MLPf(Xs, dX, sd)
    pend_eval({"SINDy-A(多项式)": lambda X, XiA=XiA: lib_A(X) @ XiA, "SINDy-B(含sin)": lambda X, XiB=XiB: lib_B(X) @ XiB, "MLP": mlp}, pend_res)
    Xw, dXw = pend_data(rng(800 + sd), 2.5)
    XiAw = stlsq(lib_A(Xw), dXw, LAM); XiBw = stlsq(lib_B(Xw), dXw, LAM)
    wide_support.append(support_metrics(XiBw, XI_B))
    pend_eval({"SINDy-A(多项式)": lambda X, XiA=XiAw: lib_A(X) @ XiA, "SINDy-B(含sin)": lambda X, XiB=XiBw: lib_B(X) @ XiB}, wide_res)
res["pendulum"] = {
    "theta_rmse": {k: {t: mean_ci(v[t]) for t in v} for k, v in pend_res.items()},
    "theta_rmse_per_seed": pend_res,
    "libB_support": agg(pend_support),
    "libB_support_per_seed_exact": [d["exact"] for d in pend_support],
    "libA_omega_dot_coefs_mean": {n: float(np.mean([c[i] for c in pend_coefA])) for i, n in enumerate(NAMES_A)},
    "libA_omega_dot_coefs_ci": {n: mean_ci([c[i] for c in pend_coefA])[1] for i, n in enumerate(NAMES_A)},
    "posthoc_wide_amplitude": {"theta_rmse": {k: {t: mean_ci(v[t]) for t in v} for k, v in wide_res.items()},
                               "theta_rmse_per_seed": wide_res, "libB_support": agg(wide_support)},
}
res["runtime_s"] = time.time() - T0

# ---------------- figures
P = plt()
od = results_dir(__file__)
fig, ax = P.subplots(1, 3, figsize=(15, 4))
xs = [max(e, 3e-4) for e in NOISES]
for m in METHODS:
    ax[0].errorbar(xs, [res["noise_sweep_T10"][m][str(e)]["exact"][0] for e in NOISES],
                   yerr=[res["noise_sweep_T10"][m][str(e)]["exact"][1] for e in NOISES], marker="o", capsize=3, label=m)
    ax[1].errorbar(xs, [res["noise_sweep_T10"][m][str(e)]["coef_err"][0] for e in NOISES],
                   yerr=[res["noise_sweep_T10"][m][str(e)]["coef_err"][1] for e in NOISES], marker="o", capsize=3, label=f"SINDy {m}")
ax[1].plot(xs, [res["dense_ls_coef_err_sg"][str(e)][0] for e in NOISES], "k--", marker="x", label="Dense-LS (sg)")
for a in ax[:2]:
    a.set_xscale("log"); a.set_xlabel("噪声 η（0 画在 3e-4 处）"); a.legend(fontsize=7)
ax[0].set_ylabel("精确恢复率"); ax[0].set_title("Lorenz：结构精确恢复 vs 噪声 (T=10)")
ax[1].set_yscale("log"); ax[1].set_ylabel("系数相对误差"); ax[1].set_title("系数误差 vs 噪声")
for m in ["fd", "sg"]:
    ax[2].errorbar(LENGTHS, [res["length_sweep_eta0.01"][m][str(L)]["exact"][0] for L in LENGTHS],
                   yerr=[res["length_sweep_eta0.01"][m][str(L)]["exact"][1] for L in LENGTHS], marker="o", capsize=3, label=m)
ax[2].set_xscale("log"); ax[2].set_xlabel("数据长度 T"); ax[2].set_ylabel("精确恢复率"); ax[2].set_title("η=0.01 时 vs 数据长度")
ax[2].legend()
fig.tight_layout(); fig.savefig(od / "sindy_recovery.png"); P.close(fig)

fig, ax = P.subplots(1, 2, figsize=(11, 4))
ks = list(vpt)
ax[0].bar(ks, [res["forecast"]["vpt_lyapunov_times"][k][0] for k in ks], yerr=[res["forecast"]["vpt_lyapunov_times"][k][1] for k in ks], capsize=4)
ax[0].set_ylabel("有效预测时间（Lyapunov 时间）"); ax[0].set_title("Lorenz 吸引子上的预测")
ax[1].bar(ks, [res["forecast"]["extrap_nrmse_1tu"][k][0] for k in ks], yerr=[res["forecast"]["extrap_nrmse_1tu"][k][1] for k in ks], capsize=4)
ax[1].set_yscale("log"); ax[1].set_ylabel("归一化 RMSE（前 1 时间单位）"); ax[1].set_title("外推：远离吸引子的初值")
fig.tight_layout(); fig.savefig(od / "forecast.png"); P.close(fig)

fig, ax = P.subplots(figsize=(6, 4))
kk = list(pend_res)
w = 0.35
ax.bar(np.arange(3) - w / 2, [res["pendulum"]["theta_rmse"][k]["in"][0] for k in kk], w, yerr=[res["pendulum"]["theta_rmse"][k]["in"][1] for k in kk], label="分布内 θ0=0.8", capsize=3)
ax.bar(np.arange(3) + w / 2, [res["pendulum"]["theta_rmse"][k]["out"][0] for k in kk], w, yerr=[res["pendulum"]["theta_rmse"][k]["out"][1] for k in kk], label="外推 θ0=2.8", capsize=3)
ax.set_xticks(range(3)); ax.set_xticklabels(kk); ax.set_yscale("log"); ax.set_ylabel("θ RMSE (rad)")
ax.set_title("单摆：函数库即先验"); ax.legend()
fig.tight_layout(); fig.savefig(od / "pendulum.png"); P.close(fig)

save_results(__file__, res)
print(f"done in {time.time()-T0:.1f}s")
