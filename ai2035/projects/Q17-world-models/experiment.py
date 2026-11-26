"""Q17 具身智能与世界模型：从随机交互数据学习单摆动力学，并用于 MPC 摆起。

设计（参数在运行前固定）：
- 系统：Gym Pendulum-v1 同款动力学（g=10, m=l=1, dt=0.05, |u|≤2, |θ̇|≤8），
  代价 c = wrap(θ)^2 + 0.1 θ̇^2 + 0.001 u^2；任务：从下垂 θ=π 摆起并保持，200 步。
- 数据：随机交互（初态 θ~U[-π,π], θ̇~U[-1,1]，动作 u~U[-2,2]，每段 25 步），
  预算 N ∈ {50, 200, 1000, 5000} 个转移。
- 世界模型（预测 Δθ, Δθ̇）：
  phys   : 岭回归，特征 [sinθ, cosθ, θ̇, u, 1]（含正确结构 = 先验知识）
  linear : 岭回归，特征 [θ, θ̇, u, 1]（错设：sinθ≈θ，代表“模型偏差”）
  mlp    : MLP(64,64,tanh)，输入 [cosθ, sinθ, θ̇/8, u/2]，单步训练（Adam，约 2000 次小批量更新）
  mlp_dad: 同上 + Data-as-Demonstrator 多步训练（Venkatraman et al. 2015），3 轮，展开 10 步
  linear_dad: 错设线性模型 + DaD（第一轮运行后增加的探索性对照：检验 DaD 是否只在模型类错设时有益）
- 复合误差：200 条测试轨迹，随机动作开环展开 h=1..50，误差为 (cos,sin,θ̇/8) 的 RMSE。
- 规划：CEM-MPC，视界 20、种群 100、精英 10、3 次迭代、热启动；真实动力学 MPC 为上界。
- 无模型基线（同样的数据）：拟合 Q 迭代 FQI + ExtraTrees（Ernst et al. 2005），γ=0.95，
  25 轮，20 棵树，动作离散化 {-2,-1,0,1,2}。
- 归一化回报：(R - R_zero)/(R_trueMPC - R_zero)，R_zero 为零力矩回报。
- 5 个种子；每种子每条件 1 个评估回合（初态 θ=π+U(-0.1,0.1)）。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng, save_results  # noqa: E402

from scipy import stats  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
T0 = time.time()
SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [50, 200, 1000, 5000]
SEG = 25
DT, G, MAXU, MAXV = 0.05, 10.0, 2.0, 8.0
EP_LEN = 200
H_ROLL = 50
N_TEST = 200
CEM = dict(H=20, pop=100, elite=10, iters=3)
N_EVAL_EP = 1
MODELS = ["phys", "linear", "linear_dad", "mlp", "mlp_dad"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def true_step(th, thd, u):
    u = np.clip(u, -MAXU, MAXU)
    nthd = np.clip(thd + (3 * G / 2 * np.sin(th) + 3.0 * u) * DT, -MAXV, MAXV)
    return th + nthd * DT, nthd


def cost(th, thd, u):
    return wrap(th) ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2


def collect(n, g):
    nseg = int(np.ceil(n / SEG))
    th = g.uniform(-np.pi, np.pi, nseg)
    thd = g.uniform(-1, 1, nseg)
    S, U, S2 = [], [], []
    for _ in range(SEG):
        u = g.uniform(-MAXU, MAXU, nseg)
        th2, thd2 = true_step(th, thd, u)
        S.append(np.stack([th, thd], 1)); U.append(u); S2.append(np.stack([th2, thd2], 1))
        th, thd = th2, thd2
    # arrays shaped (nseg, SEG, ...) kept for DaD, flattened to n transitions
    S = np.stack(S, 1)[:, :, :]; U = np.stack(U, 1); S2 = np.stack(S2, 1)
    return S, U, S2


# ---------------------------------------------------------------- models
class Ridge:
    def __init__(self, kind, lam=1e-6):
        self.kind, self.lam = kind, lam

    def feats(self, th, thd, u):
        one = np.ones_like(th)
        if self.kind == "phys":
            return np.stack([np.sin(th), np.cos(th), thd, u, one], -1)
        return np.stack([wrap(th), thd, u, one], -1)

    def fit(self, th, thd, u, dth, dthd, warm=False):
        F = self.feats(th, thd, u)
        A = F.T @ F + self.lam * np.eye(F.shape[1])
        self.W = np.linalg.solve(A, F.T @ np.stack([dth, dthd], 1))
        return self

    def step(self, th, thd, u):
        d = self.feats(th, thd, np.clip(u, -MAXU, MAXU)) @ self.W
        return th + d[..., 0], np.clip(thd + d[..., 1], -MAXV, MAXV)


class MLPModel:
    def __init__(self, seed):
        self.net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", learning_rate_init=3e-3,
                                batch_size=128, max_iter=150, alpha=1e-5, random_state=seed, tol=1e-6,
                                n_iter_no_change=1000)

    @staticmethod
    def inp(th, thd, u):
        return np.stack([np.cos(th), np.sin(th), thd / MAXV, np.clip(u, -MAXU, MAXU) / MAXU], -1)

    def fit(self, th, thd, u, dth, dthd, warm=False):
        Y = np.stack([dth, dthd], 1)
        nb = int(np.ceil(len(th) / 128))  # mini-batches per epoch
        if not warm:
            self.ys = Y.std(0) + 1e-8
            self.net.set_params(max_iter=int(np.clip(2000 // nb, 30, 1000)))  # ~2000 updates
        else:
            self.net.set_params(warm_start=True, max_iter=int(np.clip(600 // nb, 10, 300)))  # ~600 updates
        self.net.fit(self.inp(th, thd, u), Y / self.ys)
        self.Ws, self.bs = self.net.coefs_, self.net.intercepts_
        return self

    def step(self, th, thd, u):
        a = self.inp(th, thd, u)
        shp = a.shape[:-1]
        a = a.reshape(-1, 4)
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            a = a @ W + b
            if i < len(self.Ws) - 1:
                a = np.tanh(a)
        d = (a * self.ys).reshape(*shp, 2)
        return th + d[..., 0], np.clip(thd + d[..., 1], -MAXV, MAXV)


class TrueModel:
    def step(self, th, thd, u):
        return true_step(th, thd, u)


def fit_model(name, S, U, S2, seed):
    th, thd, u = S[..., 0].ravel(), S[..., 1].ravel(), U.ravel()
    dth = (S2[..., 0] - S[..., 0]).ravel(); dthd = (S2[..., 1] - S[..., 1]).ravel()
    if name in ("phys", "linear", "linear_dad"):
        m = Ridge("phys" if name == "phys" else "linear").fit(th, thd, u, dth, dthd)
    else:
        m = MLPModel(seed).fit(th, thd, u, dth, dthd)
    if name.endswith("_dad"):
        # Data-as-Demonstrator: roll model along training segments (horizon 10), add corrected targets
        Xth, Xthd, Xu, Ydth, Ydthd = [th], [thd], [u], [dth], [dthd]
        for _ in range(3):
            pth, pthd = S[:, 0, 0].copy(), S[:, 0, 1].copy()
            for t in range(SEG - 1):
                if t % 10 == 0:  # restart from true state every 10 steps
                    pth, pthd = S[:, t, 0].copy(), S[:, t, 1].copy()
                nth, nthd = m.step(pth, pthd, U[:, t])
                if t % 10 != 0:
                    Xth.append(pth); Xthd.append(pthd); Xu.append(U[:, t])
                    Ydth.append(S2[:, t, 0] - pth); Ydthd.append(S2[:, t, 1] - pthd)
                pth, pthd = nth, nthd
            m.fit(np.concatenate(Xth), np.concatenate(Xthd), np.concatenate(Xu),
                  np.concatenate(Ydth), np.concatenate(Ydthd), warm=True)
    return m


# ---------------------------------------------------------------- evaluation
def rollout_error(model, g):
    th0 = g.uniform(-np.pi, np.pi, N_TEST); thd0 = g.uniform(-1, 1, N_TEST)
    U = g.uniform(-MAXU, MAXU, (N_TEST, H_ROLL))
    th, thd, pth, pthd = th0, thd0, th0.copy(), thd0.copy()
    err = []
    for h in range(H_ROLL):
        th, thd = true_step(th, thd, U[:, h])
        pth, pthd = model.step(pth, pthd, U[:, h])
        e = (np.cos(th) - np.cos(pth)) ** 2 + (np.sin(th) - np.sin(pth)) ** 2 + ((thd - pthd) / MAXV) ** 2
        err.append(float(np.sqrt(np.mean(e))))
    return np.array(err)


def cem_mpc_episode(model, g, th0):
    th, thd = th0, 0.0
    mu = np.zeros(CEM["H"])
    R = 0.0
    for _ in range(EP_LEN):
        m, sd = mu.copy(), np.ones(CEM["H"]) * 1.0
        for _ in range(CEM["iters"]):
            A = np.clip(m + sd * g.standard_normal((CEM["pop"], CEM["H"])), -MAXU, MAXU)
            pth, pthd = np.full(CEM["pop"], th), np.full(CEM["pop"], thd)
            c = np.zeros(CEM["pop"])
            for k in range(CEM["H"]):
                c += cost(pth, pthd, A[:, k])
                pth, pthd = model.step(pth, pthd, A[:, k])
                pth = np.where(np.isfinite(pth), pth, 0.0); pthd = np.where(np.isfinite(pthd), pthd, 0.0)
            el = A[np.argsort(c)[: CEM["elite"]]]
            m, sd = el.mean(0), el.std(0) + 0.05
        u = float(m[0])
        R -= cost(th, thd, u)
        th, thd = true_step(th, thd, u)
        mu = np.concatenate([m[1:], [0.0]])
    return R


def zero_episode(th0):
    th, thd, R = th0, 0.0, 0.0
    for _ in range(EP_LEN):
        R -= cost(th, thd, 0.0)
        th, thd = true_step(th, thd, 0.0)
    return R


ACTS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def fqi(S, U, S2, seed, gamma=0.95, iters=25):
    th, thd, u = S[..., 0].ravel(), S[..., 1].ravel(), U.ravel()
    th2, thd2 = S2[..., 0].ravel(), S2[..., 1].ravel()
    r = -cost(th, thd, u)
    X = np.stack([np.cos(th), np.sin(th), thd / MAXV, u / MAXU], 1)
    n = len(th)
    X2 = np.concatenate([np.stack([np.cos(th2), np.sin(th2), thd2 / MAXV, np.full(n, a / MAXU)], 1) for a in ACTS])
    q = None
    y = r.copy()
    for _ in range(iters):
        q = ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2, random_state=seed, n_jobs=1).fit(X, y)
        y = r + gamma * q.predict(X2).reshape(len(ACTS), n).max(0)
    return q


def fqi_episode(q, th0):
    th, thd, R = th0, 0.0, 0.0
    for _ in range(EP_LEN):
        Xa = np.stack([np.full(5, np.cos(th)), np.full(5, np.sin(th)), np.full(5, thd / MAXV), ACTS / MAXU], 1)
        u = float(ACTS[np.argmax(q.predict(Xa))])
        R -= cost(th, thd, u)
        th, thd = true_step(th, thd, u)
    return R


res = {"config": dict(seeds=SEEDS, budgets=BUDGETS, seg=SEG, dt=DT, ep_len=EP_LEN, cem=CEM,
                      h_roll=H_ROLL, n_test=N_TEST, n_eval_ep=N_EVAL_EP, models=MODELS)}
roll = {m: {b: [] for b in BUDGETS} for m in MODELS}
ret = {m: {b: [] for b in BUDGETS} for m in MODELS + ["fqi"]}
ret_true, ret_zero = [], []
train_time = {m: [] for m in MODELS + ["fqi"]}

for sd in SEEDS:
    g = rng(100 * sd)
    th0s = np.pi + g.uniform(-0.1, 0.1, N_EVAL_EP)
    rz = [zero_episode(t0) for t0 in th0s]
    rt = [cem_mpc_episode(TrueModel(), rng(100 * sd + 1 + i), t0) for i, t0 in enumerate(th0s)]
    ret_zero.append(float(np.mean(rz))); ret_true.append(float(np.mean(rt)))
    S_all, U_all, S2_all = collect(max(BUDGETS), rng(100 * sd + 7))
    for b in BUDGETS:
        ns = b // SEG
        S, U, S2 = S_all[:ns], U_all[:ns], S2_all[:ns]
        for mname in MODELS:
            t1 = time.time()
            model = fit_model(mname, S, U, S2, sd)
            train_time[mname].append(time.time() - t1)
            roll[mname][b].append(rollout_error(model, rng(100 * sd + 50)))
            r = [cem_mpc_episode(model, rng(100 * sd + 1 + i), t0) for i, t0 in enumerate(th0s)]
            ret[mname][b].append(float(np.mean(r)))
        t1 = time.time()
        q = fqi(S, U, S2, sd)
        train_time["fqi"].append(time.time() - t1)
        ret["fqi"][b].append(float(np.mean([fqi_episode(q, t0) for t0 in th0s])))
    print(f"seed {sd} done {time.time()-T0:.1f}s", flush=True)

ret_true = np.array(ret_true); ret_zero = np.array(ret_zero)
res["return_true_mpc"] = mean_ci(ret_true)
res["return_zero"] = mean_ci(ret_zero)
res["return"], res["norm_return"] = {}, {}
for m in MODELS + ["fqi"]:
    res["return"][m] = {str(b): mean_ci(ret[m][b]) for b in BUDGETS}
    res["norm_return"][m] = {str(b): mean_ci((np.array(ret[m][b]) - ret_zero) / (ret_true - ret_zero)) for b in BUDGETS}

H = np.arange(1, H_ROLL + 1)
res["rollout_error"] = {}
for m in MODELS:
    d = {}
    for b in BUDGETS:
        E = np.array(roll[m][b])  # seeds x H
        slope = []
        for e in E:
            sel = H <= 30
            slope.append(np.polyfit(np.log(H[sel]), np.log(e[sel] + 1e-12), 1)[0])
        d[str(b)] = {"h1": mean_ci(E[:, 0]), "h10": mean_ci(E[:, 9]), "h30": mean_ci(E[:, 29]),
                     "h50": mean_ci(E[:, 49]), "growth_exponent_h1_30": mean_ci(slope),
                     "ratio_h30_h1": mean_ci(E[:, 29] / E[:, 0])}
    res["rollout_error"][m] = d


def paired(a, b):
    a, b = np.array(a), np.array(b)
    st = stats.ttest_rel(a, b)
    return {"diff": mean_ci(a - b), "t": float(st.statistic), "p": float(st.pvalue)}


res["tests"] = {
    "dad_vs_mlp_h30_err_rel_reduction": {str(b): mean_ci(1 - np.array(roll["mlp_dad"][b])[:, 29] / np.array(roll["mlp"][b])[:, 29]) for b in BUDGETS},
    "dad_vs_mlp_h30_paired": {str(b): paired(np.array(roll["mlp_dad"][b])[:, 29], np.array(roll["mlp"][b])[:, 29]) for b in BUDGETS},
    "mlp_vs_fqi_return": {str(b): paired(ret["mlp"][b], ret["fqi"][b]) for b in BUDGETS},
    "linear_1000_vs_5000_return": paired(ret["linear"][5000], ret["linear"][1000]),
    "lineardad_vs_linear_h30_err_rel_reduction": {str(b): mean_ci(1 - np.array(roll["linear_dad"][b])[:, 29] / np.array(roll["linear"][b])[:, 29]) for b in BUDGETS},
    "lineardad_vs_linear_h10_err_rel_reduction": {str(b): mean_ci(1 - np.array(roll["linear_dad"][b])[:, 9] / np.array(roll["linear"][b])[:, 9]) for b in BUDGETS},
    "lineardad_vs_linear_return": {str(b): paired(ret["linear_dad"][b], ret["linear"][b]) for b in BUDGETS},
    "dad_vs_mlp_return": {str(b): paired(ret["mlp_dad"][b], ret["mlp"][b]) for b in BUDGETS},
}
res["train_time_s"] = {m: mean_ci(v) for m, v in train_time.items()}
res["runtime_s"] = time.time() - T0

# ---------------------------------------------------------------- figures
P = plt()
od = results_dir(__file__)
fig, ax = P.subplots(1, 3, figsize=(14, 4))
PLOT_B = [50, 1000, 5000]
names = {"phys": "岭回归+sinθ 特征(正确结构)", "linear": "线性(θ,θ̇,u)(错设)", "linear_dad": "线性(错设)+DaD", "mlp": "MLP 单步训练", "mlp_dad": "MLP 多步(DaD)"}
for i, b in enumerate(PLOT_B):
    for m in MODELS:
        E = np.array(roll[m][b])
        mu = E.mean(0); ci = 1.96 * E.std(0, ddof=1) / np.sqrt(len(E))
        ax[i].plot(H, mu, label=names[m]); ax[i].fill_between(H, mu - ci, mu + ci, alpha=0.2)
    ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_title(f"开环展开误差，N={b}")
    ax[i].set_xlabel("展开步数 h"); ax[i].set_ylabel("RMSE")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(od / "rollout_error.png"); P.close(fig)

fig, ax = P.subplots(figsize=(6, 4))
for m, lab in list(names.items()) + [("fqi", "无模型 FQI(ExtraTrees)")]:
    mu = [res["norm_return"][m][str(b)][0] for b in BUDGETS]
    ci = [res["norm_return"][m][str(b)][1] for b in BUDGETS]
    ax.errorbar(BUDGETS, mu, yerr=ci, marker="o", capsize=3, label=lab)
ax.axhline(1, color="k", ls="--", label="真实动力学 MPC")
ax.axhline(0, color="gray", ls=":", label="零力矩")
ax.set_xscale("log"); ax.set_xlabel("交互样本数 N"); ax.set_ylabel("归一化回报")
ax.set_title("摆起任务：样本效率"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(od / "sample_efficiency.png"); P.close(fig)

save_results(__file__, res)
print(f"done in {time.time()-T0:.1f}s")
