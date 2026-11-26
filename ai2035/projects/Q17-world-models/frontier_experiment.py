"""Q17 前沿实验：世界模型的“保真度 ≠ 决策效用”——排序保真度判据与决策感知数据分配。

问题（2025–2026 争论）：世界模型应以预测/视觉保真度评估，还是以下游决策效用评估？
WorldArena (arXiv:2602.08971)、MiraBench (2605.29360)、LRC (2610.37588)、RLVR-World (2505.13934)。

设计（参数在运行前固定）：
- 环境：Gym Pendulum 动力学（同第一阶段），摆起任务 200 步，CEM-MPC（H=20, pop=100, elite=10, 3 iters）。
- 模型动物园（每种子 20 个）：
  学习模型：随机傅里叶特征岭回归 RFF(M) M∈{25,100} × 随机交互数据 N∈{200,2000}（首轮含 M=400，为满足 4 分钟限制删去）；
            错设线性（N=2000）；含 sinθ 的物理特征岭回归（N=2000）。
  受控扰动的真实动力学：动作增益误差 k∈{0.7,1.3,0.4,1.6}（作用于动作项，“差分型”误差）；
            重力误差与状态偏置 δ·sin(2θ)（与动作无关，“共模型”误差），二者的幅度
            按一步 RMSE 与增益误差 |k−1|∈{0.3,0.6} 逐一匹配（二分标定）。
- 指标：(a) 均匀分布上一步 RMSE；(b) 30 步开环 RMSE；(c) 在真实 MPC 轨迹状态上的一步 RMSE；
  (d) 排序保真度：在真实 MPC 轨迹上取 30 个状态，每个状态取 64 条候选动作序列——
      全局候选 N(0,1) 与局部候选（真实规划解 + N(0,0.3²)）——计算模型代价与真实代价的
      Spearman ρ，以及归一化后悔 (J(argmin Ĵ) − min J)/std(J)；
  (e) 闭环归一化回报 (R − R_zero)/(R_true − R_zero)。
- 干预（决策感知数据分配，Ross & Bagnell 2012 式 DAgger）：RFF(M=25)，总预算 2000：
  A=2000 条随机交互；B=1200 条随机 + 2 轮各 400 条“当前模型 MPC 策略（加 N(0,0.3²) 探索噪声）”在线数据。
- 6 个种子（首轮 8 个超时）；跨模型 Spearman 相关取每种子值再求均值±CI；配对 t 检验。
- 事后补充（首轮结果后加入，非预注册）：(i) 策略分布上的“乐观度” = 每步所选计划的真实代价 −
  模型预测代价（/H），正值表示模型乐观；离线乐观度 = 在 15 个固定评估状态上用模型做 CEM 规划，
  计算同一量（无需用模型闭环运行）；(ii) DAgger 也作用于错设线性模型。
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt, results_dir, rng  # noqa: E402

from scipy import stats  # noqa: E402

T0 = time.time()
SEEDS = list(range(6))
DT, G, MAXU, MAXV = 0.05, 10.0, 2.0, 8.0
EP_LEN, SEG = 200, 25
CEM = dict(H=20, pop=100, elite=10, iters=3)
N_RANK_STATES, N_CAND = 30, 64
GAIN_LEVELS = [0.3, 0.6]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def cost(th, thd, u):
    return wrap(th) ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2


class Dyn:
    """true dynamics with optional perturbations"""

    def __init__(self, gain=1.0, grav=1.0, bias=0.0):
        self.gain, self.grav, self.bias = gain, grav, bias

    def step(self, th, thd, u):
        u = np.clip(u, -MAXU, MAXU)
        acc = 3 * G * self.grav / 2 * np.sin(th) + 3.0 * self.gain * u + self.bias * np.sin(2 * th) / DT
        nthd = np.clip(thd + acc * DT, -MAXV, MAXV)
        return th + nthd * DT, nthd


TRUE = Dyn()


class Ridge:
    def __init__(self, kind, M=0, g=None, lam=1e-4):
        self.kind, self.lam = kind, lam
        if kind == "rff":
            self.Om = g.standard_normal((4, M)) * 1.0
            self.ph = g.uniform(0, 2 * np.pi, M)
            self.M = M

    def feats(self, th, thd, u):
        u = np.clip(u, -MAXU, MAXU)
        if self.kind == "phys":
            return np.stack([np.sin(th), np.cos(th), thd, u, np.ones_like(th)], -1)
        if self.kind == "linear":
            return np.stack([wrap(th), thd, u, np.ones_like(th)], -1)
        z = np.stack([np.cos(th), np.sin(th), thd / MAXV, u / MAXU], -1)
        f = np.sqrt(2.0 / self.M) * np.cos(z @ self.Om + self.ph)
        return np.concatenate([f, z, np.ones(z.shape[:-1] + (1,))], -1)

    def fit(self, th, thd, u, th2, thd2):
        F = self.feats(th, thd, u)
        Y = np.stack([th2 - th, thd2 - thd], 1)
        self.W = np.linalg.solve(F.T @ F + self.lam * len(th) * np.eye(F.shape[1]) / 1000, F.T @ Y)
        return self

    def step(self, th, thd, u):
        d = self.feats(th, thd, u) @ self.W
        return th + d[..., 0], np.clip(thd + d[..., 1], -MAXV, MAXV)


def collect_random(n, g):
    nseg = int(np.ceil(n / SEG))
    th = g.uniform(-np.pi, np.pi, nseg); thd = g.uniform(-1, 1, nseg)
    D = []
    for _ in range(SEG):
        u = g.uniform(-MAXU, MAXU, nseg)
        th2, thd2 = TRUE.step(th, thd, u)
        D.append(np.stack([th, thd, u, th2, thd2], 1))
        th, thd = th2, thd2
    return np.concatenate(D)[:n]


def cem_plan(model, th, thd, mu, g):
    m, sd = mu.copy(), np.ones(CEM["H"])
    for _ in range(CEM["iters"]):
        A = np.clip(m + sd * g.standard_normal((CEM["pop"], CEM["H"])), -MAXU, MAXU)
        pth, pthd = np.full(CEM["pop"], th), np.full(CEM["pop"], thd)
        c = np.zeros(CEM["pop"])
        for k in range(CEM["H"]):
            c += cost(pth, pthd, A[:, k])
            pth, pthd = model.step(pth, pthd, A[:, k])
            pth = np.nan_to_num(pth); pthd = np.nan_to_num(pthd)
        el = A[np.argsort(c)[: CEM["elite"]]]
        m, sd = el.mean(0), el.std(0) + 0.05
    return m


def episode(model, g, th0, noise=0.0, record=False, optim=None):
    th, thd, R = th0, 0.0, 0.0
    mu = np.zeros(CEM["H"]); traj = []
    for _ in range(EP_LEN):
        m = cem_plan(model, th, thd, mu, g)
        if optim is not None:  # on-policy optimism: true minus predicted cost of the chosen plan (per step)
            A1 = np.clip(m, -MAXU, MAXU)[None]
            optim.append(float((seq_cost(TRUE, th, thd, A1) - seq_cost(model, th, thd, A1))[0] / CEM["H"]))
        u = float(np.clip(m[0] + noise * g.standard_normal(), -MAXU, MAXU))
        if record:
            traj.append((th, thd, m.copy()))
        R -= cost(th, thd, u)
        th2, thd2 = TRUE.step(th, thd, u)
        if noise > 0:
            traj.append((th, thd, u, th2, thd2))
        th, thd = th2, thd2
        mu = np.concatenate([m[1:], [0.0]])
    return R, traj


def zero_return(th0):
    th, thd, R = th0, 0.0, 0.0
    for _ in range(EP_LEN):
        R -= cost(th, thd, 0.0); th, thd = TRUE.step(th, thd, 0.0)
    return R


def seq_cost(model, th, thd, A):
    pth, pthd = np.full(len(A), th), np.full(len(A), thd)
    c = np.zeros(len(A))
    for k in range(A.shape[1]):
        c += cost(pth, pthd, A[:, k]); pth, pthd = model.step(pth, pthd, A[:, k])
        pth = np.nan_to_num(pth); pthd = np.nan_to_num(pthd)
    return c


def state_err(model, th, thd, u):
    a1, b1 = TRUE.step(th, thd, u); a2, b2 = model.step(th, thd, u)
    e = (np.cos(a1) - np.cos(a2)) ** 2 + (np.sin(a1) - np.sin(a2)) ** 2 + ((b1 - b2) / MAXV) ** 2
    return float(np.sqrt(np.mean(e)))


def metrics(model, ev):
    m = {"mse1_uniform": state_err(model, *ev["uni"]), "mse1_onpolicy": state_err(model, *ev["onp"])}
    th, thd = ev["roll0"]; pth, pthd = th.copy(), thd.copy()
    for k in range(30):
        th, thd = TRUE.step(th, thd, ev["rollU"][:, k]); pth, pthd = model.step(pth, pthd, ev["rollU"][:, k])
    e = (np.cos(th) - np.cos(pth)) ** 2 + (np.sin(th) - np.sin(pth)) ** 2 + ((thd - pthd) / MAXV) ** 2
    m["rollout30"] = float(np.sqrt(np.mean(e)))
    # offline optimism (post-hoc): at half of the evaluation states, plan with the model (CEM) and compare
    # true vs predicted cost of the model's own plan -- needs no closed-loop rollout with the model
    gp = rng(4242); opts = []
    for (s_th, s_thd, A, Jt) in ev["rank"]["local"][::2]:
        m_plan = np.clip(cem_plan(model, s_th, s_thd, np.zeros(CEM["H"]), gp), -MAXU, MAXU)[None]
        opts.append(float((seq_cost(TRUE, s_th, s_thd, m_plan) - seq_cost(model, s_th, s_thd, m_plan))[0] / CEM["H"]))
    m["optimism_offline"] = float(np.mean(opts))
    for kind in ("global", "local"):
        rhos, regs = [], []
        for (s_th, s_thd, A, Jt) in ev["rank"][kind]:
            Jm = seq_cost(model, s_th, s_thd, A)
            rhos.append(stats.spearmanr(Jm, Jt).statistic)
            regs.append((Jt[np.argmin(Jm)] - Jt.min()) / (Jt.std() + 1e-9))
        m[f"rank_rho_{kind}"] = float(np.nanmean(rhos)); m[f"regret_{kind}"] = float(np.mean(regs))
    return m


def calibrate(target, kind, ev, sign):
    """find magnitude so that one-step uniform RMSE == target"""
    lo, hi = 0.0, 1.0 if kind == "grav" else 0.2
    for _ in range(40):
        mid = (lo + hi) / 2
        d = Dyn(grav=1 + sign * mid) if kind == "grav" else Dyn(bias=sign * mid)
        if state_err(d, *ev["uni"]) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


rows = []  # per (seed, model)
dagger = {}
true_ret, zero_ret = [], []
for sd in SEEDS:
    g = rng(500 + sd)
    th0 = np.pi + g.uniform(-0.1, 0.1)
    R_true, traj = episode(TRUE, rng(600 + sd), th0, record=True)
    R_zero = zero_return(th0)
    true_ret.append(R_true); zero_ret.append(R_zero)
    # evaluation sets
    ge = rng(700 + sd)
    n = 2000
    ev = {"uni": (ge.uniform(-np.pi, np.pi, n), ge.uniform(-MAXV, MAXV, n), ge.uniform(-MAXU, MAXU, n))}
    on_th = np.array([t[0] for t in traj]); on_thd = np.array([t[1] for t in traj]); on_u = np.array([t[2][0] for t in traj])
    ev["onp"] = (on_th, on_thd, on_u)
    ev["roll0"] = (ge.uniform(-np.pi, np.pi, 200), ge.uniform(-1, 1, 200)); ev["rollU"] = ge.uniform(-MAXU, MAXU, (200, 30))
    idx = np.linspace(0, EP_LEN - 1, N_RANK_STATES).astype(int)
    ev["rank"] = {"global": [], "local": []}
    for i in idx:
        s_th, s_thd, plan = traj[i]
        for kind in ("global", "local"):
            A = ge.standard_normal((N_CAND, CEM["H"])) if kind == "global" else plan + 0.3 * ge.standard_normal((N_CAND, CEM["H"]))
            A = np.clip(A, -MAXU, MAXU)
            ev["rank"][kind].append((s_th, s_thd, A, seq_cost(TRUE, s_th, s_thd, A)))
    # model zoo
    zoo = {}
    D200, D2000 = collect_random(200, rng(800 + sd)), collect_random(2000, rng(900 + sd))
    for M in (25, 100):
        for N, D in ((200, D200), (2000, D2000)):
            zoo[f"RFF{M}_N{N}"] = ("learned", Ridge("rff", M, rng(1000 + sd + M)).fit(*D.T))
    zoo["linear_N2000"] = ("learned", Ridge("linear").fit(*D2000.T))
    zoo["phys_N2000"] = ("learned", Ridge("phys").fit(*D2000.T))
    for lvl in GAIN_LEVELS:
        for sgn in (-1, 1):
            dg = Dyn(gain=1 + sgn * lvl)
            zoo[f"gain{1+sgn*lvl:.1f}"] = ("diff", dg)
            tgt = state_err(dg, *ev["uni"])
            zoo[f"grav_m{lvl}{'+' if sgn > 0 else '-'}"] = ("common", Dyn(grav=1 + sgn * calibrate(tgt, "grav", ev, 1)))
            zoo[f"bias_m{lvl}{'+' if sgn > 0 else '-'}"] = ("common", Dyn(bias=sgn * calibrate(tgt, "bias", ev, 1)))
    for name, (typ, model) in zoo.items():
        opt = []
        R, _ = episode(model, rng(600 + sd), th0, optim=opt)
        m = metrics(model, ev)
        m.update(seed=sd, model=name, type=typ, ret=R, nret=(R - R_zero) / (R_true - R_zero),
                 optimism_onpolicy=float(np.mean(opt)), optimism_onpolicy_abs=float(np.mean(np.abs(opt))))
        rows.append(m)
    # DAgger intervention: pre-registered on RFF(25); post-hoc also on misspecified linear model
    for cls in ("rff25", "linear"):
        for arm in ("random", "dagger"):
            gd = rng(1100 + sd)
            mk = (lambda: Ridge("rff", 25, rng(1200 + sd))) if cls == "rff25" else (lambda: Ridge("linear"))
            if arm == "random":
                D = collect_random(2000, gd)
                model = mk().fit(*D.T)
            else:
                D = collect_random(1200, gd)
                model = mk().fit(*D.T)
                for it in range(2):
                    new = []
                    for j in range(2):
                        _, tr = episode(model, rng(1300 + 10 * sd + 2 * it + j), gd.uniform(-np.pi, np.pi), noise=0.3)
                        new += tr
                    D = np.concatenate([D, np.array(new)])
                    model = mk().fit(*D.T)
            R, _ = episode(model, rng(600 + sd), th0)
            key = arm if cls == "rff25" else "lin_" + arm
            dagger.setdefault(key, []).append((R - R_zero) / (R_true - R_zero))
            dagger.setdefault(key + "_mse", []).append(state_err(model, *ev["uni"]))
            dagger.setdefault(key + "_onp", []).append(state_err(model, *ev["onp"]))
    print(f"seed {sd} done {time.time()-T0:.1f}s", flush=True)

# ------------------------------------------------------------------ analysis
METRICS = ["mse1_uniform", "mse1_onpolicy", "rollout30", "rank_rho_global", "rank_rho_local", "regret_global", "regret_local",
           "optimism_onpolicy", "optimism_onpolicy_abs", "optimism_offline"]
out = {"config": dict(seeds=SEEDS, cem=CEM, n_rank_states=N_RANK_STATES, n_cand=N_CAND, gain_levels=GAIN_LEVELS),
       "true_return": mean_ci(true_ret), "zero_return": mean_ci(zero_ret)}
corr = {k: [] for k in METRICS}
for sd in SEEDS:
    rs = [r for r in rows if r["seed"] == sd]
    y = np.array([r["nret"] for r in rs])
    for k in METRICS:
        x = np.array([r[k] for r in rs])
        corr[k].append(stats.spearmanr(x, y).statistic)
out["spearman_with_return"] = {k: mean_ci(v) for k, v in corr.items()}
out["spearman_per_seed"] = corr
absd = lambda a, b: np.abs(corr[a]) - np.abs(corr[b])  # noqa: E731
out["tests"] = {}
for a, b in [("regret_local", "mse1_uniform"), ("regret_local", "rollout30"), ("rank_rho_local", "mse1_uniform"),
             ("regret_local", "regret_global"), ("mse1_onpolicy", "mse1_uniform"),
             ("optimism_onpolicy", "mse1_uniform"), ("optimism_onpolicy", "regret_local"),
             ("optimism_offline", "mse1_uniform")]:
    d = absd(a, b)
    out["tests"][f"|rho|({a}) - |rho|({b})"] = {"diff": mean_ci(d), "p": float(stats.ttest_1samp(d, 0).pvalue)}
# per-model summary
names = sorted(set(r["model"] for r in rows), key=lambda s: [r["model"] for r in rows].index(s))
out["models"] = {}
for nm in names:
    rs = [r for r in rows if r["model"] == nm]
    out["models"][nm] = {"type": rs[0]["type"], **{k: mean_ci([r[k] for r in rs]) for k in METRICS + ["nret"]}}
# matched-MSE comparison: differential (gain) vs common-mode (grav, bias)
mm = {}
for lvl in GAIN_LEVELS:
    for kind in ("grav", "bias"):
        dif, com_, dr, cr, dm, cm = [], [], [], [], [], []
        for sd in SEEDS:
            for sgn, tag in ((-1, "-"), (1, "+")):
                rg = next(r for r in rows if r["seed"] == sd and r["model"] == f"gain{1+sgn*lvl:.1f}")
                rc = next(r for r in rows if r["seed"] == sd and r["model"] == f"{kind}_m{lvl}{tag}")
                dif.append(rg["nret"]); com_.append(rc["nret"]); dr.append(rg["regret_local"]); cr.append(rc["regret_local"])
                dm.append(rg["mse1_uniform"]); cm.append(rc["mse1_uniform"])
        mm[f"level{lvl}_gain_vs_{kind}"] = {
            "mse_gain": mean_ci(dm), "mse_common": mean_ci(cm),
            "nret_gain": mean_ci(dif), "nret_common": mean_ci(com_),
            "regret_local_gain": mean_ci(dr), "regret_local_common": mean_ci(cr),
            "nret_diff_common_minus_gain": mean_ci(np.array(com_) - np.array(dif)),
            "p_nret": float(stats.ttest_rel(com_, dif).pvalue),
            "p_regret": float(stats.ttest_rel(cr, dr).pvalue)}
out["matched_mse"] = mm
out["dagger"] = {k: mean_ci(v) for k, v in dagger.items()}
out["dagger"]["nret_diff"] = mean_ci(np.array(dagger["dagger"]) - np.array(dagger["random"]))
out["dagger"]["p_nret"] = float(stats.ttest_rel(dagger["dagger"], dagger["random"]).pvalue)
out["dagger"]["mse_ratio"] = mean_ci(np.array(dagger["dagger_mse"]) / np.array(dagger["random_mse"]))
out["dagger"]["onp_ratio"] = mean_ci(np.array(dagger["dagger_onp"]) / np.array(dagger["random_onp"]))
out["dagger"]["lin_nret_diff"] = mean_ci(np.array(dagger["lin_dagger"]) - np.array(dagger["lin_random"]))
out["dagger"]["lin_p_nret"] = float(stats.ttest_rel(dagger["lin_dagger"], dagger["lin_random"]).pvalue)
out["dagger"]["lin_mse_ratio"] = mean_ci(np.array(dagger["lin_dagger_mse"]) / np.array(dagger["lin_random_mse"]))
out["dagger"]["lin_onp_ratio"] = mean_ci(np.array(dagger["lin_dagger_onp"]) / np.array(dagger["lin_random_onp"]))
fail = [r for r in rows if r["nret"] < 0.9]
out["failure_analysis"] = {"n_rows": len(rows), "n_fail": len(fail),
                           "frac_fail_with_positive_optimism": float(np.mean([r["optimism_onpolicy"] > 0 for r in fail])) if fail else None,
                           "frac_nonfail_with_positive_optimism": float(np.mean([r["optimism_onpolicy"] > 0 for r in rows if r["nret"] >= 0.9]))}
out["dagger"]["per_seed"] = dagger
out["rows"] = rows
out["runtime_s"] = time.time() - T0
od = results_dir(__file__)
(od / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))

# ------------------------------------------------------------------ figures
P = plt()
fig, ax = P.subplots(1, 3, figsize=(15, 4.3))
cols = {"learned": "C0", "diff": "C3", "common": "C2"}
lab = {"learned": "学习模型", "diff": "差分型扰动(动作增益)", "common": "共模型扰动(重力/偏置)"}
for k, (xk, xl) in enumerate([("mse1_uniform", "均匀分布一步 RMSE"), ("rollout30", "30 步开环 RMSE"), ("regret_local", "局部排序后悔（标准差单位）")]):
    for t in cols:
        rs = [r for r in rows if r["type"] == t]
        ax[k].scatter([max(r[xk], 1e-5) for r in rs], [r["nret"] for r in rs], s=12, alpha=0.6, color=cols[t], label=lab[t])
    ax[k].set_xscale("log"); ax[k].set_xlabel(xl); ax[k].set_ylabel("归一化回报")
    ax[k].set_title(f"Spearman ρ = {out['spearman_with_return'][xk][0]:.2f}±{out['spearman_with_return'][xk][1]:.2f}")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(od / "frontier_fidelity_vs_utility.png"); P.close(fig)

fig, ax = P.subplots(1, 2, figsize=(11, 4))
keys = list(mm)
x = np.arange(len(keys))
ax[0].bar(x - 0.2, [mm[k]["nret_gain"][0] for k in keys], 0.4, yerr=[mm[k]["nret_gain"][1] for k in keys], label="动作增益误差", capsize=3)
ax[0].bar(x + 0.2, [mm[k]["nret_common"][0] for k in keys], 0.4, yerr=[mm[k]["nret_common"][1] for k in keys], label="匹配 RMSE 的共模误差", capsize=3)
ax[0].set_xticks(x); ax[0].set_xticklabels(keys, fontsize=7, rotation=15); ax[0].set_ylabel("归一化回报"); ax[0].legend(fontsize=8)
ax[0].set_title("一步 RMSE 相同，决策效用不同")
arms = ["random", "dagger"]
ax[1].bar([0, 1], [out["dagger"][a][0] for a in arms], yerr=[out["dagger"][a][1] for a in arms], capsize=4, color=["C7", "C1"])
ax2 = ax[1].twinx()
ax2.plot([0, 1], [out["dagger"][a + "_mse"][0] for a in arms], "ks--", label="均匀一步 RMSE")
ax2.plot([0, 1], [out["dagger"][a + "_onp"][0] for a in arms], "k^:", label="策略分布一步 RMSE")
ax2.set_ylabel("一步 RMSE"); ax2.legend(fontsize=7, loc="upper left")
ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["2000 随机", "1200 随机+800 在线(DAgger)"]); ax[1].set_ylabel("归一化回报")
ax[1].set_title("RFF(25)：决策感知的数据分配")
fig.tight_layout(); fig.savefig(od / "frontier_matched_and_dagger.png"); P.close(fig)
print(f"done {time.time()-T0:.1f}s")
