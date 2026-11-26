"""Q01 认知与计算的关系 —— Hopfield 联想记忆作为“认知↔计算”的最小桥梁。

实验 A：经典 Hebbian Hopfield 网络的存储容量 α_c 与有限尺寸效应（N=100/200/400）。
实验 B：稠密联想记忆（Krotov–Hopfield 多项式能量 n=2,3,4；Demircigil 指数能量）
        容量 P_c 随 N 的标度（n=2 即经典 Hebbian，作为对照）。
实验 C：线索破坏（cue corruption）下的检索稳健性（吸引域）。
实验 D：连续型现代 Hopfield 一步更新 = softmax 注意力；逆温度 β 的作用，
        与线性（经典 Hebbian 一步）读出对照。

运行：python ai2035/projects/Q01-cognition-computation/experiment.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt as get_plt, results_dir, rng, save_results  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)
SUCC_M = 0.95  # 检索成功判据：收敛态与目标模式的重叠 m >= 0.95（预先固定）


def rand_patterns(g, P, N):
    return g.choice(np.array([-1.0, 1.0]), size=(P, N))


def corrupt(g, X, rho):
    flip = g.random(X.shape) < rho
    return np.where(flip, -X, X)


# ---------------------------------------------------------------- 动力学
def run_dense(X, S, kind, g, max_sweeps=15):
    """异步更新的稠密联想记忆。X:(P,N) 存储模式；S:(B,N) 初始态。

    能量 E = -sum_mu F(xi_mu . sigma)；kind = ('poly', n) 或 ('exp', None)。
    n=2 时与零对角 Hebbian 规则 W = X^T X 完全等价（经典 Hopfield）。
    """
    S = S.copy()
    M = S @ X.T  # (B,P) 各模式重叠
    N = X.shape[1]
    for _ in range(max_sweeps):
        changed = 0
        for i in g.permutation(N):
            xi = X[:, i]
            si = S[:, i]
            r = M - si[:, None] * xi[None, :]  # 去掉第 i 位后的重叠
            if kind[0] == "poly":
                # F(x)=x^n：sum_mu [(r+xi)^n-(r-xi)^n] = sum_mu xi*g(r)，利用 xi^2=1 化简（正常数因子不影响符号）
                n = kind[1]
                if n == 2:
                    gr = r
                elif n == 3:
                    gr = 3 * r * r + 1
                else:
                    gr = r * (r * r + 1)
                h = gr @ xi
            else:  # exp：sum_mu e^{r}(e^{xi}-e^{-xi}) ∝ sum_mu e^{r-max} xi
                w = np.exp(r - r.max(1, keepdims=True))
                h = w @ xi
            new = np.where(h > 0, 1.0, np.where(h < 0, -1.0, si))
            d = new - si
            if np.any(d):
                changed += int(np.count_nonzero(d))
                M += d[:, None] * xi[None, :]
                S[:, i] = new
        if changed == 0:
            break
    return S


def overlaps(S, T):
    return (S * T).sum(1) / S.shape[1]


# ---------------------------------------------------------------- 实验 A
def exp_A():
    Ns = [100, 200, 400]
    alphas = np.round(np.arange(0.06, 0.451, 0.01), 3)
    seeds = 6
    succ = {N: np.zeros((seeds, len(alphas))) for N in Ns}
    mbar = {N: np.zeros((seeds, len(alphas))) for N in Ns}
    for N in Ns:
        for s in range(seeds):
            g = rng(1000 + 37 * s + N)
            for j, a in enumerate(alphas):
                P = max(1, int(round(a * N)))
                X = rand_patterns(g, P, N)
                idx = g.choice(P, size=min(P, 60), replace=False)  # 每个网络最多检验 60 个模式
                S = run_dense(X, X[idx], ("poly", 2), g)
                m = overlaps(S, X[idx])
                succ[N][s, j] = np.mean(m >= SUCC_M)
                mbar[N][s, j] = m.mean()

    def crossing(y, x, level=0.5):
        # 第一次跌破 level 的线性插值位置
        for k in range(1, len(y)):
            if y[k - 1] >= level > y[k]:
                return x[k - 1] + (y[k - 1] - level) * (x[k] - x[k - 1]) / (y[k - 1] - y[k])
        return np.nan

    res = {"alphas": alphas, "seeds": seeds, "N": {}}
    for N in Ns:
        ac = [crossing(succ[N][s], alphas, 0.5) for s in range(seeds)]
        a90 = [crossing(succ[N][s], alphas, 0.9) for s in range(seeds)]
        a10 = [crossing(succ[N][s], alphas, 0.1) for s in range(seeds)]
        width = np.array(a10) - np.array(a90)
        res["N"][str(N)] = {
            "success_mean": succ[N].mean(0), "overlap_mean": mbar[N].mean(0),
            "alpha_c50": mean_ci(ac), "alpha_c50_per_seed": ac,
            "alpha_90": mean_ci(a90), "alpha_10": mean_ci(a10),
            "width_90_10": mean_ci(width), "width_per_seed": width,
        }
    # 有限尺寸：宽度随 N 下降的趋势检验（Spearman）与 α_c(N) 外推到 1/N→0
    from scipy import stats
    Ws = np.concatenate([res["N"][str(N)]["width_per_seed"] for N in Ns])
    Nv = np.repeat(Ns, seeds)
    rho, p = stats.spearmanr(Nv, Ws)
    res["width_vs_N_spearman"] = {"rho": rho, "p": p}
    acs = np.array([res["N"][str(N)]["alpha_c50"][0] for N in Ns])
    sl, ic, r, pp, se = stats.linregress(1 / np.sqrt(np.array(Ns)), acs)
    res["alpha_c_extrap_sqrtN"] = {"intercept": ic, "slope": sl, "note": "alpha_c50 = ic + sl/sqrt(N) 拟合，三点仅作粗略外推"}
    # 宽度 ~ N^-nu
    wm = np.array([res["N"][str(N)]["width_90_10"][0] for N in Ns])
    sl2, ic2, *_ = stats.linregress(np.log(Ns), np.log(wm))
    res["width_scaling_exponent"] = sl2
    return res, succ, mbar


# ---------------------------------------------------------------- 实验 B
def capacity_curve(kind, N, g, Pgrid, B=32):
    out = []
    low = 0
    for P in Pgrid:
        X = rand_patterns(g, P, N)
        idx = g.choice(P, size=min(B, P), replace=False)
        S = run_dense(X, X[idx], kind, g, max_sweeps=6)
        sr = float(np.mean(overlaps(S, X[idx]) >= SUCC_M))
        out.append(sr)
        low = low + 1 if sr < 0.05 else 0
        if low >= 2:
            break
    out += [0.0] * (len(Pgrid) - len(out))
    return np.array(out)


def exp_B():
    Ns = [8, 12, 16, 24, 32, 48, 64]
    Pgrid = np.unique(np.round(2 ** np.arange(1, 12.01, 0.5)).astype(int))  # 2..4096
    models = {"poly2(经典)": ("poly", 2), "poly3": ("poly", 3), "poly4": ("poly", 4), "exp": ("exp", None)}
    Ns_model = {"poly2(经典)": Ns, "poly3": Ns, "poly4": Ns, "exp": [6, 8, 10, 12, 14]}  # 指数模型容量增长太快，用更小的 N
    seeds = 5
    res = {"Ns": Ns, "Ns_model": Ns_model, "Pgrid": Pgrid, "seeds": seeds, "models": {}}
    for name, kind in models.items():
        Ns = Ns_model[name]
        caps = np.full((seeds, len(Ns)), np.nan)
        cens = np.zeros((seeds, len(Ns)), bool)
        curves = {}
        for k, N in enumerate(Ns):
            cs = []
            for s in range(seeds):
                g = rng(5000 + 101 * s + N)
                c = capacity_curve(kind, N, g, Pgrid)
                cs.append(c)
                lp = np.log2(Pgrid)
                pc = np.nan
                for q in range(1, len(c)):
                    if c[q - 1] >= 0.5 > c[q]:
                        pc = 2 ** (lp[q - 1] + (c[q - 1] - 0.5) * (lp[q] - lp[q - 1]) / (c[q - 1] - c[q]))
                        break
                if np.isnan(pc) and c[-1] >= 0.5:
                    cens[s, k] = True  # 网格上限内未失效
                if np.isnan(pc) and c[0] < 0.5:
                    pc = Pgrid[0]
                caps[s, k] = pc
            curves[str(N)] = np.mean(cs, 0)
        # 对未截尾的 N 拟合 log P_c = a + b log N（逐种子）
        slopes = []
        for s in range(seeds):
            ok = ~np.isnan(caps[s]) & ~cens[s] & (np.array(Ns) >= 16)
            if ok.sum() >= 3:
                slopes.append(np.polyfit(np.log(np.array(Ns)[ok]), np.log(caps[s][ok]), 1)[0])
        # 指数模型：log P_c 对 N 线性
        lin_slopes = []
        for s in range(seeds):
            ok = ~np.isnan(caps[s]) & ~cens[s]
            if ok.sum() >= 3:
                lin_slopes.append(np.polyfit(np.array(Ns)[ok], np.log(caps[s][ok]), 1)[0])
        res["models"][name] = {
            "capacity_mean": np.nanmean(np.where(cens, np.nan, caps), 0),
            "capacity_per_seed": caps, "censored": cens,
            "censored_any": cens.any(0),
            "loglog_slope": mean_ci(slopes) if slopes else None, "loglog_slope_n": len(slopes),
            "loglin_slope_perN": mean_ci(lin_slopes) if lin_slopes else None,
            "success_curves": curves,
        }
    return res


# ---------------------------------------------------------------- 实验 C
def exp_C():
    N = 100
    loads = [10, 30]
    rhos = np.round(np.arange(0.0, 0.501, 0.05), 3)
    models = {"poly2(经典)": ("poly", 2), "poly3": ("poly", 3), "exp": ("exp", None)}
    seeds = 6
    res = {"N": N, "loads": loads, "rhos": rhos, "seeds": seeds, "curves": {}}
    for P in loads:
        for name, kind in models.items():
            sr = np.zeros((seeds, len(rhos)))
            mo = np.zeros((seeds, len(rhos)))
            for s in range(seeds):
                g = rng(9000 + 13 * s + P)
                X = rand_patterns(g, P, N)
                for j, r in enumerate(rhos):
                    S = run_dense(X, corrupt(g, X, r), kind, g)
                    m = overlaps(S, X)
                    sr[s, j] = np.mean(m >= SUCC_M)
                    mo[s, j] = m.mean()
            # 吸引域半径：成功率仍 >= 0.9 的最大 rho（逐种子）
            radius = [float(rhos[np.where(sr[s] >= 0.9)[0].max()]) if np.any(sr[s] >= 0.9) else 0.0 for s in range(seeds)]
            res["curves"][f"P{P}_{name}"] = {"success": sr.mean(0), "overlap": mo.mean(0),
                                             "basin_radius90": mean_ci(radius), "basin_per_seed": radius}
    return res


# ---------------------------------------------------------------- 实验 D
def exp_D():
    N, P = 64, 256
    betas = np.array([0.5, 1, 2, 4, 8, 16, 32, 64, 128])
    noise = 0.5
    seeds = 10
    cos_soft = np.zeros((seeds, len(betas)))
    maxw = np.zeros((seeds, len(betas)))
    top1 = np.zeros((seeds, len(betas)))
    cos_lin = np.zeros(seeds)
    top1_lin = np.zeros(seeds)
    for s in range(seeds):
        g = rng(12000 + s)
        X = g.standard_normal((P, N))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        Q = X + noise * g.standard_normal((P, N)) / np.sqrt(N)  # 每个模式一个带噪查询
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        A = Q @ X.T  # (P,P)
        for j, b in enumerate(betas):
            Z = b * A
            Z -= Z.max(1, keepdims=True)
            Wt = np.exp(Z)
            Wt /= Wt.sum(1, keepdims=True)
            R = Wt @ X  # 一步更新 = softmax 注意力（Q 查询，X 为键和值）
            c = (R * X).sum(1) / np.linalg.norm(R, axis=1)
            cos_soft[s, j] = c.mean()
            maxw[s, j] = Wt.max(1).mean()
            top1[s, j] = np.mean(np.argmax(R @ X.T, 1) == np.arange(P))
        RL = A @ X  # 线性（经典 Hebbian 一步）读出：X^T X q
        cos_lin[s] = ((RL * X).sum(1) / np.linalg.norm(RL, axis=1)).mean()
        top1_lin[s] = np.mean(np.argmax(RL @ X.T, 1) == np.arange(P))
    return {
        "N": N, "P": P, "noise": noise, "betas": betas, "seeds": seeds,
        "cos_softmax": [mean_ci(cos_soft[:, j]) for j in range(len(betas))],
        "max_attention_weight": [mean_ci(maxw[:, j]) for j in range(len(betas))],
        "top1_softmax": [mean_ci(top1[:, j]) for j in range(len(betas))],
        "cos_linear_hebbian": mean_ci(cos_lin), "top1_linear_hebbian": mean_ci(top1_lin),
        "cos_query_baseline_note": "查询与目标的初始余弦约为 1/sqrt(1+noise^2)",
        "cos_query_baseline": float(1 / np.sqrt(1 + noise ** 2)),
    }


# ---------------------------------------------------------------- 主程序
if __name__ == "__main__":
    plt = get_plt()
    tA = time.time()
    rA, succA, mA = exp_A()
    tB = time.time()
    rB = exp_B()
    tC = time.time()
    rC = exp_C()
    tD = time.time()
    rD = exp_D()
    tE = time.time()

    # --- 图 A
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for N in [100, 200, 400]:
        ax[0].plot(rA["alphas"], rA["N"][str(N)]["success_mean"], "o-", ms=3, label=f"N={N}")
        ax[1].plot(rA["alphas"], rA["N"][str(N)]["overlap_mean"], "o-", ms=3, label=f"N={N}")
    for a in ax:
        a.axvline(0.138, color="k", ls="--", lw=1)
        a.set_xlabel("负载 α = P/N")
        a.legend()
    ax[0].set_ylabel(f"检索成功率 (m≥{SUCC_M})")
    ax[1].set_ylabel("平均终态重叠 m")
    ax[0].set_title("经典 Hebbian Hopfield：容量与有限尺寸效应")
    fig.tight_layout()
    fig.savefig(OUT / "A_capacity_classical.png")
    plt.close(fig)

    # --- 图 B
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, d in rB["models"].items():
        Ns = np.array(rB["Ns_model"][name])
        caps = np.array(d["capacity_per_seed"], float)
        cens = np.array(d["censored"])
        cm = np.nanmean(np.where(cens, np.nan, caps), 0)
        ax.plot(Ns, cm, "o-", label=name)
        cz = cens.any(0)
        if cz.any():
            ax.plot(Ns[cz], [rB["Pgrid"][-1]] * cz.sum(), "^", color=ax.lines[-1].get_color(), ms=9, mfc="none")
    Ns = np.array(rB["Ns"])
    ax.plot(Ns, 0.138 * Ns, "k--", lw=1, label="0.138N")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("神经元数 N"); ax.set_ylabel("容量 P_c（50% 检索成功）")
    ax.set_title("稠密联想记忆的容量标度（空心三角=网格上限截尾）")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "B_capacity_scaling.png"); plt.close(fig)

    # --- 图 C
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, P in enumerate(rC["loads"]):
        for name in ["poly2(经典)", "poly3", "exp"]:
            ax[k].plot(rC["rhos"], rC["curves"][f"P{P}_{name}"]["success"], "o-", ms=3, label=name)
        ax[k].set_title(f"N=100, P={P}（α={P/100:.1f}）")
        ax[k].set_xlabel("线索翻转比例 ρ"); ax[k].set_ylabel("检索成功率"); ax[k].legend()
    fig.tight_layout(); fig.savefig(OUT / "C_cue_robustness.png"); plt.close(fig)

    # --- 图 D
    fig, ax = plt.subplots(figsize=(6, 4))
    cs = np.array([c[0] for c in rD["cos_softmax"]])
    t1 = np.array([c[0] for c in rD["top1_softmax"]])
    ax.semilogx(rD["betas"], cs, "o-", label="softmax 一步更新：与目标余弦")
    ax.semilogx(rD["betas"], t1, "s-", label="softmax：top-1 正确率")
    ax.axhline(rD["cos_linear_hebbian"][0], color="gray", ls="--", label="线性 Hebbian 一步：余弦")
    ax.axhline(rD["cos_query_baseline"], color="k", ls=":", label="带噪查询本身的余弦")
    ax.set_xlabel("逆温度 β"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title(f"现代 Hopfield = 注意力（N={rD['N']}, P={rD['P']}）")
    fig.tight_layout(); fig.savefig(OUT / "D_attention_retrieval.png"); plt.close(fig)

    p = save_results(__file__, {
        "success_criterion_overlap": SUCC_M,
        "A_classical_capacity": rA, "B_dense_capacity": rB, "C_cue_robustness": rC, "D_attention": rD,
        "runtime_sec": {"A": tB - tA, "B": tC - tB, "C": tD - tC, "D": tE - tD, "total": time.time() - T0},
    })
    print("saved", p, "total sec", time.time() - T0)
