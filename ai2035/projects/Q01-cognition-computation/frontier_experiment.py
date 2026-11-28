"""Q01 前沿实验：softmax 注意力作为稠密联想记忆——长上下文检索的相图、键相关性与温度缩放。

理论（随机能量模型 REM 近似，见 paper.md §4）：
  键 ξ_μ = norm(√c·u + √(1-c)·z_μ)（共享方向 u 引入两两相关 c），查询 q = norm(ξ_t + ν g)。
  目标重叠 m_t ≈ 1/Z，干扰重叠 m_μ ~ N(c/Z, s²)，s² = ((1-c)²+ν²(1-c))/(N Z²)（共模噪声抵消），Z = √(1+ν²)。
  干扰配分函数 ln Z_d = ln P + βμ + β²s²/2（非冻结，β < √(2lnP)/s），否则 β(μ + s√(2lnP))（冻结）。
  检索成功（目标注意力权重 > 1/2）⇔ ln Z_d < β m_t。由此得到临界上下文长度 P*(β, c, ν)，
  其最大值 ln P_max = N(1-c)/(2(1-c+ν²))；温度缩放 β = κ ln P 在 κ = 2/Δ 时达到 P_max。
实验：
  E1 β–P 相图（c=0, ν=0.5, N=64）：模拟 50% 成功的 ln P50 与理论 ln P* 对比。
  E2 键相关性 c 与查询噪声 ν：固定 β=8 时 ln P50 随 c 的变化；“β/(1-c) 重标”是否使曲线塌缩。
  E3 长上下文：固定 β vs 对数长度缩放 β=κ ln P（Scalable-Softmax 式）在 P=16…131072 上的检索率；
     并记录失败时的“混合/原型态”（输出与共享方向 u 的余弦）。
运行：python ai2035/projects/Q01-cognition-computation/frontier_experiment.py
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
N = 64
SEEDS = 10
B = 100  # 每个上下文中的查询数
PS = np.unique(np.round(2 ** np.arange(2, 17.01, 0.5)).astype(int))  # 4 … 131072


# ---------------------------------------------------------------- 理论
def theory_params(c, nu, n=N):
    Z = np.sqrt(1 + nu ** 2)
    mt = 1 / Z
    mu = c / Z
    # 查询噪声中沿共享方向 u 的分量对所有键（含目标）相同，在 softmax 中抵消，只剩 ν²(1-c) 的独立部分
    s = np.sqrt(((1 - c) ** 2 + nu ** 2 * (1 - c)) / n) / Z
    return mt, mu, s


def lnZd(beta, lnP, mu, s):
    bc = np.sqrt(2 * lnP) / s
    return np.where(beta < bc, lnP + beta * mu + beta ** 2 * s ** 2 / 2, beta * (mu + s * np.sqrt(2 * lnP)))


def theory_lnPstar(beta, c, nu, n=N):
    """满足 lnZd(β, lnP) = β m_t 的最大 lnP（对 lnP 二分；lnZd 对 lnP 单调增）。"""
    mt, mu, s = theory_params(c, nu, n)
    lo, hi = 1e-6, 200.0
    if lnZd(beta, lo, mu, s) >= beta * mt:
        return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if lnZd(beta, mid, mu, s) < beta * mt:
            lo = mid
        else:
            hi = mid
    return lo


def theory_lnPmax(c, nu, n=N):
    return n * (1 - c) / (2 * ((1 - c) + nu ** 2))


# ---------------------------------------------------------------- 模拟
def make_context(g, P, c, nu):
    u = g.standard_normal(N); u /= np.linalg.norm(u)
    Zr = g.standard_normal((P, N)); Zr /= np.linalg.norm(Zr, axis=1, keepdims=True)
    X = np.sqrt(c) * u + np.sqrt(1 - c) * Zr
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    nb = min(B, P)
    Q = X[:nb] + nu * g.standard_normal((nb, N)) / np.sqrt(N)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    return X, Q, u


def retrieval_stats(X, Q, u, betas):
    """返回每个 β 的 (成功率[w_t>1/2], top1, 输出与目标余弦, 输出与原型 u 余弦)。"""
    A = Q @ X.T  # (nb, P)
    nb = Q.shape[0]
    idx = np.arange(nb)
    out = []
    for b in betas:
        Zs = b * A
        Zs -= Zs.max(1, keepdims=True)
        W = np.exp(Zs)
        W /= W.sum(1, keepdims=True)
        wt = W[idx, idx]
        R = W @ X
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        out.append((np.mean(wt > 0.5), np.mean(np.argmax(A, 1) == idx), np.mean((Rn * X[:nb]).sum(1)), np.mean(Rn @ u)))
    return np.array(out)  # (len(betas), 4)


def p50_crossing(lnPs, succ):
    for k in range(1, len(succ)):
        if succ[k - 1] >= 0.5 > succ[k]:
            return lnPs[k - 1] + (succ[k - 1] - 0.5) * (lnPs[k] - lnPs[k - 1]) / (succ[k - 1] - succ[k])
    return np.nan if succ[0] < 0.5 else np.inf


def sweep(c, nu, betas, seed_base, Ps=PS):
    S = np.zeros((SEEDS, len(Ps), len(betas), 4))
    for s in range(SEEDS):
        for j, P in enumerate(Ps):
            g = rng(seed_base + 7919 * s + P)
            X, Q, u = make_context(g, P, c, nu)
            S[s, j] = retrieval_stats(X, Q, u, betas)
    return S


if __name__ == "__main__":
    plt = get_plt()
    lnPs = np.log(PS - 1)  # 干扰数 P-1
    res = {"N": N, "seeds": SEEDS, "queries_per_context": B, "Ps": PS}

    # ---------------- E1 相图
    betas1 = np.array([2, 3, 4, 6, 8, 10, 12, 16], float)
    c1, nu1 = 0.0, 0.5
    S1 = sweep(c1, nu1, betas1, 100)
    e1 = {"c": c1, "nu": nu1, "betas": betas1, "rows": []}
    sim, th = [], []
    for k, b in enumerate(betas1):
        per = [p50_crossing(lnPs, S1[s, :, k, 0]) for s in range(SEEDS)]
        per = np.array(per)
        finite = per[np.isfinite(per)]
        t = theory_lnPstar(b, c1, nu1)
        row = {"beta": b, "lnP50_sim": mean_ci(finite) if len(finite) >= 2 else None,
               "n_censored_above": int(np.sum(np.isinf(per))), "lnP50_theory": t,
               "frozen_at_theory_point": bool(b > np.sqrt(2 * max(t, 1e-9)) / theory_params(c1, nu1)[2])}
        e1["rows"].append(row)
        if len(finite) >= 2:
            sim.append(finite.mean()); th.append(t)
    sim, th = np.array(sim), np.array(th)
    e1["lnPmax_theory"] = theory_lnPmax(c1, nu1)
    e1["beta_opt_theory"] = float((1 / np.sqrt(1 + nu1 ** 2)) / theory_params(c1, nu1)[2] ** 2)
    e1["mean_abs_err_lnP"] = float(np.mean(np.abs(sim - th)))
    e1["max_abs_err_lnP"] = float(np.max(np.abs(sim - th)))
    e1["pearson_sim_theory"] = float(stats.pearsonr(sim, th)[0])
    # top-1（硬检索，β→∞）的 50% 点 vs 冻结极限
    t1 = np.array([p50_crossing(lnPs, S1[s, :, 0, 1]) for s in range(SEEDS)])
    mt, mu, s_ = theory_params(c1, nu1)
    e1["top1_lnP50_sim"] = mean_ci(t1[np.isfinite(t1)]) if np.sum(np.isfinite(t1)) >= 2 else None
    e1["top1_lnP50_theory_frozen"] = float(((mt - mu) / s_) ** 2 / 2)
    res["E1_phase_diagram"] = e1

    # ---------------- E2 相关性
    cs = [0.0, 0.3, 0.6, 0.8]
    nus = [0.0, 0.5]
    beta2 = 8.0
    e2 = {"beta": beta2, "cs": cs, "nus": nus, "rows": []}
    PS2 = PS[PS <= 4096]
    lnPs2 = np.log(PS2 - 1)
    for nu in nus:
        for c in cs:
            bpair = np.array([beta2, beta2 / (1 - c)])
            S2 = sweep(c, nu, bpair, 2000 + int(100 * c) + int(10 * nu), Ps=PS2)
            for k, mode in enumerate(["fixed", "rescaled"]):
                b = bpair[k]
                per = np.array([p50_crossing(lnPs2, S2[s, :, k, 0]) for s in range(SEEDS)])
                fin = per[np.isfinite(per)]
                e2["rows"].append({"nu": nu, "c": c, "mode": mode, "beta": b,
                                   "lnP50_sim": mean_ci(fin) if len(fin) >= 2 else None,
                                   "n_censored_above": int(np.sum(np.isinf(per))),
                                   "lnP50_theory": theory_lnPstar(b, c, nu),
                                   "lnPmax_theory": theory_lnPmax(c, nu)})
    # 理论初版（未考虑共模噪声抵消，s² = ((1-c)²+ν²)/(N Z²)）在 ν=0.5、c=0.8、β=40 的预测，用于如实披露
    Zv = np.sqrt(1.25); mtv, muv, sv = 1 / Zv, 0.8 / Zv, np.sqrt((0.04 + 0.25) / N) / Zv
    lo, hi = 1e-6, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if lnZd(40.0, mid, muv, sv) < 40.0 * mtv else (lo, mid)
    e2["theory_v1_uncorrected_nu0.5_c0.8_beta40"] = lo
    errs = np.array([abs(r["lnP50_sim"][0] - r["lnP50_theory"]) for r in e2["rows"] if r["lnP50_sim"]])
    e2["mean_abs_err_lnP"] = float(errs.mean()); e2["max_abs_err_lnP"] = float(errs.max())
    res["E2_correlation"] = e2

    # ---------------- E3 长上下文：固定 β vs β=κ lnP
    c3, nu3 = 0.3, 0.5
    mt3, mu3, s3 = theory_params(c3, nu3)
    Delta3 = mt3 - mu3
    kappa = 2 / Delta3  # 理论最优 κ（预先由 §4 推导确定）
    fixed_betas = [8.0, 16.0]
    e3 = {"c": c3, "nu": nu3, "Delta": Delta3, "kappa": kappa, "lnPmax_theory": theory_lnPmax(c3, nu3),
          "fixed_betas": fixed_betas, "succ": {}, "proto_cos": {}, "target_cos": {}}
    succ = {f"fixed{b:g}": np.zeros((SEEDS, len(PS))) for b in fixed_betas}
    succ["scaled"] = np.zeros((SEEDS, len(PS)))
    proto = {k: np.zeros((SEEDS, len(PS))) for k in succ}
    tcos = {k: np.zeros((SEEDS, len(PS))) for k in succ}
    for s in range(SEEDS):
        for j, P in enumerate(PS):
            g = rng(5000 + 7919 * s + P)
            X, Q, u = make_context(g, P, c3, nu3)
            bs = np.array(fixed_betas + [kappa * np.log(P - 1 if P > 2 else 2)])
            r = retrieval_stats(X, Q, u, bs)
            for k, key in enumerate([f"fixed{b:g}" for b in fixed_betas] + ["scaled"]):
                succ[key][s, j] = r[k, 0]; tcos[key][s, j] = r[k, 2]; proto[key][s, j] = r[k, 3]
    for key in succ:
        e3["succ"][key] = {"mean": succ[key].mean(0), "ci": [mean_ci(succ[key][:, j])[1] for j in range(len(PS))]}
        e3["proto_cos"][key] = proto[key].mean(0)
        e3["target_cos"][key] = tcos[key].mean(0)
        e3[f"lnP50_{key}"] = mean_ci(np.array([p50_crossing(lnPs, succ[key][s]) for s in range(SEEDS)]))
    for b in fixed_betas:
        e3[f"lnPstar_theory_fixed{b:g}"] = theory_lnPstar(b, c3, nu3)
    # 理论：缩放 β 下成功直到 lnP ≈ lnPmax（=Δ²/(2s²)）
    e3["theory_scaled_lnP_limit"] = float(Delta3 ** 2 / (2 * s3 ** 2))
    jbig = len(PS) - 1
    e3["at_Pmax_grid"] = {"P": int(PS[jbig]), "scaled": mean_ci(succ["scaled"][:, jbig]),
                          "fixed8": mean_ci(succ["fixed8"][:, jbig]), "fixed16": mean_ci(succ["fixed16"][:, jbig])}
    wins = int(np.sum(succ["scaled"][:, jbig] > succ["fixed16"][:, jbig]))
    e3["sign_test_scaled_vs_fixed16_at_Pmax"] = {"wins": wins, "n": SEEDS,
                                                 "p": float(stats.binomtest(wins, SEEDS, 0.5, alternative="greater").pvalue)}
    res["E3_long_context"] = e3
    res["runtime_sec"] = time.time() - T0

    # ---------------- 图
    fig, ax = plt.subplots(figsize=(6, 4.3))
    bb = np.linspace(1, 40, 200)
    ax.plot(bb, [theory_lnPstar(b, c1, nu1) for b in bb], "k-", label="REM 理论 ln P*(β)")
    ys = [r["lnP50_sim"][0] if r["lnP50_sim"] else np.nan for r in e1["rows"]]
    es = [r["lnP50_sim"][1] if r["lnP50_sim"] else 0 for r in e1["rows"]]
    ax.errorbar(betas1, ys, yerr=es, fmt="o", color="C3", label="模拟 ln P50（10 种子）")
    ax.axhline(e1["lnPmax_theory"], color="gray", ls=":", label="理论上限 ln P_max")
    ax.axhline(np.log(PS[-1]), color="C0", ls="--", lw=0.8, label="模拟网格上限")
    ax.set_xlabel("逆温度 β"); ax.set_ylabel("可检索的上下文长度 ln P")
    ax.set_title(f"注意力检索相图（N={N}, c=0, ν=0.5）"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "frontier_phase_diagram.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, nu in enumerate(nus):
        for mode, mk in [("fixed", "o-"), ("rescaled", "s--")]:
            rows = [r for r in e2["rows"] if r["nu"] == nu and r["mode"] == mode]
            ax[i].plot(cs, [r["lnP50_sim"][0] if r["lnP50_sim"] else np.nan for r in rows], mk,
                       label=f"模拟 β={'8' if mode=='fixed' else '8/(1-c)'}")
            ax[i].plot(cs, [r["lnP50_theory"] for r in rows], "k:" if mode == "fixed" else "k-.", lw=1,
                       label=f"理论 β={'8' if mode=='fixed' else '8/(1-c)'}")
        ax[i].set_xlabel("键两两相关 c"); ax[i].set_ylabel("ln P50"); ax[i].set_title(f"查询噪声 ν={nu}"); ax[i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "frontier_correlation.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, lab in [("fixed8", "固定 β=8"), ("fixed16", "固定 β=16"), ("scaled", f"β=κ·lnP（κ={kappa:.2f}）")]:
        ax[0].semilogx(PS, succ[key].mean(0), "o-", ms=3, label=lab)
        ax[1].semilogx(PS, proto[key].mean(0), "o-", ms=3, label=lab)
    ax[0].axvline(np.exp(e3["lnPmax_theory"]), color="gray", ls=":", label="理论 P_max")
    ax[0].set_xlabel("上下文长度 P"); ax[0].set_ylabel("检索成功率（目标权重>1/2）"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"长上下文检索（c={c3}, ν={nu3}）")
    ax[1].set_xlabel("上下文长度 P"); ax[1].set_ylabel("输出与共享原型 u 的余弦"); ax[1].set_title("失败模式：滑向原型/混合态")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "frontier_long_context.png"); plt.close(fig)

    p = OUT / "frontier.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_jsonable))
    print("saved", p, time.time() - T0)
    for r in e1["rows"]:
        print(r)
    print(e1["mean_abs_err_lnP"], e1["max_abs_err_lnP"], e1["top1_lnP50_sim"], e1["top1_lnP50_theory_frozen"])
    for r in e2["rows"]:
        print(r)
    print({k: v for k, v in e3.items() if k.startswith("lnP") or k in ("at_Pmax_grid", "sign_test_scaled_vs_fixed16_at_Pmax", "kappa", "theory_scaled_lnP_limit")})
