"""Q05 前沿实验：可验证奖励强化学习（RLVR）是“扩展推理边界”还是“锐化基础分布”？

最小模型（paper.md §3–4）：S 个“解题策略”，N 道题；策略 s 能否解题 i 由 A[i,s] ∈ {0,1} 给定
（策略通用性 q_s ~ Beta(0.3, 3)，少数通才、多数专才）。题目 i 的策略分布 π_i = softmax(θ + φ_i)：
θ 为共享参数（被 RL 训练），φ_i 为基础模型的题目特异偏好（固定）。单次成功率 p_i = Σ_s π_is A_is。
对照的“表格化”参数化：每题独立的可训练 θ_i（无参数共享）。
目标：f(p)=p（pass@1）、f(p)=2·arcsin√p（GRPO 标准差归一化的理想化）、f(p)=1−(1−p)^8（pass@8 目标）、
      以及 pass@1 + 熵正则 τ·H(π_i)。期望梯度：∇_θ f(p_i) = f'(p_i) · π_i ⊙ (A_i − p_i)（复制子动力学）。
另有采样版 GRPO（每题 G=8 个样本，组内标准化优势；组内全错时无梯度）。
指标：pass@k = mean_i 1−(1−p_i)^k，k=1…1024；边界扩展/收缩 = 基础 pass@1024 与 RL 后 pass@1024 跨越 0.5 的题目比例。
运行：python ai2035/projects/Q05-rl-theory/frontier_experiment.py
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
S, N = 200, 1000
SEEDS = 5
STEPS = 400
LR = 0.05
G = 8
KS = 2 ** np.arange(0, 11)  # 1 … 1024


def make_world(seed, zipf=1.0, phi_scale=1.5):
    g = rng(100 + seed + int(1000 * zipf))
    q = g.beta(0.3, 3.0, S)
    A = (g.random((N, S)) < q[None, :]).astype(float)
    theta0 = -zipf * np.log(np.arange(1, S + 1)) + 0.5 * g.standard_normal(S)  # 基础模型对策略的全局偏好（Zipf 型）
    theta0 = theta0[g.permutation(S)]
    phi = phi_scale * g.standard_normal((N, S))
    return A, theta0, phi, g


def probs(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def passk(p, ks=KS):
    return np.array([np.mean(1 - (1 - p) ** k) for k in ks])


class Adam:
    def __init__(self, shape):
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, x, grad):  # 梯度上升
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        return x + LR * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


def fprime(name, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    if name == "pass1" or name.startswith("ent"):
        return np.ones_like(p)
    if name == "grpo_std":
        return 1 / np.sqrt(p * (1 - p))
    if name == "passk8":
        return 8 * (1 - p) ** 7
    raise ValueError(name)


def train_exact(A, theta0, phi, name, tau=0.0, shared=True):
    th = theta0.copy() if shared else np.tile(theta0, (N, 1))
    opt = Adam(th.shape)
    for _ in range(STEPS):
        P = probs(phi + th)
        p = (P * A).sum(1)
        Gi = fprime(name, p)[:, None] * P * (A - p[:, None])  # (N,S)：每题对 θ 的梯度
        if tau > 0:
            logP = np.log(np.maximum(P, 1e-300))
            H = -(P * logP).sum(1, keepdims=True)
            Gi += tau * (-P * (logP + H))  # ∇_logits H(π) = −π ⊙ (log π + H)
        grad = Gi.mean(0) if shared else Gi
        th = opt.step(th, grad)
    P = probs(phi + th)
    return (P * A).sum(1), P


def train_grpo(A, theta0, phi, g, shared=True):
    th = theta0.copy() if shared else np.tile(theta0, (N, 1))
    opt = Adam(th.shape)
    rows = np.arange(N)
    zero_frac = []
    for _ in range(STEPS):
        P = probs(phi + th)
        cum = np.cumsum(P, 1)
        U = g.random((N, G))
        sidx = np.minimum((U[:, :, None] > cum[:, None, :]).sum(2), S - 1)  # (N,G)
        r = A[rows[:, None], sidx]
        mu = r.mean(1, keepdims=True); sd = r.std(1, keepdims=True)
        adv = np.where(sd > 0, (r - mu) / (sd + 1e-8), 0.0)
        zero_frac.append(float(np.mean(sd[:, 0] == 0)))
        # ∇ log π_i(s) = e_s − π_i
        Gi = -P * adv.sum(1, keepdims=True)
        np.add.at(Gi, (np.repeat(rows, G), sidx.ravel()), adv.ravel())
        Gi /= G
        grad = Gi.mean(0) if shared else Gi
        th = opt.step(th, grad)
    P = probs(phi + th)
    return (P * A).sum(1), float(np.mean(zero_frac))


def run_world(zipf, phi_scale, variants):
    R ={v[0]: [] for v in variants}
    R["grpo_sampled_shared"] = []; R["grpo_sampled_tabular"] = []; R["base"] = []
    extra = {"zero_adv_frac_shared": [], "zero_adv_frac_tabular": [], "expand_shrink": {}, "coverage_strategies": []}
    for s in range(SEEDS):
        A, theta0, phi, g = make_world(s, zipf, phi_scale)
        p0 = (probs(phi + theta0) * A).sum(1)
        R["base"].append(passk(p0))
        finals = {}
        for name, obj, tau, shared in variants:
            p, P = train_exact(A, theta0, phi, obj, tau, shared)
            R[name].append(passk(p)); finals[name] = p
        p, zf = train_grpo(A, theta0, phi, g, shared=True); R["grpo_sampled_shared"].append(passk(p)); finals["grpo_sampled_shared"] = p
        extra["zero_adv_frac_shared"].append(zf)
        p, zf = train_grpo(A, theta0, phi, g, shared=False); R["grpo_sampled_tabular"].append(passk(p)); finals["grpo_sampled_tabular"] = p
        extra["zero_adv_frac_tabular"].append(zf)
        # 边界扩展 / 收缩（以 pass@1024 是否 > 0.5 判定“在边界内”）
        b0 = 1 - (1 - p0) ** 1024 > 0.5
        for name, p in finals.items():
            b1 = 1 - (1 - p) ** 1024 > 0.5
            extra["expand_shrink"].setdefault(name, []).append((float(np.mean(~b0 & b1)), float(np.mean(b0 & ~b1))))
        extra["coverage_strategies"].append(float(np.mean(A.max(1))))
    res = {"world": {"zipf": zipf, "phi_scale": phi_scale}, "methods": {}}
    base = np.array(R["base"])
    for name, vals in R.items():
        v = np.array(vals)
        d = {"passk_mean": v.mean(0), "passk_ci": [mean_ci(v[:, j])[1] for j in range(len(KS))],
             "pass1": mean_ci(v[:, 0]), "pass256": mean_ci(v[:, list(KS).index(256)]), "pass1024": mean_ci(v[:, -1])}
        if name != "base":
            diff = v - base
            d["delta_pass1"] = mean_ci(diff[:, 0]); d["delta_pass256"] = mean_ci(diff[:, list(KS).index(256)])
            d["delta_pass1024"] = mean_ci(diff[:, -1])
            # 交叉点：RL − base 首次 < 0 的 k（逐种子）
            cross = []
            for s in range(SEEDS):
                neg = np.where(diff[s] < 0)[0]
                cross.append(int(KS[neg[0]]) if len(neg) else None)
            d["crossover_k_per_seed"] = cross
            d["n_seeds_pass256_below_base"] = int(np.sum(diff[:, list(KS).index(256)] < 0))
            d["n_seeds_all_k_ge_base"] = int(np.sum(np.all(diff >= -1e-9, 1)))
            d["paired_t_pass256_vs_base"] = {"t": float(stats.ttest_rel(v[:, list(KS).index(256)], base[:, list(KS).index(256)]).statistic),
                                             "p": float(stats.ttest_rel(v[:, list(KS).index(256)], base[:, list(KS).index(256)]).pvalue)}
            es = np.array(extra["expand_shrink"][name])
            d["frac_problems_expanded"] = mean_ci(es[:, 0]); d["frac_problems_shrunk"] = mean_ci(es[:, 1])
        res["methods"][name] = d
    res["zero_advantage_fraction"] = {"shared": mean_ci(extra["zero_adv_frac_shared"]), "tabular": mean_ci(extra["zero_adv_frac_tabular"])}
    res["oracle_coverage_any_strategy"] = mean_ci(extra["coverage_strategies"])
    # 帕累托检验：是否存在同时 pass@1 提升且 pass@256 不低于基础（−0.01 容差）的方法
    res["pareto_improvers"] = [n for n, d in res["methods"].items() if n != "base" and d["delta_pass1"][0] > 0 and d["delta_pass256"][0] >= -0.01]
    return res


if __name__ == "__main__":
    plt = get_plt()
    variants = [("exact_pass1_shared", "pass1", 0.0, True), ("exact_pass1_tabular", "pass1", 0.0, False),
                ("exact_grpostd_shared", "grpo_std", 0.0, True), ("exact_passk8_shared", "passk8", 0.0, True),
                ("exact_ent0.02_shared", "ent", 0.02, True), ("exact_ent0.1_shared", "ent", 0.1, True)]
    main_res = run_world(1.0, 1.5, variants)
    hard_res = run_world(2.0, 3.0, [variants[0], variants[1], variants[3], variants[5]])
    res = {"setup": {"S": S, "N": N, "seeds": SEEDS, "steps": STEPS, "lr_adam": LR, "G": G, "ks": KS,
                     "q_s": "Beta(0.3,3)", "theta0": "−zipf·log(rank)+0.5·N(0,1)，随机排列",
                     "worlds": {"main": "zipf=1.0, phi_scale=1.5", "hard": "zipf=2.0, phi_scale=3.0（基础覆盖不足，检验边界扩展）"}},
           "main": main_res, "hard": hard_res}
    res["runtime_sec"] = {"total": time.time() - T0, "cpu_process_time": time.process_time()}
    res_main = main_res

    # ---------------- 图
    R_ = res_main
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    show = [("base", "基础模型", "k-"), ("exact_pass1_shared", "pass@1 RL（共享参数）", "C0-"),
            ("grpo_sampled_shared", "GRPO 采样（共享）", "C0--"), ("exact_pass1_tabular", "pass@1 RL（表格化）", "C1-"),
            ("grpo_sampled_tabular", "GRPO 采样（表格化）", "C1--"), ("exact_grpostd_shared", "arcsin√p 目标（共享）", "C2-"),
            ("exact_passk8_shared", "pass@8 目标（共享）", "C3-"), ("exact_ent0.1_shared", "pass@1 + 熵 0.1（共享）", "C4-")]
    for name, lab, sty in show:
        ax[0].semilogx(KS, R_["methods"][name]["passk_mean"], sty, label=lab)
    ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k"); ax[0].legend(fontsize=7); ax[0].set_title("RL 前后的 pass@k 曲线（5 种子均值）")
    for name, lab, sty in show[1:]:
        d = R_["methods"][name]
        ax[1].errorbar(d["delta_pass1"][0], d["delta_pass256"][0], xerr=d["delta_pass1"][1], yerr=d["delta_pass256"][1], fmt="o", color=sty[:2], label=lab)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("Δ pass@1（相对基础）"); ax[1].set_ylabel("Δ pass@256（相对基础）"); ax[1].legend(fontsize=7)
    ax[1].set_title("锐化 vs 覆盖的权衡")
    fig.tight_layout(); fig.savefig(OUT / "frontier_rlvr_boundary.png"); plt.close(fig)

    pth = OUT / "frontier.json"
    pth.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_jsonable))
    print("saved", pth, res["runtime_sec"])
    for n, d in R_["methods"].items():
        print(n, np.round(d["pass1"], 3), np.round(d["pass256"], 3), d.get("delta_pass1"), d.get("delta_pass256"), d.get("crossover_k_per_seed"), d.get("frac_problems_expanded"), d.get("frac_problems_shrunk"))
    print(R_["zero_advantage_fraction"], R_["oracle_coverage_any_strategy"], R_["pareto_improvers"])
    for n, d in hard_res["methods"].items():
        print("HARD", n, np.round(d["pass1"], 3), np.round(d["pass256"], 3), d.get("delta_pass256"), d.get("frac_problems_expanded"), d.get("frac_problems_shrunk"))
    print("HARD", hard_res["zero_advantage_fraction"], hard_res["pareto_improvers"])
