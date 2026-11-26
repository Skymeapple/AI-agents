"""Q13 前沿实验：单次训练隐私审计（one-run auditing）能有多紧？——金丝雀干扰、白盒/黑盒与理论天花板.

DP-SGD（Poisson 采样、逐样本裁剪 C=1、高斯噪声、RDP 记账，沿用第一阶段实现）训练 digits 奇偶逻辑回归，
在一次训练中插入 m 个金丝雀，每个以 1/2 概率被纳入训练集（Steinke, Nasr & Jagielski 2023）。
金丝雀类型：
  (a) Dirac 梯度金丝雀（白盒）：被采样时向梯度和注入固定向量 C·u_c，u_c 为互相正交的额外坐标；
  (b) 干扰金丝雀（白盒）：u_c 为 k 维共享子空间中的随机单位向量（m/k 越大干扰越强）；
  (c) 输入空间金丝雀（黑盒）：标签翻转的真实图像，以最终模型上的损失为分数。
理论预测（§4）：白盒分数的信噪比 μ_eff = qT / sqrt(σ²T + Tq(1-q) + (m-1)(qT)²/(2k))，
据此由 μ-GDP 换算得到"审计天花板" ε_GDP(μ_eff)；单次审计的 Steinke 二项检验下界只能更低。
另含"多次训练"对照（单个 Dirac 金丝雀，每臂 R 次训练，Clopper–Pearson 下界）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp
from scipy.stats import beta, binom, norm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from common.util import SEED, mean_ci, plt, results_dir, rng

T0 = time.time()
DELTA = 1e-6   # 首轮用 1e-5 时 Steinke 界中的 2mδ 项(m=1000 时 0.02)超过 Bonferroni 显著性预算 0.01，使下界恒为 0；改为 1e-6（见 paper §5 偏差记录）
ORDERS = np.arange(2, 257)
SEEDS = list(range(5))
Q, STEPS, LR, CLIP = 0.02, 1000, 0.5, 1.0
EPSILONS = [1.0, 2.0, 4.0, 8.0]
BETA_CONF = 0.05


# ------------------------------- RDP accountant (stage 1) -------------------------------
def rdp_sg(q, sigma, a):
    k = np.arange(a + 1)
    logc = gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
    terms = logc + (a - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma**2)
    return float(logsumexp(terms) / (a - 1))


def eps_rdp(q, sigma, T, delta=DELTA):
    return min(T * rdp_sg(q, sigma, int(a)) + np.log(1 / delta) / (a - 1) for a in ORDERS)


def calib_sigma(eps, q, T):
    lo, hi = 0.3, 100.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if eps_rdp(q, mid, T) > eps else (lo, mid)
    return hi


def eps_from_mu(mu, delta=DELTA):
    """μ-GDP → (ε,δ)：δ(ε) = Φ(-ε/μ + μ/2) - e^ε Φ(-ε/μ - μ/2)（Dong, Roth & Su 2019）。"""
    if mu <= 0:
        return 0.0
    f = lambda e: norm.cdf(-e / mu + mu / 2) - np.exp(e) * norm.cdf(-e / mu - mu / 2) - delta
    if f(0) <= 0:
        return 0.0
    return float(brentq(f, 0, 100))


# ----------------------------------- data -----------------------------------
dg = load_digits()
Xall, yall = dg.data / 16.0, (dg.target % 2).astype(int)
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.25, stratify=yall, random_state=0)
NREAL, DIN = len(Xtr), Xtr.shape[1] + 1
Xtr_b = np.hstack([Xtr, np.ones((NREAL, 1))])
Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def dpsgd(g, sigma, extra_dim, grad_canaries=None, input_canaries=None):
    """grad_canaries: (U [m x extra_dim], included mask)；input_canaries: (Xc, yc, included mask)。
    返回最终参数 w（前 DIN 维为模型，后 extra_dim 维为金丝雀坐标）。"""
    Xd, yd = Xtr_b, ytr
    if input_canaries is not None:
        Xc, yc, inc = input_canaries
        Xd = np.vstack([Xd, Xc[inc]])
        yd = np.r_[yd, yc[inc]]
    n_records = len(Xd) + (0 if grad_canaries is None else int(grad_canaries[1].sum()))
    exp_b = Q * n_records
    w = np.zeros(DIN + extra_dim)
    if grad_canaries is not None:
        U, incg = grad_canaries
        Uin = U[incg]
    for _ in range(STEPS):
        mk = g.random(len(Xd)) < Q
        xb, yb = Xd[mk], yd[mk]
        G = (sigmoid(xb @ w[:DIN]) - yb)[:, None] * xb
        G = G / np.maximum(1.0, np.linalg.norm(G, axis=1, keepdims=True) / CLIP)
        tot = np.zeros_like(w)
        tot[:DIN] = G.sum(0)
        if grad_canaries is not None and len(Uin):
            sel = g.random(len(Uin)) < Q
            tot[DIN:] += -CLIP * Uin[sel].sum(0)          # 金丝雀梯度：把 w 推向 +u_c
        tot += sigma * CLIP * g.standard_normal(w.size)
        w -= LR * tot / exp_b
    return w


def steinke_lb(scores, inc, m, beta_conf=BETA_CONF):
    """Steinke et al. 2023 单次审计：对得分最高/最低各 r/2 个金丝雀猜"在/不在"，
    用二项尾界（pure-ε 形式）+ δ 修正项 2mδ 求 ε 的 1-β 置信下界；r 网格 Bonferroni 校正。"""
    order = np.argsort(scores)
    best = 0.0
    grid = [max(2, int(m * f)) for f in [0.05, 0.1, 0.2, 0.5, 1.0]]
    b = beta_conf / len(grid)
    for r in grid:
        h = r // 2
        guess_in, guess_out = order[-h:], order[:h]
        v = int(inc[guess_in].sum() + (~inc[guess_out]).sum())
        r_ = 2 * h

        def pval(e):
            return binom.sf(v - 1, r_, np.exp(e) / (1 + np.exp(e))) + 2 * m * DELTA

        if pval(0.0) > b:
            continue
        lo, hi = 0.0, 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if pval(mid) <= b else (lo, mid)
        best = max(best, lo)
    return best


def mu_hat(scores, inc):
    a, b = scores[inc], scores[~inc]
    return float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))


def mu_theory(sigma, m, k):
    qT = Q * STEPS
    interf = 0.0 if k is None else (m - 1) * qT**2 / (2 * k)
    return qT / np.sqrt(sigma**2 * STEPS + STEPS * Q * (1 - Q) + interf)


sigmas = {e: calib_sigma(e, Q, STEPS) for e in EPSILONS}
rec = []


def run(kind, eps, m, k, seed):
    g = rng(10_000 * seed + 100 * int(min(eps, 50)) + m // 10 + (0 if k is None else k))
    sig = 0.0 if np.isinf(eps) else sigmas[eps]
    inc = g.random(m) < 0.5
    if kind in ("dirac", "interf"):
        if kind == "dirac":
            U = np.eye(m)
            ed = m
        else:
            U = g.standard_normal((m, k))
            U /= np.linalg.norm(U, axis=1, keepdims=True)
            ed = k
        w = dpsgd(g, sig, ed, grad_canaries=(U, inc))
        scores = U @ w[DIN:]
    else:  # input-space mislabeled canaries, black-box loss score
        idx = g.choice(len(Xte), m, replace=False)          # 取自测试池的真实图像，标签翻转
        Xc, yc = Xte_b[idx], 1 - yte[idx]
        w = dpsgd(g, sig, 0, input_canaries=(Xc, yc, inc))
        p = np.clip(sigmoid(Xc @ w[:DIN]), 1e-12, 1 - 1e-12)
        scores = -(-(yc * np.log(p) + (1 - yc) * np.log(1 - p)))   # 高分 = 低损失 = 更像成员
        keep = np.ones(len(Xte), bool)
        keep[idx] = False
    acc = float(((sigmoid(Xte_b @ w[:DIN]) > 0.5) == yte).mean())
    row = dict(kind=kind, eps=eps, m=m, k=k, seed=seed, sigma=sig, acc=acc,
               mu_hat=mu_hat(scores, inc), eps_lb_steinke=steinke_lb(scores, inc, m),
               auc=float(stats.mannwhitneyu(scores[inc], scores[~inc]).statistic / (inc.sum() * (~inc).sum())))
    row["mu_theory"] = float(mu_theory(sig, m, k)) if kind != "input" else float("nan")
    row["eps_gdp_from_mu_hat"] = eps_from_mu(row["mu_hat"])
    # 高斯启发式下界（Agrawal et al. 2026 思路的简化版）：μ̂ 的 95% 单侧置信下限 → ε_GDP
    se = np.sqrt(4.0 / m + row["mu_hat"] ** 2 / (2 * m))
    row["eps_gdp_lower"] = eps_from_mu(row["mu_hat"] - 1.645 * se)
    rec.append(row)


for seed in SEEDS:
    for eps in EPSILONS:
        for m in [100, 1000]:
            run("dirac", eps, m, None, seed)
    for eps in [2.0, 8.0]:
        for k in [50, 200]:
            run("interf", eps, 1000, k, seed)
        run("input", eps, 200, None, seed)
    run("input", np.inf, 200, None, seed)
    print(f"seed {seed} {time.time()-T0:.0f}s", flush=True)

# ----------------------- multi-run baseline: single Dirac canary -----------------------
MR_RUNS = 60
multi = {}
for eps in [2.0, 8.0]:
    sc = {True: [], False: []}
    for r in range(MR_RUNS):
        for incl in (True, False):
            g = rng(90_000 + 2 * r + incl + int(eps) * 1000)
            w = dpsgd(g, sigmas[eps], 1, grad_canaries=(np.ones((1, 1)), np.array([incl])))
            sc[incl].append(w[DIN])
    a, b = np.array(sc[True]), np.array(sc[False])
    h = MR_RUNS // 2
    cands = np.unique(np.r_[a[:h], b[:h]])
    t = cands[int(np.argmax([(a[:h] >= c).mean() - (b[:h] >= c).mean() for c in cands]))]
    fn, fp, n2 = int((a[h:] < t).sum()), int((b[h:] >= t).sum()), MR_RUNS - h
    up = lambda k_, n_: 1.0 if k_ >= n_ else float(beta.ppf(1 - BETA_CONF / 2, k_ + 1, n_ - k_))
    lb = max(np.log(max(1 - DELTA - up(fp, n2), 1e-12) / up(fn, n2)), np.log(max(1 - DELTA - up(fn, n2), 1e-12) / up(fp, n2)), 0.0)
    mu_mr = float((a.mean() - b.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))))
    multi[f"eps={eps}"] = dict(runs_per_arm=MR_RUNS, total_trainings=2 * MR_RUNS, eps_lb_cp=float(lb), mu_hat=mu_mr,
                               mu_theory=float(mu_theory(sigmas[eps], 1, None)), eps_gdp_from_mu_hat=eps_from_mu(mu_mr))
print(f"multi-run {time.time()-T0:.0f}s", flush=True)


# --------------------------------- aggregation ---------------------------------
def ag(rows, key):
    m_, h = mean_ci([r[key] for r in rows])
    return {"mean": m_, "ci95": h, "n": len(rows)}


def ek(e):
    return "inf" if np.isinf(e) else str(e)


out = {"config": dict(q=Q, steps=STEPS, lr=LR, clip=CLIP, delta=DELTA, epsilons=EPSILONS, seeds=SEEDS, n_real=NREAL,
                      beta_conf=BETA_CONF, steinke_r_grid="m*{0.05,0.1,0.2,0.5,1.0}, Bonferroni, delta term 2*m*delta",
                      multi_run_runs_per_arm=MR_RUNS),
       "sigma": {ek(e): s for e, s in sigmas.items()},
       # 所有中间迭代均可见时 DP-SGD 的 GDP 中心极限近似 μ_CLT = q·sqrt(T(e^{1/σ²}-1))（Dong et al. 2019），对比末次迭代审计天花板
       "all_iterates_gdp_clt": {ek(e): {"mu": float(Q * np.sqrt(STEPS * np.expm1(1 / sigmas[e] ** 2))),
                                         "eps": eps_from_mu(float(Q * np.sqrt(STEPS * np.expm1(1 / sigmas[e] ** 2))))} for e in EPSILONS},
       "theory_ceiling_eps_gdp_dirac_m1": {ek(e): eps_from_mu(mu_theory(sigmas[e], 1, None)) for e in EPSILONS}}
groups = {}
for r in rec:
    groups.setdefault(f"{r['kind']}|eps={ek(r['eps'])}|m={r['m']}|k={r['k']}", []).append(r)
out["conditions"] = {}
for key, rows in groups.items():
    e = {k: ag(rows, k) for k in ["mu_hat", "eps_lb_steinke", "eps_gdp_from_mu_hat", "eps_gdp_lower", "auc", "acc"]}
    e["mu_theory"] = rows[0]["mu_theory"]
    e["eps_gdp_ceiling_from_mu_theory"] = eps_from_mu(rows[0]["mu_theory"]) if np.isfinite(rows[0]["mu_theory"]) else None
    e["eps_accountant"] = rows[0]["eps"] if np.isfinite(rows[0]["eps"]) else None
    out["conditions"][key] = e
out["multi_run"] = multi
T = {}
for eps in [2.0, 8.0]:
    a = [r["eps_lb_steinke"] for r in rec if r["kind"] == "dirac" and r["eps"] == eps and r["m"] == 1000]
    for k in [50, 200]:
        b = [r["eps_lb_steinke"] for r in rec if r["kind"] == "interf" and r["eps"] == eps and r["k"] == k]
        T[f"eps={eps}|dirac_minus_interf_k{k}"] = {"diff": float(np.mean(np.array(a) - b)), "paired_t_p": float(stats.ttest_rel(a, b).pvalue)}
    c = [r["eps_lb_steinke"] for r in rec if r["kind"] == "input" and r["eps"] == eps]
    T[f"eps={eps}|dirac_minus_input"] = {"diff": float(np.mean(np.array(a) - c)), "paired_t_p": float(stats.ttest_rel(a, c).pvalue)}
for eps in EPSILONS:
    a = [r["eps_lb_steinke"] for r in rec if r["kind"] == "dirac" and r["eps"] == eps and r["m"] == 1000]
    b = [r["eps_lb_steinke"] for r in rec if r["kind"] == "dirac" and r["eps"] == eps and r["m"] == 100]
    T[f"eps={eps}|dirac_m1000_minus_m100"] = {"diff": float(np.mean(np.array(a) - b)), "paired_t_p": float(stats.ttest_rel(a, b).pvalue) if np.std(np.array(a) - b) > 0 else 1.0}
wb = [(r["mu_theory"], r["mu_hat"]) for r in rec if r["kind"] != "input"]
T["mu_hat_vs_mu_theory"] = {"pearson_r": float(stats.pearsonr(*zip(*wb)).statistic),
                            "median_rel_error": float(np.median([abs(h - t) / t for t, h in wb]))}
out["tests"] = T
out["per_run"] = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in rec]
out["runtime_sec"] = time.time() - T0
out["_meta"] = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
R = results_dir(__file__)
(R / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: None if o != o else o))

# ----------------------------------- figures -----------------------------------
P = plt()
fig, axes = P.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
ax.plot(EPSILONS, EPSILONS, "k--", label="记账上界 ε")
for m, c in [(100, "C0"), (1000, "C1")]:
    ce = [out["conditions"][f"dirac|eps={ek(e)}|m={m}|k=None"] for e in EPSILONS]
    ax.errorbar(EPSILONS, [x["eps_lb_steinke"]["mean"] for x in ce], yerr=[x["eps_lb_steinke"]["ci95"] for x in ce],
                marker="o", capsize=3, color=c, label=f"单次审计 Steinke 下界（Dirac, m={m}）")
    ax.plot(EPSILONS, [x["eps_gdp_ceiling_from_mu_theory"] for x in ce], ":", color=c, label=f"理论天花板 ε_GDP(μ_eff), m={m}")
for e, mk in [(2.0, "s"), (8.0, "^")]:
    ax.scatter([e], [out["conditions"][f"interf|eps={ek(e)}|m=1000|k=50"]["eps_lb_steinke"]["mean"]], marker=mk, color="C3",
               label="干扰金丝雀 m/k=20" if e == 2.0 else None)
    ax.scatter([e], [out["conditions"][f"input|eps={ek(e)}|m=200|k=None"]["eps_lb_steinke"]["mean"]], marker=mk, color="C4",
               label="黑盒输入金丝雀" if e == 2.0 else None)
    ax.scatter([e], [multi[f"eps={e}"]["eps_lb_cp"]], marker="*", s=120, color="C5", label=f"多次训练 ({2*MR_RUNS} 次)" if e == 2.0 else None)
ax.set_xlabel("记账 ε (δ=1e-6)")
ax.set_ylabel("经验 ε 下界 (95%)")
ax.set_title("单次审计能逼近记账 ε 吗？")
ax.legend(fontsize=7)
ax = axes[1]
th = [r["mu_theory"] for r in rec if r["kind"] != "input"]
hh = [r["mu_hat"] for r in rec if r["kind"] != "input"]
cc = ["C0" if r["kind"] == "dirac" else "C3" for r in rec if r["kind"] != "input"]
ax.scatter(th, hh, c=cc, s=12)
lim = [0, max(th + hh) * 1.05]
ax.plot(lim, lim, "k--")
ax.set_xlabel("理论 μ_eff")
ax.set_ylabel("实测 μ̂（成员/非成员分数的标准化均值差）")
ax.set_title("白盒金丝雀信噪比：理论 vs 实测（蓝=Dirac，红=干扰）")
fig.tight_layout()
fig.savefig(R / "frontier_audit.png")
print(f"done {time.time()-T0:.0f}s")
