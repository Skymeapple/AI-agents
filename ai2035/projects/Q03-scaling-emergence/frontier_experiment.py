"""Q03 前沿实验：测试时计算（重复采样）的规模律——覆盖率幂律、多数投票天花板与不完美验证器的“奖励黑客”拐点。

理论（paper.md §4）：
  T1 覆盖率：pass@k = 1 − E[(1−p)^k]。若题目难度 p ~ Beta(a,b)，则 1 − pass@k ≈ Γ(a+b)Γ(a)/(Γ(b)·B(a,b)) … ∝ k^{−a}
     （精确：E[(1−p)^k] = B(a, b+k)/B(a,b) ~ Γ(a+b)/Γ(b) · k^{−a}）。指数只由 p→0 处的密度决定。
  T2 多数投票：k→∞ 时每题正确 ⇔ p > q_max（最强错误答案的概率），maj@∞ = P(p > q_max) < pass@∞；
     p < q_max 的题目随 k 增大准确率下降（逆向缩放）。
  T3 带分数验证器的 Best-of-n：正确样本分数 ~ N(μ1, σ1²)，错误样本 ~ N(μ0, σ0²)。
     A(n) = ∫ n·p·f1(s)·[p F1(s) + (1−p) F0(s)]^{n−1} ds（精确）；σ0 > σ1 时错误样本的上尾终将胜出，A(n) 先升后降。
实验：模拟 2000 道题 × 4096 次采样 × 5 种子，逐条与理论对照；并比较“从小 k 外推大 k”的两种方法。
运行：python ai2035/projects/Q03-scaling-emergence/frontier_experiment.py
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
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import _jsonable, mean_ci, plt as get_plt, results_dir, rng  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)
NPROB, KMAX, SEEDS = 2000, 4096, 5
KS = 2 ** np.arange(0, 13)  # 1 … 4096
A_BETA, B_BETA = 0.3, 2.0  # 难度分布（p 接近 0 处密度 ∝ p^{a-1}）
M_WRONG = 4  # 错误答案的类别数
DIR_W = 0.5  # 错误答案在各类别间的 Dirichlet 集中度（小 → 错误答案集中，易出现“强错误吸引子”）


def theory_passk(a, b, k):
    return 1 - np.exp(special.betaln(a, b + k) - special.betaln(a, b))


def bon_theory(p, n, mu1, s1, mu0, s0, grid=None):
    """精确 BoN 准确率（逐题积分）；p 数组，n 标量。"""
    if grid is None:
        grid = np.linspace(min(mu0 - 8 * s0, mu1 - 8 * s1), max(mu0 + 8 * s0, mu1 + 8 * s1), 4001)
    f1 = stats.norm.pdf(grid, mu1, s1); F1 = stats.norm.cdf(grid, mu1, s1); F0 = stats.norm.cdf(grid, mu0, s0)
    mix = p[:, None] * F1[None, :] + (1 - p[:, None]) * F0[None, :]
    with np.errstate(divide="ignore"):
        logmix = np.where(mix > 0, np.log(np.maximum(mix, 1e-300)), -np.inf)
    integrand = n * p[:, None] * f1[None, :] * np.exp((n - 1) * logmix)
    return np.trapezoid(integrand, grid, axis=1)


def unbiased_passk(c, n, k):
    """Chen et al. (2021) 无偏估计：1 − C(n−c,k)/C(n,k)，逐题后取平均。"""
    ok = (n - c) >= k
    cc = np.where(ok, c, 0)
    logr = special.gammaln(n - cc + 1) - special.gammaln(n - cc - k + 1) - special.gammaln(n + 1) + special.gammaln(n - k + 1)
    return float(np.mean(np.where(ok, 1 - np.exp(logr), 1.0)))


def simulate(seed, unsolvable=0.0, a=A_BETA, b=B_BETA):
    g = rng(300 + seed)
    p = g.beta(a, b, NPROB)
    if unsolvable > 0:
        p[g.random(NPROB) < unsolvable] = 0.0
    wrong = g.dirichlet(np.full(M_WRONG, DIR_W), NPROB) * (1 - p)[:, None]
    qmax = wrong.max(1)
    cum = np.cumsum(np.hstack([p[:, None], wrong]), 1)  # 类别 0 = 正确
    U = g.random((NPROB, KMAX))
    ans = (U[:, :, None] > cum[:, None, :-1]).sum(2).astype(np.int8)  # 0..M_WRONG
    correct = ans == 0
    # pass@k（以首 k 个样本是否至少一次正确；对每题用无偏估计 1 - C(n-c,k)/C(n,k) 更平滑，这里用全部 KMAX 样本的无偏估计）
    c = correct.sum(1)
    passk = np.array([unbiased_passk(c, KMAX, k) for k in KS])
    # 多数投票：前 k 个样本中计数最多的答案（并列时按并列数分摊）
    counts = np.zeros((NPROB, M_WRONG + 1), np.int32)
    maj = []
    last = 0
    for k in KS:
        for j in range(M_WRONG + 1):
            counts[:, j] += (ans[:, last:k] == j).sum(1)
        last = k
        mx = counts.max(1)
        tied = (counts == mx[:, None]).sum(1)
        maj_prob = (counts[:, 0] == mx) / tied
        maj.append(np.mean(maj_prob))
    maj = np.array(maj)
    # 验证器分数（两种验证器共享样本流）
    Z = g.standard_normal((NPROB, KMAX))
    bon = {}
    for name, (mu1, s1, mu0, s0) in VERIFIERS.items():
        sc = np.where(correct, mu1 + s1 * Z, mu0 + s0 * Z)
        cmax_c = np.maximum.accumulate(np.where(correct, sc, -np.inf), 1)
        cmax_w = np.maximum.accumulate(np.where(correct, -np.inf, sc), 1)
        bon[name] = np.array([np.mean(cmax_c[:, k - 1] > cmax_w[:, k - 1]) for k in KS])
    # 小样本（前 32 个）用于外推
    c32 = correct[:, :32].sum(1)
    return {"p": p, "qmax": qmax, "passk": passk, "maj": maj, "bon": bon, "c32": c32, "maj_final_prob": maj_prob}


VERIFIERS = {"equal_var": (1.0, 1.0, 0.0, 1.0), "heavy_wrong_tail": (1.0, 1.0, 0.0, 1.6)}


def fit_beta_binomial(c, n):
    """对每题 n 次试验中的成功数 c 做 Beta-Binomial 极大似然，返回 (a,b)。"""
    def nll(th):
        a, b = np.exp(th)
        return -np.sum(special.betaln(c + a, n - c + b) - special.betaln(a, b))
    r = optimize.minimize(nll, [np.log(0.5), np.log(2.0)], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000})
    return np.exp(r.x)


if __name__ == "__main__":
    plt = get_plt()
    res = {"setup": {"n_problems": NPROB, "k_max": KMAX, "seeds": SEEDS, "beta_a": A_BETA, "beta_b": B_BETA,
                     "m_wrong": M_WRONG, "dirichlet_wrong": DIR_W, "verifiers(mu1,s1,mu0,s0)": VERIFIERS}}
    sims = [simulate(s) for s in range(SEEDS)]
    sims_u = [simulate(100 + s, unsolvable=0.1) for s in range(SEEDS)]
    sims_e = [simulate(200 + s, a=2.0, b=1.0) for s in range(SEEDS)]  # 较易题目分布（均值 0.67）
    t_sim = time.time() - T0

    # ---------------- H1 覆盖率幂律
    th = theory_passk(A_BETA, B_BETA, KS)
    pk = np.array([s["passk"] for s in sims])
    big = KS >= 32
    expo = [-np.polyfit(np.log(KS[big]), np.log(1 - pk[s, big]), 1)[0] for s in range(SEEDS)]
    expo_th = -np.polyfit(np.log(KS[big]), np.log(1 - th[big]), 1)[0]
    pku = np.array([s["passk"] for s in sims_u])
    expo_u = [-np.polyfit(np.log(KS[big]), np.log(1 - pku[s, big]), 1)[0] for s in range(SEEDS)]
    # 外推：只用每题前 32 个样本
    ext = {"betabin": [], "powerlaw": [], "true": []}
    for s in range(SEEDS):
        a_hat, b_hat = fit_beta_binomial(sims[s]["c32"], 32)
        ext["betabin"].append(theory_passk(a_hat, b_hat, 4096))
        # 朴素：在 k=1..32 上用 32 样本的无偏 pass@k 拟合 log(1−pass@k) ~ log k
        c32 = sims[s]["c32"]
        small = np.array([1, 2, 4, 8, 16, 32])
        pk32 = np.array([unbiased_passk(c32, 32, k) for k in small])
        sl, ic = np.polyfit(np.log(small[1:]), np.log(1 - pk32[1:]), 1)
        ext["powerlaw"].append(1 - np.exp(ic) * 4096 ** sl)
        ext["true"].append(pk[s, -1])
        ext.setdefault("a_hat", []).append(a_hat)
    res["H1_coverage"] = {
        "ks": KS, "passk_sim_mean": pk.mean(0), "passk_sim_ci": [mean_ci(pk[:, j])[1] for j in range(len(KS))],
        "passk_theory": th, "max_abs_err": float(np.max(np.abs(pk.mean(0) - th))),
        "exponent_sim_k32_4096": mean_ci(expo), "exponent_theory_same_range": expo_th, "exponent_asymptotic": A_BETA,
        "with_10pct_unsolvable": {"passk_sim_mean": pku.mean(0), "exponent_sim_k32_4096": mean_ci(expo_u),
                                  "ceiling_theory": 0.9 * 1.0},
        "extrapolation_k4096_from_32": {"true": mean_ci(ext["true"]), "betabinomial": mean_ci(ext["betabin"]),
                                        "naive_powerlaw": mean_ci(ext["powerlaw"]), "a_hat": mean_ci(ext["a_hat"]),
                                        "abs_err_betabinomial": mean_ci(np.abs(np.array(ext["betabin"]) - ext["true"])),
                                        "abs_err_naive": mean_ci(np.abs(np.array(ext["powerlaw"]) - ext["true"]))},
    }

    # ---------------- H2 多数投票
    mj = np.array([s["maj"] for s in sims])
    ceil = np.array([np.mean(s["p"] > s["qmax"]) for s in sims])
    frac_inv = np.array([np.mean((s["p"] < s["qmax"]) & (s["p"] > 0.05)) for s in sims])
    # 逆向缩放检验：p<q_max 子集上 maj@1 与 maj@4096（重新计算子集准确率需要逐题结果，此处用理论：maj@1 = p，maj@∞ = 0）
    sub_p = np.array([s["p"][(s["p"] < s["qmax"])].mean() for s in sims])
    res["H2_majority"] = {
        "maj_sim_mean": mj.mean(0), "maj_sim_ci": [mean_ci(mj[:, j])[1] for j in range(len(KS))],
        "maj_ceiling_theory": mean_ci(ceil), "maj_at_4096": mean_ci(mj[:, -1]), "pass_at_4096": mean_ci(pk[:, -1]),
        "gap_ceiling_minus_maj4096": mean_ci(ceil - mj[:, -1]),
        "frac_problems_p_lt_qmax": mean_ci(np.array([np.mean(s["p"] < s["qmax"]) for s in sims])),
        "mean_p_on_p_lt_qmax_subset(=maj@1 there)": mean_ci(sub_p),
        "maj_peak_k": int(KS[np.argmax(mj.mean(0))]),
        "subset_p_lt_qmax_maj_at_4096": mean_ci(np.array([s["maj_final_prob"][s["p"] < s["qmax"]].mean() for s in sims])),
        "subset_p_gt_qmax_maj_at_4096": mean_ci(np.array([s["maj_final_prob"][s["p"] > s["qmax"]].mean() for s in sims])),
        "easy_beta(2,1)": {"maj_sim_mean": np.array([s["maj"] for s in sims_e]).mean(0),
                           "passk_sim_mean": np.array([s["passk"] for s in sims_e]).mean(0),
                           "maj_ceiling_theory": mean_ci(np.array([np.mean(s["p"] > s["qmax"]) for s in sims_e])),
                           "maj_at_4096": mean_ci(np.array([s["maj"][-1] for s in sims_e])),
                           "maj_at_1": mean_ci(np.array([s["maj"][0] for s in sims_e])),
                           "pass_at_4096": mean_ci(np.array([s["passk"][-1] for s in sims_e]))},
        "residual_error_loglog_slope_k256_4096": mean_ci([np.polyfit(np.log(KS[KS >= 256]), np.log(1 - mj[s, KS >= 256]), 1)[0] for s in range(SEEDS)]),
    }

    # ---------------- H3 验证器 BoN
    h3 = {}
    for name, (mu1, s1, mu0, s0) in VERIFIERS.items():
        bs = np.array([s["bon"][name] for s in sims])
        thb = np.array([[bon_theory(s["p"], k, mu1, s1, mu0, s0).mean() for k in KS] for s in sims])
        peak_sim = [int(KS[np.argmax(bs[s])]) for s in range(SEEDS)]
        peak_th = [int(KS[np.argmax(thb[s])]) for s in range(SEEDS)]
        h3[name] = {"bon_sim_mean": bs.mean(0), "bon_sim_ci": [mean_ci(bs[:, j])[1] for j in range(len(KS))],
                    "bon_theory_mean": thb.mean(0), "max_abs_err": float(np.max(np.abs(bs.mean(0) - thb.mean(0)))),
                    "peak_k_sim_per_seed": peak_sim, "peak_k_theory_per_seed": peak_th,
                    "peak_acc_sim": mean_ci(bs.max(1)), "acc_at_4096_sim": mean_ci(bs[:, -1]),
                    "drop_from_peak_to_4096": mean_ci(bs.max(1) - bs[:, -1]),
                    "auc": float(stats.norm.cdf((mu1 - mu0) / np.sqrt(s1 ** 2 + s0 ** 2)))}
        # 更大 n 的理论外推（模拟之外）
        h3[name]["theory_large_n"] = {str(n): float(bon_theory(sims[0]["p"], n, mu1, s1, mu0, s0).mean()) for n in [2 ** 14, 2 ** 17, 2 ** 20]}
    res["H3_verifier_bon"] = h3
    res["runtime_sec"] = {"simulate": t_sim, "total": time.time() - T0}

    # ---------------- 图
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].loglog(KS, 1 - pk.mean(0), "o", label="模拟 1−pass@k")
    ax[0].loglog(KS, 1 - th, "k-", label=f"理论 B(a,b+k)/B(a,b)，a={A_BETA}")
    ax[0].loglog(KS, 1 - pku.mean(0), "s", label="含 10% 无解题")
    ax[0].set_xlabel("采样数 k"); ax[0].set_ylabel("1 − pass@k"); ax[0].legend(fontsize=8); ax[0].set_title("覆盖率幂律")
    ax[1].semilogx(KS, pk.mean(0), "o-", label="pass@k（神谕验证）")
    ax[1].semilogx(KS, mj.mean(0), "s-", label="多数投票 maj@k")
    ax[1].axhline(ceil.mean(), color="C1", ls=":", label="理论 maj@∞ = P(p > q_max)")
    me = np.array([s["maj"] for s in sims_e]).mean(0)
    ax[1].semilogx(KS, me, "d--", color="C1", alpha=0.6, label="maj@k（易题分布 Beta(2,1)）")
    for name, lab in [("equal_var", "BoN 等方差验证器"), ("heavy_wrong_tail", "BoN 错误重尾验证器")]:
        ax[1].semilogx(KS, h3[name]["bon_sim_mean"], "^-", label=lab)
        ax[1].semilogx(KS, h3[name]["bon_theory_mean"], "k:", lw=1)
    ax[1].set_xlabel("采样数 k / n"); ax[1].set_ylabel("准确率"); ax[1].legend(fontsize=7); ax[1].set_title("三种聚合方式（虚线=理论）")
    nn = 2 ** np.arange(0, 21)
    for name in VERIFIERS:
        mu1, s1, mu0, s0 = VERIFIERS[name]
        ax[2].semilogx(nn, [bon_theory(sims[0]["p"], n, mu1, s1, mu0, s0).mean() for n in nn], label=name)
    ax[2].axhline(pk.mean(0)[-1], color="gray", ls=":", label="pass@4096")
    ax[2].set_xlabel("n（理论外推到 10⁶）"); ax[2].set_ylabel("BoN 准确率"); ax[2].legend(fontsize=8); ax[2].set_title("验证器尾部竞争")
    fig.tight_layout(); fig.savefig(OUT / "frontier_test_time_scaling.png"); plt.close(fig)

    pth = OUT / "frontier.json"
    pth.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=_jsonable))
    print("saved", pth, res["runtime_sec"])
    print(json.dumps({k: v for k, v in res.items() if k != "setup"}, default=_jsonable, ensure_ascii=False)[:5000])
