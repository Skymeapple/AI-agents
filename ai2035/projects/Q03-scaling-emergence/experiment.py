"""Q03 规模律与涌现 —— 教师-学生任务上的模型规模律、外推与“涌现=度量假象”检验。

任务：8 维高斯输入 x；固定的随机教师网络（tanh, 64 隐元）输出 K=8 个连续量，
      每个量按教师分布的分位数离散成 C=10 个“token”（等概率桶，随机猜测准确率 0.1）。
学生：随机 ReLU 特征模型（宽度 N = 16 … 2048，跨 2.1 个数量级），岭回归训练输出层；
      对同一种子，各宽度的特征是嵌套子集（大模型包含小模型的特征），只改变规模。
度量：(1) 连续损失 L = 测试 MSE（已标准化，预测常数时 L≈1）；
      (2) 逐 token 准确率 p（预测连续值落入正确桶）；
      (3) k-token 精确匹配 EM_k（前 k 个 token 全对），k∈{1,2,4,8}。
分析：L(N)=a N^-b + c 拟合 + 种子自助法 CI；小规模拟合外推大规模；
      陡峭度指数（单次规模翻倍所贡献的最大归一化增幅）比较 p 与 EM；
      检验 EM_k ≈ Π p_j（独立近似）。

运行：python ai2035/projects/Q03-scaling-emergence/experiment.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt as get_plt, results_dir, rng, save_results  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)
D, H, K, C = 8, 64, 8, 10
NS = [16, 32, 64, 128, 256, 512, 1024, 2048]
FIT_SMALL = 6  # 外推：只用前 6 个规模（16..512）拟合
NTR, NTE = 12000, 4000
SEEDS = 5
KS = [1, 2, 4, 8]

# ---------------------------------------------------------------- 教师
gT = rng(0)
W1 = gT.standard_normal((D, H)) / np.sqrt(D)
b1 = 0.5 * gT.standard_normal(H)
W2 = gT.standard_normal((H, K)) / np.sqrt(H)


def teacher_raw(X):
    return np.tanh(X @ W1 + b1) @ W2


_ref = teacher_raw(gT.standard_normal((200000, D)))
MU, SD = _ref.mean(0), _ref.std(0)
EDGES = [np.quantile((_ref[:, j] - MU[j]) / SD[j], np.linspace(0, 1, C + 1)[1:-1]) for j in range(K)]


def teacher(X):
    return (teacher_raw(X) - MU) / SD


def tokens(Y):
    return np.stack([np.searchsorted(EDGES[j], Y[:, j]) for j in range(K)], 1)


# ---------------------------------------------------------------- 单个种子
def run_seed(s):
    g = rng(100 + s)
    Xtr = g.standard_normal((NTR, D)); Xte = g.standard_normal((NTE, D))
    Ytr, Yte = teacher(Xtr), teacher(Xte)
    Tte = tokens(Yte)
    Wf = g.standard_normal((D, NS[-1])) / np.sqrt(D)
    bf = g.standard_normal(NS[-1]) * 0.5
    Ftr = np.maximum(Xtr @ Wf + bf, 0)
    Fte = np.maximum(Xte @ Wf + bf, 0)
    G = Ftr.T @ Ftr
    R = Ftr.T @ Ytr
    out = {"L": [], "p_tok": [], "p_tok_each": [], "EM": {k: [] for k in KS}, "EM_indep": {k: [] for k in KS}}
    for N in NS:
        Gn = G[:N, :N]
        lam = 1e-4 * np.trace(Gn) / N
        A = np.linalg.solve(Gn + lam * np.eye(N), R[:N])
        P = Fte[:, :N] @ A
        out["L"].append(np.mean((P - Yte) ** 2))
        corr = tokens(P) == Tte  # (n,K)
        pe = corr.mean(0)
        out["p_tok_each"].append(pe)
        out["p_tok"].append(pe.mean())
        for k in KS:
            out["EM"][k].append(np.mean(np.all(corr[:, :k], 1)))
            out["EM_indep"][k].append(np.prod(pe[:k]))
    return out


# ---------------------------------------------------------------- 拟合
def pl(logN, a, b, c):
    return a * np.exp(-b * logN) + c


def fit_pl(Ns, L):
    x = np.log(np.asarray(Ns, float)); y = np.asarray(L)

    def loss(th):
        la, b, lc = th
        return np.log(np.exp(la) * np.exp(-b * x) + np.exp(lc)) - np.log(y)
    best = None
    for b0 in [0.2, 0.5, 1.0]:
        for c0 in [1e-4, 1e-2, 0.5 * y.min()]:
            th0 = [np.log(max(y[0] - c0, 1e-6)) + b0 * x[0], b0, np.log(c0)]
            r = optimize.least_squares(loss, th0, bounds=([-20, 0, -20], [20, 5, 5]))
            if best is None or r.cost < best.cost:
                best = r
    la, b, lc = best.x
    return float(np.exp(la)), float(b), float(np.exp(lc))


def fit_pure(Ns, L):
    b, la = np.polyfit(np.log(Ns), np.log(L), 1)
    return float(np.exp(la)), float(-b)


def sharpness(M):
    M = np.asarray(M)
    span = M.max() - M.min()
    return float(np.max(np.diff(M)) / span) if span > 0 else np.nan


if __name__ == "__main__":
    plt = get_plt()
    runs = [run_seed(s) for s in range(SEEDS)]
    t_train = time.time() - T0
    L = np.array([r["L"] for r in runs])
    ptok = np.array([r["p_tok"] for r in runs])
    EM = {k: np.array([r["EM"][k] for r in runs]) for k in KS}
    EMi = {k: np.array([r["EM_indep"][k] for r in runs]) for k in KS}
    Ns = np.array(NS, float)

    # ---- H1：规模律拟合（全部规模） + 种子自助法
    a, b, c = fit_pl(Ns, L.mean(0))
    bt = rng(999)
    boot = []
    for _ in range(300):
        idx = bt.integers(0, SEEDS, SEEDS)
        boot.append(fit_pl(Ns, L[idx].mean(0)))
    boot = np.array(boot)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    per_seed_b = [fit_pl(Ns, L[s])[1] for s in range(SEEDS)]
    pred_full = pl(np.log(Ns), a, b, c)
    r2_log = 1 - np.sum((np.log(L.mean(0)) - np.log(pred_full)) ** 2) / np.sum((np.log(L.mean(0)) - np.log(L.mean(0)).mean()) ** 2)
    # 局部对数斜率（相邻规模）
    local_slope = -np.diff(np.log(L.mean(0))) / np.diff(np.log(Ns))

    # ---- H2：外推（拟合 16..512，预测 1024, 2048），逐种子
    ext = {"sat": [], "pure": []}
    for s in range(SEEDS):
        a_s, b_s, c_s = fit_pl(Ns[:FIT_SMALL], L[s, :FIT_SMALL])
        ap, bp = fit_pure(Ns[:FIT_SMALL], L[s, :FIT_SMALL])
        ext["sat"].append([(pl(np.log(N), a_s, b_s, c_s) - L[s, j]) / L[s, j] for j, N in zip(range(FIT_SMALL, len(NS)), NS[FIT_SMALL:])])
        ext["pure"].append([(ap * N ** -bp - L[s, j]) / L[s, j] for j, N in zip(range(FIT_SMALL, len(NS)), NS[FIT_SMALL:])])
    ext = {k: np.array(v) for k, v in ext.items()}

    # ---- H3：涌现 = 度量假象
    sharp = {"p_tok": [sharpness(ptok[s]) for s in range(SEEDS)],
             "neg_logL": [sharpness(-np.log(L[s])) for s in range(SEEDS)]}
    for k in KS:
        sharp[f"EM{k}"] = [sharpness(EM[k][s]) for s in range(SEEDS)]
    from scipy import stats
    tt = stats.ttest_rel(sharp["EM8"], sharp["p_tok"])
    # 独立近似误差
    indep_err = {k: float(np.max(np.abs(EMi[k].mean(0) - EM[k].mean(0)))) for k in KS}
    # 从小规模外推 EM8：(i) 经由逐 token 错误率的饱和幂律，EM≈Π p_j ≈ p^8；(ii) 对 EM8 本身在 log N 上线性外推
    em_ext = {"via_token": [], "via_token_pure": [], "direct_linear": [], "true": []}
    for s in range(SEEDS):
        e = 1 - ptok[s]
        ae, be, ce = fit_pl(Ns[:FIT_SMALL], e[:FIT_SMALL])
        pe_hat = 1 - pl(np.log(Ns[FIT_SMALL:]), ae, be, ce)
        em_ext["via_token"].append(pe_hat ** K)
        ap_, bp_ = fit_pure(Ns[:FIT_SMALL], e[:FIT_SMALL])  # 探索性：错误率用纯幂律外推
        em_ext["via_token_pure"].append((1 - ap_ * Ns[FIT_SMALL:] ** -bp_) ** K)
        sl, ic = np.polyfit(np.log(Ns[FIT_SMALL - 3:FIT_SMALL]), EM[8][s, FIT_SMALL - 3:FIT_SMALL], 1)
        em_ext["direct_linear"].append(sl * np.log(Ns[FIT_SMALL:]) + ic)
        em_ext["true"].append(EM[8][s, FIT_SMALL:])
    em_ext = {k: np.array(v) for k, v in em_ext.items()}

    # ---------------------------------------------------------------- 图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].errorbar(Ns, L.mean(0), yerr=[mean_ci(L[:, j])[1] for j in range(len(NS))], fmt="o", label="实测（均值±95%CI）")
    xx = np.logspace(np.log10(16), np.log10(4096), 100)
    ax[0].plot(xx, pl(np.log(xx), a, b, c), "-", label=f"全量拟合 L={a:.2f}N^-{b:.2f}+{c:.3f}")
    a6, b6, c6 = fit_pl(Ns[:FIT_SMALL], L.mean(0)[:FIT_SMALL])
    ax[0].plot(xx, pl(np.log(xx), a6, b6, c6), "--", label="仅用 N≤512 拟合（外推）")
    ap6, bp6 = fit_pure(Ns[:FIT_SMALL], L.mean(0)[:FIT_SMALL])
    ax[0].plot(xx, ap6 * xx ** -bp6, ":", label="纯幂律 N≤512（外推）")
    ax[0].axvline(512 * 1.4, color="gray", lw=0.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("模型规模 N（特征数）"); ax[0].set_ylabel("测试损失 L（MSE）")
    ax[0].legend(fontsize=7); ax[0].set_title("规模律与外推")
    ax[1].plot(Ns, ptok.mean(0), "o-", label="逐 token 准确率 p")
    for k in KS[1:]:
        ax[1].plot(Ns, EM[k].mean(0), "s-", label=f"精确匹配 EM_{k}")
    ax[1].plot(Ns, ptok.mean(0) ** 8, "k:", label="p^8（独立近似）")
    ax[1].set_xscale("log"); ax[1].set_xlabel("模型规模 N"); ax[1].set_ylabel("准确率（线性轴）")
    ax[1].legend(fontsize=8); ax[1].set_title("同一组模型：平滑度量 vs 精确匹配")
    fig.tight_layout(); fig.savefig(OUT / "scaling_and_emergence.png"); plt.close(fig)

    p = save_results(__file__, {
        "setup": {"D": D, "teacher_hidden": H, "K_tokens": K, "C_bins": C, "Ns": NS, "n_train": NTR, "n_test": NTE,
                  "seeds": SEEDS, "ridge_rel": 1e-4, "fit_small_upto": NS[FIT_SMALL - 1]},
        "loss_mean": L.mean(0), "loss_ci": [mean_ci(L[:, j])[1] for j in range(len(NS))],
        "p_tok_mean": ptok.mean(0), "p_tok_ci": [mean_ci(ptok[:, j])[1] for j in range(len(NS))],
        "EM_mean": {str(k): EM[k].mean(0) for k in KS}, "EM_ci": {str(k): [mean_ci(EM[k][:, j])[1] for j in range(len(NS))] for k in KS},
        "EM_indep_mean": {str(k): EMi[k].mean(0) for k in KS}, "EM_indep_maxabs_err": indep_err,
        "H1_powerlaw_full": {"a": a, "b": b, "c": c, "b_boot_ci": ci(boot[:, 1]), "c_boot_ci": ci(boot[:, 2]),
                             "b_per_seed": per_seed_b, "b_per_seed_meanci": mean_ci(per_seed_b), "R2_log": r2_log,
                             "local_slopes": local_slope},
        "H2_extrapolation": {"targets": NS[FIT_SMALL:],
                             "sat_relerr_mean": ext["sat"].mean(0), "sat_relerr_ci": [mean_ci(ext["sat"][:, j])[1] for j in range(ext["sat"].shape[1])],
                             "sat_abs_relerr_mean": np.abs(ext["sat"]).mean(0),
                             "pure_relerr_mean": ext["pure"].mean(0), "pure_relerr_ci": [mean_ci(ext["pure"][:, j])[1] for j in range(ext["pure"].shape[1])],
                             "pure_abs_relerr_mean": np.abs(ext["pure"]).mean(0)},
        "H3_sharpness": {k: mean_ci(v) for k, v in sharp.items()}, "H3_sharp_per_seed": sharp,
        "H3_paired_t_EM8_vs_ptok": {"t": float(tt.statistic), "p": float(tt.pvalue)},
        "H3_EM8_extrapolation": {k: v.mean(0) for k, v in em_ext.items()},
        "H3_EM8_extrap_abs_err": {"via_token": np.abs(em_ext["via_token"] - em_ext["true"]).mean(0),
                                  "via_token_pure": np.abs(em_ext["via_token_pure"] - em_ext["true"]).mean(0),
                                  "direct_linear": np.abs(em_ext["direct_linear"] - em_ext["true"]).mean(0)},
        "runtime_sec": {"train": t_train, "total": time.time() - T0},
    })
    print("saved", p, time.time() - T0)
    print("L", np.round(L.mean(0), 4)); print("p", np.round(ptok.mean(0), 3)); print("EM8", np.round(EM[8].mean(0), 4))
    print("fit", a, b, c, "sharp", {k: np.mean(v) for k, v in sharp.items()})
