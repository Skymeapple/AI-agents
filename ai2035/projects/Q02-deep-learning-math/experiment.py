"""Q02 深度学习的数学机理 —— 频率原则 / 谱偏好 与 逼近速率。

实验 A：自写 numpy MLP（ReLU, 2 隐层×128, Adam）拟合等幅多频目标
        f(x)=Σ_k sin(2πkx), k∈{1,5,10,20}；每隔若干步对残差做 FFT，
        追踪各频率相对误差 Δ_k(t) 与收敛时刻 t_k。
实验 B：随机特征模型（固定第一层，只训练输出层）——
        (B1) 梯度下降：用核 K=ΦΦᵀ/n 的 Rayleigh 商 λ_k 预测各频率衰减速率；
        (B2) Adam：与 MLP 同优化器的对照。
实验 C：单隐层 ReLU 网络逼近误差随宽度 m 的标度：随机特征+最小二乘 vs 训练的 MLP。

运行：python ai2035/projects/Q02-deep-learning-math/experiment.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import mean_ci, plt as get_plt, results_dir, rng, save_results  # noqa: E402

T0 = time.time()
OUT = results_dir(__file__)
FREQS = [1, 5, 10, 20]
NGRID = 256
THRESH = 0.1  # 收敛判据：Δ_k < 0.1（预先固定）
SEEDS = 5


def target(x):
    return sum(np.sin(2 * np.pi * k * x) for k in FREQS)


def rel_freq_err(resid, y):
    R = np.abs(np.fft.rfft(resid))
    Y = np.abs(np.fft.rfft(y))
    return np.array([R[k] / Y[k] for k in FREQS])


# ------------------------------------------------------------ 简单 MLP + Adam
class MLP:
    def __init__(self, sizes, g):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            lim = 1 / np.sqrt(a)  # PyTorch 默认初始化（kaiming_uniform a=√5 等价于 U(-1/√fan_in, 1/√fan_in)）
            self.W.append(g.uniform(-lim, lim, (a, b)))
            self.b.append(g.uniform(-lim, lim, b))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, X):
        hs = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            hs.append(h)
        return hs

    def step(self, X, y, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        hs = self.forward(X)
        out = hs[-1][:, 0]
        n = X.shape[0]
        d = (2 / n) * (out - y)[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (hs[i] > 0)
        grads = gW + gb
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)
        return out


def first_cross(curve, steps, th):
    idx = np.where(curve < th)[0]
    return float(steps[idx[0]]) if len(idx) else np.nan


# ------------------------------------------------------------ 实验 A：MLP
def exp_A(steps=8000, log_every=50):
    x = np.arange(NGRID) / NGRID
    y = target(x)
    X = (2 * x - 1)[:, None]
    logs = np.arange(0, steps + 1, log_every)
    curves = np.zeros((SEEDS, len(logs), len(FREQS)))
    final_mse = []
    for s in range(SEEDS):
        g = rng(100 + s)
        net = MLP([1, 128, 128, 128, 1], g)
        j = 0
        for t in range(steps + 1):
            if t % log_every == 0:
                out = net.forward(X)[-1][:, 0]
                curves[s, j] = rel_freq_err(out - y, y)
                j += 1
            if t < steps:
                net.step(X, y, lr=2e-3)
        final_mse.append(np.mean((net.forward(X)[-1][:, 0] - y) ** 2))
    return logs, curves, final_mse


# ------------------------------------------------------------ 实验 B：随机特征
def rf_features(g, x, m):
    X = (2 * x - 1)[:, None]
    W = g.uniform(-1, 1, (1, m))
    b = g.uniform(-1, 1, m)
    return np.maximum(X @ W + b, 0)


def exp_B(steps=8000, log_every=50, m=1024):
    x = np.arange(NGRID) / NGRID
    y = target(x)
    n = NGRID
    logs = np.arange(0, steps + 1, log_every)
    cur_gd = np.zeros((SEEDS, len(logs), len(FREQS)))
    cur_adam = np.zeros_like(cur_gd)
    lam_k = np.zeros((SEEDS, len(FREQS)))
    pred_rate = np.zeros((SEEDS, len(FREQS)))
    emp_rate = np.zeros((SEEDS, len(FREQS)))
    kk = np.arange(1, 41)
    lam_all = np.zeros((SEEDS, len(kk)))
    eig_slope, eig_rho, eig_domfreq = [], [], []
    lam_eig = np.zeros((SEEDS, len(FREQS)))
    eig_spec_slope = []
    for s in range(SEEDS):
        g = rng(300 + s)
        Phi = rf_features(g, x, m) / np.sqrt(m)
        Phi = np.hstack([Phi, np.ones((n, 1)) / np.sqrt(m)])
        K = Phi @ Phi.T  # 输出层 GD 的核（损失 1/n Σ r² 下动力学 r ← (I - 2η K/n) r）
        evmax = np.linalg.eigvalsh(K).max()
        eta = 0.9 * n / (2 * evmax)  # 稳定步长（预先固定的规则）
        # 各频率 Rayleigh 商（sin 与 cos 两分量平均）
        for q, k in enumerate(FREQS):
            es = np.sin(2 * np.pi * k * x); es /= np.linalg.norm(es)
            ec = np.cos(2 * np.pi * k * x); ec /= np.linalg.norm(ec)
            lam_k[s, q] = 0.5 * (es @ K @ es + ec @ K @ ec)
        for q, k in enumerate(kk):
            es = np.sin(2 * np.pi * k * x); es /= np.linalg.norm(es)
            ec = np.cos(2 * np.pi * k * x); ec /= np.linalg.norm(ec)
            lam_all[s, q] = 0.5 * (es @ K @ es + ec @ K @ ec)
        eig_slope.append(np.polyfit(np.log(kk), np.log(lam_all[s]), 1)[0])
        ev, Vfull = np.linalg.eigh(K)
        ev, Vfull = ev[::-1], Vfull[:, ::-1]
        V = Vfull[:, :40]
        dom = np.argmax(np.abs(np.fft.rfft(V, axis=0)), axis=0)  # 前 40 个特征向量的主频
        eig_domfreq.append(dom)
        eig_rho.append(stats.spearmanr(np.arange(40), dom)[0])
        # 各频率对应的核特征值：与 span{sin_k, cos_k} 投影能量最大的特征向量之特征值
        for q, k in enumerate(FREQS):
            es = np.sin(2 * np.pi * k * x); es /= np.linalg.norm(es)
            ec = np.cos(2 * np.pi * k * x); ec /= np.linalg.norm(ec)
            proj = (Vfull.T @ es) ** 2 + (Vfull.T @ ec) ** 2
            lam_eig[s, q] = ev[int(np.argmax(proj))]
        # 特征谱 vs 主频：log λ_i 对 log 主频 的斜率（主频 1..40）
        domall = np.argmax(np.abs(np.fft.rfft(Vfull, axis=0)), axis=0)
        okd = (domall >= 1) & (domall <= 40) & (ev > ev[0] * 1e-13)
        if s == 0:
            eig0 = {"ev": ev[:150], "domfreq": domall[:150]}
        eig_spec_slope.append(np.polyfit(np.log(domall[okd]), np.log(ev[okd]), 1)[0])
        pred_rate[s] = 2 * eta * lam_eig[s] / n  # 线性化理论：该特征方向残差每步衰减率
        # B1：GD
        a = np.zeros(Phi.shape[1])
        j = 0
        for t in range(steps + 1):
            out = Phi @ a
            if t % log_every == 0:
                cur_gd[s, j] = rel_freq_err(out - y, y); j += 1
            a -= eta * (2 / n) * Phi.T @ (out - y)
        # 经验衰减率：-ln Δ_k(t*) / t*，t* 为 Δ_k 首次 <0.5 的步数（若未达到则取最后一步）
        for q in range(len(FREQS)):
            c = cur_gd[s, :, q]
            idx = np.where(c < 0.5)[0]
            jstar = idx[0] if len(idx) else len(c) - 1
            jstar = max(jstar, 1)
            emp_rate[s, q] = -np.log(c[jstar]) / logs[jstar]
        # B2：Adam（与 MLP 相同 lr）
        a = np.zeros(Phi.shape[1]); mm = np.zeros_like(a); vv = np.zeros_like(a)
        j = 0
        for t in range(steps + 1):
            out = Phi @ a
            if t % log_every == 0:
                cur_adam[s, j] = rel_freq_err(out - y, y); j += 1
            gr = (2 / n) * Phi.T @ (out - y)
            mm = 0.9 * mm + 0.1 * gr; vv = 0.999 * vv + 0.001 * gr * gr
            a -= 2e-3 * (mm / (1 - 0.9 ** (t + 1))) / (np.sqrt(vv / (1 - 0.999 ** (t + 1))) + 1e-8)
    extra = {"eig_lambda_k": {f"k{k}": mean_ci(lam_eig[:, q]) for q, k in enumerate(FREQS)},
             "eig_spectrum_loglog_slope_vs_domfreq": mean_ci(eig_spec_slope), "eig_seed0": eig0,
             "rayleigh_loglog_slope_k1_40": mean_ci(eig_slope), "rayleigh_k1_40_mean": lam_all.mean(0),
             "eigvec_rank_vs_domfreq_spearman": mean_ci(eig_rho), "eigvec_domfreq_seed0_top20": eig_domfreq[0][:20]}
    return logs, cur_gd, cur_adam, lam_k, pred_rate, emp_rate, extra


# ------------------------------------------------------------ 实验 C：逼近速率
def exp_C():
    ntr, nte = 1024, 4096
    xtr = (np.arange(ntr) + 0.5) / ntr
    xte = np.random.default_rng(7).random(nte)
    ytr, yte = target(xtr), target(xte)
    widths_rf = [16, 32, 64, 128, 256, 512, 1024, 2048]
    ntr_rf = 4096  # 随机特征最小二乘用更密的训练网格，保证 m < n（非插值区）
    xtr_rf = (np.arange(ntr_rf) + 0.5) / ntr_rf
    widths_mlp = [16, 32, 64, 128, 256, 512]
    rf_err = np.zeros((SEEDS, len(widths_rf)))
    mlp_err = np.zeros((SEEDS, len(widths_mlp)))
    for s in range(SEEDS):
        for j, m in enumerate(widths_rf):
            g = rng(700 + 31 * s + m)
            W = g.uniform(-1, 1, (1, m)); b = g.uniform(-1, 1, m)
            F = lambda x: np.hstack([np.maximum((2 * x - 1)[:, None] @ W + b, 0), np.ones((len(x), 1))])
            coef, *_ = np.linalg.lstsq(F(xtr_rf), target(xtr_rf), rcond=None)
            rf_err[s, j] = np.sqrt(np.mean((F(xte) @ coef - yte) ** 2))
        for j, m in enumerate(widths_mlp):
            g = rng(900 + 31 * s + m)
            net = MLP([1, m, 1], g)
            Xtr = (2 * xtr - 1)[:, None]
            for t in range(2000):
                net.step(Xtr, ytr, lr=3e-3)
            mlp_err[s, j] = np.sqrt(np.mean((net.forward((2 * xte - 1)[:, None])[-1][:, 0] - yte) ** 2))

    def fit(widths, err, mmin=64, mmax=512):
        w = np.array(widths); ok = (w >= mmin) & (w <= mmax)
        sl = [np.polyfit(np.log(w[ok]), np.log(err[s, ok]), 1)[0] for s in range(SEEDS)]
        return mean_ci(sl), sl
    rf_ext = fit(widths_rf, rf_err, 256, 2048)
    return widths_rf, rf_err, fit(widths_rf, rf_err), widths_mlp, mlp_err, fit(widths_mlp, mlp_err), rf_ext


if __name__ == "__main__":
    plt = get_plt()
    t1 = time.time()
    logs, curA, mseA = exp_A()
    t2 = time.time()
    logsB, cur_gd, cur_adam, lam_k, pred_rate, emp_rate, extraB = exp_B()
    t3 = time.time()
    wr, rf_err, (rf_fit, rf_sl), wm, mlp_err, (mlp_fit, mlp_sl), (rf_fit_ext, rf_sl_ext) = exp_C()
    t4 = time.time()

    def conv_times(cur, lg):
        return np.array([[first_cross(cur[s, :, q], lg, THRESH) for q in range(len(FREQS))] for s in range(SEEDS)])

    tA = conv_times(curA, logs)
    tA5 = np.array([[first_cross(curA[s, :, q], logs, 0.5) for q in range(len(FREQS))] for s in range(SEEDS)])
    tAD5 = np.array([[first_cross(cur_adam[s, :, q], logsB, 0.5) for q in range(len(FREQS))] for s in range(SEEDS)])
    tGD = conv_times(cur_gd, logsB)
    tAD = conv_times(cur_adam, logsB)

    def order_stats(T):
        # 逐种子：收敛时刻是否随频率严格单调递增；Kendall τ（截尾记为 +inf）
        Tf = np.where(np.isnan(T), np.inf, T)
        mono = [all((Tf[s][i] < Tf[s][i + 1]) or (np.isinf(Tf[s][i]) and np.isinf(Tf[s][i + 1]))
                    for i in range(len(FREQS) - 1)) for s in range(SEEDS)]
        taus = [stats.kendalltau(FREQS, np.where(np.isinf(Tf[s]), 1e12, Tf[s]))[0] for s in range(SEEDS)]
        return {"monotone_per_seed": mono, "n_monotone": int(sum(mono)), "kendall_tau": mean_ci(taus)}

    def tsum(T):
        out = {}
        for q, k in enumerate(FREQS):
            v = T[:, q]
            out[f"k{k}"] = {"mean_ci": mean_ci(v[~np.isnan(v)]) if np.sum(~np.isnan(v)) >= 2 else None,
                            "n_censored": int(np.isnan(v).sum()), "per_seed": v}
        return out

    # 各模型最终（8000 步）各频率误差
    def final_err(cur):
        return {f"k{k}": mean_ci(cur[:, -1, q]) for q, k in enumerate(FREQS)}

    # MLP vs RF-Adam：t_20/t_1 比值（配对：同一种子编号，不同模型，作描述）
    # B1 预测 vs 实测衰减率
    ratio = emp_rate / pred_rate
    rate_rho = [stats.spearmanr(pred_rate[s], emp_rate[s])[0] for s in range(SEEDS)]

    # 图 A/B
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for a_, cur, lg, ttl in [(ax[0], curA, logs, "MLP（3×128 ReLU，Adam）"),
                              (ax[1], cur_adam, logsB, "随机特征 1024（Adam）"),
                              (ax[2], cur_gd, logsB, "随机特征 1024（GD）")]:
        for q, k in enumerate(FREQS):
            mu = cur[:, :, q].mean(0)
            a_.semilogy(lg, mu, label=f"k={k}")
        a_.axhline(THRESH, color="k", ls=":", lw=1)
        a_.set_title(ttl); a_.set_xlabel("训练步数"); a_.legend()
    ax[0].set_ylabel("相对频率误差 Δ_k")
    fig.tight_layout(); fig.savefig(OUT / "A_frequency_principle.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    kk = np.arange(1, 41)
    lm = np.array(extraB["rayleigh_k1_40_mean"])
    e0 = extraB["eig_seed0"]
    okd = (np.array(e0["domfreq"]) >= 1) & (np.array(e0["domfreq"]) <= 40)
    ax.loglog(np.array(e0["domfreq"])[okd], np.array(e0["ev"])[okd], "o", ms=3, label="核特征值 vs 特征向量主频（种子 0）")
    ax.loglog(kk, lm, "x", ms=3, color="gray", label="Fourier 模 Rayleigh 商（含泄漏）")
    sl = extraB["eig_spectrum_loglog_slope_vs_domfreq"][0]
    ax.set_xlabel("频率 k"); ax.set_ylabel("λ"); ax.legend(fontsize=7); ax.set_title(f"随机特征核谱随频率衰减（特征值斜率 {sl:.2f}）")
    fig.tight_layout(); fig.savefig(OUT / "B_kernel_spectrum.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.loglog(wr, rf_err.mean(0), "o-", label=f"随机特征+最小二乘（斜率 64–512: {rf_fit[0]:.2f}; 256–2048: {rf_fit_ext[0]:.2f}）")
    ax.loglog(wm, mlp_err.mean(0), "s-", label=f"训练的单隐层 MLP（斜率 {mlp_fit[0]:.2f}）")
    w = np.array(wr, float)
    ax.loglog(w, rf_err.mean(0)[-1] * (w / w[-1]) ** -2.0, "k:", label="参考 m^-2")
    ax.set_xlabel("宽度 m"); ax.set_ylabel("测试 RMSE"); ax.legend(fontsize=8); ax.set_title("逼近误差 vs 宽度")
    fig.tight_layout(); fig.savefig(OUT / "C_width_scaling.png"); plt.close(fig)

    p = save_results(__file__, {
        "freqs": FREQS, "threshold": THRESH, "seeds": SEEDS, "steps": int(logs[-1]),
        "A_mlp": {"arch": "1-128-128-128-1 ReLU", "lr": 2e-3, "conv_time": tsum(tA), "order": order_stats(tA), "conv_time_th0.5": tsum(tA5), "order_th0.5": order_stats(tA5), "final_rel_err": final_err(curA),
                  "final_train_mse": mean_ci(mseA),
                  "rel_err_at_steps": {str(st): {f"k{k}": float(curA[:, list(logs).index(st), q].mean()) for q, k in enumerate(FREQS)} for st in [500, 1000, 2000, 4000, 8000]}},
        "B_rf_gd": {"conv_time": tsum(tGD), "order": order_stats(tGD), "final_rel_err": final_err(cur_gd),
                    "rayleigh_lambda_k": {f"k{k}": mean_ci(lam_k[:, q]) for q, k in enumerate(FREQS)},
                    "lambda_ratio_k1_over_k20": mean_ci(lam_k[:, 0] / lam_k[:, 3]),
                    "pred_rate": {f"k{k}": mean_ci(pred_rate[:, q]) for q, k in enumerate(FREQS)},
                    "emp_rate": {f"k{k}": mean_ci(emp_rate[:, q]) for q, k in enumerate(FREQS)},
                    "emp_over_pred": {f"k{k}": mean_ci(ratio[:, q]) for q, k in enumerate(FREQS)},
                    "spearman_pred_emp": mean_ci(rate_rho), "kernel_spectrum": extraB},
        "B_rf_adam": {"conv_time": tsum(tAD), "order": order_stats(tAD), "conv_time_th0.5": tsum(tAD5), "order_th0.5": order_stats(tAD5), "final_rel_err": final_err(cur_adam)},
        "C_width": {"widths": wr, "rf_rmse_mean": rf_err.mean(0), "rf_rmse_ci": [mean_ci(rf_err[:, j])[1] for j in range(len(wr))],
                    "rf_slope_m_ge_64": rf_fit, "rf_slope_per_seed": rf_sl,
                    "mlp_rmse_mean": mlp_err.mean(0), "mlp_rmse_ci": [mean_ci(mlp_err[:, j])[1] for j in range(len(wm))],
                    "mlp_slope_m_ge_64": mlp_fit, "mlp_slope_per_seed": mlp_sl,
                    "rf_slope_m_256_2048": rf_fit_ext, "rf_slope_ext_per_seed": rf_sl_ext,
                    "rf_local_slopes": np.diff(np.log(rf_err.mean(0))) / np.diff(np.log(wr)),
                    "mlp_steps": 2000, "mlp_lr": 3e-3, "target_std": float(np.std(target(np.arange(4096) / 4096)))},
        "runtime_sec": {"A": t2 - t1, "B": t3 - t2, "C": t4 - t3, "total": time.time() - T0},
    })
    print("saved", p, time.time() - T0)
