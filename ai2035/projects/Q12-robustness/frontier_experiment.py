"""Q12 前沿实验：随机平滑（Cohen et al. 2019）认证的紧性、可靠性与意义边界（digits）.

检验四个理论预测：
  T1 线性紧性：二分类线性模型的平滑分类器，Cohen 半径（精确 pA）= 到超平面的距离；
     蒙特卡洛（MC）证书只会更保守，保守程度由 Clopper–Pearson 下界决定。
  T2 采样预算上限：N 个噪声样本、置信 1-α 下，可认证半径不超过 R_max = σ Φ^{-1}(α^{1/N})。
  T3 种子方差：同一数据、不同训练种子的模型，在逐样本层面上的证书是否一致（Shi et al. 2026, arXiv 2601.13303 的论断）。
  T4 认证-经验差距与 L∞ 含义：SmoothAdv 式 L2 攻击下的经验稳健准确率 vs 认证准确率；L2 证书换算到 L∞ 需除以 √d。
输出：results/frontier.json，results/frontier_*.png
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
from scipy.stats import beta, norm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from common.util import SEED, mean_ci, plt, results_dir, rng

T0 = time.time()
SIGMAS = [0.12, 0.25, 0.5]
SEEDS = list(range(5))
ALPHA = 0.001
N0, N_MAIN = 100, 2000
N_SWEEP = [100, 1000, 10000]
N_CERT = 300           # 认证的测试点数（固定同一批点，便于跨种子逐样本比较）
RADII = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
D = 64


# --------------------------------- models ---------------------------------
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, g, h=128):
        self.W1 = g.standard_normal((D, h)) * np.sqrt(2 / D)
        self.b1 = np.zeros(h)
        self.W2 = g.standard_normal((h, 10)) * np.sqrt(2 / h)
        self.b2 = np.zeros(10)

    def logits(self, X):
        return np.maximum(X @ self.W1 + self.b1, 0) @ self.W2 + self.b2

    def grad_input_ce(self, X, y_soft_target_fn):
        raise NotImplementedError


def train_mlp(X, y, g, sigma, epochs=60, lr=2e-3, bs=64):
    net = MLP(g)
    P = [net.W1, net.b1, net.W2, net.b2]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    t = 0
    for _ in range(epochs):
        idx = g.permutation(len(X))
        for s in range(0, len(X), bs):
            b = idx[s:s + bs]
            xb = X[b] + sigma * g.standard_normal(X[b].shape)        # 高斯噪声增强（Cohen et al. 2019）
            h = np.maximum(xb @ net.W1 + net.b1, 0)
            p = softmax(h @ net.W2 + net.b2)
            p[np.arange(len(b)), y[b]] -= 1
            p /= len(b)
            dh = (p @ net.W2.T) * (h > 0)
            G = [xb.T @ dh + 1e-4 * net.W1, dh.sum(0), h.T @ p + 1e-4 * net.W2, p.sum(0)]
            t += 1
            for i in range(4):
                m[i] = 0.9 * m[i] + 0.1 * G[i]
                v[i] = 0.999 * v[i] + 0.001 * G[i] ** 2
                P[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return net


def counts(predict, x, sigma, n, g, ncls=10, chunk=5000):
    c = np.zeros(ncls, int)
    done = 0
    while done < n:
        b = min(chunk, n - done)
        pr = predict(x[None, :] + sigma * g.standard_normal((b, x.size)))
        c += np.bincount(pr, minlength=ncls)
        done += b
    return c


def cp_lower(k, n, alpha=ALPHA):
    return 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))


def certify(predict, x, sigma, n0, n, g):
    """Cohen et al. 2019 CERTIFY. 返回 (预测类别 或 -1 弃权, 半径)。"""
    cA = int(counts(predict, x, sigma, n0, g).argmax())
    c = counts(predict, x, sigma, n, g)
    pA = cp_lower(c[cA], n)
    if pA < 0.5:
        return -1, 0.0
    return cA, float(sigma * norm.ppf(pA))


def cert_acc(preds, radii, y, r):
    return float(np.mean((preds == y) & (radii >= r)))


# ------------------------------------ data ------------------------------------
dg = load_digits()
X, Y = dg.data / 16.0, dg.target
Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=0)
Xc, yc = Xte[:N_CERT], yte[:N_CERT]

# ===================== T1: linear tightness (binary 3 vs 8) =====================
m38tr = np.isin(ytr, [3, 8])
m38te = np.isin(yte, [3, 8])
Xb, yb = Xtr[m38tr], (ytr[m38tr] == 8).astype(int)
Xbt, ybt = Xte[m38te], (yte[m38te] == 8).astype(int)
from sklearn.linear_model import LogisticRegression

T1 = []
for seed in SEEDS:
    gS = rng(100 + seed)
    boot = gS.integers(0, len(Xb), len(Xb))                       # 种子 = bootstrap 训练集
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(Xb[boot], yb[boot])
    w, b0 = clf.coef_[0], clf.intercept_[0]
    pred_lin = lambda Z, w=w, b0=b0: (Z @ w + b0 > 0).astype(int)
    for sigma in SIGMAS:
        for N in N_SWEEP:
            g = rng(200 + 10 * seed + N % 97 + int(100 * sigma))
            ratios, sound = [], []
            for x, yy in zip(Xbt, ybt):
                marg = x @ w + b0
                r_exact = abs(marg) / np.linalg.norm(w)            # 平滑线性分类器的精确稳健半径
                cls, r = certify(lambda Z: pred_lin(Z), x, sigma, N0, N, g)
                if cls == -1:
                    ratios.append(0.0)
                    sound.append(True)
                    continue
                ratios.append(r / r_exact)
                sound.append(r <= r_exact + 1e-9)
            # 理论：若 pA 精确，半径 = r_exact；MC + CP 下界后的期望比值
            T1.append(dict(seed=seed, sigma=sigma, N=N, median_ratio=float(np.median(ratios)),
                           mean_ratio=float(np.mean(ratios)), sound_frac=float(np.mean(sound)),
                           cap=float(sigma * norm.ppf(ALPHA ** (1 / N)))))
print(f"T1 {time.time()-T0:.0f}s", flush=True)

# ===================== main: MLP smoothing, seeds x sigmas =====================
main = {}
nets = {}
for sigma in SIGMAS:
    for seed in SEEDS:
        net = train_mlp(Xtr, ytr, rng(300 + seed), sigma)
        nets[(sigma, seed)] = net
        predict = lambda Z, net=net: net.logits(Z).argmax(1)
        g = rng(400 + seed + int(100 * sigma))
        preds, radii = np.zeros(N_CERT, int), np.zeros(N_CERT)
        for i in range(N_CERT):
            preds[i], radii[i] = certify(predict, Xc[i], sigma, N0, N_MAIN, g)
        clean_base = float((net.logits(Xte).argmax(1) == yte).mean())
        main[(sigma, seed)] = dict(preds=preds, radii=radii, clean_base=clean_base)
    print(f"main sigma={sigma} {time.time()-T0:.0f}s", flush=True)

# ===================== T2: sample-budget cap (sigma=0.25, seed 0) =====================
T2 = []
for N in N_SWEEP:
    net = nets[(0.25, 0)]
    predict = lambda Z, net=net: net.logits(Z).argmax(1)
    g = rng(500 + N % 991)
    nn_ = 150
    preds, radii = np.zeros(nn_, int), np.zeros(nn_)
    for i in range(nn_):
        preds[i], radii[i] = certify(predict, Xc[i], 0.25, N0, N, g)
    cap = 0.25 * norm.ppf(ALPHA ** (1 / N))
    T2.append(dict(N=N, cap_theory=float(cap), max_radius=float(radii.max()),
                   frac_at_cap=float(np.mean(np.isclose(radii, cap, rtol=1e-6))),
                   cert_acc={str(r): cert_acc(preds, radii, yc[:nn_], r) for r in RADII}))
print(f"T2 {time.time()-T0:.0f}s", flush=True)


# ===================== T4: empirical attack on the smoothed classifier =====================
def smooth_attack(net, x, y, sigma, eps, g, steps=20, m=32):
    """SmoothAdv（Salman et al. 2019）式 L2-PGD：最大化平均 softmax 的负对数概率。"""
    delta = np.zeros_like(x)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        E = sigma * g.standard_normal((m, x.size))
        Z = x + delta + E
        h = np.maximum(Z @ net.W1 + net.b1, 0)
        p = softmax(h @ net.W2 + net.b2)
        pbar = p[:, y].mean()
        # d(-log pbar)/dZ_j = -(1/(m pbar)) * dp_y/dZ_j ; dp_y/dlogits = p_y (e_y - p)
        dlog = -(p[:, [y]] * (np.eye(10)[y] - p)) / (m * pbar)
        dh = (dlog @ net.W2.T) * (h > 0)
        gx = (dh @ net.W1.T).sum(0)
        gn = np.linalg.norm(gx) + 1e-12
        delta = delta + alpha * gx / gn
        nd = np.linalg.norm(delta)
        if nd > eps:
            delta *= eps / nd
        delta = np.clip(x + delta, 0, 1) - x
    return x + delta


T4 = []
N_ATT = 120
for sigma in SIGMAS:
    net = nets[(sigma, 0)]
    predict = lambda Z, net=net: net.logits(Z).argmax(1)
    g = rng(600 + int(100 * sigma))
    preds, radii = main[(sigma, 0)]["preds"][:N_ATT], main[(sigma, 0)]["radii"][:N_ATT]
    for eps in [0.25, 0.5, 1.0]:
        ok = []
        for i in range(N_ATT):
            xa = smooth_attack(net, Xc[i], yc[i], sigma, eps, g)
            c = counts(predict, xa, sigma, 500, g)
            ok.append(int(c.argmax()) == yc[i])
        T4.append(dict(sigma=sigma, eps=eps, empirical_robust_acc=float(np.mean(ok)),
                       certified_acc=cert_acc(preds, radii, yc[:N_ATT], eps)))
print(f"T4 {time.time()-T0:.0f}s", flush=True)

# ================================= aggregation =================================
out = {"config": dict(sigmas=SIGMAS, seeds=SEEDS, alpha=ALPHA, n0=N0, n_main=N_MAIN, n_sweep=N_SWEEP, n_cert=N_CERT,
                      radii=RADII, d=D, base="MLP 64-128-10, Gaussian noise augmentation, 60 epochs",
                      linear="LogisticRegression digits 3 vs 8, seeds = bootstrap resamples", n_attack=N_ATT)}
out["T1_linear_tightness"] = {}
for sigma in SIGMAS:
    for N in N_SWEEP:
        rows = [r for r in T1 if r["sigma"] == sigma and r["N"] == N]
        out["T1_linear_tightness"][f"sigma={sigma}|N={N}"] = {
            k: dict(zip(["mean", "ci95"], mean_ci([r[k] for r in rows]))) for k in ["median_ratio", "mean_ratio", "sound_frac"]}
        out["T1_linear_tightness"][f"sigma={sigma}|N={N}"]["cap_theory"] = rows[0]["cap"]
out["main"] = {}
pair_stats = {}
for sigma in SIGMAS:
    e = {}
    for r in RADII:
        e[f"cert_acc@{r}"] = dict(zip(["mean", "ci95"], mean_ci([cert_acc(main[(sigma, s)]["preds"], main[(sigma, s)]["radii"], yc, r) for s in SEEDS])))
        e[f"cert_acc@{r}_sd_across_seeds"] = float(np.std([cert_acc(main[(sigma, s)]["preds"], main[(sigma, s)]["radii"], yc, r) for s in SEEDS], ddof=1))
    e["cap_theory_N_main"] = float(sigma * norm.ppf(ALPHA ** (1 / N_MAIN)))
    e["frac_at_cap_N_main"] = dict(zip(["mean", "ci95"], mean_ci([np.mean(np.isclose(main[(sigma, s)]["radii"], sigma * norm.ppf(ALPHA ** (1 / N_MAIN)), rtol=1e-6)) for s in SEEDS])))
    e["clean_base"] = dict(zip(["mean", "ci95"], mean_ci([main[(sigma, s)]["clean_base"] for s in SEEDS])))
    e["abstain_rate"] = dict(zip(["mean", "ci95"], mean_ci([np.mean(main[(sigma, s)]["preds"] == -1) for s in SEEDS])))
    # test-set bootstrap sd of certified accuracy at r=0.5 (seed 0)
    gb = rng(700)
    ok = (main[(sigma, 0)]["preds"] == yc) & (main[(sigma, 0)]["radii"] >= 0.5)
    e["cert_acc@0.5_sd_test_bootstrap"] = float(np.std([ok[gb.integers(0, N_CERT, N_CERT)].mean() for _ in range(1000)]))
    # per-instance disagreement across seed pairs
    dis, sp, dis_any = [], [], []
    for a in SEEDS:
        for b in SEEDS:
            if a < b:
                ca = (main[(sigma, a)]["preds"] == yc) & (main[(sigma, a)]["radii"] >= 0.5)
                cb = (main[(sigma, b)]["preds"] == yc) & (main[(sigma, b)]["radii"] >= 0.5)
                dis.append(float(np.mean(ca != cb)))
                ra = np.where(main[(sigma, a)]["preds"] == yc, main[(sigma, a)]["radii"], 0)
                rb = np.where(main[(sigma, b)]["preds"] == yc, main[(sigma, b)]["radii"], 0)
                sp.append(float(stats.spearmanr(ra, rb).statistic))
    allc = np.array([(main[(sigma, s)]["preds"] == yc) & (main[(sigma, s)]["radii"] >= 0.5) for s in SEEDS])
    e["pairwise_disagreement@0.5"] = dict(zip(["mean", "ci95"], mean_ci(dis)))
    e["pairwise_spearman_radius"] = dict(zip(["mean", "ci95"], mean_ci(sp)))
    # 命题 4：若各测试点的证书在种子间独立翻转，则聚合认证准确率的跨种子标准差 ≈ sqrt(D/(2n))
    e["pred_seed_sd_if_independent"] = float(np.sqrt(np.mean(dis) / (2 * N_CERT)))
    e["frac_certified_by_all_seeds@0.5"] = float(allc.all(0).mean())
    e["frac_certified_by_some_seed@0.5"] = float(allc.any(0).mean())
    # L∞ implication: L2 radius R certifies L∞ ball of radius R/sqrt(d)
    for einf in [0.05, 0.1]:
        e[f"cert_acc_Linf@{einf}"] = dict(zip(["mean", "ci95"], mean_ci([cert_acc(main[(sigma, s)]["preds"], main[(sigma, s)]["radii"], yc, einf * np.sqrt(D)) for s in SEEDS])))
    out["main"][f"sigma={sigma}"] = e
out["T2_budget"] = T2
out["T4_empirical_vs_certified"] = T4
out["runtime_sec"] = time.time() - T0
out["_meta"] = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
R = results_dir(__file__)
(R / "frontier.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

# ----------------------------------- figures -----------------------------------
P = plt()
fig, axes = P.subplots(1, 3, figsize=(15, 4.2))
for sigma in SIGMAS:
    e = out["main"][f"sigma={sigma}"]
    axes[0].errorbar(RADII, [e[f"cert_acc@{r}"]["mean"] for r in RADII], yerr=[e[f"cert_acc@{r}"]["ci95"] for r in RADII],
                     marker="o", capsize=3, label=f"σ={sigma}")
for sigma, c in zip(SIGMAS, ["C0", "C1", "C2"]):
    rows = [t for t in T4 if t["sigma"] == sigma]
    axes[0].plot([t["eps"] for t in rows], [t["empirical_robust_acc"] for t in rows], "x--", color=c)
axes[0].set_xlabel("L2 半径")
axes[0].set_ylabel("准确率")
axes[0].set_title("认证准确率（实线, 5 种子）vs SmoothAdv 经验稳健准确率（×虚线, 种子 0）")
axes[0].legend(fontsize=8)
for N, mk in zip(N_SWEEP, ["o", "s", "^"]):
    t2 = [t for t in T2 if t["N"] == N][0]
    axes[1].plot(RADII, [t2["cert_acc"][str(r)] for r in RADII], marker=mk, label=f"N={N}（理论上限 {t2['cap_theory']:.2f}）")
    axes[1].axvline(t2["cap_theory"], ls=":", color="gray")
axes[1].set_xlabel("L2 半径")
axes[1].set_title("采样预算决定可认证半径上限（σ=0.25）")
axes[1].legend(fontsize=8)
xs = np.arange(len(SIGMAS))
axes[2].bar(xs - 0.2, [out["main"][f"sigma={s}"]["frac_certified_by_some_seed@0.5"] for s in SIGMAS], 0.4, label="至少一个种子认证")
axes[2].bar(xs + 0.2, [out["main"][f"sigma={s}"]["frac_certified_by_all_seeds@0.5"] for s in SIGMAS], 0.4, label="5 个种子都认证")
axes[2].set_xticks(xs, [f"σ={s}" for s in SIGMAS])
axes[2].set_title("同一测试点在 r=0.5 的证书跨种子一致性")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "frontier_smoothing.png")
print(f"done {time.time()-T0:.0f}s")
