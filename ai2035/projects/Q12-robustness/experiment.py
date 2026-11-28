"""Q12 稳健性与对抗安全：digits 上的 FGSM/PGD、随机噪声对照、对抗训练权衡与迁移性.

模型全部用 numpy 自实现（以便精确求输入梯度）：
  * LR  : 多类 softmax 逻辑回归（= 无隐藏层网络）
  * MLP : 64-256-256-10 ReLU 网络
攻击：FGSM、PGD（L∞，随机起点，20 步）、同 L∞ 范数的随机符号噪声（对照）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from common.util import mean_ci, plt, results_dir, rng, save_results

T0 = time.time()
SEEDS = list(range(5))
EPS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPS_TRAIN = [0.0, 0.05, 0.1, 0.2, 0.3]
EPS_REF = 0.1          # 迁移与权衡的参考扰动
PGD_STEPS, PGD_TRAIN_STEPS = 20, 5
ADV_EPOCHS = 40


# ------------------------------- numpy network -------------------------------
class Net:
    def __init__(self, sizes, g):
        self.W = [g.standard_normal((a, b)) * np.sqrt(2.0 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X, keep=False):
        hs, h = [X], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
            hs.append(h)
        return (h, hs) if keep else h

    def backward(self, hs, dlogits):
        """returns param grads and input grad"""
        gW, gb, d = [], [], dlogits
        for i in reversed(range(len(self.W))):
            gW.insert(0, hs[i].T @ d)
            gb.insert(0, d.sum(0))
            d = d @ self.W[i].T
            if i > 0:
                d = d * (hs[i] > 0)
        return gW, gb, d

    def loss_grad_input(self, X, y):
        logits, hs = self.forward(X, keep=True)
        p = softmax(logits)
        d = p.copy()
        d[np.arange(len(y)), y] -= 1
        _, _, dx = self.backward(hs, d)  # gradient of per-example CE wrt its input
        return ce(p, y), dx

    def predict(self, X):
        return self.forward(X).argmax(1)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ce(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


def train(sizes, X, y, g, epochs=60, lr=2e-3, wd=1e-4, adv_eps=0.0, bs=64):
    net = Net(sizes, g)
    mW = [np.zeros_like(w) for w in net.W] + [np.zeros_like(b) for b in net.b]
    vW = [np.zeros_like(w) for w in mW]
    t = 0
    for ep in range(epochs):
        idx = g.permutation(len(X))
        for s in range(0, len(X), bs):
            bi = idx[s:s + bs]
            xb, yb = X[bi], y[bi]
            if adv_eps > 0:  # Madry-style: train on PGD examples only
                xb = pgd(net, xb, yb, adv_eps, PGD_TRAIN_STEPS, g)
            logits, hs = net.forward(xb, keep=True)
            d = softmax(logits)
            d[np.arange(len(yb)), yb] -= 1
            d /= len(yb)
            gW, gb, _ = net.backward(hs, d)
            grads = [gw + wd * w for gw, w in zip(gW, net.W)] + gb
            params = net.W + net.b
            t += 1
            for k, (p_, g_) in enumerate(zip(params, grads)):
                mW[k] = 0.9 * mW[k] + 0.1 * g_
                vW[k] = 0.999 * vW[k] + 0.001 * g_**2
                p_ -= lr * (mW[k] / (1 - 0.9**t)) / (np.sqrt(vW[k] / (1 - 0.999**t)) + 1e-8)
    return net


# ---------------------------------- attacks ----------------------------------
def fgsm(net, X, y, eps):
    if eps == 0:
        return X
    _, g = net.loss_grad_input(X, y)
    return np.clip(X + eps * np.sign(g), 0, 1)


def pgd(net, X, y, eps, steps, g):
    if eps == 0:
        return X
    alpha = 2.5 * eps / steps
    Xa = np.clip(X + g.uniform(-eps, eps, X.shape), 0, 1)
    for _ in range(steps):
        _, gr = net.loss_grad_input(Xa, y)
        Xa = np.clip(np.clip(Xa + alpha * np.sign(gr), X - eps, X + eps), 0, 1)
    return Xa


def rand_sign(X, eps, g):
    return np.clip(X + eps * g.choice([-1.0, 1.0], X.shape), 0, 1)


def acc(net, X, y):
    return float((net.predict(X) == y).mean())


# ---------------------------------- main loop --------------------------------
digits = load_digits()
Xall, yall = digits.data / 16.0, digits.target
ARCH = {"LR": [64, 10], "MLP": [64, 256, 256, 10]}
rec = {k: [] for k in ["curve", "linear", "tradeoff", "transfer"]}
for seed in SEEDS:
    g = rng(seed)
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, stratify=yall, random_state=seed)
    nets = {"LR": train(ARCH["LR"], Xtr, ytr, g, epochs=100, lr=1e-2),
            "MLP": train(ARCH["MLP"], Xtr, ytr, g),
            "MLP_b": train(ARCH["MLP"], Xtr, ytr, rng(100 + seed))}  # 同数据、不同初始化
    # (1) accuracy vs eps curves
    for name in ["LR", "MLP"]:
        net = nets[name]
        for eps in EPS:
            rec["curve"].append(dict(seed=seed, model=name, eps=eps,
                                     fgsm=acc(net, fgsm(net, Xte, yte, eps), yte),
                                     pgd=acc(net, pgd(net, Xte, yte, eps, PGD_STEPS, g), yte),
                                     rand=acc(net, rand_sign(Xte, eps, g), yte)))
        # (2) linearity diagnostics at small eps
        L0, gr = net.loss_grad_input(Xte, yte)
        l1, l2 = np.abs(gr).sum(1), np.sqrt((gr**2).sum(1))
        for eps in [0.01, 0.05, 0.1]:
            La, _ = net.loss_grad_input(np.clip(Xte + eps * np.sign(gr), 0, 1), yte)
            Lr, _ = net.loss_grad_input(rand_sign(Xte, eps, g), yte)
            ok = l1 > 1e-8
            rec["linear"].append(dict(
                seed=seed, model=name, eps=eps,
                l1_over_l2=float(np.median(l1[ok] / l2[ok])),
                # 一阶预测的损失增量 eps*||g||_1 与实际增量之比（裁剪到 [0,1] 会削弱实际增量）
                actual_over_firstorder=float(np.median((La - L0)[ok] / (eps * l1[ok]))),
                # 线性模型下随机符号噪声 |g·s| 的期望约为 ||g||_2*sqrt(2/pi)，故预测比值为 ||g||_1/(||g||_2*sqrt(2/pi))
                pred_adv_over_rand=float(np.mean(l1) / (np.mean(l2) * np.sqrt(2 / np.pi))),
                adv_over_rand_loss_increase=float(np.mean(La - L0) / max(np.mean(np.abs(Lr - L0)), 1e-12))))
    # (3) adversarial training trade-off (MLP)
    adv_nets = {}
    for et in EPS_TRAIN:
        net = nets["MLP"] if et == 0 else train(ARCH["MLP"], Xtr, ytr, rng(200 + seed), epochs=ADV_EPOCHS, adv_eps=et)
        adv_nets[et] = net
        row = dict(seed=seed, eps_train=et, clean=acc(net, Xte, yte))
        for ee in [0.05, 0.1, 0.2]:
            row[f"pgd@{ee}"] = acc(net, pgd(net, Xte, yte, ee, PGD_STEPS, g), yte)
        row["train_clean"] = acc(net, Xtr, ytr)
        rec["tradeoff"].append(row)
    # (4) transferability at EPS_REF (PGD)
    srcs = {"LR": nets["LR"], "MLP": nets["MLP"], "MLP_b": nets["MLP_b"], "MLP_adv0.1": adv_nets[0.1]}
    tgts = {"LR": nets["LR"], "MLP": nets["MLP"], "MLP_b": nets["MLP_b"], "MLP_adv0.1": adv_nets[0.1]}
    for sn, sm in srcs.items():
        Xa = pgd(sm, Xte, yte, EPS_REF, PGD_STEPS, g)
        for tn, tm in tgts.items():
            rec["transfer"].append(dict(seed=seed, src=sn, tgt=tn, acc=acc(tm, Xa, yte), clean=acc(tm, Xte, yte)))
    print(f"seed {seed} done {time.time()-T0:.0f}s", flush=True)


def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


out = {"config": dict(seeds=SEEDS, eps=EPS, eps_train=EPS_TRAIN, eps_ref=EPS_REF, pgd_steps=PGD_STEPS,
                      pgd_train_steps=PGD_TRAIN_STEPS, adv_epochs=ADV_EPOCHS, std_epochs=60, arch=ARCH, input_range="[0,1] (pixel/16)", d=64)}
out["curve"] = {f"{m}|eps={e}": {k: ag(rec["curve"], k, model=m, eps=e) for k in ["fgsm", "pgd", "rand"]}
                for m in ["LR", "MLP"] for e in EPS}
out["linear"] = {f"{m}|eps={e}": {k: ag(rec["linear"], k, model=m, eps=e)
                                  for k in ["l1_over_l2", "actual_over_firstorder", "pred_adv_over_rand", "adv_over_rand_loss_increase"]}
                 for m in ["LR", "MLP"] for e in [0.01, 0.05, 0.1]}
out["tradeoff"] = {f"eps_train={et}": {k: ag(rec["tradeoff"], k, eps_train=et)
                                       for k in ["clean", "train_clean", "pgd@0.05", "pgd@0.1", "pgd@0.2"]}
                   for et in EPS_TRAIN}
names = ["LR", "MLP", "MLP_b", "MLP_adv0.1"]
out["transfer"] = {f"{s}->{t}": ag(rec["transfer"], "acc", src=s, tgt=t) for s in names for t in names}
out["transfer_clean"] = {t: ag(rec["transfer"], "clean", src="LR", tgt=t) for t in names}

# ----------------------------- statistical tests -----------------------------
T = {}
for m in ["LR", "MLP"]:
    for e in [0.05, 0.1]:
        a = [r["pgd"] for r in rec["curve"] if r["model"] == m and r["eps"] == e]
        b = [r["rand"] for r in rec["curve"] if r["model"] == m and r["eps"] == e]
        c = [r["fgsm"] for r in rec["curve"] if r["model"] == m and r["eps"] == e]
        T[f"{m}|eps={e}|rand_minus_pgd"] = {"diff": float(np.mean(np.array(b) - a)), "paired_t_p": float(stats.ttest_rel(b, a).pvalue)}
        T[f"{m}|eps={e}|fgsm_minus_pgd"] = {"diff": float(np.mean(np.array(c) - a)), "paired_t_p": float(stats.ttest_rel(c, a).pvalue)}
c0 = [r["clean"] for r in rec["tradeoff"] if r["eps_train"] == 0]
r0 = [r["pgd@0.1"] for r in rec["tradeoff"] if r["eps_train"] == 0]
for et in EPS_TRAIN[1:]:
    c1 = [r["clean"] for r in rec["tradeoff"] if r["eps_train"] == et]
    r1 = [r["pgd@0.1"] for r in rec["tradeoff"] if r["eps_train"] == et]
    T[f"tradeoff|eps_train={et}|clean_drop"] = {"diff": float(np.mean(np.array(c0) - c1)), "paired_t_p": float(stats.ttest_rel(c0, c1).pvalue)}
    T[f"tradeoff|eps_train={et}|robust_gain@0.1"] = {"diff": float(np.mean(np.array(r1) - r0)), "paired_t_p": float(stats.ttest_rel(r1, r0).pvalue)}
# same-architecture transfer vs cross-architecture transfer (drop from clean accuracy)
def drop(s, t):
    a = np.array([r["clean"] - r["acc"] for r in rec["transfer"] if r["src"] == s and r["tgt"] == t])
    return a
for a_, b_ in [(("MLP", "MLP_b"), ("LR", "MLP_b")), (("MLP", "MLP_b"), ("MLP", "LR"))]:
    x, y_ = drop(*a_), drop(*b_)
    T[f"transfer_drop {a_[0]}->{a_[1]} vs {b_[0]}->{b_[1]}"] = {"diff": float(np.mean(x - y_)), "paired_t_p": float(stats.ttest_rel(x, y_).pvalue)}
out["tests"] = T
out["runtime_sec"] = time.time() - T0

# ---------------------------------- figures ----------------------------------
P = plt()
R = results_dir(__file__)
fig, axes = P.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, m in zip(axes, ["LR", "MLP"]):
    for k, lab, c in [("pgd", "PGD-20", "C3"), ("fgsm", "FGSM", "C1"), ("rand", "随机符号噪声(同 L∞)", "C0")]:
        ys = [out["curve"][f"{m}|eps={e}"][k]["mean"] for e in EPS]
        es = [out["curve"][f"{m}|eps={e}"][k]["ci95"] for e in EPS]
        ax.errorbar(EPS, ys, yerr=es, marker="o", capsize=3, label=lab, color=c)
    ax.set_title(f"{m}：准确率 vs 扰动预算 ε（像素∈[0,1]）")
    ax.set_xlabel("ε (L∞)")
axes[0].set_ylabel("测试准确率")
axes[0].legend()
fig.tight_layout()
fig.savefig(R / "acc_vs_eps.png")

fig, ax = P.subplots(figsize=(5.5, 4.2))
xs = [out["tradeoff"][f"eps_train={et}"]["clean"]["mean"] for et in EPS_TRAIN]
ys = [out["tradeoff"][f"eps_train={et}"]["pgd@0.1"]["mean"] for et in EPS_TRAIN]
ax.plot(xs, ys, "o-")
for et, x_, y_ in zip(EPS_TRAIN, xs, ys):
    ax.annotate(f"ε_train={et}", (x_, y_), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("干净准确率")
ax.set_ylabel("PGD 稳健准确率 (ε=0.1)")
ax.set_title("对抗训练：干净 vs 稳健准确率")
fig.tight_layout()
fig.savefig(R / "tradeoff.png")

fig, ax = P.subplots(figsize=(5.5, 4.5))
M = np.array([[out["transfer"][f"{s}->{t}"]["mean"] for t in names] for s in names])
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(len(names)), names, rotation=20)
ax.set_yticks(range(len(names)), names)
ax.set_xlabel("目标模型")
ax.set_ylabel("生成对抗样本的源模型")
ax.set_title("迁移攻击后目标模型准确率 (PGD, ε=0.1)")
fig.colorbar(im)
fig.tight_layout()
fig.savefig(R / "transfer.png")

save_results(__file__, out)
print(f"done {time.time()-T0:.0f}s")
