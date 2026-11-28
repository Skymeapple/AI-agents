"""Q07 持续学习与灾难性遗忘：Split-digits（5 个任务 × 2 类），类增量 / 域增量 / 任务增量三种场景（van de Ven & Tolias 2019）。

自写 numpy MLP（64-128-10，ReLU，SGD）。方法（全部超参数事先固定，λ 以预注册网格全部报告）：
  naive 顺序微调；经验回放（类均衡缓冲区，总容量 M∈{20,100,500}）；
  EWC（对角经验 Fisher，λ∈{1e2,1e3,1e4}）；L2-到上一任务权重（λ∈{0.1,1}）；
  累积联合训练（上界：每个阶段用全部已见数据训练）。
指标：最终平均准确率 ACC、遗忘 F、反向迁移 BWT、学习准确率 LA（可塑性）。10 个种子（类顺序与划分随机）。
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, save_results, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

T0 = time.time()
OUT = results_dir(__file__)
SEEDS = 10
H, LR, BS, EPOCHS = 128, 0.05, 32, 30
NT = 5

METHODS = [("naive", {}), ("replay", {"M": 20}), ("replay", {"M": 100}), ("replay", {"M": 500}),
           ("ewc", {"lam": 1e2}), ("ewc", {"lam": 1e3}), ("ewc", {"lam": 1e4}),
           ("l2", {"lam": 0.1}), ("l2", {"lam": 1.0}), ("joint", {})]


def mname(m, p):
    return {"naive": "朴素微调", "joint": "累积联合(上界)"}.get(m) or (
        f"回放 M={p['M']}" if m == "replay" else f"EWC λ={p['lam']:g}" if m == "ewc" else f"L2 λ={p['lam']:g}")


class MLP:
    def __init__(self, g, d=64, h=H, k=10):
        self.p = {"W1": g.standard_normal((d, h)) * np.sqrt(2 / d), "b1": np.zeros(h),
                  "W2": g.standard_normal((h, k)) * np.sqrt(1 / h), "b2": np.zeros(k)}

    def forward(self, X):
        z = X @ self.p["W1"] + self.p["b1"]; hdn = np.maximum(z, 0)
        return z, hdn, hdn @ self.p["W2"] + self.p["b2"]

    def grads(self, X, y, mask, per_sample_sq=False):
        """mask: (n,10) bool of allowed logits. Returns mean grads (and sum of squared per-sample grads)."""
        z, hdn, o = self.forward(X)
        o = np.where(mask, o, -1e9); o = o - o.max(1, keepdims=True)
        P = np.exp(o); P /= P.sum(1, keepdims=True)
        n = len(y); D = P.copy(); D[np.arange(n), y] -= 1   # dL/do per sample
        dh = (D @ self.p["W2"].T) * (z > 0)
        g = {"W2": hdn.T @ D / n, "b2": D.mean(0), "W1": X.T @ dh / n, "b1": dh.mean(0)}
        if per_sample_sq:
            sq = {"W2": (hdn ** 2).T @ (D ** 2), "b2": (D ** 2).sum(0),
                  "W1": (X ** 2).T @ (dh ** 2), "b1": (dh ** 2).sum(0)}
            return g, sq
        return g

    def predict(self, X, mask):
        o = self.forward(X)[2]
        return np.where(mask, o, -np.inf).argmax(1)


def make_mask(y, task_of, task_classes, scenario, seen):
    m = np.zeros((len(y), 10), bool)
    if scenario == "domain":
        m[:, :2] = True
    elif scenario == "task":
        for i, c in enumerate(y):
            m[i, task_classes[task_of[c]]] = True
    else:
        m[:, seen] = True
    return m


def run(seed, scenario, method, prm, data):
    g = rng(1000 * seed + 7)
    Xtr, ytr, Xte, yte, order = data
    task_classes = [order[2 * t:2 * t + 2] for t in range(NT)]
    task_of = {c: t for t, cs in enumerate(task_classes) for c in cs}
    pos = np.zeros(10, int)
    for cs in task_classes:
        pos[cs[1]] = 1
    tgt = (lambda yy: pos[yy]) if scenario == "domain" else (lambda yy: yy)   # Domain-IL: shared 2-way head
    net = MLP(g)
    R = np.zeros((NT, NT))
    anchors = []            # EWC / L2: list of (params, fisher)
    buf_X, buf_y = np.zeros((0, 64)), np.zeros(0, int)
    seen = []
    for t in range(NT):
        seen = seen + list(task_classes[t])
        idx = np.isin(ytr, task_classes[t])
        if method == "joint":
            idx = np.isin(ytr, seen)
        X, y = Xtr[idx], ytr[idx]
        for ep in range(EPOCHS):
            perm = g.permutation(len(y))
            for b in range(0, len(y), BS):
                bi = perm[b:b + BS]; Xb, yb = X[bi], y[bi]
                if method == "replay" and len(buf_y):
                    ri = g.integers(0, len(buf_y), len(bi))
                    Xb = np.vstack([Xb, buf_X[ri]]); yb = np.r_[yb, buf_y[ri]]
                mask = make_mask(yb, task_of, task_classes, scenario, seen)
                gr = net.grads(Xb, tgt(yb), mask)
                for k in gr:
                    net.p[k] -= LR * gr[k]
                if anchors:
                    # quadratic penalty 0.5*sum_a c_a (θ-θ_a)^2 applied as an exact proximal (implicit) step,
                    # which is stable for any λ (explicit steps diverge when LR*λ*F > 2)
                    for k in net.p:
                        num, den = net.p[k].copy(), np.ones_like(net.p[k])
                        for (pa, F) in anchors:
                            c = LR * prm["lam"] * (F[k] if method == "ewc" else 1.0)
                            num += c * pa[k]; den += c
                        net.p[k] = num / den
        # after task t
        if method == "ewc":
            mask = make_mask(y, task_of, task_classes, scenario, seen)
            _, sq = net.grads(X, tgt(y), mask, per_sample_sq=True)
            F = {k: v / len(y) for k, v in sq.items()}
            anchors.append(({k: v.copy() for k, v in net.p.items()}, F))
        elif method == "l2":
            anchors = [({k: v.copy() for k, v in net.p.items()}, None)]
        elif method == "replay":
            per = prm["M"] // len(seen)
            bx, by = [], []
            for c in seen:
                src = (buf_X[buf_y == c], buf_y[buf_y == c]) if c not in task_classes[t] else (X[y == c], y[y == c])
                sel = g.permutation(len(src[1]))[:per]
                bx.append(src[0][sel]); by.append(src[1][sel])
            buf_X, buf_y = np.vstack(bx), np.concatenate(by)
        for i in range(t + 1):
            te = np.isin(yte, task_classes[i])
            mask = make_mask(yte[te], task_of, task_classes, scenario, seen)
            R[t, i] = (net.predict(Xte[te], mask) == tgt(yte[te])).mean()
    acc = R[-1].mean()
    bwt = np.mean([R[-1, i] - R[i, i] for i in range(NT - 1)])
    fgt = np.mean([R[:NT - 1, i].max() - R[-1, i] for i in range(NT - 1)])
    la = np.mean([R[i, i] for i in range(NT)])
    return {"ACC": acc, "BWT": bwt, "F": fgt, "LA": la, "R": R}


Xall, yall = load_digits(return_X_y=True); Xall = Xall / 16.0
raw = {sc: {mname(m, p): [] for m, p in METHODS} for sc in ["class", "domain", "task"]}
for s in range(SEEDS):
    g = rng(s)
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, stratify=yall, random_state=int(g.integers(1e6)))
    order = list(g.permutation(10))
    for sc in ["class", "domain", "task"]:
        for m, p in METHODS:
            raw[sc][mname(m, p)].append(run(s, sc, m, p, (Xtr, ytr, Xte, yte, order)))
    print("seed", s, round(time.time() - T0, 1), flush=True)

summary = {}
for sc in raw:
    summary[sc] = {}
    naive = np.array([r["ACC"] for r in raw[sc]["朴素微调"]])
    for name, runs in raw[sc].items():
        ent = {}
        for met in ["ACC", "BWT", "F", "LA"]:
            mu, ci = mean_ci([r[met] for r in runs]); ent[met] = {"mean": mu, "ci95": ci}
        a = np.array([r["ACC"] for r in runs])
        if name != "朴素微调":
            d = a - naive; ent["ACC_diff_vs_naive"] = {"mean": mean_ci(d)[0], "ci95": mean_ci(d)[1],
                                                        "p_paired": float(stats.ttest_rel(a, naive).pvalue)}
        ent["R_mean"] = np.mean([r["R"] for r in runs], axis=0)
        summary[sc][name] = ent
# paired comparisons of interest
cmp = {}
for sc in raw:
    def arr(n): return np.array([r["ACC"] for r in raw[sc][n]])
    for a_, b_ in [("EWC λ=1000", "朴素微调"), ("回放 M=100", "EWC λ=1000"), ("回放 M=20", "EWC λ=10000"), ("累积联合(上界)", "回放 M=500")]:
        d = arr(a_) - arr(b_)
        cmp[f"{sc}: {a_} - {b_}"] = {"mean": mean_ci(d)[0], "ci95": mean_ci(d)[1],
                                     "p_paired": float(stats.ttest_rel(arr(a_), arr(b_)).pvalue)}

# ---- figures ----
P = plt()
names = [mname(m, p) for m, p in METHODS]
fig, axes = P.subplots(1, 3, figsize=(20, 4.8), sharey=True)
for ax, sc, tt in zip(axes, ["class", "domain", "task"], ["类增量 (Class-IL)", "域增量 (Domain-IL)", "任务增量 (Task-IL)"]):
    xs = np.arange(len(names)); w = 0.27
    for j, met in enumerate(["ACC", "F", "LA"]):
        mu = [summary[sc][n][met]["mean"] for n in names]; ci = [summary[sc][n][met]["ci95"] for n in names]
        ax.bar(xs + (j - 1) * w, mu, w, yerr=ci, capsize=2,
               label={"ACC": "最终平均准确率 ACC", "F": "遗忘 F", "LA": "学习准确率 LA"}[met])
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_title(tt); ax.grid(axis="y", alpha=.3)
axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "metrics.png"); P.close(fig)

fig, axes = P.subplots(3, 4, figsize=(15, 10.5))
for row, sc in enumerate(["class", "domain", "task"]):
    for ax, n in zip(axes[row], ["朴素微调", "EWC λ=1000", "回放 M=100", "累积联合(上界)"]):
        im = ax.imshow(summary[sc][n]["R_mean"], vmin=0, vmax=1, cmap="viridis")
        R = summary[sc][n]["R_mean"]
        for i in range(NT):
            for j in range(i + 1):
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if R[i, j] < .6 else "k")
        ax.set_title(f"{sc.capitalize()}-IL: {n}", fontsize=9)
        ax.set_xlabel("评估任务 i"); ax.set_ylabel("训练完任务 l")
fig.tight_layout(); fig.savefig(OUT / "acc_matrices.png"); P.close(fig)

# accuracy on the average of seen tasks after each stage
fig, axes = P.subplots(1, 3, figsize=(17, 4.2), sharey=True)
for ax, sc in zip(axes, ["class", "domain", "task"]):
    for n in ["朴素微调", "EWC λ=1000", "L2 λ=1", "回放 M=20", "回放 M=100", "回放 M=500", "累积联合(上界)"]:
        curves = np.array([[r["R"][t, :t + 1].mean() for t in range(NT)] for r in raw[sc][n]])
        mu = curves.mean(0); ci = 1.96 * curves.std(0, ddof=1) / np.sqrt(len(curves))
        ax.errorbar(np.arange(1, NT + 1), mu, yerr=ci, marker="o", capsize=2, label=n)
    ax.set_xlabel("已学任务数"); ax.set_ylabel("已见任务平均准确率"); ax.set_title(sc.capitalize() + "-IL")
    ax.grid(alpha=.3)
axes[2].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "stage_curves.png"); P.close(fig)

runtime = time.time() - T0
save_results(__file__, {"config": {"seeds": SEEDS, "hidden": H, "lr": LR, "batch": BS, "epochs_per_task": EPOCHS,
                                   "scenarios": "class-IL: single 10-way head over seen classes; domain-IL: shared 2-way head (first/second class of pair), no task id; task-IL: task-specific 2-way masked head",
                                   "tasks": "5 tasks x 2 classes, class order permuted per seed; 70/30 stratified split",
                                   "replay": "class-balanced buffer, replay batch same size as current batch",
                                   "ewc": "diagonal empirical Fisher per task, separate quadratic penalty per task",
                                   "l2": "penalty to weights after previous task",
                                   "penalty_update": "proximal (implicit) step for quadratic penalties"},
                        "summary": summary, "paired_comparisons": cmp, "runtime_sec": runtime})
print("runtime", runtime)
