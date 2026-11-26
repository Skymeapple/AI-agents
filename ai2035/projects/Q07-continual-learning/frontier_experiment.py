"""Q07 前沿实验：可塑性丧失（loss of plasticity）——机制、诊断指标与补救方法。

设置：digits 中固定 300 张图像（输入支撑集固定），每个任务给它们重新随机分配 10 类标签（随机标签记忆任务，
      可塑性研究的标准压力测试）；64-128-128-10 ReLU MLP，Adam(lr=1e-3)，每任务 20 epoch（200 步），共 60 个任务。
方法（超参数事先固定）：基线；L2（解耦权重衰减 λ=1）；L2-init（解耦拉回初始化 λ=1）；
      Shrink&Perturb（任务边界 θ←0.8θ+0.1·初始化尺度噪声）；ReDo（任务边界重置“全零激活”单元）。
理论（paper.md §4）：命题 1——输入支撑集固定时，第一层“死亡”单元在无正则的（Adam/SGD）训练下是吸收态；
      纯 L2 只能把死亡单元的边际推向 0⁻，不能复活；L2-init 使死亡单元沿 θ0+(1-ηλ)^t(θd-θ0) 确定性返回，
      复活步数可逐单元精确预测；噪声/重置以概率或直接方式离开吸收态。
诊断：任务开始时的死亡比例、特征有效秩、权重范数、初始梯度范数，与该任务结束时训练准确率（可训练性）的 Spearman 相关。
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits

T0 = time.time()
OUT = results_dir(__file__)
SEEDS, NT, EPOCHS, BS, NS_ = 5, 60, 20, 30, 300
H, LR = 128, 1e-3
LAM = 1.0
METHODS = ["基线", "L2", "L2-init", "Shrink&Perturb", "ReDo"]
Xall, _ = load_digits(return_X_y=True); Xall = Xall / 16.0


def init(g):
    return {"W1": g.standard_normal((64, H)) * np.sqrt(2 / 64), "b1": np.zeros(H),
            "W2": g.standard_normal((H, H)) * np.sqrt(2 / H), "b2": np.zeros(H),
            "W3": g.standard_normal((H, 10)) * np.sqrt(1 / H), "b3": np.zeros(10)}


def forward(P, x):
    z1 = x @ P["W1"] + P["b1"]; h1 = np.maximum(z1, 0)
    z2 = h1 @ P["W2"] + P["b2"]; h2 = np.maximum(z2, 0)
    return z1, h1, z2, h2, h2 @ P["W3"] + P["b3"]


def grads(P, x, y):
    z1, h1, z2, h2, o = forward(P, x)
    o = o - o.max(1, keepdims=True); p = np.exp(o); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    d = p; d[np.arange(len(y)), y] -= 1; d /= len(y)
    g = {"W3": h2.T @ d, "b3": d.sum(0)}
    dh2 = d @ P["W3"].T * (z2 > 0); g["W2"] = h1.T @ dh2; g["b2"] = dh2.sum(0)
    dh1 = dh2 @ P["W2"].T * (z1 > 0); g["W1"] = x.T @ dh1; g["b1"] = dh1.sum(0)
    return g, loss


def erank(F):
    s = np.linalg.svd(F - F.mean(0), compute_uv=False)
    if s.sum() == 0:
        return 0.0
    q = s / s.sum(); q = q[q > 0]
    return float(np.exp(-(q * np.log(q)).sum()))


def run(method, seed):
    g = rng(100 * seed + 7)
    X = Xall[rng(seed).choice(len(Xall), NS_, replace=False)]
    xt = np.c_[X, np.ones(NS_)]                       # for layer-1 margins
    P = init(g); P0 = {k: v.copy() for k, v in P.items()}
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    it = 0
    rec = {"acc": [], "loss": [], "dead1": [], "dead2": [], "erank2": [], "wnorm": [], "gnorm0": [], "feat_energy": [],
           "revive1": [], "revive2": [], "n_dead1": [], "n_dead2": []}
    revival_pred_obs = []        # (predicted step, observed step) for L2-init layer-1 revivals
    prev_dead1 = prev_dead2 = None
    for t in range(NT):
        y = g.integers(0, 10, NS_)
        # ---- task-boundary interventions ----
        if t > 0 and method == "Shrink&Perturb":
            for k in P:
                scale = np.sqrt(2 / P[k].shape[0]) if k.startswith("W") else 0.0
                P[k] = 0.8 * P[k] + 0.1 * scale * g.standard_normal(P[k].shape)
        if t > 0 and method == "ReDo":
            z1, h1, z2, h2, _ = forward(P, X)
            for lay, hh in [("1", h1), ("2", h2)]:
                dead = hh.max(0) <= 0
                if dead.any():
                    fan = P["W" + lay].shape[0]
                    P["W" + lay][:, dead] = g.standard_normal((fan, dead.sum())) * np.sqrt(2 / fan)
                    P["b" + lay][dead] = 0
                    nxt = "W" + str(int(lay) + 1); P[nxt][dead, :] = 0
                    for st in (m, v2):
                        st["W" + lay][:, dead] = 0; st["b" + lay][dead] = 0; st[nxt][dead, :] = 0
        # ---- diagnostics at task start ----
        z1, h1, z2, h2, _ = forward(P, X)
        dead1 = h1.max(0) <= 0; dead2 = h2.max(0) <= 0
        if prev_dead1 is not None:
            rec["revive1"].append(float((prev_dead1 & ~dead1).sum())); rec["n_dead1"].append(float(prev_dead1.sum()))
            rec["revive2"].append(float((prev_dead2 & ~dead2).sum())); rec["n_dead2"].append(float(prev_dead2.sum()))
        g0, _ = grads(P, X, y)
        rec["dead1"].append(dead1.mean()); rec["dead2"].append(dead2.mean()); rec["erank2"].append(erank(h2))
        rec["wnorm"].append(float(np.sqrt(sum((v ** 2).sum() for v in P.values()))))
        rec["gnorm0"].append(float(np.sqrt(sum((v ** 2).sum() for v in g0.values()))))
        rec["feat_energy"].append(float((h2 ** 2).mean()))
        # L2-init: predict revival step of every dead layer-1 unit assuming zero data gradient
        tracking = {}
        if method == "L2-init" and dead1.any():
            for j in np.where(dead1)[0]:
                thd = np.r_[P["W1"][:, j], P["b1"][j]]; th0 = np.r_[P0["W1"][:, j], P0["b1"][j]]
                # margin_i(s) = A_i + c^s B_i ; first s with some margin_i > 0 (closed form)
                A_ = xt @ th0; B_ = xt @ (thd - th0); c_ = 1 - LR * LAM
                ok = (A_ > 0) & (B_ < 0)
                pred = int(np.ceil(np.min(np.log(A_[ok] / -B_[ok]) / np.log(c_)))) if ok.any() else None
                tracking[j] = [pred, None]
        # ---- train ----
        step_in_task = 0
        for ep in range(EPOCHS):
            perm = g.permutation(NS_)
            for b in range(0, NS_, BS):
                bi = perm[b:b + BS]
                gr, _ = grads(P, X[bi], y[bi]); it += 1; step_in_task += 1
                for k in P:
                    m[k] = 0.9 * m[k] + 0.1 * gr[k]; v2[k] = 0.999 * v2[k] + 0.001 * gr[k] ** 2
                    P[k] -= LR * (m[k] / (1 - 0.9 ** it)) / (np.sqrt(v2[k] / (1 - 0.999 ** it)) + 1e-8)
                    if method == "L2":
                        P[k] -= LR * LAM * P[k]
                    elif method == "L2-init":
                        P[k] -= LR * LAM * (P[k] - P0[k])
                if tracking:
                    marg = (X @ P["W1"] + P["b1"]).max(0)
                    for j, pv in tracking.items():
                        if pv[1] is None and marg[j] > 0:
                            pv[1] = step_in_task
        for j, pv in tracking.items():
            revival_pred_obs.append(pv)
        _, _, _, _, o = forward(P, X)
        rec["acc"].append(float((o.argmax(1) == y).mean()))
        prev_dead1, prev_dead2 = dead1, dead2
    return rec, revival_pred_obs


raw = {m: [] for m in METHODS}
revivals = []
for m in METHODS:
    for s in range(SEEDS):
        rec, rp = run(m, s)
        raw[m].append(rec)
        if m == "L2-init":
            revivals += rp
    print(m, round(time.time() - T0, 1), flush=True)

# ---------------- summaries ----------------
summ = {}
first10_all = np.mean([np.mean(r["acc"][:1]) for m in METHODS for r in raw[m]])   # fresh-network reference = task 0
for m in METHODS:
    A = np.array([r["acc"] for r in raw[m]])
    first = A[:, :10].mean(1); last = A[:, -10:].mean(1)
    e = {"acc_first10": dict(zip(["mean", "ci95"], mean_ci(first))), "acc_last10": dict(zip(["mean", "ci95"], mean_ci(last))),
         "acc_task0": dict(zip(["mean", "ci95"], mean_ci(A[:, 0]))),
         "decline_first_minus_last": dict(zip(["mean", "ci95"], mean_ci(first - last))),
         "p_decline": float(stats.ttest_rel(first, last).pvalue)}
    for k in ["dead1", "dead2", "erank2", "wnorm", "gnorm0"]:
        Z = np.array([r[k] for r in raw[m]])
        e[k + "_last"] = dict(zip(["mean", "ci95"], mean_ci(Z[:, -1])))
        e[k + "_first"] = float(Z[:, 0].mean())
    nd1 = sum(sum(r["n_dead1"]) for r in raw[m]); rv1 = sum(sum(r["revive1"]) for r in raw[m])
    nd2 = sum(sum(r["n_dead2"]) for r in raw[m]); rv2 = sum(sum(r["revive2"]) for r in raw[m])
    e["revival_rate_layer1"] = rv1 / nd1 if nd1 else None; e["dead_unit_tasks_layer1"] = nd1
    e["revival_rate_layer2"] = rv2 / nd2 if nd2 else None; e["dead_unit_tasks_layer2"] = nd2
    e["curve_mean"] = A.mean(0)
    summ[m] = e
# paired: each remedy vs baseline (last10)
base_last = np.array([np.mean(r["acc"][-10:]) for r in raw["基线"]])
for m in METHODS[1:]:
    l = np.array([np.mean(r["acc"][-10:]) for r in raw[m]])
    summ[m]["last10_vs_baseline"] = {"diff": float((l - base_last).mean()), "p_paired": float(stats.ttest_rel(l, base_last).pvalue)}

# diagnostics -> trainability (Spearman), pooled over all methods/seeds/tasks, and within baseline
diag = {}
for scope, ms_ in [("all_methods", METHODS), ("baseline_only", ["基线"])]:
    acc = np.concatenate([r["acc"] for m in ms_ for r in raw[m]])
    diag[scope] = {}
    for k in ["dead1", "dead2", "erank2", "wnorm", "gnorm0", "feat_energy"]:
        val = np.concatenate([r[k] for m in ms_ for r in raw[m]])
        rho = stats.spearmanr(val, acc)
        diag[scope][k] = {"spearman": float(rho.correlation), "p": float(rho.pvalue)}
# within-method partial view: per method pooled
diag["per_method"] = {m: {k: float(stats.spearmanr(np.concatenate([r[k] for r in raw[m]]), np.concatenate([r["acc"] for r in raw[m]])).correlation)
                          for k in ["dead2", "erank2", "wnorm", "gnorm0"]} for m in METHODS}

# revival prediction (L2-init)
rv = [(p_, o_) for p_, o_ in revivals if p_ is not None]
obs = [(p_, o_) for p_, o_ in rv if o_ is not None]
revival = {"tracked_units": len(revivals), "predicted_within_task": int(sum(1 for p_, o_ in rv if p_ <= EPOCHS * NS_ // BS)),
           "observed_within_task": int(sum(1 for p_, o_ in revivals if o_ is not None))}
if obs:
    pa = np.array([a for a, b in obs], float); oa = np.array([b for a, b in obs], float)
    revival.update({"n_pairs": len(obs), "spearman": float(stats.spearmanr(pa, oa).correlation),
                    "median_abs_err_steps": float(np.median(np.abs(pa - oa))), "median_pred": float(np.median(pa)),
                    "median_obs": float(np.median(oa)), "frac_obs_le_pred": float(np.mean(oa <= pa))})

# ---------------- figures ----------------
Pl = plt()
fig, axes = Pl.subplots(1, 4, figsize=(21, 4.3))
for m in METHODS:
    A = np.array([r["acc"] for r in raw[m]]); mu = A.mean(0); ci = 1.96 * A.std(0, ddof=1) / np.sqrt(SEEDS)
    axes[0].plot(range(NT), mu, label=m); axes[0].fill_between(range(NT), mu - ci, mu + ci, alpha=.2)
    for ax, k in zip(axes[1:], ["dead2", "erank2", "wnorm"]):
        Z = np.array([r[k] for r in raw[m]]); ax.plot(range(NT), Z.mean(0), label=m)
for ax, t in zip(axes, ["任务结束时训练准确率（可训练性）", "第二层死亡单元比例", "第二层特征有效秩", "权重总范数"]):
    ax.set_title(t); ax.set_xlabel("任务序号"); ax.grid(alpha=.3)
axes[0].axhline(0.1, color="grey", ls=":"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "frontier_plasticity.png"); Pl.close(fig)
if obs:
    fig, ax = Pl.subplots(figsize=(5, 4.2))
    ax.loglog(pa, oa, ".", alpha=.5); lim = [min(pa.min(), oa.min()), max(pa.max(), oa.max())]
    ax.plot(lim, lim, "k--", lw=1, label="y=x")
    ax.set_xlabel("理论预测复活步数（零数据梯度假设）"); ax.set_ylabel("实测复活步数"); ax.set_title("L2-init：第一层死亡单元的复活"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "frontier_revival.png"); Pl.close(fig)

runtime = time.time() - T0
res = {"config": {"seeds": SEEDS, "tasks": NT, "epochs_per_task": EPOCHS, "batch": BS, "n_images": NS_, "hidden": H, "lr_adam": LR,
                  "lambda_l2_and_l2init": LAM, "shrink_perturb": "0.8, noise 0.1*init std at task boundaries",
                  "redo": "reinit units with zero activation on all 300 inputs at task boundaries; outgoing weights zeroed"},
       "summary": summ, "diagnostics_spearman": diag, "l2init_revival_prediction": revival, "runtime_sec": runtime}
(OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
print("runtime", runtime)
