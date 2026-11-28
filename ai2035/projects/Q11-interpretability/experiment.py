"""Q11 可解释性：已知真值下归因方法的忠实度检验.

合成表格数据：d=12 个特征，只有 S={x0..x4} 进入真实 logit：
    logit = 1.5 x0 - 1.0 x1 + 0.7 x2 + 1.5 x3*x4        (x3, x4 只以交互项出现)
干扰特征：x5 ~ rho*x0 + sqrt(1-rho^2) e,  x6 ~ rho*x1 + ...（与真特征相关但本身无因果作用），
x7..x11 为独立噪声。
模型：逻辑回归（LR）、梯度提升树（GBDT, HistGradientBoosting）、MLP。
归因：梯度×输入（数值梯度）、遮挡（置为均值）、采样 Shapley（干预式，自实现）、
置换重要性（log-loss 增量）、系数幅度（仅 LR）、随机归因（对照）。
指标：precision@k、与"真值 Shapley"（对真实 logit 计算）的 Spearman、相关代理特征的
"泄漏率"、方法间一致性 Kendall tau；以及 Adebayo et al. 2018 式的健全性检验
（权重随机化 / 标签随机化）。
"""
from __future__ import annotations

import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # single-core budget
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from common.util import mean_ci, plt, results_dir, rng, save_results

T0 = time.time()
D, K = 12, 5
TRUE = np.array([0, 1, 2, 3, 4])
RHOS = [0.0, 0.5, 0.8, 0.9, 0.99]
SEEDS = list(range(5))
N_TRAIN, N_TEST, N_EXPLAIN, M_PERM = 3000, 1000, 150, 48
METHODS = ["grad_x_input", "occlusion", "shapley", "permutation", "coef", "random"]
MODELS = ["LR", "GBDT", "MLP"]


def make_data(g, n, rho):
    X = g.standard_normal((n, D))
    X[:, 5] = rho * X[:, 0] + np.sqrt(1 - rho**2) * g.standard_normal(n)
    X[:, 6] = rho * X[:, 1] + np.sqrt(1 - rho**2) * g.standard_normal(n)
    y = (g.random(n) < 1 / (1 + np.exp(-true_logit(X)))).astype(int)
    return X, y


def true_logit(X):
    return 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * X[:, 2] + 1.5 * X[:, 3] * X[:, 4]


def logit_fn(model):
    def f(X):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    return f


def fit_models(X, y, seed):
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed).fit(X, y)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=400,
                        early_stopping=True, random_state=seed).fit(X, y)
    return {"LR": lr, "GBDT": gb, "MLP": mlp}


# ---------------- attribution methods (local: n_explain x D) ----------------
def grad_x_input(f, X, h=1e-2):
    G = np.zeros_like(X)
    for j in range(D):
        Xp, Xm = X.copy(), X.copy()
        Xp[:, j] += h
        Xm[:, j] -= h
        G[:, j] = (f(Xp) - f(Xm)) / (2 * h)
    return G * X  # baseline = 0 = population mean


def occlusion(f, X, mu):
    base = f(X)
    A = np.zeros_like(X)
    for j in range(D):
        Xo = X.copy()
        Xo[:, j] = mu[j]
        A[:, j] = base - f(Xo)
    return A


def shapley_sampling(f, X, background, g, m=M_PERM):
    """Interventional Shapley via permutation sampling (Strumbelj & Kononenko 2014)."""
    n = X.shape[0]
    perms = np.argsort(g.random((n, m, D)), axis=2)          # n,m,D
    Z = background[g.integers(0, len(background), (n, m))]   # n,m,D
    rows = np.repeat(Z[:, :, None, :], D + 1, axis=2)        # n,m,D+1,D
    for s in range(1, D + 1):
        idx = perms[:, :, s - 1]                              # feature added at step s
        mask = np.zeros((n, m, D), bool)
        np.put_along_axis(mask, perms[:, :, :s], True, axis=2)
        rows[:, :, s, :] = np.where(mask, X[:, None, :], Z)
    vals = f(rows.reshape(-1, D)).reshape(n, m, D + 1)
    deltas = np.diff(vals, axis=2)                            # n,m,D  marginal contribution at step s
    phi = np.zeros((n, m, D))
    np.put_along_axis(phi, perms, deltas, axis=2)
    return phi.mean(axis=1)


def permutation_importance(model, X, y, g, reps=5):
    def ll(Xe):
        p = np.clip(model.predict_proba(Xe)[:, 1], 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    base = ll(X)
    imp = np.zeros(D)
    for j in range(D):
        for _ in range(reps):
            Xp = X.copy()
            Xp[:, j] = g.permutation(Xp[:, j])
            imp[j] += (ll(Xp) - base) / reps
    return imp


def global_imp(local):
    return np.abs(local).mean(axis=0)


def prec_at_k(imp):
    top = np.argsort(-imp)[:K]
    return len(set(top) & set(TRUE)) / K


def all_attributions(name, model, Xtr, Xte, yte, Xe, g):
    f = logit_fn(model)
    mu = Xtr.mean(axis=0)
    loc = {
        "grad_x_input": grad_x_input(f, Xe),
        "occlusion": occlusion(f, Xe, mu),
        "shapley": shapley_sampling(f, Xe, Xtr[:500], g),
    }
    glob = {k: global_imp(v) for k, v in loc.items()}
    glob["permutation"] = permutation_importance(model, Xte, yte, g)
    if name == "LR":
        glob["coef"] = np.abs(model.coef_[0]) * Xtr.std(axis=0)
    glob["random"] = g.random(D)
    return loc, glob


def randomize_weights(model, name, g):
    import copy
    m = copy.deepcopy(model)
    if name == "LR":
        m.coef_ = g.standard_normal(m.coef_.shape) * np.std(model.coef_)
        m.intercept_ = np.zeros_like(m.intercept_)
    elif name == "MLP":
        m.coefs_ = [g.standard_normal(W.shape) * np.sqrt(2.0 / W.shape[0]) for W in model.coefs_]
        m.intercepts_ = [np.zeros_like(b) for b in model.intercepts_]
    return m


def mean_local_spearman(A, B):
    r = []
    for a, b in zip(np.abs(A), np.abs(B)):
        c = stats.spearmanr(a, b).statistic
        r.append(0.0 if np.isnan(c) else c)
    return float(np.mean(r))


# ------------------------------------------------------------------------------
records = []      # per (seed, rho, model, method)
agree = []        # method agreement
sanity = []       # sanity checks
acc_rec = []
for seed in SEEDS:
    for rho in RHOS:
        g = rng(1000 * seed + int(rho * 100))
        Xtr, ytr = make_data(g, N_TRAIN, rho)
        Xte, yte = make_data(g, N_TEST, rho)
        Xe = Xte[:N_EXPLAIN]
        models = fit_models(Xtr, ytr, seed)
        oracle = global_imp(shapley_sampling(true_logit, Xe, Xtr[:500], g, m=M_PERM))
        bayes_acc = np.mean((true_logit(Xte) > 0) == yte)
        for name, model in models.items():
            acc_rec.append(dict(seed=seed, rho=rho, model=name, acc=float(model.score(Xte, yte)), bayes=float(bayes_acc)))
            loc, glob = all_attributions(name, model, Xtr, Xte, yte, Xe, g)
            for meth, imp in glob.items():
                records.append(dict(
                    seed=seed, rho=rho, model=name, method=meth,
                    prec=prec_at_k(imp),
                    spearman=float(stats.spearmanr(imp, oracle).statistic),
                    leak0=float(imp[5] / (imp[0] + imp[5] + 1e-12)),
                    leak1=float(imp[6] / (imp[1] + imp[6] + 1e-12)),
                    proxy0=float(imp[5] / (imp[7:].mean() + 1e-12)),
                    proxy1=float(imp[6] / (imp[7:].mean() + 1e-12)),
                    inter=float(np.isin([3, 4], np.argsort(-imp)[:K]).mean()),
                    imp=imp / (imp.sum() + 1e-12)))
            ms = [m for m in ["grad_x_input", "occlusion", "shapley", "permutation"]]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    agree.append(dict(seed=seed, rho=rho, model=name, pair=f"{ms[i]}~{ms[j]}",
                                      tau=float(stats.kendalltau(glob[ms[i]], glob[ms[j]]).statistic)))
            if rho == 0.0:
                # sanity checks (Adebayo et al. 2018)
                checks = {}
                ytr_perm = g.permutation(ytr)
                mrand_lab = fit_models(Xtr, ytr_perm, seed)[name]
                checks["label_rand"] = mrand_lab
                if name in ("LR", "MLP"):
                    checks["weight_rand"] = randomize_weights(model, name, g)
                for ck, mr in checks.items():
                    loc_r, glob_r = all_attributions(name, mr, Xtr, Xte, yte, Xe, g)
                    for meth in ["grad_x_input", "occlusion", "shapley", "permutation"]:
                        row = dict(seed=seed, model=name, check=ck, method=meth,
                                   global_sp=float(np.nan_to_num(stats.spearmanr(glob[meth], glob_r[meth]).statistic)))
                        if meth in loc:
                            row["local_sp"] = mean_local_spearman(loc[meth], loc_r[meth])
                            row["local_sp_vs_absx"] = mean_local_spearman(loc_r[meth], Xe)
                        sanity.append(row)
    print(f"seed {seed} done  {time.time()-T0:.0f}s", flush=True)


# ------------------------------- aggregation ---------------------------------
def agg(rows, key, **flt):
    v = [r[key] for r in rows if all(r.get(a) == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


out = {"config": dict(D=D, K=K, true=TRUE.tolist(), rhos=RHOS, seeds=SEEDS, n_train=N_TRAIN, n_test=N_TEST,
                      n_explain=N_EXPLAIN, shapley_perms=M_PERM,
                      logit="1.5x0 - 1.0x1 + 0.7x2 + 1.5x3*x4; x5~x0, x6~x1 (corr rho)")}
out["accuracy"] = {f"{m}|rho={r}": agg(acc_rec, "acc", model=m, rho=r) for m in MODELS for r in RHOS}
out["bayes_acc"] = {f"rho={r}": agg(acc_rec, "bayes", model="LR", rho=r) for r in RHOS}
out["faithfulness"] = {}
for r in RHOS:
    for m in MODELS:
        for meth in METHODS:
            if meth == "coef" and m != "LR":
                continue
            key = f"{m}|{meth}|rho={r}"
            out["faithfulness"][key] = {k: agg(records, k, model=m, method=meth, rho=r)
                                        for k in ["prec", "spearman", "leak0", "leak1", "proxy0", "proxy1", "inter"]}
            out["faithfulness"][key]["imp_mean"] = np.mean(
                [x["imp"] for x in records if x["model"] == m and x["method"] == meth and x["rho"] == r], axis=0).round(4)
# paired tests: Shapley vs grad_x_input precision on GBDT / MLP at rho=0
tests = {}
for m in MODELS:
    for r in [0.0, 0.9]:
        a = [x["spearman"] for x in records if x["model"] == m and x["method"] == "shapley" and x["rho"] == r]
        b = [x["spearman"] for x in records if x["model"] == m and x["method"] == "grad_x_input" and x["rho"] == r]
        tests[f"{m}|shapley_vs_gradxinput_spearman|rho={r}"] = {
            "diff_mean": float(np.mean(np.array(a) - np.array(b))),
            "paired_t_p": float(stats.ttest_rel(a, b).pvalue) if np.std(np.array(a) - np.array(b)) > 0 else 1.0}
# leakage: compare models at rho=0.9 using shapley (Welch)
for meth in ["shapley", "permutation"]:
    for m1, m2 in [("LR", "GBDT"), ("MLP", "GBDT"), ("LR", "MLP")]:
        a = [x["leak0"] for x in records if x["model"] == m1 and x["method"] == meth and x["rho"] == 0.9]
        b = [x["leak0"] for x in records if x["model"] == m2 and x["method"] == meth and x["rho"] == 0.9]
        tests[f"leak0|{meth}|{m1}_vs_{m2}|rho=0.9"] = {"diff_mean": float(np.mean(a) - np.mean(b)),
                                                      "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue)}
out["tests"] = tests
out["agreement"] = {}
pairs = sorted(set(a["pair"] for a in agree))
for m in MODELS:
    for r in [0.0, 0.9, 0.99]:
        out["agreement"][f"{m}|rho={r}"] = {p: agg(agree, "tau", model=m, rho=r, pair=p) for p in pairs}
# cross-model agreement of shapley importances (same data)
cross = {}
for r in RHOS:
    taus = []
    for s in SEEDS:
        imps = {x["model"]: x["imp"] for x in records if x["seed"] == s and x["rho"] == r and x["method"] == "shapley"}
        taus.append(np.mean([stats.kendalltau(imps[a], imps[b]).statistic for a, b in [("LR", "GBDT"), ("LR", "MLP"), ("GBDT", "MLP")]]))
    cross[f"rho={r}"] = dict(zip(["mean", "ci95"], mean_ci(taus)))
out["cross_model_tau_shapley"] = cross
out["sanity"] = {}
for m in MODELS:
    for ck in ["label_rand", "weight_rand"]:
        for meth in ["grad_x_input", "occlusion", "shapley", "permutation"]:
            rows = [x for x in sanity if x["model"] == m and x["check"] == ck and x["method"] == meth]
            if not rows:
                continue
            e = {"global_sp": agg(rows, "global_sp")}
            if "local_sp" in rows[0]:
                e["local_sp"] = agg(rows, "local_sp")
                e["local_sp_vs_absx"] = agg(rows, "local_sp_vs_absx")
            out["sanity"][f"{m}|{ck}|{meth}"] = e
for m in ["LR", "MLP", "GBDT"]:
    for ck in ["label_rand", "weight_rand"]:
        a = [x["local_sp"] for x in sanity if x["model"] == m and x["check"] == ck and x["method"] == "grad_x_input"]
        b = [x["local_sp"] for x in sanity if x["model"] == m and x["check"] == ck and x["method"] == "shapley"]
        if a:
            out["tests"][f"sanity_local|{m}|{ck}|gradxinput_minus_shapley"] = {
                "diff_mean": float(np.mean(np.array(a) - np.array(b))),
                "paired_t_p": float(stats.ttest_rel(a, b).pvalue)}
# reference: local spearman between original attributions and |x| (input-only "edge detector" analogue)
out["runtime_sec"] = time.time() - T0

# ---------------------------------- figures ----------------------------------
P = plt()
R = results_dir(__file__)
fig, axes = P.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, m in zip(axes, MODELS):
    for meth in ["grad_x_input", "occlusion", "shapley", "permutation", "coef", "random"]:
        if meth == "coef" and m != "LR":
            continue
        ys = [out["faithfulness"][f"{m}|{meth}|rho={r}"]["prec"]["mean"] for r in RHOS]
        es = [out["faithfulness"][f"{m}|{meth}|rho={r}"]["prec"]["ci95"] for r in RHOS]
        ax.errorbar(range(len(RHOS)), ys, yerr=es, marker="o", capsize=3, label=meth,
                    color=f"C{METHODS.index(meth)}")
    ax.set_xticks(range(len(RHOS)), [str(r) for r in RHOS])
    ax.set_title(f"{m}")
    ax.set_xlabel("干扰特征与真特征相关系数 ρ")
axes[0].set_ylabel("precision@5")
axes[0].legend(fontsize=8)
fig.suptitle("归因方法恢复真实特征的 precision@5（均值±95%CI，5 种子）")
fig.tight_layout()
fig.savefig(R / "precision_vs_rho.png")

fig, ax = P.subplots(figsize=(6.5, 4))
for m, c in zip(MODELS, ["C0", "C1", "C2"]):
    for meth, ls in [("shapley", "-"), ("permutation", "--")]:
        ys = [out["faithfulness"][f"{m}|{meth}|rho={r}"]["leak0"]["mean"] for r in RHOS]
        es = [out["faithfulness"][f"{m}|{meth}|rho={r}"]["leak0"]["ci95"] for r in RHOS]
        ax.errorbar(RHOS, ys, yerr=es, color=c, ls=ls, marker="o", capsize=3, label=f"{m}-{meth}")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("ρ(x0, x5)")
ax.set_ylabel("泄漏率 imp(x5)/(imp(x0)+imp(x5))")
ax.set_title("相关干扰特征 x5 分走的归因份额")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(R / "leakage_vs_rho.png")

fig, axes = P.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, m in zip(axes, MODELS):
    meths = [x for x in METHODS if x != "random" and (x != "coef" or m == "LR")]
    M = np.array([out["faithfulness"][f"{m}|{x}|rho=0.9"]["imp_mean"] for x in meths])
    ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(meths)), meths)
    ax.set_xticks(range(D), [f"x{j}" for j in range(D)], fontsize=7)
    ax.set_title(f"{m}（ρ=0.9，归一化重要性）")
fig.tight_layout()
fig.savefig(R / "importance_heatmap_rho09.png")

save_results(__file__, out)
print(f"done in {time.time()-T0:.0f}s")
