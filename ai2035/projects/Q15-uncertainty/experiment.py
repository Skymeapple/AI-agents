"""Q15 不确定性量化：校准、温度缩放、深度集成 OOD 检测、分裂共形预测及其在分布偏移下的失效.

数据：sklearn digits，像素加高斯噪声 σ0=4（0–16 灰度）使任务不过于简单；
      分布内（ID）类别 0–6，分布外（OOD）类别 7–9。
模型：逻辑回归 LR、HistGradientBoosting (GBDT)、过拟合 MLP（200-200，几乎无正则），
      5 个 MLP 的深度集成（仅随机初始化/数据打乱顺序不同，不用 bootstrap，与 Lakshminarayanan 等的建议一致）。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.datasets import load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from common.util import mean_ci, plt, results_dir, rng, save_results

warnings.filterwarnings("ignore", category=ConvergenceWarning)
T0 = time.time()
SEEDS = list(range(5))
SIGMA0, SHIFT_SIGMAS = 4.0, [6.0, 8.0]
ID_CLASSES, N_ENS, BINS = 7, 5, 15
ALPHAS = [0.05, 0.1, 0.2]
N_RESPLIT = 50  # 每个种子内对 校准/测试 集的随机重划分次数（共形覆盖率分布）


def ece(P, y, bins=BINS):
    conf, pred = P.max(1), P.argmax(1)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def nll(P, y):
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, None))).sum(1)


def softmax_T(logP, T):
    z = logP / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_T(logP, y):
    r = minimize_scalar(lambda lt: nll(softmax_T(logP, np.exp(lt)), y), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def conformal(Pcal, ycal, Ptest, ytest, alpha):
    n = len(ycal)
    s = 1 - Pcal[np.arange(n), ycal]
    qhat = np.quantile(s, min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0), method="inverted_cdf")  # 第 ceil((n+1)(1-α)) 小的分数
    sets = (1 - Ptest) <= qhat
    cov = float(sets[np.arange(len(ytest)), ytest].mean()) if ytest is not None else np.nan
    return cov, float(sets.sum(1).mean()), qhat


digits = load_digits()
rec, conf_rec, rel_store = [], [], {}
for seed in SEEDS:
    g = rng(seed)
    Xn = (digits.data + g.normal(0, SIGMA0, digits.data.shape)) / 16.0
    y_all = digits.target
    idm = y_all < ID_CLASSES
    Xid, yid = Xn[idm], y_all[idm]
    Xood = Xn[~idm]
    perm = g.permutation(len(yid))
    n = len(perm)
    tr, ca, te = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
    # shifted copies of the ID test images: stronger noise on the *clean* images
    clean_id = digits.data[idm] / 16.0
    Xshift = {s: clean_id[te] + g.normal(0, s, clean_id[te].shape) / 16.0 for s in SHIFT_SIGMAS}
    models = {
        "LR": LogisticRegression(C=1.0, max_iter=3000).fit(Xid[tr], yid[tr]),
        "GBDT": HistGradientBoostingClassifier(random_state=seed).fit(Xid[tr], yid[tr]),
    }
    mlps = [MLPClassifier((200, 200), alpha=1e-5, max_iter=1000, random_state=1000 * seed + k).fit(Xid[tr], yid[tr])
            for k in range(N_ENS)]
    probs = {}
    for name, m in models.items():
        probs[name] = {k: m.predict_proba(X) for k, X in [("cal", Xid[ca]), ("test", Xid[te]), ("ood", Xood)] +
                       [(f"shift{s}", Xshift[s]) for s in SHIFT_SIGMAS]}
    sets_ = [("cal", Xid[ca]), ("test", Xid[te]), ("ood", Xood)] + [(f"shift{s}", Xshift[s]) for s in SHIFT_SIGMAS]
    mlp_p = [{k: m.predict_proba(X) for k, X in sets_} for m in mlps]
    probs["MLP"] = mlp_p[0]
    probs["Ensemble"] = {k: np.mean([mp[k] for mp in mlp_p], 0) for k, _ in sets_}
    # temperature-scaled versions (T fitted on calibration split by NLL)
    for base in ["MLP", "GBDT", "LR", "Ensemble"]:
        lp = {k: np.log(np.clip(v, 1e-12, None)) for k, v in probs[base].items()}
        T = fit_T(lp["cal"], yid[ca])
        probs[base + "+TS"] = {k: softmax_T(v, T) for k, v in lp.items()}
        rec.append(dict(seed=seed, model=base + "+TS", metric="T", value=T))
    for name, P in probs.items():
        yt = yid[te]
        r = dict(acc=float((P["test"].argmax(1) == yt).mean()), ece=ece(P["test"], yt), nll=nll(P["test"], yt),
                 brier=float(((P["test"] - np.eye(ID_CLASSES)[yt]) ** 2).sum(1).mean()))
        ent_id, ent_ood = entropy(P["test"]), entropy(P["ood"])
        r["ood_auroc_entropy"] = float(roc_auc_score(np.r_[np.zeros(len(ent_id)), np.ones(len(ent_ood))], np.r_[ent_id, ent_ood]))
        r["ood_auroc_maxprob"] = float(roc_auc_score(np.r_[np.zeros(len(ent_id)), np.ones(len(ent_ood))],
                                                     -np.r_[P["test"].max(1), P["ood"].max(1)]))
        r["ood_mean_maxprob"] = float(P["ood"].max(1).mean())
        for s in SHIFT_SIGMAS:
            r[f"acc_shift{s}"] = float((P[f"shift{s}"].argmax(1) == yt).mean())
            r[f"ece_shift{s}"] = ece(P[f"shift{s}"], yt)
        for k, v in r.items():
            rec.append(dict(seed=seed, model=name, metric=k, value=v))
        if seed == 0:
            rel_store[name] = (P["test"].max(1), (P["test"].argmax(1) == yt))
    # mutual information (epistemic) for the ensemble
    ens_ent = {k: entropy(probs["Ensemble"][k]) for k in ["test", "ood"]}
    exp_ent = {k: np.mean([entropy(mp[k]) for mp in mlp_p], 0) for k in ["test", "ood"]}
    mi_id, mi_ood = ens_ent["test"] - exp_ent["test"], ens_ent["ood"] - exp_ent["ood"]
    rec.append(dict(seed=seed, model="Ensemble", metric="ood_auroc_mutualinfo",
                    value=float(roc_auc_score(np.r_[np.zeros(len(mi_id)), np.ones(len(mi_ood))], np.r_[mi_id, mi_ood]))))
    # ---------------- split conformal prediction ----------------
    for name in ["LR", "MLP+TS", "Ensemble+TS", "GBDT"]:
        Pc, Pt = probs[name]["cal"], probs[name]["test"]
        Pall, yall_ = np.vstack([Pc, Pt]), np.r_[yid[ca], yid[te]]
        ncal = len(ca)
        for a in ALPHAS:
            # (i) re-split cal/test many times to get the coverage distribution (exchangeable case)
            covs, sizes = [], []
            for rr in range(N_RESPLIT):
                pp = rng(10_000 * seed + rr).permutation(len(yall_))
                c_, t_ = pp[:ncal], pp[ncal:]
                cv, sz, _ = conformal(Pall[c_], yall_[c_], Pall[t_], yall_[t_], a)
                covs.append(cv)
                sizes.append(sz)
            row = dict(seed=seed, model=name, alpha=a, cov_id=float(np.mean(covs)), size_id=float(np.mean(sizes)),
                       cov_id_sd=float(np.std(covs)))
            # (ii) shift: calibration on original split, test on shifted images
            for s in SHIFT_SIGMAS:
                cv, sz, _ = conformal(Pc, yid[ca], probs[name][f"shift{s}"], yid[te], a)
                row[f"cov_shift{s}"], row[f"size_shift{s}"] = cv, sz
            # (iii) label shift: test resampled so that the 3 classes with lowest cal accuracy make up 80%
            cls_acc = np.array([(Pc.argmax(1) == yid[ca])[yid[ca] == c].mean() for c in range(ID_CLASSES)])
            hard = np.argsort(cls_acc)[:3]
            yt = yid[te]
            w_ = np.where(np.isin(yt, hard), 0.8 / np.isin(yt, hard).sum(), 0.2 / (~np.isin(yt, hard)).sum())
            idx = rng(20_000 + seed).choice(len(yt), size=2000, p=w_ / w_.sum())
            cv, sz, _ = conformal(Pc, yid[ca], Pt[idx], yt[idx], a)
            row["cov_labelshift"], row["size_labelshift"] = cv, sz
            # (iv) OOD inputs: the true label is never in the set; report set size & fraction of empty sets
            _, sz_ood, qh = conformal(Pc, yid[ca], probs[name]["ood"], None, a)
            row["size_ood"] = sz_ood
            row["empty_ood"] = float((((1 - probs[name]["ood"]) <= qh).sum(1) == 0).mean())
            row["n_cal"] = ncal
            conf_rec.append(row)
    print(f"seed {seed} done {time.time()-T0:.0f}s", flush=True)


def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


MODELS = ["LR", "GBDT", "MLP", "Ensemble", "LR+TS", "GBDT+TS", "MLP+TS", "Ensemble+TS"]
METRICS = ["acc", "ece", "nll", "brier", "ood_auroc_entropy", "ood_auroc_maxprob", "ood_mean_maxprob"] + \
          [f"{k}_shift{s}" for s in SHIFT_SIGMAS for k in ["acc", "ece"]]
out = {"config": dict(seeds=SEEDS, sigma0=SIGMA0, shift_sigmas=SHIFT_SIGMAS, id_classes=list(range(ID_CLASSES)),
                      ood_classes=[7, 8, 9], n_ens=N_ENS, ece_bins=BINS, alphas=ALPHAS, n_resplit=N_RESPLIT,
                      split="ID: 50% train / 25% calibration(+TS) / 25% test")}
out["calibration"] = {m: {k: ag(rec, "value", model=m, metric=k) for k in METRICS} for m in MODELS}
out["temperature"] = {m: ag(rec, "value", model=m, metric="T") for m in ["LR+TS", "GBDT+TS", "MLP+TS", "Ensemble+TS"]}
out["ensemble_mutualinfo_auroc"] = ag(rec, "value", model="Ensemble", metric="ood_auroc_mutualinfo")


def vals(m, k):
    return np.array([r["value"] for r in rec if r["model"] == m and r["metric"] == k])


T = {}
for a, b, k in [("MLP", "LR", "ece"), ("MLP", "MLP+TS", "ece"), ("MLP", "MLP+TS", "acc"), ("MLP", "MLP+TS", "nll"),
                ("Ensemble", "MLP", "ood_auroc_entropy"), ("Ensemble", "MLP", "nll"), ("Ensemble", "MLP", "ece"),
                ("Ensemble", "MLP+TS", "ood_auroc_entropy"), ("MLP+TS", "MLP", "ood_auroc_entropy"),
                ("Ensemble", "LR", "ood_auroc_entropy"), ("GBDT", "GBDT+TS", "ece")]:
    x, y_ = vals(a, k), vals(b, k)
    T[f"{k}|{a}_minus_{b}"] = {"diff": float(np.mean(x - y_)), "paired_t_p": float(stats.ttest_rel(x, y_).pvalue)}
T["ece|MLP+TS_rel_reduction"] = {"mean": float(np.mean(1 - vals("MLP+TS", "ece") / vals("MLP", "ece")))}
out["tests"] = T
out["conformal"] = {}
for m in ["LR", "GBDT", "MLP+TS", "Ensemble+TS"]:
    for a in ALPHAS:
        ks = ["cov_id", "cov_id_sd", "size_id", "cov_labelshift", "size_labelshift", "size_ood", "empty_ood"] + \
             [f"{p}_shift{s}" for s in SHIFT_SIGMAS for p in ["cov", "size"]]
        out["conformal"][f"{m}|alpha={a}"] = {k: ag(conf_rec, k, model=m, alpha=a) for k in ks}
        ncal = conf_rec[0]["n_cal"]
        out["conformal"][f"{m}|alpha={a}"]["theory_upper"] = 1 - a + 1 / (ncal + 1)
        # 可交换且分数无并列时的精确期望覆盖率 = ceil((n+1)(1-α))/(n+1)
        out["conformal"][f"{m}|alpha={a}"]["theory_expected"] = float(np.ceil((ncal + 1) * (1 - a)) / (ncal + 1))
out["n_cal"] = conf_rec[0]["n_cal"]
out["runtime_sec"] = time.time() - T0

# ---------------------------------- figures ----------------------------------
P_ = plt()
R = results_dir(__file__)
fig, axes = P_.subplots(1, 4, figsize=(15, 3.8))
for ax, name in zip(axes, ["LR", "GBDT", "MLP", "MLP+TS"]):
    conf, corr = rel_store[name]
    edges = np.linspace(0, 1, 11)
    idx = np.minimum((conf * 10).astype(int), 9)
    accs = [corr[idx == b].mean() if (idx == b).sum() > 0 else np.nan for b in range(10)]
    cnts = [(idx == b).sum() for b in range(10)]
    ax.bar(edges[:-1] + 0.05, accs, width=0.1, edgecolor="k", alpha=0.7, label="准确率")
    ax.plot([0, 1], [0, 1], "r--", label="完美校准")
    e_ = out["calibration"][name]["ece"]["mean"]
    ax.set_title(f"{name}  ECE={e_:.3f}")
    ax.set_xlabel("置信度")
    for b in range(10):
        if cnts[b] > 0:
            ax.text(edges[b] + 0.05, 0.02, str(cnts[b]), ha="center", fontsize=6, rotation=90)
axes[0].set_ylabel("该置信度区间内准确率")
axes[0].legend(fontsize=7)
fig.suptitle("可靠性图（种子 0，ID 测试集；柱内数字为样本数）")
fig.tight_layout()
fig.savefig(R / "reliability.png")

fig, axes = P_.subplots(1, 2, figsize=(11, 4))
cms = ["LR", "GBDT", "MLP+TS", "Ensemble+TS"]
w = 0.2
for i, m in enumerate(cms):
    xs = np.arange(len(ALPHAS)) + (i - 1.5) * w
    axes[0].bar(xs, [out["conformal"][f"{m}|alpha={a}"]["cov_id"]["mean"] for a in ALPHAS], w, label=f"{m} ID", color=f"C{i}")
    axes[0].scatter(xs, [out["conformal"][f"{m}|alpha={a}"]["cov_shift8.0"]["mean"] for a in ALPHAS], color="k", marker="x", zorder=3)
    axes[0].scatter(xs, [out["conformal"][f"{m}|alpha={a}"]["cov_labelshift"]["mean"] for a in ALPHAS], color="r", marker="v", zorder=3)
for j, a in enumerate(ALPHAS):
    axes[0].hlines(1 - a, j - 0.45, j + 0.45, color="k", ls="--")
axes[0].set_xticks(range(len(ALPHAS)), [f"α={a}" for a in ALPHAS])
axes[0].set_ylim(0.35, 1.0)
axes[0].set_ylabel("经验覆盖率")
axes[0].set_title("分裂共形：柱=ID，×=噪声偏移 σ=8，▼=标签偏移；虚线=1-α")
axes[0].legend(fontsize=7)
for i, m in enumerate(["LR", "GBDT", "MLP", "Ensemble", "MLP+TS"]):
    axes[1].bar(i, out["calibration"][m]["ood_auroc_entropy"]["mean"], yerr=out["calibration"][m]["ood_auroc_entropy"]["ci95"],
                color=f"C{i}", capsize=3)
axes[1].bar(5, out["ensemble_mutualinfo_auroc"]["mean"], yerr=out["ensemble_mutualinfo_auroc"]["ci95"], color="C5", capsize=3)
axes[1].set_xticks(range(6), ["LR", "GBDT", "MLP", "集成(熵)", "MLP+TS", "集成(互信息)"], fontsize=8)
axes[1].set_ylim(0.5, 1.0)
axes[1].set_ylabel("AUROC")
axes[1].set_title("OOD 检测（ID=0–6，OOD=7–9）")
fig.tight_layout()
fig.savefig(R / "conformal_ood.png")

save_results(__file__, out)
print(f"done {time.time()-T0:.0f}s")
