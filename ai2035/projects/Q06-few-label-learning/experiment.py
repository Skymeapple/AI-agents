"""Q06 小样本与缺标签学习：无标签数据何时有用、何时有害。

实验 A（digits，归纳式）：每类标注数 k∈{1,2,5,10,20}，比较
  监督基线 LR / 1-NN；图半监督 LabelSpreading；自训练 SelfTraining(LR)；
  “预训练表示”代理：在全部无标签数据上拟合 PCA(30) 后线性探针。
实验 B（two-moons，直推式）：同样方法，表示代理改为谱嵌入（拉普拉斯特征映射）。
实验 C（反例，预注册）：
  C1 聚类假设被违反——两个分离的高斯团沿 x1 排列，但标签由 x2 符号决定（决策边界穿过高密度区）。
  C2 表示先验被违反——判别信息位于低方差方向，高方差方向为噪声，PCA 表示丢弃判别信息。
所有参数在运行前固定；10 个种子；对 SSL−监督 的差值做配对 t 检验。
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, save_results, mean_ci, plt, results_dir

import numpy as np
from scipy import stats
from sklearn.datasets import load_digits, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

T0 = time.time()
SEEDS = 10
BUDGETS = [1, 2, 5, 10, 20]
OUT = results_dir(__file__)


def sample_labeled(y, k, g):
    idx = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        idx.extend(g.choice(cand, size=k, replace=False))
    return np.array(idx)


def lr():
    return LogisticRegression(C=1.0, max_iter=2000)


def run_methods(Xp, yp, Xt, yt, k, g, rep="pca", transductive=False, n_pca=30):
    """Xp: pool (labeled+unlabeled); Xt,yt: evaluation set.
    If transductive, Xt is a subset of the pool (evaluate on unlabeled pool points)."""
    L = sample_labeled(yp, k, g)
    mask = np.zeros(len(yp), bool); mask[L] = True
    if transductive:
        Xt, yt = Xp[~mask], yp[~mask]
    res = {}
    res["LR(监督)"] = lr().fit(Xp[L], yp[L]).score(Xt, yt)
    res["1-NN(监督)"] = KNeighborsClassifier(1).fit(Xp[L], yp[L]).score(Xt, yt)
    ysemi = yp.copy(); ysemi[~mask] = -1
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.99, max_iter=2000).fit(Xp, ysemi)
    res["LabelSpreading"] = ls.score(Xt, yt)
    # sklearn default alpha=0.2: propagation decays fast and the tol-based stop freezes far nodes
    ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=2000).fit(Xp, ysemi)
    res["LabelSpreading(α=0.2默认)"] = ls.score(Xt, yt)
    st = SelfTrainingClassifier(lr(), threshold=0.9, max_iter=10).fit(Xp, ysemi)
    res["自训练(阈值0.9)"] = st.score(Xt, yt)
    st = SelfTrainingClassifier(lr(), criterion="k_best", k_best=100, max_iter=10).fit(Xp, ysemi)
    res["自训练(每轮top100)"] = st.score(Xt, yt)
    if rep == "pca":
        P = PCA(n_components=n_pca, random_state=0).fit(Xp)
        Zp, Zt = P.transform(Xp), P.transform(Xt)
        res["PCA表示+线性探针"] = lr().fit(Zp[L], yp[L]).score(Zt, yt)
        # non-linear unsupervised features: k-means "triangle" encoding (Coates et al. 2011)
        km = MiniBatchKMeans(n_clusters=100, n_init=3, random_state=int(g.integers(1e6))).fit(Xp)
        def tri(Z):
            D = km.transform(Z); return np.maximum(0, D.mean(1, keepdims=True) - D)
        Fp, Ft = tri(Xp), tri(Xt); sc = StandardScaler().fit(Fp)
        res["k-means特征+线性探针"] = lr().fit(sc.transform(Fp)[L], yp[L]).score(sc.transform(Ft), yt)
    elif rep == "spectral":
        Z = SpectralEmbedding(n_components=2, n_neighbors=10, random_state=0).fit_transform(Xp)
        Z = StandardScaler().fit_transform(Z)
        res["谱嵌入表示+线性探针"] = lr().fit(Z[L], yp[L]).score(Z[~mask], yp[~mask])
    return res


def summarize(raw):
    """raw[budget][method] -> list over seeds"""
    out = {}
    for b, d in raw.items():
        out[b] = {}
        base = np.array(d["LR(监督)"])
        for m, v in d.items():
            v = np.array(v); mu, ci = mean_ci(v)
            ent = {"mean": mu, "ci95": ci}
            if m != "LR(监督)":
                diff = v - base
                dmu, dci = mean_ci(diff)
                p = stats.ttest_rel(v, base).pvalue if np.any(diff != 0) else 1.0
                ent.update({"diff_vs_LR": dmu, "diff_ci95": dci, "p_paired": float(p)})
            out[b][m] = ent
    return out


# ---------------- A: digits ----------------
X, y = load_digits(return_X_y=True)
X = X / 16.0
rawA = {k: {} for k in BUDGETS}
for s in range(SEEDS):
    g = rng(s)
    perm = g.permutation(len(y))
    te, po = perm[:500], perm[500:]
    for k in BUDGETS:
        r = run_methods(X[po], y[po], X[te], y[te], k, g, rep="pca")
        for m, v in r.items():
            rawA[k].setdefault(m, []).append(v)
A = summarize(rawA)
print("A done", time.time() - T0)

# ---------------- B: two moons (transductive) ----------------
rawB = {k: {} for k in BUDGETS}
for s in range(SEEDS):
    g = rng(100 + s)
    Xm, ym = make_moons(n_samples=1000, noise=0.1, random_state=int(g.integers(1e9)))
    for k in BUDGETS:
        r = run_methods(Xm, ym, None, None, k, g, rep="spectral", transductive=True)
        for m, v in r.items():
            rawB[k].setdefault(m, []).append(v)
B = summarize(rawB)
print("B done", time.time() - T0)


# ---------------- C1: cluster assumption violated ----------------
def make_c1(n, g):
    # two well-separated blobs along x1 (centres ±3), label = sign(x2): boundary cuts through density
    c = g.integers(0, 2, n) * 2 - 1
    X = np.c_[3.0 * c + 0.6 * g.standard_normal(n), g.standard_normal(n)]
    y = (X[:, 1] > 0).astype(int)
    return X, y

rawC1 = {k: {} for k in BUDGETS}
for s in range(SEEDS):
    g = rng(200 + s)
    Xc, yc = make_c1(1500, g)
    for k in BUDGETS:
        r = run_methods(Xc[500:], yc[500:], Xc[:500], yc[:500], k, g, rep="none")
        for m, v in r.items():
            rawC1[k].setdefault(m, []).append(v)
C1 = summarize(rawC1)


# ---------------- C2: representation prior violated ----------------
def make_c2(n, g, d=20):
    # dims 0..d-2: high-variance noise (std 3); last dim: low-variance, carries the label
    y = g.integers(0, 2, n)
    X = 3.0 * g.standard_normal((n, d))
    X[:, -1] = (2 * y - 1) * 1.0 + 0.3 * g.standard_normal(n)
    # a fixed random rotation would hide axis alignment but keep variance structure
    return X, y

rawC2 = {k: {} for k in BUDGETS}
for s in range(SEEDS):
    g = rng(300 + s)
    Xc, yc = make_c2(1500, g)
    for k in BUDGETS:
        r = run_methods(Xc[500:], yc[500:], Xc[:500], yc[:500], k, g, rep="pca", n_pca=5)
        for m, v in r.items():
            rawC2[k].setdefault(m, []).append(v)
C2 = summarize(rawC2)
print("C done", time.time() - T0)

# ---------------- figures ----------------
P = plt()
fig, axes = P.subplots(1, 4, figsize=(19, 4.2))
for ax, (name, S) in zip(axes, [("A: digits (500 测试)", A), ("B: two-moons (直推)", B),
                                 ("C1: 违反聚类假设", C1), ("C2: 判别信息在低方差方向", C2)]):
    methods = list(S[BUDGETS[0]].keys())
    for m in methods:
        mu = np.array([S[k][m]["mean"] for k in BUDGETS]); ci = np.array([S[k][m]["ci95"] for k in BUDGETS])
        ax.errorbar(BUDGETS, mu, yerr=ci, marker="o", capsize=3, label=m)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
    ax.set_xlabel("每类标注样本数 k"); ax.set_ylabel("测试准确率"); ax.set_title(name); ax.grid(alpha=.3)
    ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "learning_curves.png"); P.close(fig)

# gain plot
fig, ax = P.subplots(figsize=(6.5, 4.2))
for name, S, m in [("digits LabelSpreading", A, "LabelSpreading"), ("digits PCA探针", A, "PCA表示+线性探针"),
                   ("digits k-means探针", A, "k-means特征+线性探针"), ("digits 自训练(阈值0.9)", A, "自训练(阈值0.9)"), ("C1 自训练(top100)", C1, "自训练(每轮top100)"), ("C1 LabelSpreading", C1, "LabelSpreading"),
                   ("C2 PCA探针", C2, "PCA表示+线性探针")]:
    mu = [S[k][m]["diff_vs_LR"] for k in BUDGETS]; ci = [S[k][m]["diff_ci95"] for k in BUDGETS]
    ax.errorbar(BUDGETS, mu, yerr=ci, marker="o", capsize=3, label=name)
ax.axhline(0, color="k", lw=.8); ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
ax.set_xlabel("每类标注样本数 k"); ax.set_ylabel("相对监督 LR 的准确率增益"); ax.set_title("无标签数据的增益（正=有益，负=有害）")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "ssl_gain.png"); P.close(fig)

# illustrate C1
g = rng(999); Xc, yc = make_c1(600, g)
fig, ax = P.subplots(figsize=(5, 4))
ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="coolwarm", s=8)
ax.set_title("C1 数据：簇沿 x1 分离，标签由 x2 决定"); fig.tight_layout(); fig.savefig(OUT / "c1_data.png"); P.close(fig)

runtime = time.time() - T0
save_results(__file__, {
    "config": {"seeds": SEEDS, "budgets_per_class": BUDGETS, "digits_test": 500,
               "LabelSpreading": "knn k=7 alpha=0.99 (Zhou et al. 2004) max_iter=2000; variant alpha=0.2 (sklearn default)", "SelfTraining": "LR; (a) threshold 0.9 (b) k_best=100 per iter; max_iter 10", "kmeans_features": "MiniBatchKMeans 100, triangle encoding, standardized",
               "PCA": 30, "moons": "n=1000 noise=0.1, transductive, spectral embedding 2D",
               "C1": "blobs at x1=±3 (sd .6), label=sign(x2), n_pool=1000, n_test=500",
               "C2": "d=20, 19 noise dims sd 3, label dim ±1+N(0,.3^2), PCA 5 comps / k-means 100"},
    "A_digits": A, "B_moons": B, "C1_cluster_violated": C1, "C2_lowvar_signal": C2,
    "runtime_sec": runtime})
print("runtime", runtime)
