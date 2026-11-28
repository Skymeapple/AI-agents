"""Q13 数据共享与隐私保护：DP-SGD + RDP 记账、成员推断攻击、非 IID 联邦平均.

(A) DP-SGD（逐样本裁剪 + 高斯噪声 + Poisson 子采样）训练逻辑回归；
    RDP 记账：Poisson 子采样高斯机制的整数阶 RDP（Mironov, Talwar & Zhang 2019 的精确公式），
    T 步线性组合，再用标准转换 eps = min_a [T*rdp(a) + log(1/delta)/(a-1)]。
    隐私-效用曲线：eps ∈ {0.5,1,2,4,8,∞}，两个数据集（breast_cancer、digits 奇偶二分类）。
(B) 成员推断（Yeom et al. 2018 损失阈值攻击）：刻意的小样本过拟合场景，
    比较经验优势与 (eps,delta)-DP 的理论上界。
(C) FedAvg：digits 10 类、20 个客户端、Dirichlet 标签划分控制异质性。
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
from scipy.special import gammaln, logsumexp
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.util import mean_ci, plt, results_dir, rng, save_results

T0 = time.time()
EPSILONS = [0.5, 1.0, 2.0, 4.0, 8.0, np.inf]
DELTA = 1e-5
ORDERS = np.arange(2, 257)
N_SEEDS_UTIL, N_SEEDS_MIA, N_SEEDS_FL = 10, 10, 5


# ------------------------------- RDP accountant -------------------------------
def rdp_subsampled_gaussian(q, sigma, alpha):
    """RDP of the Poisson-subsampled Gaussian mechanism at integer order alpha (exact, MTZ 2019)."""
    k = np.arange(alpha + 1)
    logc = gammaln(alpha + 1) - gammaln(k + 1) - gammaln(alpha - k + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = logc + (alpha - k) * np.log1p(-q) + k * np.log(q) + (k * k - k) / (2 * sigma**2)
    if q == 1.0:
        terms = np.where(k == alpha, (alpha * alpha - alpha) / (2 * sigma**2), -np.inf)
    return float(logsumexp(terms) / (alpha - 1))


def eps_from_rdp(q, sigma, steps, delta=DELTA):
    best = np.inf
    for a in ORDERS:
        e = steps * rdp_subsampled_gaussian(q, sigma, int(a)) + np.log(1 / delta) / (a - 1)
        best = min(best, e)
    return best


def calibrate_sigma(target_eps, q, steps, delta=DELTA):
    lo, hi = 0.3, 200.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if eps_from_rdp(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


# ----------------------------------- DP-SGD -----------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def dp_sgd_logreg(X, y, g, sigma, q, steps, lr, clip=1.0, wd=0.0):
    """Poisson-subsampled DP-SGD; sigma=0 -> clipped non-private SGD."""
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.hstack([X, np.ones((n, 1))])
    exp_b = q * n
    for _ in range(steps):
        m = g.random(n) < q
        xb, yb = Xb[m], y[m]
        G = (sigmoid(xb @ w) - yb)[:, None] * xb                 # per-example gradients
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        G = G / np.maximum(1.0, norms / clip)
        noisy = G.sum(0) + sigma * clip * g.standard_normal(d + 1)
        w -= lr * (noisy / exp_b + wd * w)
    return w


def predict_proba(w, X):
    return sigmoid(np.hstack([X, np.ones((len(X), 1))]) @ w)


def losses(w, X, y):
    p = np.clip(predict_proba(w, X), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def load(name):
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data, d.target
    d = load_digits()
    return d.data, (d.target % 2).astype(int)


# accountant sanity check: q=1 must equal alpha/(2 sigma^2)
acct_check = {f"alpha={a}": {"code": rdp_subsampled_gaussian(1.0, 2.0, a), "closed_form": a / (2 * 4.0)} for a in [2, 8, 32]}

# ------------------------------ (A) utility curve ------------------------------
UTIL_CFG = dict(batch=32, epochs=30, lr=0.5, clip=1.0, wd=1e-3)
util = []
sigmas = {}
for ds in ["breast_cancer", "digits_parity"]:
    X0, y0 = load(ds)
    for s in range(N_SEEDS_UTIL):
        Xtr, Xte, ytr, yte = train_test_split(X0, y0, test_size=0.25, stratify=y0, random_state=s)
        sc = StandardScaler().fit(Xtr)        # 注：标准化统计量未做隐私化（见效度威胁）
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        q = UTIL_CFG["batch"] / len(Xtr)
        steps = int(UTIL_CFG["epochs"] / q)
        skl = LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte)
        for e in EPSILONS:
            key = (ds, e)
            if key not in sigmas:
                sigmas[key] = 0.0 if np.isinf(e) else calibrate_sigma(e, q, steps)
            g = rng(10_000 + 100 * s + int(10 * min(e, 99)))
            w = dp_sgd_logreg(Xtr, ytr, g, sigmas[key], q, steps, UTIL_CFG["lr"], UTIL_CFG["clip"], UTIL_CFG["wd"])
            acc = float(((predict_proba(w, Xte) > 0.5) == yte).mean())
            util.append(dict(ds=ds, seed=s, eps=e, acc=acc, sklearn=skl, n_train=len(Xtr), q=q, steps=steps))
print(f"(A) done {time.time()-T0:.0f}s", flush=True)

# ------------------------------- (B) membership -------------------------------
MIA_CFG = dict(n_member=150, batch=15, epochs=100, lr=1.0, clip=1.0, wd=0.0)
X0, y0 = load("digits_parity")
X0 = X0 / 16.0
mia = []
mia_sig = {}
for s in range(N_SEEDS_MIA):
    g0 = rng(50_000 + s)
    idx = g0.permutation(len(X0))
    mem, non = idx[:MIA_CFG["n_member"]], idx[MIA_CFG["n_member"]:2 * MIA_CFG["n_member"]]
    test = idx[2 * MIA_CFG["n_member"]:]
    Xm, ym, Xn, yn = X0[mem], y0[mem], X0[non], y0[non]
    q = MIA_CFG["batch"] / len(Xm)
    steps = int(MIA_CFG["epochs"] / q)
    for e in EPSILONS:
        if e not in mia_sig:
            mia_sig[e] = 0.0 if np.isinf(e) else calibrate_sigma(e, q, steps)
        g = rng(60_000 + 100 * s + int(10 * min(e, 99)))
        w = dp_sgd_logreg(Xm, ym, g, mia_sig[e], q, steps, MIA_CFG["lr"], MIA_CFG["clip"], MIA_CFG["wd"])
        lm, ln = losses(w, Xm, ym), losses(w, Xn, yn)
        tau = lm.mean()                                     # Yeom：阈值=平均训练损失（攻击者已知）
        adv_yeom = float((lm <= tau).mean() - (ln <= tau).mean())
        allv = np.concatenate([lm, ln])
        lab = np.r_[np.ones(len(lm)), np.zeros(len(ln))]
        # 最优阈值（对攻击者偏乐观的上估计）：在半数样本上选阈值，在另一半上评估
        h = g.permutation(len(allv))
        A, B = h[: len(h) // 2], h[len(h) // 2:]
        cands = np.unique(allv[A])
        adv_A = [((allv[A] <= t) & (lab[A] == 1)).sum() / max((lab[A] == 1).sum(), 1)
                 - ((allv[A] <= t) & (lab[A] == 0)).sum() / max((lab[A] == 0).sum(), 1) for t in cands]
        tbest = cands[int(np.argmax(adv_A))]
        adv_split = float(((allv[B] <= tbest) & (lab[B] == 1)).sum() / (lab[B] == 1).sum()
                          - ((allv[B] <= tbest) & (lab[B] == 0)).sum() / (lab[B] == 0).sum())
        auc = float(roc_auc_score(lab, -allv))
        mia.append(dict(seed=s, eps=e, adv_yeom=adv_yeom, adv_split=adv_split, auc=auc,
                        train_acc=float(((predict_proba(w, Xm) > .5) == ym).mean()),
                        test_acc=float(((predict_proba(w, X0[test]) > .5) == y0[test]).mean()),
                        gen_gap_loss=float(ln.mean() - lm.mean())))
print(f"(B) done {time.time()-T0:.0f}s", flush=True)


# ---------------- (B2) worst-case audit with a canary (Jagielski et al. 2020) ----------------
AUDIT_CFG = dict(n_member=150, batch=15, epochs=30, lr=1.0, clip=1.0, runs_per_arm=120, eps=[1.0, 4.0, 8.0, np.inf])
from scipy.stats import beta as _beta


def cp_upper(k, n, a=0.05):
    return 1.0 if k >= n else float(_beta.ppf(1 - a / 2, k + 1, n - k))


audit = {}
g0 = rng(80_000)
base_idx = g0.permutation(len(X0))[:AUDIT_CFG["n_member"]]
Xa, ya = X0[base_idx], y0[base_idx]
x_can, y_can = np.ones(64), 0           # 全亮像素 + 标签 0：分布外且与多数同类样本冲突的"金丝雀"
qa = AUDIT_CFG["batch"] / (AUDIT_CFG["n_member"] + 1)
steps_a = int(AUDIT_CFG["epochs"] / qa)
for e in AUDIT_CFG["eps"]:
    sig = 0.0 if np.isinf(e) else calibrate_sigma(e, qa, steps_a)
    sc_in, sc_out = [], []
    for r in range(AUDIT_CFG["runs_per_arm"]):
        for with_c, store in [(True, sc_in), (False, sc_out)]:
            Xr = np.vstack([Xa, x_can]) if with_c else Xa
            yr = np.r_[ya, y_can] if with_c else ya
            w = dp_sgd_logreg(Xr, yr, rng(81_000 + 2 * r + with_c + int(1000 * min(e, 99))), sig, qa, steps_a,
                              AUDIT_CFG["lr"], AUDIT_CFG["clip"])
            store.append(float(losses(w, x_can[None], np.array([y_can]))[0]))
    sc_in, sc_out = np.array(sc_in), np.array(sc_out)
    h = len(sc_in) // 2                    # 前一半选阈值，后一半估计（避免选择偏差）
    cands = np.unique(np.r_[sc_in[:h], sc_out[:h]])
    advs = [(sc_in[:h] <= t).mean() - (sc_out[:h] <= t).mean() for t in cands]
    t = cands[int(np.argmax(advs))]
    n2 = len(sc_in) - h
    fn = int((sc_in[h:] > t).sum())       # 有金丝雀却判为"无"
    fp = int((sc_out[h:] <= t).sum())
    fnr_hi, fpr_hi = cp_upper(fn, n2), cp_upper(fp, n2)
    lb = max(np.log(max(1 - DELTA - fpr_hi, 1e-12) / fnr_hi), np.log(max(1 - DELTA - fnr_hi, 1e-12) / fpr_hi), 0.0)
    audit[e] = dict(sigma=sig, fnr=fn / n2, fpr=fp / n2, eps_lower_95=float(lb),
                    adv=float(1 - fn / n2 - fp / n2), auc=float(roc_auc_score(np.r_[np.ones(len(sc_in)), np.zeros(len(sc_out))], -np.r_[sc_in, sc_out])))
print(f"(B2) done {time.time()-T0:.0f}s", flush=True)


def bound_kairouz(e, d=DELTA):
    return 1.0 if np.isinf(e) else min(1.0, (np.exp(e) - 1 + 2 * d) / (np.exp(e) + 1))


def bound_yeom(e):
    return 1.0 if np.isinf(e) else min(1.0, np.exp(e) - 1)


# ---------------------------------- (C) FedAvg ---------------------------------
class MLP:
    def __init__(self, g, sizes=(64, 64, 10)):
        self.P = [g.standard_normal((sizes[0], sizes[1])) * np.sqrt(2 / sizes[0]), np.zeros(sizes[1]),
                  g.standard_normal((sizes[1], sizes[2])) * np.sqrt(2 / sizes[1]), np.zeros(sizes[2])]


def mlp_logits(P, X):
    return np.maximum(X @ P[0] + P[1], 0) @ P[2] + P[3]


def mlp_sgd(P, X, y, g, epochs, lr, bs=16):
    P = [p.copy() for p in P]
    for _ in range(epochs):
        idx = g.permutation(len(X))
        for s in range(0, len(X), bs):
            b = idx[s:s + bs]
            h = np.maximum(X[b] @ P[0] + P[1], 0)
            z = h @ P[2] + P[3]
            z -= z.max(1, keepdims=True)
            pr = np.exp(z); pr /= pr.sum(1, keepdims=True)
            pr[np.arange(len(b)), y[b]] -= 1
            pr /= len(b)
            gW2, gb2 = h.T @ pr, pr.sum(0)
            dh = (pr @ P[2].T) * (h > 0)
            gW1, gb1 = X[b].T @ dh, dh.sum(0)
            for p_, g_ in zip(P, [gW1, gb1, gW2, gb2]):
                p_ -= lr * g_
    return P


def dirichlet_split(y, n_clients, alpha, g):
    if alpha == "patho2":   # McMahan et al. 2017 式病态划分：按标签排序切 2K 片，每客户端 2 片
        order = np.argsort(y + 0.01 * g.random(len(y)), kind="stable")
        shards = np.array_split(order, 2 * n_clients)
        sid = g.permutation(2 * n_clients)
        return [np.concatenate([shards[sid[2 * k]], shards[sid[2 * k + 1]]]) for k in range(n_clients)]
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        ids = g.permutation(np.where(y == c)[0])
        prop = np.full(n_clients, 1 / n_clients) if np.isinf(alpha) else g.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(prop) * len(ids)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(ids, cuts)):
            parts[k].extend(chunk.tolist())
    return [np.array(p, int) for p in parts]


FL_CFG = dict(clients=20, rounds=40, lr=0.05, alphas=[np.inf, 1.0, 0.3, 0.1, "patho2"], local_epochs=[1, 5])
dg = load_digits()
XD, yD = dg.data / 16.0, dg.target
fl = []
curves = {}
for s in range(N_SEEDS_FL):
    Xtr, Xte, ytr, yte = train_test_split(XD, yD, test_size=0.3, stratify=yD, random_state=s)
    init = MLP(rng(70_000 + s)).P
    # 中心化基线：同等样本-梯度步数预算（rounds*E 个 epoch）
    for E in FL_CFG["local_epochs"]:
        Pc = mlp_sgd(init, Xtr, ytr, rng(71_000 + s), FL_CFG["rounds"] * E, FL_CFG["lr"])
        fl.append(dict(seed=s, alpha="central", E=E, acc=float((mlp_logits(Pc, Xte).argmax(1) == yte).mean())))
    for a in FL_CFG["alphas"]:
        parts = dirichlet_split(ytr, FL_CFG["clients"], a, rng(72_000 + s + (999 if a == "patho2" else int(10 * min(a, 99)))))
        # 标签分布异质性：客户端标签分布与全局分布的平均总变差距离
        glob = np.bincount(ytr, minlength=10) / len(ytr)
        tv = float(np.mean([0.5 * np.abs(np.bincount(ytr[p], minlength=10) / max(len(p), 1) - glob).sum()
                            for p in parts if len(p) > 0]))
        for E in FL_CFG["local_epochs"]:
            P = [p.copy() for p in init]
            g = rng(73_000 + s)
            curve = []
            for r in range(FL_CFG["rounds"]):
                new, wts = [], []
                for p in parts:
                    if len(p) == 0:
                        continue
                    new.append(mlp_sgd(P, Xtr[p], ytr[p], g, E, FL_CFG["lr"]))
                    wts.append(len(p))
                wts = np.array(wts) / sum(wts)
                P = [sum(w_ * n_[i] for w_, n_ in zip(wts, new)) for i in range(4)]
                curve.append(float((mlp_logits(P, Xte).argmax(1) == yte).mean()))
            fl.append(dict(seed=s, alpha=str(a), E=E, acc=curve[-1], tv=tv))
            curves.setdefault(f"alpha={a}|E={E}", []).append(curve)
print(f"(C) done {time.time()-T0:.0f}s", flush=True)


# --------------------------------- aggregation ---------------------------------
def ag(rows, key, **flt):
    v = [r[key] for r in rows if all(r[a] == b for a, b in flt.items())]
    m, h = mean_ci(v)
    return {"mean": m, "ci95": h, "n": len(v)}


def ek(e):
    if isinstance(e, str):
        return e
    return "inf" if np.isinf(e) else str(e)


out = {"config": dict(epsilons=[ek(e) for e in EPSILONS], delta=DELTA, rdp_orders="2..256 (integer)",
                      util=UTIL_CFG, mia=MIA_CFG, fl={**FL_CFG, "alphas": [ek(a) for a in FL_CFG["alphas"]]},
                      seeds=dict(util=N_SEEDS_UTIL, mia=N_SEEDS_MIA, fl=N_SEEDS_FL)),
       "accountant_check_q1": acct_check}
out["sigma"] = {f"{ds}|eps={ek(e)}": v for (ds, e), v in sigmas.items()}
out["sigma_mia"] = {f"eps={ek(e)}": v for e, v in mia_sig.items()}
out["utility"] = {}
for ds in ["breast_cancer", "digits_parity"]:
    for e in EPSILONS:
        out["utility"][f"{ds}|eps={ek(e)}"] = ag(util, "acc", ds=ds, eps=e)
    out["utility"][f"{ds}|sklearn_nonprivate"] = ag(util, "sklearn", ds=ds, eps=np.inf)
    r0 = [u for u in util if u["ds"] == ds][0]
    out["utility"][f"{ds}|setup"] = dict(n_train=r0["n_train"], q=r0["q"], steps=r0["steps"])
out["mia"] = {}
for e in EPSILONS:
    d_ = {k: ag(mia, k, eps=e) for k in ["adv_yeom", "adv_split", "auc", "train_acc", "test_acc", "gen_gap_loss"]}
    d_["bound_kairouz"] = bound_kairouz(e)
    d_["bound_yeom_e^eps-1"] = bound_yeom(e)
    out["mia"][f"eps={ek(e)}"] = d_
out["mia_setup"] = {"q": MIA_CFG["batch"] / MIA_CFG["n_member"], "steps": int(MIA_CFG["epochs"] * MIA_CFG["n_member"] / MIA_CFG["batch"])}
out["audit_canary"] = {"config": {**AUDIT_CFG, "eps": [ek(e) for e in AUDIT_CFG["eps"]], "q": qa, "steps": steps_a},
                       **{f"eps={ek(e)}": {**v, "bound_kairouz": bound_kairouz(e)} for e, v in audit.items()}}
out["fedavg"] = {}
for a in FL_CFG["alphas"]:
    for E in FL_CFG["local_epochs"]:
        out["fedavg"][f"alpha={ek(a)}|E={E}"] = {"acc": ag(fl, "acc", alpha=str(a), E=E), "label_TV": ag(fl, "tv", alpha=str(a), E=E)}
for E in FL_CFG["local_epochs"]:
    out["fedavg"][f"central|epochs={FL_CFG['rounds']*E}"] = {"acc": ag(fl, "acc", alpha="central", E=E)}
T = {}
for E in FL_CFG["local_epochs"]:
    a_ = [r["acc"] for r in fl if r["alpha"] == str(np.inf) and r["E"] == E]
    for het in ["0.1", "patho2"]:
        b_ = [r["acc"] for r in fl if r["alpha"] == het and r["E"] == E]
        T[f"fedavg|E={E}|iid_minus_{het}"] = {"diff": float(np.mean(np.array(a_) - b_)), "paired_t_p": float(stats.ttest_rel(a_, b_).pvalue)}
for a in FL_CFG["alphas"]:
    a_ = [r["acc"] for r in fl if r["alpha"] == str(a) and r["E"] == 1]
    b_ = [r["acc"] for r in fl if r["alpha"] == str(a) and r["E"] == 5]
    T[f"fedavg|alpha={ek(a)}|E5_minus_E1"] = {"diff": float(np.mean(np.array(b_) - a_)), "paired_t_p": float(stats.ttest_rel(b_, a_).pvalue)}
for ds in ["breast_cancer", "digits_parity"]:
    base = [u["acc"] for u in util if u["ds"] == ds and np.isinf(u["eps"])]
    for e in EPSILONS[:-1]:
        x = [u["acc"] for u in util if u["ds"] == ds and u["eps"] == e]
        T[f"utility|{ds}|eps={e}|drop_vs_inf"] = {"diff": float(np.mean(np.array(base) - x)), "paired_t_p": float(stats.ttest_rel(base, x).pvalue)}
inf_adv = [m["adv_yeom"] for m in mia if np.isinf(m["eps"])]
for e in EPSILONS[:-1]:
    x = [m["adv_yeom"] for m in mia if m["eps"] == e]
    T[f"mia|eps={e}|adv_yeom_inf_minus_eps"] = {"diff": float(np.mean(np.array(inf_adv) - x)), "paired_t_p": float(stats.ttest_rel(inf_adv, x).pvalue)}
out["tests"] = T
out["runtime_sec"] = time.time() - T0

# ----------------------------------- figures -----------------------------------
P_ = plt()
R = results_dir(__file__)
xs = np.arange(len(EPSILONS))
fig, axes = P_.subplots(1, 2, figsize=(11, 4))
for ds, c in [("breast_cancer", "C0"), ("digits_parity", "C1")]:
    m = [out["utility"][f"{ds}|eps={ek(e)}"]["mean"] for e in EPSILONS]
    h = [out["utility"][f"{ds}|eps={ek(e)}"]["ci95"] for e in EPSILONS]
    axes[0].errorbar(xs, m, yerr=h, marker="o", capsize=3, color=c, label=ds)
    axes[0].axhline(out["utility"][f"{ds}|sklearn_nonprivate"]["mean"], color=c, ls=":", lw=1)
axes[0].set_xticks(xs, [ek(e) for e in EPSILONS])
axes[0].set_xlabel("ε (δ=1e-5)")
axes[0].set_ylabel("测试准确率")
axes[0].set_title("DP-SGD 逻辑回归：隐私-效用曲线（点线=sklearn 非私有）")
axes[0].legend()
for k, lab, c in [("adv_yeom", "Yeom 阈值攻击优势", "C3"), ("adv_split", "最优阈值(交叉验证)优势", "C4")]:
    m = [out["mia"][f"eps={ek(e)}"][k]["mean"] for e in EPSILONS]
    h = [out["mia"][f"eps={ek(e)}"][k]["ci95"] for e in EPSILONS]
    axes[1].errorbar(xs, m, yerr=h, marker="o", capsize=3, color=c, label=lab)
axes[1].plot(xs, [bound_kairouz(e) for e in EPSILONS], "k--", label="(ε,δ)-DP 上界 (e^ε-1+2δ)/(e^ε+1)")
axes[1].plot(xs, [bound_yeom(e) for e in EPSILONS], "k:", label="Yeom 上界 e^ε-1")
axes[1].set_xticks(xs, [ek(e) for e in EPSILONS])
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_xlabel("ε")
axes[1].set_ylabel("成员推断优势 TPR-FPR")
axes[1].set_title("成员推断：经验优势 vs 理论上界")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(R / "privacy_utility_mia.png")

fig, ax = P_.subplots(figsize=(6.5, 4))
for i, a in enumerate(FL_CFG["alphas"]):
    for E, ls in [(1, "--"), (5, "-")]:
        cv = np.array(curves[f"alpha={a}|E={E}"])
        ax.plot(np.arange(1, FL_CFG["rounds"] + 1), cv.mean(0), ls=ls, color=f"C{i}", label=f"α={ek(a)}, E={E}")
ax.set_xlabel("通信轮数")
ax.set_ylabel("测试准确率")
ax.set_title("FedAvg（20 客户端，Dirichlet 标签划分）")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(R / "fedavg.png")

save_results(__file__, out)
print(f"done {time.time()-T0:.0f}s")
