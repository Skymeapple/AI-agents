"""Q08 因果驱动学习：不变性预测 vs 经验风险最小化；观测数据因果发现的可识别性边界。

Part 1（多环境线性 SCM，IRM/ICP 设定）：
  X1 ~ N(mu_e, s_e^2), X2 ~ N(0, s_e^2)            （Y 的原因，环境改变其分布）
  Y  = 1.0*X1 + 0.5*X2 + N(0,1)                       （不变机制）
  X3 = c_e*Y + N(0, 0.5^2)                            （Y 的结果，"虚假"相关强度随环境变化）
  X4 = d_e*Y + X2 + N(0, tau_e^2)                     （Y 的结果，训练中 d=1、噪声随环境变化）
  X5 ~ N(0,1)                                         （无关变量）
  训练环境 e1:(mu,s,c,tau)=(0,1,1,0.3), e2:(1,1.5,2,1.0)；测试环境 A：(-1,2,-1,0.3,d=1)（仅 X3 翻转）；
  测试环境 B：同 A 但 d=-1（两个子节点机制都被干预）。
  方法：ERM 汇总 OLS；Oracle OLS(X1,X2)；ICP（Peters et al. 2016，残差均值+方差检验，α=0.05）；
        IRMv1 线性版（Arjovsky et al. 2019），λ∈{1,10,100,1000} 全部报告。每环境样本数 n∈{50,200,1000,5000}。
Part 2（可识别性）：
  (a) 三变量链/叉/对撞，线性高斯，PC 式条件独立检验（Fisher-z，α=0.01）+ 最大似然比较；
  (b) 两变量 X→Y 方向判别（回归残差-自变量 HSIC 独立性，LiNGAM/ANM 思想），高斯 vs 均匀噪声。
"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, save_results, mean_ci, plt, results_dir

import numpy as np
from scipy import stats, optimize

T0 = time.time()
OUT = results_dir(__file__)
SEEDS = 20
NS = [50, 200, 1000, 5000]
TRAIN_ENVS = [dict(mu=0, s=1.0, c=1.0, tau=0.3), dict(mu=1, s=1.5, c=2.0, tau=1.0)]
TEST_ENVS = {"A: 仅翻转 X3": dict(mu=-1, s=2.0, c=-1.0, tau=0.3, d=1.0),
             "B: 翻转 X3 且翻转 X4": dict(mu=-1, s=2.0, c=-1.0, tau=0.3, d=-1.0)}
TA, TB = list(TEST_ENVS)
LAMS = [1, 10, 100, 1000]
ALPHA = 0.05


def sample_env(n, e, g):
    X1 = e["mu"] + e["s"] * g.standard_normal(n)
    X2 = e["s"] * g.standard_normal(n)
    Y = 1.0 * X1 + 0.5 * X2 + g.standard_normal(n)
    X3 = e["c"] * Y + 0.5 * g.standard_normal(n)
    X4 = e.get("d", 1.0) * Y + X2 + e["tau"] * g.standard_normal(n)
    X5 = g.standard_normal(n)
    return np.c_[X1, X2, X3, X4, X5], Y


def ols(X, y):
    A = np.c_[np.ones(len(y)), X]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w


def pred(w, X):
    return w[0] + X @ w[1:]


def icp(Xs, ys):
    """Invariant causal prediction: intersection of accepted sets. Returns (set, accepted_sets)."""
    X = np.vstack(Xs); y = np.concatenate(ys); env = np.concatenate([np.full(len(v), i) for i, v in enumerate(ys)])
    d = X.shape[1]; accepted = []
    for r in range(d + 1):
        for S in itertools.combinations(range(d), r):
            w = ols(X[:, S], y) if S else np.array([y.mean()])
            res = y - (pred(w, X[:, S]) if S else w[0])
            groups = [res[env == i] for i in range(len(ys))]
            p_mean = stats.f_oneway(*groups).pvalue
            p_var = stats.levene(*groups).pvalue
            p = min(1.0, 2 * min(p_mean, p_var))          # Bonferroni over the two tests
            if p > ALPHA:
                accepted.append(set(S))
    if not accepted:
        return set(), accepted                              # all rejected -> model misspecified; return empty
    return set.intersection(*accepted), accepted


def irm_linear(Xs, ys, lam, w0):
    """IRMv1 for linear regression with dummy scalar classifier: sum_e R_e + lam * sum_e (dR_e/dv |_{v=1})^2."""
    As = [np.c_[np.ones(len(y)), X] for X, y in zip(Xs, ys)]

    def obj(w):
        tot = 0.0; grad = np.zeros_like(w)
        for A, y in zip(As, ys):
            f = A @ w; r = f - y; n = len(y)
            R = np.mean(r ** 2); gR = 2 * A.T @ r / n
            D = 2 * np.mean(r * f)                           # dR/dv at v=1 for prediction v*f
            gD = 2 * A.T @ (2 * f - y) / n
            tot += R + lam * D ** 2; grad += gR + lam * 2 * D * gD
        return tot, grad
    sol = optimize.minimize(obj, w0, jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    return sol.x


# ----------------------- Part 1 -----------------------
VARS = ["X1", "X2", "X3", "X4", "X5"]
methods = ["ERM(汇总)", "Oracle(X1,X2)", "ICP"] + [f"IRMv1 λ={l}" for l in LAMS]
P1 = {n: {m: {"mse_test": [], "mse_testB": [], "mse_train": [], "coef": []} for m in methods} for n in NS}
icp_sets = {n: [] for n in NS}
for n in NS:
    for s in range(SEEDS):
        g = rng(10 * n + s)
        Xs, ys = zip(*[sample_env(n, e, g) for e in TRAIN_ENVS])
        Xte, yte = sample_env(5000, TEST_ENVS[TA], g)
        XteB, yteB = sample_env(5000, TEST_ENVS[TB], g)
        Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
        w_erm = ols(Xtr, ytr)
        out = {"ERM(汇总)": w_erm}
        wo = ols(Xtr[:, [0, 1]], ytr); w = np.zeros(6); w[0] = wo[0]; w[[1, 2]] = wo[1:]; out["Oracle(X1,X2)"] = w
        S, acc = icp(Xs, ys); icp_sets[n].append(sorted(S))
        w = np.zeros(6)
        if S:
            Sl = sorted(S); wi = ols(Xtr[:, Sl], ytr); w[0] = wi[0]; w[[i + 1 for i in Sl]] = wi[1:]
        else:
            w[0] = ytr.mean()
        out["ICP"] = w
        for l in LAMS:
            out[f"IRMv1 λ={l}"] = irm_linear(Xs, ys, l, w_erm)
        for m, w in out.items():
            P1[n][m]["mse_test"].append(np.mean((pred(w, Xte) - yte) ** 2))
            P1[n][m]["mse_testB"].append(np.mean((pred(w, XteB) - yteB) ** 2))
            P1[n][m]["mse_train"].append(np.mean((pred(w, Xtr) - ytr) ** 2))
            P1[n][m]["coef"].append(w[1:])
print("part1", time.time() - T0, flush=True)

S1 = {}
for n in NS:
    S1[n] = {}
    for m in methods:
        te = np.array(P1[n][m]["mse_test"]); teB = np.array(P1[n][m]["mse_testB"]); tr = np.array(P1[n][m]["mse_train"]); C = np.array(P1[n][m]["coef"])
        e = {"mse_test": dict(zip(["mean", "ci95"], mean_ci(te))), "mse_testB": dict(zip(["mean", "ci95"], mean_ci(teB))),
             "mse_test_median": float(np.median(te)), "mse_testB_median": float(np.median(teB)), "mse_train": dict(zip(["mean", "ci95"], mean_ci(tr))),
             "coef_mean": dict(zip(VARS, C.mean(0))), "coef_X3_ci95": mean_ci(C[:, 2])[1]}
        if m != "Oracle(X1,X2)":
            o = np.array(P1[n]["Oracle(X1,X2)"]["mse_test"])
            e["p_vs_oracle_wilcoxon"] = float(stats.wilcoxon(te, o).pvalue) if np.any(te != o) else 1.0
            oB = np.array(P1[n]["Oracle(X1,X2)"]["mse_testB"])
            e["p_vs_oracle_wilcoxon_B"] = float(stats.wilcoxon(teB, oB).pvalue) if np.any(teB != oB) else 1.0
        S1[n][m] = e
    sets = icp_sets[n]
    S1[n]["ICP_selection"] = {
        "exact_{X1,X2}": float(np.mean([s == [0, 1] for s in sets])),
        "subset_of_parents": float(np.mean([set(s) <= {0, 1} for s in sets])),
        "empty": float(np.mean([len(s) == 0 for s in sets])),
        "includes_child_X3_or_X4": float(np.mean([bool(set(s) & {2, 3}) for s in sets])),
        "freq_per_var": {VARS[i]: float(np.mean([i in s for s in sets])) for i in range(5)}}
# irreducible test MSE for the causal predictor: Var(eps_Y)=1
print({n: S1[n]["ICP_selection"] for n in NS})

# ----------------------- Part 2a: Markov equivalence -----------------------
def fisher_z_p(C, i, j, cond, n):
    idx = [i, j] + list(cond)
    Pm = np.linalg.inv(C[np.ix_(idx, idx)])
    r = -Pm[0, 1] / np.sqrt(Pm[0, 0] * Pm[1, 1])
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - len(cond) - 3)
    return 2 * stats.norm.sf(abs(z))


def gen3(kind, n, g):
    a, b = 0.8, 0.8
    if kind == "链 X→Y→Z":
        X = g.standard_normal(n); Y = a * X + g.standard_normal(n); Z = b * Y + g.standard_normal(n)
    elif kind == "叉 X←Y→Z":
        Y = g.standard_normal(n); X = a * Y + g.standard_normal(n); Z = b * Y + g.standard_normal(n)
    elif kind == "反向链 X←Y←Z":
        Z = g.standard_normal(n); Y = b * Z + g.standard_normal(n); X = a * Y + g.standard_normal(n)
    else:
        X = g.standard_normal(n); Z = g.standard_normal(n); Y = a * X + b * Z + g.standard_normal(n)
    return np.c_[X, Y, Z]


def gauss_dag_loglik(D, parents):
    """max log-likelihood of linear-Gaussian DAG (parents: dict node->list)."""
    n = len(D); ll = 0.0
    for v, pa in parents.items():
        A = np.c_[np.ones(n), D[:, pa]] if pa else np.ones((n, 1))
        r = D[:, v] - A @ np.linalg.lstsq(A, D[:, v], rcond=None)[0]
        s2 = np.mean(r ** 2); ll += -0.5 * n * (np.log(2 * np.pi * s2) + 1)
    return ll


DAGS = {"链 X→Y→Z": {0: [], 1: [0], 2: [1]}, "叉 X←Y→Z": {1: [], 0: [1], 2: [1]},
        "反向链 X←Y←Z": {2: [], 1: [2], 0: [1]}, "对撞 X→Y←Z": {0: [], 2: [], 1: [0, 2]}}
KINDS = list(DAGS)
P2a = {k: {"pc_output": {}, "loglik_gap_vs_true": {d: [] for d in DAGS}} for k in KINDS}
for k in KINDS:
    for s in range(50):
        g = rng(5000 + s); n = 1000
        D = gen3(k, n, g); C = np.corrcoef(D.T)
        indep_marg = fisher_z_p(C, 0, 2, [], n) > 0.01
        indep_cond = fisher_z_p(C, 0, 2, [1], n) > 0.01
        if indep_cond and not indep_marg:
            outp = "X—Y—Z（不可定向：链/叉/反向链等价类）"
        elif indep_marg and not indep_cond:
            outp = "X→Y←Z（对撞，可定向）"
        else:
            outp = "其他/检验错误"
        P2a[k]["pc_output"][outp] = P2a[k]["pc_output"].get(outp, 0) + 1
        lt = gauss_dag_loglik(D, DAGS[k])
        for d, pa in DAGS.items():
            P2a[k]["loglik_gap_vs_true"][d].append(gauss_dag_loglik(D, pa) - lt)
for k in KINDS:
    P2a[k]["loglik_gap_vs_true"] = {d: {"mean": float(np.mean(v)), "max_abs": float(np.max(np.abs(v)))}
                                    for d, v in P2a[k]["loglik_gap_vs_true"].items()}

# ----------------------- Part 2b: direction from non-Gaussianity -----------------------
def hsic(a, b):
    n = len(a)
    def K(x):
        d2 = (x[:, None] - x[None, :]) ** 2
        med = np.median(d2[d2 > 0]); return np.exp(-d2 / med)
    H = np.eye(n) - 1.0 / n
    return np.trace(K(a) @ H @ K(b) @ H) / n ** 2


def direction_correct(x, y):
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std()
    rf = y - np.polyfit(x, y, 1)[0] * x; rb = x - np.polyfit(y, x, 1)[0] * y
    return hsic(x, rf) < hsic(y, rb)


P2b = {}
for noise in ["高斯", "均匀"]:
    P2b[noise] = {}
    for n in [100, 300, 1000]:
        hits = []
        for s in range(40):
            g = rng(9000 + 31 * n + s)
            e1 = g.standard_normal(n) if noise == "高斯" else g.uniform(-np.sqrt(3), np.sqrt(3), n)
            e2 = g.standard_normal(n) if noise == "高斯" else g.uniform(-np.sqrt(3), np.sqrt(3), n)
            x = e1; y = 0.8 * x + 0.6 * e2
            hits.append(direction_correct(x, y))
        hits = np.array(hits, float)
        bt = stats.binomtest(int(hits.sum()), len(hits), 0.5)
        ci = bt.proportion_ci()
        P2b[noise][n] = {"acc": float(hits.mean()), "ci95_low": ci.low, "ci95_high": ci.high,
                         "p_vs_chance": float(bt.pvalue), "trials": len(hits)}
print("part2", time.time() - T0, flush=True)

# ----------------------- figures -----------------------
Pl = plt()
fig, axes = Pl.subplots(1, 4, figsize=(22, 4.4))
for ax, key, tt in [(axes[0], "mse_test", "测试环境 A（仅 X3 翻转）"), (axes[1], "mse_testB", "测试环境 B（X3、X4 均翻转）")]:
    for m in methods:
        mu = [S1[n][m][key]["mean"] for n in NS]; ci = [S1[n][m][key]["ci95"] for n in NS]
        ax.errorbar(NS, mu, yerr=ci, marker="o", capsize=3, label=m)
    ax.axhline(1.0, color="k", ls=":", lw=1, label="不可约噪声 Var(ε_Y)=1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("每个训练环境样本数 n"); ax.set_ylabel("测试 MSE")
    ax.set_title(tt); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
ax = axes[2]
for m in methods:
    mu = [S1[n][m]["mse_train"]["mean"] for n in NS]
    ax.plot(NS, mu, marker="o", label=m)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("n"); ax.set_ylabel("训练（汇总）MSE"); ax.set_title("训练误差：ERM 最小，但依赖非因果特征"); ax.grid(alpha=.3)
ax = axes[3]
xs = np.arange(len(NS)); w = 0.25
for j, key in enumerate(["exact_{X1,X2}", "subset_of_parents", "includes_child_X3_or_X4"]):
    ax.bar(xs + (j - 1) * w, [S1[n]["ICP_selection"][key] for n in NS], w,
           label={"exact_{X1,X2}": "恰为 {X1,X2}", "subset_of_parents": "⊆ 真实父节点（无假阳性）",
                  "includes_child_X3_or_X4": "含子节点（假阳性）"}[key])
ax.axhline(1 - ALPHA, color="k", ls=":", lw=1); ax.set_xticks(xs); ax.set_xticklabels(NS)
ax.set_xlabel("n"); ax.set_ylabel("频率（20 种子）"); ax.set_title("ICP 选择的变量集"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "invariance.png"); Pl.close(fig)

fig, ax = Pl.subplots(figsize=(6, 4))
for noise, mk in [("高斯", "o"), ("均匀", "s")]:
    ns = [100, 300, 1000]
    mu = [P2b[noise][n]["acc"] for n in ns]
    lo = [P2b[noise][n]["acc"] - P2b[noise][n]["ci95_low"] for n in ns]
    hi = [P2b[noise][n]["ci95_high"] - P2b[noise][n]["acc"] for n in ns]
    ax.errorbar(ns, mu, yerr=[lo, hi], marker=mk, capsize=3, label=f"{noise}噪声")
ax.axhline(0.5, color="k", ls=":"); ax.set_xscale("log"); ax.set_ylim(0, 1.05)
ax.set_xlabel("样本数 n"); ax.set_ylabel("方向判别正确率"); ax.set_title("仅凭观测数据判别 X→Y：依赖非高斯性")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "direction.png"); Pl.close(fig)

runtime = time.time() - T0
save_results(__file__, {"config": {"seeds_part1": SEEDS, "n_per_env": NS, "train_envs": TRAIN_ENVS, "test_envs": TEST_ENVS,
                                   "icp_alpha": ALPHA, "irm_lambdas": LAMS, "test_n": 5000,
                                   "part2a": "n=1000, 50 seeds, Fisher-z alpha=0.01, edge weights 0.8",
                                   "part2b": "x=e1, y=0.8x+0.6e2, unit-variance noise, 40 trials, HSIC gaussian kernel median heuristic"},
                        "part1": S1, "part2a_markov_equivalence": P2a, "part2b_direction": P2b, "runtime_sec": runtime})
print("runtime", runtime)
