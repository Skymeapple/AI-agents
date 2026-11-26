"""Q06 前沿实验：上下文学习（ICL）何时被示范“帮助”、何时被“误导”——一个可解的线性注意力模型。

任务：x~N(0,I_d), w~N(0,I/d), 示范 y_i = s_i·x_iᵀw + σε_i，s_i=-1 的概率为 ρ（被“翻转”的错误示范）。
ICL 学习器：单层线性自注意力（LSA）的预测 ŷ = x_qᵀ Γ h，h=(1/n)Σ y_i x_i；Γ 在预训练提示上用最小二乘拟合
           （对 Γ 线性，故全局最优可精确求得）。对照：岭回归（ρ=0 时的 Bayes 最优）、已知 ρ 的稳健 EM（近似 Bayes）、
           离散先验 Bayes（dMMSE，用于任务多样性实验）。
理论（paper.md §4）：a=1-2ρ，D=d(1+σ²)。最优 Γ*=γ*I，γ*=a_tr n/((n-1)a_tr²+2+D)；
  测试风险 R = 1 - 2γ a_te + γ²[(n-1)a_te²+2+D]/n；匹配时 R=1-a²n/((n-1)a²+2+D)；n→∞ 时 R→(1-a_te/a_tr)²。
"""
import os, sys, time, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.util import rng, mean_ci, plt, results_dir

import numpy as np
from scipy import stats

T0 = time.time()
OUT = results_dir(__file__)
D, SIG = 10, 0.25
DD = D * (1 + SIG ** 2)
SEEDS = 5
NPRE, NTEST, CHUNK = 8000, 3000, 1000


def prompts(m, n, rho, g, W=None):
    """returns X (m,n,d), y (m,n), xq (m,d), yq clean-signal (m,), w (m,d)."""
    if W is None:
        w = g.standard_normal((m, D)) / np.sqrt(D)
    else:
        w = W[g.integers(0, len(W), m)]
    X = g.standard_normal((m, n, D))
    s = np.where(g.uniform(size=(m, n)) < rho, -1.0, 1.0)
    y = s * np.einsum("mnd,md->mn", X, w) + SIG * g.standard_normal((m, n))
    xq = g.standard_normal((m, D))
    return X, y, xq, np.einsum("md,md->m", xq, w), w


def h_of(X, y):
    return np.einsum("mnd,mn->md", X, y) / X.shape[1]


def fit_lsa(n, rho, g, W=None):
    A = np.zeros((D * D, D * D)); b = np.zeros(D * D)
    for _ in range(NPRE // CHUNK):
        X, y, xq, fq, _ = prompts(CHUNK, n, rho, g, W)
        F = np.einsum("mi,mj->mij", xq, h_of(X, y)).reshape(CHUNK, -1)
        yq = fq + SIG * g.standard_normal(CHUNK)
        A += F.T @ F; b += F.T @ yq
    return np.linalg.solve(A + 1e-8 * np.eye(D * D), b).reshape(D, D)


def risk_lsa(G, X, y, xq, fq):
    pred = np.einsum("mi,ij,mj->m", xq, G, h_of(X, y))
    return (pred - fq) ** 2


def risk_ridge(X, y, xq, fq, lam=SIG ** 2 * D):
    A = np.einsum("mni,mnj->mij", X, X) + lam * np.eye(D)
    wh = np.linalg.solve(A, np.einsum("mni,mn->mi", X, y)[..., None])[..., 0]
    return (np.einsum("mi,mi->m", xq, wh) - fq) ** 2


def risk_em(X, y, xq, fq, rho, iters=30):
    """robust EM for sign-flip contamination (knows rho): ridge on y_i(1-2π_i)."""
    lam = SIG ** 2 * D
    XtX = np.einsum("mni,mnj->mij", X, X) + lam * np.eye(D)
    a = 1 - 2 * rho
    wh = np.linalg.solve(XtX, np.einsum("mni,mn->mi", X, y)[..., None])[..., 0] / max(a, 0.05)
    for _ in range(iters):
        f = np.einsum("mni,mi->mn", X, wh)
        lp = np.log(1 - rho) - (y - f) ** 2 / (2 * SIG ** 2); lm = np.log(max(rho, 1e-12)) - (y + f) ** 2 / (2 * SIG ** 2)
        pi = 1 / (1 + np.exp(np.clip(lp - lm, -50, 50)))
        yt = y * (1 - 2 * pi)
        wh = np.linalg.solve(XtX, np.einsum("mni,mn->mi", X, yt)[..., None])[..., 0]
    return (np.einsum("mi,mi->m", xq, wh) - fq) ** 2


def gamma_star(n, a): return a * n / ((n - 1) * a ** 2 + 2 + DD)
def risk_theory(gam, n, a_te): return 1 - 2 * gam * a_te + gam ** 2 * ((n - 1) * a_te ** 2 + 2 + DD) / n


def ms(v):
    mu, ci = mean_ci(v); return {"mean": mu, "ci95": ci}


# ---------------- E1: matched corruption, risk vs n ----------------
NS = [5, 10, 20, 40, 80, 160]
RHOS = [0.0, 0.1, 0.2, 0.3, 0.4]
E1 = {}
for rho in RHOS:
    E1[rho] = {}
    for n in NS:
        r_lsa, r_ridge, r_em, gam_fit = [], [], [], []
        for s in range(SEEDS):
            g = rng(1000 * RHOS.index(rho) + 10 * NS.index(n) + s)
            G = fit_lsa(n, rho, g)
            gam_fit.append(np.trace(G) / D)
            X, y, xq, fq, _ = prompts(NTEST, n, rho, g)
            r_lsa.append(risk_lsa(G, X, y, xq, fq).mean())
            r_ridge.append(risk_ridge(X, y, xq, fq).mean())
            if n >= 10 and rho > 0:
                r_em.append(risk_em(X[:1000], y[:1000], xq[:1000], fq[:1000], rho).mean())
        a = 1 - 2 * rho
        E1[rho][n] = {"LSA": ms(r_lsa), "ridge": ms(r_ridge), "theory_LSA": risk_theory(gamma_star(n, a), n, a),
                      "gamma_fit": ms(gam_fit), "gamma_theory": gamma_star(n, a)}
        if r_em:
            E1[rho][n]["robust_EM"] = ms(r_em)
    print("E1", rho, round(time.time() - T0, 1), flush=True)

# ---------------- E2: corruption mismatch between pretraining and test ----------------
RHO_TR = [0.0, 0.2, 0.4]
RHO_TE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
E2 = {}
for n in [20, 160]:
    E2[n] = {}
    for rt in RHO_TR:
        E2[n][rt] = {}
        Gs = [fit_lsa(n, rt, rng(50000 + 100 * n + 10 * RHO_TR.index(rt) + s)) for s in range(SEEDS)]
        for re_ in RHO_TE:
            rr = []
            for s, G in enumerate(Gs):
                g = rng(60000 + 100 * n + 10 * RHO_TR.index(rt) + s + 1000 * RHO_TE.index(re_))
                X, y, xq, fq, _ = prompts(NTEST, n, re_, g)
                rr.append(risk_lsa(G, X, y, xq, fq).mean())
            a_tr, a_te = 1 - 2 * rt, 1 - 2 * re_
            E2[n][rt][re_] = {"LSA": ms(rr), "theory": risk_theory(gamma_star(n, a_tr), n, a_te),
                              "theory_asymptotic": (1 - a_te / a_tr) ** 2}
    print("E2", n, round(time.time() - T0, 1), flush=True)
# key paired comparison: LSA trained at rho=0.2, n=160 -- clean test demos vs matched (rho=0.2) demos
E2_key = {}
for rt in [0.2, 0.4]:
    clean = E2[160][rt][0.0]["LSA"]["mean"]; matched = E2[160][rt][rt]["LSA"]["mean"]
    E2_key[rt] = {"clean_demos_risk": clean, "matched_demos_risk": matched,
                  "theory_clean": E2[160][rt][0.0]["theory"], "theory_matched": E2[160][rt][rt]["theory"]}

# ---------------- E3: task diversity ----------------
MS = [1, 2, 4, 8, 16, 32, 64, 256]
N3 = 40
E3 = {}
for M in MS + ["∞"]:
    rows = {"LSA_new": [], "LSA_train": [], "ridge_new": [], "dMMSE_new": [], "dMMSE_train": [], "theory_new": [], "theory_train": []}
    for s in range(SEEDS):
        g = rng(90000 + 100 * (MS.index(M) if M != "∞" else 99) + s)
        W = None if M == "∞" else g.standard_normal((M, D)) / np.sqrt(D)
        G = fit_lsa(N3, 0.0, g, W)
        Xn, yn, xqn, fqn, _ = prompts(NTEST, N3, 0.0, g)
        rows["LSA_new"].append(risk_lsa(G, Xn, yn, xqn, fqn).mean())
        rows["ridge_new"].append(risk_ridge(Xn, yn, xqn, fqn).mean())
        # theory for the trained-on-M optimum
        Sw = np.eye(D) / D if W is None else W.T @ W / M
        CA = (N3 + 1) / N3 * Sw + (np.trace(Sw) + SIG ** 2) / N3 * np.eye(D)
        Gth = Sw @ np.linalg.inv(CA)
        c_iso = ((N3 + 1) + DD) / (N3 * D)
        rows["theory_new"].append(1 - 2 * np.trace(Gth) / D + c_iso * np.trace(Gth @ Gth.T))
        rows["theory_train"].append(np.trace(Sw) - 2 * np.trace(Gth @ Sw) + np.trace(Gth @ CA @ Gth.T))
        if W is not None:
            Xt, yt, xqt, fqt, _ = prompts(NTEST, N3, 0.0, g, W)
            rows["LSA_train"].append(risk_lsa(G, Xt, yt, xqt, fqt).mean())
            for tag, (Xa, ya, xqa, fqa) in [("new", (Xn, yn, xqn, fqn)), ("train", (Xt, yt, xqt, fqt))]:
                ll = -((ya[:, None, :] - np.einsum("mnd,kd->mkn", Xa, W)) ** 2).sum(-1) / (2 * SIG ** 2)
                ll -= ll.max(1, keepdims=True); pw = np.exp(ll); pw /= pw.sum(1, keepdims=True)
                wh = pw @ W
                rows[f"dMMSE_{tag}"].append(((np.einsum("md,md->m", xqa, wh) - fqa) ** 2).mean())
    E3[str(M)] = {k: ms(v) for k, v in rows.items() if v}
print("E3", round(time.time() - T0, 1), flush=True)

# ---------------- E4: context length mismatch ----------------
E4 = {}
NTR = 20
Gs = [fit_lsa(NTR, 0.0, rng(70000 + s)) for s in range(SEEDS)]
for nte in [5, 10, 20, 40, 80, 160, 320]:
    rr, rg = [], []
    for s, G in enumerate(Gs):
        X, y, xq, fq, _ = prompts(2000, nte, 0.0, rng(71000 + 10 * nte + s))
        rr.append(risk_lsa(G, X, y, xq, fq).mean()); rg.append(risk_ridge(X, y, xq, fq).mean())
    E4[nte] = {"LSA_trained_n20": ms(rr), "ridge": ms(rg), "theory": risk_theory(gamma_star(NTR, 1.0), nte, 1.0),
               "theory_matched_n": risk_theory(gamma_star(nte, 1.0), nte, 1.0)}
E4["theory_plateau"] = (1 - gamma_star(NTR, 1.0)) ** 2
print("E4", round(time.time() - T0, 1), flush=True)

# theory-vs-measurement summary
devs = [abs(E1[r][n]["LSA"]["mean"] - E1[r][n]["theory_LSA"]) / E1[r][n]["theory_LSA"] for r in RHOS for n in NS]
devs2 = [abs(E2[n][a][b]["LSA"]["mean"] - E2[n][a][b]["theory"]) / E2[n][a][b]["theory"] for n in E2 for a in RHO_TR for b in RHO_TE]
summary = {"E1_rel_dev_median": float(np.median(devs)), "E1_rel_dev_max": float(np.max(devs)),
           "E2_rel_dev_median": float(np.median(devs2)), "E2_rel_dev_max": float(np.max(devs2))}

# ---------------- figures ----------------
P = plt()
fig, axes = P.subplots(1, 3, figsize=(18, 4.6))
ax = axes[0]
for i, rho in enumerate(RHOS):
    col = f"C{i}"
    ax.errorbar(NS, [E1[rho][n]["LSA"]["mean"] for n in NS], yerr=[E1[rho][n]["LSA"]["ci95"] for n in NS], fmt="o", color=col, capsize=2, label=f"LSA ρ={rho}")
    ax.plot(NS, [E1[rho][n]["theory_LSA"] for n in NS], "-", color=col, lw=1)
    if rho in (0.0, 0.3):
        ax.plot(NS, [E1[rho][n]["ridge"]["mean"] for n in NS], "s--", ms=3, color=col, lw=.8, label=f"岭回归 ρ={rho}")
    if rho == 0.3:
        nn = [n for n in NS if "robust_EM" in E1[rho][n]]
        ax.plot(nn, [E1[rho][n]["robust_EM"]["mean"] for n in nn], "^:", ms=4, color=col, label="稳健 EM ρ=0.3")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("示范数 n"); ax.set_ylabel("查询超额风险")
ax.set_title("E1 预训练与测试污染率一致（线=理论）"); ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=.3)
ax = axes[1]
for i, rt in enumerate(RHO_TR):
    col = f"C{i}"
    ax.errorbar(RHO_TE, [E2[160][rt][r]["LSA"]["mean"] for r in RHO_TE], yerr=[E2[160][rt][r]["LSA"]["ci95"] for r in RHO_TE], fmt="o", color=col, capsize=2, label=f"预训练 ρ={rt}")
    ax.plot(RHO_TE, [E2[160][rt][r]["theory"] for r in RHO_TE], "-", color=col, lw=1)
ax.axhline(1.0, color="k", ls=":", lw=1, label="零样本风险=1")
ax.set_xlabel("测试示范污染率 ρ_te"); ax.set_ylabel("查询超额风险"); ax.set_yscale("log")
ax.set_title("E2 预训练/测试污染失配（n=160，线=理论）"); ax.legend(fontsize=7); ax.grid(alpha=.3)
ax = axes[2]
xs = MS + [1024]
for key, lab, st in [("LSA_new", "LSA：新任务", "o-"), ("LSA_train", "LSA：训练任务", "o--"), ("dMMSE_new", "离散先验 Bayes：新任务", "s-"),
                     ("dMMSE_train", "离散先验 Bayes：训练任务", "s--"), ("theory_new", "LSA 理论：新任务", "k:")]:
    vals = [(x, E3[str(M)][key]["mean"]) for x, M in zip(xs, MS + ["∞"]) if key in E3[str(M)]]
    ax.plot([v[0] for v in vals], [v[1] for v in vals], st, ms=4, label=lab)
ax.axhline(E3["∞"]["ridge_new"]["mean"], color="grey", ls="-.", lw=1, label="岭回归（Bayes 最优）")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("预训练任务数 M（最右为 ∞）"); ax.set_ylabel("查询超额风险")
ax.set_title(f"E3 任务多样性（d={D}, n={N3}）"); ax.legend(fontsize=6.5); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "frontier_icl.png"); P.close(fig)

fig, ax = P.subplots(figsize=(5.6, 4.2))
ns4 = [5, 10, 20, 40, 80, 160, 320]
ax.errorbar(ns4, [E4[n]["LSA_trained_n20"]["mean"] for n in ns4], yerr=[E4[n]["LSA_trained_n20"]["ci95"] for n in ns4], fmt="o", capsize=2, label="LSA（在 n=20 上预训练）")
ax.plot(ns4, [E4[n]["theory"] for n in ns4], "-", lw=1, label="理论")
ax.plot(ns4, [E4[n]["ridge"]["mean"] for n in ns4], "s--", ms=3, label="岭回归")
ax.axhline(E4["theory_plateau"], color="grey", ls=":", label="理论平台 (1-γ*)²")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("测试示范数"); ax.set_ylabel("查询超额风险"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("E4 上下文长度失配"); fig.tight_layout(); fig.savefig(OUT / "frontier_context_length.png"); P.close(fig)

runtime = time.time() - T0
res = {"config": {"d": D, "sigma": SIG, "seeds": SEEDS, "n_pretrain_prompts": NPRE, "n_test_prompts": NTEST,
                  "prior": "w~N(0,I/d)", "corruption": "label sign flip with prob rho", "risk": "E(ŷ - x_qᵀw)² (noise-free target)"},
       "E1_matched": E1, "E2_mismatch": E2, "E2_key": E2_key, "E3_task_diversity": E3, "E4_context_length": E4,
       "theory_vs_measured": summary, "runtime_sec": runtime}
(OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print("runtime", runtime)
